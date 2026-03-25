#!/usr/bin/env python3
import argparse
import json
import sys
import base64
import requests
import warnings
import uiautomator2 as u2
import os
import subprocess
import tempfile
import contextlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from io import BytesIO
import re

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

try:
    from requests import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

DEVICE = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_MARKERS_JSON = Path(__file__).parent / "markers.json"
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:35b-a3b")
DEFAULT_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/hunter-alpha")
DEFAULT_SCREEN_OVERVIEW_PROVIDER = os.environ.get("DEFAULT_SCREEN_OVERVIEW_PROVIDER", "glm-ocr")
DEFAULT_OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DEFAULT_GLM_OCR_SERVER_URL = os.environ.get("GLM_OCR_SERVER_URL", "http://192.168.0.212:8765/ocr")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Z.AI MaaS API (預設使用這個!)
# 重要：API Key 請透過環境變數或 .env 設定，切勿寫在 code 裡！
ZAI_API_URL = os.environ.get("ZAI_API_URL", "https://api.z.ai/api/paas/v4/layout_parsing")
ZAI_API_KEY = os.environ.get("ZAI_API_KEY", "")  # 必需設定環境變數
USE_LOCAL_GLM_OCR = os.environ.get("USE_LOCAL_GLM_OCR", "true").lower() == "true"  # 預設使用本地模型

OMNIPARSER_INSTANCE = None
OMNIPARSER_DIR = Path(__file__).parent / "OmniParser"
OMNIPARSER_CONFIG = {
    "som_model_path": str(OMNIPARSER_DIR / "weights" / "icon_detect" / "model.pt"),
    "caption_model_name": "florence2",
    "caption_model_path": str(OMNIPARSER_DIR / "weights" / "icon_caption_florence"),
    "BOX_TRESHOLD": 0.05,
}

def get_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = u2.connect()
    return DEVICE

def get_screen():
    d = get_device()
    return d.info


def get_screenshot_png_base64() -> str:
    d = get_device()
    img = d.screenshot()
    if img is None:
        raise RuntimeError("無法取得截圖")

    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            png_bytes = buffer.getvalue()
        else:
            png_bytes = bytes(img) if hasattr(img, '__bytes__') else img
    except Exception as e:
        raise RuntimeError(f"截圖轉換失敗: {e}")

    if not png_bytes:
        raise RuntimeError("截圖資料為空")

    return base64.b64encode(png_bytes).decode("utf-8")


def ollama_chat(model: str, prompt: str, image_base64: Optional[str] = None, timeout: int = 120) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "think": False
    }
    if image_base64:
        payload["messages"][0]["images"] = [image_base64]

    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"無法連線到 Ollama API: {e}")

    if response.status_code != 200:
        try:
            err = response.json().get("error", response.text)
        except Exception:
            err = response.text
        raise RuntimeError(f"Ollama API 失敗: {err}")

    data = response.json()
    return str(data.get("message", {}).get("content", "")).strip()


def openrouter_chat(
    model: str,
    prompt: str,
    api_key: Optional[str] = None,
    image_base64: Optional[str] = None,
    timeout: int = 120,
) -> str:
    resolved_api_key = (api_key or os.environ.get("OPENROUTER_API_KEY") or DEFAULT_OPENROUTER_API_KEY).strip()
    if not resolved_api_key:
        raise RuntimeError("缺少 OpenRouter API Key，請設定 OPENROUTER_API_KEY 或使用 --api-key")

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_base64:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                },
            }
        )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": user_content,
            }
        ],
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"無法連線到 OpenRouter API: {e}")

    if response.status_code != 200:
        try:
            err = response.json().get("error", response.text)
        except Exception:
            err = response.text
        raise RuntimeError(f"OpenRouter API 失敗: {err}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter 回應缺少 choices")
    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()


def glm_ocr_server_chat(
    image_base64: Optional[str],
    server_url: str = DEFAULT_GLM_OCR_SERVER_URL,
    timeout: int = 120,
) -> str:
    payload: Dict[str, Any] = {
        "image_b64": image_base64 or "",
    }

    try:
        response = requests.post(
            server_url,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"無法連線到 GLM-OCR 本機 server ({server_url}): {e}")

    if response.status_code != 200:
        raise RuntimeError(f"GLM-OCR server 失敗 (HTTP {response.status_code}): {response.text}")

    try:
        data = response.json()
    except Exception:
        text = response.text.strip()
        if not text:
            raise RuntimeError("GLM-OCR server 回應為空")
        return text

    if isinstance(data, dict):
        if data.get("error"):
            raise RuntimeError(f"GLM-OCR server 錯誤: {data.get('error')}")
        if data.get("detail"):
            raise RuntimeError(f"GLM-OCR server 錯誤: {data.get('detail')}")
        if data.get("message") and "result" not in data and "text" not in data and "content" not in data and "ocr" not in data:
            raise RuntimeError(f"GLM-OCR server 錯誤: {data.get('message')}")
        for key in ["result", "text", "content", "ocr"]:
            if key in data and data.get(key) is not None:
                return str(data.get(key)).strip()
        return json.dumps(data, ensure_ascii=False)

    return str(data).strip()


def glm_ocr_api_chat(
    image_base64: str,
    api_url: str = ZAI_API_URL,
    api_key: str = ZAI_API_KEY,
    timeout: int = 120,
) -> str:
    """Call Z.AI MaaS GLM-OCR API directly (預設模式)."""
    if not api_key:
        raise RuntimeError(
            "ZAI_API_KEY 未設定！請在 .env 檔案中設定 ZAI_API_KEY，"
            "或 export ZAI_API_KEY=your_api_key"
        )
    
    payload = {
        "model": "glm-ocr",
        "file": f"data:image/png;base64,{image_base64}"
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"無法連線到 Z.AI API: {e}")
    
    if response.status_code != 200:
        raise RuntimeError(f"Z.AI API 失敗 (HTTP {response.status_code}): {response.text}")
    
    result = response.json()
    
    # Parse Z.AI response format: layout_details[page_index][items]
    # Each item has: bbox_2d, content, label, native_label, height, width, index
    layout_details = result.get("layout_details", [])
    if not layout_details or not layout_details[0]:
        # Fallback to markdown result
        md = result.get("md_results", "")
        if md:
            return md
        return json.dumps(result, ensure_ascii=False)
    
    items = layout_details[0]
    output_lines = []
    
    # 圖片資訊
    data_info = result.get("data_info", {})
    if data_info:
        pages = data_info.get("pages", [{}])
        if pages:
            img_h = pages[0].get("height", 0)
            img_w = pages[0].get("width", 0)
            output_lines.append(f"[IMAGE] size={img_w}x{img_h}")
    
    for item in items:
        bbox = item.get("bbox_2d", [])
        content = item.get("content", "")
        label = item.get("label", "text")
        native_label = item.get("native_label", "")
        
        if content:
            output_lines.append(f"[{label}] bbox={bbox} \"{content}\"")
        else:
            output_lines.append(f"[{label}] bbox={bbox}")
    
    return "\n".join(output_lines)


def strip_reasoning_output(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    if "Final Answer:" in cleaned:
        cleaned = cleaned.split("Final Answer:", 1)[1].strip()

    lines = cleaned.splitlines()
    while lines and lines[0].strip().lower().startswith(("thinking", "thinking process", "思考")):
        lines.pop(0)

    return "\n".join(lines).strip()


@contextlib.contextmanager
def _suppress_output(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _get_omniparser_instance(verbose: bool = False):
    global OMNIPARSER_INSTANCE
    if OMNIPARSER_INSTANCE is not None:
        return OMNIPARSER_INSTANCE

    if not OMNIPARSER_DIR.exists():
        raise RuntimeError(f"OmniParser 目錄不存在: {OMNIPARSER_DIR}")

    omni_dir_str = str(OMNIPARSER_DIR)
    if omni_dir_str not in sys.path:
        sys.path.insert(0, omni_dir_str)

    try:
        with _suppress_output(enabled=not verbose):
            from util.omniparser import Omniparser  # type: ignore
            OMNIPARSER_INSTANCE = Omniparser(config=OMNIPARSER_CONFIG)
    except Exception as e:
        raise RuntimeError(f"載入 OmniParser 失敗: {e}")

    return OMNIPARSER_INSTANCE


def _to_pixel_bbox(bbox: List[float], width: int, height: int) -> List[int]:
    return [
        int(bbox[0] * width),
        int(bbox[1] * height),
        int(bbox[2] * width),
        int(bbox[3] * height),
    ]


def omniparser_screen_overview(image_base64: str, verbose: bool = False) -> str:
    parser = _get_omniparser_instance(verbose=verbose)
    with _suppress_output(enabled=not verbose):
        _, parsed_content_list = parser.parse(image_base64)

    from PIL import Image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size

    pixel_items: List[Dict[str, Any]] = []
    for i, item in enumerate(parsed_content_list):
        elem = dict(item)
        bbox = elem.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            elem["bbox_ratio"] = bbox
            elem["bbox"] = _to_pixel_bbox(bbox, width, height)
        elem["id"] = i
        pixel_items.append(elem)

    return "\n".join([f"icon {i}: {v}" for i, v in enumerate(pixel_items)])


def _clean_llm_text(text: str) -> str:
    cleaned = (text or "").replace("<|user|>", "").replace("<|assistant|>", "").strip()
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _extract_json_field(text: str, field: str) -> str:
    if not text:
        return ""
    m = re.search(rf'"{re.escape(field)}"\s*:\s*"(.*?)"', text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).replace('\\n', '\n').replace('\\"', '"').strip()


def _extract_gemini_text(response_json: Dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts: List[str] = []
    for p in parts:
        t = p.get("text")
        if t:
            texts.append(t)
    return "\n".join(texts).strip()


def _crop_to_base64(image, bbox_ratio: List[float]) -> str:
    w, h = image.size
    x1 = max(0, min(w, int(bbox_ratio[0] * w)))
    y1 = max(0, min(h, int(bbox_ratio[1] * h)))
    x2 = max(0, min(w, int(bbox_ratio[2] * w)))
    y2 = max(0, min(h, int(bbox_ratio[3] * h)))
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = image.crop((x1, y1, x2, y2))
    buf = BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def full_screen_overview_pipeline(
    image_base64: str,
    glm_ocr_url: str = DEFAULT_GLM_OCR_SERVER_URL,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    verbose: bool = False,
) -> str:
    """Full pipeline: OmniParser -> GLM-OCR per box -> Gemini refine per box -> pixel bbox output."""
    parser = _get_omniparser_instance(verbose=verbose)
    with _suppress_output(enabled=not verbose):
        _, parsed_content_list = parser.parse(image_base64)

    from PIL import Image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # Step 1: GLM-OCR text replacement on each box
    for item in parsed_content_list:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        crop_b64 = _crop_to_base64(image, bbox)
        if not crop_b64:
            continue
        try:
            new_text = _clean_llm_text(
                glm_ocr_server_chat(image_base64=crop_b64, server_url=glm_ocr_url, timeout=120)
            )
            if new_text:
                item["content"] = new_text
                item["source"] = "glm_ocr_api"
        except Exception:
            continue

    # Step 2: Gemini refine on each box
    if GEMINI_API_KEY:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{gemini_model}:generateContent?key={GEMINI_API_KEY}"
        )
        headers = {"Content-Type": "application/json"}
        instruction = (
            "You are a strict UI element recognizer. "
            "Return exactly one line JSON: "
            "{\"refined_content\":\"...\",\"icon_label\":\"...\"}."
        )

        for idx, item in enumerate(parsed_content_list):
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            crop_b64 = _crop_to_base64(image, bbox)
            if not crop_b64:
                continue

            content = _clean_llm_text(str(item.get("content") or ""))
            prompt = (
                f"{instruction}\n"
                f"box_id={idx}\n"
                f"type={item.get('type', 'unknown')}\n"
                f"existing_content={content}\n"
                "Refine this UI element."
            )

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/png", "data": crop_b64}},
                        ]
                    }
                ],
                "generationConfig": {"temperature": 0.1},
            }

            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
                if resp.status_code != 200:
                    continue
                out = _extract_gemini_text(resp.json())
                refined = _extract_json_field(out, "refined_content") or _clean_llm_text(out)
                icon_label = _extract_json_field(out, "icon_label")
                if refined:
                    item["content"] = refined
                    item["source"] = "gemini_refined"
                if icon_label:
                    item["icon_label"] = icon_label
            except Exception:
                continue

    # Step 3: ratio bbox -> pixel bbox for output
    pixel_items: List[Dict[str, Any]] = []
    for i, item in enumerate(parsed_content_list):
        elem = dict(item)
        bbox = elem.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            elem["bbox_ratio"] = bbox
            elem["bbox"] = _to_pixel_bbox(bbox, width, height)
        elem["id"] = i
        pixel_items.append(elem)

    return "\n".join([f"icon {i}: {v}" for i, v in enumerate(pixel_items)])


def load_markers_data(json_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(json_path).expanduser() if json_path else DEFAULT_MARKERS_JSON
    if not path.exists():
        raise FileNotFoundError(f"找不到標記 JSON：{path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("標記 JSON 格式錯誤：根節點必須是物件")
    if not isinstance(data.get("states"), list):
        raise ValueError("標記 JSON 格式錯誤：缺少 states 陣列")
    return data


def get_base_resolution(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    raw = data.get("base_resolution")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return {"width": int(raw[0]), "height": int(raw[1])}
        except Exception:
            return None
    if isinstance(raw, dict):
        try:
            return {
                "width": int(raw.get("width", 0)),
                "height": int(raw.get("height", 0)),
            }
        except Exception:
            return None
    return None


def find_state(data: Dict[str, Any], state_name: str) -> Dict[str, Any]:
    for state in data.get("states", []):
        if str(state.get("name", "")).strip() == state_name:
            return state
    raise ValueError(f"找不到狀態：{state_name}")


def find_marker(state: Dict[str, Any], marker_name: str) -> Dict[str, Any]:
    for marker in state.get("markers", []):
        if str(marker.get("name", "")).strip() == marker_name:
            return marker
    raise ValueError(f"在狀態「{state.get('name', '')}」找不到標記：{marker_name}")


def cmd_list_states(json_path: Optional[str] = None):
    data = load_markers_data(json_path)
    base_resolution = get_base_resolution(data)
    states = data.get("states", [])
    output = {
        "base_resolution": base_resolution,
        "count": len(states),
        "states": [
            {
                "name": str(state.get("name", "")),
                "description": str(state.get("description", "")),
                "marker_count": len(state.get("markers", [])),
            }
            for state in states
        ],
    }
    return json.dumps(output, ensure_ascii=False)


def cmd_list_markers(state_name: str, json_path: Optional[str] = None):
    data = load_markers_data(json_path)
    state = find_state(data, state_name)
    markers = state.get("markers", [])
    output = {
        "state": str(state.get("name", "")),
        "description": str(state.get("description", "")),
        "count": len(markers),
        "markers": markers,
    }
    return json.dumps(output, ensure_ascii=False)


def cmd_run_marker(state_name: str, marker_name: str, json_path: Optional[str] = None, duration: float = 0.5):
    data = load_markers_data(json_path)
    state = find_state(data, state_name)
    marker = find_marker(state, marker_name)

    marker_type = str(marker.get("type", "tap")).strip().lower()
    x = int(marker.get("x", 0))
    y = int(marker.get("y", 0))

    if marker_type in ["swipe", "slide", "滑動"]:
        bx = int(marker.get("bx", x))
        by = int(marker.get("by", y))
        cmd_swipe(x, y, bx, by, duration)
        action = {
            "type": "swipe",
            "from": [x, y],
            "to": [bx, by],
            "duration": duration,
        }
    else:
        cmd_click(x, y)
        action = {
            "type": "tap",
            "at": [x, y],
        }

    output = {
        "status": "OK",
        "state": state_name,
        "marker": marker_name,
        "action": action,
        "target_state": marker.get("target_state"),
    }
    return json.dumps(output, ensure_ascii=False)


def cmd_run_marker_and_follow(state_name: str, marker_name: str, json_path: Optional[str] = None, duration: float = 0.5):
    data = load_markers_data(json_path)
    state = find_state(data, state_name)
    marker = find_marker(state, marker_name)

    run_result = json.loads(cmd_run_marker(state_name, marker_name, json_path=json_path, duration=duration))
    target_state_name = marker.get("target_state")
    if not target_state_name:
        run_result["next_state"] = None
        return json.dumps(run_result, ensure_ascii=False)

    try:
        target_state = find_state(data, str(target_state_name))
        run_result["next_state"] = {
            "name": str(target_state.get("name", "")),
            "description": str(target_state.get("description", "")),
            "marker_count": len(target_state.get("markers", [])),
        }
    except Exception:
        run_result["next_state"] = {
            "name": str(target_state_name),
            "description": "",
            "marker_count": 0,
            "warning": "target_state 在 JSON 中不存在",
        }

    return json.dumps(run_result, ensure_ascii=False)


def cmd_screen_overview(
    use_image: bool = True,
    provider: str = "api",
    glm_ocr_url: str = DEFAULT_GLM_OCR_SERVER_URL,
    debug: bool = False,
):
    """
    Screen overview with three modes:
    - "ocr": Uses local GLM-OCR server (DEFAULT)
    - "api": Uses Z.AI MaaS GLM-OCR API directly (set USE_LOCAL_GLM_OCR=false to switch)
    - "full": Full pipeline with OmniParser + GLM-OCR per box + Gemini refine (detailed)
    
    預設使用本地模型，若要使用 Z.AI API 請設環境變數 USE_LOCAL_GLM_OCR=false
    """
    # 根據 USE_LOCAL_GLM_OCR 環境變數決定預設 provider
    # 預設使用本地模型（USE_LOCAL_GLM_OCR 預設 true）
    # 若要使用 Z.AI API，設 USE_LOCAL_GLM_OCR=false
    provider = "ocr" if USE_LOCAL_GLM_OCR else "api"
    
    provider_norm = (provider or "").strip().lower()
    if provider_norm not in ["api", "ocr", "full"]:
        return json.dumps({"error": f"不支援的 provider: {provider}，僅支援 'api'、'ocr' 或 'full'"})

    debug_info: Dict[str, Any] = {
        "provider": provider_norm,
        "glm_ocr_url": glm_ocr_url,
        "image_attached": False,
        "image_base64_length": 0,
    }

    image_base64 = None
    if use_image:
        try:
            image_base64 = get_screenshot_png_base64()
            debug_info["image_attached"] = True
            debug_info["image_base64_length"] = len(image_base64)
        except Exception as e:
            if provider_norm == "full":
                return json.dumps({"error": f"無法取得截圖: {e}"})
            debug_info["warning"] = f"無法附上截圖: {e}"

    if not image_base64:
        if provider_norm == "full":
            return json.dumps({"error": "需要截圖影像"})
        return json.dumps({
            "error": "無法取得截圖",
            "debug": debug_info,
        })

    try:
        if provider_norm == "api":
            # Z.AI MaaS API (預設模式)
            content = glm_ocr_api_chat(
                image_base64=image_base64,
                timeout=120,
            )
        elif provider_norm == "ocr":
            # 本地 GLM-OCR server
            content = glm_ocr_server_chat(
                image_base64=image_base64,
                server_url=glm_ocr_url,
                timeout=120,
            )
        else:  # full
            content = full_screen_overview_pipeline(
                image_base64=image_base64,
                glm_ocr_url=glm_ocr_url,
                verbose=debug,
            )
    except Exception as e:
        return json.dumps({"error": str(e)})

    if debug:
        return f"[DEBUG] {json.dumps(debug_info, ensure_ascii=False)}\n\n{content}"
    return content

def cmd_screenshot():
    d = get_device()
    img = d.screenshot()
    if img is None:
        return ""
    try:
        from PIL import Image
        from io import BytesIO
        if isinstance(img, Image.Image):
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            data = buffer.getvalue()
        else:
            data = bytes(img) if hasattr(img, '__bytes__') else img
    except Exception:
        data = b""

    # Save screenshot to PhoneUse/screenshot/ with timestamp and app name
    try:
        script_dir = Path(__file__).parent
        screenshot_dir = script_dir / "screenshot"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        app_name = ""
        try:
            app_info = d.app_current()
            package = app_info.get("package", "")
            if package.startswith("com."):
                app_name = "_" + package
        except Exception:
            pass

        filename = f"{timestamp}{app_name}.png"
        filepath = screenshot_dir / filename

        from PIL import Image
        from io import BytesIO
        pil_img = Image.open(BytesIO(data))
        pil_img.save(str(filepath), format='PNG')
    except Exception:
        pass

    return "screenshot saved, filename: " + filename

def cmd_hierarchy():
    d = get_device()
    xml = d.dump_hierarchy(compressed=True)
    return xml

def cmd_click(x, y):
    d = get_device()
    d.click(x, y)
    return "OK"

def cmd_long_click(x, y, duration=1.0):
    d = get_device()
    d.long_click(x, y, duration)
    return "OK"

def cmd_swipe(sx, sy, ex, ey, duration=0.5):
    d = get_device()
    d.swipe(sx, sy, ex, ey, duration)
    return "OK"

def cmd_press(key):
    d = get_device()
    d.press(key)
    return "OK"

def cmd_text(text):
    d = get_device()
    d.set_fastinput_ime(True)
    d.send_keys(text)
    return "OK"

def cmd_clear():
    d = get_device()
    d.set_fastinput_ime(True)
    d.clear_text()
    return "OK"

def cmd_tap_text(text, fuzzy=False):
    d = get_device()
    if fuzzy:
        d(text=text).click()
    else:
        d(text=text).click()
    return "OK"

def cmd_exists(text):
    d = get_device()
    exists = d(text=text).exists
    return "true" if exists else "false"

def cmd_wait_id(resource_id, timeout=10):
    d = get_device()
    el = d(resourceId=resource_id, timeout=timeout)
    return "true" if el.exists else "false"

def cmd_current_app():
    d = get_device()
    return d.app_current()

def cmd_start_app(package):
    d = get_device()
    d.app_start(package)
    return "OK"

def cmd_stop_app(package):
    d = get_device()
    d.app_stop(package)
    return "OK"

def cmd_screen_size():
    d = get_device()
    return f"{d.info['displayWidth']}x{d.info['displayHeight']}"

def cmd_annotate():
    """Annotate UI elements using Gemini 2.0 Flash Vision API."""
    if not GEMINI_API_KEY:
        return json.dumps({"error": "GEMINI_API_KEY 未設定"})

    try:
        image_base64 = get_screenshot_png_base64()
    except Exception as e:
        return json.dumps({"error": f"無法取得截圖: {e}"})

    screen_size = cmd_screen_size()
    width, height = map(int, screen_size.split('x'))

    prompt = f"""You are a UI interaction annotation assistant for Android app/game screenshots.
Screen resolution: {screen_size}

Identify interactable UI elements only:
- Buttons, CTAs, confirm/cancel
- Input fields, search bars, dropdowns
- Checkboxes, toggles, radio buttons
- Tabs, navigation bars, back buttons
- Scrollable lists (annotate container)
- Game controls: skill buttons, item slots, joysticks, attack buttons
- Swipeable cards, links, tappable text

Exclude:
- Decorative images, backgrounds
- Static labels, non-tappable text
- HUD elements (HP bars, scores) unless tappable
- Loading indicators unless stoppable

Output JSON array only, no other text:
[
  {{
    "id": "elem_01",
    "type": "button",
    "label": "Start Game button",
    "action": "tap",
    "bbox": [x_min, y_min, x_max, y_max],
    "center": [x, y],
    "confidence": "high"
  }}
]

bbox values should be absolute pixels based on screen resolution {width}x{height}. Include ALL interactable elements with confidence level."""

    try:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-3-flash-preview:generateContent?key={GEMINI_API_KEY}"
        )
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": image_base64}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "top_p": 0.95,
            },
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            return json.dumps({"error": f"API 失敗 (HTTP {resp.status_code}): {resp.text}"})

        candidates = resp.json().get("candidates") or []
        if not candidates:
            return json.dumps({"error": "API 回應無 candidates"})

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()

        # Try to extract JSON array
        json_match = re.search(r'(\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list):
                    return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass

        # Fallback: try whole text
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        return json.dumps({"error": "無法解析 JSON", "raw": text[:500]})

    except requests.RequestException as e:
        return json.dumps({"error": f"網路錯誤: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description='PhoneUse - Agent-controlled phone automation')
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('screenshot', help='Get screenshot as base64')
    sub.add_parser('hierarchy', help='Get UI hierarchy as XML')
    sub.add_parser('screen_size', help='Get screen resolution')
    sub.add_parser('current_app', help='Get current running app')

    p = sub.add_parser('click', help='Click at coordinates')
    p.add_argument('x', type=int)
    p.add_argument('y', type=int)

    p = sub.add_parser('long_click', help='Long click at coordinates')
    p.add_argument('x', type=int)
    p.add_argument('y', type=int)
    p.add_argument('--duration', type=float, default=1.0)

    p = sub.add_parser('swipe', help='Swipe from (sx,sy) to (ex,ey)')
    p.add_argument('sx', type=int)
    p.add_argument('sy', type=int)
    p.add_argument('ex', type=int)
    p.add_argument('ey', type=int)
    p.add_argument('--duration', type=float, default=0.5)

    p = sub.add_parser('press', help='Press key (home/back/menu/volume_up/volume_down/power)')
    p.add_argument('key')

    p = sub.add_parser('text', help='Input text')
    p.add_argument('content')

    sub.add_parser('clear', help='Clear input text')

    p = sub.add_parser('tap_text', help='Tap element by text')
    p.add_argument('text')
    p.add_argument('--fuzzy', action='store_true')

    p = sub.add_parser('exists', help='Check if element with text exists')
    p.add_argument('text')

    p = sub.add_parser('wait_id', help='Wait for element by resource ID')
    p.add_argument('resource_id')
    p.add_argument('--timeout', type=int, default=10)

    p = sub.add_parser('start_app', help='Start app by package name')
    p.add_argument('package')

    p = sub.add_parser('stop_app', help='Stop app by package name')
    p.add_argument('package')

    sub.add_parser('annotate', help='Annotate UI elements using Gemini vision')

    p = sub.add_parser('list_states', help='List states from markers JSON')
    p.add_argument('--json', dest='json_path', default=None, help='Path to markers JSON')

    p = sub.add_parser('list_markers', help='List markers in a state from markers JSON')
    p.add_argument('state_name')
    p.add_argument('--json', dest='json_path', default=None, help='Path to markers JSON')

    p = sub.add_parser('run_marker', help='Run one marker action from markers JSON')
    p.add_argument('state_name')
    p.add_argument('marker_name')
    p.add_argument('--json', dest='json_path', default=None, help='Path to markers JSON')
    p.add_argument('--duration', type=float, default=0.5, help='Swipe duration')

    p = sub.add_parser('run_marker_follow', help='Run marker and return target_state brief info')
    p.add_argument('state_name')
    p.add_argument('marker_name')
    p.add_argument('--json', dest='json_path', default=None, help='Path to markers JSON')
    p.add_argument('--duration', type=float, default=0.5, help='Swipe duration')

    p = sub.add_parser('screen_overview', help='Describe current screen - default uses local model (set USE_LOCAL_GLM_OCR=false for cloud API)')
    p.add_argument('--provider', default='ocr', choices=['api', 'ocr', 'full'], help='Provider: ocr (local GLM-OCR server, default), api (Z.AI MaaS API, set USE_LOCAL_GLM_OCR=false to use this)')
    p.add_argument('--glm-ocr-url', dest='glm_ocr_url', default=DEFAULT_GLM_OCR_SERVER_URL, help='GLM-OCR local server URL')
    p.add_argument('--no-image', action='store_true', help='Do not attach screenshot image')
    p.add_argument('--debug', action='store_true', help='Show debug metadata')

    args = parser.parse_args()

    try:
        if args.cmd == 'screenshot':
            print(cmd_screenshot())
        elif args.cmd == 'hierarchy':
            print(cmd_hierarchy())
        elif args.cmd == 'screen_size':
            print(cmd_screen_size())
        elif args.cmd == 'current_app':
            print(json.dumps(cmd_current_app()))
        elif args.cmd == 'click':
            print(cmd_click(args.x, args.y))
        elif args.cmd == 'long_click':
            print(cmd_long_click(args.x, args.y, args.duration))
        elif args.cmd == 'swipe':
            print(cmd_swipe(args.sx, args.sy, args.ex, args.ey, args.duration))
        elif args.cmd == 'press':
            print(cmd_press(args.key))
        elif args.cmd == 'text':
            print(cmd_text(args.content))
        elif args.cmd == 'clear':
            print(cmd_clear())
        elif args.cmd == 'tap_text':
            print(cmd_tap_text(args.text, args.fuzzy))
        elif args.cmd == 'exists':
            print(cmd_exists(args.text))
        elif args.cmd == 'wait_id':
            print(cmd_wait_id(args.resource_id, args.timeout))
        elif args.cmd == 'start_app':
            print(cmd_start_app(args.package))
        elif args.cmd == 'stop_app':
            print(cmd_stop_app(args.package))
        elif args.cmd == 'annotate':
            print(cmd_annotate())
        elif args.cmd == 'list_states':
            print(cmd_list_states(args.json_path))
        elif args.cmd == 'list_markers':
            print(cmd_list_markers(args.state_name, args.json_path))
        elif args.cmd == 'run_marker':
            print(cmd_run_marker(args.state_name, args.marker_name, args.json_path, args.duration))
        elif args.cmd == 'run_marker_follow':
            print(cmd_run_marker_and_follow(args.state_name, args.marker_name, args.json_path, args.duration))
        elif args.cmd == 'screen_overview':
            print(
                cmd_screen_overview(
                    use_image=not args.no_image,
                    provider=args.provider,
                    glm_ocr_url=args.glm_ocr_url,
                    debug=args.debug,
                )
            )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
