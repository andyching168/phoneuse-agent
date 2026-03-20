# from ultralytics import YOLO
import os
import io
import base64
import time
from PIL import Image, ImageDraw, ImageFont
import json
import requests
# utility function
import os
from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()

import json
import sys
import os
import cv2
import numpy as np
# %matplotlib inline
import easyocr
from paddleocr import PaddleOCR

# GLM-OCR for better text recognition
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import requests

reader = easyocr.Reader(['en'])
paddle_ocr = PaddleOCR(
    lang='en',  # other lang also available
    use_angle_cls=False,
    use_gpu=False,  # using cuda will conflict with pytorch in the same process
    max_batch_size=1024,
    use_dilation=True,  # improves accuracy
    det_db_score_mode='slow',  # improves accuracy
    rec_batch_num=1024)

# GLM-OCR model initialization (lazy loading)
glm_processor = None
glm_model = None
GLM_OCR_API_URL = "http://192.168.0.212:8765/ocr"  # Default API endpoint
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://192.168.0.212:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:0.8b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))
OLLAMA_RETRY = int(os.environ.get("OLLAMA_RETRY", "1"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "120"))
GEMINI_RETRY = int(os.environ.get("GEMINI_RETRY", "1"))

def get_glm_ocr_model():
    """Lazy load GLM-OCR model"""
    global glm_processor, glm_model
    if glm_processor is None:
        print("Loading GLM-OCR model...")
        try:
            from transformers import AutoProcessor
            glm_processor = AutoProcessor.from_pretrained(
                "zai-org/GLM-OCR",
                trust_remote_code=True
            )
            glm_model = AutoModelForImageTextToText.from_pretrained(
                "zai-org/GLM-OCR",
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            print("GLM-OCR model loaded!")
        except Exception as e:
            print(f"Failed to load GLM-OCR model: {e}")
            print("Will use HTTP API instead if available")
            return None, None
    return glm_processor, glm_model

def recognize_text_with_glm_ocr(cropped_image: Image.Image, use_api: bool = False) -> str:
    """
    Recognize text from a cropped image using GLM-OCR
    
    Args:
        cropped_image: PIL Image object (cropped region)
        use_api: If True, use HTTP API endpoint instead of local model
    
    Returns:
        Recognized text string
    """
    import tempfile
    
    # Try API first if requested or if model fails to load
    if use_api:
        try:
            # Send bytes directly so API works across different hosts/filesystems.
            buffer = io.BytesIO()
            cropped_image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

            # Call API
            response = requests.post(
                GLM_OCR_API_URL,
                json={"image_b64": image_b64},
                timeout=180  # 3 minutes for CPU inference
            )
            
            if response.status_code == 200:
                result = response.json().get("result", "")
                # Clean up markdown formatting and check if actual text exists
                cleaned = result.strip()
                # Remove markdown code blocks
                if cleaned.startswith("```"):
                    # Extract content between code blocks
                    lines = cleaned.split('\n')
                    content_lines = []
                    in_code_block = False
                    for line in lines:
                        if line.strip() == "```":
                            in_code_block = not in_code_block
                            continue
                        if not in_code_block and line.strip():
                            content_lines.append(line)
                    cleaned = '\n'.join(content_lines).strip()
                
                # Return empty if no meaningful text
                if not cleaned:
                    return ""
                return cleaned
            else:
                print(f"[GLM-OCR API] Failed: {response.status_code}")
                return ""
        except Exception as e:
            print(f"[GLM-OCR API] Error: {e}")
            return ""
    
    # Use local model
    processor, model = get_glm_ocr_model()
    if processor is None or model is None:
        print("GLM-OCR not available")
        return ""

    # Save to temp file for GLM-OCR
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cropped_image.save(tmp.name)
        img_src = tmp.name

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": img_src},
                {"type": "text", "text": "Screenshot Recognition:"}
            ]
        }]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            think=False,
            return_dict=True, return_tensors="pt"
        ).to(model.device)
        inputs.pop("token_type_ids", None)

        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        result = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )

        # Clean up temp file
        os.unlink(img_src)

        # Extract text from result (GLM-OCR returns structured output)
        return result.strip()
    except Exception as e:
        print(f"Local model error: {e}")
        if os.path.exists(img_src):
            os.unlink(img_src)
        return ""


def _extract_json_field(text: str, field: str) -> str:
    """Extract a simple JSON string field from model output without strict JSON parsing."""
    if not text:
        return ""
    pattern = rf'"{re.escape(field)}"\s*:\s*"(.*?)"'
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        return ""
    value = m.group(1)
    return value.replace('\\n', '\n').replace('\\"', '"').strip()


def _extract_gemini_text(response_json: dict) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        return ""
    content = (candidates[0].get("content") or {}).get("parts") or []
    texts = []
    for part in content:
        txt = part.get("text")
        if txt:
            texts.append(txt)
    return "\n".join(texts).strip()


def _clean_model_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("<|user|>", "").replace("<|assistant|>", "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def refine_boxes_with_gemini(
    image_source: Image.Image,
    parsed_content_list: list[dict],
    api_key: str = GEMINI_API_KEY,
    model: str = GEMINI_MODEL,
    timeout: int = GEMINI_TIMEOUT,
    retry: int = GEMINI_RETRY,
    verbose: bool = True,
) -> tuple[list[dict], int]:
    """Refine box text/icon semantics using Gemini with image crop + existing description."""
    if not parsed_content_list:
        return parsed_content_list, 0
    if not api_key:
        print("[GEMINI] Skip refinement: GEMINI_API_KEY is empty")
        return parsed_content_list, 0

    image_source = image_source.convert("RGB")
    w, h = image_source.size
    refined_count = 0
    total = len(parsed_content_list)

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    instruction = (
        "You are a strict UI element recognizer. "
        "Given a cropped GUI element image and an existing description, refine the element meaning. "
        "Return exactly one line of JSON: "
        "{\"refined_content\":\"...\",\"icon_label\":\"...\"}. "
        "Keep refined_content short and practical. If uncertain, preserve existing meaning."
    )

    if verbose:
        print(
            f"[GEMINI] Start refinement: total_boxes={total}, model={model}, timeout={timeout}s, retry={retry}"
        )

    for idx, item in enumerate(parsed_content_list):
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            if verbose:
                print(f"[GEMINI][{idx+1}/{total}] SKIP invalid bbox: {bbox}")
            continue

        x1 = max(0, min(w, int(bbox[0] * w)))
        y1 = max(0, min(h, int(bbox[1] * h)))
        x2 = max(0, min(w, int(bbox[2] * w)))
        y2 = max(0, min(h, int(bbox[3] * h)))
        if x2 <= x1 or y2 <= y1:
            if verbose:
                print(f"[GEMINI][{idx+1}/{total}] SKIP empty crop after clamp: {(x1, y1, x2, y2)}")
            continue

        crop = image_source.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        existing_content = _clean_model_text(str(item.get("content") or "").strip())
        item_type = str(item.get("type") or "unknown")
        source = str(item.get("source") or "")

        if verbose:
            print(
                f"[GEMINI][{idx+1}/{total}] START type={item_type} source={source} "
                f"bbox={(x1, y1, x2, y2)} old='{existing_content[:120]}'"
            )

        prompt = (
            f"box_id={idx}\n"
            f"type={item_type}\n"
            f"source={source}\n"
            f"existing_content={existing_content}\n"
            "Refine this into better UI/icon meaning."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{instruction}\n\n{prompt}"},
                        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.1},
        }

        headers = {"Content-Type": "application/json"}
        success = False
        last_error = ""

        for attempt in range(1, retry + 2):
            started = time.time()
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    last_error = f"status={resp.status_code} body={resp.text[:200]}"
                    if verbose:
                        print(f"[GEMINI][{idx+1}/{total}] FAIL attempt={attempt} {last_error}")
                    continue

                data = resp.json()
                output = _extract_gemini_text(data)
                refined_content = _extract_json_field(output, "refined_content")
                icon_label = _extract_json_field(output, "icon_label")

                if not refined_content:
                    refined_content = _clean_model_text(output)

                elapsed = time.time() - started
                if refined_content:
                    item["content"] = refined_content
                    item["source"] = "gemini_refined"
                    refined_count += 1
                if icon_label:
                    item["icon_label"] = icon_label

                if verbose:
                    raw_preview = output.replace("\n", " ")[:160]
                    print(
                        f"[GEMINI][{idx+1}/{total}] OK attempt={attempt} elapsed={elapsed:.2f}s "
                        f"new='{str(item.get('content', ''))[:120]}' icon_label='{item.get('icon_label', '')}' raw='{raw_preview}'"
                    )
                success = True
                break
            except requests.exceptions.Timeout:
                elapsed = time.time() - started
                last_error = f"timeout after {elapsed:.2f}s"
                if verbose:
                    print(f"[GEMINI][{idx+1}/{total}] TIMEOUT attempt={attempt} {last_error}")
            except Exception as e:
                elapsed = time.time() - started
                last_error = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"[GEMINI][{idx+1}/{total}] ERROR attempt={attempt} elapsed={elapsed:.2f}s {last_error}")

        if not success and verbose:
            print(f"[GEMINI][{idx+1}/{total}] GIVEUP last_error={last_error}")

    if verbose:
        print(f"[GEMINI] Done refinement: refined={refined_count}/{total}")
    return parsed_content_list, refined_count


def refine_boxes_with_ollama(
    image_source: Image.Image,
    parsed_content_list: list[dict],
    model: str = OLLAMA_MODEL,
    api_url: str = OLLAMA_API_URL,
    timeout: int = OLLAMA_TIMEOUT,
    retry: int = OLLAMA_RETRY,
    verbose: bool = True,
) -> tuple[list[dict], int]:
    """Refine box text/icon semantics using local Ollama with image crop + existing description."""
    if not parsed_content_list:
        return parsed_content_list, 0

    image_source = image_source.convert("RGB")
    w, h = image_source.size
    refined_count = 0

    system_prompt = (
        "You are a strict UI element recognizer. "
        "Given a cropped GUI element image and an existing description, refine the element meaning. "
        "Return exactly one line of JSON: "
        "{\"refined_content\":\"...\",\"icon_label\":\"...\"}. "
        "Keep refined_content short and practical. If uncertain, preserve existing meaning."
    )

    total = len(parsed_content_list)
    if verbose:
        print(
            f"[OLLAMA] Start refinement: total_boxes={total}, model={model}, "
            f"timeout={timeout}s, retry={retry}, api={api_url}"
        )

    for idx, item in enumerate(parsed_content_list):
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            if verbose:
                print(f"[OLLAMA][{idx+1}/{total}] SKIP invalid bbox: {bbox}")
            continue

        x1 = max(0, min(w, int(bbox[0] * w)))
        y1 = max(0, min(h, int(bbox[1] * h)))
        x2 = max(0, min(w, int(bbox[2] * w)))
        y2 = max(0, min(h, int(bbox[3] * h)))
        if x2 <= x1 or y2 <= y1:
            if verbose:
                print(f"[OLLAMA][{idx+1}/{total}] SKIP empty crop after clamp: {(x1, y1, x2, y2)}")
            continue

        crop = image_source.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        existing_content = str(item.get("content") or "").strip()
        item_type = str(item.get("type") or "unknown")
        source = str(item.get("source") or "")
        old_preview = existing_content[:120] if existing_content else ""

        if verbose:
            print(
                f"[OLLAMA][{idx+1}/{total}] START type={item_type} source={source} "
                f"bbox={(x1, y1, x2, y2)} old='{old_preview}'"
            )

        user_prompt = (
            f"box_id={idx}\n"
            f"type={item_type}\n"
            f"source={source}\n"
            f"existing_content={existing_content}\n"
            "Refine this into better UI/icon meaning."
        )

        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [image_b64]},
            ],
            "options": {"temperature": 0.1},
        }

        success = False
        last_error = ""
        for attempt in range(1, retry + 2):
            started = time.time()
            try:
                resp = requests.post(api_url, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    body = resp.text[:400]
                    if "image" in body.lower() or "vision" in body.lower() or "multimodal" in body.lower():
                        if verbose:
                            print(f"[OLLAMA][{idx+1}/{total}] attempt={attempt} text-only fallback")
                        text_only_payload = {
                            "model": model,
                            "stream": False,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            "options": {"temperature": 0.1},
                        }
                        resp = requests.post(api_url, json=text_only_payload, timeout=timeout)
                        if resp.status_code != 200:
                            last_error = f"status={resp.status_code} body={resp.text[:200]}"
                            if verbose:
                                print(f"[OLLAMA][{idx+1}/{total}] FAIL attempt={attempt} {last_error}")
                            continue
                    else:
                        last_error = f"status={resp.status_code} body={resp.text[:200]}"
                        if verbose:
                            print(f"[OLLAMA][{idx+1}/{total}] FAIL attempt={attempt} {last_error}")
                        continue

                data = resp.json()
                output = (data.get("message") or {}).get("content", "")
                if not output:
                    output = data.get("response", "")

                refined_content = _extract_json_field(output, "refined_content")
                icon_label = _extract_json_field(output, "icon_label")

                if not refined_content:
                    refined_content = output.strip()

                elapsed = time.time() - started
                if refined_content:
                    item["content"] = refined_content
                    item["source"] = "ollama_qwen35_refined"
                    refined_count += 1
                if icon_label:
                    item["icon_label"] = icon_label

                if verbose:
                    new_preview = (item.get("content") or "")[:120]
                    raw_preview = output.replace("\n", " ")[:160]
                    print(
                        f"[OLLAMA][{idx+1}/{total}] OK attempt={attempt} elapsed={elapsed:.2f}s "
                        f"new='{new_preview}' icon_label='{item.get('icon_label', '')}' raw='{raw_preview}'"
                    )
                success = True
                break
            except requests.exceptions.Timeout:
                elapsed = time.time() - started
                last_error = f"timeout after {elapsed:.2f}s"
                if verbose:
                    print(f"[OLLAMA][{idx+1}/{total}] TIMEOUT attempt={attempt} {last_error}")
            except Exception as e:
                elapsed = time.time() - started
                last_error = f"{type(e).__name__}: {e}"
                if verbose:
                    print(f"[OLLAMA][{idx+1}/{total}] ERROR attempt={attempt} elapsed={elapsed:.2f}s {last_error}")

        if not success and verbose:
            print(f"[OLLAMA][{idx+1}/{total}] GIVEUP last_error={last_error}")

    if verbose:
        print(f"[OLLAMA] Done refinement: refined={refined_count}/{total}")
    return parsed_content_list, refined_count

import time
import base64

import os
import ast
import torch
from typing import Tuple, List, Union
from torchvision.ops import box_convert
import re
from torchvision.transforms import ToPILImage
import supervision as sv
import torchvision.transforms as T
from util.box_annotator import BoxAnnotator 


def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float32
        ) 
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
            model_name_or_path, device_map=None, torch_dtype=torch.float16
        ).to(device)
    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM 
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    return {'model': model.to(device), 'processor': processor}


def get_yolo_model(model_path):
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(model_path)
    return model


@torch.inference_mode()
def get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=None, batch_size=128):
    # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except:
            continue

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"
    
    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        start = time.time()
        batch = croped_pil_image[i:i+batch_size]
        t1 = time.time()
        if model.device.type == 'cuda':
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
        if 'florence' in model.config.name_or_path:
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
        else:
            generated_ids = model.generate(**inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True, num_return_sequences=1) # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)
    
    return generated_texts



def get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox):]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
        ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = caption_model_processor['model'], caption_model_processor['processor']
    device = model.device
    messages = [{"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}] 
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i:i+batch_size]
        image_inputs = [processor.image_processor(x, return_tensors="pt") for x in images]
        inputs ={'input_ids': [], 'attention_mask': [], 'pixel_values': [], 'image_sizes': []}
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(image_inputs[i], txt, return_tensors="pt")
            inputs['input_ids'].append(input['input_ids'])
            inputs['attention_mask'].append(input['attention_mask'])
            inputs['pixel_values'].append(input['pixel_values'])
            inputs['image_sizes'].append(input['image_sizes'])
        max_len = max([x.shape[1] for x in inputs['input_ids']])
        for i, v in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = torch.cat([processor.tokenizer.pad_token_id * torch.ones(1, max_len - v.shape[1], dtype=torch.long), v], dim=1)
            inputs['attention_mask'][i] = torch.cat([torch.zeros(1, max_len - v.shape[1], dtype=torch.long), inputs['attention_mask'][i]], dim=1)
        inputs_cat = {k: torch.concatenate(v).to(device) for k, v in inputs.items()}

        generation_args = { 
            "max_new_tokens": 25, 
            "temperature": 0.01, 
            "do_sample": False, 
        } 
        generate_ids = model.generate(**inputs_cat, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
        # # remove input tokens 
        generate_ids = generate_ids[:, inputs_cat['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = [res.strip('\n').strip() for res in response]
        generated_texts.extend(response)

    return generated_texts

def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            # keep the smaller box
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                # only add the box if it does not overlap with any ocr bbox
                if not any(IoU(box1, box3) > iou_threshold and not is_inside(box1, box3) for k, box3 in enumerate(ocr_bbox)):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1): # ocr inside icon
                            # box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            # break
                        elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1)
    return filtered_boxes # torch.tensor(filtered_boxes)


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def predict(model, image, caption, box_threshold, text_threshold):
    """ Use huggingface model to replace the original model
    """
    model, processor = model['model'], model['processor']
    device = model.device

    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold, # 0.4,
        text_threshold=text_threshold, # 0.3,
        target_sizes=[image.size[::-1]]
    )[0]
    boxes, logits, phrases = results["boxes"], results["scores"], results["labels"]
    return boxes, logits, phrases


def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    if scale_img:
        result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        iou=iou_threshold, # default 0.7
        )
    else:
        result = model.predict(
        source=image,
        conf=box_threshold,
        iou=iou_threshold, # default 0.7
        )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area

def get_som_labeled_img(image_source: Union[str, Image.Image], model=None, BOX_TRESHOLD=0.01, output_coord_in_ratio=False, ocr_bbox=None, text_scale=0.4, text_padding=5, draw_bbox_config=None, caption_model_processor=None, ocr_text=[], use_local_semantics=True, iou_threshold=0.9,prompt=None, scale_img=False, imgsz=None, batch_size=128):
    """Process either an image path or Image object
    
    Args:
        image_source: Either a file path (str) or PIL Image object
        ...
    """
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    image_source = image_source.convert("RGB") # for CLIP
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)
    # print('image size:', w, h)
    xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None

    ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0] 
    xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
    filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
    
    # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
    # get the index of the first 'content': None
    starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
    filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
    print('len(filtered_boxes):', len(filtered_boxes), starting_idx)

    # get parsed icon local semantics
    time1 = time.time()
    if use_local_semantics:
        caption_model = caption_model_processor['model']
        if 'phi3_v' in caption_model.config.model_type: 
            parsed_content_icon = get_parsed_content_icon_phi3v(filtered_boxes, ocr_bbox, image_source, caption_model_processor)
        else:
            parsed_content_icon = get_parsed_content_icon(filtered_boxes, starting_idx, image_source, caption_model_processor, prompt=prompt,batch_size=batch_size)
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        # fill the filtered_boxes_elem None content with parsed_content_icon in order
        for i, box in enumerate(filtered_boxes_elem):
            if box['content'] is None:
                box['content'] = parsed_content_icon.pop(0)
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text
    print('time to get parsed content:', time.time()-time1)

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]
    
    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)
    
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, filtered_boxes_elem


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def get_xywh_yolo(input):
    x, y, w, h = input[0], input[1], input[2] - input[0], input[3] - input[1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def check_ocr_box(image_source: Union[str, Image.Image], display_img = True, output_bb_format='xywh', goal_filtering=None, easyocr_args=None, use_paddleocr=False):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size
    if use_paddleocr:
        if easyocr_args is None:
            text_threshold = 0.5
        else:
            text_threshold = easyocr_args['text_threshold']
        result = paddle_ocr.ocr(image_np, cls=False)[0]
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        #  matplotlib expects RGB
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering