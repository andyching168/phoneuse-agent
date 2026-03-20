# ocr_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum

try:
    from mistral_common.protocol.instruct import request as mistral_request

    if not hasattr(mistral_request, "ReasoningEffort"):
        class ReasoningEffort(str, Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        mistral_request.ReasoningEffort = ReasoningEffort
except Exception:
    pass

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch, base64, tempfile, os

app = FastAPI()
MODEL_PATH = "zai-org/GLM-OCR"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)
print("Model ready!")

class OCRRequest(BaseModel):
    image_path: str | None = None   # 本地路徑
    image_b64: str | None = None    # base64 字串

@app.post("/ocr")
def ocr(req: OCRRequest):
    if req.image_path:
        img_src = req.image_path
        tmp = None
    elif req.image_b64:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(base64.b64decode(req.image_b64))
        tmp.close()
        img_src = tmp.name
    else:
        raise HTTPException(400, "Need image_path or image_b64")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "url": img_src},
            {"type": "text", "text": "Screenshot Recognition:"}
        ]
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    result = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False
    )
    if tmp:
        os.unlink(tmp.name)
    return {"result": result}
