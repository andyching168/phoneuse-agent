from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io


import base64, os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img, recognize_text_with_glm_ocr, refine_boxes_with_ollama, refine_boxes_with_gemini
import torch
from PIL import Image

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    use_glmocr,
    glmocr_mode,
    refinement_provider,
    imgsz
) -> Optional[Image.Image]:
    import time
    
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_input, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    
    # Refine OCR text using GLM-OCR if enabled
    if use_glmocr and parsed_content_list:
        use_api = (glmocr_mode == "API (Port 8000)")
        print(f"Refining OCR text with GLM-OCR ({'API' if use_api else 'Local Model'})...")
        w, h = image_input.size
        
        refined_count = 0
        for idx, item in enumerate(parsed_content_list):
            # Process any item with content (text or icon with OCR text)
            content = item.get('content')
            if content and isinstance(content, str) and content.strip():
                bbox = item.get('bbox')
                if bbox:
                    # Crop the region from the original image
                    x1 = int(bbox[0] * w)
                    y1 = int(bbox[1] * h)
                    x2 = int(bbox[2] * w)
                    y2 = int(bbox[3] * h)
                    
                    # Add some padding
                    pad = 2
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)
                    
                    cropped = image_input.crop((x1, y1, x2, y2))
                    
                    # Use GLM-OCR to recognize text
                    try:
                        new_text = recognize_text_with_glm_ocr(cropped, use_api=use_api)
                        if new_text:
                            old_content = item.get('content', '')
                            item['content'] = new_text
                            item['source'] = 'glm_ocr_api' if use_api else 'glm_ocr_local'
                            refined_count += 1
                            print(f"  [{refined_count}] '{old_content[:30]}' -> '{new_text[:30]}'")
                    except Exception as e:
                        print(f"  Error processing item {idx}: {e}")
        
        print(f"GLM-OCR refined {refined_count} text boxes")

        provider = (refinement_provider or "Gemini").strip().lower()
        if provider == "gemini":
            print("Running Gemini refinement for all boxes...")
            parsed_content_list, refined_count_2 = refine_boxes_with_gemini(
                image_input,
                parsed_content_list,
            )
            print(f"Gemini refined {refined_count_2} boxes")
        else:
            print("Running Ollama refinement for all boxes...")
            parsed_content_list, refined_count_2 = refine_boxes_with_ollama(
                image_input,
                parsed_content_list,
            )
            print(f"Ollama refined {refined_count_2} boxes")
    
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    w, h = image_input.size
    for item in parsed_content_list:
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            item["bbox_ratio"] = bbox
            item["bbox"] = [
                int(bbox[0] * w),
                int(bbox[1] * h),
                int(bbox[2] * w),
                int(bbox[3] * h),
            ]
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    # parsed_content_list = str(parsed_content_list)
    return image, str(parsed_content_list)

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            # set the threshold for removing the bounding boxes with large overlap, default is 0.1
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            use_paddleocr_component = gr.Checkbox(
                label='Use PaddleOCR', value=True)
            use_glmocr_component = gr.Checkbox(
                label='Use GLM-OCR (Better text recognition)', value=False)
            glmocr_mode_component = gr.Radio(
                choices=["API (Port 8000)", "Local Model"],
                value="API (Port 8000)",
                label="GLM-OCR Mode",
                info="API mode requires ocrServer.py running on port 8000"
            )
            refinement_provider_component = gr.Radio(
                choices=["Gemini", "Ollama"],
                value="Gemini",
                label="Refinement Provider",
                info="Default uses Gemini. Ollama can be used as fallback."
            )
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            use_glmocr_component,
            glmocr_mode_component,
            refinement_provider_component,
            imgsz_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# demo.launch(debug=False, show_error=True, share=True)
demo.launch(share=True, server_port=7861, server_name='127.0.0.1')
