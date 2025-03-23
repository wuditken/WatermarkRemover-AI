import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
from enum import Enum
from loguru import logger

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

app = Flask(__name__)

# Global variables to store models
florence_model = None
florence_processor = None
model_manager = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Base directory for image files
#IMAGE_DIR = "/data/imgdata"
IMAGE_DIR = "/Users/wudi/conpany/aitootls/WatermarkRemover-AI"

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    """Detect bounding box for objects and OCR text"""

def identify(task_prompt: TaskType, image: MatLike, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    if not isinstance(task_prompt, TaskType):
        raise ValueError(f"task_prompt must be a TaskType, but {task_prompt} is of type {type(task_prompt)}")

    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt.value, image_size=(image.width, image.height)
    )

def get_watermark_mask(image: MatLike, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = identify(task_prompt, image, text_input, model, processor, device)

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        image_area = image.width * image.height
        for bbox in parsed_answer[detection_key]["bboxes"]:
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)
            if (bbox_area / image_area) * 100 <= max_bbox_percent:
                draw.rectangle([x1, y1, x2, y2], fill=255)
            else:
                logger.warning(f"Skipping large bounding box: {bbox} covering {bbox_area / image_area:.2%} of the image")

    return mask

def process_image_with_lama(image: MatLike, mask: MatLike, model_manager: ModelManager):
    config = Config(
        ldm_steps=50,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_crop_margin=64,
        hd_strategy_crop_trigger_size=800,
        hd_strategy_resize_limit=1600,
    )
    result = model_manager(image, mask, config)

    if result.dtype in [np.float64, np.float32]:
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def make_region_transparent(image: Image.Image, mask: Image.Image):
    image = image.convert("RGBA")
    mask = mask.convert("L")
    transparent_image = Image.new("RGBA", image.size)
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) > 0:
                transparent_image.putpixel((x, y), (0, 0, 0, 0))
            else:
                transparent_image.putpixel((x, y), image.getpixel((x, y)))
    return transparent_image

def load_models():
    global florence_model, florence_processor, model_manager, device
    
    logger.info(f"Using device: {device}")
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")
    
    model_manager = ModelManager(name="lama", device=device)
    logger.info("LaMa model loaded")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": device})

@app.route('/process', methods=['POST'])
def process_image():
    # Check if models are loaded
    global florence_model, florence_processor, model_manager
    if florence_model is None or florence_processor is None or model_manager is None:
        return jsonify({"error": "Models not loaded yet"}), 503
    
    # Get parameters from request
    if not request.json:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Get image filenames
    if 'image_files' not in request.json:
        return jsonify({"error": "No image_files provided"}), 400
    
    image_files = request.json['image_files']
    if not isinstance(image_files, list):
        image_files = [image_files]
    
    # Get options
    transparent = request.json.get('transparent', False)
    max_bbox_percent = float(request.json.get('max_bbox_percent', 10.0))
    output_format = request.json.get('format', 'PNG').upper()
    if output_format not in ["PNG", "WEBP", "JPG", "JPEG"]:
        output_format = "PNG"
    
    results = []
    
    for image_file in image_files:
        try:
            # Construct full path
            image_path = os.path.join(IMAGE_DIR, image_file)
            
            # Check if file exists
            if not os.path.exists(image_path):
                results.append({
                    "filename": image_file,
                    "error": f"File not found: {image_path}"
                })
                continue
            
            # Open image file
            image = Image.open(image_path).convert("RGB")
            
            # Process image
            start_time = time.time()
            
            # Get mask for watermark
            mask_image = get_watermark_mask(image, florence_model, florence_processor, device, max_bbox_percent)
            
            # Process based on transparent option
            if transparent:
                result_image = make_region_transparent(image, mask_image)
                # Force PNG for transparency
                current_format = "PNG"
            else:
                lama_result = process_image_with_lama(np.array(image), np.array(mask_image), model_manager)
                result_image = Image.fromarray(cv2.cvtColor(lama_result, cv2.COLOR_BGR2RGB))
                current_format = output_format
            
            # Convert to output format
            if current_format == "JPG":
                current_format = "JPEG"
            
            # Create output filename
            file_base, file_ext = os.path.splitext(image_file)
            output_file = f"{file_base}_processed.{current_format.lower()}"
            output_path = os.path.join(IMAGE_DIR, output_file)
            
            # Save processed image
            result_image.save(output_path, format=current_format)
            
            processing_time = time.time() - start_time
            
            results.append({
                "original_filename": image_file,
                "processed_filename": output_file,
                "output_path": output_path,
                "format": current_format.lower(),
                "processing_time": processing_time,
                "transparent": transparent
            })
            
        except Exception as e:
            logger.error(f"Error processing image {image_file}: {str(e)}")
            results.append({
                "filename": image_file,
                "error": f"Error processing image: {str(e)}"
            })
    
    return jsonify({
        "results": results
    })

if __name__ == "__main__":
    # Load models before starting server
    load_models()
    
    # Create image directory if it doesn't exist
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Start Flask server
    app.run(host='0.0.0.0', port=9000, debug=False)