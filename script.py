import cv2
import os
import subprocess
from PIL import Image
import easyocr
from spellchecker import SpellChecker
import numpy as np
import webcolors
from collections import Counter
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import tensorflow as tf
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# constants
BARRIER = "********\n"

# Check if a model is in the cache
def is_model_downloaded(model_name, cache_directory):
    model_path = os.path.join(cache_directory, model_name.replace('/', '_'))
    return os.path.exists(model_path)

# Convert color to the closest name
def closest_colour(requested_colour):
    min_colours = {}
    css3_names = webcolors.names("css3")
    for name in css3_names:
        hex_value = webcolors.name_to_hex(name, spec='css3')
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        distance = rd + gd + bd
        min_colours[distance] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    """
    Returns a tuple: (exact_name, closest_name).
    If an exact match fails, 'exact_name' is None, use the 'closest_name' fallback.
    """
    try:
        actual_name = webcolors.rgb_to_name(requested_colour, spec='css3')
        closest_name = actual_name
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def get_most_frequent_color(pixels, bin_size=10):
    """
    Returns the most frequent color among the given pixels,
    using a binning approach (default bin size=10).
    """
    bins = np.arange(0, 257, bin_size)
    r_bins = np.digitize(pixels[:, 0], bins) - 1
    g_bins = np.digitize(pixels[:, 1], bins) - 1
    b_bins = np.digitize(pixels[:, 2], bins) - 1
    combined_bins = r_bins * 10000 + g_bins * 100 + b_bins
    bin_counts = Counter(combined_bins)
    most_common_bin = bin_counts.most_common(1)[0][0]

    r_bin = most_common_bin // 10000
    g_bin = (most_common_bin % 10000) // 100
    b_bin = most_common_bin % 100
    r_value = bins[r_bin] + bin_size // 2
    g_value = bins[g_bin] + bin_size // 2
    b_value = bins[b_bin] + bin_size // 2

    return (r_value, g_value, b_value)

def get_most_frequent_alpha(alphas, bin_size=10):
    bins = np.arange(0, 257, bin_size)
    alpha_bins = np.digitize(alphas, bins) - 1
    bin_counts = Counter(alpha_bins)
    most_common_bin = bin_counts.most_common(1)[0][0]
    alpha_value = bins[most_common_bin] + bin_size // 2
    return alpha_value

# downscale images for OCR. TODO change dim to a suitable one
def downscale_for_ocr(image_cv, max_dim=600):
    """
    If either dimension of `image_cv` is bigger than `max_dim`,
    scale it down proportionally. This speeds up EasyOCR on large images.
    """
    h, w = image_cv.shape[:2]
    if w <= max_dim and h <= max_dim:
        return image_cv  # No downscale needed

    scale = min(max_dim / float(w), max_dim / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_resized

# Worker function to process a single bounding box
def process_single_region(
    idx, bounding_box, image, sr, reader, spell, icon_model,
    processor, model, device, no_captioning, output_json,
    cropped_imageview_images_dir, base_name, save_images,
    model_to_use, log_prefix=""
):
    """
    Processes one bounding box (region)
    Returns a dict with:
      * "region_dict" (for JSON)
      * "text_log" (file/captions output)
    """
    (x_min, y_min, x_max, y_max, class_id) = bounding_box
    class_names = {0: 'View', 1: 'ImageView', 2: 'Text', 3: 'Line'}
    class_name = class_names.get(class_id, f'Unknown Class {class_id}')
    region_idx = idx + 1

    # collect text output in a list, then join at the end
    logs = []

    logs.append(f"\n{log_prefix}Region {region_idx} - Class ID: {class_id} ({class_name})")
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    logs.append(f"{log_prefix}Coordinates: x_center={x_center}, y_center={y_center}")
    width = x_max - x_min
    height = y_max - y_min
    logs.append(f"{log_prefix}Size: width={width}, height={height}")

    region_dict = {
        "id": f"region_{region_idx}_class_{class_name}",
        "x_coordinates_center": x_center,
        "y_coordinates_center": y_center,
        "width": width,
        "height": height
    }

    # Crop region
    cropped_image_region = image[y_min:y_max, x_min:x_max]
    if cropped_image_region.size == 0:
        logs.append(f"{log_prefix}Empty crop for Region {region_idx}, skipping...")
        return {"region_dict": region_dict, "text_log": "\n".join(logs)}

    # Save cropped region
    if class_id == 0:
        # Save as PNG if it's a View
        cropped_path = os.path.join(
            cropped_imageview_images_dir, f"region_{region_idx}_class_{class_id}.png"
        )
        cv2.imwrite(cropped_path, cropped_image_region)
    else:
        # Save as JPG
        cropped_path = os.path.join(
            cropped_imageview_images_dir, f"region_{region_idx}_class_{class_id}.jpg"
        )
        cv2.imwrite(cropped_path, cropped_image_region)


    def open_and_upscale_image(img_path, cid):

        if cid == 2: # Text 
            MAX_WIDTH, MAX_HEIGHT = 30, 30
        else:
            MAX_WIDTH, MAX_HEIGHT = 10, 10

        def is_small(w, h):
            return w <= MAX_WIDTH and h <= MAX_HEIGHT

        if cid == 0:  # "View" - use PIL to preserve alpha
            pil_image = Image.open(img_path).convert("RGBA")
            w, h = pil_image.size
            if not is_small(w, h):
                logs.append(f"{log_prefix}Skipping upscale for large View (size={w}×{h}).")
                return pil_image

            # If super-resolution is provided, use it
            if sr:
                image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
                upscaled = sr.upsample(image_cv)
                return Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGBA))
            else:
                return pil_image.resize((w * 4, h * 4), resample=Image.BICUBIC)
        else:
            # For other classes, load the image with OpenCV (BGR)
            cv_image = cv2.imread(img_path)
            if cv_image is None or cv_image.size == 0:
                logs.append(f"{log_prefix}Empty image at {img_path}, skipping.")
                return None

            h, w = cv_image.shape[:2]
            if not is_small(w, h):
                logs.append(f"{log_prefix}Skipping upscale for large region (size={w}×{h}).")
                return cv_image

            if sr:
                return sr.upsample(cv_image)
            else:
                return cv2.resize(cv_image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # for LLaMA (ollama)
    def call_ollama(prompt_text, rid, task_type):
        model_name = "llama3.2-vision:11b"
        cmd = ["ollama", "run", model_name, prompt_text]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                logs.append(f"{log_prefix}Error generating {task_type} for Region {rid}: {result.stderr}")
                return None
            else:
                response = result.stdout.strip()
                logs.append(f"{log_prefix}Generated {task_type.capitalize()} for Region {rid}: {response}")
                return response
        except Exception as e:
            logs.append(f"{log_prefix}An error occurred while generating {task_type} for Region {rid}: {e}")
            return None

    # for BLIP-2
    def generate_caption_blip(img_path):
        pil_image = Image.open(img_path).convert('RGB')
        inputs = processor(images=pil_image, return_tensors="pt").to(device, torch.float16)
        gen_ids = model.generate(**inputs)
        return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # Handle each class type
    if class_id == 1:  # ImageView
        if no_captioning:
            logs.append(f"{log_prefix}(Icon-image detection + captioning disabled by --no-captioning.)")
            if not output_json:
                block = (
                    f"Image: region_{region_idx}_class_{class_id} ({class_name})\n"
                    f"Coordinates: x_center={(x_min + x_max) // 2}, y_center={(y_min + y_max) // 2}\n"
                    f"Size: width={width}, height={height}\n"
                    f"{BARRIER}"
                )
                logs.append(block)
        else:
            upscaled = open_and_upscale_image(cropped_path, class_id)
            if upscaled is None:
                return {"region_dict": region_dict, "text_log": "\n".join(logs)}

            # Icon detection
            if icon_model:
                icon_input_size = (224, 224)
                if isinstance(upscaled, Image.Image):
                    upscaled_cv = cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGBA2BGR)
                else:
                    upscaled_cv = upscaled
                resized = cv2.resize(upscaled_cv, icon_input_size)
                rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
                rgb_img = np.expand_dims(rgb_img, axis=0)
                pred = icon_model.predict(rgb_img)
                logs.append(f"{log_prefix}Prediction output for Region {region_idx}: {pred}")
                if pred.shape == (1, 1):
                    probability = pred[0][0]
                    threshold = 0.5
                    predicted_class = 1 if probability >= threshold else 0
                    logs.append(f"{log_prefix}Probability of class 1: {probability}")
                elif pred.shape == (1, 2):
                    predicted_class = np.argmax(pred[0])
                    logs.append(f"{log_prefix}Class probabilities: {pred[0]}")
                else:
                    logs.append(f"{log_prefix}Unexpected prediction shape: {pred.shape}")
                    return {"region_dict": region_dict, "text_log": "\n".join(logs)}

                pred_text = "Icon/Mobile UI Element" if predicted_class == 1 else "Normal Image"
                region_dict["prediction"] = pred_text
                if predicted_class == 1:
                    prompt_text = "Describe the mobile UI element on this image. Keep it short."
                else:
                    prompt_text = "Describe what is in the image briefly. It's not an icon or typical UI element."
            else:
                logs.append(f"{log_prefix}Icon detection model not provided; skipping icon detection.")
                region_dict["prediction"] = "Icon detection skipped"
                prompt_text = "Describe what is in this image briefly."

            # Caption
            temp_image_path = os.path.abspath(
                os.path.join(cropped_imageview_images_dir, f"imageview_{region_idx}.jpg")
            )
            if isinstance(upscaled, Image.Image): # TODO check optimization
                upscaled.save(temp_image_path)
            else:
                cv2.imwrite(temp_image_path, upscaled)

            response = ""
            if model and processor and model_to_use == 'blip':
                response = generate_caption_blip(temp_image_path)
            else: # TODO check optimization
                resp = call_ollama(prompt_text + " " + temp_image_path, region_idx, "description")
                response = resp if resp else "Error generating description"

            region_dict["description"] = response

            if not output_json:
                block = (
                    f"Image: region_{region_idx}_class_{class_id} ({class_name})\n"
                    f"Coordinates: x_center={(x_min + x_max) // 2}, y_center={(y_min + y_max) // 2}\n"
                    f"Size: width={width}, height={height}\n"
                    f"Prediction: {region_dict['prediction']}\n"
                    f"{response}\n"
                    f"{BARRIER}"
                )
                logs.append(block)

            if os.path.exists(temp_image_path) and not save_images:
                os.remove(temp_image_path)

    elif class_id == 2:  # Text
        upscaled = open_and_upscale_image(cropped_path, class_id)
        if upscaled is None:
            return {"region_dict": region_dict, "text_log": "\n".join(logs)}

        if isinstance(upscaled, Image.Image):
            upscaled_cv = cv2.cvtColor(np.array(upscaled), cv2.COLOR_RGBA2BGR)
        else:
            upscaled_cv = upscaled


        # TODO use other lib to improve the performance
        upscaled_cv = downscale_for_ocr(upscaled_cv, max_dim=600)
        gray = cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2GRAY)
        result_ocr = reader.readtext(gray, detail=0, batch_size=8)
        text = ' '.join(result_ocr).strip()

        # TODO use other lib to improve performance
        correction_cache = {}
        corrected_words = []
        for w in text.split():
            if w not in correction_cache:
                correction_cache[w] = spell.correction(w) or w
            corrected_words.append(correction_cache[w])
        corrected_text = ' '.join(corrected_words)


        logs.append(f"{log_prefix}Extracted Text for Region {region_idx}: {text}")
        logs.append(f"{log_prefix}Corrected Text for Region {region_idx}: {corrected_text}")

        region_dict["extractedText"] = text
        region_dict["correctedText"] = corrected_text

        if not output_json:
            block = (
                f"Text: region_{region_idx}_class_{class_id} ({class_name})\n"
                f"Coordinates: x_center={(x_min + x_max) // 2}, y_center={(y_min + y_max) // 2}\n"
                f"Size: width={width}, height={height}\n"
                f"Extracted Text: {text}\n"
                f"Corrected Text: {corrected_text}\n"
                f"{BARRIER}"
            )
            logs.append(block)

    elif class_id == 0:  # View
        upscaled = open_and_upscale_image(cropped_path, class_id)
        if upscaled is None:
            return {"region_dict": region_dict, "text_log": "\n".join(logs)}

        data = np.array(upscaled)
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGRA)
        elif data.shape[-1] == 3:
            b, g, r = cv2.split(data)
            a = np.full_like(b, 255)
            data = cv2.merge((b, g, r, a))

        pixels = data.reshape((-1, 4))
        opaque_pixels = pixels[pixels[:, 3] > 0]

        if len(opaque_pixels) == 0:
            logs.append(f"{log_prefix}No opaque pixels found in Region {region_idx}, cannot determine background color.")
            color_name = "Unknown"
        else:
            dom_color = get_most_frequent_color(opaque_pixels[:, :3], bin_size=10)
            exact_name, closest_name = get_colour_name(dom_color)
            color_name = exact_name if exact_name else closest_name

        alphas = pixels[:, 3]
        dominant_alpha = get_most_frequent_alpha(alphas, bin_size=10)
        transparency = "opaque" if dominant_alpha >= 245 else "transparent"

        response = (
            f"1. The background color of the container is {color_name}.\n"
            f"2. The container is {transparency}."
        )
        logs.append(f"{log_prefix}{response}")
        region_dict["view_color"] = f"The background color of the container is {color_name}."
        region_dict["view_alpha"] = f"The container is {transparency}."

        if not output_json:
            block = (
                f"View: region_{region_idx}_class_{class_id} ({class_name})\n"
                f"Coordinates: x_center={(x_min + x_max) // 2}, y_center={(y_min + y_max) // 2}\n"
                f"Size: width={width}, height={height}\n"
                f"{response}\n"
                f"{BARRIER}"
            )
            logs.append(block)

    elif class_id == 3:  # Line
        logs.append(f"{log_prefix}Processing Line in Region {region_idx}")
        line_img = cv2.imread(cropped_path, cv2.IMREAD_UNCHANGED)
        if line_img is None:
            logs.append(f"{log_prefix}Failed to read image at {cropped_path}")
            return {"region_dict": region_dict, "text_log": "\n".join(logs)}

        hh, ww = line_img.shape[:2]
        logs.append(f"{log_prefix}Image dimensions: width={ww}, height={hh}")

        data = np.array(line_img)
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGRA)
        elif data.shape[-1] == 3:
            b, g, r = cv2.split(data)
            a = np.full_like(b, 255)
            data = cv2.merge((b, g, r, a))

        pixels = data.reshape((-1, 4))
        opaque_pixels = pixels[pixels[:, 3] > 0]

        if len(opaque_pixels) == 0:
            logs.append(f"{log_prefix}No opaque pixels found in Region {region_idx}, cannot determine line color.")
            color_name = "Unknown"
        else:
            dom_color = get_most_frequent_color(opaque_pixels[:, :3], bin_size=10)
            exact_name, closest_name = get_colour_name(dom_color)
            color_name = exact_name if exact_name else closest_name

        alphas = pixels[:, 3]
        dom_alpha = get_most_frequent_alpha(alphas, bin_size=10)
        transparency = "opaque" if dom_alpha >= 245 else "transparent"

        response = (
            f"1. The color of the line is {color_name}.\n"
            f"2. The line is {transparency}."
        )
        logs.append(f"{log_prefix}{response}")
        region_dict["line_color"] = f"The color of the line is {color_name}."
        region_dict["line_alpha"] = f"The line is {transparency}."

        if not output_json:
            block = (
                f"Line: region_{region_idx}_class_{class_id} ({class_name})\n"
                f"Coordinates: x_center={(x_min + x_max) // 2}, y_center={(y_min + y_max) // 2}\n"
                f"Size: width={width}, height={height}\n"
                f"{response}\n"
                f"{BARRIER}"
            )
            logs.append(block)

    else:
        logs.append(f"{log_prefix}Class ID {class_id} not handled.")

    # Remove intermediate if not saving
    if os.path.exists(cropped_path) and not save_images:
        os.remove(cropped_path)

    return {
        "region_dict": region_dict,
        "text_log": "\n".join(logs),
    }


# Main function
def process_image(
    input_image_path,
    yolo_output_path,
    model_to_use='llama',     
    save_images=False,
    icon_model_path=None,
    cache_directory='./models_cache',
    huggingface_token='your_token', # for blip2
    no_captioning=False,
    output_json=False,
    sr=None,
    reader=None,
    spell=None
):
    # Prepare JSON output dictionary
    json_output = {
        "image": {
            "path": input_image_path,
            "size": {
                "width": None,
                "height": None
            }
        },
        "elements": []
    }

    start_time = time.perf_counter()
    print("super-resolution initialization start (in script.py)")
    # Super-resolution initialization
    if sr is None:
        print("No sr reference passed; performing local init ...")
        model_path = 'EDSR_x4.pb'
        if hasattr(cv2, 'dnn_superres'):
            print("dnn_superres module is available.")
            import cv2.dnn_superres as dnn_superres
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                print("Using DnnSuperResImpl_create()")
            except AttributeError:
                sr = cv2.dnn_superres.DnnSuperResImpl()
                print("Using DnnSuperResImpl()")
            sr.readModel(model_path)
            sr.setModel('edsr', 4)
        else:
            print("dnn_superres module is NOT available; skipping super-resolution.")
    else:
        print("Using pre-initialized sr reference.")


    elapsed = time.perf_counter() - start_time
    print(f"super-resoulution init (in script.py) took {elapsed:.3f} seconds.")


    start_time = time.perf_counter()
    print("OCR init start (in script.py)")
    if reader is None:
        print("No EasyOCR reference passed; performing local init ...")
        reader = easyocr.Reader(['en'], gpu=True)
    else:
        print("Using pre-initialized EasyOCR reference.")

    if spell is None:
        print("No SpellChecker reference passed; performing local init ...")
        spell = SpellChecker()
    else:
        print("Using pre-initialized SpellChecker reference.")
    elapsed = time.perf_counter() - start_time
    print(f"OCR init (in script.py) took {elapsed:.3f} seconds.")


    start_time = time.perf_counter()
    print("icon-model init start (in script.py)")
    # Load icon detection model (if provided)
    if icon_model_path:
        icon_model = tf.keras.models.load_model(icon_model_path)
        print(f"Icon detection model loaded: {icon_model_path}")
    else:
        icon_model = None

    elapsed = time.perf_counter() - start_time
    print(f"icon-model init (in script.py) took {elapsed:.3f} seconds.")

    
    # Load the original image
    image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Image at {input_image_path} could not be loaded.")
        return

    image_height, image_width = image.shape[:2]


    # Read YOLO labels
    with open(yolo_output_path, 'r') as f:
        lines = f.readlines()

    # Check torch device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Conditionally load the captioning model
    processor, model = None, None
    if not no_captioning:
        if model_to_use == 'blip':
            print("Loading BLIP-2 model...")
            blip_model_name = "Salesforce/blip2-opt-2.7b"
            if not is_model_downloaded(blip_model_name, cache_directory):
                print("Model not found in cache. Downloading...")
            else:
                print("Model found in cache. Loading...")
            processor = AutoProcessor.from_pretrained(
                blip_model_name,
                use_auth_token=huggingface_token,
                cache_dir=cache_directory,
                resume_download=True
            )
            model = Blip2ForConditionalGeneration.from_pretrained(
                blip_model_name,
                device_map='auto',
                torch_dtype=torch.float16,
                use_auth_token=huggingface_token,
                cache_dir=cache_directory,
                resume_download=True
            ).to(device)
        else:
            print("Using LLaMA model via external call (ollama).")
    else:
        print("--no-captioning flag is set; skipping model loading.")

    # Prepare bounding boxes from YOLO
    bounding_boxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
        x_center = x_center_norm * image_width
        y_center = y_center_norm * image_height
        box_width = width_norm * image_width
        box_height = height_norm * image_height
        x_min = int(x_center - box_width / 2)
        y_min = int(y_center - box_height / 2)
        x_max = int(x_center + box_width / 2)
        y_max = int(y_center + box_height / 2)
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width - 1, x_max)
        y_max = min(image_height - 1, y_max)
        bounding_boxes.append((x_min, y_min, x_max, y_max, class_id))


    # Create output dirs
    cropped_dir = "cropped_imageview_images"
    os.makedirs(cropped_dir, exist_ok=True)
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    captions_filename = f"{base_name}_regions_captions.txt"
    captions_file_path = os.path.join(result_dir, captions_filename)


    # Write initial image info
    def put_image_size_in_output_file():
        h, w = image.shape[:2]
        if output_json:
            json_output["image"]["size"]["width"] = w
            json_output["image"]["size"]["height"] = h
        else:
            with open(captions_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Image path: {input_image_path}\n")
                f.write(f"Image Size: width={w}, height={h}\n")
                f.write(BARRIER)
        print(f"Image path: {input_image_path}")
        print(f"Image Size: width={w}, height={h}")
        print(BARRIER)

    put_image_size_in_output_file()

    # Number of workers can be increased if hardware is suitable for it. But testing is needed
    start_time = time.perf_counter()
    print("Process single region start (in script.py)")
    all_results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures_map = {}
        for idx, box in enumerate(bounding_boxes):
            future = executor.submit(
                process_single_region,
                idx, box, image, sr, reader, spell,
                icon_model, processor, model, device,
                no_captioning, output_json,
                cropped_dir, base_name, save_images,
                model_to_use, log_prefix=""
            )
            futures_map[future] = idx

        for future in as_completed(futures_map):
            item = future.result()  # { "region_dict":..., "text_log":... }
            all_results.append(item)

    elapsed = time.perf_counter() - start_time
    print(f"Process single region (in script.py) took {elapsed:.3f} seconds.")

    # Finalyzing JSON or captions text
    if output_json:
        for item in all_results:
            region_info = item["region_dict"]
            json_output["elements"].append(region_info)
        json_file = os.path.join(result_dir, f"{base_name}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"JSON output written to {json_file}")
    else:
        with open(captions_file_path, 'a', encoding='utf-8') as f:
            for item in all_results:
                f.write(item["text_log"])

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image and its YOLO labels.')
    parser.add_argument('input_image', help='Path to the input YOLO image.')
    parser.add_argument('input_labels', help='Path to the input YOLO labels file.')
    parser.add_argument('--model_to_use', choices=['llama', 'blip'], default='llama',
                        help='Model for captioning (llama or blip).')
    parser.add_argument('--save_images', action='store_true',
                        help='Flag to save intermediate images.')
    parser.add_argument('--icon_detection_path', help='Path to icon detection model.')
    parser.add_argument('--cache_directory', default='./models_cache',
                        help='Cache directory for Hugging Face models.')
    parser.add_argument('--huggingface_token', default='your_token',
                        help='Hugging Face token for model downloads.')
    parser.add_argument('--no-captioning', action='store_true',
                        help='Disable any image captioning.')
    parser.add_argument('--json', dest='output_json', action='store_true',
                        help='Output the image data in JSON format')
    args = parser.parse_args()

    process_image(
        input_image_path=args.input_image,
        yolo_output_path=args.input_labels,
        model_to_use=args.model_to_use,
        save_images=args.save_images,
        icon_model_path=args.icon_detection_path,
        cache_directory=args.cache_directory,
        huggingface_token=args.huggingface_token,
        no_captioning=args.no_captioning,
        output_json=args.output_json
    )
