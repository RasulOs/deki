# %%
import cv2
import os
import subprocess
from PIL import Image
import easyocr
from spellchecker import SpellChecker
import numpy as np  
import webcolors  # Import webcolors for color name conversion
from collections import Counter
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import tensorflow as tf  
import argparse  # command-line argument parsing
import json

# %%
# Define constants
BARRIER = "********\n"

# %%
# Initialize the argument parser
parser = argparse.ArgumentParser(description='Process an image and its YOLO labels.')
parser.add_argument('input_image', help='Path to the input YOLO image.')
parser.add_argument('input_labels', help='Path to the input YOLO labels file.')
parser.add_argument('--model_to_use', choices=['llama', 'blip'], default='llama', help='Model to use for captioning (default: llama).')
parser.add_argument('--save_images', action='store_true', help='Flag to save intermediate images.')
parser.add_argument('--icon_detection_path', help='Path to the icon detection model. If not provided, icon detection will be skipped.')
parser.add_argument('--cache_directory', default='./models_cache', help='Cache directory for models.')
parser.add_argument('--huggingface_token', default='your_token', help='Hugging Face token for model downloads.')
parser.add_argument('--no-captioning', action='store_true', help='Disable any image captioning.')
parser.add_argument('--json', dest='output_json', action='store_true', help='Output the image data in JSON format')

args = parser.parse_args()

# Assign arguments to variables
input_image_path = args.input_image
yolo_output_path = args.input_labels
model_to_use = args.model_to_use
save_images = args.save_images
icon_model_path = args.icon_detection_path
cache_directory = args.cache_directory
huggingface_token = args.huggingface_token
no_captioning = args.no_captioning
output_json = args.output_json # bool

# %%
# Store "image" info + "elements".
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

# %%
# Initialize the super-resolution model if available
model_path = 'EDSR_x4.pb'  # Ensure this file is in the same directory

print("OpenCV version:", cv2.__version__)

# Check if dnn_superres module is available
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
    print("dnn_superres module is NOT available.")
    sr = None  # Super-resolution not available

# %%
# Initialize EasyOCR and SpellChecker
reader = easyocr.Reader(['en'])  # Add more languages if needed
spell = SpellChecker()

# %%
# Load the icon detection model (if it was provided)
if icon_model_path:
    icon_model = tf.keras.models.load_model(icon_model_path)
else:
    icon_model = None

# %%
# Load the original image
image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED to load alpha channel
if image is None:
    print(f"Image at {input_image_path} could not be loaded.")
    exit(1)
image_height, image_width = image.shape[:2]

# Read the YOLO output file
with open(yolo_output_path, 'r') as f:
    lines = f.readlines()

# %%
# Check for device compatibility
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# %%
# Function to check if a model is downloaded in the cache
def is_model_downloaded(model_name, cache_directory):
    model_path = os.path.join(cache_directory, model_name.replace('/', '_'))
    return os.path.exists(model_path)

# Conditionally load the captioning model only if no-captioning is False
if not no_captioning:
    if model_to_use == 'blip':
        print("Loading BLIP-2 model...")
        model_name = "Salesforce/blip2-opt-2.7b"

        if not is_model_downloaded(model_name, cache_directory):
            print("Model not found in cache. Downloading...")
        else:
            print("Model found in cache. Loading...")

        # Load the processor and model
        processor = AutoProcessor.from_pretrained(
            model_name,
            use_auth_token=huggingface_token,
            cache_dir=cache_directory,
            resume_download=True
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float16,
            use_auth_token=huggingface_token,
            cache_dir=cache_directory,
            resume_download=True
        ).to(device)
    else:
        print("Using LLaMA model via Ollama")
        processor = None  # no processor needed for LLaMA external call
        model = None
else:
    print("--no-captioning flag is set; skipping model loading.")
    processor = None
    model = None

# %%
# Initialize a list to store bounding boxes with their class IDs
bounding_boxes = []

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])

    # Convert normalized coordinates to absolute pixel values
    x_center = x_center_norm * image_width
    y_center = y_center_norm * image_height
    box_width = width_norm * image_width
    box_height = height_norm * image_height

    x_min = int(x_center - box_width / 2)
    y_min = int(y_center - box_height / 2)
    x_max = int(x_center + box_width / 2)
    y_max = int(y_center + box_height / 2)

    # Clamp the coordinates
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width - 1, x_max)
    y_max = min(image_height - 1, y_max)

    bounding_boxes.append((x_min, y_min, x_max, y_max, class_id))

# %%
# Create directories to save cropped images and results
cropped_imageview_images_dir = "cropped_imageview_images"
os.makedirs(cropped_imageview_images_dir, exist_ok=True)

result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# Extract the base filename (without extension) from the input image
base_name = os.path.splitext(os.path.basename(input_image_path))[0]
# Create a captions filename like "bb_1_regions_captions.txt"
captions_filename = f"{base_name}_regions_captions.txt"
# Prepare the captions file
captions_file_path = os.path.join(result_dir, captions_filename)

# %%
def open_and_upscale_image(image_path, class_id, upscale_imageview=True):
    """
    Opens an image and optionally upscales it based on the class ID and flags.
    Skips (or limits) super-resolution for very large images.
    """

    # Thresholds for deciding when to skip or limit SR
    MAX_WIDTH = 200
    MAX_HEIGHT = 200

    if class_id == 0:
        # PIL to handle images with alpha channel
        pil_image = Image.open(image_path).convert('RGBA')

        # Check if upscaling is needed
        if sr:
            # If the original image is too large, skip or limit the upscale
            if pil_image.width > MAX_WIDTH or pil_image.height > MAX_HEIGHT:
                print(f"Skipping 4× super-resolution for large View (size={pil_image.width}×{pil_image.height}).")
                upscaled_image = pil_image
            else:
                # Upscale the image using super-resolution
                image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
                upscaled_image_cv = sr.upsample(image_cv)
                upscaled_image = Image.fromarray(cv2.cvtColor(upscaled_image_cv, cv2.COLOR_BGR2RGBA))
        else:
            if pil_image.width > MAX_WIDTH or pil_image.height > MAX_HEIGHT:
                print(f"Skipping 4× super-resolution for large View (size={pil_image.width}×{pil_image.height}).")
                upscaled_image = pil_image
            else:
                upscaled_image = pil_image.resize(
                (pil_image.width * 4, pil_image.height * 4),
                resample=Image.BICUBIC
                )
        return upscaled_image

    else:
        # For other classes, read image without alpha channel
        image = cv2.imread(image_path)  # Read in BGR format

        # Check if the image is empty
        if image is None or image.size == 0:
            print(f"Empty image at {image_path}, skipping...")
            return None

        # Decide whether to upscale based on class_id and upscale_imageview flag
        if class_id == 1 and not upscale_imageview:
            # Return the original image without upscaling
            return image
        else:
            if sr:
                # Check thresholds for large images
                h, w = image.shape[:2]
                if w > MAX_WIDTH or h > MAX_HEIGHT:
                    print(f"Skipping 4× super-resolution for large region (size={w}×{h}).")
                    upscaled_image = image
                else:
                    # Use the 4× super-resolution
                    upscaled_image = sr.upsample(image)
                return upscaled_image
            else:
                # Upscale using OpenCV's resize if super-resolution is not available
                upscaled_image = cv2.resize(
                    image,
                    (image.shape[1] * 4, image.shape[0] * 4),
                    interpolation=cv2.INTER_CUBIC
                )
                return upscaled_image

# %%
# Function to call Ollama with a given prompt
def call_ollama(prompt_text, idx, task_type):
    # Prepare the command to call Ollama
    model_name = "llama3.2-vision:11b"

    # Construct the command
    command = [
        "ollama", "run", model_name,
        prompt_text
    ]

    # Execute the command
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Check for errors
        if result.returncode != 0:
            print(f"Error generating {task_type} for Region {idx+1}:")
            print(result.stderr)
            return None
        else:
            response = result.stdout.strip()
            print(f"Generated {task_type.capitalize()}:")
            print(response)
            return response
    except Exception as e:
        print(f"An error occurred while generating {task_type} for Region {idx+1}: {e}")
        return None

# %%
# Function to generate captions using BLIP-2
def generate_caption_blip(image_path):
    # Open the image
    raw_image = Image.open(image_path).convert('RGB')
    # Process the image
    inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)
    # Generate the caption
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

# %%
# Helper functions for color name conversion
def closest_colour(requested_colour):
    min_colours = {}
    # Get the list of CSS3 color names and their RGB values
    css3_names = webcolors.names("css3")
    for name in css3_names:
        hex_value = webcolors.name_to_hex(name, spec='css3')
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        distance = rd + gd + bd
        min_colours[distance] = name
    closest_name = min_colours[min(min_colours.keys())]
    return closest_name

def get_colour_name(requested_colour):
    try:
        actual_name = webcolors.rgb_to_name(requested_colour, spec='css3')
        closest_name = actual_name
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

# %%
def get_most_frequent_color(pixels, bin_size=10):
    # Bin the color values
    bins = np.arange(0, 257, bin_size)
    r_bins = np.digitize(pixels[:, 0], bins) - 1
    g_bins = np.digitize(pixels[:, 1], bins) - 1
    b_bins = np.digitize(pixels[:, 2], bins) - 1

    # Combine the bins to create a single value
    combined_bins = r_bins * 10000 + g_bins * 100 + b_bins

    # Find the most frequent bin
    bin_counts = Counter(combined_bins)
    most_common_bin = bin_counts.most_common(1)[0][0]

    # Extract the bin indices
    r_bin = most_common_bin // 10000
    g_bin = (most_common_bin % 10000) // 100
    b_bin = most_common_bin % 100

    # Calculate the average color of the most frequent bin
    r_value = bins[r_bin] + bin_size // 2
    g_value = bins[g_bin] + bin_size // 2
    b_value = bins[b_bin] + bin_size // 2

    return (r_value, g_value, b_value)

# %%
# Function to find the most frequent alpha value
def get_most_frequent_alpha(alphas, bin_size=10):
    bins = np.arange(0, 257, bin_size)
    alpha_bins = np.digitize(alphas, bins) - 1
    bin_counts = Counter(alpha_bins)
    most_common_bin = bin_counts.most_common(1)[0][0]
    alpha_value = bins[most_common_bin] + bin_size // 2
    return alpha_value

# %%
# General function to process each region based on class ID
def process_region(image_path, idx, class_id, captions_file, x_min, y_min, x_max, y_max):
    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min

    # Class names mapping
    class_names = {0: 'View', 1: 'ImageView', 2: 'Text', 3: 'Line'}

    # Get the class name
    class_name = class_names.get(class_id, f'Unknown Class {class_id}')

    # Print coordinates and sizes to the terminal
    print(f"Region {idx+1} - Class ID: {class_id} ({class_name})")
    print(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
    print(f"Size: width={width}, height={height}")

    # Build base dictionary for JSON if --json flag is passed
    region_dict = {
        "id": f"region_{idx+1}_class_{class_id}",
        "type": class_name,
        "coordinates": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        },
        "size": {
            "width": width,
            "height": height
        }
    }

    # ImageView
    if class_id == 1:
        if no_captioning:
            print(f"(Icon-image detection and image captioning disabled by --no-captioning.)")
            # For JSON, only set the type, no additional info
            if not output_json:
                with open(captions_file, 'a', encoding='utf-8') as f:
                    f.write(f"Image: region_{idx+1}_class_{class_id} ({class_name})\n")
                    f.write(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}\n")
                    f.write(f"Size: width={width}, height={height}\n")
                    f.write(BARRIER)
        else:
            upscaled_image = open_and_upscale_image(image_path, class_id)
            if upscaled_image is None:
                return

            # If an icon detection model is provided, run the prediction; otherwise, skip icon detection.
            if icon_model:
                icon_input_size = (224, 224)
                icon_image = cv2.resize(upscaled_image, icon_input_size)
                icon_image = cv2.cvtColor(icon_image, cv2.COLOR_BGR2RGB)
                icon_image = icon_image / 255.0 # Normalize
                icon_image = np.expand_dims(icon_image, axis=0)  # batch dimension

                prediction = icon_model.predict(icon_image)
                print(f"Prediction output for Region {idx+1}: {prediction}")
                print(f"Prediction shape: {prediction.shape}")

                # Interpret the prediction
                if prediction.shape == (1, 1):
                    # Sigmoid activation (probability of class 1)
                    probability = prediction[0][0]
                    threshold = 0.5  # Adjust the threshold if needed
                    predicted_class = 1 if probability >= threshold else 0
                    print(f"Probability of class 1: {probability}")
                elif prediction.shape == (1, 2):
                    # Softmax activation
                    predicted_class = np.argmax(prediction[0])
                    print(f"Class probabilities: {prediction[0]}")
                else:
                    print(f"Unexpected prediction shape: {prediction.shape}")
                    return

                print(f"Prediction for Region {idx+1}: {'Icon/Mobile UI Element' if predicted_class == 1 else 'Normal Image'}")
                prompt_text = (
                    "Describe the mobile UI element on this image. Try to be short."
                    if predicted_class == 1
                    else "Describe what is in the image. This image is not related to icons or mobile UI elements. Try to be short."
                )
                region_dict["prediction"] = "Icon/Mobile UI Element" if predicted_class == 1 else "Normal Image"
            else:
                print("Icon detection model not provided. Skipping icon detection.")
                prompt_text = "Describe what is in the image. Try to be short."
                region_dict["prediction"] = "Icon detection skipped"

            if not no_captioning:
                print(f"Using prompt: {prompt_text}")

            # Save the processed image temporarily
            temp_image_path = os.path.abspath(os.path.join(cropped_imageview_images_dir, f"imageview_{idx+1}.jpg"))
            cv2.imwrite(temp_image_path, upscaled_image)

            if not no_captioning:
                if model_to_use == 'blip':
                    response = generate_caption_blip(temp_image_path)
                else:
                    # For LLaMA, incorporate the prompt with the image path
                    prompt_with_image = prompt_text + " " + temp_image_path
                    response = call_ollama(prompt_with_image, idx, 'description')
                    if response is None:
                        response = "Error generating description"
            else:
                print(f"(Captioning disabled by --no-captioning.)")
                response = ""

            # Store the description in the JSON dictionary
            region_dict["description"] = response

            if not output_json:
                with open(captions_file, 'a', encoding='utf-8') as f:
                    f.write(f"Image: region_{idx+1}_class_{class_id} ({class_name})\n")
                    f.write(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}\n")
                    f.write(f"Size: width={width}, height={height}\n")
                    if icon_model:
                        f.write(f"Prediction: {'Icon/Mobile UI Element' if predicted_class == 1 else 'Normal Image'}\n")
                    else:
                        f.write("Prediction: Icon detection skipped\n")
                    f.write(f"{response}\n")
                    f.write(BARRIER)

            if os.path.exists(temp_image_path) and save_images == False:
                os.remove(temp_image_path)

    elif class_id == 2:
        # Text
        upscaled_image = open_and_upscale_image(image_path, class_id)
        if upscaled_image is None:
            return

        # Convert to grayscale for OCR
        gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

        # Perform OCR using EasyOCR
        result = reader.readtext(gray_image)
        text = ' '.join([res[1] for res in result]).strip()

        # Spell checking
        words = text.split()
        corrected_words = [
            spell.correction(word) if spell.correction(word) else word
            for word in words
        ]
        corrected_text = ' '.join(corrected_words)

        print(f"Extracted Text for Region {idx+1}: {text}")
        print(f"Corrected Text for Region {idx+1}: {corrected_text}")

        # For JSON
        region_dict["extractedText"] = text
        region_dict["correctedText"] = corrected_text

        if not output_json:
            with open(captions_file, 'a', encoding='utf-8') as f:
                f.write(f"Text: region_{idx+1}_class_{class_id} ({class_name})\n")
                f.write(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}\n")
                f.write(f"Size: width={width}, height={height}\n")
                f.write(f"Extracted Text: {text}\n")
                f.write(f"Corrected Text: {corrected_text}\n")
                f.write(BARRIER)

        print(f"Text for Region {idx+1} written to {captions_file}")

    elif class_id == 0:
        # View
        upscaled_image = open_and_upscale_image(image_path, class_id)
        if upscaled_image is None:
            return

        # Analyze the background color
        # Convert image data to numpy array
        data = np.array(upscaled_image)
        r, g, b, a = data.T  # Separate color channels

        # Flatten the arrays
        pixels = data.reshape((-1, 4))

        # Exclude fully transparent pixels
        opaque_pixels = pixels[pixels[:, 3] > 0]

        if len(opaque_pixels) == 0:
            print(f"No opaque pixels found in Region {idx+1}, cannot determine background color.")
            color_name = "Unknown"
        else:
            # Find the most frequent color
            dominant_color = get_most_frequent_color(opaque_pixels[:, :3], bin_size=10)
            actual_name, closest_name = get_colour_name(dominant_color)
            color_name = actual_name if actual_name else closest_name

        # Analyze transparency
        alphas = pixels[:, 3]
        # Find the most frequent alpha value
        dominant_alpha = get_most_frequent_alpha(alphas, bin_size=10)
        transparency = "opaque" if dominant_alpha >= 255 - 10 else "transparent"  # Allow a small margin

        # Prepare the response
        response = f"1. The background color of the container is {color_name}.\n"
        response += f"2. The container is {transparency}."
        print(response)

        # For JSON
        region_dict["properties"] = [
            f"The background color of the container is {color_name}.",
            f"The container is {transparency}."
        ]

        if not output_json:
            with open(captions_file, 'a', encoding='utf-8') as f:
                f.write(f"View: region_{idx+1}_class_{class_id} ({class_name})\n")
                f.write(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}\n")
                f.write(f"Size: width={width}, height={height}\n")
                f.write(f"{response}\n")
                f.write(BARRIER)

        print(f"Analysis for Region {idx+1} written to {captions_file}")

    elif class_id == 3:
        # Line
        print(f"Processing Line in Region {idx+1}")

        # Read the cropped image directly
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to read image at {image_path}")
            return

        # Get the dimensions of the cropped image
        height, width = image.shape[:2]
        print(f"Image dimensions: width={width}, height={height}")

        # Proceed to analyze the image directly without further cropping
        data = np.array(image)
        if len(data.shape) == 2:
            # Grayscale image
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGRA)
        elif data.shape[2] == 3:
            # Image has no alpha channel
            b, g, r = cv2.split(data)
            a = np.full_like(b, 255)
            data = cv2.merge((b, g, r, a))
        elif data.shape[2] == 4:
            # Image has alpha channel
            pass
        else:
            print(f"Unexpected number of channels in image: {data.shape}")
            return

        # Flatten the arrays
        pixels = data.reshape((-1, 4))

        # Exclude fully transparent pixels
        opaque_pixels = pixels[pixels[:, 3] > 0]

        if len(opaque_pixels) == 0:
            print(f"No opaque pixels found in Region {idx+1}, cannot determine line color.")
            color_name = "Unknown"
        else:
            # Find the most frequent color
            dominant_color = get_most_frequent_color(opaque_pixels[:, :3], bin_size=10)
            actual_name, closest_name = get_colour_name(dominant_color)
            color_name = actual_name if actual_name else closest_name

        # Analyze transparency
        alphas = pixels[:, 3]
        # Find the most frequent alpha value
        dominant_alpha = get_most_frequent_alpha(alphas, bin_size=10)
        transparency = "opaque" if dominant_alpha >= 255 - 10 else "transparent"  # Allow a small margin

        # Prepare the response
        response = f"1. The color of the line is {color_name}.\n"
        response += f"2. The line is {transparency}."
        print(response)

        # For JSON
        region_dict["properties"] = [
            f"The color of the line is {color_name}.",
            f"The line is {transparency}."
        ]

        if not output_json:
            with open(captions_file, 'a', encoding='utf-8') as f:
                f.write(f"Line: region_{idx+1}_class_{class_id} ({class_name})\n")
                f.write(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}\n")
                f.write(f"Size: width={width}, height={height}\n")
                f.write(f"{response}\n")
                f.write(BARRIER)

        print(f"Details for Line Region {idx+1} written to {captions_file}")

    else:
        # For other classes, we can skip or add handling as needed
        print(f"Class ID {class_id} not handled.")

    # Append region_dict to json_output["elements"] if --json
    if output_json:
        json_output["elements"].append(region_dict)

def put_image_size_in_output_file(captions_file, input_image_path):
    # Get image size
    image_cv = cv2.imread(input_image_path)
    height, width, _ = image_cv.shape

    # Store in the JSON structure if --json
    if output_json:
        json_output["image"]["size"]["width"] = width
        json_output["image"]["size"]["height"] = height
    else:
        with open(captions_file, 'w', encoding='utf-8') as f:
            f.write(f"Image path: {input_image_path}\n")
            f.write(f"Image Size: width={width}, height={height}\n")
            f.write(BARRIER)

    print(f"Image path: {input_image_path}")
    print(f"Image Size: width={width}, height={height}")
    print(BARRIER)

# %%
put_image_size_in_output_file(captions_file_path, input_image_path)

# %%
# Process each bounding box and write captions
for idx, (x_min, y_min, x_max, y_max, class_id) in enumerate(bounding_boxes):
    print(f"\nProcessing Region {idx+1}, Class ID: {class_id}")
    # Crop the region from the original image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Check if the cropped image is empty
    if cropped_image.size == 0:
        print(f"Empty crop for Region {idx+1}, skipping...")
        continue

    # Save the cropped image for processing
    if class_id == 0:
        # Save as PNG to preserve alpha channel
        cropped_image_path = os.path.join(cropped_imageview_images_dir, f"region_{idx+1}_class_{class_id}.png")
        cv2.imwrite(cropped_image_path, cropped_image)
    else:
        # Save as JPG for other classes
        cropped_image_path = os.path.join(cropped_imageview_images_dir, f"region_{idx+1}_class_{class_id}.jpg")
        cv2.imwrite(cropped_image_path, cropped_image)

    # Process the region, pass coordinates and sizes
    process_region(cropped_image_path, idx, class_id, captions_file_path, x_min, y_min, x_max, y_max)

    if os.path.exists(cropped_image_path) and save_images == False:
        os.remove(cropped_image_path)

# %%
# Dump JSON to the output file
if output_json:
    json_output_filename = os.path.join(result_dir, f"{base_name}.json")
    with open(json_output_filename, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"JSON output written to {json_output_filename}")

