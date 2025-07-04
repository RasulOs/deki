import os
import sys
import argparse
from ultralytics import YOLO
from os.path import basename, splitext
import time

from yolo_script import process_yolo

from script import process_image

def process_image_description(
    input_image: str,
    weights_file: str,
    output_dir: str,
    model_to_use: str = 'llama',
    save_images: bool = False,
    icon_detection_path: str = None,
    cache_directory: str = './models_cache',
    huggingface_token: str = 'your_token',
    no_captioning: bool = False,
    output_json: bool = False,
    json_mini: bool = False,
    model_obj: YOLO = None,
    sr=None,
    reader=None,
    spell=None,
    skip_ocr=False,
    skip_spell=False,
) -> None:
    """
    Processes an image by running YOLO detection (via the imported process_yolo function)
    and then calling process_image() from script.py to do the image description work.
    
    Parameters:
      - input_image: Path to the input image.
      - weights_file: Path to the YOLO weights file.
      - output_dir: Directory for YOLO output
      - model_to_use: Which model to use for captioning ('llama' or 'blip').
      - save_images: Whether to save intermediate images.
      - icon_detection_path: Optional path to an icon detection model.
      - cache_directory: Cache directory for models.
      - huggingface_token: Hugging Face token for model downloads.
      - no_captioning: If True, disable image captioning.
      - output_json: If True, output the results in JSON format.
      - json_mini: same as output_json but has more compact json output.
      - model_obj: YOLO object that was initialized at a startup time (optional)
      - sr: Super resolution object (optional)
      - reader: EasyOCR object (optional)
      - spell: Spell checker object (optional)
    """

    base_name = splitext(basename(input_image))[0]
    
    process_yolo(input_image, weights_file, output_dir, model_obj=model_obj)
    
    labels_dir = os.path.join(output_dir, 'labels')
    label_file = os.path.join(labels_dir, base_name + '.txt')

    if not os.path.isfile(label_file):
        raise FileNotFoundError(f"Labels file not found at expected path: {label_file}")

    process_image(
        input_image_path=input_image,
        yolo_output_path=label_file,
        output_dir=output_dir,
        model_to_use=model_to_use,
        save_images=save_images,
        icon_model_path=icon_detection_path,
        cache_directory=cache_directory,
        huggingface_token=huggingface_token,
        no_captioning=no_captioning,
        output_json=output_json,
        json_mini=json_mini,
        sr=sr,
        reader=reader,
        spell=spell,
        skip_ocr=skip_ocr,
        skip_spell=skip_spell,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Wrapper script to run YOLO detection and image description in sequence.'
    )
    parser.add_argument('--input_image', required=True, help='Path to the input image.')
    parser.add_argument('--weights_file', required=True, help='Path to the YOLO weights file.')
    parser.add_argument('--output_dir', default='./output', help='Output directory for YOLO results.')
    parser.add_argument('--model_to_use', choices=['llama', 'blip'], default='llama',
                        help='Model for captioning.')
    parser.add_argument('--save_images', action='store_true',
                        help='Flag to save intermediate images.')
    parser.add_argument('--icon_detection_path', help='Path to the icon detection model.')
    parser.add_argument('--cache_directory', default='./models_cache',
                        help='Cache directory for models.')
    parser.add_argument('--huggingface_token', default='your_token',
                        help='Hugging Face token for model downloads.')
    parser.add_argument('--no-captioning', action='store_true',
                        help='Disable any image captioning')
    parser.add_argument('--json', dest='output_json', action='store_true',
                        help='Output the image data in JSON format')
    parser.add_argument('--json-mini', action='store_true',
                        help='JSON output in a more condensed format')
    parser.add_argument('--skip-ocr', action='store_true',
                        help='Disable OCR & spell-checking (faster).')
    parser.add_argument('--skip-spell', action='store_true', help='Run OCR but skip spell-check')
    
    args = parser.parse_args()
    
    try:
        print("Running YOLO detection...")
        yolo_output_dir = args.output_dir
        os.makedirs(yolo_output_dir, exist_ok=True)
        process_yolo(args.input_image, args.weights_file, yolo_output_dir)

        base_name = splitext(basename(args.input_image))[0]
        labels_dir = os.path.join(yolo_output_dir, 'labels')
        label_file = os.path.join(labels_dir, base_name + '.txt')
        if not os.path.isfile(label_file):
            raise FileNotFoundError(f"Labels file not found: {label_file}")

        print("Running image description...")
        process_image(
            input_image_path=args.input_image,
            yolo_output_path=label_file,
            model_to_use=args.model_to_use,
            save_images=args.save_images,
            icon_model_path=args.icon_detection_path,
            cache_directory=args.cache_directory,
            huggingface_token=args.huggingface_token,
            no_captioning=args.no_captioning,
            output_json=args.output_json,
            json_mini=args.json_mini,
            skip_ocr=args.skip_ocr,
            skip_spell=args.skip_spell
        )
    except Exception as e:
        print(e)
        sys.exit(1)

