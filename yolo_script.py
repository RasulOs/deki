import ultralytics
import cv2
from ultralytics import YOLO
import os
import glob
import argparse
import sys
import numpy as np
import uuid

def iou(box1, box2):
    # Convert normalized coordinates to (x1, y1, x2, y2)
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2

    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def load_image(input_image, base_name: str = None):

    if isinstance(input_image, str):
        img = cv2.imread(input_image)
        if img is None:
            raise ValueError(f"Unable to load image from path: {input_image}")
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(input_image))[0]
        return img, base_name
    else:
        # Assume input_image is raw bytes or a file-like object
        if isinstance(input_image, bytes):
            image_bytes = input_image
        else:
            image_bytes = input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image from input bytes or file-like object.")
        if base_name is None:
            base_name = str(uuid.uuid4())
        return img, base_name

def process_yolo(input_image, weights_file: str, output_dir: str = './yolo_run', model_obj: YOLO = None, base_name: str = None) -> str:

    # Load image (from disk or memory) and determine a base name
    orig_image, inferred_base_name = load_image(input_image, base_name)
    base_name = inferred_base_name  # use provided or inferred base name
    
    os.makedirs(output_dir, exist_ok=True)
    # Determine file extension: if input_image is a file path, use its extension. Otherwise, default to .png
    if isinstance(input_image, str):
        ext = os.path.splitext(input_image)[1]
    else:
        ext = ".png"
    output_image_name = f"{base_name}_yolo{ext}"
    updated_output_image_name = f"{base_name}_yolo_updated{ext}"

    print(ultralytics.checks())

    # If input_image is a file path, call YOLO with the path to preserve filename-based labeling.
    # Otherwise, if processing in-memory, YOLO might default to a generic name.
    if isinstance(input_image, str):
        source_input = input_image
    else:
        source_input = orig_image

    # Use provided model or load from weights_file.
    if model_obj is None:
        model = YOLO(weights_file)
    else:
        model = model_obj

    # Delete existing label file for this image (if it exists) so that detections are rewritten and not appended
    labels_dir = os.path.join(output_dir, "yolo_labels_output", "labels")
    expected_label_file = os.path.join(labels_dir, f"{base_name}.txt")
    if os.path.exists(expected_label_file):
        print(f"Deleting existing label file: {expected_label_file}")
        os.remove(expected_label_file)

    results = model(
        source=source_input, 
        save_txt=True, 
        project=output_dir, 
        name="yolo_labels_output",
        exist_ok=True
    )

    # Save the initial inference image.
    img_with_boxes = results[0].plot(font_size=2, line_width=1)
    output_image_path = os.path.join(output_dir, output_image_name)
    cv2.imwrite(output_image_path, img_with_boxes)
    print(f"Image saved as '{output_image_path}'")

    # Directory containing label files.
    labels_dir = os.path.join(output_dir, 'yolo_labels_output', 'labels')
    # Search for txt files whose filenames contain the original base name.
    label_files = [f for f in glob.glob(os.path.join(labels_dir, '*.txt')) if base_name in os.path.basename(f)]
    if not label_files:
        raise FileNotFoundError(f"No label files found for the image '{base_name}'.")
    label_file = label_files[0]

    with open(label_file, 'r') as f:
        lines = f.readlines()

    boxes = []
    for idx, line in enumerate(lines):
        tokens = line.strip().split()
        class_id = int(tokens[0])
        x_center = float(tokens[1])
        y_center = float(tokens[2])
        width = float(tokens[3])
        height = float(tokens[4])
        boxes.append({
            'class_id': class_id,
            'bbox': [x_center, y_center, width, height],
            'line': line,
            'index': idx
        })

    boxes.sort(key=lambda b: b['bbox'][1] - (b['bbox'][3] / 2))

    # Perform NMS.
    keep_indices = []
    suppressed = [False] * len(boxes)
    num_removed = 0
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep_indices.append(i)
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            if boxes[i]['class_id'] == boxes[j]['class_id']:
                iou_value = iou(boxes[i]['bbox'], boxes[j]['bbox'])
                if iou_value > 0.7:
                    suppressed[j] = True
                    num_removed += 1

    with open(label_file, 'w') as f:
        for idx in keep_indices:
            f.write(boxes[idx]['line'])

    print(f"Number of bounding boxes removed: {num_removed}")

    # Draw updated bounding boxes on the original image (loaded in memory).
    drawn_image = orig_image.copy()
    h_img, w_img, _ = drawn_image.shape

    for i, idx in enumerate(keep_indices):
        box = boxes[idx]
        x_center, y_center, w_norm, h_norm = box['bbox']
        x_center *= w_img
        y_center *= h_img
        w_box = w_norm * w_img
        h_box = h_norm * h_img
        x1 = int(x_center - w_box / 2)
        y1 = int(y_center - h_box / 2)
        x2 = int(x_center + w_box / 2)
        y2 = int(y_center + h_box / 2)
        cv2.rectangle(drawn_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(drawn_image, str(i + 1), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (143, 10, 18), 1)

    updated_output_image_path = os.path.join(output_dir, updated_output_image_name)
    cv2.imwrite(updated_output_image_path, drawn_image)
    print(f"Updated image saved as '{updated_output_image_path}'")
    
    return updated_output_image_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process YOLO inference and NMS on an image.')
    parser.add_argument('input_image', help='Path to the input image or pass raw bytes via a file-like object.')
    parser.add_argument('weights_file', help='Path to the YOLO weights file.')
    parser.add_argument('output_dir', nargs='?', default='./yolo_run', help='Output directory (optional).')
    parser.add_argument('--base_name', help='Optional base name for output files (without extension).')
    args = parser.parse_args()
    
    try:
        process_yolo(args.input_image, args.weights_file, args.output_dir, base_name=args.base_name)
    except Exception as e:
        print(e)
        sys.exit(1)
