import cv2
import base64
import numpy as np
from typing import Tuple

def preprocess_image(image_bytes: bytes, threshold: int = 1000, scale: float = 0.5, fmt: str = "png") -> Tuple[bytes, str]:
    """
    Preprocesses the image by checking its dimensions and downscaling it if needed.
    
    Parameters:
      - image_bytes: Raw image bytes.
      - threshold: Maximum allowed width or height (in pixels). If either dimension exceeds this,
                   the image will be downscaled.
      - scale: Scale factor to use for resizing if the image is too large.
      - fmt: Format for re-encoding the image (e.g., "png" or "jpg").
    
    Returns:
      - A tuple (new_image_bytes, new_base64_str) where:
          new_image_bytes: The re-encoded image bytes after potential downscaling.
          new_base64_str: The base64 string (without header) of the new image bytes.
    """
    # Convert raw bytes to a NumPy array then decode with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if cv_image is None:
        raise ValueError("Failed to decode image with OpenCV.")
    
    h, w = cv_image.shape[:2]
    
    # If either dimension is greater than threshold, resize the image.
    if h > threshold or w > threshold:
        new_w, new_h = int(w * scale), int(h * scale)
        cv_image = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    ret, buf = cv2.imencode(f'.{fmt}', cv_image)
    
    if not ret:
        raise ValueError("Failed to re-encode image.")
    
    new_image_bytes = buf.tobytes()
    new_base64_str = base64.b64encode(new_image_bytes).decode("utf-8")
    
    return new_image_bytes, new_base64_str
