import os
import base64
import time
import easyocr
from spellchecker import SpellChecker
import cv2
import cv2.dnn_superres as dnn_superres
import json
from fastapi.responses import JSONResponse

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai

from ultralytics import YOLO

from wrapper import process_image_description
from utils.pills import preprocess_image

app = FastAPI(title="deki-automata API")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Something like a small password that you set and
# only clients that know this token can use functionalities of the server
API_TOKEN = os.environ.get("API_TOKEN")
if not OPENAI_API_KEY or not API_TOKEN:
    raise RuntimeError("OPENAI_API_KEY and API_TOKEN must be set in environment variables.")

openai.api_key = OPENAI_API_KEY

YOLO_MODEL = None
GLOBAL_SR = None
GLOBAL_READER = None
GLOBAL_SPELL = None

@app.on_event("startup")
def load_models():
    """
    Called once when FastAPI starts.
    """
    global YOLO_MODEL, GLOBAL_SR, GLOBAL_READER, GLOBAL_SPELL
    print("Loading YOLO model ...")
    YOLO_MODEL = YOLO("best.pt")  # or /code/best.pt in Docker
    print("YOLO model loaded successfully.")

    # Super-resolution
    print("Loading super-resolution model ...")
    start_time = time.perf_counter()
    sr = None
    model_path = "EDSR_x4.pb"
    if hasattr(cv2, 'dnn_superres'):
        print("dnn_superres module is available.")
        try:
            sr = dnn_superres.DnnSuperResImpl_create()
            print("Using DnnSuperResImpl_create()")
        except AttributeError:
            sr = dnn_superres.DnnSuperResImpl()
            print("Using DnnSuperResImpl()")
        sr.readModel(model_path)
        sr.setModel('edsr', 4)
        GLOBAL_SR = sr
    else:
        print("dnn_superres module is NOT available; skipping super-resolution.")
        GLOBAL_SR = None
    print(f"Super-resolution initialization took {time.perf_counter()-start_time:.3f}s.")

    # EasyOCR + SpellChecker
    print("Loading OCR + SpellChecker ...")
    start_time = time.perf_counter()
    GLOBAL_READER = easyocr.Reader(['en'], gpu=True)
    GLOBAL_SPELL = SpellChecker()
    print(f"OCR + SpellChecker init took {time.perf_counter()-start_time:.3f}s.")

class ActionRequest(BaseModel):
    image: str  # Base64-encoded image
    prompt: str  # User prompt (like "Open whatsapp and write my friend user_name that I will be late for 15 minutes")

class AnalyzeRequest(BaseModel):
    image: str  # Base64-encoded image

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return credentials.credentials

def run_wrapper(image_path: str) -> str:
    """
    Calls `process_image_description()` to perform YOLO detection and image description,
    then loads the resulting JSON or text file from ./result.
    """
    start_time = time.perf_counter()
    print("run_wrapper start")

    weights_file = "/code/best.pt"
    no_captioning = True
    output_json = True

    process_image_description(
        input_image=image_path,
        weights_file=weights_file,
        no_captioning=no_captioning,
        output_json=output_json,
        model_obj=YOLO_MODEL,
        sr=GLOBAL_SR,
        reader=GLOBAL_READER,
        spell=GLOBAL_SPELL
    )
    elapsed = time.perf_counter() - start_time
    print(f"process_image_description (run_wrapper) took {elapsed:.3f} seconds.")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = os.path.join(os.getcwd(), "result")
    json_file = os.path.join(result_dir, f"{base_name}.json")
    txt_file = os.path.join(result_dir, f"{base_name}.txt")

    description = None
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                description = f.read()
        except Exception as e:
            raise Exception(f"Failed to read JSON description file: {e}")
    elif os.path.exists(txt_file):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                description = f.read()
        except Exception as e:
            raise Exception(f"Failed to read TXT description file: {e}")
    else:
        raise FileNotFoundError("No image description file was generated.")

    return description

@app.post("/action")
async def action(request: ActionRequest, token: str = Depends(verify_token)):
    """
    Processes the input image (in base64 format) and a user prompt:
    1. Decodes and saves the image.
    2. Runs the wrapper.py script (via run_wrapper) to generate an image description.
    3. Constructs a detailed prompt for ChatGPT that includes:
       - Allowed commands.
       - The user prompt.
       - The generated image description.
       - The original image in a base64 format. (can be resized by 0.5 if image is big)
    4. Sends the prompt to ChatGPT (via OpenAI API) and returns the command response.
    """
    start_time = time.perf_counter()
    print("action endpoint start")

    # Remove data URI header if present
    image_data = request.image
    if image_data.startswith("data:image"):
        _, image_data = image_data.split(",", 1)
    
    start_time_decode = time.perf_counter()
    print("Decode image start (action endpoint)")

    try:
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    elapsed_decode = time.perf_counter() - start_time_decode
    print(f"decode image (action endpoint) took {elapsed_decode:.3f} seconds.")
    
    # Save the image to a temporary file
    image_path = "./res/uploaded_image_action.png"
    try:
        with open(image_path, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
    

    start_time_wrapper = time.perf_counter()
    print("run_wrapper start (action endpoint)")
    # Generate image description using run_wrapper (which reads the output file from ./result)
    try:
        image_description = run_wrapper(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

    elapsed_wrapper = time.perf_counter() - start_time_wrapper
    print(f"run wrapper (action endpoint) took {elapsed_wrapper:.3f} seconds.")
    

    start_time_gpt = time.perf_counter()
    print("Send request to gpt (action endpoint)")
    try:
        new_bytes, new_b64 = preprocess_image(img_bytes, threshold=500, scale=0.5, fmt="png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {e}")
    # Prepare the image as a Data URL (with the "data:image/png;base64," prefix)
    base64_image_url = f"data:image/png;base64,{new_b64}"
    
    # Construct the messages for ChatGPT using a multi-part structure
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": '''You are controlling a mobile device. You can only respond with one of these commands (in quotes):
1. "Swipe left"
2. "Swipe right"
3. "Swipe top"
4. "Swipe bottom"
5. "Go home"
6. "Open com.whatsapp" (or other app)
7. "Tap coordinates 300,400" (or other coordinates)
8. "Insert text 300,400:Hello world" (or other coordinates and text)
9. "Finished" (task is finished)
10. "Can't proceed" (Can't understand what to do or image has problem etc.. Better to not continue)

The user said: "{0}"

Please respond with exactly one valid command from the list (formatted precisely), without extra words or explanation.

I will share the screenshot of the current state of the phone and the description (sizes and coordinates) of UI elements.
Description:
"{1}"'''.format(request.prompt, image_description)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image_url,
                        "detail": "low"
                    }
                }
            ]
        }
    ]
    
    # Call the OpenAI API with the prepared messages.
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or use "gpt-4o"
            messages=messages,
            max_tokens=4000,
            temperature=0.2
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    
    elapsed_gpt = time.perf_counter() - start_time_gpt
    print(f"gpt call (action endpoint) took {elapsed_gpt:.3f} seconds.")
    command_response = response.choices[0].message.content.strip()
    elapsed = time.perf_counter() - start_time
    print(f"process_image_description (action endpoint) took {elapsed:.3f} seconds.")
    return {"response": command_response}

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, token: str = Depends(verify_token)):
    """
    Processes the input image (in base64 format) to return the image description as a JSON object.
    """
    # Remove data URI header if present
    image_data = request.image
    if image_data.startswith("data:image"):
        _, image_data = image_data.split(",", 1)
    
    try:
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    image_path = "./res/uploaded_image_analyze.png"
    try:
        with open(image_path, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")

    try:
        image_description = run_wrapper(image_path)
        analyzed_description = json.loads(image_description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

    return JSONResponse(content={"description": analyzed_description})
