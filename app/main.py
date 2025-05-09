import os
import base64
import time
import easyocr
from spellchecker import SpellChecker
import cv2
import cv2.dnn_superres as dnn_superres
import json
import asyncio
import functools
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
from openai import OpenAI

from ultralytics import YOLO

from wrapper import process_image_description
from utils.pills import preprocess_image

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

app = FastAPI(title="deki-automata API")

# Global semaphore limit is set to 1 because the ML tasks take almost all RAM for the current server.
# Can be increased in the future
CONCURRENT_LIMIT = 1
concurrency_semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

def with_semaphore(timeout: float = 20):
    """
    Decorator to limit concurrent access by acquiring the semaphore
    before the function runs, and releasing it afterward.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                await asyncio.wait_for(concurrency_semaphore.acquire(), timeout=timeout)
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="Service busy, please try again later.")
            try:
                return await func(*args, **kwargs)
            finally:
                concurrency_semaphore.release()
        return wrapper
    return decorator

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
API_TOKEN = os.environ.get("API_TOKEN")

if not OPENAI_API_KEY or not API_TOKEN:
    logging.error("OPENAI_API_KEY and API_TOKEN must be set in environment variables.")
    raise RuntimeError("OPENAI_API_KEY and API_TOKEN must be set in environment variables.")

openai.api_key = OPENAI_API_KEY

YOLO_MODEL = None
GLOBAL_SR = None
GLOBAL_READER = None
GLOBAL_SPELL = None
LLM_CLIENT = None

os.makedirs("./res", exist_ok=True)
os.makedirs("./result", exist_ok=True)
os.makedirs("./output", exist_ok=True)


# for action step tracking
ACTION_STEPS_LIMIT = 10 # can be updated
action_step_history = []
action_step_count = 0

@app.on_event("startup")
def load_models():
    """
    Called once when FastAPI starts.
    """
    global YOLO_MODEL, GLOBAL_SR, GLOBAL_READER, GLOBAL_SPELL, LLM_CLIENT
    logging.info("Loading YOLO model ...")
    YOLO_MODEL = YOLO("best.pt")  # or /code/best.pt in Docker
    logging.info("YOLO model loaded successfully.")

    # Super-resolution
    logging.info("Loading super-resolution model ...")
    start_time = time.perf_counter()
    sr = None
    model_path = "EDSR_x4.pb"
    if hasattr(cv2, 'dnn_superres'):
        logging.info("dnn_superres module is available.")
        try:
            sr = dnn_superres.DnnSuperResImpl_create()
            logging.info("Using DnnSuperResImpl_create()")
        except AttributeError:
            sr = dnn_superres.DnnSuperResImpl()
            logging.info("Using DnnSuperResImpl()")

        if os.path.exists(model_path):
            sr.readModel(model_path)
            sr.setModel('edsr', 4)
            GLOBAL_SR = sr
            logging.info("Super-resolution model loaded.")
        else:
            logging.warning(f"Super-resolution model file not found: {model_path}. Skipping SR.")
            GLOBAL_SR = None
    else:
        logging.info("dnn_superres module is NOT available; skipping super-resolution.")
        GLOBAL_SR = None
    logging.info(f"Super-resolution initialization took {time.perf_counter()-start_time:.3f}s.")

    # EasyOCR + SpellChecker
    logging.info("Loading OCR + SpellChecker ...")
    start_time = time.perf_counter()
    GLOBAL_READER = easyocr.Reader(['en'], gpu=True)
    GLOBAL_SPELL = SpellChecker()
    logging.info(f"OCR + SpellChecker init took {time.perf_counter()-start_time:.3f}s.")
    LLM_CLIENT = OpenAI()

class ActionRequest(BaseModel):
    image: str  # Base64-encoded image
    prompt: str  # User prompt (like "Open whatsapp and write my friend user_name that I will be late for 15 minutes")

class AnalyzeRequest(BaseModel):
    image: str  # Base64-encoded image

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        logging.warning("Invalid API token attempt.")
        raise HTTPException(status_code=401, detail="Invalid API token")
    return credentials.credentials

def save_base64_image(image_data: str, file_path: str) -> bytes:
    """
    Decode base64 image data (removing any data URI header) and save to the specified file.
    Returns the raw image bytes.
    """
    if image_data.startswith("data:image"):
        _, image_data = image_data.split(",", 1)
    try:
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        logging.exception("Error decoding base64 image data.")
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
    
    try:
        with open(file_path, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        logging.exception("Error saving image file.")
        raise HTTPException(status_code=500, detail=f"Failed to save image: {e}")
    
    return img_bytes

def log_request_data(request, endpoint: str):
    """
    Log the user prompt (if any) and a preview of the image data.
    """
    logging.info(f"{endpoint} request received:")
    if hasattr(request, 'prompt'):
        logging.info(f"User prompt: {request.prompt}")
    image_preview = request.image[:100] + "..." if len(request.image) > 100 else request.image
    logging.info(f"User image data (base64 preview): {image_preview}")

def run_wrapper(image_path: str, skip_ocr: bool = False, skip_spell: bool = False) -> str:
    """
    Calls process_image_description() to perform YOLO detection and image description,
    then reads the resulting JSON or text file from ./result.
    """
    start_time = time.perf_counter()
    logging.info("run_wrapper start")

    weights_file = "best.pt"
    no_captioning = True
    output_json = True

    process_image_description(
        input_image=image_path,
        weights_file=weights_file,
        no_captioning=no_captioning,
        output_json=output_json,
        model_obj=YOLO_MODEL,
        sr=GLOBAL_SR,
        spell=None if skip_ocr else GLOBAL_SPELL,
        reader=None if skip_ocr else GLOBAL_READER,
        skip_ocr=skip_ocr,
        skip_spell=skip_spell
    )
    elapsed = time.perf_counter() - start_time
    logging.info(f"process_image_description (run_wrapper) took {elapsed:.3f} seconds.")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = os.path.join(os.getcwd(), "result")
    json_file = os.path.join(result_dir, f"{base_name}.json")
    txt_file = os.path.join(result_dir, f"{base_name}.txt")

    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.exception("Failed to read JSON description file.")
            raise Exception(f"Failed to read JSON description file: {e}")
    elif os.path.exists(txt_file):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logging.exception("Failed to read TXT description file.")
            raise Exception(f"Failed to read TXT description file: {e}")
    else:
        logging.error("No image description file was generated.")
        raise FileNotFoundError("No image description file was generated.")

@app.get("/")
async def root():
    return {"message": "deki"}

@app.post("/action")
@with_semaphore(timeout=60)
async def action(request: ActionRequest, token: str = Depends(verify_token)):
    """
    Processes the input image (in base64 format) and a user prompt:
    1. Decodes and saves the original image.
    2. Runs the wrapper to generate an image description and the YOLO-updated image file.
    3. Reads the YOLO-updated image file.
    4. Preprocesses the YOLO-updated image.
    5. Constructs a prompt for ChatGPT (using description + preprocessed YOLO image) and sends it.
    6. Returns the command response.

    This endpoint tracks previous steps and current step count.
    If the number of calls reaches a limit (ACTION_STEPS_LIMIT), it returns
    that step limits are reached and resets the history.
    """
    global action_step_history, action_step_count

    # Check if the step limit is reached.
    if action_step_count >= ACTION_STEPS_LIMIT:
        action_step_history = []
        action_step_count = 0
        return {"response": "Step limit is reached"}

    start_time = time.perf_counter()
    logging.info("action endpoint start")
    log_request_data(request, "/action")

    original_image_path = "./res/uploaded_image_action.png"
    base_name = "uploaded_image_action"
    yolo_updated_image_path = f"./output/{base_name}_yolo_updated.png"

    start_time_decode = time.perf_counter()
    logging.info(f"Decoding and saving original image to: {original_image_path}")
    _ = save_base64_image(request.image, original_image_path)
    logging.info(f"Original image decode and save took {time.perf_counter()-start_time_decode:.3f} seconds.")

    start_time_wrapper = time.perf_counter()
    logging.info("Running wrapper for image description (action endpoint)")
    try:
        image_description = run_wrapper(original_image_path, skip_ocr=False, skip_spell=True)
    except Exception as e:
        logging.exception("Image processing failed in action endpoint.")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")
    logging.info(f"run_wrapper took {time.perf_counter()-start_time_wrapper:.3f} seconds.")

    start_time_yolo_read_prep = time.perf_counter()
    try:
        logging.info(f"Reading YOLO updated image from: {yolo_updated_image_path}")
        if not os.path.exists(yolo_updated_image_path):
            logging.error(f"YOLO updated image not found at {yolo_updated_image_path}")
            raise HTTPException(status_code=500, detail="YOLO updated image generation failed or not found.")
        with open(yolo_updated_image_path, "rb") as f:
            yolo_updated_img_bytes = f.read()
    except Exception as e:
        logging.exception(f"Error reading YOLO updated image from {yolo_updated_image_path}")
        raise HTTPException(status_code=500, detail=f"Failed to read YOLO updated image: {e}")

    logging.info("Preprocessing YOLO updated image for GPT (action endpoint)")
    try:
        new_bytes, new_b64 = preprocess_image(yolo_updated_img_bytes, threshold=2000, scale=0.5, fmt="png")
    except Exception as e:
        logging.exception("YOLO updated image preprocessing failed.")
        raise HTTPException(status_code=500, detail=f"YOLO updated image preprocessing failed: {e}")
    logging.info(f"YOLO image read & preprocess took {time.perf_counter()-start_time_yolo_read_prep:.3f} seconds.")

    base64_image_url = f"data:image/png;base64,{new_b64}"

    # Construct previous steps text and include current step number
    current_step = action_step_count + 1
    if action_step_history:
        previous_steps_text = "\nPrevious steps:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(action_step_history))
    else:
        previous_steps_text = ""

    prompt_text = f"""You are an AI agent that controls a mobile device and sees the content of screen.
User can ask you about some information or to do some task and you need to do these tasks.
You can only respond with one of these commands (in quotes) but some variables are dynamic
and can be changed based on the context:
1. "Swipe left. From start coordinates 300, 400" (or other coordinates) (Goes right)
2. "Swipe right. From start coordinates 500, 650" (or other coordinates) (Goes left)
3. "Swipe top. From start coordinates 600, 510" (or other coordinates) (Goes bottom)
4. "Swipe bottom. From start coordinates 640, 500" (or other coordinates) (Goes top)
5. "Go home"
6. "Go back"
8. "Open com.whatsapp" (or other app)
9. "Tap coordinates 160, 820" (or other coordinates)
10. "Insert text 210, 820:Hello world" (or other coordinates and text)
11. "Screen is in a loading state. Try again" (send image again)
12. "Answer: There are no new important mails today" (or other answer)
13. "Finished" (task is finished)
14. "Can't proceed" (can't understand what to do or image has problem etc.)


The user said: "{request.prompt}"

Current step: {current_step}
{previous_steps_text}

I will share the screenshot of the current state of the phone (with UI elements highlighted and the corresponding 
index of these UI elements) and the description (sizes, coordinates and indexes) of UI elements.
Description:
"{image_description}" """
    # Log the prompt for debugging (remove later)
    logging.info("Constructed prompt for LLM:")
    logging.info(prompt_text)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image_url,
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    start_time_gpt = time.perf_counter()
    logging.info("Sending request to GPT (action endpoint)")
    try:
        response = LLM_CLIENT.chat.completions.create(
            model="gpt-4.1",  # or mini
            messages=messages,
            temperature=0.2,
        )
    except Exception as e:
        logging.exception("OpenAI API error.")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    logging.info(f"GPT call took {time.perf_counter()-start_time_gpt:.3f} seconds.")
    command_response = response.choices[0].message.content.strip()

    action_step_history.append(command_response)
    action_step_count += 1

    command_lower = command_response.strip().strip('\'"').lower()

    if command_lower.startswith("answer:") or command_lower.startswith("finished") or command_lower.startswith("can't proceed"):
        logging.info(f"Terminal command received ('{command_response}'). Resetting history and count.")
        action_step_history = []
        action_step_count = 0

    logging.info(f"action endpoint total processing time: {time.perf_counter()-start_time:.3f} seconds.")
    logging.info(f"response: {command_response}")
    return {"response": command_response}


@app.post("/generate")
@with_semaphore(timeout=60)
async def generate(request: ActionRequest, token: str = Depends(verify_token)):
    """
    Processes the input image (in base64 format) and a user prompt:
    1. Decodes and saves the image.
    2. Runs the wrapper to generate an image description.
    3. Constructs a prompt for ChatGPT and sends it.
    4. Returns the command response.
    """
    start_time = time.perf_counter()
    logging.info("generate endpoint start")
    log_request_data(request, "/generate")

    image_path = "./res/uploaded_image_generate.png"
    start_time_decode = time.perf_counter()
    logging.info("Decoding and saving image (generate endpoint)")
    img_bytes = save_base64_image(request.image, image_path)
    logging.info(f"Image decode and save took {time.perf_counter()-start_time_decode:.3f} seconds.")

    start_time_wrapper = time.perf_counter()
    logging.info("Running wrapper for image description (generate endpoint)")
    try:
        image_description = run_wrapper(image_path)
    except Exception as e:
        logging.exception("Image processing failed in generate endpoint.")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")
    logging.info(f"run_wrapper took {time.perf_counter()-start_time_wrapper:.3f} seconds.")

    start_time_gpt = time.perf_counter()
    logging.info("Preprocessing image for GPT (generate endpoint)")
    try:
        new_bytes, new_b64 = preprocess_image(img_bytes, threshold=1500, scale=0.5, fmt="png")
    except Exception as e:
        logging.exception("Image preprocessing failed.")
        raise HTTPException(status_code=500, detail=f"Image preprocessing failed: {e}")

    base64_image_url = f"data:image/png;base64,{new_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": '''"Prompt: {0}" 
                    Image description:
                    "{1}"'''.format(request.prompt,
                    image_description)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image_url,
                        "detail": "high" # or low
                    }
                }
            ]
        }
    ]

    logging.info("Sending request to GPT (generate endpoint)")
    try:
        response = LLM_CLIENT.chat.completions.create(
            model="gpt-4o",  # or "gpt-4o-mini"
            messages=messages
        )
    except Exception as e:
        logging.exception("OpenAI API error.")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

    logging.info(f"GPT call took {time.perf_counter()-start_time_gpt:.3f} seconds.")
    command_response = response.choices[0].message.content.strip()
    logging.info(f"generate endpoint total processing time: {time.perf_counter()-start_time:.3f} seconds.")
    return {"response": command_response}

@app.post("/analyze")
@with_semaphore(timeout=60)
async def analyze(request: AnalyzeRequest, token: str = Depends(verify_token)):
    """
    Processes the input image (in base64 format) to return the image description as a JSON object.
    """
    logging.info("analyze endpoint start")
    log_request_data(request, "/analyze")
    image_path = "./res/uploaded_image_analyze.png"
    img_bytes = save_base64_image(request.image, image_path)

    try:
        image_description = run_wrapper(image_path)
        analyzed_description = json.loads(image_description)
    except Exception as e:
        logging.exception("Image processing failed in analyze endpoint.")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

    return JSONResponse(content={"description": analyzed_description})


@app.post("/analyze_and_get_yolo")
@with_semaphore(timeout=60)
async def analyze_and_get_yolo(request: AnalyzeRequest, token: str = Depends(verify_token)):
    """
    Processes the input image (in base64 format) to:
      1. Return the image description as a JSON object.
      2. Return the YOLO-updated image (base64 encoded) generated at ./output/uploaded_image_analyze_and_get_yolo_yolo_updated.
    """
    logging.info("analyze_and_get_yolo endpoint start")
    log_request_data(request, "/analyze_and_get_yolo")
    
    # Save the input image to the res folder.
    image_path = "./res/uploaded_image_analyze_and_get_yolo.png"
    img_bytes = save_base64_image(request.image, image_path)
    
    # Run the wrapper to generate the image description.
    try:
        image_description = run_wrapper(image_path)
        analyzed_description = json.loads(image_description)
    except Exception as e:
        logging.exception("Image processing failed in analyze_and_get_yolo endpoint.")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")
    
    # Load the YOLO-updated image from the output folder.
    print(os.getcwd())
    yolo_image_path = "./output/uploaded_image_analyze_and_get_yolo_yolo_updated.png"
    if not os.path.exists(yolo_image_path):
        logging.error("YOLO updated image not found.")
        raise HTTPException(status_code=500, detail="YOLO updated image not generated.")
    
    try:
        with open(yolo_image_path, "rb") as f:
            yolo_img_bytes = f.read()
        yolo_b64 = base64.b64encode(yolo_img_bytes).decode("utf-8")
        yolo_image_encoded = f"data:image/png;base64,{yolo_b64}"
    except Exception as e:
        logging.exception("Error reading YOLO updated image.")
        raise HTTPException(status_code=500, detail=f"Error reading YOLO updated image: {e}")
    
    # Return both the description and the YOLO image.
    return JSONResponse(content={
        "description": analyzed_description,
        "yolo_image": yolo_image_encoded
    })
