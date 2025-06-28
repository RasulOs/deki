# Code modifications to run the Android Control benchmark

To make it possible to test deki by using screensuite project 
deki and screensuite was temporarily modified only for running this 
benchmark.

## Updates

/action endpoint in app/main.py:

```python
@app.post("/action", response_model=ActionResponse)
@with_semaphore(timeout=60)
async def action(request: ActionRequest, token: str = Depends(verify_token)):
    start_time = time.perf_counter()
    logging.info("action endpoint start")
    log_request_data(request, "/action")

    with tempfile.TemporaryDirectory() as temp_dir:
        request_id = str(uuid.uuid4())
        original_image_path = os.path.join(temp_dir, f"{request_id}.png")
        save_base64_image(request.image, original_image_path)
        yolo_updated_image_path = os.path.join(temp_dir, f"{request_id}_yolo_updated.png")
        try:
            loop = asyncio.get_running_loop()
            image_description = await loop.run_in_executor(
                None, run_wrapper, original_image_path, temp_dir, False, True, True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

        if not os.path.exists(yolo_updated_image_path):
            raise HTTPException(status_code=500, detail="YOLO updated image generation failed.")
        with open(yolo_updated_image_path, "rb") as f:
            yolo_updated_img_bytes = f.read()
        _, new_b64 = preprocess_image(yolo_updated_img_bytes, threshold=2000, scale=0.5, fmt="png")

    base64_image_url = f"data:image/png;base64,{new_b64}"

    prompt_text = f"""You are a precise AI agent for a benchmark. Your only goal is to output a single, exact command based on an instruction and screen analysis.

**CRITICAL RULES:**
1.  **NEVER refuse a task.** You must always choose an action from the Approved List below.
2.  **Scrolling vs. Swiping:**
    *   If the user says "scroll up" or "swipe down", the content on screen moves UP. Use `scroll(direction='up')`.
    *   If the user says "scroll down" or "swipe up", the content on screen moves DOWN. Use `scroll(direction='down')`.
3.  **Typing:** If the instruction is to "type" or "enter" text, your action MUST be `type(...)`. Do not `click` a text field that is already focused.
4.  **State Awareness:** If an action was just performed and the screen seems to be loading or has not changed in a meaningful way, the correct action is likely `wait()`.
5.  **Final Output:** Your response MUST BE ONLY the command. Do not add any other words, explanations, or markdown formatting.

**Approved Action Formats:**
- `click(x, y)`
- `long_press(x, y)`
- `type("text to type here")`
- `scroll(direction)` where direction is one of 'up', 'down', 'left', or 'right'
- `open_app("AppName")`
- `navigate_home()`
- `navigate_back()`
- `wait()`

---
**Current Task**

**User Instruction:** "{request.prompt}"

**Screen Analysis (JSON description of UI elements):**
{image_description}

Based on all available information, what is the single, precise action to perform?
Action:"""

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": base64_image_url, "detail": "high"}}]}
    ]

    try:
        response = LLM_CLIENT.chat.completions.create(model="gpt-4.1", messages=messages, temperature=0.0)
        command_response = response.choices[0].message.content.strip()
        return ActionResponse(response=command_response, history=[])
    except Exception as e:
        logging.exception("OpenAI API error.")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
```

---

## deki adapter to run screensuite

```python

import requests
import base64
from PIL import Image
import io

from smolagents import Model, ChatMessage

class DekiAgent(Model):
    """
    An adapter to make the deki agent compatible with the 
    ScreenSuite evaluation framework.
    """
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url
        self.api_token = api_token
        self.model_id = "deki_agent_v1"

    def generate(self, messages: list[dict], **kwargs) -> ChatMessage:
        if not messages or not isinstance(messages, list):
            raise ValueError("Invalid 'messages' input: must be a non-empty list.")

        message_dict = messages[0]
        content_list = message_dict.get('content', [])
        
        if not isinstance(content_list, list):
            raise ValueError("Invalid message format: 'content' is not a list.")

        task_instruction = None
        latest_image = None

        for part in content_list:
            if part.get('type') == 'text' and 'Task:' in part.get('text', ''):
                for line in part['text'].split('\n'):
                    if line.startswith('Task:'):
                        task_instruction = line.replace('Task:', '').strip()
            
            if part.get('type') == 'image' and 'image' in part and isinstance(part['image'], Image.Image):
                latest_image = part['image']

        if not task_instruction or not latest_image:
            raise ValueError("Could not extract task instruction and latest image from the input.")

        buffered = io.BytesIO()
        latest_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"
        
        payload = {"image": image_b64, "prompt": task_instruction, "history": []}
        headers = {"Authorization": f"Bearer {self.api_token}"}

        try:
            print(f"\n[INFO] Sending task '{task_instruction}' to deki API...")
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            deki_response = response.json()["response"].strip().strip('\'"`')
            
            print(f"[INFO] Deki responded: '{deki_response}'")
            
            return ChatMessage(role="assistant", content=deki_response)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error calling deki API: {e}")
            return ChatMessage(role="assistant", content="wait()")
```

---

## Script to evaluate deki results

```python
import json
import os
import traceback
from datetime import datetime

import datasets
from dotenv import load_dotenv

from deki_adapter import DekiAgent
from screensuite import EvaluationConfig, get_registry

load_dotenv()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "deki_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_deki_evaluation():
    registry = get_registry()
    benchmarks_list = registry.get(["android_control"])
    
    if not benchmarks_list:
        print("Error: 'android_control' benchmark not found in registry.")
        return
        
    benchmark = benchmarks_list[0]
    print(f"Found benchmark: {benchmark.name}")

    def custom_load(*args, **kwargs):
        print("Loading a small slice of the dataset (25 examples) via custom_load...")
        
        small_dataset_split = datasets.load_dataset(
            "smolagents/android-control",
            split='test',
            streaming=True
        ).take(25)

        benchmark.dataset = list(small_dataset_split)
        
        print("Dataset slice loaded successfully.")

    benchmark.load = custom_load

    deki_api_url = "http://localhost:8000/action"
    deki_api_token = os.getenv("API_TOKEN") 
    
    if not deki_api_token:
        raise ValueError("API_TOKEN not found. Please set it in your environment or a .env file.")

    my_agent = DekiAgent(api_url=deki_api_url, api_token=deki_api_token)

    run_name = f"deki_eval_on_{benchmark.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    
    config = EvaluationConfig(
        parallel_workers=1,
        run_name=run_name,
        max_samples_to_test=25,
    )

    print(f"Starting evaluation on {config.max_samples_to_test} samples...")
    try:
        results = benchmark.evaluate(my_agent, evaluation_config=config)
        
        print("\n--- Evaluation Complete ---")
        print(f"Results: {results._metrics}")

        output_file = f"{RESULTS_DIR}/results_{run_name}.json"
        with open(output_file, "w") as f:
            json.dump({"benchmark_name": benchmark.name, "metrics": results._metrics}, f, indent=4)
        print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_deki_evaluation()
```

---
