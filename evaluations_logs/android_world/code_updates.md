# Code modifications to run the Android World benchmark

In android_world/agents/deki.py

```python
import requests
import base64
from PIL import Image
import io
import json

from android_world.agents import base_agent
from android_world.env import adb_utils
from android_world.env import interface
import android_world.env.json_action as json_action
from android_world.env.representation_utils import BoundingBox

def _find_element_index_at_coords(
    elements: list, x: int, y: int
) -> int | None:
    for i, element in enumerate(elements):
        if hasattr(element, 'bbox_pixels') and element.bbox_pixels:
            bbox: BoundingBox = element.bbox_pixels
            if (bbox.x_min <= x <= bbox.x_max) and (bbox.y_min <= y <= bbox.y_max):
                return i
    return None

class DekiWorldAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(self, env: interface.AsyncEnv):
        super().__init__(env)
        self.deki_api_url = "http://localhost:8000/action"
        self.deki_api_token = "YOUR_TOKEN_FOR_DEKI_SERVER"
        self.action_history = []

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        print(f"DekiWorldAgent received goal: {goal}")

        try:
            state: interface.State = self.get_post_transition_state()
            screenshot = Image.fromarray(state.pixels)
            installed_packages = adb_utils.get_all_package_names(self.env.controller)

        except Exception as e:
            print(f"Error getting state from environment: {e}")
            agent_action = json_action.JSONAction(action_type="wait")
            self.env.execute_action(agent_action)
            return base_agent.AgentInteractionResult(done=False, data={})

        enriched_prompt = (
            f"Given the following installed apps: {', '.join(installed_packages)}. "
            f"Perform the following task: {goal}"
        )
        
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64 = f"data:image/png;base64,{img_str}"

        payload = {"image": image_b64, "prompt": enriched_prompt, "history": self.action_history}
        headers = {"Authorization": f"Bearer {self.deki_api_token}"}

        try:
            print(f"\n[INFO] Sending enriched task to deki API...")
            response = requests.post(self.deki_api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            action_and_explanation_str = response.json()["response"]
            print(f"Deki server proposed: {action_and_explanation_str}")
            
            self.action_history.append(action_and_explanation_str)
            
            parsed_data = json.loads(action_and_explanation_str)
            action_params = parsed_data.get('action', {'action_type': 'wait'})
            
            if action_params.get('action_type') == 'swipe' and action_params.get('x') is not None:
                print("Swipe with coordinates detected. Translating to indexed scroll.")
                target_index = _find_element_index_at_coords(
                    state.ui_elements, action_params['x'], action_params['y']
                )
                
                if target_index is not None:
                    print(f"Found element at index {target_index}. Creating indexed scroll action.")
                    final_action = json_action.JSONAction(
                        action_type='scroll',
                        index=target_index,
                        direction=action_params.get('direction', 'left')
                    )
                else:
                    print("Could not find element at coordinates. Falling back to generic swipe.")
                    final_action = json_action.JSONAction(
                        action_type='swipe',
                        direction=action_params.get('direction', 'left')
                    )
            else:
                final_action = json_action.JSONAction(**action_params)

        except Exception as e:
            print(f"Error processing action from deki server: {e}. Defaulting to wait().")
            final_action = json_action.JSONAction(action_type="wait")

        self.env.execute_action(final_action)

        is_task_finished = (final_action.action_type == 'status' and final_action.goal_status == 'complete')
        
        step_data = {
            'screenshot': state.pixels,
            'goal': goal,
            'action': final_action.json_str()
        }
        
        return base_agent.AgentInteractionResult(
            done=is_task_finished,
            data=step_data
        )

    def reset(self, go_home: bool = False):
        print(f"DekiWorldAgent reset called. Go home: {go_home}")
        self.action_history = []
        super().reset(go_home=go_home)
```

In infer.py:309

```python 
              'url': f'data:image/jpeg;base64,{self.encode_image(image)}',"detail": "high"
```

In app/main.py: /action endpoint

```python

@app.post("/action", response_model=ActionResponse)
@with_semaphore(timeout=60)
async def action(request: ActionRequest, token: str = Depends(verify_token)):
    start_time = time.perf_counter()
    logging.info("action endpoint start")
    log_request_data(request, "/action")

    action_step_history = request.history
    action_step_count = len(action_step_history)

    if action_step_count >= ACTION_STEPS_LIMIT:
        logging.warning(f"Step limit of {ACTION_STEPS_LIMIT} reached. Resetting history.")
        return ActionResponse(response="Step limit is reached", history=[])

    # Use a temporary directory to isolate all files for this request.
    with tempfile.TemporaryDirectory() as temp_dir:
        request_id = str(uuid.uuid4())
        
        original_image_path = os.path.join(temp_dir, f"{request_id}.png")
        yolo_updated_image_path = os.path.join(temp_dir, f"{request_id}_yolo_updated.png")

        save_base64_image(request.image, original_image_path)

        try:
            loop = asyncio.get_running_loop()
            image_processing_task = loop.run_in_executor(
                None,
                run_wrapper,
                original_image_path,
                temp_dir,
                False,
                True,
                True,
            )
            image_description = await asyncio.wait_for(image_processing_task, timeout=60.0)
        except asyncio.TimeoutError:
            logging.error("Image processing in run_wrapper timed out.")
            raise HTTPException(
                status_code=504,
                detail="Image processing task timed out. The device may be unresponsive."
            )
        except Exception as e:
            logging.exception("Image processing failed in action endpoint.")
            raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")

        try:
            if not os.path.exists(yolo_updated_image_path):
                logging.error(f"YOLO updated image not found at {yolo_updated_image_path}")
                raise HTTPException(status_code=500, detail="YOLO updated image generation failed or not found.")
            with open(yolo_updated_image_path, "rb") as f:
                yolo_updated_img_bytes = f.read()
        except Exception as e:
            logging.exception(f"Error reading YOLO updated image from {yolo_updated_image_path}")
            raise HTTPException(status_code=500, detail=f"Failed to read YOLO updated image: {e}")
        
        try:
            _, new_b64 = preprocess_image(yolo_updated_img_bytes, threshold=2000, scale=0.5, fmt="png")
        except Exception as e:
            logging.exception("YOLO updated image preprocessing failed.")
            raise HTTPException(status_code=500, detail=f"YOLO updated image preprocessing failed: {e}")

    base64_image_url = f"data:image/png;base64,{new_b64}"
    history_log = "\n".join(f"- Step {i+1}: {step}" for i, step in enumerate(request.history))
    if not history_log:
        history_log = "No actions have been taken yet."

    print(f"history_log: {history_log}")

      
    prompt_text = f"""You are a methodical AI agent operating an Android phone. Your goal is to reason about the next step to complete a high-level goal, write everything you see on the image, then output your reasoning and a single action.

    **GUIDELINES & BEST PRACTICES:**
    - **Task Completion:** When the task is fully complete, you MUST use `status(goal_status="complete")`.
    - **Click:** When you want to click on something try to click on the interactable elements, they are usually icons, texts or images. Rarely containers.
    - **Text Input:** To enter text, use `input_text`. If a text field already contains text, you must clear it first. **THE WAY TO  DO THIS**: long press on some word, then context menu will be shown, **THEN CLICK ON `SELECT ALL` AND THEN PRESSING BACKSPACE BUTTON**. DON'T FORGET TO LONG PRESS. If you are just adding text, remember what is already there, clear everything, and then type the full combined text. DON'T JUST APPEND THE TEXT. FIRST REMOVE THE CURRENT ONE BY LONG PRESSING AND THE CLEARING ALL.
    - **Update the content of the text file:** If the task to update the content of some text file, then first remove everything that was there (by selecting all) and removing and then add what you need.
    - **Focus before input text:** If you don't see the keyboard then it means the input field is not focuessed. **YOU NEED TO FOCUS ON AN INPUT TEXT FIELD (IF IT NOS ALREADT FOCUSSED) BEFORE DOING THE INPUT TEXT COMMAND.**
    - **If keyboard is open and you need to enter text:** Just use input text command, no need to click on keyboard numbers one by one.
    - **Video Recording:** If the task is to record a video, open the Camera app, **SWIPE RIGHT TO ENTER THE VIDEO MODE**, press the record button to start, and press it again to stop.
    - **Take a picture:** To take a picture, open the camera, click on camera icon and it is done.
    - **Browser Setup:** When opening a browser (like Chrome) for the first time, you will likely see setup screens. **DOUBLE TAP on "Accept & continue", THEN DOUBLE TAP ON CONTINUE WITHOUT ACCOUNT (USE DOUBLE TAPS ON GOOGLE CHROME SETUP BUTTONS (Like, accept and continue or no thanks on turn on sync screen), DON'T SPEND TIME ON SINGLE CLICKS).** If you seem stuck, try the same action again; it might be a one-time setup issue. Don't click on add account etc. If something went wrong, you  need to re-attempt opening it with Chrome because the chrome has some returns you back after making a first time set-up. Don't stop, or go home etc. Continue. It is known problem in chrome that after setup it can return you to the previous screen but you should know that it means that set up was completed and now the default app for openning some files is chrome, just make a double tap on a file. Don't navigate back, navigate home etc. **IF YOU SEE THAT THERE IS URI ADDRESS IN ADDRESS BAR BUT NO CONTENT IT IS HIGHLY LIKELY LOADING, IF IT IS THE CASE - JUST WAIT**
    - **Memory/Scratchpad:** Use your "Reasoning" to keep track of important information. If a task requires you to remember a number, write it down in your reasoning.
    - **Click Accuracy:** If a click doesn't work (or screen did not change after the click), you should click on the center of the target element again or a little bit tweak x and y coordinates (maybe the tapable area is close but the exact). If it still fails, reconsider your plan.

    - ** Swipes**
            * You must provide the x and y coordinates where the swipe should begin.
            * How to get coordinates: Find the element you want to swipe (e.g., the last visible category button) in the JSON screen description. Use its coordinates for your x and y.
            * How to choose the direction: The direction is the direction of your finger movement. 
            * Or just start from the center if you don't care and just want make the swipe movement.
            * IF YOU SEE THAT SWIPES TO SOME DIRECTIONS DON'T WORK YOU SHOULD TRY THE OPPOSITE DIRECTION.
            * IF YOU WANT TO SWIPE AND YOU SEE THAT KEYBOARD IS OPEN THEN FIRST CLOSE THE KEYBOARD BY PRESSING BACK (NAVIGATING BACK) 1 time.

    - **Answering Questions:** If the goal is a question, you MUST use the `answer` action.
    - **Recording Audio or creating a note:** If the goal is to record the audio/creating a note with a specific name, then first record it (by clicking apply or new or +icon) and then it will ask about the name of the file. You can choose the name of the file after the recording.
    - **Focus:** If the focus is not on some element and you want to click it, it can require you to click twice, 1 click to get the focus and the second to make the click.
    - **Open with:** If the task is to open some file in a with a specific app (like chrome browser) then **FIRST LONG PRESS ON THAT FILE INSTEAD OF CLICKING on that file**. **DON'T SPEND TIME ON SINGLE CLICKS FIRST**. Then click on something like 3 dots or similar to open a context menu.
    - **Renaming file:** If the task is to rename some file, **USUALLY YOU NEED TO EXIT FROM THE FILE ITSELF (IF YOU WERE IN THE FILE) TO SEE THE LIST WITH ALL FILE NAMES IN THE FOLDER AND THEN LONG PRESS ON THE FILE NAME AND CLICK ON THE APPROPRIATE ICON**. **IT CAN BE A LETTER ICON WITH A CURSOR (HIGHEST POSSIBILITY), OR PENCIL (HIGHEST POSSIBILITY), OR 3 DOTS. START FROM HIGHEST PROBABILITY TO THE LOWEST. Don't spend time on renaming the filename if you are in the file.**
    - **Open app:** If the task is to open some app then select the most suitable apps from the list agent provided you. Don't navigate the home screen, make swipes/scrolls in the home screen or open the app drawer etc. Just directly open the app from the installed apps.
    - **Browser:** If the task is to open some file in chrome browser, sometimes when you open some file in the browser the loading can take some time, wait 1 time if the content is blank and URI was enetered. Don't navigate home or navigate back. If you see that there is file's content URI address but no content it highly likely loading.
    - **Reading from file or image:** If the task is about to read something and write this into other place then just copy ruqired part. Don't add the characters to the text you see on image or in the file. Like, if there is no dot at the end OF sentence, don't add it.

    - **Search or moving the file:** If the goal to move some specific file from one folder to another then serch for that specific file, because there can be files with similar names.
    - **Remove the element:** **IF THE GOAL TO REMOVE SOME ELEMENT CLICK ON `TRASH` ICON, `X` ICONS OR OR SOMETHING LIKE THAT, AND AFTER REMOVING IT, CHECK IF IT WAS REMOVED BY CHECKING THE SCREENSHOT AGAIN. USUALLY TO DELETE SOMETHING YOU NEED TO CLICK ON TRASH ICON OR X ICON OR DELETE TITLE ETC.**


    **ACTION SPACE (Your action must be a single JSON object from this list):**
    *   `{{"action_type": "click", "x": <int>, "y": <int>}}`
    *   `{{"action_type": "double_tap", "x": <int>, "y": <int>}}`
    *   `{{"action_type": "long_press", "x": <int>, "y": <int>}}`
    *   `{{"action_type": "input_text", "text": "text to type"}}`
    *   `{{"action_type": "swipe", "x": <int>, "y": <int>, "direction": "up|down|left|right"}}`
    *   `{{"action_type": "navigate_home"}}`
    *   `{{"action_type": "navigate_back"}}`
    *   `{{"action_type": "keyboard_enter"}}`
    *   `{{"action_type": "open_app", "app_name": "AppName"}}`
    *   `{{"action_type": "wait"}}`
    *   `{{"action_type": "status", "goal_status": "complete|infeasible"}}`
    *   `{{"action_type": "answer", "text": "your answer"}}`

    ---
    **CURRENT TASK**

    **Goal:** "{request.prompt}"

    **Action History (Your Memory):**
    {history_log}

    **Screen Analysis (Vision Model JSON Output):**
    ```json
    {image_description}

    Your Response:
    Your response MUST be in this exact format, with "Reason:" and "Action:" on separate lines.

    Reason: [Your detailed reasoning and analysis of the screen and history goes here. You should write all the details of the most important parts of the screen as you think. Like, texts, figures etc.]
    Action: [A single JSON object from the ACTION SPACE list above.]
    """

    try:
        gcfg = types.GenerateContentConfig(
            temperature=0.0,
        )
        image_part = types.Part.from_bytes(
            data=yolo_updated_img_bytes,
            mime_type='image/png'
        )
        contents = [image_part, prompt_text]
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=contents,
            config=gcfg,
        )
        if not response:
            logging.error("Gemini API returned a null response object.")
            raise HTTPException(status_code=500, detail="Gemini API returned a null response.")

        if not response.candidates:
            block_reason = "Unknown"
            if response.prompt_feedback:
                 block_reason = response.prompt_feedback.block_reason
            logging.error(
                f"Gemini response was blocked or empty. Reason: {block_reason}. "
                f"Full feedback: {response.prompt_feedback}"
            )
            raise HTTPException(
                status_code=500, 
                detail=f"Gemini API call failed: Response was blocked due to {block_reason}"
            )

        command_response = response.text.strip()
        
        reason_match = re.search(r'Reason:(.*?)Action:', command_response, re.DOTALL)
        action_match = re.search(r'Action:(.*)', command_response, re.DOTALL)

        reason_text = ""
        action_json_str = None

        if reason_match:
            reason_text = reason_match.group(1).strip()
        if action_match:
            action_block = action_match.group(1).strip()
            json_match_in_action = re.search(r'\{.*\}', action_block, re.DOTALL)
            if json_match_in_action:
                action_json_str = json_match_in_action.group(0)
        
        if not action_json_str:
            final_json_payload = {
                "action": {"action_type": "wait"},
                "explanation": f"Could not parse valid action from LLM response: {command_response}"
            }
        else:
            try:
                parsed_action = json.loads(action_json_str)

                # This block translates the LLM's intent ("show me content on the left")
                # into the correct physical finger movement ("swipe right").
                # It is required only for Android World benchmark
                DIRECTION_INVERSION_MAP = {
                    "up": "down",
                    "down": "up",
                    "left": "right",
                    "right": "left",
                }
                action_type = parsed_action.get("action_type")
                if action_type in ["swipe", "scroll"]:
                    original_direction = parsed_action.get("direction")
                    if original_direction in DIRECTION_INVERSION_MAP:
                        inverted_direction = DIRECTION_INVERSION_MAP[original_direction]
                        parsed_action["direction"] = inverted_direction
                        logging.info(
                            f"Inverting direction for '{action_type}': "
                            f"{original_direction} -> {inverted_direction}"
                        )

                final_json_payload = {
                    "action": parsed_action,
                    "explanation": reason_text
                }
            except json.JSONDecodeError:
                final_json_payload = {
                    "action": {"action_type": "wait"},
                    "explanation": f"Could not decode the action JSON from LLM: {action_json_str}"
                }

        final_response_str = json.dumps(final_json_payload)
        new_history = request.history + [final_response_str]
        
        return ActionResponse(response=final_response_str, history=new_history)

    except Exception as e:
        logging.exception("Gemini API error.")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")
```

run: 
```bash
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
```

```bash 
python run.py --agent_name=deki --tasks=AudioRecorderRecordAudio,AudioRecorderRecordAudioWithFileName,BrowserDraw,BrowserMaze
```

