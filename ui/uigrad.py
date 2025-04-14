import gradio as gr
import requests
import base64
import io
from PIL import Image

DEFAULT_API_TOKEN = ""

# Load default images from the relative paths
default_image_1 = Image.open("./res/bb_1.jpeg")
default_image_2 = Image.open("./res/mfa_1.jpeg")

def load_example_action_1():
    return default_image_1, "Open and read Umico partner"

def load_example_action_2():
    return default_image_2, "Sign up in the application"

def load_example_analyze_1():
    return default_image_1

def load_example_analyze_2():
    return default_image_2

def load_example_yolo_1():
    return default_image_1

def load_example_yolo_2():
    return default_image_2

def load_example_generate_1():
    # Both examples use the same prompt.
    return default_image_1, "Generate the code for this screen for Android XML. Try to use constraint layout"

def load_example_generate_2():
    return default_image_2, "Generate the code for this screen for Android XML. Try to use constraint layout"

# API calls
def call_action_endpoint(token: str, image: Image.Image, prompt: str) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    data = {
        "image": f"data:image/png;base64,{img_str}",
        "prompt": prompt
    }
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post("http://localhost:8000/action", json=data, headers=headers)
        response.raise_for_status()
        result = response.json()["response"]
    except Exception as e:
        result = f"Error: {e}"
    return result

def call_analyze_endpoint(token: str, image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    data = {"image": f"data:image/png;base64,{img_str}"}
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post("http://localhost:8000/analyze", json=data, headers=headers)
        response.raise_for_status()
        result = response.json()["description"]
    except Exception as e:
        result = f"Error: {e}"
    return str(result)

def call_analyze_and_get_yolo_endpoint(token: str, image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    data = {"image": f"data:image/png;base64,{img_str}"}
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post("http://localhost:8000/analyze_and_get_yolo", json=data, headers=headers)
        response.raise_for_status()
        resp_json = response.json()
        description = resp_json.get("description", "")
        yolo_image_data = resp_json.get("yolo_image", "")
        
        # Remove data URI header if present.
        if yolo_image_data.startswith("data:image"):
            _, encoded = yolo_image_data.split(",", 1)
        else:
            encoded = yolo_image_data
        
        yolo_image_bytes = base64.b64decode(encoded)
        yolo_image = Image.open(io.BytesIO(yolo_image_bytes))
    except Exception as e:
        description = f"Error: {e}"
        yolo_image = None
    return description, yolo_image

def call_generate_endpoint(token: str, image: Image.Image, prompt: str) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    data = {
        "image": f"data:image/png;base64,{img_str}",
        "prompt": prompt
    }
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post("http://localhost:8000/generate", json=data, headers=headers)
        response.raise_for_status()
        result = response.json()["response"]
    except Exception as e:
        result = f"Error: {e}"
    return result

with gr.Blocks() as demo:
    gr.Markdown("# deki automata Gradio UI")
    
    # Global token input visible across all tabs.
    token_input = gr.Textbox(value=DEFAULT_API_TOKEN, label="API Token", placeholder="Enter API Token")
    
    with gr.Tabs():
        with gr.TabItem("Action Endpoint"):
            gr.Markdown("### Call the /action endpoint")
            with gr.Row():
                image_input = gr.Image(value=None, type="pil", label="Upload Image")
                prompt_input = gr.Textbox(value="", lines=2, placeholder="Enter prompt", label="Prompt")
            action_output = gr.Textbox(label="Response")
            action_button = gr.Button("Call Action Endpoint")
            action_button.click(fn=call_action_endpoint, inputs=[token_input, image_input, prompt_input], outputs=action_output)
            # Load Example buttons below the call button.
            with gr.Row():
                example_action_btn1 = gr.Button("Load Example 1")
                example_action_btn2 = gr.Button("Load Example 2")
            example_action_btn1.click(fn=load_example_action_1, outputs=[image_input, prompt_input])
            example_action_btn2.click(fn=load_example_action_2, outputs=[image_input, prompt_input])
        
        with gr.TabItem("Analyze Endpoint"):
            gr.Markdown("### Call the /analyze endpoint")
            image_input_analyze = gr.Image(value=None, type="pil", label="Upload Image")
            analyze_output = gr.Textbox(label="Response")
            analyze_button = gr.Button("Call Analyze Endpoint")
            analyze_button.click(fn=call_analyze_endpoint, inputs=[token_input, image_input_analyze], outputs=analyze_output)
            with gr.Row():
                example_analyze_btn1 = gr.Button("Load Example 1")
                example_analyze_btn2 = gr.Button("Load Example 2")
            example_analyze_btn1.click(fn=load_example_analyze_1, outputs=image_input_analyze)
            example_analyze_btn2.click(fn=load_example_analyze_2, outputs=image_input_analyze)
        
        with gr.TabItem("Analyze & Get YOLO Endpoint"):
            gr.Markdown("### Call the /analyze_and_get_yolo endpoint")
            image_input_yolo = gr.Image(value=None, type="pil", label="Upload Image")
            description_output = gr.Textbox(label="Description")
            yolo_image_output = gr.Image(label="YOLO Updated Image")
            yolo_button = gr.Button("Call Analyze & Get YOLO Endpoint")
            yolo_button.click(
                fn=call_analyze_and_get_yolo_endpoint, 
                inputs=[token_input, image_input_yolo], 
                outputs=[description_output, yolo_image_output]
            )
            with gr.Row():
                example_yolo_btn1 = gr.Button("Load Example 1")
                example_yolo_btn2 = gr.Button("Load Example 2")
            example_yolo_btn1.click(fn=load_example_yolo_1, outputs=image_input_yolo)
            example_yolo_btn2.click(fn=load_example_yolo_2, outputs=image_input_yolo)
        
        with gr.TabItem("Generate Endpoint"):
            gr.Markdown("### Call the /generate endpoint")
            with gr.Row():
                image_input_generate = gr.Image(value=None, type="pil", label="Upload Image")
                prompt_input_generate = gr.Textbox(value="", lines=2, placeholder="Enter prompt", label="Prompt")
            generate_output = gr.Textbox(label="Response")
            generate_button = gr.Button("Call Generate Endpoint")
            generate_button.click(fn=call_generate_endpoint, inputs=[token_input, image_input_generate, prompt_input_generate], outputs=generate_output)
            with gr.Row():
                example_generate_btn1 = gr.Button("Load Example 1")
                example_generate_btn2 = gr.Button("Load Example 2")
            example_generate_btn1.click(fn=load_example_generate_1, outputs=[image_input_generate, prompt_input_generate])
            example_generate_btn2.click(fn=load_example_generate_2, outputs=[image_input_generate, prompt_input_generate])

demo.launch(share=True)
