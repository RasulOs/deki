import base64
import argparse
import requests
import json

def encode_image(image_path):
    """Reads an image file and returns a base64-encoded Data URL (assuming PNG)."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def main():
    parser = argparse.ArgumentParser(description="Test deki-automata API")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--prompt", default="Open whatsapp and write my friend user_name that I will be late for 15 minutes",
                        help="User prompt for the API (only used with the action endpoint)")
    parser.add_argument("--api_url", required=True, help="API endpoint URL (e.g., http://localhost:8000/action or http://localhost:8000/analyze)")
    parser.add_argument("--token", required=True, help="API token for Authorization header")
    args = parser.parse_args()

    # Convert the image to base64
    image_data = encode_image(args.image_path)

    # Build the payload:
    # If the API URL contains "action", include the prompt; for analyze, only image is needed.
    payload = {"image": image_data}
    if "action" in args.api_url.lower():
        payload["prompt"] = args.prompt

    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json"
    }

    # Send the POST request
    response = requests.post(args.api_url, json=payload, headers=headers)
    
    print("Status Code:", response.status_code)
    try:
        print("Formatted Response JSON:", json.dumps(response.json(), indent=2))
    except Exception:
        print("Response Text:", response.text)

if __name__ == "__main__":
    main()
