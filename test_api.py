import base64
import argparse
import requests

def encode_image(image_path):
    """Reads an image file and returns a base64-encoded Data URL (assuming PNG)."""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def main():
    parser = argparse.ArgumentParser(description="Test deki-automata API")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--prompt", default="Open whatsapp and write my friend user_name that I will be late for 15 minutes",
                        help="User prompt for the API")
    parser.add_argument("--api_url", default="http://localhost:80/analyze", help="API endpoint URL")
    parser.add_argument("--token", required=True, help="API token for Authorization header")
    args = parser.parse_args()

    # Convert the image to base64
    image_data = encode_image(args.image_path)

    # Build the payload
    payload = {
        "prompt": args.prompt,
        "image": image_data
    }

    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json"
    }

    # Send the POST request
    response = requests.post(args.api_url, json=payload, headers=headers)
    
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception:
        print("Response Text:", response.text)

if __name__ == "__main__":
    main()
