import streamlit as st
from PIL import Image
import imagehash
import base64
import json
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Paths and configurations
UPLOAD_FOLDER = 'static/uploads/'
SIMILAR_PHOTOS_FOLDER = 'static/similar_photos/'
IMAGE_INFO_FILE = 'static/image_info.json'

# Create folders if they do not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SIMILAR_PHOTOS_FOLDER, exist_ok=True)
if not os.path.exists(IMAGE_INFO_FILE):
    with open(IMAGE_INFO_FILE, 'w') as f:
        json.dump([], f)

# Define functions
def categorize_image(image_path):
    if 'jeans' in image_path.lower():
        return 'jeans'
    elif 'shirt' in image_path.lower():
        return 'shirt'
    else:
        return 'unknown'

def detect_color(image):
    dominant_color = image.convert('RGB').getpixel((image.width // 2, image.height // 2))
    if dominant_color[0] > dominant_color[1] and dominant_color[0] > dominant_color[2]:
        return 'red'
    elif dominant_color[1] > dominant_color[0] and dominant_color[1] > dominant_color[2]:
        return 'green'
    elif dominant_color[2] > dominant_color[0] and dominant_color[2] > dominant_color[1]:
        return 'blue'
    else:
        return 'unknown'

def is_vision_model_loaded():
    try:
        completion = client.chat.completions.create(
            model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            messages=[
                {"role": "system", "content": "Always answer in rhymes."},
                {"role": "user", "content": "Introduce yourself."}
            ],
            temperature=0.7,
        )
        return True
    except openai.error.BadRequestError as e:
        if 'Vision model is not loaded' in str(e):
            return False
        raise e

def find_similar_images(uploaded_image_path, uploaded_category, uploaded_color):
    uploaded_image = Image.open(uploaded_image_path)
    uploaded_image_hash = imagehash.average_hash(uploaded_image)
    
    with open(IMAGE_INFO_FILE, 'r') as f:
        image_info = json.load(f)
    
    similar_images = set()
    for image in image_info:
        current_image_hash = imagehash.hex_to_hash(image['hash'])
        similarity_score = uploaded_image_hash - current_image_hash
        if similarity_score < 10 and image.get('category') == uploaded_category and image.get('color') == uploaded_color:
            similar_images.add(image['filename'])
    
    return similar_images

def store_image_info(file_path, category, color):
    image = Image.open(file_path)
    image_hash = imagehash.average_hash(image)

    image_info = []
    if os.path.exists(IMAGE_INFO_FILE):
        with open(IMAGE_INFO_FILE, 'r') as f:
            image_info = json.load(f)

    image_info.append({'filename': os.path.basename(file_path), 'hash': str(image_hash), 'category': category, 'color': color})

    with open(IMAGE_INFO_FILE, 'w') as f:
        json.dump(image_info, f)

# Streamlit app
def main():
    st.title("Image Upload and Analysis")

    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if uploaded_files:
        if not is_vision_model_loaded():
            st.error("Vision model is not loaded. Please load the Vision model and try again.")
            return

        similar_images = set()
        response_content = ""

        for uploaded_file in uploaded_files:
            if uploaded_file:
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                category = categorize_image(file_path)
                image = Image.open(file_path)
                color = detect_color(image)

                store_image_info(file_path, category, color)
                
                # Display the uploaded image
                st.image(file_path, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
                
                # Convert image to base64
                with open(file_path, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode("utf-8")

                completion = client.chat.completions.create(
                    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                    messages=[
                        {
                            "role": "system",
                            "content": "This is a chat between a user and an assistant. The assistant is helping the user to describe an image.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What’s in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=1000,
                    stream=True
                )

                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta and 'content' in chunk.choices[0].delta:
                        response_content += chunk.choices[0].delta.content

                # Find similar images
                similar_images.update(find_similar_images(file_path, category, color))

        st.write("Description of uploaded images:")
        st.write(response_content)

        st.write("Similar images found:")
        for similar_image in similar_images:
            st.image(os.path.join(SIMILAR_PHOTOS_FOLDER, similar_image), use_column_width=True)

if __name__ == "__main__":
    main()
