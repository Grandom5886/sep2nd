import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import logging
import requests
from PIL import Image
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT").rstrip('/')
)

# Azure Computer Vision API credentials from environment variables
subscription_key = os.getenv('AZURE_COMPUTER_VISION_SUBSCRIPTION_KEY')
endpoint = os.getenv('AZURE_COMPUTER_VISION_ENDPOINT')

# Streamlit UI
st.title("Text to Image and Image Captioning")

# Text to Image
st.header("Generate Image from Text")
text = st.text_input("Enter text to generate an image:")

if st.button("Generate Image"):
    if text:
        try:
            logger.debug("Attempting to generate image with Azure OpenAI")
            response = client.images.generate(
                model="dalle2",  # Replace with your actual deployment name
                prompt=text,
                n=1,
                size="256x256"
            )
            logger.debug(f"Azure OpenAI API response: {response}")
            
            if not response.data:
                st.error("No image generated")
            else:
                image_url = response.data[0].url
                st.image(image_url, caption="Generated Image", use_column_width=True)
                
                # Find similar images (placeholder implementation)
                similar_images = find_similar_images(image_url)
                st.subheader("Similar Images")
                for img_url in similar_images:
                    st.image(img_url, use_column_width=True)
        except Exception as e:
            logger.exception(f"An error occurred: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("No text provided")

# Image Captioning
st.header("Caption Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the uploaded image to a temporary file
    image_path = os.path.join('temp', uploaded_file.name)
    image.save(image_path)
    
    try:
        # Use Azure Computer Vision API to get captions
        captions = get_image_captions(image_path)
        st.subheader("Captions")
        for caption in captions:
            st.write(caption)
    finally:
        # Ensure the temporary image file is removed
        if os.path.exists(image_path):
            os.remove(image_path)

# Helper functions
def find_similar_images(image_url):
    # Placeholder logic to find similar images
    return ['https://via.placeholder.com/150', 'https://via.placeholder.com/150']

def get_image_captions(image_path):
    image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key, 'Content-Type': 'application/octet-stream'}
    params = {'maxCandidates': '1'}
    vision_url = f"{endpoint}/vision/v3.1/describe"
    logger.debug(f"Sending request to Azure Computer Vision API: {vision_url}")
    response = requests.post(vision_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    analysis = response.json()
    logger.debug(f"Azure API response: {analysis}")
    
    # Extract captions
    captions = [caption['text'] for caption in analysis['description']['captions']]
    return captions

if not os.path.exists('temp'):
    os.makedirs('temp')
