import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from diffusers import StableDiffusionPipeline
import requests
from PIL import Image
from io import BytesIO

# Load dataset
fashion_data = pd.read_csv("fashion.csv")

# Cache model loading
@st.cache_resource
def load_models():
    # Load the Sentence Transformer model for image embeddings
    model = SentenceTransformer('clip-ViT-B-32')
    # Load the Stable Diffusion model using diffusers
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to("cpu")  # Use CPU instead of CUDA
    return model, pipe

model, pipe = load_models()

# Function to generate an image based on text
@st.cache_data
def generate_image(text):
    image = pipe(text).images[0]
    return image

# Function to get image embeddings
def get_image_embedding(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_tensor = model.encode([img])
    return img_tensor

# Create a sidebar for user input
st.sidebar.header("Fashion Search")
user_query = st.sidebar.text_input("Describe the product you're looking for:")

if user_query:
    # Generate an image based on user query
    st.write("Generating image based on your query...")
    generated_image = generate_image(user_query)

    st.image(generated_image, caption="Generated Image", use_column_width=True)

    # Get the embedding of the generated image
    generated_embedding = model.encode([generated_image])

    # Find similar images in the dataset
    st.write("Finding similar images in the dataset...")
    fashion_data['embedding'] = fashion_data['img'].apply(get_image_embedding)

    # Calculate cosine similarity
    similarities = util.cos_sim(generated_embedding, torch.stack(fashion_data['embedding'].tolist())).squeeze()
    top_matches = similarities.topk(5).indices

    # Display the top matches
    for idx in top_matches:
        st.image(fashion_data.iloc[idx]['img'], caption=fashion_data.iloc[idx]['name'], use_column_width=True)
        st.write(f"Price: {fashion_data.iloc[idx]['price']}")
        st.write(f"Brand: {fashion_data.iloc[idx]['brand']}")
        st.write(f"Rating: {fashion_data.iloc[idx]['avg_rating']}/5 from {fashion_data.iloc[idx]['ratingCount']} reviews")

else:
    st.write("Please enter a query to start searching.")
