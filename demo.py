import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st

# Constants
OUTPUT_LABELS = ['acne', 'clear', 'wrinkle']
IMG_SIZE = (128, 128)

# Helper Functions
def center_crop(image, target_size):
    """Center crop the image to the smallest dimension and then resize to target_size."""
    h, w, _ = image.shape
    crop_size = min(h, w)
    if h != w:
        image = tf.image.central_crop(image, central_fraction=crop_size / max(h, w))
    image = tf.image.resize(image, target_size)
    return image.numpy()

def identify_disease(image, model):
    """Predict the skin condition from the image."""
    img = center_crop(image, IMG_SIZE)  # Center crop and resize
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    disease_index = np.argmax(prediction)
    output = OUTPUT_LABELS[disease_index]
    confidence = prediction[0][disease_index]
    return output, confidence

# Load Pre-trained Model
@st.cache_resource
def load_skin_model():
    return load_model('best_skin_classification.keras')


model = load_skin_model()

# Streamlit Interface
st.title("Skin Disease Classification")
st.write("Upload an image or capture a photo to classify skin conditions into Acne, Clear, or Wrinkle.")

# Upload or Capture Image
option = st.radio("Select Image Input Method:", ["Upload Image", "Use Webcam"])
image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image = np.array(image)

elif option == "Use Webcam":
    picture = st.camera_input("Capture a photo")
    if picture is not None:
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_container_width=True)
        image = np.array(image)

# Perform Prediction
if image is not None:
    result, confidence = identify_disease(image, model)
    st.write(f"**Prediction:** {result}")

# Instructions for Deployment
st.sidebar.title("Instructions")
st.sidebar.write("""
- **Upload Image:** Select an image file for analysis.
- **Use Webcam:** Capture a photo directly.
- Ensure the image focuses on the skin area for accurate results.
""")
