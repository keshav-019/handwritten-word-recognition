import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from PIL import Image
import pickle

# Load the model and mappings
model = load_model('final_model.keras')

with open('char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)

# Reverse the char_to_index mapping for decoding
index_to_char = {index: char for char, index in char_to_index.items()}

def preprocess_image(image, target_size=(64, 32)):
    """Preprocess the uploaded image."""
    img = image.convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to target size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, target_size[1], target_size[0], 1)  # Add batch dimension
    return img_array

def decode_output(prediction, index_to_char):
    """Decode the model output to Hindi word."""
    decoded_word = "".join([index_to_char[idx] for idx in prediction if idx != 0])
    return decoded_word

# Streamlit app
st.title("Devanagari Handwritten Word Recognition")
st.write("Upload an image of a handwritten Hindi word to predict the text.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        preprocessed_image = preprocess_image(image)

        # Get the model prediction
        prediction = model.predict(preprocessed_image)
        predicted_indices = np.argmax(prediction, axis=-1)[0]  # Get the predicted indices

        # Decode the prediction to Hindi word
        predicted_word = decode_output(predicted_indices, index_to_char)

        # Display the result
        st.write(f"**Predicted Word:** {predicted_word}")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload an image to proceed.")
