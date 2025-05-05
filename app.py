# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# from PIL import Image
# import pickle

# # Load the model and mappings
# model = load_model('final_model.keras')

# with open('char_to_index.pkl', 'rb') as f:
#     char_to_index = pickle.load(f)

# # Reverse the char_to_index mapping for decoding
# index_to_char = {index: char for char, index in char_to_index.items()}

# def preprocess_image(image, target_size=(64, 32)):
#     """Preprocess the uploaded image."""
#     img = image.convert('L')  # Convert to grayscale
#     img = img.resize(target_size)  # Resize to target size
#     img_array = np.array(img) / 255.0  # Normalize
#     img_array = img_array.reshape(1, target_size[1], target_size[0], 1)  # Add batch dimension
#     return img_array

# def decode_output(prediction, index_to_char):
#     """Decode the model output to Hindi word."""
#     decoded_word = "".join([index_to_char[idx] for idx in prediction if idx != 0])
#     return decoded_word

# # Streamlit app
# st.title("Devanagari Handwritten Word Recognition")
# st.write("Upload an image of a handwritten Hindi word to predict the text.")

# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         # Load and preprocess the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#         preprocessed_image = preprocess_image(image)

#         # Get the model prediction
#         prediction = model.predict(preprocessed_image)
#         predicted_indices = np.argmax(prediction, axis=-1)[0]  # Get the predicted indices

#         # Decode the prediction to Hindi word
#         predicted_word = decode_output(predicted_indices, index_to_char)

#         # Display the result
#         st.write(f"**Predicted Word:** {predicted_word}")
#     except Exception as e:
#         st.error(f"Error processing the image: {e}")
# else:
#     st.info("Please upload an image to proceed.")


import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import time

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #4285F4;
        color: white;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .title {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("Devanagari Character Recognition")
st.markdown("---")

# Load model and labels
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('devanagari_cnn.keras')
    label_mapping = np.load('label_mapping.npy', allow_pickle=True).item()
    return model, label_mapping

model, label_mapping = load_model()

# File upload section
uploaded_file = st.file_uploader("Upload a Devanagari character image", 
                                type=["png", "jpg", "jpeg"],
                                accept_multiple_files=False)

col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)

with col2:
    if uploaded_file:
        # Preprocess and predict
        img_array = np.array(image.convert('L').resize((64, 64))) / 255.0
        img_array = img_array.reshape(1, 64, 64, 1)
        
        with st.spinner('Predicting...'):
            time.sleep(1)  # Simulate processing time
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_char = label_mapping[predicted_class]
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color: #2c3e50;'>Prediction Result</h3>
                <p style='font-size: 24px;'>Character: <strong>{predicted_char}</strong></p>
                <p>Confidence: {predictions[0][predicted_class]*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>Devanagari Character Recognition System</p>
</div>
""", unsafe_allow_html=True)



