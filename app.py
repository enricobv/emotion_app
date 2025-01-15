import streamlit as st
from fer import FER
from PIL import Image
import cv2
import numpy as np

# Title of the app
st.title('Emotion Recognition from Faces using FER')

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert the image to a numpy array
    img = np.array(image)

    # Convert RGB to BGR (required by OpenCV)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Initialize the FER detector
    detector = FER()

    # Detect emotions in the image
    emotion, score = detector.top_emotion(img_bgr)

    # Display the result
    st.write(f"Dominant Emotion: {emotion} with a confidence of {score}")
