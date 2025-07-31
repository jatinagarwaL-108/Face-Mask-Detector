import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import os
import gdown

model_file = "face_detect_model1.h5"

if not os.path.exists(model_file):
    url = "https://drive.google.com/uc?id=1faU_ThFAiQxuWU2eSKGBHkmIsnugNMvR"
    gdown.download(url, model_file, quiet=False)

model = tf.keras.models.load_model(model_file)
haar = cv2.CascadeClassifier("haarcascade_frontalface_default (1).xml")

# Face detection
def detect_face(img):
    return haar.detectMultiScale(img)

# Predict mask
def detect_image(img):
    y_pred = model.predict(img.reshape(1, 224, 224, 3))
    return y_pred[0][0]

# Draw label
def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    end_x = pos[0] + text_size[0][0] + 2
    end_y = pos[1] + text_size[0][1] + 2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness=-1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

# Streamlit UI
st.set_page_config(page_title="Face Mask Detection", layout="centered")
st.markdown("<h1 style='text-align: center;'>😷 Face Mask Detection</h1>", unsafe_allow_html=True)
st.sidebar.title("Choose Mode")

mode = st.sidebar.radio("Select input type", ["📷 Live Webcam", "🖼️ Upload Image"])

if mode == "📷 Live Webcam":
    img_file = st.camera_input("Take a photo")
    FRAME_WINDOW = st.image([])
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        resized = cv2.resize(frame, (224, 224))
        pred = detect_image(resized)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_face(gray)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)

        if pred < 1e-2:
            draw_label(frame, "with_Mask", (30, 30), (0, 255, 0))
        else:
            draw_label(frame, "without_mask", (30, 30), (0, 0, 255))

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        frame = np.array(img)
        display_img = frame.copy()

        resized_img = cv2.resize(frame, (224, 224))
        y_pred = detect_image(resized_img)
        gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
        faces = detect_face(gray)

        for x, y, w, h in faces:
            cv2.rectangle(display_img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=3)

        if y_pred < 1e-2:
            st.markdown("<h2 style='color:green; text-align:center;'>😷 Person is wearing a Mask</h2>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:red; text-align:center;'>❌ Person is NOT wearing a Mask</h2>",
                        unsafe_allow_html=True)

        st.image(display_img, caption="Detected Result", channels="BGR")
