# Prediction

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2  # Added for face detection

# Constants
IMG_SIZE = 64

# Preprocess input image
def preprocess_image(img_path, predict_mode=False):
    try:
        img = Image.open(img_path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # channel
        if predict_mode:
            img = np.expand_dims(img, axis=0)  # batch
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Predict Gender and Age (Regression) from New Image
def predict_gender_and_age(img_path):
    gender_map = {0: "Male", 1: "Female"}

    # Use OpenCV to check if a face is present
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("Error: Cannot read image with OpenCV.")
        return
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("Error: No face detected in the image.")
        return

    # Continue with prediction only if face is detected
    img = preprocess_image(img_path, predict_mode=True)
    if img is None:
        print("Invalid image")
        return

    # Load and predict gender
    gender_model = tf.keras.models.load_model("gender_cnn_model.h5")
    gender_pred = gender_model.predict(img)
    gender_index = np.argmax(gender_pred)
    gender_conf = gender_pred[0][gender_index]

    # Load and predict age
    age_model = tf.keras.models.load_model("age_regression_model.h5")
    age_pred = age_model.predict(img)
    predicted_age = age_pred[0][0]

    # Results
    print(f"\nPredicted Gender: {gender_map[gender_index]} ({gender_conf:.2f} confidence)")
    print(f"Predicted Age: {predicted_age:.1f} years")

# TEST: Replace with your image path
test_image_path = "photos/kid4.jpg"
if os.path.exists(test_image_path):
    predict_gender_and_age(test_image_path)
else:
    print(f"Test image not found: {test_image_path}")