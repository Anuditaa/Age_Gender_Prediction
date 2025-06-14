#Prediction

import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Constants
IMG_SIZE = 64

#Preprocess input image
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

    img = preprocess_image(img_path, predict_mode=True)
    if img is None:
        print("Invalid image")
        return

    # Load and predict gender
    gender_model = tf.keras.models.load_model("gender_cnn_model.h5")
    gender_pred = gender_model.predict(img)
    gender_index = np.argmax(gender_pred)
    gender_conf = gender_pred[0][gender_index]

    # Load and predict continuous age
    age_model = tf.keras.models.load_model("age_regression_model.h5")
    age_pred = age_model.predict(img)
    predicted_age = age_pred[0][0]

    # Results
    print(f"\nPredicted Gender: {gender_map[gender_index]} ({gender_conf:.2f} confidence)")
    print(f"Predicted Age: {predicted_age:.1f} years")

# TEST: Replace with your image path
test_image_path = "photos/pic.jpg"
if os.path.exists(test_image_path):
    predict_gender_and_age(test_image_path)
else:
    print(f"Test image not found: {test_image_path}")
