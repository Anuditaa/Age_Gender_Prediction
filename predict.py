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

#Predict Gender and Age Group from New Image
def predict_gender_and_age(img_path):
    gender_map = {0: "Male", 1: "Female"}
    age_group_labels = {
        0: "0–12", 1: "13–19", 2: "20–29", 3: "30–39",
        4: "40–49", 5: "50–59", 6: "60–74", 7: "75+"
    }

    img = preprocess_image(img_path, predict_mode=True)
    if img is None:
        print("Invalid image")
        return

    # Load and predict gender
    gender_model = tf.keras.models.load_model("gender_cnn_model.h5")
    gender_pred = gender_model.predict(img)
    gender_index = np.argmax(gender_pred)
    gender_conf = gender_pred[0][gender_index]

    # Load and predict age group
    age_model = tf.keras.models.load_model("age_group_cnn_model.h5")
    age_pred = age_model.predict(img)
    age_group_index = np.argmax(age_pred)
    age_group_conf = age_pred[0][age_group_index]

    #Results
    print(f"\nPredicted Gender: {gender_map[gender_index]} ({gender_conf:.2f} confidence)")
    print(f"Predicted Age Group: {age_group_labels[age_group_index]} ({age_group_conf:.2f} confidence)")

#TEST: Replace with your image path
test_image_path = "photos/anudita.jpg"
if os.path.exists(test_image_path):
    predict_gender_and_age(test_image_path)
else:
    print(f"Test image not found: {test_image_path}")
