import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox

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
        print(f"❌ Error processing image: {e}")
        return None

# Predict gender and age
def predict_gender_and_age(img_path):
    gender_map = {0: "Male", 1: "Female"}

    img = preprocess_image(img_path, predict_mode=True)
    if img is None:
        return "❌ Image processing failed"

    try:
        gender_model = tf.keras.models.load_model("gender_cnn_model.h5")
        gender_pred = gender_model.predict(img)
        gender_index = np.argmax(gender_pred)
        gender_conf = gender_pred[0][gender_index]

        age_model = tf.keras.models.load_model("age_regression_model.h5")
        age_pred = age_model.predict(img)
        predicted_age = age_pred[0][0]

 
        result = f"🧠 Gender: {gender_map[gender_index]}\n🎂 Age: {predicted_age:.1f} years"
        return result
    except Exception as e:
        return f"❌ Prediction error: {e}"

# GUI App
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        result = predict_gender_and_age(file_path)
        result_label.config(text=result)

# Build GUI
root = tk.Tk()
root.title("Gender and Age Predictor")
root.geometry("400x300")
root.resizable(False, False)

tk.Label(root, text="👤 Select an Image to Predict", font=("Arial", 14)).pack(pady=20)
tk.Button(root, text="Browse Image", command=browse_image, font=("Arial", 12)).pack(pady=10)

result_label = tk.Label(root, text="", wraplength=350, font=("Arial", 12))
result_label.pack(pady=20)

root.mainloop()
