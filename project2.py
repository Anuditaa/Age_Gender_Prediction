import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import kagglehub

#Download UTKFace dataset
dataset_path = kagglehub.dataset_download("jangedoo/utkface-new")
img_base_path = os.path.join(dataset_path, "UTKFace")

#Hyperparameters
IMG_SIZE = 64
EPOCHS = 10
FOLDS = 5
BATCH_SIZE = 32

#Extract age and gender from filename
def parse_filename(fname):
    try:
        parts = fname.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        return age, gender
    except:
        return None, None

#Image preprocessing
def preprocess_image(img_path, predict_mode=False):
    try:
        img = Image.open(img_path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)  # channel dimension
        if predict_mode:
            img = np.expand_dims(img, axis=0)  # batch dimension
        return img
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

#Load images and extract labels
print("Loading and preprocessing UTKFace images...")
images = []
gender_labels = []
age_labels = []

for fname in os.listdir(img_base_path):
    if not fname.endswith(".jpg"):
        continue
    age, gender = parse_filename(fname)
    if gender is None or age is None:
        continue
    img_path = os.path.join(img_base_path, fname)
    img = preprocess_image(img_path)
    if img is not None:
        images.append(img)
        gender_labels.append(gender)
        age_labels.append(age)

if not images:
    raise ValueError("No images were successfully loaded. Please check dataset and structure.")

print(f"Loaded {len(images)} images.")

#Prepare data
X = np.array(images)
y_gender = LabelEncoder().fit_transform(gender_labels)
y_gender_cat = to_categorical(y_gender)
y_age_reg = np.array(age_labels, dtype=np.float32)  # Regression target

#CNN model definitions
def create_gender_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_age_regression_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Continuous age prediction
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

#Train Gender Model
print("\nTraining Gender Prediction Model...")
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
gender_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_gender)):
    print(f"\nGender - Fold {fold + 1}")
    model_gender = create_gender_model()
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_gender_cat[train_idx], y_gender_cat[val_idx]

    model_gender.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_data=(X_val, y_val), verbose=1)
    loss, acc = model_gender.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    gender_accuracies.append(acc)

model_gender.save("gender_cnn_model.h5")
print("\nGender model saved as gender_cnn_model.h5")
print(f"Avg Gender Accuracy: {np.mean(gender_accuracies):.4f}")

#Train Age Regression Model
print("\nTraining Age Regression Model...")
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
age_maes = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_age_reg)):
    print(f"\nAge Regression - Fold {fold + 1}")
    model_age_reg = create_age_regression_model()
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_age_reg[train_idx], y_age_reg[val_idx]

    model_age_reg.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_data=(X_val, y_val), verbose=1)
    loss, mae = model_age_reg.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold + 1} MAE: {mae:.2f}")
    age_maes.append(mae)

model_age_reg.save("age_regression_model.h5")
print("\nAge regression model saved as age_regression_model.h5")
print(f"Avg Age MAE: {np.mean(age_maes):.2f}")

#Predict Gender and Age (Regression) from New Image
def predict_gender_and_age(img_path):
    gender_map = {0: "Male", 1: "Female"}

    img = preprocess_image(img_path, predict_mode=True)
    if img is None:
        print("Invalid image")
        return

    # Predict gender
    gender_model = tf.keras.models.load_model("gender_cnn_model.h5")
    gender_pred = gender_model.predict(img)
    gender_index = np.argmax(gender_pred)
    gender_conf = gender_pred[0][gender_index]

    # Predict age (regression)
    age_model = tf.keras.models.load_model("age_regression_model.h5")
    age_pred = age_model.predict(img)
    predicted_age = age_pred[0][0]

    print(f"\nPredicted Gender: {gender_map[gender_index]} ({gender_conf:.2f} confidence)")
    print(f"Predicted Age: {predicted_age:.1f} years")

#TEST: Replace with your image
test_image_path = "pic.jpg"
if os.path.exists(test_image_path):
    predict_gender_and_age(test_image_path)
else:
    print(f"Test image not found: {test_image_path}")





