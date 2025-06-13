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

# ⬇️ Download UTKFace dataset
dataset_path = kagglehub.dataset_download("jangedoo/utkface-new")
img_base_path = os.path.join(dataset_path, "UTKFace")

# 📌 Hyperparameters
IMG_SIZE = 64
EPOCHS = 3
FOLDS = 3
BATCH_SIZE = 32

# ⬇️ Extract age and gender from filename
def parse_filename(fname):
    try:
        parts = fname.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        return age, gender
    except:
        return None, None

# 🧼 Image preprocessing
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
        print(f"⚠️ Error processing {img_path}: {e}")
        return None

# 📥 Load images and extract labels
print("📥 Loading and preprocessing UTKFace images...")
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
    raise ValueError("❌ No images were successfully loaded. Please check dataset and structure.")

print(f"✅ Loaded {len(images)} images.")

# ✅ Prepare data
X = np.array(images)
y_gender = LabelEncoder().fit_transform(gender_labels)
y_gender_cat = to_categorical(y_gender)

# 🎂 Convert actual age to age group (classification)
def get_age_group(age):
    if age <= 12:
        return 0
    elif age <= 19:
        return 1
    elif age <= 29:
        return 2
    elif age <= 39:
        return 3
    elif age <= 49:
        return 4
    elif age <= 59:
        return 5
    elif age <= 74:
        return 6
    else:
        return 7

y_age_group = np.array([get_age_group(age) for age in age_labels])
y_age_group_cat = to_categorical(y_age_group, num_classes=8)

# 🧠 CNN model definitions
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

def create_age_group_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 🔁 Train Gender Model
print("\n🔁 Training Gender Prediction Model...")
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
gender_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_gender)):
    print(f"\n📂 Gender - Fold {fold + 1}")
    model_gender = create_gender_model()
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_gender_cat[train_idx], y_gender_cat[val_idx]

    model_gender.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=1)
    loss, acc = model_gender.evaluate(X_val, y_val, verbose=0)
    print(f"✅ Fold {fold + 1} Accuracy: {acc:.4f}")
    gender_accuracies.append(acc)

model_gender.save("gender_cnn_model.h5")
print("\n💾 Gender model saved as gender_cnn_model.h5")
print(f"🎯 Avg Gender Accuracy: {np.mean(gender_accuracies):.4f}")

# 🔁 Train Age Group Model
print("\n🔁 Training Age Group Prediction Model...")
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
age_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_age_group)):
    print(f"\n📂 Age Group - Fold {fold + 1}")
    model_age_group = create_age_group_model()
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_age_group_cat[train_idx], y_age_group_cat[val_idx]

    model_age_group.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=1)
    loss, acc = model_age_group.evaluate(X_val, y_val, verbose=0)
    print(f"✅ Fold {fold + 1} Accuracy: {acc:.4f}")
    age_accuracies.append(acc)

model_age_group.save("age_group_cnn_model.h5")
print("\n💾 Age group model saved as age_group_cnn_model.h5")
print(f"🎯 Avg Age Group Accuracy: {np.mean(age_accuracies):.4f}")

# 👤 Predict Gender and Age Group from New Image
def predict_gender_and_age(img_path):
    gender_map = {0: "Male", 1: "Female"}
    age_group_labels = {
        0: "0–12", 1: "13–19", 2: "20–29", 3: "30–39",
        4: "40–49", 5: "50–59", 6: "60–74", 7: "75+"
    }

    img = preprocess_image(img_path, predict_mode=True)
    if img is None:
        print("❌ Invalid image")
        return

    # Predict gender
    gender_model = tf.keras.models.load_model("gender_cnn_model.h5")
    gender_pred = gender_model.predict(img)
    gender_index = np.argmax(gender_pred)
    gender_conf = gender_pred[0][gender_index]

    # Predict age group
    age_model = tf.keras.models.load_model("age_group_cnn_model.h5")
    age_pred = age_model.predict(img)
    age_group_index = np.argmax(age_pred)
    age_group_conf = age_pred[0][age_group_index]

    print(f"\n🧠 Predicted Gender: {gender_map[gender_index]} ({gender_conf:.2f} confidence)")
    print(f"🎂 Predicted Age Group: {age_group_labels[age_group_index]} ({age_group_conf:.2f} confidence)")

# 🖼️ TEST: Replace path with your image
test_image_path = "pic.jpg"
if os.path.exists(test_image_path):
    predict_gender_and_age(test_image_path)
else:
    print(f"🖼️ Test image not found: {test_image_path}")
