
# === IMPORTS ===
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# === DATA PATHS ===
healthy_path = r"C:\Users\HP\Desktop\DIP PROJECT\PlantVillage\Pepper__bell___healthy"
bacterial_path = r"C:\Users\HP\Desktop\DIP PROJECT\PlantVillage\Pepper__bell___Bacterial_spot"

# === LOAD & PREPROCESS IMAGES ===
image_size = (128, 128)
X, y = [], []

for label, folder in enumerate([healthy_path, bacterial_path]):
    count = 0
    for file in os.listdir(folder):
        if count >= 5: break
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None: continue
        img_resized = cv2.resize(img, image_size)
        X.append(img_resized)
        y.append(label)
        count += 1

X = np.array(X) / 255.0
y = to_categorical(np.array(y), num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL DEFINITION ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === TRAIN ===
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), callbacks=[early_stop])

# === EVALUATE ===
preds = model.predict(X_test)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test, axis=1)
print(classification_report(true_labels, pred_labels))

# === SAVE MODEL ===
model.save(r"\\PASCAL-OCHIENG\Users\HP\Desktop\DIP PROJECT\leaf_disease_model.keras")

# === TEST IMAGE FUNCTION ===
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img_resized = cv2.resize(img, (128, 128))
    return cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# === TEST ===
test_paths = [
    r"C:\Users\HP\Desktop\leaf_test.jpg",
    r"C:\Users\HP\Desktop\leaf_test2.jpg"
]

class_names = ["Healthy", "Unhealthy"]
for path in test_paths:
    test_img = preprocess_image(path)
    if test_img is not None:
        input_img = test_img.reshape(1, 128, 128, 3) / 255.0
        prediction = model.predict(input_img)
        class_idx = np.argmax(prediction)
        class_name = class_names[class_idx]
        print(f"Prediction for {os.path.basename(path)}: {class_name}")
        plt.imshow(test_img)
        plt.title(f"Predicted: {class_name}")
        plt.axis("off")
        plt.show()
    else:
        print(f"❌ Failed to load: {path}")
