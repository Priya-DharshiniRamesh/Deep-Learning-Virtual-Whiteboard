
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Set dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "shape_dataset.npz")

# 2. Load dataset
data = np.load(DATASET_PATH)

X = data["X"]
y = data["y"]

print("Dataset loaded successfully!")
print("Images:", X.shape)
print("Labels:", y.shape)

# 3. Prepare images for CNN
X = X.astype("float32")
X = X.reshape(-1, 28, 28, 1)

# 4. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

# 5. Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(4, activation="softmax")
])

# 6. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. Save the best model during training
MODEL_PATH = os.path.join(BASE_DIR, "shape_cnn_model.keras")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]

# 8. Train CNN model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks
)

# 9. Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nShape CNN Test Results")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# 10. Predict test images
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

class_names = ["Circle", "Square", "Triangle", "Rectangle"]

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=class_names,
    zero_division=0
))

# 11. Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

plt.figure(figsize=(7, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Shape Recognition CNN - Confusion Matrix")
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.xticks(range(4), class_names, rotation=20)
plt.yticks(range(4), class_names)
plt.colorbar()

for i in range(4):
    for j in range(4):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# 12. Plot training accuracy
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("CNN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nTraining completed!")
print("Model saved at:", MODEL_PATH)
