
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Set project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "gesture_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "gesture_cnn_model.keras")

# 2. Load gesture dataset
df = pd.read_csv(CSV_PATH)

print("Gesture dataset loaded!")
print("Dataset shape:", df.shape)
print("\nGesture counts:")
print(df["label"].value_counts())

# 3. Separate features and labels
X = df.drop(columns=["label"]).values.astype("float32")
y = df["label"].values

# 4. Encode gesture names into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nGesture label mapping:")
for index, name in enumerate(encoder.classes_):
    print(index, ":", name)

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# 6. Normalize features using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Reshape for 1D CNN
X_train = X_train.reshape(-1, 21, 3)
X_test = X_test.reshape(-1, 21, 3)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 8. Build 1D CNN model
model = models.Sequential([
    layers.Input(shape=(21, 3)),

    layers.Conv1D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(64, kernel_size=3, activation="relu"),
    layers.GlobalMaxPooling1D(),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(len(encoder.classes_), activation="softmax")
])

# 9. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 10. Save preprocessing information
np.savez(
    os.path.join(BASE_DIR, "gesture_preprocessing.npz"),
    mean=scaler.mean_,
    scale=scaler.scale_,
    classes=encoder.classes_
)

# 11. Train model
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)

# 12. Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nGesture CNN Test Results")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")

# 13. Predict gestures
predictions = model.predict(X_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

# 14. Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(encoder.classes_))
)

plt.figure(figsize=(9, 7))
plt.imshow(cm, cmap="Blues")
plt.title("Gesture Recognition CNN - Confusion Matrix")
plt.xlabel("Predicted Gesture")
plt.ylabel("Actual Gesture")
plt.xticks(
    range(len(encoder.classes_)),
    encoder.classes_,
    rotation=45
)
plt.yticks(range(len(encoder.classes_)), encoder.classes_)

for i in range(len(encoder.classes_)):
    for j in range(len(encoder.classes_)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

# 15. Training accuracy graph
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Gesture CNN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nGesture CNN training completed!")
print("Model saved at:", MODEL_PATH)
print("Preprocessing saved at: gesture_preprocessing.npz")
