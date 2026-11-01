
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Get the folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Load trained CNN model
MODEL_PATH = os.path.join(BASE_DIR, "shape_cnn_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

print("Trained CNN model loaded successfully!")

# 3. Load dataset
DATASET_PATH = os.path.join(BASE_DIR, "shape_dataset.npz")
data = np.load(DATASET_PATH)

X = data["X"]
y = data["y"]

# 4. Shape names
class_names = ["Circle", "Square", "Triangle", "Rectangle"]

# 5. Select one sample from each class
sample_images = []
sample_labels = []

for class_id in range(4):
    index = np.where(y == class_id)[0][0]
    sample_images.append(X[index])
    sample_labels.append(y[index])

# 6. Prepare images for prediction
sample_images = np.array(sample_images).astype("float32")
sample_images = sample_images.reshape(-1, 28, 28, 1)

# 7. Predict shapes
predictions = model.predict(sample_images, verbose=0)

# 8. Display results
plt.figure(figsize=(10, 4))

for i in range(4):
    predicted_class = np.argmax(predictions[i])
    confidence = predictions[i][predicted_class] * 100

    print("\nActual Shape:", class_names[sample_labels[i]])
    print("Predicted Shape:", class_names[predicted_class])
    print("Confidence:", round(confidence, 2), "%")

    plt.subplot(1, 4, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap="gray")
    plt.title(
        f"Actual: {class_names[sample_labels[i]]}\n"
        f"Predicted: {class_names[predicted_class]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nShape prediction testing completed!")
