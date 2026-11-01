
import numpy as np
import matplotlib.pyplot as plt
import os

# Load combined dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "shape_dataset.npz")

data = np.load(file_path)

X = data["X"]
y = data["y"]

# Shape labels
labels = {
    0: "Circle",
    1: "Square",
    2: "Triangle",
    3: "Rectangle"
}

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Count images in each class
for number, name in labels.items():
    count = np.sum(y == number)
    print(f"{name}: {count} images")

# Display 16 random images
indices = np.random.choice(len(X), 16, replace=False)

plt.figure(figsize=(8, 8))

for i, index in enumerate(indices):
    plt.subplot(4, 4, i + 1)

    plt.imshow(X[index], cmap="gray")
    plt.title(labels[int(y[index])])
    plt.axis("off")

plt.tight_layout()
plt.show()
