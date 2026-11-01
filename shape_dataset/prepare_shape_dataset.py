
import numpy as np
import cv2
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_PER_CLASS = 5000
IMAGE_SIZE = 28
MAX_SHAPE_SIZE = 20

shape_files = {
    "circle": "full_numpy_bitmap_circle.npy",
    "square": "full_numpy_bitmap_square.npy",
    "triangle": "full_numpy_bitmap_triangle.npy"
}

label_mapping = {
    "circle": 0,
    "square": 1,
    "triangle": 2,
    "rectangle": 3
}


def preprocess_image(image):
    """Crop, resize and centre a shape in a 28x28 black image."""

    image = image.astype(np.uint8)

    # Find the foreground pixels
    _, binary = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

    points = cv2.findNonZero(binary)

    if points is None:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    # Crop around the shape
    x, y, width, height = cv2.boundingRect(points)
    cropped = image[y:y + height, x:x + width]

    # Resize while preserving the original proportions
    scale = MAX_SHAPE_SIZE / max(width, height)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    # Place the shape in the centre
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    start_x = (IMAGE_SIZE - new_width) // 2
    start_y = (IMAGE_SIZE - new_height) // 2

    canvas[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = resized

    return canvas


all_images = []
all_labels = []

# Process circle, square and triangle images
for shape, filename in shape_files.items():

    file_path = os.path.join(BASE_DIR, filename)

    print("Loading:", shape)

    data = np.load(file_path, mmap_mode="r")

    if len(data) < SAMPLES_PER_CLASS:
        raise ValueError(f"Not enough images for {shape}")

    indices = np.random.default_rng(42).choice(
        len(data),
        SAMPLES_PER_CLASS,
        replace=False
    )

    processed_images = []

    for index in indices:
        image = np.asarray(data[index]).reshape(28, 28)
        processed_images.append(preprocess_image(image))

    all_images.extend(processed_images)
    all_labels.extend([shape] * SAMPLES_PER_CLASS)

    print("Processed:", shape, len(processed_images))


# Process rectangle images
rectangle_folder = os.path.join(BASE_DIR, "rectangle")
rectangle_files = glob.glob(os.path.join(rectangle_folder, "*.png"))

if len(rectangle_files) < SAMPLES_PER_CLASS:
    raise ValueError("Not enough rectangle images. Generate them first.")

print("Loading: rectangle")

rng = np.random.default_rng(42)

selected_files = rng.choice(
    rectangle_files,
    SAMPLES_PER_CLASS,
    replace=False
)

rectangle_images = []

for file_path in selected_files:

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    rectangle_images.append(preprocess_image(image))

if len(rectangle_images) != SAMPLES_PER_CLASS:
    raise ValueError(
        f"Expected {SAMPLES_PER_CLASS} rectangles, "
        f"but loaded {len(rectangle_images)}"
    )

all_images.extend(rectangle_images)
all_labels.extend(["rectangle"] * SAMPLES_PER_CLASS)

print("Processed: rectangle", len(rectangle_images))


# Convert images and labels into arrays
X = np.array(all_images, dtype=np.float32) / 255.0

y = np.array([
    label_mapping[label] for label in all_labels
], dtype=np.int32)

# Shuffle the complete dataset
indices = np.random.default_rng(42).permutation(len(X))

X = X[indices]
y = y[indices]

# Save the dataset
output_path = os.path.join(BASE_DIR, "shape_dataset.npz")

np.savez_compressed(
    output_path,
    X=X,
    y=y
)

print("\nDataset created successfully!")
print("Image shape:", X.shape)
print("Labels shape:", y.shape)

for label, shape in enumerate(label_mapping):
    print(shape, ":", np.sum(y == label), "images")

print("Minimum pixel value:", X.min())
print("Maximum pixel value:", X.max())
