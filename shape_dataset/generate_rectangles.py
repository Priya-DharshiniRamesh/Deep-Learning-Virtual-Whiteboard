
import cv2
import numpy as np
import os
import random

output_folder = "rectangle"
os.makedirs(output_folder, exist_ok=True)

number_of_images = 10000

for i in range(number_of_images):

    image = np.zeros((28, 28), dtype=np.uint8)

    # Generate different rectangle sizes
    width = random.randint(10, 24)
    height = random.randint(6, 20)

    # Avoid squares
    if width == height:
        height = 8 if width != 8 else 12

    x = random.randint(1, 27 - width)
    y = random.randint(1, 27 - height)

    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        255,
        1
    )

    cv2.imwrite(
        os.path.join(output_folder, f"rectangle_{i}.png"),
        image
    )

print("Rectangle dataset generated successfully!")
