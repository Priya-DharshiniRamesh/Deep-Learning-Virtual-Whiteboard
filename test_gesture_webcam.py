
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained gesture model
MODEL_PATH = os.path.join(BASE_DIR, "gesture_cnn_model.keras")
PREPROCESS_PATH = os.path.join(BASE_DIR, "gesture_preprocessing.npz")

model = tf.keras.models.load_model(MODEL_PATH)

# Load preprocessing information
preprocessing = np.load(PREPROCESS_PATH, allow_pickle=True)

mean = preprocessing["mean"]
scale = preprocessing["scale"]
classes = preprocessing["classes"]

print("Gesture CNN loaded successfully!")
print("Recognized gestures:", classes)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found!")
    hands.close()
    raise SystemExit

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks (x, y, z)
            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Convert to NumPy array
            features = np.array(landmarks, dtype=np.float32)

            # Apply the same scaling used during training
            features = (features - mean) / scale

            # Reshape for 1D CNN
            features = features.reshape(1, 21, 3)

            # Predict gesture
            prediction = model.predict(features, verbose=0)

            predicted_index = np.argmax(prediction[0])
            confidence = prediction[0][predicted_index] * 100

            gesture = classes[predicted_index]

            # Display prediction
            display_text = f"{gesture}: {confidence:.1f}%"

            cv2.putText(
                frame,
                display_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    else:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Gesture Recognition - CNN", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()

print("Webcam testing completed!")
