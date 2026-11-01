import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# CSV file
csv_file = "gesture_data.csv"

# Create CSV with header if it doesn't exist
if not os.path.exists(csv_file):
    header = ["label"]

    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Ask for gesture name
label = input("Enter gesture name: ")

# Open webcam
cap = cv2.VideoCapture(0)

print("Press SPACE to save a sample.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam.")
        break

    # Flip camera for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        f"Gesture: {label}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save landmark data
    if key == 32:  # SPACE
        if results.multi_hand_landmarks:

            landmarks = results.multi_hand_landmarks[0]

            row = [label]

            for landmark in landmarks.landmark:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"Saved sample for: {label}")

        else:
            print("No hand detected.")

    # Quit
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("Data collection completed.")