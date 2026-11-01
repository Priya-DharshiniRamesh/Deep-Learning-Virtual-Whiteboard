import os
import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque


# ============================================
# 1. PROJECT PATHS
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GESTURE_MODEL_PATH = os.path.join(BASE_DIR, "gesture_cnn_model.keras")
GESTURE_PREPROCESS_PATH = os.path.join(BASE_DIR, "gesture_preprocessing.npz")
SHAPE_MODEL_PATH = os.path.join(BASE_DIR, "shape_dataset", "shape_cnn_model.keras")


# ============================================
# 2. LOAD TRAINED MODELS
# ============================================

print("Loading models...")

for path in [GESTURE_MODEL_PATH, GESTURE_PREPROCESS_PATH, SHAPE_MODEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

gesture_model = tf.keras.models.load_model(GESTURE_MODEL_PATH)

gesture_data = np.load(GESTURE_PREPROCESS_PATH, allow_pickle=True)
gesture_mean = gesture_data["mean"]
gesture_scale = np.where(gesture_data["scale"] == 0, 1.0, gesture_data["scale"])
gesture_classes = gesture_data["classes"]

shape_model = tf.keras.models.load_model(SHAPE_MODEL_PATH)

shape_classes = ["Circle", "Square", "Triangle", "Rectangle"]

print("Both CNN models loaded successfully!")


# ============================================
# 3. WHITEBOARD SETTINGS
# ============================================

WIDTH = 1280
HEIGHT = 655

DRAW_COLOR = (0, 255, 255)
SELECT_COLOR = (255, 0, 255)
# Selected fill colour is stored in OpenCV BGR format.
FILL_COLORS = {
    "Black": (0, 0, 0),
    "Red": (0, 0, 255),
    "Green": (0, 128, 0),
    "Dark Blue": (139, 0, 0),
    "Yellow": (0, 255, 255),
    "Pink": (180, 105, 255),
    "Orange": (0, 165, 255),
    "Violet": (238, 130, 238),
    "Sky Blue": (235, 206, 135),
    "Gray": (128, 128, 128),
    "Brown": (42, 42, 165),
    "Light Pink": (193, 182, 255),
}
selected_fill_name = "Red"
selected_fill_color = FILL_COLORS[selected_fill_name]

BRUSH_SIZE = 8
ERASER_SIZE = 50

CONFIDENCE_THRESHOLD = 70

SMARTSHAPE_HOLD_TIME = 1.0
SMARTSHAPE_GRACE_TIME = 0.40
SHAPE_CONFIDENCE_THRESHOLD = 70

objects = []

current_layer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

selected_object = None

previous_x = None
previous_y = None

drag_previous = None

resize_source = None
resize_start_distance = None
resize_object_index = None

smartshape_start = None
smartshape_anchor = None
smartshape_triggered = False
smartshape_last_seen = None

gesture_prediction_history = deque(maxlen=5)

interaction_mode = "SELECT"
resize_mode = False

WINDOW_NAME = "Deep Learning Virtual Whiteboard"
MODE_BUTTON = (1040, 12, 1255, 48)
RESIZE_BUTTON = (1040, 56, 1255, 92)

# Colour palette: all 12 colours in one horizontal row at the top.
PALETTE_X = 16
PALETTE_Y = 14
# Compact rectangular swatches with a small gap between adjacent colours.
PALETTE_CELL_W = 42
PALETTE_CELL_H = 40
PALETTE_SWATCH_W = 38
PALETTE_SWATCH_H = 34
PALETTE_COLUMNS = 12
PALETTE_PREVIEW_BOX = (530, 14, 568, 48)

fill_triggered = False

last_detected_shape = None
last_shape_confidence = 0.0

last_message = "Draw something!"

MAX_HISTORY = 10
undo_history = []
redo_history = []

draw_action_active = False
erase_action_active = False
drag_action_active = False
resize_action_active = False

# Closed-fist stop control: suppress Draw/Drag flicker after stopping.
drawing_stop_latched = False
drawing_stop_release_count = 0
DRAWING_STOP_RELEASE_FRAMES = 3


# ============================================
# 4. FIND CONTOURS IN AN OBJECT
# ============================================

def get_object_contour(layer):
    gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 100:
        return None

    return contour


# ============================================
# 5. SAVE CURRENT DRAWING AS AN OBJECT
# ============================================

def commit_current_layer():
    global current_layer

    if np.any(current_layer):
        objects.append({"layer": current_layer.copy()})

        print("Drawing saved as object. Total objects:", len(objects))

        current_layer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


# ============================================
# 6. UNDO AND REDO
# ============================================

def save_history_state():
    state = {
        "objects": [{"layer": obj["layer"].copy()} for obj in objects],
        "current_layer": current_layer.copy(),
        "selected_object": selected_object
    }

    undo_history.append(state)

    if len(undo_history) > MAX_HISTORY:
        undo_history.pop(0)

    redo_history.clear()


def restore_history_state(state):
    global objects, current_layer, selected_object
    global previous_x, previous_y
    global drag_previous, resize_source
    global resize_start_distance, resize_object_index

    objects = [{"layer": obj["layer"].copy()} for obj in state["objects"]]
    current_layer = state["current_layer"].copy()
    selected_object = state["selected_object"]

    previous_x = None
    previous_y = None
    drag_previous = None
    resize_source = None
    resize_start_distance = None
    resize_object_index = None


def undo_action():
    if not undo_history:
        print("Nothing to undo.")
        return

    current_state = {
        "objects": [{"layer": obj["layer"].copy()} for obj in objects],
        "current_layer": current_layer.copy(),
        "selected_object": selected_object
    }

    redo_history.append(current_state)
    previous_state = undo_history.pop()
    restore_history_state(previous_state)

    print("Undo successful.")


def redo_action():
    if not redo_history:
        print("Nothing to redo.")
        return

    current_state = {
        "objects": [{"layer": obj["layer"].copy()} for obj in objects],
        "current_layer": current_layer.copy(),
        "selected_object": selected_object
    }

    undo_history.append(current_state)

    if len(undo_history) > MAX_HISTORY:
        undo_history.pop(0)

    next_state = redo_history.pop()
    restore_history_state(next_state)

    print("Redo successful.")


# ============================================
# 7. FIND OBJECT UNDER THE HAND
# ============================================

def find_object_at(x, y):
    containing_objects = []
    nearby_objects = []

    for index, obj in enumerate(objects):
        contour = get_object_contour(obj["layer"])

        if contour is None:
            continue

        distance = cv2.pointPolygonTest(contour, (float(x), float(y)), True)

        if distance >= 0:
            containing_objects.append((distance, index))
        else:
            nearby_objects.append((abs(distance), index))

    if containing_objects:
        return max(containing_objects, key=lambda item: item[0])[1]

    if nearby_objects:
        distance, index = min(nearby_objects, key=lambda item: item[0])

        if distance <= 40:
            return index

    return None


# ============================================
# 8. RECOGNIZE SHAPE USING 2D CNN
# ============================================

def recognize_shape(layer):
    contour = get_object_contour(layer)

    if contour is None:
        return None, 0.0

    x, y, w, h = cv2.boundingRect(contour)

    gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    cropped = gray[y:y + h, x:x + w]

    if cropped.size == 0:
        return None, 0.0

    _, binary = cv2.threshold(cropped, 10, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(binary)

    if points is None:
        return None, 0.0

    bx, by, bw, bh = cv2.boundingRect(points)
    cropped = cropped[by:by + bh, bx:bx + bw]

    height, width = cropped.shape
    scale = 20.0 / max(height, width)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    resized = cv2.resize(
        cropped, (new_width, new_height), interpolation=interpolation
    )

    image = np.zeros((28, 28), dtype=np.uint8)

    start_x = (28 - new_width) // 2
    start_y = (28 - new_height) // 2

    image[start_y:start_y + new_height, start_x:start_x + new_width] = resized

    image = image.astype(np.float32) / 255.0
    image = image.reshape(1, 28, 28, 1)

    prediction = shape_model(image, training=False).numpy()[0]

    index = int(np.argmax(prediction))
    confidence = float(prediction[index] * 100)

    return shape_classes[index], confidence


# ============================================
# 9. AUTOMATIC GEOMETRIC SHAPE CORRECTION
# ============================================

def correct_shape(layer, shape_name):
    contour = get_object_contour(layer)

    if contour is None:
        return False

    x, y, w, h = cv2.boundingRect(contour)

    if w < 10 or h < 10:
        return False

    cv2.drawContours(layer, [contour], -1, (0, 0, 0), cv2.FILLED)

    margin = max(2, BRUSH_SIZE)
    left = max(margin, x + margin)
    top = max(margin, y + margin)
    right = min(WIDTH - margin - 1, x + w - margin)
    bottom = min(HEIGHT - margin - 1, y + h - margin)

    shape_width = right - left
    shape_height = bottom - top

    if shape_width < 5 or shape_height < 5:
        return False

    if shape_name == "Circle":
        center = ((left + right) // 2, (top + bottom) // 2)
        radius = max(2, min(shape_width, shape_height) // 2)

        cv2.circle(layer, center, radius, DRAW_COLOR, BRUSH_SIZE)

    elif shape_name == "Square":
        side = min(shape_width, shape_height)
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        half = side // 2

        square_left = max(0, center_x - half)
        square_top = max(0, center_y - half)
        square_right = min(WIDTH - 1, center_x + half)
        square_bottom = min(HEIGHT - 1, center_y + half)

        cv2.rectangle(
            layer,
            (square_left, square_top),
            (square_right, square_bottom),
            DRAW_COLOR,
            BRUSH_SIZE
        )

    elif shape_name == "Rectangle":
        cv2.rectangle(layer, (left, top), (right, bottom), DRAW_COLOR, BRUSH_SIZE)

    elif shape_name == "Triangle":
        points = np.array([
            [(left + right) // 2, top],
            [left, bottom],
            [right, bottom]
        ], dtype=np.int32)

        cv2.polylines(layer, [points], True, DRAW_COLOR, BRUSH_SIZE)

    else:
        return False

    return True


# ============================================
# 10. MOVE SELECTED OBJECT
# ============================================
def move_object(index, dx, dy):
    if index is None:
        return

    layer = objects[index]["layer"]

    # Find the complete drawing boundary.
    gray = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero(gray)

    if points is None:
        return

    x, y, w, h = cv2.boundingRect(points)

    # Prevent the drawing from moving outside the canvas.
    dx = int(np.clip(dx, -x, WIDTH - (x + w)))
    dy = int(np.clip(dy, -y, HEIGHT - (y + h)))

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    objects[index]["layer"] = cv2.warpAffine(
        layer,
        matrix,
        (WIDTH, HEIGHT)
    )

# ============================================
# 11. RESIZE SELECTED OBJECT
# ============================================

def resize_object(index, source_layer, scale):
    if index is None:
        return source_layer

    contour = get_object_contour(source_layer)

    if contour is None:
        return source_layer

    x, y, w, h = cv2.boundingRect(contour)

    center_x = x + w // 2
    center_y = y + h // 2

    matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)

    return cv2.warpAffine(source_layer, matrix, (WIDTH, HEIGHT))


# ============================================
# 12. FILL SELECTED OBJECT
# ============================================

def fill_object(index):
    if index is None:
        return False

    if not (0 <= index < len(objects)):
        return False

    layer = objects[index]["layer"]
    contour = get_object_contour(layer)

    if contour is None:
        return False

    area = cv2.contourArea(contour)

    if area < 500:
        return False

    cv2.drawContours(layer, [contour], -1, selected_fill_color, cv2.FILLED)
    cv2.drawContours(layer, [contour], -1, DRAW_COLOR, BRUSH_SIZE)

    objects[index]["layer"] = layer

    return True


# ============================================
# 13. CREATE DISPLAY CANVAS
# ============================================

def create_canvas():
    display = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for obj in objects:
        display = cv2.add(display, obj["layer"])

    display = cv2.add(display, current_layer)

    return display


def create_saved_image():
    """Create an export containing only the drawing on a black background."""
    saved_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    for obj in objects:
        saved_image = cv2.add(saved_image, obj["layer"])

    saved_image = cv2.add(saved_image, current_layer)
    return saved_image


# ============================================
# 14. INITIALIZE MEDIAPIPE
# ============================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================
# 15. START WEBCAM
# ============================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise SystemExit("Webcam not found!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

print("Webcam started!")
print("Virtual Whiteboard is ready!")


# ============================================
# ON-SCREEN SELECT / ERASE / RESIZE BUTTONS
# ============================================

def handle_mouse_click(event, mouse_x, mouse_y, flags, param):
    global interaction_mode, resize_mode
    global previous_x, previous_y
    global erase_action_active, drag_previous
    global resize_source, resize_start_distance, resize_object_index
    global selected_fill_name, selected_fill_color

    if event == cv2.EVENT_LBUTTONDOWN:

        # Select a fill colour by clicking its swatch in the top row.
        for color_index, color_name in enumerate(FILL_COLORS):
            px1 = PALETTE_X + color_index * PALETTE_CELL_W
            py1 = PALETTE_Y
            px2 = px1 + PALETTE_SWATCH_W
            py2 = py1 + PALETTE_SWATCH_H

            if px1 <= mouse_x <= px2 and py1 <= mouse_y <= py2:
                selected_fill_name = color_name
                selected_fill_color = FILL_COLORS[color_name]
                print(f"Selected fill colour: {selected_fill_name}")
                return

        x1, y1, x2, y2 = MODE_BUTTON

        if x1 <= mouse_x <= x2 and y1 <= mouse_y <= y2:
            interaction_mode = (
                "ERASE" if interaction_mode == "SELECT" else "SELECT"
            )

            resize_mode = False
            gesture_prediction_history.clear()

            previous_x = previous_y = None
            erase_action_active = False
            drag_previous = None

            resize_source = None
            resize_start_distance = None
            resize_object_index = None

            print(f"Interaction mode: {interaction_mode}")

        rx1, ry1, rx2, ry2 = RESIZE_BUTTON

        if rx1 <= mouse_x <= rx2 and ry1 <= mouse_y <= ry2:
            resize_mode = not resize_mode
            gesture_prediction_history.clear()

            previous_x = previous_y = None
            drag_previous = None

            resize_source = None
            resize_start_distance = None
            resize_object_index = None

            print(f"Resize mode: {'ON' if resize_mode else 'OFF'}")


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, handle_mouse_click)


# ============================================
# 16. MAIN PROGRAM
# ============================================

try:
    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "No Gesture"
        confidence = 0.0

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks
            landmarks = []

            for point in hand.landmark:
                landmarks.extend([point.x, point.y, point.z])

            features = np.array(landmarks, dtype=np.float32)
            features = (features - gesture_mean) / gesture_scale
            features = features.reshape(1, 21, 3)

            # Gesture CNN prediction
            prediction = gesture_model.predict(features, verbose=0)

            predicted_index = int(np.argmax(prediction[0]))
            raw_confidence = float(prediction[0][predicted_index] * 100)
            raw_gesture = str(gesture_classes[predicted_index])

            gesture_prediction_history.append((raw_gesture, raw_confidence))

            vote_counts = {}

            for saved_gesture, saved_confidence in gesture_prediction_history:
                if saved_confidence >= CONFIDENCE_THRESHOLD:
                    vote_counts[saved_gesture] = (
                        vote_counts.get(saved_gesture, 0) + 1
                    )

            gesture = "No Gesture"
            confidence = 0.0

            if vote_counts:
                most_common_gesture = max(vote_counts, key=vote_counts.get)

                if vote_counts[most_common_gesture] >= 3:
                    gesture = most_common_gesture

                    matching = [
                        c for g, c in gesture_prediction_history
                        if g == gesture and c >= CONFIDENCE_THRESHOLD
                    ]

                    confidence = float(np.mean(matching))

            # RESIZE BUTTON MODE:
            # When active, bypass CNN gesture labels and use pinch resizing.
            # Other gesture actions are paused until Resize Mode is turned off.
            if resize_mode:
                # Button ON: use pinch resizing and pause other CNN actions.
                gesture = "Resize"
                confidence = 100.0
            elif gesture == "Resize":
                # Button OFF: completely ignore the CNN Resize class.
                # Resizing is available only when the RESIZE MODE button is ON.
                gesture = "No Gesture"
                confidence = 0.0

            # ====================================
            # STOP DRAWING WITH CLOSED FIST
            # ====================================

            # Drag is the trained label for a closed fist. If a stroke is
            # still being drawn, use that fist to stop and save the stroke.
            if not resize_mode and gesture == "Drag" and np.any(current_layer):
                drawing_stop_latched = True
                drawing_stop_release_count = 0
                last_message = "Drawing stopped - release fist to draw again"

            if drawing_stop_latched:
                if gesture in ("Draw", "Drag"):
                    # Ignore temporary CNN prediction flicker while the fist
                    # is held or while the hand is transitioning away from it.
                    gesture = "StopDrawing"
                    confidence = 100.0
                    drawing_stop_release_count = 0
                else:
                    drawing_stop_release_count += 1

                    # Require a short stable release before allowing a new stroke.
                    if drawing_stop_release_count >= DRAWING_STOP_RELEASE_FRAMES:
                        drawing_stop_latched = False
                        drawing_stop_release_count = 0

            # Index finger position
            index_tip = hand.landmark[8]

            x = int(index_tip.x * WIDTH)
            y = int(index_tip.y * HEIGHT)

            x = int(np.clip(x, 0, WIDTH - 1))
            y = int(np.clip(y, 0, HEIGHT - 1))

            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

            # Commit drawing when drawing ends.
            if gesture != "Draw" and np.any(current_layer):
                commit_current_layer()
                previous_x = None
                previous_y = None

            # ====================================
            # DRAW
            # ====================================

            if gesture == "Draw" and confidence >= CONFIDENCE_THRESHOLD:

                if previous_x is not None:

                    if not draw_action_active:
                        save_history_state()
                        draw_action_active = True

                    cv2.line(
                        current_layer,
                        (previous_x, previous_y),
                        (x, y),
                        DRAW_COLOR,
                        BRUSH_SIZE
                    )

                previous_x = x
                previous_y = y

                smartshape_start = None
                smartshape_anchor = None
                smartshape_triggered = False

            # ====================================
            # ERASE
            # ====================================

            elif (
                gesture in ("Erase", "Select")
                and interaction_mode == "ERASE"
                and confidence >= CONFIDENCE_THRESHOLD
            ):

                if not erase_action_active:
                    save_history_state()
                    erase_action_active = True

                cv2.circle(
                    current_layer, (x, y), ERASER_SIZE, (0, 0, 0), -1
                )

                for obj in objects:
                    cv2.circle(
                        obj["layer"], (x, y), ERASER_SIZE, (0, 0, 0), -1
                    )

                previous_x = None
                previous_y = None

            # ====================================
            # SMARTSHAPE
            # ====================================

            elif (
                gesture == "SmartShape"
                and confidence >= CONFIDENCE_THRESHOLD
            ):

                previous_x = None
                previous_y = None

                now = time.time()
                smartshape_last_seen = now

                if smartshape_start is None:
                    smartshape_start = now
                    smartshape_anchor = (x, y)
                    smartshape_triggered = False

                elapsed = now - smartshape_start

                cv2.putText(
                    frame,
                    f"Hold: {elapsed:.1f}s",
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                if elapsed >= SMARTSHAPE_HOLD_TIME and not smartshape_triggered:
                    smartshape_triggered = True

                    # Pinky-only gesture folds the index finger.
                    # Therefore, correct the previously selected object.
                    object_index = selected_object

                    if (
                        object_index is not None
                        and not (0 <= object_index < len(objects))
                    ):
                        object_index = None

                    if object_index is not None:

                        detected, shape_confidence = recognize_shape(
                            objects[object_index]["layer"]
                        )

                        if (
                            detected is not None
                            and shape_confidence >= SHAPE_CONFIDENCE_THRESHOLD
                        ):

                            save_history_state()

                            corrected = correct_shape(
                                objects[object_index]["layer"],
                                detected
                            )

                            if corrected:
                                last_detected_shape = detected
                                last_shape_confidence = shape_confidence
                                last_message = f"Corrected to: {detected}"
                            else:
                                last_message = (
                                    f"Recognized: {detected}, correction failed"
                                )

                            print(
                                detected,
                                "Confidence:",
                                round(shape_confidence, 2),
                                "%"
                            )

                        else:
                            last_message = (
                                "Shape not recognized - draw a clearer closed shape"
                            )

                    else:
                        last_message = "Select a shape first"

            # ====================================
            # SELECT
            # ====================================

            elif (
                gesture in ("Select", "Erase")
                and interaction_mode == "SELECT"
                and confidence >= CONFIDENCE_THRESHOLD
            ):

                previous_x = None
                previous_y = None

                candidate_object = find_object_at(x, y)

                if candidate_object is not None:
                    selected_object = candidate_object
                    last_message = "Object selected"

                elif (
                    selected_object is not None
                    and selected_object < len(objects)
                ):
                    last_message = "Selection locked"

                else:
                    selected_object = None
                    last_message = "Point at a shape to select"

                drag_previous = None

                resize_source = None
                resize_start_distance = None
                resize_object_index = None

            # ====================================
            # DRAG
            # ====================================

            elif (
                gesture == "Drag"
                and confidence >= CONFIDENCE_THRESHOLD
            ):

                previous_x = None
                previous_y = None

                if selected_object is not None:

                    if drag_previous is None:
                        drag_previous = (x, y)

                    else:
                        dx = x - drag_previous[0]
                        dy = y - drag_previous[1]

                        if not drag_action_active:
                            save_history_state()
                            drag_action_active = True

                        move_object(selected_object, dx, dy)

                        drag_previous = (x, y)
                        last_message = "Moving selected shape"

            # ====================================
            # RESIZE
            # ====================================

            elif (
                gesture == "Resize"
                and confidence >= CONFIDENCE_THRESHOLD
            ):

                previous_x = None
                previous_y = None
                drag_previous = None

                thumb_tip = hand.landmark[4]
                index_tip = hand.landmark[8]

                pinch_distance = np.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2
                    + (thumb_tip.y - index_tip.y) ** 2
                )

                if selected_object is not None:

                    if 0 <= selected_object < len(objects):

                        if resize_source is None:

                            save_history_state()
                            resize_action_active = True

                            resize_source = objects[selected_object]["layer"].copy()

                            resize_start_distance = max(
                                pinch_distance, 0.001
                            )

                            resize_object_index = selected_object

                        if resize_object_index == selected_object:

                            scale = (
                                pinch_distance / resize_start_distance
                            )

                            scale = float(np.clip(scale, 0.5, 2.0))

                            objects[selected_object]["layer"] = resize_object(
                                selected_object,
                                resize_source,
                                scale
                            )

                            last_message = f"Resize: {scale:.2f}x"

                    else:
                        last_message = "Selected object is not available"

                else:
                    last_message = "Select an object before resizing"

            # ====================================
            # PAINT FILL
            # ====================================

            elif (
                gesture == "PaintFill"
                and confidence >= CONFIDENCE_THRESHOLD
            ):

                previous_x = None
                previous_y = None

                if not fill_triggered:

                    fill_triggered = True

                    if selected_object is not None:
                        save_history_state()

                    if fill_object(selected_object):
                        last_message = "Shape filled!"
                        print("Selected shape filled!")
                    else:
                        last_message = "Select a closed shape first"

            # ====================================
            # RESET GESTURE-SPECIFIC STATES
            # ====================================

            if gesture != "Draw":
                draw_action_active = False

            if not (
                gesture in ("Erase", "Select")
                and interaction_mode == "ERASE"
            ):
                erase_action_active = False

            if gesture != "Drag":
                drag_previous = None
                drag_action_active = False

            # Keep resize data while the resize button is active.
            # Clear it only after Resize Mode is switched off.
            if gesture != "Resize":
                resize_action_active = False
                resize_source = None
                resize_start_distance = None
                resize_object_index = None

            # Keep SmartShape hold timer through brief missed predictions.
            if (
                smartshape_start is not None
                and smartshape_last_seen is not None
                and time.time() - smartshape_last_seen > SMARTSHAPE_GRACE_TIME
            ):
                smartshape_start = None
                smartshape_anchor = None
                smartshape_triggered = False
                smartshape_last_seen = None

            if gesture != "PaintFill":
                fill_triggered = False

        else:
            gesture_prediction_history.clear()
            previous_x = None
            previous_y = None

            # No hand detected: clear the stop latch so the next gesture can start fresh.
            drawing_stop_latched = False
            drawing_stop_release_count = 0

            draw_action_active = False
            erase_action_active = False
            drag_action_active = False

            if (
                smartshape_start is not None
                and smartshape_last_seen is not None
                and time.time() - smartshape_last_seen > SMARTSHAPE_GRACE_TIME
            ):
                smartshape_start = None
                smartshape_anchor = None
                smartshape_triggered = False
                smartshape_last_seen = None

            drag_previous = None

            # Do not discard the resize snapshot when the hand is briefly
            # not detected. Keep it while Resize Mode remains active.
            if not resize_mode:
                resize_action_active = False
                resize_source = None
                resize_start_distance = None
                resize_object_index = None

            fill_triggered = False


        # ====================================
        # DISPLAY SELECTED OBJECT
        # ====================================

        if selected_object is not None:
            if selected_object < len(objects):
                contour = get_object_contour(
                    objects[selected_object]["layer"]
                )

                if contour is not None:
                    cv2.drawContours(
                        frame, [contour], -1, SELECT_COLOR, 3
                    )

        # ====================================
        # DISPLAY INFORMATION
        # ====================================


        cv2.putText(
            frame,
            f"Active Mode: {interaction_mode}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255) if interaction_mode == "ERASE" else (255, 200, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            last_message,
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if last_detected_shape is not None:
            cv2.putText(
                frame,
                f"Shape: {last_detected_shape} ({last_shape_confidence:.1f}%)",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # ====================================
        # COMBINE WEBCAM AND WHITEBOARD
        # ====================================

        canvas = create_canvas()

        combined = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

        # ====================================
        # DISPLAY ALL 12 COLOURS IN ONE TOP ROW
        # ====================================

        for color_index, color_name in enumerate(FILL_COLORS):
            px = PALETTE_X + color_index * PALETTE_CELL_W
            top_left = (px, PALETTE_Y)
            bottom_right = (
                px + PALETTE_SWATCH_W,
                PALETTE_Y + PALETTE_SWATCH_H,
            )
            swatch_color = FILL_COLORS[color_name]

            # Compact rectangular colour swatch; no names or text labels.
            cv2.rectangle(combined, top_left, bottom_right, swatch_color, -1)
            cv2.rectangle(
                combined, top_left, bottom_right, (230, 230, 230), 1
            )

            # A thicker white border indicates the selected fill colour.
            if color_name == selected_fill_name:
                cv2.rectangle(
                    combined,
                    (px - 2, PALETTE_Y - 2),
                    (px + PALETTE_SWATCH_W + 2,
                     PALETTE_Y + PALETTE_SWATCH_H + 2),
                    (255, 255, 255),
                    2,
                )

        # Rectangular preview changes when a colour is clicked.
        preview_x1, preview_y1, preview_x2, preview_y2 = PALETTE_PREVIEW_BOX
        cv2.rectangle(
            combined,
            (preview_x1, preview_y1),
            (preview_x2, preview_y2),
            selected_fill_color,
            -1,
        )
        cv2.rectangle(
            combined,
            (preview_x1, preview_y1),
            (preview_x2, preview_y2),
            (255, 255, 255),
            2,
        )

        # Compact live hand/gesture status on the same top row as the palette.
        # The existing gesture prediction and hand tracking logic is unchanged.
        status_text = "Show your hand" if gesture == "No Gesture" else f"Gesture: {gesture}"
        cv2.putText(
            combined,
            status_text,
            (590, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0) if gesture != "No Gesture" else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # Select / Erase toggle button
        bx1, by1, bx2, by2 = MODE_BUTTON

        button_color = (
            (0, 0, 220) if interaction_mode == "ERASE"
            else (180, 80, 0)
        )

        cv2.rectangle(combined, (bx1, by1), (bx2, by2), button_color, -1)
        cv2.rectangle(
            combined, (bx1, by1), (bx2, by2), (255, 255, 255), 2
        )

        button_text = (
            "ERASE MODE (CLICK)" if interaction_mode == "ERASE"
            else "SELECT MODE (CLICK)"
        )

        cv2.putText(
            combined,
            button_text,
            (bx1 + 7, by1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1
        )

        # Separate Resize Mode button
        rx1, ry1, rx2, ry2 = RESIZE_BUTTON

        resize_button_color = (
            (0, 150, 0) if resize_mode else (100, 100, 100)
        )

        cv2.rectangle(
            combined,
            (rx1, ry1),
            (rx2, ry2),
            resize_button_color,
            -1
        )

        cv2.rectangle(
            combined,
            (rx1, ry1),
            (rx2, ry2),
            (255, 255, 255),
            2
        )

        resize_button_text = (
            "RESIZE MODE: ON" if resize_mode
            else "RESIZE MODE: OFF"
        )

        cv2.putText(
            combined,
            resize_button_text,
            (rx1 + 7, ry1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        # Keyboard shortcuts: use a dedicated high-contrast footer so the
        # controls remain readable and are not clipped at the bottom edge.
        shortcut_y1 = HEIGHT - 48
        shortcut_y2 = HEIGHT - 8
        cv2.rectangle(
            combined,
            (8, shortcut_y1),
            (WIDTH - 8, shortcut_y2),
            (35, 35, 35),
            -1,
        )
        cv2.putText(
            combined,
            "Q: QUIT   C: CLEAR   S: SAVE   Z: UNDO   Y: REDO   |   PINKY 1s: SHAPE CORRECTION",
            (18, HEIGHT - 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, combined)

        # ====================================
        # KEYBOARD CONTROLS
        # ====================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key in (ord("z"), 26):
            undo_action()
            last_message = "Undo"

        elif key in (ord("y"), 25):
            redo_action()
            last_message = "Redo"

        elif key == ord("c"):
            save_history_state()

            objects.clear()

            current_layer = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            selected_object = None

            last_detected_shape = None
            last_shape_confidence = 0.0

            resize_source = None
            resize_start_distance = None
            resize_object_index = None

            last_message = "Canvas cleared"

            print("Canvas cleared!")

        elif key == ord("s"):
            commit_current_layer()

            # Create drawing-only image with a black background.
            output = create_saved_image()

            # Save using a unique filename.
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                BASE_DIR,
                f"whiteboard_output_{timestamp}_{time.time_ns()}.png"
            )

            if cv2.imwrite(save_path, output):
                print("Whiteboard saved successfully:", save_path)
            else:
                print("Error: could not save whiteboard image.")

# ============================================
# 17. RELEASE RESOURCES
# ============================================

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("Virtual Whiteboard closed successfully!")
