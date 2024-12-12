import os
import pickle
import mediapipe as mp
import cv2

# Mediapipe Hand detection initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = './data'

# Data and labels containers
data = []
labels = []

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    print(f"Data directory {DATA_DIR} does not exist.")
    exit()

# Iterate through the classes (directories) and images
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Check if it is a directory, skip files like .gitignore
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            if img is None:
                print(f"Image {img_path} could not be loaded.")
                continue

            # Convert the image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to detect hand landmarks
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect x, y coordinates of the landmarks
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalize the coordinates based on the minimum values
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append the processed data and label
                data.append(data_aux)
                labels.append(dir_)

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Close the Mediapipe hands instance
hands.close()

print("Data collection and saving completed successfully.")
