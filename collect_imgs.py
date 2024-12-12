import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Open the camera at index 0 (you can change this if needed)
cap = cv2.VideoCapture(0)  # Try with 0 or other indices like 1, 2, etc.

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for j in range(number_of_classes):
    # Create class directories if they don't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            continue

        # Display the frame with text
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Break the loop if 'Q' is pressed
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            continue

        # Display the frame
        cv2.imshow('frame', frame)

        # Save the frame as an image file
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
