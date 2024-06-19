import cv2
from ultralytics import YOLO
import os

# Loading trained yolo mode;
model = YOLO("runs/pose/train/weights/best.pt")

# Path to the video file
video_path = "beess.mp4"

# Opening video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Using model to detect bees in image
    results = model.predict(frame, save=False, save_txt=False, show_labels=False, show_conf=False)

    # Get the image with predictions
    pred_img = results[0].plot(labels=False, show=False)

    # Display the resulting frame
    cv2.imshow('Bee Tracking', pred_img)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
