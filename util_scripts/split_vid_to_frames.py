# Script for splitting video feed into individual frames
import cv2
import os

# Define the path to your video file
video_path = './footage/five.mp4'

# Directory to save the extracted frames
output_dir = './footage/training'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_frame_count = 0
max_frames_to_save = 40


while cap.isOpened() and saved_frame_count < max_frames_to_save:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every other frame
    if frame_count % 2 == 0:
        frame_filename = os.path.join(output_dir, f'frame_vid5_{saved_frame_count:06d}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f'Total frames saved: {saved_frame_count}')
