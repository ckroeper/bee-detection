import cv2
import os
import random
import shutil

# Directory containing the video files
video_dir = './footage/videos'
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

# Directory to save the extracted frames
output_dir = './footage/frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Directories for training and validation sets
train_dir = './footage/training/images'
val_dir = './footage/validation/images'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

max_frames_to_save = 40

for video_path in video_files:
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_frame_count = 0
    
    while cap.isOpened() and saved_frame_count < max_frames_to_save:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every other frame
        if frame_count % 2 == 0:
            frame_filename = os.path.join(output_dir, f'frame_{os.path.basename(video_path).split(".")[0]}_{saved_frame_count:06d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()

cv2.destroyAllWindows()

print(f'Total frames saved for video {video_path}: {saved_frame_count}')

# Split frames into training and validation sets
frames = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
random.shuffle(frames)

split_ratio = 0.8
split_index = int(len(frames) * split_ratio)
train_frames = frames[:split_index]
val_frames = frames[split_index:]

# Move frames to respective directories
for frame in train_frames:
    shutil.move(os.path.join(output_dir, frame), os.path.join(train_dir, frame))

for frame in val_frames:
    shutil.move(os.path.join(output_dir, frame), os.path.join(val_dir, frame))

print(f'Total training frames: {len(train_frames)}')
print(f'Total validation frames: {len(val_frames)}')
