import os
import cv2
import random
from ultralytics import YOLO
from tqdm import tqdm

# Directory paths
video_dir = './footage/videos'
train_dir = './footage/training/images'
val_dir = './footage/validation/images'
train_labels_dir = './footage/labels/train'
val_labels_dir = './footage/labels/val'

# Create necessary directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Load pre-trained model for labeling
labeling_model = YOLO("runs/pose/train/weights/best.pt")

def extract_frames(video_path, max_frames=40):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0
    frames = []

    while cap.isOpened() and saved_frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 2 == 0:
            frame_filename = f'frame_{os.path.basename(video_path).split(".")[0]}_{saved_frame_count:06d}.jpg'
            frames.append((frame, frame_filename))
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    return frames

def label_frame(frame, filename, model, confidence_threshold=0.5):
    results = model.predict(source=frame, save=False)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None, None

    result = results[0]
    filtered_results = [box for box in result.boxes if box.conf > confidence_threshold]

    if len(filtered_results) == 0:
        return None, None

    label_content = []
    for box in filtered_results:
        cls = int(box.cls)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        img_width, img_height = frame.shape[1], frame.shape[0]
        x_norm, y_norm = x / img_width, y / img_height
        w_norm, h_norm = w / img_width, h / img_height
        px1_norm, py1_norm = x1 / img_width, y1 / img_height
        px2_norm, py2_norm = x2 / img_width, y2 / img_height
        
        label_content.append(f"{cls} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f} {px1_norm:.6f} {py1_norm:.6f} {px2_norm:.6f} {py2_norm:.6f}")

    return frame, '\n'.join(label_content)

def process_videos():
    video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    all_frames = []

    for video_path in video_files:
        frames = extract_frames(video_path)
        all_frames.extend(frames)

    random.shuffle(all_frames)
    split_index = int(len(all_frames) * 0.8)
    train_frames, val_frames = all_frames[:split_index], all_frames[split_index:]

    for frame_set, img_dir, label_dir in [(train_frames, train_dir, train_labels_dir), (val_frames, val_dir, val_labels_dir)]:
        for frame, filename in tqdm(frame_set, desc=f"Processing {'training' if img_dir == train_dir else 'validation'} frames"):
            labeled_frame, label_content = label_frame(frame, filename, labeling_model)
            if labeled_frame is not None and label_content is not None:
                cv2.imwrite(os.path.join(img_dir, filename), labeled_frame)
                with open(os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt'), 'w') as f:
                    f.write(label_content)

def main():
    print("Processing videos, extracting frames, and labeling...")
    process_videos()
    print("Video processing and labeling completed successfully!")

if __name__ == "__main__":
    main()