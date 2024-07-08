import os
import random
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO("runs/pose/train/weights/best.pt")

# Directories containing the training and validation images
train_images_dir = './footage/training/images'
val_images_dir = './footage/validation/images'

# Get the list of images
training_images = [os.path.join(train_images_dir, img) for img in os.listdir(train_images_dir) if img.endswith(('.jpg', '.png'))]
validation_images = [os.path.join(val_images_dir, img) for img in os.listdir(val_images_dir) if img.endswith(('.jpg', '.png'))]

# Directories to save pseudo-labels
pseudo_labels_train_dir = './footage/labels/train'
pseudo_labels_val_dir = './footage/labels/val'

os.makedirs(pseudo_labels_train_dir, exist_ok=True)
os.makedirs(pseudo_labels_val_dir, exist_ok=True)

def generate_pseudo_labels(image_paths, labels_dir):
    for image_path in image_paths:
        # Predict using the previously trained model
        results = model.predict(source=image_path, save=False)
        
        # Extract the first result (assuming batch size of 1)
        if len(results) == 0:
            print(f"No results found for image: {image_path}")
            continue
        
        result = results[0]
        
        # Filter results based on confidence threshold
        confidence_threshold = 0.7  # 70%
        filtered_results = [box for box in result.boxes if box.conf > confidence_threshold]
        
        if len(filtered_results) == 0:
            print(f"No high-confidence results for image: {image_path}")
            continue
        
        # Save pseudo-labels
        label_path = os.path.join(labels_dir, os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(label_path, 'w') as f:
            for box in filtered_results:
                cls = int(box.cls)  # Classification
                conf = box.conf.item()  # Confidence
                x, y, w, h = box.xywh.tolist()[0]  # Bounding box location and dimensions
                f.write(f"{cls} {x} {y} {w} {h} {conf}\n")
        
        print(f"Pseudo-labels saved for image: {image_path}")

# Generate pseudo-labels for training images
generate_pseudo_labels(training_images, pseudo_labels_train_dir)

# Generate pseudo-labels for validation images
generate_pseudo_labels(validation_images, pseudo_labels_val_dir)
