# Script for creating pseudo labels
from ultralytics import YOLO
import os

# Load your YOLOv8 model
model = YOLO("runs/pose/train/weights/best.pt")

# Directory containing the unlabeled images
unlabeled_images_dir = './footage/training'
unlabeled_images = [os.path.join(unlabeled_images_dir, img) for img in os.listdir(unlabeled_images_dir) if img.endswith(('.jpg', '.png'))]

# Directory to save pseudo-labels
pseudo_labels_dir = './footage/labels'
if not os.path.exists(pseudo_labels_dir):
    os.makedirs(pseudo_labels_dir)

# Generating pseudo-labels for each image
for image_path in unlabeled_images:
    # Predict using the previously trained model
    results = model.predict(source=image_path, save=False)
    
    # Extract the first result (assuming batch size of 1)
    if len(results) == 0:
        print(f"No results found for image: {image_path}")
        continue
    
    result = results[0]
    
    # Filter results based on confidence threshold
    confidence_threshold = 0.7 # 70%
    filtered_results = [box for box in result.boxes if box.conf > confidence_threshold]
    
    if len(filtered_results) == 0:
        print(f"No high-confidence results for image: {image_path}")
        continue
    
    # Save pseudo-labels
    label_path = os.path.join(pseudo_labels_dir, os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(label_path, 'w') as f:
        for box in filtered_results:
            cls = int(box.cls) # Classification
            conf = box.conf.item() # Confidence
            x, y, w, h = box.xywh.tolist()[0] # Bounding box location, and dimensions
            f.write(f"{cls} {x} {y} {w} {h} {conf}\n")
    
    print(f"Pseudo-labels saved for image: {image_path}")
