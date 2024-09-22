import os
import shutil
from ultralytics import YOLO
import cv2
from tqdm import tqdm

# Directory paths
train_dir = './footage/training/images'
val_dir = './footage/validation/images'
train_labels_dir = './footage/labels/train'
val_labels_dir = './footage/labels/val'
split_dir = './footage/split'

# Split directories
split_train_images_dir = os.path.join(split_dir, 'train/images')
split_train_labels_dir = os.path.join(split_dir, 'train/labels')
split_val_images_dir = os.path.join(split_dir, 'val/images')
split_val_labels_dir = os.path.join(split_dir, 'val/labels')

# Create split directories
os.makedirs(split_train_images_dir, exist_ok=True)
os.makedirs(split_train_labels_dir, exist_ok=True)
os.makedirs(split_val_images_dir, exist_ok=True)
os.makedirs(split_val_labels_dir, exist_ok=True)

def copy_files(images_dir, labels_dir, split_images_dir, split_labels_dir):
    for img in os.listdir(images_dir):
        label_file = os.path.splitext(img)[0] + '.txt'
        img_path = os.path.join(images_dir, img)
        label_path = os.path.join(labels_dir, label_file)
        if os.path.exists(label_path):
            shutil.copyfile(img_path, os.path.join(split_images_dir, img))
            shutil.copyfile(label_path, os.path.join(split_labels_dir, label_file))

def create_data_yaml():
    content = f"""
train: {os.path.abspath(split_train_images_dir)}
val: {os.path.abspath(split_val_images_dir)}
names:
  0: bee
nc: 1
nk: 2
kpt_shape: [2, 2]
"""
    data_yaml_path = os.path.join(split_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(content)
    return data_yaml_path

def train_model(data_yaml_path):
    model = YOLO("runs/pose/train/weights/best.pt")
    model.train(data=data_yaml_path, epochs=100, imgsz=640, batch=32)

def validate_model():
    best_model = YOLO("runs/pose/train/weights/best.pt")
    results = best_model.val(save=False, plots=True, save_hybrid=True, conf=False, show_labels=False, show_boxes=False, verbose=False, show=True)

def save_predictions():
    output_dir = 'output_new'
    os.makedirs(output_dir, exist_ok=True)
    best_model = YOLO("runs/pose/train/weights/best.pt")

    for img_file in tqdm(os.listdir(split_val_images_dir)):
        img_path = os.path.join(split_val_images_dir, img_file)
        original_img = cv2.imread(img_path)
        results = best_model.predict(img_path, save=False, save_txt=False, show_labels=False, show_conf=False)
        pred_img = results[0].plot(labels=False, show=False)
        side_by_side = cv2.hconcat([original_img, pred_img])
        output_path = os.path.join(output_dir, f'side_by_side_{img_file}')
        cv2.imwrite(output_path, side_by_side)

def main():
    print("Copying files to split directories...")
    copy_files(train_dir, train_labels_dir, split_train_images_dir, split_train_labels_dir)
    copy_files(val_dir, val_labels_dir, split_val_images_dir, split_val_labels_dir)

    print("Creating data.yaml file...")
    data_yaml_path = create_data_yaml()

    print("Training model...")
    train_model(data_yaml_path)

    print("Validating model...")
    validate_model()

    print("Saving predictions...")
    save_predictions()

    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()