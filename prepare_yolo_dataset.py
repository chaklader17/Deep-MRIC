"""
YOLOv8 Dataset Preparation Script

This script prepares the brain tumor classification dataset for YOLOv8 object detection.
Since the original dataset is for classification (not detection with bounding boxes),
this script creates a YOLOv8-compatible structure.

Note: For proper YOLOv8 training, you would need bounding box annotations.
This script creates a basic structure. You may need to manually annotate images
or use an annotation tool like LabelImg to create proper bounding box labels.

Expected Input Structure:
    data/vgg16_classification/
    ├── train/
    │   ├── NO_TUMOR/
    │   ├── GLIOMA/
    │   ├── MENINGIOMA/
    │   └── PITUITARY/
    ├── val/
    └── test/

Output Structure:
    data/yolov8/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

# Configuration
SOURCE_DIR = 'data/vgg16_classification'
YOLO_DIR = 'data/yolov8'
CLASSES = ['NO_TUMOR', 'GLIOMA', 'MENINGIOMA', 'PITUITARY']

# YOLOv8 class mapping
# For detection, we'll treat all tumor types as class 0 (tumor)
# NO_TUMOR images won't have labels (no tumor to detect)
CLASS_TO_YOLO = {
    'NO_TUMOR': None,  # No label file (no tumor present)
    'GLIOMA': 0,       # Class 0: tumor
    'MENINGIOMA': 0,   # Class 0: tumor
    'PITUITARY': 0     # Class 0: tumor
}


def create_yolo_label(image_path, class_id, output_label_path):
    """
    Create a YOLO format label file.
    
    Since we don't have bounding box annotations, this creates a placeholder
    that covers the entire image. For real detection, you should use proper
    bounding box coordinates from annotation tools.
    
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    
    Args:
        image_path: Path to the image file
        class_id: YOLO class ID (0 for tumor)
        output_label_path: Path to save the label file
    """
    # Read image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # Create a label that covers the entire image
    # This is a placeholder - for real detection, use actual bounding boxes
    center_x = 0.5
    center_y = 0.5
    width = 1.0
    height = 1.0
    
    # Write YOLO format label
    with open(output_label_path, 'w') as f:
        f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
    
    return True


def prepare_yolo_dataset():
    """
    Prepare YOLOv8 dataset structure from classification dataset.
    """
    print("Preparing YOLOv8 dataset structure...")
    print("=" * 60)
    
    # Create YOLOv8 directory structure
    yolo_images_train = os.path.join(YOLO_DIR, 'images', 'train')
    yolo_images_val = os.path.join(YOLO_DIR, 'images', 'val')
    yolo_labels_train = os.path.join(YOLO_DIR, 'labels', 'train')
    yolo_labels_val = os.path.join(YOLO_DIR, 'labels', 'val')
    
    for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process train and validation sets
    for split in ['train', 'val']:
        print(f"\nProcessing {split} set...")
        source_split_dir = os.path.join(SOURCE_DIR, split)
        
        if not os.path.exists(source_split_dir):
            print(f"Warning: {source_split_dir} not found. Skipping.")
            continue
        
        image_count = 0
        label_count = 0
        
        for class_name in CLASSES:
            class_dir = os.path.join(source_split_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files, desc=f"  {class_name}"):
                source_img_path = os.path.join(class_dir, img_file)
                
                # Copy image to YOLOv8 images directory
                dest_img_path = os.path.join(YOLO_DIR, 'images', split, img_file)
                shutil.copy2(source_img_path, dest_img_path)
                image_count += 1
                
                # Create label file if this is a tumor class
                yolo_class_id = CLASS_TO_YOLO[class_name]
                if yolo_class_id is not None:
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(YOLO_DIR, 'labels', split, label_file)
                    
                    if create_yolo_label(source_img_path, yolo_class_id, label_path):
                        label_count += 1
        
        print(f"  {split.capitalize()} set: {image_count} images, {label_count} labels")
    
    print("\n" + "=" * 60)
    print("YOLOv8 dataset preparation complete!")
    print(f"\nDataset structure created at: {YOLO_DIR}")
    print("\nDirectory structure:")
    print(f"  {YOLO_DIR}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/")
    print(f"    │   └── val/")
    print(f"    └── labels/")
    print(f"        ├── train/")
    print(f"        └── val/")
    print("\n⚠️  IMPORTANT NOTE:")
    print("The label files created are placeholders covering the entire image.")
    print("For proper YOLOv8 training, you should:")
    print("  1. Use an annotation tool (e.g., LabelImg) to create proper bounding boxes")
    print("  2. Or use a pre-annotated dataset with bounding box coordinates")
    print("  3. Update the label files with actual tumor bounding box coordinates")
    print("\nThe tumor_data.yaml configuration file is ready to use.")


if __name__ == "__main__":
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Source directory '{SOURCE_DIR}' not found.")
        print("Please run the preprocessing script first to create the classification dataset.")
        exit(1)
    
    # Run dataset preparation
    prepare_yolo_dataset()