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
import pandas as pd

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
    Uses CSV metadata to prevent data leakage between train and val/test sets.
    """
    print("Preparing YOLOv8 dataset structure...")
    print("Using CSV metadata to prevent data leakage...")
    print("=" * 60)
    
    # Load CSV metadata files
    augmented_metadata_path = 'data/augmented_dataset_metadata.csv'
    original_metadata_path = 'data/dataset_metadata.csv'
    
    aug_metadata_df = pd.DataFrame()
    orig_metadata_df = pd.DataFrame()
    
    if os.path.exists(augmented_metadata_path):
        aug_metadata_df = pd.read_csv(augmented_metadata_path)
        print(f"Loaded augmented metadata: {len(aug_metadata_df)} rows")
    else:
        print(f"Warning: {augmented_metadata_path} not found.")
    
    if os.path.exists(original_metadata_path):
        orig_metadata_df = pd.read_csv(original_metadata_path)
        print(f"Loaded original metadata: {len(orig_metadata_df)} rows")
    else:
        print(f"Warning: {original_metadata_path} not found.")
    
    # Filter train data to exclude test/val images
    train_df_filtered = pd.DataFrame()
    if len(aug_metadata_df) > 0 and len(orig_metadata_df) > 0:
        # Get test and val original filenames
        test_originals = set(orig_metadata_df[orig_metadata_df['split'] == 'test']['filename'].unique())
        val_originals = set(orig_metadata_df[orig_metadata_df['split'] == 'val']['filename'].unique())
        excluded_originals = test_originals.union(val_originals)
        
        # Filter train data: only keep images whose original_filename is NOT in test/val
        train_df = aug_metadata_df[aug_metadata_df['split'] == 'train'].copy()
        train_df_filtered = train_df[~train_df['original_filename'].isin(excluded_originals)]
        
        print(f"Filtered training data: {len(train_df_filtered)} rows (from {len(train_df)} total train rows)")
        print(f"Excluded {len(train_df) - len(train_df_filtered)} rows that overlap with test/val sets")
    
    # Filter val data to exclude train images
    val_df_filtered = pd.DataFrame()
    if len(orig_metadata_df) > 0 and len(aug_metadata_df) > 0:
        # Get train original filenames
        train_originals = set(aug_metadata_df[aug_metadata_df['split'] == 'train']['original_filename'].unique())
        
        # Filter val data: only keep images whose filename is NOT in train
        val_df = orig_metadata_df[orig_metadata_df['split'] == 'val'].copy()
        val_df_filtered = val_df[~val_df['filename'].isin(train_originals)]
        
        print(f"Filtered validation data: {len(val_df_filtered)} rows (from {len(val_df)} total val rows)")
        print(f"Excluded {len(val_df) - len(val_df_filtered)} rows that overlap with train set")
    
    # Create YOLOv8 directory structure
    yolo_images_train = os.path.join(YOLO_DIR, 'images', 'train')
    yolo_images_val = os.path.join(YOLO_DIR, 'images', 'val')
    yolo_labels_train = os.path.join(YOLO_DIR, 'labels', 'train')
    yolo_labels_val = os.path.join(YOLO_DIR, 'labels', 'val')
    
    for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process train set
    print(f"\nProcessing train set...")
    train_image_count = 0
    train_label_count = 0
    
    if len(train_df_filtered) > 0:
        # Use CSV metadata to get filtered images
        for _, row in tqdm(train_df_filtered.iterrows(), total=len(train_df_filtered), desc="  Processing train images"):
            # Use full_path if available, otherwise construct from SOURCE_DIR and image_path
            if pd.notna(row.get('full_path')):
                source_img_path = row['full_path']
            else:
                source_img_path = os.path.join(SOURCE_DIR, row['image_path'])
            
            # Normalize path separators
            source_img_path = source_img_path.replace('\\', '/')
            
            if os.path.exists(source_img_path):
                img_file = os.path.basename(source_img_path)
                dest_img_path = os.path.join(YOLO_DIR, 'images', 'train', img_file)
                shutil.copy2(source_img_path, dest_img_path)
                train_image_count += 1
                
                # Create label file if this is a tumor class
                yolo_class_id = CLASS_TO_YOLO.get(row['class'])
                if yolo_class_id is not None:
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(YOLO_DIR, 'labels', 'train', label_file)
                    
                    if create_yolo_label(source_img_path, yolo_class_id, label_path):
                        train_label_count += 1
    else:
        # Fallback to directory loading if CSV not available
        print("Warning: CSV metadata not available. Using directory loading (may have data leakage).")
        source_split_dir = os.path.join(SOURCE_DIR, 'train')
        if os.path.exists(source_split_dir):
            for class_name in CLASSES:
                class_dir = os.path.join(source_split_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in tqdm(image_files, desc=f"  {class_name}"):
                    source_img_path = os.path.join(class_dir, img_file)
                    dest_img_path = os.path.join(YOLO_DIR, 'images', 'train', img_file)
                    shutil.copy2(source_img_path, dest_img_path)
                    train_image_count += 1
                    
                    yolo_class_id = CLASS_TO_YOLO[class_name]
                    if yolo_class_id is not None:
                        label_file = os.path.splitext(img_file)[0] + '.txt'
                        label_path = os.path.join(YOLO_DIR, 'labels', 'train', label_file)
                        if create_yolo_label(source_img_path, yolo_class_id, label_path):
                            train_label_count += 1
    
    print(f"  Train set: {train_image_count} images, {train_label_count} labels")
    
    # Process validation set
    print(f"\nProcessing val set...")
    val_image_count = 0
    val_label_count = 0
    
    if len(val_df_filtered) > 0:
        # Use CSV metadata to get filtered images
        for _, row in tqdm(val_df_filtered.iterrows(), total=len(val_df_filtered), desc="  Processing val images"):
            # Use full_path if available, otherwise construct from SOURCE_DIR and image_path
            if pd.notna(row.get('full_path')):
                source_img_path = row['full_path']
            else:
                source_img_path = os.path.join(SOURCE_DIR, row['image_path'])
            
            # Normalize path separators
            source_img_path = source_img_path.replace('\\', '/')
            
            if os.path.exists(source_img_path):
                img_file = os.path.basename(source_img_path)
                dest_img_path = os.path.join(YOLO_DIR, 'images', 'val', img_file)
                shutil.copy2(source_img_path, dest_img_path)
                val_image_count += 1
                
                # Create label file if this is a tumor class
                yolo_class_id = CLASS_TO_YOLO.get(row['class'])
                if yolo_class_id is not None:
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    label_path = os.path.join(YOLO_DIR, 'labels', 'val', label_file)
                    
                    if create_yolo_label(source_img_path, yolo_class_id, label_path):
                        val_label_count += 1
    else:
        # Fallback to directory loading if CSV not available
        print("Warning: CSV metadata not available. Using directory loading (may have data leakage).")
        source_split_dir = os.path.join(SOURCE_DIR, 'val')
        if os.path.exists(source_split_dir):
            for class_name in CLASSES:
                class_dir = os.path.join(source_split_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                image_files = [f for f in os.listdir(class_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                for img_file in tqdm(image_files, desc=f"  {class_name}"):
                    source_img_path = os.path.join(class_dir, img_file)
                    dest_img_path = os.path.join(YOLO_DIR, 'images', 'val', img_file)
                    shutil.copy2(source_img_path, dest_img_path)
                    val_image_count += 1
                    
                    yolo_class_id = CLASS_TO_YOLO[class_name]
                    if yolo_class_id is not None:
                        label_file = os.path.splitext(img_file)[0] + '.txt'
                        label_path = os.path.join(YOLO_DIR, 'labels', 'val', label_file)
                        if create_yolo_label(source_img_path, yolo_class_id, label_path):
                            val_label_count += 1
    
    print(f"  Val set: {val_image_count} images, {val_label_count} labels")
    
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