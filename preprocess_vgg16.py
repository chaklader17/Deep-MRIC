"""
VGG16 Data Preprocessing Script for Brain Tumor Classification

This script preprocesses brain MRI images for VGG16 classification model training.
It performs the following operations:
1. Loads raw MRI images from organized class folders
2. Crops brain regions using contour detection (removes black background)
3. Resizes images to 224x224 (VGG16 input size)
4. Performs stratified train/validation/test split (70/15/15)
5. Saves processed images in organized directory structure

Expected Input Structure:
    data/raw_dataset/
    ├── NO_TUMOR/
    ├── GLIOMA/
    ├── MENINGIOMA/
    └── PITUITARY/

Output Structure:
    data/vgg16_classification/
    ├── train/
    │   ├── NO_TUMOR/
    │   ├── GLIOMA/
    │   ├── MENINGIOMA/
    │   └── PITUITARY/
    ├── val/
    │   └── [same class folders]
    └── test/
        └── [same class folders]

This is part of the Deep-MRIC project for brain tumor detection and classification.
See README.md for more information about the full pipeline.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
# Directory containing raw MRI images organized by class
RAW_DATA_DIR = 'data/raw_dataset/'

# Output directory for processed VGG16-ready images
VGG_OUTPUT_DIR = 'data/vgg16_classification/'

# VGG16 requires 224x224 input images
VGG_IMG_SIZE = 224 

# Train/Validation/Test split ratios
SPLIT_RATIO = [0.70, 0.15, 0.15]  # Train:Val:Test

# Expected Class Folders in RAW_DATA_DIR (Update this based on your dataset)
CLASSES = ['NO_TUMOR', 'GLIOMA', 'MENINGIOMA', 'PITUITARY']
SPLIT_NAMES = ['train', 'val', 'test']


# --- Utility Function: Image Cropping ---

def crop_brain_region(img):
    """
    Identifies the brain region by contour detection and crops the image.
    This effectively removes non-brain black background padding.
    
    The function:
    1. Converts image to grayscale and applies Gaussian blur
    2. Uses thresholding to separate brain tissue from black background
    3. Applies morphological operations (erosion/dilation) to clean up
    4. Finds the largest contour (assumed to be the brain)
    5. Crops the image to the bounding box with a small buffer
    
    Args:
        img: Input BGR image (numpy array from cv2.imread)
    
    Returns:
        Cropped image (numpy array) or None if processing fails
    """
    if img is None:
        return None

    # Convert to grayscale and apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to separate the brain from the background
    # Threshold value of 45 works well for MRI images with black backgrounds
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up the thresholded image
    # Erosion removes small noise, dilation fills gaps
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours (boundaries of white regions)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: if no contour found, return original image
        print("Warning: No contours found, returning original image")
        return img 

    # Find the largest contour (assumed to be the brain)
    c = max(contours, key=cv2.contourArea)

    # Get bounding box (x, y, width, height)
    x, y, w, h = cv2.boundingRect(c)
    
    # Crop the image with a small buffer for safety (prevents edge clipping)
    buffer = 10
    cropped_img = img[
        max(0, y - buffer):y + h + buffer, 
        max(0, x - buffer):x + w + buffer
    ]

    return cropped_img


# --- Main Preprocessing Function ---

def prepare_vgg_data():
    """
    Main preprocessing pipeline for VGG16 classification.
    
    Steps:
    1. Validates input directory exists
    2. Creates output directory structure
    3. Collects all image files from class folders
    4. Performs stratified train/val/test split
    5. Processes each image (crop + resize)
    6. Saves processed images to organized folders
    
    The stratified split ensures each class is proportionally represented
    in train, validation, and test sets.
    """
    
    # 0. Setup and Data Collection
    
    # Ensure raw data exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"❌ Error: Raw data directory '{RAW_DATA_DIR}' not found.")
        print(f"Please create the directory and organize your images as follows:")
        print(f"  {RAW_DATA_DIR}")
        for class_name in CLASSES:
            print(f"    ├── {class_name}/")
        return

    # Check if any class folders exist
    class_found = False
    for class_name in CLASSES:
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        if os.path.isdir(class_path):
            class_found = True
            break
    
    if not class_found:
        print(f"❌ Error: No class folders found in '{RAW_DATA_DIR}'")
        print(f"Expected folders: {', '.join(CLASSES)}")
        return

    # Clear previous output and create the final structure
    if os.path.exists(VGG_OUTPUT_DIR):
        print(f"⚠️  Removing existing output directory: {VGG_OUTPUT_DIR}")
        shutil.rmtree(VGG_OUTPUT_DIR)
    
    # Create directory structure: output/split/class/
    print(f"📁 Creating output directory structure...")
    for split in SPLIT_NAMES:
        for class_name in CLASSES:
            os.makedirs(os.path.join(VGG_OUTPUT_DIR, split, class_name), exist_ok=True)
            
    # Collect all file paths and their classes
    print(f"📂 Collecting images from {RAW_DATA_DIR}...")
    all_files = []
    for class_name in CLASSES:
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        if os.path.isdir(class_path):
            image_count = 0
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_files.append((os.path.join(class_path, file_name), class_name))
                    image_count += 1
            if image_count > 0:
                print(f"  ✓ Found {image_count} images in {class_name}/")
    
    if len(all_files) == 0:
        print(f"❌ Error: No image files found in {RAW_DATA_DIR}")
        print("Supported formats: .jpg, .jpeg, .png")
        return
    
    # 1. Stratified Train/Val/Test Split
    
    print(f"\n📊 Performing stratified train/val/test split...")
    file_paths = [f[0] for f in all_files]
    class_labels = [f[1] for f in all_files]
    
    # Split 1: Train vs (Val + Test)
    # This ensures proportional class distribution in training set
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, class_labels, 
        train_size=SPLIT_RATIO[0], 
        stratify=class_labels, 
        random_state=42  # Fixed seed for reproducibility
    )
    
    # Split 2: Val vs Test
    # Calculate ratio for second split
    val_test_ratio = SPLIT_RATIO[2] / (SPLIT_RATIO[1] + SPLIT_RATIO[2])
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, 
        test_size=val_test_ratio, 
        stratify=temp_labels, 
        random_state=42
    )

    splits = {
        'train': list(zip(train_paths, train_labels)), 
        'val': list(zip(val_paths, val_labels)), 
        'test': list(zip(test_paths, test_labels))
    }

    print(f"\n📈 Dataset Statistics:")
    print(f"  Total Images: {len(all_files)}")
    print(f"  Train: {len(train_paths)} ({len(train_paths)/len(all_files)*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} ({len(val_paths)/len(all_files)*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} ({len(test_paths)/len(all_files)*100:.1f}%)")

    # 2. Process and Save Files
    
    print(f"\n🔄 Processing images...")
    failed_count = 0
    
    for split_name, file_list in splits.items():
        print(f"\n  Processing {split_name} set ({len(file_list)} images)...")
        for file_path, class_name in tqdm(file_list, desc=f"  {split_name}"):
            
            # Read image
            img = cv2.imread(file_path)
            
            if img is None:
                print(f"\n⚠️  Warning: Failed to read image {file_path}. Skipping.")
                failed_count += 1
                continue
            
            # Step 1: Crop the brain region (remove black background)
            cleaned_img = crop_brain_region(img)
            
            if cleaned_img is None:
                print(f"\n⚠️  Warning: Failed to process image {file_path}. Skipping.")
                failed_count += 1
                continue

            # Step 2: Resize image to VGG-specific size (224x224)
            # VGG16 was trained on ImageNet with 224x224 images
            vgg_img = cv2.resize(cleaned_img, (VGG_IMG_SIZE, VGG_IMG_SIZE))
            
            # Step 3: Save the image into the final directory
            final_dir = os.path.join(VGG_OUTPUT_DIR, split_name, class_name)
            output_path = os.path.join(final_dir, os.path.basename(file_path))
            
            # Save with same extension as original
            success = cv2.imwrite(output_path, vgg_img)
            if not success:
                print(f"\n⚠️  Warning: Failed to save image {output_path}")
                failed_count += 1

    print(f"\n✅ VGG16 Preprocessing complete!")
    print(f"📁 Output saved to: {VGG_OUTPUT_DIR}")
    if failed_count > 0:
        print(f"⚠️  {failed_count} images failed to process")


if __name__ == "__main__":
    prepare_vgg_data()

