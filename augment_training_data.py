"""
Data Augmentation Script for Brain Tumor Classification Training

This script generates augmented training images to create a larger training dataset.
It applies comprehensive augmentation transforms to each training image and saves
multiple augmented versions to a new directory.

Usage:
    python augment_training_data.py

The script will:
1. Load images from data/vgg16_classification/train/
2. Apply augmentation transforms to each image
3. Save augmented images to data/vgg16_classification/train_augmented/
4. Create multiple augmented versions per original image (configurable)

After running this script, update your training notebooks to use:
    DATA_DIR = 'data/vgg16_classification'
    Train from: train_augmented/ (instead of train/)
"""

import os
import csv
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
from torchvision import transforms

# Configuration
INPUT_TRAIN_DIR = 'data/vgg16_classification/train'
OUTPUT_AUGMENTED_DIR = 'data/vgg16_classification/train_augmented'
AUGMENTATIONS_PER_IMAGE = 5  # Number of augmented versions to create per original image

# Class names
CLASS_NAMES = ['NO_TUMOR', 'GLIOMA', 'MENINGIOMA', 'PITUITARY']


def apply_augmentation(image, aug_type='random'):
    """
    Apply augmentation to a single image.
    
    Args:
        image: PIL Image object
        aug_type: Type of augmentation ('random', 'rotation', 'flip', 'color', 'affine', 'blur', 'sharpness')
    
    Returns:
        Augmented PIL Image
    """
    if aug_type == 'random':
        # Randomly select one augmentation type
        aug_type = random.choice(['rotation', 'flip', 'color', 'affine', 'blur', 'sharpness', 'crop'])
    
    if aug_type == 'rotation':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(224),
        ])
    elif aug_type == 'flip':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomCrop(224),
        ])
    elif aug_type == 'color':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        ])
    elif aug_type == 'affine':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.RandomCrop(224),
        ])
    elif aug_type == 'blur':
        # Use GaussianBlur if available, otherwise skip blur augmentation
        try:
            from torchvision.transforms import GaussianBlur
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        except (ImportError, AttributeError):
            # Fallback: use a simple combination without blur
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
    elif aug_type == 'sharpness':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
        ])
    elif aug_type == 'crop':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
        ])
    else:
        # Default: combination of multiple augmentations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    return transform(image)


def augment_training_data():
    """
    Main function to augment training data.
    """
    print("=" * 70)
    print("Data Augmentation for Brain Tumor Classification")
    print("=" * 70)
    print(f"\nInput directory: {INPUT_TRAIN_DIR}")
    print(f"Output directory: {OUTPUT_AUGMENTED_DIR}")
    print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    print("\n" + "-" * 70)
    
    # Check if input directory exists
    if not os.path.exists(INPUT_TRAIN_DIR):
        print(f"ERROR: Input directory '{INPUT_TRAIN_DIR}' does not exist!")
        print("Please run the preprocessing script first (preprocess_vgg16.py)")
        return
    
    # Create output directory structure
    os.makedirs(OUTPUT_AUGMENTED_DIR, exist_ok=True)
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(OUTPUT_AUGMENTED_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Statistics
    total_original = 0
    total_augmented = 0
    augmentation_types = ['rotation', 'flip', 'color', 'affine', 'blur', 'sharpness', 'crop']
    
    # Metadata list for CSV
    metadata = []
    
    # Process each class
    for class_name in CLASS_NAMES:
        class_input_dir = os.path.join(INPUT_TRAIN_DIR, class_name)
        class_output_dir = os.path.join(OUTPUT_AUGMENTED_DIR, class_name)
        
        if not os.path.exists(class_input_dir):
            print(f"Warning: Class directory '{class_input_dir}' does not exist. Skipping...")
            continue
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(Path(class_input_dir).glob(f'*{ext}'))
        
        if len(image_files) == 0:
            print(f"Warning: No images found in '{class_input_dir}'. Skipping...")
            continue
        
        print(f"\nProcessing class: {class_name}")
        print(f"  Original images: {len(image_files)}")
        
        total_original += len(image_files)
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"  Augmenting {class_name}"):
            try:
                # Load original image
                img = Image.open(img_path).convert('RGB')
                original_name = img_path.stem
                original_ext = img_path.suffix
                
                # Save original image first (copy to augmented directory)
                original_filename = f"{original_name}{original_ext}"
                original_output_path = os.path.join(class_output_dir, original_filename)
                img.save(original_output_path)
                total_augmented += 1
                
                # Add original image to metadata
                relative_path = os.path.join('train_augmented', class_name, original_filename)
                metadata.append({
                    'image_path': relative_path.replace(os.sep, '/'),  # Use forward slashes for consistency
                    'full_path': os.path.abspath(original_output_path).replace(os.sep, '/'),
                    'class': class_name,
                    'split': 'train',
                    'filename': original_filename,
                    'augmentation_type': 'original',
                    'original_filename': original_filename
                })
                
                # Create augmented versions
                for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                    # Select augmentation type (cycle through types)
                    aug_type = augmentation_types[aug_idx % len(augmentation_types)]
                    
                    # Apply augmentation
                    aug_img = apply_augmentation(img, aug_type=aug_type)
                    
                    # Save augmented image
                    aug_filename = f"{original_name}_aug{aug_idx+1}_{aug_type}{original_ext}"
                    aug_output_path = os.path.join(class_output_dir, aug_filename)
                    aug_img.save(aug_output_path)
                    total_augmented += 1
                    
                    # Add augmented image to metadata
                    relative_path = os.path.join('train_augmented', class_name, aug_filename)
                    metadata.append({
                        'image_path': relative_path.replace(os.sep, '/'),
                        'full_path': os.path.abspath(aug_output_path).replace(os.sep, '/'),
                        'class': class_name,
                        'split': 'train',
                        'filename': aug_filename,
                        'augmentation_type': aug_type,
                        'original_filename': original_filename
                    })
                    
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue
        
        print(f"  Total images created: {len(image_files) * (AUGMENTATIONS_PER_IMAGE + 1)}")
    
    # Create metadata CSV
    csv_output = 'data/augmented_dataset_metadata.csv'
    print("\n" + "-" * 70)
    print("Creating metadata CSV file...")
    
    if len(metadata) == 0:
        print("WARNING: No images found for CSV creation.")
    else:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_output), exist_ok=True)
        
        # Write to CSV
        with open(csv_output, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['image_path', 'full_path', 'class', 'split', 'filename', 'augmentation_type', 'original_filename']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in metadata:
                writer.writerow(row)
        
        print(f"Metadata CSV saved to: {csv_output}")
        print(f"Total entries: {len(metadata)}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Augmentation Complete!")
    print("=" * 70)
    print(f"Original images: {total_original}")
    print(f"Total augmented images: {total_augmented}")
    print(f"Augmentation factor: {total_augmented / total_original:.2f}x")
    print(f"\nAugmented data saved to: {OUTPUT_AUGMENTED_DIR}")
    print(f"Metadata CSV saved to: {csv_output}")
    print("\nNext steps:")
    print("1. Training notebooks are already configured to use 'train_augmented' directory")
    print("2. Start training with the augmented dataset!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    augment_training_data()
