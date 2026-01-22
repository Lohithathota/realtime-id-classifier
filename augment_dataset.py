"""
Data augmentation script to create more training samples from limited images.
Applies various transformations to existing images to expand the dataset.
"""

import os
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random

# Seed for reproducibility
random.seed(42)

DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

def augment_image(image_path: str, output_dir: Path, num_variations: int = 8):
    """
    Create multiple augmented versions of a single image.
    
    Args:
        image_path: Path to the original image
        output_dir: Directory to save augmented images
        num_variations: Number of variations to create
    """
    try:
        img = Image.open(image_path).convert('RGB')
        filename = Path(image_path).stem
        extension = Path(image_path).suffix
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_variations):
            aug_img = img.copy()
            
            # Rotation
            if i % 4 == 0:
                angle = random.randint(-15, 15)
                aug_img = aug_img.rotate(angle, expand=False)
            
            # Horizontal flip
            if i % 3 == 0:
                aug_img = ImageOps.mirror(aug_img)
            
            # Brightness adjustment
            if i % 2 == 0:
                enhancer = ImageEnhance.Brightness(aug_img)
                factor = random.uniform(0.8, 1.2)
                aug_img = enhancer.enhance(factor)
            
            # Contrast adjustment
            if i % 2 == 1:
                enhancer = ImageEnhance.Contrast(aug_img)
                factor = random.uniform(0.8, 1.2)
                aug_img = enhancer.enhance(factor)
            
            # Sharpness adjustment
            if i % 3 == 1:
                enhancer = ImageEnhance.Sharpness(aug_img)
                factor = random.uniform(0.5, 1.5)
                aug_img = enhancer.enhance(factor)
            
            # Zoom (crop and resize)
            if i % 5 == 0:
                crop_margin = random.randint(5, 20)
                width, height = aug_img.size
                crop_box = (crop_margin, crop_margin, width - crop_margin, height - crop_margin)
                aug_img = aug_img.crop(crop_box)
                aug_img = aug_img.resize((width, height), Image.LANCZOS)
            
            # Save augmented image
            output_path = output_dir / f"{filename}_aug_{i:02d}{extension}"
            aug_img.save(output_path)
            print(f"✓ Created: {output_path}")
        
    except Exception as e:
        print(f"✗ Error processing {image_path}: {str(e)}")


def main():
    """
    Augment all images in the training dataset.
    """
    print("=" * 60)
    print("Starting Dataset Augmentation")
    print("=" * 60)
    
    # Process training data
    train_classes = ['aadhaar', 'pan']
    
    for class_name in train_classes:
        class_dir = TRAIN_DIR / class_name
        if not class_dir.exists():
            print(f"✗ Directory not found: {class_dir}")
            continue
        
        print(f"\n📂 Processing class: {class_name}")
        print(f"   Directory: {class_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in class_dir.iterdir() 
                 if f.suffix.lower() in image_extensions]
        
        print(f"   Found {len(images)} original images")
        
        for img_file in images:
            print(f"\n   Processing: {img_file.name}")
            augment_image(str(img_file), class_dir, num_variations=10)
    
    print("\n" + "=" * 60)
    print("✓ Dataset augmentation complete!")
    print("=" * 60)
    
    # Count total images
    total_images = 0
    for class_name in train_classes:
        class_dir = TRAIN_DIR / class_name
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in class_dir.iterdir() 
                 if f.suffix.lower() in image_extensions]
        print(f"\n{class_name}: {len(images)} images")
        total_images += len(images)
    
    print(f"\nTotal training images: {total_images}")


if __name__ == "__main__":
    main()
