"""
Create a balanced dataset with proper data augmentation for training.
Only applies augmentation once per image to create realistic variations.
"""

import os
from PIL import Image
import random
from pathlib import Path


def apply_augmentation(image_path, output_dir, num_augmentations=15):
    """
    Apply realistic augmentations to an image.
    Each augmentation is applied independently (not stacked).
    
    Args:
        image_path: Path to the original image
        output_dir: Directory to save augmented images
        num_augmentations: Number of augmentations to create
    """
    try:
        img = Image.open(image_path)
        
        # Save originals to output directory
        original_name = Path(image_path).stem
        original_ext = Path(image_path).suffix
        
        # Copy original
        output_path = os.path.join(output_dir, f"{original_name}_original{original_ext}")
        img.save(output_path)
        print(f"  Saved: {output_path}")
        
        # Create augmentations
        for i in range(num_augmentations):
            img_aug = Image.open(image_path)  # Reload original each time
            
            # Random augmentations (applied independently)
            aug_type = random.choice([
                'rotate',      # Slight rotation
                'flip',        # Horizontal flip
                'brightness',  # Brightness adjustment
                'contrast',    # Contrast adjustment
            ])
            
            if aug_type == 'rotate':
                angle = random.randint(-15, 15)
                img_aug = img_aug.rotate(angle, expand=False, fillcolor='white')
            
            elif aug_type == 'flip':
                img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)
            
            elif aug_type == 'brightness':
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(img_aug)
                factor = random.uniform(0.8, 1.2)
                img_aug = enhancer.enhance(factor)
            
            elif aug_type == 'contrast':
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(img_aug)
                factor = random.uniform(0.8, 1.2)
                img_aug = enhancer.enhance(factor)
            
            # Save augmented image
            output_path = os.path.join(output_dir, f"{original_name}_aug{i:02d}{original_ext}")
            img_aug.save(output_path)
        
        print(f"  Created {num_augmentations} augmentations")
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")


def main():
    """Create augmented dataset"""
    
    base_dir = Path("dataset")
    
    # Directories
    train_aadhaar = base_dir / "train" / "aadhaar"
    train_pan = base_dir / "train" / "pan"
    val_aadhaar = base_dir / "val" / "aadhaar"
    val_pan = base_dir / "val" / "pan"
    
    # Get original images
    train_aadhaar_originals = [f for f in train_aadhaar.glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    train_aadhaar_originals += [f for f in train_aadhaar.glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    train_pan_originals = [f for f in train_pan.glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    train_pan_originals += [f for f in train_pan.glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    val_aadhaar_originals = [f for f in val_aadhaar.glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    val_aadhaar_originals += [f for f in val_aadhaar.glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    val_pan_originals = [f for f in val_pan.glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    val_pan_originals += [f for f in val_pan.glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    train_other_originals = [f for f in (base_dir / "train" / "other").glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    train_other_originals += [f for f in (base_dir / "train" / "other").glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    val_other_originals = [f for f in (base_dir / "val" / "other").glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    val_other_originals += [f for f in (base_dir / "val" / "other").glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    train_payment_originals = [f for f in (base_dir / "train" / "payment_receipt").glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    train_payment_originals += [f for f in (base_dir / "train" / "payment_receipt").glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    val_payment_originals = [f for f in (base_dir / "val" / "payment_receipt").glob("*.jpg") if "aug" not in f.name and "original" not in f.name]
    val_payment_originals += [f for f in (base_dir / "val" / "payment_receipt").glob("*.jpeg") if "aug" not in f.name and "original" not in f.name]
    
    print(f"Found {len(train_aadhaar_originals)} original Aadhaar training images")
    print(f"Found {len(train_pan_originals)} original PAN training images")
    print(f"Found {len(train_other_originals)} original Other training images")
    print(f"Found {len(train_payment_originals)} original Payment Receipt training images")
    print(f"Found {len(val_aadhaar_originals)} original Aadhaar validation images")
    print(f"Found {len(val_pan_originals)} original PAN validation images")
    print(f"Found {len(val_other_originals)} original Other validation images")
    print(f"Found {len(val_payment_originals)} original Payment Receipt validation images")
    
    # Augment training data
    print("\n=== Augmenting Training Aadhaar ===")
    for img_path in train_aadhaar_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(train_aadhaar), num_augmentations=20)
    
    print("\n=== Augmenting Training PAN ===")
    for img_path in train_pan_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(train_pan), num_augmentations=20)
    
    print("\n=== Augmenting Training Other ===")
    for img_path in train_other_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(base_dir / "train" / "other"), num_augmentations=20)
    
    print("\n=== Augmenting Validation Aadhaar ===")
    for img_path in val_aadhaar_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(val_aadhaar), num_augmentations=5)
    
    print("\n=== Augmenting Validation PAN ===")
    for img_path in val_pan_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(val_pan), num_augmentations=5)
    
    print("\n=== Augmenting Validation Other ===")
    for img_path in val_other_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(base_dir / "val" / "other"), num_augmentations=5)
    
    print("\n=== Augmenting Training Payment Receipts ===")
    for img_path in train_payment_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(base_dir / "train" / "payment_receipt"), num_augmentations=15)
    
    print("\n=== Augmenting Validation Payment Receipts ===")
    for img_path in val_payment_originals:
        print(f"Processing: {img_path.name}")
        apply_augmentation(str(img_path), str(base_dir / "val" / "payment_receipt"), num_augmentations=5)
    
    print("\n✅ Dataset creation complete!")
    
    # Count final images
    train_aadhaar_count = len(list(train_aadhaar.glob("*.jpg"))) + len(list(train_aadhaar.glob("*.jpeg")))
    train_pan_count = len(list(train_pan.glob("*.jpg"))) + len(list(train_pan.glob("*.jpeg")))
    train_other_count = len(list((base_dir / "train" / "other").glob("*.jpg"))) + len(list((base_dir / "train" / "other").glob("*.jpeg")))
    train_payment_count = len(list((base_dir / "train" / "payment_receipt").glob("*.jpg"))) + len(list((base_dir / "train" / "payment_receipt").glob("*.jpeg")))
    val_aadhaar_count = len(list(val_aadhaar.glob("*.jpg"))) + len(list(val_aadhaar.glob("*.jpeg")))
    val_pan_count = len(list(val_pan.glob("*.jpg"))) + len(list(val_pan.glob("*.jpeg")))
    val_other_count = len(list((base_dir / "val" / "other").glob("*.jpg"))) + len(list((base_dir / "val" / "other").glob("*.jpeg")))
    val_payment_count = len(list((base_dir / "val" / "payment_receipt").glob("*.jpg"))) + len(list((base_dir / "val" / "payment_receipt").glob("*.jpeg")))
    
    print(f"\nFinal dataset size:")
    print(f"  Train Aadhaar: {train_aadhaar_count}")
    print(f"  Train PAN: {train_pan_count}")
    print(f"  Train Other: {train_other_count}")
    print(f"  Train Payment Receipt: {train_payment_count}")
    print(f"  Val Aadhaar: {val_aadhaar_count}")
    print(f"  Val PAN: {val_pan_count}")
    print(f"  Val Other: {val_other_count}")
    print(f"  Val Payment Receipt: {val_payment_count}")
    print(f"  Total: {train_aadhaar_count + train_pan_count + train_other_count + train_payment_count + val_aadhaar_count + val_pan_count + val_other_count + val_payment_count}")


if __name__ == "__main__":
    main()
