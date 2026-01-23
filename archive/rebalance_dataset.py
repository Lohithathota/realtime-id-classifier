"""
Reduce the 'other' class to balance the dataset better.
This will prevent the model from being biased toward the 'other' class.
"""

from pathlib import Path
import random
import shutil

print("=" * 60)
print("Rebalancing Dataset - Removing Excess 'Other' Images")
print("=" * 60)

train_other = Path("dataset/train/other")
val_other = Path("dataset/val/other")

# Get all other images
train_other_files = sorted(list(train_other.glob("*.jpg")))
val_other_files = sorted(list(val_other.glob("*.jpg")))

print(f"\nBefore rebalancing:")
print(f"  Train other: {len(train_other_files)} images")
print(f"  Val other: {len(val_other_files)} images")

# Target: reduce other to ~350 training images (similar ratio to aadhaar/pan/receipt)
# Keep only 350 for training, 28 for validation
target_train = 350
target_val = 28

# Remove excess training images
if len(train_other_files) > target_train:
    to_remove = len(train_other_files) - target_train
    files_to_delete = random.sample(train_other_files, to_remove)
    
    for f in files_to_delete:
        f.unlink()
    
    print(f"\n  Removed {to_remove} training images")

# Remove excess validation images
if len(val_other_files) > target_val:
    to_remove = len(val_other_files) - target_val
    files_to_delete = random.sample(val_other_files, to_remove)
    
    for f in files_to_delete:
        f.unlink()
    
    print(f"  Removed {to_remove} validation images")

# Verify
train_count = len(list(train_other.glob("*.jpg")))
val_count = len(list(val_other.glob("*.jpg")))

print(f"\nAfter rebalancing:")
print(f"  Train other: {train_count} images")
print(f"  Val other: {val_count} images")

print(f"\nDataset is now better balanced!")
print(f"Classes should have similar representation:")
print(f"  - Aadhaar: 264 images")
print(f"  - PAN: 264 images")
print(f"  - Payment Receipt: 255 images")
print(f"  - Other: {train_count} images")
print("=" * 60)
