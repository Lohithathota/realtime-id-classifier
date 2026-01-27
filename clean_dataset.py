
import os
import glob
from pathlib import Path

DATASET_DIR = Path("dataset")

def clean_synthetic_data():
    print("Cleaning synthetic data (files starting with 'syn_')...")
    
    deleted_count = 0
    preserved_count = 0
    
    # Walk through all files in dataset
    for filepath in DATASET_DIR.rglob("*"):
        if filepath.is_file():
            filename = filepath.name
            
            # Check if it looks synthetic (created by generate_massive_dataset.py)
            if filename.startswith("syn_"):
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")
            else:
                preserved_count += 1
                # print(f"Preserved: {filepath}")

    print(f"Done.")
    print(f"Deleted {deleted_count} synthetic images.")
    print(f"Preserved {preserved_count} real images/files.")
    
    # Check status of classes
    for split in ["train", "val"]:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
            
        print(f"\nStatus of {split}:")
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*")))
                print(f"  - {class_dir.name}: {count} images")

if __name__ == "__main__":
    clean_synthetic_data()
