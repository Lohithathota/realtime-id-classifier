
import shutil
import os
from pathlib import Path

# Paths to the uploaded files (from the user state provided in previous turn)
# I need to use the exact paths provided in the prompt context.
uploaded_files = [
    r"C:\Users\LohithaThota\.gemini\antigravity\brain\0d12c706-e250-4be4-9de9-39d900f8a5fe\uploaded_image_0_1769153806552.png",
    r"C:\Users\LohithaThota\.gemini\antigravity\brain\0d12c706-e250-4be4-9de9-39d900f8a5fe\uploaded_image_1_1769153806552.jpg",
    r"C:\Users\LohithaThota\.gemini\antigravity\brain\0d12c706-e250-4be4-9de9-39d900f8a5fe\uploaded_image_2_1769153806552.jpg",
    r"C:\Users\LohithaThota\.gemini\antigravity\brain\0d12c706-e250-4be4-9de9-39d900f8a5fe\uploaded_image_3_1769153806552.jpg"
]

target_dir = Path("dataset/train/aadhaar")
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Copying files to {target_dir}...")

COPIES_PER_IMAGE = 25  # Increase weight of real data (4 * 25 = 100 real images)

for i, src_path in enumerate(uploaded_files):
    src = Path(src_path)
    if not src.exists():
        print(f"Warning: Source file not found: {src}")
        continue
        
    # Copy original and duplicates
    for j in range(COPIES_PER_IMAGE):
        ext = src.suffix
        dst_name = f"real_aadhaar_user_{i}_{j}{ext}"
        dst_path = target_dir / dst_name
        shutil.copy2(src, dst_path)

print(f"Successfully added {len(uploaded_files) * COPIES_PER_IMAGE} real image samples (including duplicates).")
