#!/usr/bin/env python
"""Diagnose why images are being misclassified as PAN"""
import torch
from utils import load_model, preprocess_image
from PIL import Image
from pathlib import Path
from io import BytesIO

print("DIAGNOSING PAN MISCLASSIFICATION ISSUE\n")
print("=" * 60)

model = load_model("model.pth")
model.eval()

# Test each class
test_dirs = {
    "aadhaar": "dataset/train/aadhaar",
    "pan": "dataset/train/pan",
    "payment_receipt": "dataset/train/payment_receipt",
    "other": "dataset/train/other"
}

class_names = ["aadhaar", "other", "pan", "payment_receipt"]

for test_class, test_dir in test_dirs.items():
    print(f"\nTesting {test_class.upper()} images:")
    print("-" * 60)
    
    path = Path(test_dir)
    images = sorted(list(path.glob("*.jpg")))[:3]
    
    for img_file in images:
        try:
            with open(img_file, 'rb') as f:
                img_bytes = BytesIO(f.read())
            
            img_tensor = preprocess_image(img_bytes)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
            
            pred_idx = torch.argmax(probs[0]).item()
            pred_class = class_names[pred_idx]
            confidence = probs[0][pred_idx].item()
            
            print(f"\n{img_file.name}")
            print(f"  Predicted: {pred_class} ({confidence:.4f})")
            print(f"  All scores:")
            for i, name in enumerate(class_names):
                print(f"    {name:20s}: {probs[0][i].item():.4f}")
            
            # Check if correct
            if pred_class == test_class:
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ WRONG (expected {test_class})")
        
        except Exception as e:
            print(f"  ERROR: {e}")

print("\n" + "=" * 60)
