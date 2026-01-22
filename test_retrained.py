"""
Test the retrained model to verify it correctly classifies images.
"""

import torch
from utils import load_model, preprocess_image
from PIL import Image, ImageDraw
from pathlib import Path
from io import BytesIO
import time

print("=" * 70)
print("TESTING RETRAINED MODEL")
print("=" * 70)

# Load model
print("\n1. Loading retrained model...")
try:
    model = load_model("model.pth")
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Create test doll image
print("\n2. Creating synthetic doll image...")
doll_img = Image.new('RGB', (224, 224), color=(255, 200, 180))
draw = ImageDraw.Draw(doll_img)

# Draw head
draw.ellipse([60, 40, 140, 120], fill=(240, 190, 170), outline=(200, 150, 130), width=2)
# Draw body
draw.ellipse([50, 120, 150, 200], fill=(255, 100, 100), outline=(200, 50, 50), width=2)
# Draw eyes
draw.ellipse([80, 70, 95, 85], fill=(0, 0, 0))
draw.ellipse([110, 70, 125, 85], fill=(0, 0, 0))
# Draw smile
draw.arc([80, 85, 120, 105], 0, 180, fill=(0, 0, 0), width=2)

doll_bytes = BytesIO()
doll_img.save(doll_bytes, format='JPEG')
doll_bytes.seek(0)

# Test doll
print("\n3. Testing doll image...")
try:
    img_tensor = preprocess_image(doll_bytes)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    pred_class = ["aadhaar", "other", "pan", "payment_receipt"][pred_idx.item()]
    print(f"   Predicted: {pred_class}")
    print(f"   Confidence: {conf.item():.4f}")
    print(f"   All probabilities:")
    for i, class_name in enumerate(["aadhaar", "other", "pan", "payment_receipt"]):
        print(f"     {class_name}: {probs[0][i].item():.4f}")
    
    if pred_class == "other":
        print("   ✅ CORRECT! Doll correctly classified as 'other'")
    else:
        print(f"   ❌ WRONG! Should be 'other', not '{pred_class}'")
    
except Exception as e:
    print(f"❌ Error testing doll: {e}")
    import traceback
    traceback.print_exc()

# Test some training images
print("\n4. Testing training images...")

# Test "other" images
other_path = Path("dataset/train/other")
other_files = sorted(list(other_path.glob("*.jpg")))[:5]

print(f"\n   Testing {len(other_files)} 'other' images:")
correct_other = 0
for img_file in other_files:
    try:
        with open(img_file, 'rb') as f:
            img_bytes = BytesIO(f.read())
        
        img_tensor = preprocess_image(img_bytes)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        pred_class = ["aadhaar", "other", "pan", "payment_receipt"][pred_idx.item()]
        
        if pred_class == "other":
            correct_other += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"     {status} {img_file.name[:25]:25s} → {pred_class} ({conf.item():.4f})")
    except Exception as e:
        print(f"     ❌ Error: {str(e)[:50]}")

print(f"   Correct: {correct_other}/{len(other_files)}")

# Test "aadhaar" images
aadhaar_path = Path("dataset/train/aadhaar")
aadhaar_files = sorted(list(aadhaar_path.glob("*.jpg")))[:3]

print(f"\n   Testing {len(aadhaar_files)} 'aadhaar' images:")
correct_aadhaar = 0
for img_file in aadhaar_files:
    try:
        with open(img_file, 'rb') as f:
            img_bytes = BytesIO(f.read())
        
        img_tensor = preprocess_image(img_bytes)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        pred_class = ["aadhaar", "other", "pan", "payment_receipt"][pred_idx.item()]
        
        if pred_class == "aadhaar":
            correct_aadhaar += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"     {status} {img_file.name[:25]:25s} → {pred_class} ({conf.item():.4f})")
    except Exception as e:
        print(f"     ❌ Error: {str(e)[:50]}")

print(f"   Correct: {correct_aadhaar}/{len(aadhaar_files)}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
