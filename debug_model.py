import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# Define constants
MODEL_PATH = "model.pth"
CLASS_NAMES = {0: "aadhaar", 1: "other", 2: "pan", 3: "payment_receipt"}
DEVICE = torch.device("cpu")

# Same transforms as training/inference
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    # Recreate architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 4)
    
    # Load weights
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'fc.weight' in checkpoint:
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try loading as full model
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        
    model.eval()
    return model

def test_image(model, img_path):
    print(f"\nTesting image: {img_path}")
    try:
        image = Image.open(img_path).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        print("Raw Logits:", outputs.numpy())
        print("Probabilities:", probs.numpy())
        
        conf, pred_idx = torch.max(probs, 1)
        print(f"Prediction: {CLASS_NAMES[pred_idx.item()]} ({conf.item()*100:.2f}%)")
        
    except Exception as e:
        print(f"Failed to process image: {e}")

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        return

    model = load_model()
    
    # Find one image from each class in dataset/train
    base_dir = "dataset/train"
    for cls_name in CLASS_NAMES.values():
        cls_dir = os.path.join(base_dir, cls_name)
        if os.path.exists(cls_dir):
            files = os.listdir(cls_dir)
            if files:
                # Pick the first image
                img_path = os.path.join(cls_dir, files[0])
                test_image(model, img_path)
            else:
                print(f"No images found for class {cls_name}")

if __name__ == "__main__":
    main()
