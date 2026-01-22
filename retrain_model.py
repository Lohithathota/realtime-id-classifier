#!/usr/bin/env python
"""
Direct model retraining with minimal dependencies.
Saves model immediately after each batch to ensure it's saved.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from pathlib import Path
import sys

def main():
    print("Starting model retraining...")
    
    # Configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 3
    MODEL_PATH = "model.pth"
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
    val_dataset = datasets.ImageFolder("dataset/val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Calculate class weights
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    total = sum(class_counts)
    weights = torch.tensor([total / (len(train_dataset.classes) * c) for c in class_counts], dtype=torch.float32)
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Class distribution: {dict(zip(train_dataset.classes, class_counts))}")
    print(f"Class weights: {dict(zip(train_dataset.classes, weights.tolist()))}")
    
    # Load model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, len(train_dataset.classes))
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    # Training loop
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}:")
        
        # Train
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        correct = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total_val += labels.size(0)
        
        acc = 100 * correct / total_val
        print(f"  Validation Accuracy: {acc:.2f}%")
        
        # Save if best
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Model saved (best accuracy: {best_acc:.2f}%)")
    
    print(f"\nTraining complete!")
    print(f"Final model accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    
    # Verify
    if Path(MODEL_PATH).exists():
        size = Path(MODEL_PATH).stat().st_size / (1024 * 1024)
        print(f"File size: {size:.2f} MB")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
