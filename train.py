"""
Training script for ResNet18 model on image classification.
Trains on: Aadhaar, PAN, Payment Receipt, and Other (unsupported documents)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from pathlib import Path
import sys

# Configuration
DATASET_DIR = Path("dataset")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
MODEL_SAVE_PATH = "model.pth"

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
DEVICE = torch.device("cpu")
PATIENCE = 3

# Image preprocessing transforms
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_data():
    """Load training and validation datasets."""
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(f"Dataset directories not found")
    
    print(f"Loading data from {TRAIN_DIR} and {VAL_DIR}...")
    
    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=TRAIN_TRANSFORM)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=VAL_TRANSFORM)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Calculate class weights
    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    total = sum(class_counts)
    weights = torch.tensor(
        [total / (len(train_dataset.classes) * c) for c in class_counts],
        dtype=torch.float32,
        device=DEVICE
    )
    
    print(f"Class counts: {dict(zip(train_dataset.classes, class_counts))}")
    print(f"Class weights: {dict(zip(train_dataset.classes, weights.tolist()))}")
    
    return train_loader, val_loader, len(train_dataset.classes), weights


def main():
    print("=" * 70)
    print("TRAINING IMAGE CLASSIFICATION MODEL")
    print("=" * 70)
    
    # Load data
    train_loader, val_loader, num_classes, class_weights = load_data()
    
    # Create model
    print("\nLoading pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, num_classes)
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    best_val_acc = 0
    patience_count = 0
    
    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}: Loss={loss.item():.4f}")
        
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [BEST] Model saved! (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping (no improvement for {PATIENCE} epochs)")
                break
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
