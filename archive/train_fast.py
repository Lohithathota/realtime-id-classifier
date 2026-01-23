"""
Fast training script focused on improving "other" class detection.
Uses weighted loss to emphasize the "other" class.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from pathlib import Path
import sys

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 10
DEVICE = torch.device("cpu")
MODEL_PATH = "model.pth"
DATASET_PATH = "dataset"


def get_class_weights(train_dataset):
    """Calculate class weights to emphasize minority classes"""
    class_counts = {}
    for _, label in train_dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\n=== Class Distribution ===")
    for class_idx, class_name in enumerate(train_dataset.classes):
        count = class_counts.get(class_idx, 0)
        print(f"  {class_name}: {count} samples")
    
    total = sum(class_counts.values())
    weights = {}
    for class_idx in range(len(train_dataset.classes)):
        count = class_counts.get(class_idx, 1)
        # Inverse frequency weighting
        weight = total / (len(train_dataset.classes) * count)
        weights[class_idx] = weight
    
    print("\n=== Class Weights ===")
    for class_idx, class_name in enumerate(train_dataset.classes):
        print(f"  {class_name}: {weights.get(class_idx, 1):.4f}")
    
    class_weight_tensor = torch.tensor(
        [weights.get(i, 1) for i in range(len(train_dataset.classes))],
        dtype=torch.float32,
        device=DEVICE
    )
    return class_weight_tensor


def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    print(f"Epoch Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return epoch_loss


def validate(model, val_loader):
    """Validate the model"""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    val_loss /= len(val_loader)
    
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return val_acc, val_loss


def main():
    print("=" * 60)
    print("Fast Model Training with Class Weighting")
    print("=" * 60)
    
    # Load datasets
    print(f"\nLoading dataset from {DATASET_PATH}...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = datasets.ImageFolder(
        root=f"{DATASET_PATH}/train",
        transform=transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=f"{DATASET_PATH}/val",
        transform=transform
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Get class weights
    class_weights = get_class_weights(train_dataset)
    
    # Load pre-trained model
    print("\nLoading pre-trained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify final layer for 4 classes
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)
    
    print(f"Modified model for {num_classes} classes")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    print("\n" + "=" * 60)
    print("Beginning Training with Weighted Loss")
    print("=" * 60)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_acc, val_loss = validate(model, val_loader)
        
        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Model saved! Best validation accuracy: {best_val_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final model saved to {MODEL_PATH}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
