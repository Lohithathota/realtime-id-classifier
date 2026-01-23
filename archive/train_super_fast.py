"""
Ultra-fast model retraining focused on "other" class.
Uses weighted loss and early stopping to minimize training time.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms, models, datasets
from pathlib import Path

# Ultra-fast config
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cpu")
MODEL_PATH = "model.pth"
DATASET_PATH = "dataset"

print("=" * 60)
print("Ultra-Fast Model Retraining")
print("=" * 60)

# Load datasets with faster loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(f"{DATASET_PATH}/train", transform=transform)
val_dataset = datasets.ImageFolder(f"{DATASET_PATH}/val", transform=transform)

print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Calculate weighted loss
class_counts = [0] * len(train_dataset.classes)
for _, label in train_dataset.samples:
    class_counts[label] += 1

total_samples = sum(class_counts)
weights = torch.tensor([total_samples / (len(train_dataset.classes) * c) for c in class_counts], dtype=torch.float32)
print(f"Class weights: {weights.tolist()}")

# Load model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, len(train_dataset.classes))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

print("\nTraining...")

best_val_acc = 0
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss={loss.item():.4f}")
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}%\n")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ Model saved! Best accuracy: {best_val_acc:.2f}%\n")

print("=" * 60)
print(f"Training Complete! Best accuracy: {best_val_acc:.2f}%")
print("=" * 60)
