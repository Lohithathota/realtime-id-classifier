#!/usr/bin/env python
"""Train for HIGH CONFIDENCE predictions (90-100%)"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

print("TRAINING - HIGH CONFIDENCE MODE (90-100%)\n")

BS, LR, EP = 16, 0.0005, 10
D = torch.device("cpu")

tr = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
vr = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset...")
td = datasets.ImageFolder("dataset/train", transform=tr)
vd = datasets.ImageFolder("dataset/val", transform=vr)

# Count samples per class
class_counts = [0] * len(td.classes)
for _, label in td.samples:
    class_counts[label] += 1

print(f"Classes: {td.classes}")
print(f"Counts: {dict(zip(td.classes, class_counts))}")

# Inverse frequency weighting for balance
total = sum(class_counts)
weights = torch.tensor([total / (len(td.classes) * c) for c in class_counts], dtype=torch.float32)
print(f"Weights: {dict(zip(td.classes, weights.tolist()))}\n")

tl = DataLoader(td, batch_size=BS, shuffle=True, num_workers=0)
vl = DataLoader(vd, batch_size=BS, shuffle=False, num_workers=0)

# Model
print("Loading model...")
m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
m.fc = nn.Linear(512, 4)
m.to(D)

# Use standard CrossEntropyLoss WITHOUT label smoothing for high confidence
crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.0)
# Freeze backbone, only train final layer (faster)
for param in m.layer1.parameters():
    param.requires_grad = False
for param in m.layer2.parameters():
    param.requires_grad = False
for param in m.layer3.parameters():
    param.requires_grad = False
for param in m.layer4.parameters():
    param.requires_grad = False
opt = optim.Adam(m.fc.parameters(), lr=LR)
best_acc = 0

print("EPOCH | TRAIN_LOSS | VAL_ACC")
print("-" * 40)

for e in range(EP):
    m.train()
    loss_t = 0
    for img, lbl in tl:
        img, lbl = img.to(D), lbl.to(D)
        opt.zero_grad()
        loss = crit(m(img), lbl)
        loss.backward()
        opt.step()
        loss_t += loss.item()
    
    m.eval()
    cor, tot = 0, 0
    with torch.no_grad():
        for img, lbl in vl:
            img, lbl = img.to(D), lbl.to(D)
            cor += (m(img).argmax(1) == lbl).sum().item()
            tot += lbl.size(0)
    
    acc = 100 * cor / tot
    print(f"{e+1:3d}   | {loss_t/len(tl):9.4f} | {acc:6.2f}%", end="")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(m.state_dict(), "model.pth")
        print(" [SAVED]")
    else:
        print()

print("-" * 40)
print(f"Best: {best_acc:.2f}%\nDone! Restart server.")
