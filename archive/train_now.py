#!/usr/bin/env python
"""Minimal fast training - fixes class confusion"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

print("QUICK TRAINING - FIXING AADHAAR/PAN CONFUSION\n")

# Minimal config
BS, LR, EP = 32, 0.001, 10
D = torch.device("cpu")

# Transforms
tr = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
vr = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load data
print("Loading dataset...")
td = datasets.ImageFolder("dataset/train", transform=tr)
vd = datasets.ImageFolder("dataset/val", transform=vr)
print(f"Classes: {td.classes}\nTrain: {len(td)}, Val: {len(vd)}\n")

tl = DataLoader(td, batch_size=BS, shuffle=True, num_workers=0)
vl = DataLoader(vd, batch_size=BS, shuffle=False, num_workers=0)

# Model
print("Loading model...")
m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
m.fc = nn.Linear(512, 4)
m.to(D)

# Train
crit = nn.CrossEntropyLoss()
opt = optim.SGD(m.fc.parameters(), lr=LR, momentum=0.9)
best_acc = 0

print("EPOCH | TRAIN_LOSS | VAL_ACC")
print("-" * 35)

for e in range(EP):
    m.train()
    loss_t = 0
    for img, lbl in tl:
        img, lbl = img.to(D), lbl.to(D)
        opt.zero_grad()
        loss_t += crit(m(img), lbl).item()
        crit(m(img), lbl).backward()
        opt.step()
    
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

print("-" * 35)
print(f"Best: {best_acc:.2f}%\nDone!")
