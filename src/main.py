import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter


def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # print(outputs.shape, labels.shape)
        # print(labels)
        loss = criterion(outputs, labels)
        # print(outputs.shape, labels.shape, loss.cpu())
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.detach().cpu().item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


parser = argparse.ArgumentParser()
parser.add_argument("--cls-lr", type=float, default=1e-2)
parser.add_argument("--other-lr", type=float, default=1e-4)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--noinit", action="store_true")
args = parser.parse_args()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5474404692649841, 0.5220522880554199, 0.4891655445098877], [0.24121175706386566, 0.24064411222934723, 0.24012628197669983])  # script/cal_caltech_meanvar.py
])

root = Path(__file__).resolve().parent.parent / "data/caltech-101"
log_dir = f"runs/caltech101_alexnet_LR{args.cls_lr}_lr{args.other_lr}_BS{args.bs}_{'init' if not args.noinit else 'rand'}"
save_dir = f"checkpoints/caltech101_alexnet_LR{args.cls_lr}_lr{args.other_lr}_BS{args.bs}_{'init' if not args.noinit else 'rand'}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
dataset = ImageFolder(root=root / '101_ObjectCategories', transform=transform)

train_indices = []
test_indices = []
class_to_count = {cls_idx: 0 for cls_idx in range(len(dataset.classes))}

for idx, (path, label) in enumerate(dataset.samples):
    if class_to_count[label] < 30:
        train_indices.append(idx)
        class_to_count[label] += 1
    else:
        test_indices.append(idx)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

model = alexnet(pretrained=not args.noinit)
model.classifier[6] = nn.Linear(4096, 102)

for name, param in model.named_parameters():
    if 'classifier.6' not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([
    {'params': model.classifier[6].parameters(), 'lr': args.cls_lr},
    {'params': [p for n, p in model.named_parameters() if 'classifier.6' not in n], 'lr': args.other_lr}
], momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

TR, EV = [], []
best = 0
for epoch in range(10):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    TR.append(train_acc)
    EV.append(test_acc)
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", test_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_acc, epoch)
    writer.add_scalar("Accuracy/Val", test_acc, epoch)
    print(f"Epoch {epoch}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}")
    if test_acc > best:
        best = test_acc
        torch.save(model.state_dict(), save_dir+"/best.pth")
writer.close()
with open("CONFIG.txt", 'a') as f:
    f.write(f"LR{args.cls_lr}_lr{args.other_lr}_BS{args.bs}_{'init' if not args.noinit else 'rand'}: {TR}; {EV}\n")
