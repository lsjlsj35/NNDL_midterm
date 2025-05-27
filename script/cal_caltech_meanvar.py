import os
from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
import numpy as np

data_root = Path(__file__).resolve().parent.parent / "data/caltech-101/101_ObjectCategories"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class Caltech101TrainOnly(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        dataset = datasets.ImageFolder(root=root)
        class_to_imgs = {cls_idx: [] for cls_idx in dataset.class_to_idx.values()}
        
        for path, label in dataset.samples:
            if len(class_to_imgs[label]) < 30:
                class_to_imgs[label].append((path, label))
        
        for img_list in class_to_imgs.values():
            self.samples.extend(img_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

train_dataset = Caltech101TrainOnly(data_root, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)

mean = torch.zeros(3)
std = torch.zeros(3)
nb_samples = 0

for images in train_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, 3, -1)  # [B, C, H*W]
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean.tolist()}")
print(f"Std:  {std.tolist()}")
