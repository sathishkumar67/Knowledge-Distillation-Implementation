import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet152

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import Trainer

from datasets import load_dataset

import os
import sys
from PIL import Image

from huggingface_hub import hf_hub_url, hf_hub_download, login, HfApi


# hyperparameters - next time use dataclass and argparse
EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_CLASSES = None
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
T_MAX = 50
ETA_MIN = 3e-4


# Define your transformations (augmentations)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


# import mnist dataset from pytorch
train_dataset = torchvision.datasets.Food101(root='./data', split='train', download=True, transform=transform)
test_dataset = torchvision.datasets.Food101(root='./data', split='test', download=True, transform=transform)

# unique labels
unique_labels = train_dataset.classes
NUM_CLASSES = len(unique_labels)


# dataloader for train and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# show an image
# item = next(iter(train_loader))
# img, label = item[0][1], item[1][1]

# plt.imshow(img.permute(1, 2, 0))
# plt.title(label)
# plt.axis('off')
# plt.show()

# student model (ResNet-50)
student_base = resnet50(pretrained=False)
student_base.fc = nn.Linear(2048, NUM_CLASSES)

# no of parameters
print(f"Number of parameters: {sum(p.numel() for p in student_base.parameters())/1e6}M")


# wrapper for student model using lightning module
class Student(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # storing loss
        self.train_loss = []
    
    def training_step(self, batch, batch_idx):
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        self.train_loss.append(loss)
        self.log("Train_Loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def validation_step(self, batch, batch_idx):
        batch, label = batch
        out = self.model(batch)
        loss = F.cross_entropy(out, label)
        self.log("Val_Loss", loss, prog_bar=True)
        return loss

# create a trainer for the model
trainer = Trainer(max_epochs=EPOCHS,
                  accelerator="cuda")

# create student model
student = Student(student_base)

# training
trainer.fit(student, train_loader, test_loader)
