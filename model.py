# necessary libraries
from __future__ import annotations
import scipy
# import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision.datasets import ImageNet


imagenet_dataset = ImageNet(download=True, split="train")