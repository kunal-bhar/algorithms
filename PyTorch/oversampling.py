import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import (
    WeightedRandomSampler, DataLoader,
)
import torchvision.transforms as transforms
import torch.nn as nn

# Methods v Imbalanced Datasets
# 1. Class Weighting ; brute force
# 2. Oversampling ; preferred

# Example of a class-weighted loss fxn
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor({1, 50, 12, 25, ...}))