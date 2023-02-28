# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:17:15 2023

@author: kunal
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader
    )
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm


device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 1

model = torchvision.models.vgg16(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False
    
model.avgpool = nn.Identity()
model.classifier = nn.Sequential(
    nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, num_classes)
    )
model.to(device)

train_dataset = datasets.CIFAR10(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True
    )
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    losses = []
    
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)        
       
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
    
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}')
    

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('=> Checking Accuracy on Training Data')
    else:
        print('=> Checking Accuracy on Test Data')
        
    num_correct = 0
    num_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)           
           
            scores = model(x)
            _, predictions = scores.max(1)
            
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        
    model.train()

check_accuracy(train_loader, model)

