# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:56:26 2023

@author: kunal
"""

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import (
    DataLoader
    )
from tqdm import tqdm

input_size = 28
hidden_size = 256
num_layers = 2
num_classes = 10
sequence_length = 28
learning_rate = 3e-4
batch_size = 64
num_epochs = 1

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        
        out = self.fc(out)
        return out
    
    
class RNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        
        out = self.fc(out)
        return out    
    
    
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(
            x, (h0, c0)
            )
        out = out.reshape(out.shape[0], -1)
        
        out = self.fc(out)
        return out    
    
class RNN_BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_BLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
            
        out, _ = self.blstm(
            x, (h0, c0)
            )
        out = self.fc(out[:, -1, :])
        return out
    
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = datasets.MNIST(
    root='/dataset', train=True, transform=transforms.ToTensor(), download=True
    )

test_dataset = datasets.MNIST(
    root='/dataset', train=False, transform=transforms.ToTensor(), download=True
    )

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)        
       
        scores = model(data)
        loss = criterion(scores, targets)
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
                        
            scores = model(x)
            _, predictions = scores.max(1)
            
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
            
    model.train()
    return num_correct/num_samples

print(f'Accuracy on Train Set: {check_accuracy(train_loader, model)*100:.2f}')
print(f'Accuracy on Test Set: {check_accuracy(test_loader, model)*100:.2f}')