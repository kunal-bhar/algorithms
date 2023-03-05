import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from torch import optim
from torch import nn
from torch.utils.data import (
    DataLoader, Dataset, random_split
    )
from tqdm import tqdm


class CatsAndDogsDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(self.annotations.iloc[index, 1])
        
        if self.transform:
            image = self.transform(image)
            
        return (image, y_label)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channel = 3
num_classes = 2
learning_rate = 0.001
batch_size = 32
num_epochs = 10


dataset = CatsAndDogsDataset(
    csv_file='PyTorch\Custom Datasets\cats_dogs.csv', root_dir='PyTorch\Custom Datasets\cats_dogs_resized', transform=transforms.ToTensor()
                             )
train_dataset, test_dataset = random_split(dataset, [5, 5])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model = models.googlenet(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False # freeze all layers
    
model.fc = nn.Linear(in_features=1024, out_features=num_classes) # the final layer isn't frozen
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


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
        
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}')
        
        
def check_accuracy(loader, model):
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
            
    model.train()
    return num_correct/num_samples

print(f'Accuracy on Train Set: {check_accuracy(train_loader, model)*100:.2f}')
print(f'Accuracy on Test Set: {check_accuracy(test_loader, model)*100:.2f}')