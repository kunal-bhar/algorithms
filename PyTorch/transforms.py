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


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1,
            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1,
            )
        self.fc1 = nn.Linear(16*8*8, num_classes)
               
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
        
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = 3
num_classes = 10
learning_rate = 3e-4 # Karpathy's Constant i.e. safe bet for an optimal lr
batch_size = 32
num_epochs = 5


model = CNN(in_channels=in_channels, num_classes=num_classes)
model.classifier = nn.Sequential(
    nn.Linear(512, 100),
    nn.ReLU(), 
    nn.Linear(100, 10)
)
model.to(device)


my_transforms = transforms.Compose([
    transforms.Resize((36, 36)),
    transforms.RandomCrop((32, 32)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ),    
])

train_dataset = datasets.CIFAR10(
    root='dataset/', train=True, transform=my_transforms, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    losses = []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
    print(f'Average Loss per epoch {epoch} is {sum(losses)/len(losses):.3f}')
    
def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions==y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/{num_samples} with Accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()
    
check_accuracy(train_loader, model)