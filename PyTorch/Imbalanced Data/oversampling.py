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
# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor({1, 50, 12, 25, ...}))

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose({
        # practically, you'd also add some data augmentation here
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    })
    
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)
    class_weights = [1, 50]
    
    # here we know our data is in the ratio 50:1 for classes [0, 1]
    # to generalize, we can set class_weights = [] & use the loops below as apt:
    
    # for root, subdir, files in os.walk(root_dir):
    #     if len(files) > 0:
    #         class_weights.append(1/len(files))
    
    # subdirectories = dataset.classes
    # for subdir in subdirectories:
    #     files = os.listdir(os.path.join(root_dir, subdir))
    #     class_weights.append(1/len(files))
            
    sample_weights = [0] * len(dataset)
    
    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label] # map class wt to class label
        sample_weights[idx] = class_weight # set sample wts to corresp class wts
    
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
        ) 
    # we wanna use replacement=True while oversampling, as doing otherwise
    # would mean that we'd see that particular sample only once while itr our dataset
    
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, sampler=sampler
                        )
    return loader
    

def main():
    loader = get_loader(root_dir='PyTorch\Imbalanced Data\dataset', batch_size=8)
    
    num_retrievers = 0
    num_elkhounds = 0
    
    for epoch in range(10):
        for data, labels in loader:
            num_retrievers += torch.sum(labels == 0)
            num_elkhounds += torch.sum(labels == 1)
    
    print(num_retrievers)
    print(num_elkhounds)
    
if __name__ == '__main__':
    main() 