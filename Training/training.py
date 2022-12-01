import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()

import torchvision

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import torch.optim as optim

from models import ConvNet
from models import DeConvNet
from preprocess import CenterCrop224

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torch



def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  


def train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    train_transforms = transforms.Compose([
        CenterCrop224(),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )
    
    return train_dataset
  


def val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    
    val_transforms = transforms.Compose([
	CenterCrop224(),
        transforms.ToTensor(),
        normalize_transform()
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset
  


def data_loader(data_dir, batch_size=128, workers=10, pin_memory=True):
    train_ds = train_dataset(data_dir)
    val_ds = val_dataset(data_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader



def main() :
    torch.cuda.empty_cache()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = '/ssd_scratch/cvit/shreya/Imagenet2012/Imagenet-10/'
    # Load Data
    # imagenet_data = torchvision.datasets.ImageNet('/ssd_scratch/cvit/shreya/Imagenet2012/devkit/')

    train_loader, val_loader = data_loader(data_dir, batch_size=4)


    
    # Init Net
    net = ConvNet(in_channels = 3, out_channels = 10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



    # Training Loop
    epochs = 70
    print("-----Training Loop Started-----")
    for epoch in range(epochs):  

        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # input_type = type(inputs)

            optimizer.zero_grad()
            
            # inputs = resize_input(inputs).to(DEVICE)
            # assert(type(inputs) == input_type, "Input type changed")
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print 
            running_loss += float(loss)
            if i % 200 == 199:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
	
	if ((epoch % 10) == 0) :   
		torch.save(net.state_dict(), f'~/Courses/SMAI_Project/Weights/Model{epoch}.pt')

    print('Finished Training')
    
    
    
if __name__ == '__main__':
    main()
