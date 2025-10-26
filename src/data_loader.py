import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_data_loaders(batch_size=128, data_dir='./data'):
    """
    Create and return data loaders for CIFAR-10 dataset
    """
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])

    # Just normalization for validation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])

    # Download and load training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    # Download and load test dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

def get_sample_data_loader(batch_size=64):
    """
    Get a smaller data loader for quick testing
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Use only a subset for quick testing
    subset = torch.utils.data.Subset(dataset, range(1000))
    return DataLoader(subset, batch_size=batch_size, shuffle=True)