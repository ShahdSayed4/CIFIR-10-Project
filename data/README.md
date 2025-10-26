# Data Directory

This directory contains the CIFAR-10 dataset.

## Automatic Download
The CIFAR-10 dataset will be automatically downloaded by PyTorch's torchvision 
when you run the training script for the first time.

## Files
- `cifar-10-batches-py/` - Contains the actual CIFAR-10 dataset files
  - This directory is automatically created and populated
  - It is excluded from Git via .gitignore due to size

## Dataset Information
- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images, 10,000 test images