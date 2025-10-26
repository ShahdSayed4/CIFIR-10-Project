#!/usr/bin/env python3
"""
Training script for CIFAR-10 classifier
"""

import torch
from src.data_loader import get_data_loaders
from src.model import create_model
from src.train import train_model
from src.utils import set_seed, get_device
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 Classifier')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed and device
    set_seed(args.seed)
    device = get_device()
    
    print(f"Using device: {device}")
    print(f"Training parameters: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    
    # Get data loaders
    train_loader, test_loader, classes = get_data_loaders(batch_size=args.batch_size)
    
    # Create model
    model = create_model(device=device)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()