#!/usr/bin/env python3
"""
Evaluation script for CIFAR-10 classifier
"""

import torch
from src.data_loader import get_data_loaders
from src.model import create_model
from src.evaluate import (
    evaluate_model, 
    plot_training_history, 
    plot_confusion_matrix,
    plot_sample_predictions,
    generate_classification_report
)
from src.utils import get_device, load_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate CIFAR-10 Classifier')
    parser.add_argument('--model_path', type=str, default='models/saved_models/best_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Get data loaders
    _, test_loader, classes = get_data_loaders(batch_size=args.batch_size)
    
    # Create model and load weights
    model = create_model(device=device)
    epoch, history = load_model(model, args.model_path, device=device)
    
    print(f"Loaded model from epoch {epoch}")
    
    # Evaluate model
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, classes, device)
    
    # Generate plots and reports
    if history:
        plot_training_history(history)
    
    plot_confusion_matrix(y_true, y_pred, classes)
    plot_sample_predictions(model, test_loader, classes, device)
    report = generate_classification_report(y_true, y_pred, classes)
    
    # Calculate overall accuracy
    accuracy = (y_pred == y_true).mean() * 100
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()