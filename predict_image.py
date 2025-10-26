#!/usr/bin/env python3
"""
Prediction script for single image classification
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from src.model import create_model
from src.utils import get_device, load_model
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_image(image_path, device='cpu'):
    """
    Load and preprocess image for prediction
    """
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocessing same as validation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(device), image

def predict_single_image(model, image_tensor, classes):
    """
    Make prediction on single image
    """
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

def display_prediction(original_image, prediction, confidence, probabilities, classes, image_path):
    """
    Display the image and prediction results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(original_image)
    ax1.set_title(f'Prediction: {classes[prediction]}\nConfidence: {confidence:.2%}')
    ax1.axis('off')
    
    # Display probabilities
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probabilities')
    ax2.invert_yaxis()  # labels read top-to-bottom
    
    plt.tight_layout()
    
    # Save the result
    output_path = image_path.replace('.', '_prediction.')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Predict CIFAR-10 class for an image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='models/saved_models/best_model.pth', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # CIFAR-10 classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Load model
    model = create_model(device=device)
    epoch, history = load_model(model, args.model_path, device=device)
    print(f"Loaded model from epoch {epoch}")
    
    # Load and preprocess image
    image_tensor, original_image = load_and_preprocess_image(args.image_path, device)
    print(f"Loaded image: {args.image_path}")
    
    # Make prediction
    predicted_class, confidence, all_probabilities = predict_single_image(
        model, image_tensor, classes
    )
    
    # Display results
    print(f"\n{'='*50}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"Input image: {args.image_path}")
    print(f"Predicted class: {classes[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nTop 3 predictions:")
    
    # Get top 3 predictions
    top3_indices = np.argsort(all_probabilities)[-3:][::-1]
    for i, idx in enumerate(top3_indices):
        print(f"{i+1}. {classes[idx]}: {all_probabilities[idx]:.2%}")
    
    # Display visualization
    output_path = display_prediction(
        original_image, predicted_class, confidence, 
        all_probabilities, classes, args.image_path
    )
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
