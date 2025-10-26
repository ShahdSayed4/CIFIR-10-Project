#!/usr/bin/env python3
"""
Batch prediction script for multiple images
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
from src.model import create_model
from src.utils import get_device, load_model
import argparse
import os
import glob
import pandas as pd
from tqdm import tqdm

def predict_images_in_folder(model, folder_path, device, classes):
    """
    Predict classes for all images in a folder
    """
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {folder_path}")
        return []
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    
    results = []
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get top 3 predictions
            probs_np = probabilities.cpu().numpy()[0]
            top3_indices = probs_np.argsort()[-3:][::-1]
            top3_classes = [classes[i] for i in top3_indices]
            top3_probs = [probs_np[i] for i in top3_indices]
            
            results.append({
                'image_path': image_path,
                'predicted_class': classes[predicted.item()],
                'confidence': confidence.item(),
                'top1_class': top3_classes[0],
                'top1_prob': top3_probs[0],
                'top2_class': top3_classes[1],
                'top2_prob': top3_probs[1],
                'top3_class': top3_classes[2],
                'top3_prob': top3_probs[2]
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({
                'image_path': image_path,
                'predicted_class': 'ERROR',
                'confidence': 0.0,
                'top1_class': 'ERROR',
                'top1_prob': 0.0,
                'top2_class': 'ERROR',
                'top2_prob': 0.0,
                'top3_class': 'ERROR',
                'top3_prob': 0.0
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Batch prediction for CIFAR-10 images')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing images')
    parser.add_argument('--model_path', type=str, default='models/saved_models/best_model.pth', 
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file path')
    
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
    
    # Process images
    print(f"Processing images in: {args.folder_path}")
    results = predict_images_in_folder(model, args.folder_path, device, classes)
    
    if results:
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"BATCH PREDICTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total images processed: {len(results)}")
        print(f"Successful predictions: {len([r for r in results if r['predicted_class'] != 'ERROR'])}")
        print(f"Failed predictions: {len([r for r in results if r['predicted_class'] == 'ERROR'])}")
        
        # Show class distribution
        if len(results) > 0:
            class_counts = df[df['predicted_class'] != 'ERROR']['predicted_class'].value_counts()
            print(f"\nClass distribution:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} images")
    
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
