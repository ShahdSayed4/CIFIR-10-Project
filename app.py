#!/usr/bin/env python3
"""
Simple web interface for CIFAR-10 image classification
"""

from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model import create_model
from src.utils import get_device, load_model
import io
import base64
import numpy as np

app = Flask(__name__)

# Global variables
model = None
device = None
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def load_model_once():
    global model, device
    if model is None:
        device = get_device()
        model = create_model(device=device)
        load_model(model, 'models/saved_models/best_model.pth', device=device)
        model.eval()

def preprocess_image(image):
    """Preprocess image for model prediction"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010)),
    ])
    return transform(image).unsqueeze(0).to(device)

def predict_image(image_tensor):
    """Make prediction on image tensor"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIFAR-10 Image Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-form { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .result { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .prediction { font-size: 1.2em; font-weight: bold; color: #2c3e50; }
            .confidence { color: #27ae60; }
            img { max-width: 200px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CIFAR-10 Image Classifier</h1>
            <p>Upload an image to classify it into one of 10 CIFAR-10 categories</p>
            
            <div class="upload-form">
                <form action="/predict" method="post" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*" required>
                    <input type="submit" value="Classify Image">
                </form>
            </div>
            
            <div>
                <h3>Supported Classes:</h3>
                <ul>
                    <li>✈️ Airplane</li>
                    <li>🚗 Automobile</li>
                    <li>🐦 Bird</li>
                    <li>🐱 Cat</li>
                    <li>🦌 Deer</li>
                    <li>🐕 Dog</li>
                    <li>🐸 Frog</li>
                    <li>🐴 Horse</li>
                    <li>🚢 Ship</li>
                    <li>🚚 Truck</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    load_model_once()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    try:
        # Open and process image
        image = Image.open(file.stream).convert('RGB')
        image_tensor = preprocess_image(image)
        
        # Make prediction
        predicted_class, confidence, probabilities = predict_image(image_tensor)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {'class': classes[i], 'probability': float(probabilities[i])}
            for i in top3_indices
        ]
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .result {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .prediction {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
                .confidence {{ color: #27ae60; font-size: 1.2em; }}
                img {{ max-width: 200px; margin: 10px 0; border: 1px solid #ddd; }}
                .back-btn {{ margin: 20px 0; padding: 10px 20px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prediction Result</h1>
                <button class="back-btn" onclick="window.history.back()">← Back</button>
                
                <div class="result">
                    <img src="data:image/png;base64,{img_str}" alt="Uploaded Image">
                    <div class="prediction">Predicted: {classes[predicted_class]}</div>
                    <div class="confidence">Confidence: {confidence:.2%}</div>
                    
                    <h3>Top Predictions:</h3>
                    <ol>
                        {"".join(f'<li>{pred["class"]}: {pred["probability"]:.2%}</li>' for pred in top_predictions)}
                    </ol>
                </div>
            </div>
        </body>
        </html>
        '''
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model_once()
    app.run(debug=True, host='0.0.0.0', port=5000)
