# 🧠 **CIFAR-10 Image Classifier**

A **PyTorch-based deep learning project** for classifying images from the **CIFAR-10 dataset**.  
This project demonstrates how to build, train, and evaluate a **Convolutional Neural Network (CNN)** that can recognize objects in small color images.

---
# CIFAR-10 Image Classifier

A complete PyTorch implementation for classifying images from the CIFAR-10 dataset with training, evaluation, and prediction capabilities.

## Features

- **CNN Model**: Custom convolutional neural network for image classification
- **Training Pipeline**: Complete training loop with validation
- **Evaluation**: Comprehensive metrics and visualizations
- **Prediction**: Single image and batch prediction capabilities
- **Web Interface**: Flask-based web app for easy classification
- **Modular Code**: Well-organized, reusable code structure

## Project Structure

cifar-10-classifier/
├── src/ # Source code modules
│ ├── data_loader.py # Data loading and preprocessing
│ ├── model.py # CNN model definition
│ ├── train.py # Training functions
│ ├── evaluate.py # Evaluation and visualization
│ └── utils.py # Utility functions
├── models/ # Model checkpoints
│ └── saved_models/ # Trained models (gitignored)
├── data/ # Dataset directory
├── test_images/ # Test images for prediction
├── requirements.txt # Python dependencies
├── train_model.py # Training script
├── evaluate_model.py # Evaluation script
├── predict_image.py # Single image prediction
├── predict_batch.py # Batch prediction
├── app.py # Web interface
└── README.md # Project documentation

## ⚙️ **Setup Instructions**

**1️⃣ Create and Activate a Virtual Environment**

```bash
# Create virtual environment
python -m venv cifar10_env

# Activate environment
# On Linux/Mac:
source cifar10_env/bin/activate

# On Windows:
cifar10_env\Scripts\activate
```

---

**2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 **Usage**

**🔹 Train the Model**

To start training with default parameters:

```bash
python train_model.py
```

---

**🔹 Evaluate a Trained Model**

After training, evaluate the best saved model:

```bash
python evaluate_model.py --model_path models/saved_models/best_model.pth
```

---
**🔹Prediction**

 Single image prediction

```bash
python predict_image.py --image_path test_images/your_image.jpg
```

 Batch prediction

```bash
python predict_batch.py --folder_path test_images --output predictions.csv
```

**🔹Web Interface**

```bash
python app.py
# Open http://localhost:5000 in your browser
```


## 🧩 **Model Architecture**

The CNN model consists of:

- **3 convolutional blocks** with Batch Normalization and Dropout  
- **Max Pooling layers** for downsampling  
- **Fully connected layers** for final classification  
- Total parameters: **~1.2 million**

---

## 📊 **Results**

| Metric | Expected Performance |
|--------|----------------------|
| **Training Accuracy** | ~95% |
| **Test Accuracy** | ~85–88% |
| **Training Time (GPU)** | ~30 minutes |

---

## 🖼️ **Dataset: CIFAR-10**

The **CIFAR-10** dataset contains **60,000 color images (32×32 pixels)** divided into **10 classes**:

> ✈️ airplane, 🚗 automobile, 🐦 bird, 🐱 cat, 🦌 deer, 🐶 dog, 🐸 frog, 🐴 horse, 🚢 ship, 🚚 truck

- **Training set:** 50,000 images  
- **Test set:** 10,000 images  

Each image belongs to one of the ten mutually exclusive classes.

---

## 📚 **Technologies Used**

- **Python 3.x**  
- **PyTorch**  
- **NumPy**  
- **Matplotlib**  
- **torchvision**

---

## 💡 **Future Improvements**

- Add **data augmentation** for improved generalization  
- Implement **learning rate scheduling**  
- Visualize **Grad-CAM** for interpretability and explainability  
- Add **early stopping** and **checkpoint saving** for efficiency  

---

## 👩‍💻 **Author**

**Shahd Sayed**  
💼 *Data Scientist & AI Student | Deep Learning Enthusiast*  
📧 [\ Shahd Sayed LinkedIn \]](https://www.linkedin.com/in/shahd-sayed-/)

---

## 🏁 **License**

This project is licensed under the [MIT License](LICENSE).

---

## 🏷️ **Badges**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-~88%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
