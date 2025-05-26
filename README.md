# Liver Analysis and Diseases Classification

This repository contains two related projects for liver analysis and disease classification using deep learning techniques.

## Projects Overview

### 1. Liver Analysis
A deep learning project for liver analysis using EfficientNetB0 architecture. The project focuses on analyzing liver images and classifying them into different categories.

#### Features:
- Uses EfficientNetB0 as the base model
- Implements data augmentation techniques
- Includes comprehensive evaluation metrics
- Provides visualization tools for results

#### Performance Metrics:
- Training Accuracy: 95.32%
- Validation Accuracy: 89.47%
- Training AUC: 0.9912
- Validation AUC: 0.9723
- Test Accuracy: 88.92%
- Test AUC: 0.9685

#### Directory Structure:
```
liver_analysis/
├── test/
├── train/
├── valid/
├── train_model.py
├── evaluate_model.py
└── requirements.txt
```

### 2. Liver Diseases Classification
A deep learning project for classifying different types of liver diseases using EfficientNetB2 architecture. The project aims to identify and classify various liver conditions from medical images.

#### Features:
- Uses EfficientNetB2 as the base model
- Implements advanced data preprocessing
- Includes comprehensive evaluation metrics
- Provides detailed classification reports

#### Performance Metrics:
- Training Accuracy: 92.17%
- Validation Accuracy: 55.92%
- Training AUC: 0.9930
- Validation AUC: 0.8094
- Test Accuracy: 28.73%
- Test AUC: 0.5473

#### Directory Structure:
```
liver_diseases/
├── test/
├── train/
├── valid/
├── train_model.py
├── evaluate_model.py
├── visualize_results.py
└── requirements.txt
```

#### Visualization Features:
- Training metrics visualization (accuracy, loss, AUC)
- Confusion matrix generation
- Class distribution analysis
- Sample prediction visualization with probabilities
- Comprehensive evaluation reports

## Technical Details
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (1640MB memory)
- Image size: 128x128 (optimized for memory constraints)
- Python version: 3.10
- CUDA version: 11.0
- cuDNN version: 8.1.0

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/CodreaDragos/AI-Detection-of-hepatic-disease.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Liver Analysis
1. Navigate to the liver_analysis directory:
```bash
cd liver_analysis
```
2. Train the model:
```bash
python train_model.py
```
3. Evaluate the model:
```bash
python evaluate_model.py
```

### Liver Diseases Classification
1. Navigate to the liver_diseases directory:
```bash
cd liver_diseases
```
2. Train the model:
```bash
python train_model.py
```
3. Evaluate the model:
```bash
python evaluate_model.py
```

## Requirements
- Python 3.8+
- TensorFlow 2.12.0
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
