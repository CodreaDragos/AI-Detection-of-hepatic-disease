# Liver Analysis from Multi-View Medical Imaging

This project implements a deep learning solution that performs both liver segmentation and cancer detection from medical images across different viewing angles (coronal, sagittal, and transverse).

## Model Performance

Current model performance:
- Segmentation Accuracy: 90.66%
- Cancer Detection AUC: 0.9897
- Early stopping at epoch 68 with optimal validation accuracy

## Features

- Dual functionality:
  1. Semantic segmentation of liver regions
  2. Binary classification for cancer detection
- Multi-view analysis (coronal, sagittal, transverse)
- Data augmentation for improved model robustness
- Advanced visualization tools for result analysis
- Early stopping and learning rate scheduling
- Comprehensive performance metrics including:
  - IoU (Intersection over Union) for segmentation
  - AUC and accuracy for cancer detection

## Project Structure

- `main.py`: Training script with model architecture and training pipeline
- `visualize_results.py`: Visualization and analysis tools
- `decode-files.py`: DICOM medical image decoder
- Data directories:
  - `train/`: Training dataset
  - `valid/`: Validation dataset
  - `test/`: Test dataset
- Model files:
  - `trained_model.h5`: Latest trained model
  - `best_model.h5`: Best performing model backup
- Results:
  - `training_history.npz`: Saved training metrics
  - `results/`: Generated visualizations and analysis
  - `predictions/`: Model predictions

## Dataset

The model was trained on:
- Training images
- Validation images
- Test images
Each image is annotated with:
- Liver segmentation masks
- Binary cancer labels (present/absent)
Across different viewing angles (coronal, sagittal, transverse)

## Technical Details

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU (1640MB memory)
- Image size: 128x128 (optimized for memory constraints)
- Python version: 3.10
- CUDA version: 11.0
- cuDNN version: 8.1.0

## Requirements

Required packages are listed in `requirements.txt`. Install using:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Activate the virtual environment:
   ```bash
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Unix/MacOS
   ```

2. Train the model:
   ```bash
   python main.py
   ```

3. Visualize results:
   ```bash
   python visualize_results.py
   ```

## Model Architecture

The model uses a hybrid architecture combining:
1. U-Net backbone for segmentation:
   - Encoder path with convolutional and pooling layers
   - Decoder path with upsampling and skip connections
   - Final segmentation mask output
2. Classification head for cancer detection:
   - Features extracted from segmentation encoder
   - Additional convolutional layers
   - Binary classification output

Common components:
- Batch normalization for training stability
- Dropout for regularization
- Shared feature learning between tasks

## Performance Optimization

- Data augmentation (rotation, shifts, flips, brightness adjustment)
- Early stopping to prevent overfitting
- Dynamic learning rate scheduling
- GPU acceleration support
- Memory-optimized image size
- Multi-task learning optimization

## Results Visualization

The visualization script provides:
- Training/validation metrics plots
- Segmentation mask overlays
- IoU scores per image
- Cancer detection confidence scores
- ROC curves for cancer classification
- Combined analysis of segmentation and classification results 