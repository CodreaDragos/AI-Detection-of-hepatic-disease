# Brain Cancer Detection from Multi-View Medical Imaging

This project implements a deep learning solution for detecting brain cancer from medical images across different viewing angles (coronal, sagittal, and transverse).

## Model Performance

- Test Accuracy: 90.66%
- AUC Score: 0.9897
- Early stopping at epoch 68 with optimal validation accuracy

## Features

- Multi-class classification across 6 categories:
  - Cancer - Coronal view
  - Cancer - Sagittal view
  - Cancer - Transverse view
  - Normal - Coronal view
  - Normal - Sagittal view
  - Normal - Transverse view
- Data augmentation for improved model robustness
- Advanced visualization tools for result analysis
- Early stopping and learning rate scheduling
- Comprehensive performance metrics

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
- 20,643 training images
- 1,970 validation images
- 985 test images
Across 6 classes combining cancer/normal status with viewing angles.

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

The model uses a CNN architecture with:
- Multiple convolutional and max pooling layers
- Dropout for regularization
- Dense layers for classification
- Softmax output for 6-class prediction

## Performance Optimization

- Data augmentation (rotation, shifts, flips)
- Early stopping to prevent overfitting
- Dynamic learning rate scheduling
- GPU acceleration support

## Results Visualization

The visualization script provides:
- Training/validation metrics plots
- Confusion matrix
- Sample predictions with probability distributions
- Detailed analysis of individual predictions 