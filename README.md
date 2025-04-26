# AI Detection of Hepatic Disease

This project utilizes deep learning to detect and classify different hepatic (liver) diseases from medical images.

## Features

- Liver image classification into multiple categories:
  - Homogeneous Liver (Normal)
  - Liver Tumor
  - Liver Hemangioma
  - Liver Cyst
- Training pipeline for custom models
- Visualization tools for model analysis and results interpretation

## Project Structure

- `main.py`: Main application entry point
- `visualize_results.py`: Tools for visualizing model predictions and performance
- `src/`: Source code directory
  - `model.py`: Neural network model definition
  - `train.py`: Training pipeline
  - `predict.py`: Prediction utilities
  - `data_analysis.py`: Data analysis tools
  - `data_organization.py`: Dataset organization utilities
  - `config.py`: Configuration parameters
  - `train_and_predict.py`: Combined training and prediction pipeline

## Requirements

Required packages are listed in `requirements.txt`

```
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the appropriate format
2. Configure parameters in `src/config.py`
3. Train the model using:
   ```
   python src/train.py
   ```
4. Evaluate and visualize results:
   ```
   python visualize_results.py
   ```

## Note

Training data, test data, validation data, and results are not included in this repository due to NDA restrictions. 