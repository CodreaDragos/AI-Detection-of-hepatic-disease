import os

# Data configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'studies')
IMAGE_SIZE = (224, 224)  # Standard size for EfficientNet
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model configuration
NUM_CLASSES = 4
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.1
LEARNING_RATE_PATIENCE = 5

# Class mapping
CLASS_MAPPING = {
    0: 'Homogeneous Liver',
    1: 'Liver Tumor',
    2: 'Liver Hemangioma',
    3: 'Liver Cyst'
}

# Training configuration
RANDOM_SEED = 42
USE_MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 10

# Augmentation parameters
AUGMENTATION_PARAMS = {
    'rotate_limit': 20,
    'shift_limit': 0.2,
    'scale_limit': 0.2,
    'brightness_contrast_limit': 0.2,
    'prob': 0.5
} 