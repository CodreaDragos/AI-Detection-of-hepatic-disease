import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
import albumentations as A
from ..config import *

class LiverDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=BATCH_SIZE, 
                 image_size=IMAGE_SIZE, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.augmentor = self._create_augmentor() if is_training else None
        self.indexes = np.arange(len(self.image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]

        X = np.zeros((len(batch_indexes), *self.image_size, 3), dtype=np.float32)
        y = np.zeros((len(batch_indexes), NUM_CLASSES), dtype=np.float32)

        for i, (path, label) in enumerate(zip(batch_paths, batch_labels)):
            img = self._load_image(path)
            if self.is_training and self.augmentor:
                transformed = self.augmentor(image=img)
                img = transformed['image']
            X[i] = img
            y[i, label] = 1

        return X, y

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255.0
        return img

    def _create_augmentor(self):
        return A.Compose([
            A.RandomRotate90(p=AUGMENTATION_PARAMS['prob']),
            A.Rotate(limit=AUGMENTATION_PARAMS['rotate_limit'], p=AUGMENTATION_PARAMS['prob']),
            A.HorizontalFlip(p=AUGMENTATION_PARAMS['prob']),
            A.VerticalFlip(p=AUGMENTATION_PARAMS['prob']),
            A.RandomBrightnessContrast(
                brightness_limit=AUGMENTATION_PARAMS['brightness_contrast_limit'],
                contrast_limit=AUGMENTATION_PARAMS['brightness_contrast_limit'],
                p=AUGMENTATION_PARAMS['prob']
            ),
            A.ShiftScaleRotate(
                shift_limit=AUGMENTATION_PARAMS['shift_limit'],
                scale_limit=AUGMENTATION_PARAMS['scale_limit'],
                rotate_limit=0,
                p=AUGMENTATION_PARAMS['prob']
            ),
        ])

def prepare_dataset():
    """
    Prepare the dataset by loading image paths and labels
    """
    image_paths = []
    labels = []
    
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(class_id))
        if os.path.exists(class_dir):
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_id)

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Shuffle the dataset
    indices = np.arange(len(image_paths))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
    image_paths = image_paths[indices]
    labels = labels[indices]

    # Split into train, validation, and test sets
    n_samples = len(image_paths)
    n_train = int(n_samples * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    n_val = int(n_samples * VALIDATION_SPLIT)

    train_paths = image_paths[:n_train]
    train_labels = labels[:n_train]
    
    val_paths = image_paths[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    
    test_paths = image_paths[n_train + n_val:]
    test_labels = labels[n_train + n_val:]

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) 