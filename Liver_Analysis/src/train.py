import os
import tensorflow as tf
from data.data_loader import prepare_dataset, LiverDataGenerator
from models.model import build_model, get_callbacks
from config import *

def main():
    # Enable mixed precision training
    if USE_MIXED_PRECISION:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Prepare the dataset
    print("Preparing dataset...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = prepare_dataset()

    # Create data generators
    train_generator = LiverDataGenerator(
        train_paths, 
        train_labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        is_training=True
    )

    val_generator = LiverDataGenerator(
        val_paths,
        val_labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        is_training=False
    )

    test_generator = LiverDataGenerator(
        test_paths,
        test_labels,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        is_training=False
    )

    # Build the model
    print("Building model...")
    model = build_model(input_shape=(*IMAGE_SIZE, 3))
    model.summary()

    # Get callbacks
    callbacks = get_callbacks()

    # Train the model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(
        test_generator,
        verbose=1
    )

    # Print test results
    metrics = ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']
    for metric, value in zip(metrics, test_results):
        print(f"Test {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 