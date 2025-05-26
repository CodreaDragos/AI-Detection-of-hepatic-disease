import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os
import sys
import random
from tensorflow.keras.preprocessing import image

# Add focal loss function
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_mean(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + tf.keras.backend.epsilon()) + \
                      (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + tf.keras.backend.epsilon()))
    return focal_loss_fixed

def load_training_history():
    """Load the training history from the saved numpy file."""
    try:
        history = np.load('evaluation_results/training_history.npy', allow_pickle=True).item()
        return history
    except FileNotFoundError:
        print("Error: Could not find training history file at 'evaluation_results/training_history.npy'")
        print("Please make sure you have run the training script first.")
        sys.exit(1)

def plot_training_metrics(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['accuracy'], label='Training', color='blue')
    plt.plot(history['val_accuracy'], label='Validation', color='red')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Training', color='blue')
    plt.plot(history['val_loss'], label='Validation', color='red')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(history['auc'], label='Training', color='blue')
    plt.plot(history['val_auc'], label='Validation', color='red')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution(y_true, class_names):
    """Plot class distribution in the dataset."""
    plt.figure(figsize=(10, 6))
    class_counts = np.bincount(y_true)
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('evaluation_results/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def load_model():
    """Try to load the model from different possible locations."""
    possible_paths = [
        'evaluation_results/trained_model_resnet50.h5',
        'evaluation_results/trained_model_resnet50',
        'trained_model_resnet50.h5',
        'trained_model_resnet50'
    ]
    
    # Custom objects dictionary for loading the model
    custom_objects = {
        'focal_loss_fixed': focal_loss(gamma=2.)
    }
    
    for path in possible_paths:
        try:
            print(f"Attempting to load model from: {path}")
            if os.path.exists(path):
                model = tf.keras.models.load_model(path, custom_objects=custom_objects)
                print(f"Successfully loaded model from: {path}")
                return model
            else:
                print(f"File not found: {path}")
        except Exception as e:
            print(f"Error loading from {path}: {str(e)}")
            continue
    
    print("\nError: Could not find or load the trained model file.")
    print("Please make sure you have run the training script first and the model was saved correctly.")
    print("The script looked for the model in the following locations:")
    for path in possible_paths:
        print(f"- {path}")
    print("\nChecking if evaluation_results directory exists...")
    if os.path.exists('evaluation_results'):
        print("Contents of evaluation_results directory:")
        for file in os.listdir('evaluation_results'):
            print(f"- {file}")
    else:
        print("evaluation_results directory does not exist!")
    sys.exit(1)

def plot_sample_predictions(model, test_generator, class_names, num_samples=5, output_path='evaluation_results/sample_predictions.png'):
    """Visualize random test images with true/predicted labels and class probabilities."""
    # Get a list of all test image file paths and their true labels
    filepaths = test_generator.filepaths
    y_true = test_generator.classes
    indices = list(range(len(filepaths)))
    random_samples = random.sample(indices, min(num_samples, len(indices)))

    plt.figure(figsize=(10, num_samples * 3))
    for i, idx in enumerate(random_samples):
        # Load image
        img_path = filepaths[idx]
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_exp = np.expand_dims(img_array, axis=0)
        img_array_exp = tf.keras.applications.resnet50.preprocess_input(img_array_exp)

        # Predict
        preds = model.predict(img_array_exp, verbose=0)[0]
        pred_class = np.argmax(preds)
        true_class = y_true[idx]

        # Plot image
        plt.subplot(num_samples, 2, 2 * i + 1)
        plt.imshow(np.array(img))
        plt.axis('off')
        correct = (pred_class == true_class)
        color = 'green' if correct else 'red'
        plt.title(f"Actual: {class_names[true_class]}\nPredicted: {class_names[pred_class]} ({preds[pred_class]*100:.1f}%)", color=color, fontsize=10)

        # Plot class probabilities
        plt.subplot(num_samples, 2, 2 * i + 2)
        bars = plt.barh(class_names, preds * 100, color=['green' if j == pred_class else 'gray' for j in range(len(class_names))])
        plt.xlim(0, 100)
        plt.xlabel('Class Probabilities (%)')
        plt.tight_layout()

    plt.suptitle('Sample Predictions (Random Test Images)', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions visualization saved at: {output_path}")

def main():
    # Create evaluation_results directory if it doesn't exist
    os.makedirs('evaluation_results', exist_ok=True)
    
    print("Loading training history...")
    try:
        history = load_training_history()
        print("Training history loaded successfully")
    except Exception as e:
        print(f"Error loading training history: {e}")
        sys.exit(1)
    
    print("Plotting training metrics...")
    plot_training_metrics(history)
    
    print("Loading model...")
    model = load_model()
    
    print("Loading test data...")
    # Load test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    
    try:
        test_generator = test_datagen.flow_from_directory(
            'test',
            target_size=(224, 224),
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )
    except Exception as e:
        print(f"\nError: Could not load test data: {str(e)}")
        print("Please make sure the 'test' directory exists and contains the test images.")
        sys.exit(1)
    
    print("Generating predictions...")
    # Get predictions
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    
    print("Plotting class distribution...")
    plot_class_distribution(y_true, class_names)
    
    # Print classification report
    print("\nClassification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Plot sample predictions
    print("Plotting sample predictions...")
    plot_sample_predictions(model, test_generator, class_names, num_samples=5)
    
    print("\nVisualization complete! Check the 'evaluation_results' directory for the generated plots.")

if __name__ == "__main__":
    main() 