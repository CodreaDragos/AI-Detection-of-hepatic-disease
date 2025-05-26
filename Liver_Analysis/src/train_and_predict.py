import os
import random
from model import LiverClassifier
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def train_model():
    # Create the classifier
    classifier = LiverClassifier()
    
    # Train the model
    print("Starting model training...")
    history = classifier.train(
        train_dir='train',
        validation_dir='valid',
        batch_size=32,
        epochs=50
    )
    
    # Plot training history
    classifier.plot_training_history(history)
    print("Training complete! Model saved as 'models/best_model.h5'")
    
    return classifier

def predict_random_test_image(classifier):
    # Get a random test image
    test_dir = 'test'
    view_types = ['coronal', 'sagittal', 'transverse']
    view = random.choice(view_types)
    category = random.choice(['cancer', 'normal'])
    
    category_path = os.path.join(test_dir, f'{category}-{view}')
    if not os.path.exists(category_path):
        print(f"Category path not found: {category_path}")
        return
    
    # Get all jpg files
    jpg_files = [f for f in os.listdir(category_path) if f.lower().endswith('.jpg')]
    if not jpg_files:
        print(f"No images found in {category_path}")
        return
    
    # Select a random image
    random_image = random.choice(jpg_files)
    image_path = os.path.join(category_path, random_image)
    
    try:
        # Make prediction
        predicted_class, confidence = classifier.predict(image_path)
        
        # Display the image and prediction
        img = Image.open(image_path)
        img = np.array(img)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f"Test Image\nTrue Category: {category}\nPredicted: {predicted_class}\nConfidence: {confidence:.2%}")
        plt.axis('off')
        
        # Save the prediction
        os.makedirs('predictions', exist_ok=True)
        output_path = f'predictions/test_prediction_{category}_{view}.png'
        plt.savefig(output_path)
        plt.close()
        
        print(f"\nSelected image: {random_image}")
        print(f"True Category: {category}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Prediction saved as: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # Train the model
    classifier = train_model()
    
    # Make predictions on random test images
    print("\nMaking predictions on random test images...")
    for _ in range(5):  # Predict 5 random images
        predict_random_test_image(classifier)

if __name__ == "__main__":
    main() 