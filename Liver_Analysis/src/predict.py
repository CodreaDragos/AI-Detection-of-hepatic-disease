import cv2
import numpy as np
import tensorflow as tf
from config import *
import argparse

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocess a single image for prediction
    """
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model_path, image_path):
    """
    Make a prediction for a single image
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get class name
    class_name = CLASS_MAPPING[predicted_class]
    
    return {
        'class_name': class_name,
        'class_id': predicted_class,
        'confidence': float(confidence),
        'probabilities': {CLASS_MAPPING[i]: float(prob) 
                         for i, prob in enumerate(predictions[0])}
    }

def main():
    parser = argparse.ArgumentParser(description='Predict liver image class')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to classify')
    
    args = parser.parse_args()
    
    try:
        result = predict_image(args.model, args.image)
        
        print("\nPrediction Results:")
        print(f"Predicted Class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        print("\nClass Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"{class_name}: {prob:.2%}")
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main() 