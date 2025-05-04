import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
import matplotlib.gridspec as gridspec

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the trained model
model = load_model('trained_model.h5')

# Print model summary and output shape
print("Model output shape:", model.output_shape)

# Image size from the model
IMG_SIZE = 150  # Updated to match the original successful model

# Load the training history
try:
    history = np.load('training_history.npz', allow_pickle=True)
    history = history['history'].item()
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot AUC
    if 'auc' in history and 'val_auc' in history:
        plt.subplot(1, 3, 3)
        plt.plot(history['auc'], label='Training AUC')
        plt.plot(history['val_auc'], label='Validation AUC')
        plt.title('Model AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_progress.png')
    print("Training progress graphs saved to 'results/training_progress.png'")
    plt.close()
except Exception as e:
    print(f"Error loading training history: {str(e)}")
    print("Only model testing will be performed.")

# Test the model on individual random images
test_dir = 'test'
class_dirs = []
for d in os.listdir(test_dir):
    if os.path.isdir(os.path.join(test_dir, d)):
        class_dirs.append(d)

if not class_dirs:
    print("No class directories found in test folder!")
else:
    # Map class names - adjust for the 6 classes in the model output
    # Infer the appropriate class names from the test directory structure
    print("Found class directories:", class_dirs)
    
    # Map the 6 output classes - adjust based on your specific data structure
    # Let's print a sample prediction to see the actual output dimensions
    sample_dir = class_dirs[0]
    sample_path = os.path.join(test_dir, sample_dir)
    sample_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if sample_files:
        sample_img_path = os.path.join(sample_path, sample_files[0])
        img = load_img(sample_img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        sample_pred = model.predict(img_array, verbose=0)
        print("Sample prediction shape:", sample_pred.shape)
        print("Sample prediction values:", sample_pred[0])
    
    # Define 6 classes - update with your specific class names
    class_names = [
        'cancer-coronal',
        'cancer-sagittal', 
        'cancer-transverse',
        'normal-coronal',
        'normal-sagittal',
        'normal-transverse'
    ]
    
    # Number of test images to display
    num_test_images = 6
    plt.figure(figsize=(15, 4 * num_test_images))
    
    # Track accuracy
    correct_count = 0
    sample_count = 0
    
    for idx in range(num_test_images):
        # Pick a random class directory
        random_class_dir = random.choice(class_dirs)
        class_path = os.path.join(test_dir, random_class_dir)
        
        # Get image files
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No image files found in {class_path}")
            continue
        
        random_image = random.choice(image_files)
        image_path = os.path.join(class_path, random_image)
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get the predicted class
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        predicted_class = class_names[class_idx]
        
        # Parse the actual class from directory name
        actual_class = random_class_dir
        
        # Extract the simple cancer/normal status and view for display
        pred_type = "cancer" if "cancer" in predicted_class else "normal"
        pred_view = predicted_class.split('-')[1] if '-' in predicted_class else ""
        
        actual_type = "cancer" if "cancer" in actual_class.lower() else "normal"
        actual_view = ""
        for view in ['coronal', 'sagittal', 'transverse']:
            if view in actual_class.lower():
                actual_view = view
                break
        
        # Check if prediction is correct (based on cancer/normal and view)
        is_correct = (pred_type == actual_type) and (pred_view == actual_view)
        if is_correct:
            correct_count += 1
        sample_count += 1
        
        # Display the image and prediction
        plt.subplot(num_test_images, 2, idx*2+1)
        plt.imshow(plt.imread(image_path))
        title_color = "green" if is_correct else "red"
        plt.title(f"View: {actual_view}\nActual: {actual_type} | Predicted: {pred_type} ({confidence:.1f}%)", 
                 color=title_color, fontsize=12)
        plt.axis('off')
        
        # Display the probabilities
        plt.subplot(num_test_images, 2, idx*2+2)
        probs = [predictions[0][i] * 100 for i in range(len(class_names))]
        y_pos = np.arange(len(class_names))
        bars = plt.barh(y_pos, probs)
        plt.yticks(y_pos, [cn.replace('-', '\n') for cn in class_names], fontsize=8)
        plt.xlim(0, 100)
        plt.title("Class Probabilities (%)")
        
        # Color the bars
        for i, bar in enumerate(bars):
            if class_names[i] == predicted_class and is_correct:
                bar.set_color('green')
            elif class_names[i] == predicted_class and not is_correct:
                bar.set_color('red')
    
    plt.tight_layout(pad=3.0)
    plt.savefig('results/test_predictions.png')
    print(f"Test predictions saved to 'results/test_predictions.png'")
    print(f"Accuracy on sample: {correct_count}/{sample_count} ({(correct_count/sample_count)*100:.1f}%)")
    plt.close()
    
    # Show a detailed analysis of just one image
    # Pick a random class
    random_class_dir = random.choice(class_dirs)
    class_path = os.path.join(test_dir, random_class_dir)
    
    # Get image files
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        random_image = random.choice(image_files)
        image_path = os.path.join(class_path, random_image)
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get the predicted class
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100
        predicted_class = class_names[class_idx]
        
        # Parse the actual class from directory name
        actual_class = random_class_dir
        
        # Extract the simple cancer/normal status and view for display
        pred_type = "cancer" if "cancer" in predicted_class else "normal"
        pred_view = predicted_class.split('-')[1] if '-' in predicted_class else ""
        
        actual_type = "cancer" if "cancer" in actual_class.lower() else "normal"
        actual_view = ""
        for view in ['coronal', 'sagittal', 'transverse']:
            if view in actual_class.lower():
                actual_view = view
                break
        
        # Check if prediction is correct
        is_correct = (pred_type == actual_type) and (pred_view == actual_view)
        
        # Create a figure for detailed analysis
        plt.figure(figsize=(12, 8))
        
        # Display the image
        plt.subplot(1, 2, 1)
        plt.imshow(plt.imread(image_path))
        title_color = "green" if is_correct else "red"
        plt.title(f"View type: {actual_view}", fontsize=14)
        plt.axis('off')
        
        # Display the probability bar chart
        plt.subplot(1, 2, 2)
        probs = [predictions[0][i] * 100 for i in range(len(class_names))]
        y_pos = np.arange(len(class_names))
        bars = plt.barh(y_pos, probs)
        plt.yticks(y_pos, [cn.replace('-', '\n') for cn in class_names], fontsize=9)
        plt.xlabel('Probability (%)')
        plt.xlim(0, 100)
        plt.title('Class Probabilities', fontsize=14)
        
        # Color the bars
        for i, bar in enumerate(bars):
            if class_names[i] == predicted_class and is_correct:
                bar.set_color('green')
            elif class_names[i] == predicted_class and not is_correct:
                bar.set_color('red')
            elif actual_type in class_names[i] and actual_view in class_names[i]:
                bar.set_color('blue')  # The correct class
        
        # Add an analysis text box
        if is_correct:
            analysis = f"✅ CORRECT PREDICTION\n\n"
        else:
            analysis = f"❌ INCORRECT PREDICTION\n\n"
            
        analysis += f"The AI analyzed this {actual_view} scan and"
        analysis += f" identified it as {pred_type} with {confidence:.1f}% confidence.\n\n"
        analysis += f"Actual diagnosis: {actual_type} (view: {actual_view})"
        
        if is_correct:
            if confidence > 95:
                analysis += f"\n\nThe AI is very confident in this correct prediction, which suggests it has learned strong patterns associated with {actual_type} in {actual_view} scans."
            else:
                analysis += f"\n\nThe AI correctly identified this {actual_type}, but with moderate confidence ({confidence:.1f}%)."
        else:
            analysis += f"\n\nThe AI incorrectly classified this {actual_type} as {pred_type}. This could be due to:"
            analysis += f"\n- Unusual presentation of {actual_type}"
            analysis += f"\n- Similarities between {actual_type} and {pred_type} in this particular view"
            analysis += f"\n- Limited training examples of similar cases"
            
        plt.figtext(0.5, 0.01, analysis, ha="center", fontsize=12, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.savefig('results/detailed_analysis.png')
        print(f"Detailed analysis saved to 'results/detailed_analysis.png'")
        
        # Print the analysis
        print("\n" + "="*50)
        print("DETAILED ANALYSIS OF SAMPLE IMAGE")
        print("="*50)
        print(analysis)
        print("="*50)
        
        plt.close()

print("Visualization completed!") 