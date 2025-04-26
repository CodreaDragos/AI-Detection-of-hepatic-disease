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

# Define the proper liver pathology classes
LIVER_CLASSES = [
    'Homogeneous Liver',   # Normal liver
    'Liver Tumor',         # Secondary determinations
    'Liver Hemangioma',    
    'Liver Cyst'
]

# If model outputs 6 classes (cancer/normal + view types)
# but we need 4 classes (liver pathologies), we'll need to adapt

# Check if an existing class mapping file exists
class_mapping_file = 'class_mapping.npy'
if os.path.exists(class_mapping_file):
    class_mapping = np.load(class_mapping_file, allow_pickle=True).item()
    print("Loaded existing class mapping:", class_mapping)
else:
    # If your model actually has 6 outputs but you need 4 classes,
    # you would need to create a mapping here
    # For now, we'll assume the model directly outputs the 4 classes we need
    class_mapping = {i: cls for i, cls in enumerate(LIVER_CLASSES)}
    print("Created default class mapping:", class_mapping)

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
    print("Found test directories:", class_dirs)
    
    # Sample prediction to check the output format
    sample_dir = class_dirs[0]
    sample_path = os.path.join(test_dir, sample_dir)
    sample_files = [f for f in os.listdir(sample_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if sample_files:
        print(f"Testing a sample image from {sample_dir}")
        sample_img_path = os.path.join(sample_path, sample_files[0])
        img = load_img(sample_img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        sample_pred = model.predict(img_array, verbose=1)
        print("Sample prediction shape:", sample_pred.shape)
        print("Sample prediction values:", sample_pred[0])
        
        # Get the predicted class
        pred_idx = np.argmax(sample_pred[0])
        confidence = sample_pred[0][pred_idx] * 100
        
        # Map model output to liver class names
        # If model output dimension doesn't match our 4 classes, 
        # we need to handle that discrepancy
        num_model_outputs = len(sample_pred[0])
        print(f"Model has {num_model_outputs} outputs, we defined {len(LIVER_CLASSES)} liver classes")
        
        # For visualization, use the number of classes that the model actually outputs
        if num_model_outputs == 6:  # If model has 6 outputs (cancer/normal + 3 views)
            class_names = [
                'cancer-coronal', 'cancer-sagittal', 'cancer-transverse',
                'normal-coronal', 'normal-sagittal', 'normal-transverse'
            ]
            print("Using the original 6-class model outputs (cancer/normal + views)")
        else:  # Use liver pathology classes
            class_names = LIVER_CLASSES
            print(f"Using the {len(class_names)} liver pathology classes")
        
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
        
        # Map to appropriate class name
        if len(class_names) == len(predictions[0]):
            predicted_class = class_names[class_idx]
        else:
            # If dimensions don't match, use a fallback
            predicted_class = f"Class {class_idx}"
            
        # Determine actual class from directory name
        # This mapping will depend on how your test folders are organized
        if 'cancer' in random_class_dir.lower() or 'tumor' in random_class_dir.lower():
            actual_class = 'Liver Tumor'
        elif 'hemangioma' in random_class_dir.lower():
            actual_class = 'Liver Hemangioma'
        elif 'cyst' in random_class_dir.lower():
            actual_class = 'Liver Cyst'
        else:
            actual_class = 'Homogeneous Liver'  # Normal/homogeneous
            
        # If your folders are actually already named according to the 4 classes,
        # you can simplify this mapping
        for liver_class in LIVER_CLASSES:
            if liver_class.lower() in random_class_dir.lower():
                actual_class = liver_class
                break
        
        # Check if prediction is correct
        is_correct = predicted_class == actual_class
        if is_correct:
            correct_count += 1
        sample_count += 1
        
        # Display the image and prediction
        plt.subplot(num_test_images, 2, idx*2+1)
        plt.imshow(plt.imread(image_path))
        title_color = "green" if is_correct else "red"
        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\n({confidence:.1f}%)", 
                 color=title_color, fontsize=12)
        plt.axis('off')
        
        # Display the probabilities
        plt.subplot(num_test_images, 2, idx*2+2)
        probs = [predictions[0][i] * 100 for i in range(len(predictions[0]))]
        y_pos = np.arange(len(class_names[:len(predictions[0])]))
        bars = plt.barh(y_pos, probs)
        plt.yticks(y_pos, class_names[:len(predictions[0])], fontsize=10)
        plt.xlim(0, 100)
        plt.title("Class Probabilities (%)")
        
        # Color the bars
        for i, bar in enumerate(bars):
            if class_names[i] == predicted_class and is_correct:
                bar.set_color('green')
            elif class_names[i] == predicted_class and not is_correct:
                bar.set_color('red')
            elif class_names[i] == actual_class:
                bar.set_color('blue')  # Highlight actual class
    
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
        
        # Map to appropriate class name
        if len(class_names) == len(predictions[0]):
            predicted_class = class_names[class_idx]
        else:
            # If dimensions don't match, use a fallback
            predicted_class = f"Class {class_idx}"
            
        # Determine actual class from directory name
        if 'cancer' in random_class_dir.lower() or 'tumor' in random_class_dir.lower():
            actual_class = 'Liver Tumor'
        elif 'hemangioma' in random_class_dir.lower():
            actual_class = 'Liver Hemangioma'
        elif 'cyst' in random_class_dir.lower():
            actual_class = 'Liver Cyst'
        else:
            actual_class = 'Homogeneous Liver'  # Normal/homogeneous
            
        # If your folders are actually already named according to the 4 classes,
        # you can simplify this mapping
        for liver_class in LIVER_CLASSES:
            if liver_class.lower() in random_class_dir.lower():
                actual_class = liver_class
                break
        
        # Create a figure for detailed analysis
        plt.figure(figsize=(12, 8))
        
        # Display the image
        plt.subplot(1, 2, 1)
        plt.imshow(plt.imread(image_path))
        is_correct = predicted_class == actual_class
        title_color = "green" if is_correct else "red"
        plt.title(f"Liver Image Analysis", fontsize=14)
        plt.axis('off')
        
        # Display the probability bar chart
        plt.subplot(1, 2, 2)
        probs = [predictions[0][i] * 100 for i in range(len(predictions[0]))]
        y_pos = np.arange(len(class_names[:len(predictions[0])]))
        bars = plt.barh(y_pos, probs)
        plt.yticks(y_pos, class_names[:len(predictions[0])], fontsize=10)
        plt.xlabel('Probability (%)')
        plt.xlim(0, 100)
        plt.title('Class Probabilities', fontsize=14)
        
        # Color the bars
        for i, bar in enumerate(bars):
            if class_names[i] == predicted_class and is_correct:
                bar.set_color('green')
            elif class_names[i] == predicted_class and not is_correct:
                bar.set_color('red')
            elif class_names[i] == actual_class:
                bar.set_color('blue')  # Highlight actual class
        
        # Add an analysis text box
        if is_correct:
            analysis = f"✅ CORRECT PREDICTION\n\n"
        else:
            analysis = f"❌ INCORRECT PREDICTION\n\n"
            
        analysis += f"The AI analyzed this liver image and"
        analysis += f" identified it as '{predicted_class}' with {confidence:.1f}% confidence.\n\n"
        analysis += f"Actual diagnosis: {actual_class}"
        
        if is_correct:
            if confidence > 95:
                analysis += f"\n\nThe AI is very confident in this correct prediction, which suggests it has learned strong patterns associated with {actual_class}."
            else:
                analysis += f"\n\nThe AI correctly identified this as {actual_class}, but with moderate confidence ({confidence:.1f}%)."
        else:
            analysis += f"\n\nThe AI incorrectly classified this {actual_class} as {predicted_class}. This could be due to:"
            analysis += f"\n- Unusual presentation of {actual_class}"
            analysis += f"\n- Similarities between {actual_class} and {predicted_class} features"
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