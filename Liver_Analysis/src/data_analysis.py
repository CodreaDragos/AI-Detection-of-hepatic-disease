import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def analyze_dataset(base_dir='.'):
    """
    Analyze the dataset structure and display sample images from each directory
    """
    # Define dataset splits
    splits = ['train', 'test', 'valid']
    
    # Define view types
    view_types = ['coronal', 'sagittal', 'transverse']
    
    print("Analyzing dataset structure...")
    print("-" * 50)
    
    # Collect statistics for each split
    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split} directory not found!")
            continue
            
        print(f"\nAnalyzing {split} set:")
        print("-" * 30)
        
        # Count images in each category
        for view in view_types:
            cancer_path = os.path.join(split_path, f'cancer-{view}')
            normal_path = os.path.join(split_path, f'normal-{view}')
            
            if os.path.exists(cancer_path):
                cancer_count = len([f for f in os.listdir(cancer_path) 
                                  if f.lower().endswith('.jpg')])
                print(f"Cancer {view}: {cancer_count} images")
                
            if os.path.exists(normal_path):
                normal_count = len([f for f in os.listdir(normal_path) 
                                  if f.lower().endswith('.jpg')])
                print(f"Normal {view}: {normal_count} images")

def predict_random_test_image(base_dir='.'):
    """
    Select a random image from the test set and predict its class
    """
    test_path = os.path.join(base_dir, 'test')
    if not os.path.exists(test_path):
        print("Test directory not found!")
        return
    
    # Get all view types
    view_types = ['coronal', 'sagittal', 'transverse']
    
    # Randomly select a view type
    view = random.choice(view_types)
    
    # Randomly select between cancer and normal
    category = random.choice(['cancer', 'normal'])
    
    # Get the path to the selected category
    category_path = os.path.join(test_path, f'{category}-{view}')
    
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
        # Read and display the image
        img = Image.open(image_path)
        img = np.array(img)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray')
        plt.title(f"Test Image\nCategory: {category}\nView: {view}")
        plt.axis('off')
        
        # Save the sample image
        os.makedirs('sample_images', exist_ok=True)
        output_path = f'sample_images/test_sample_{category}_{view}.png'
        plt.savefig(output_path)
        plt.close()
        
        print(f"\nSelected image: {random_image}")
        print(f"Category: {category}")
        print(f"View: {view}")
        print(f"Sample saved as: {output_path}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def main():
    # First analyze the dataset structure
    analyze_dataset()
    
    # Then predict a random test image
    print("\nSelecting and displaying a random test image...")
    predict_random_test_image()
    
    print("\nThe dataset is organized into three splits (train, test, valid) with:")
    print("1. Cancer images (coronal, sagittal, transverse views)")
    print("2. Normal images (coronal, sagittal, transverse views)")

if __name__ == "__main__":
    main() 