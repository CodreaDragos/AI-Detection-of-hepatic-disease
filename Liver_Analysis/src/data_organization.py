import os
import shutil
from pathlib import Path

def organize_dataset(source_dir='studies', dest_dir='organized_dataset'):
    """
    Organize the dataset into the four categories:
    0: Homogeneous Liver
    1: Liver Tumor
    2: Liver Hemangioma
    3: Liver Cyst
    """
    # Create destination directories
    categories = ['0_homogeneous', '1_tumor', '2_hemangioma', '3_cyst']
    for category in categories:
        os.makedirs(os.path.join(dest_dir, category), exist_ok=True)

    # Process each directory in the source
    for dir_name in os.listdir(source_dir):
        dir_path = os.path.join(source_dir, dir_name)
        if os.path.isdir(dir_path) and dir_name.isdigit():
            # Process each image in the directory
            for img_name in os.listdir(dir_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Here you'll need to implement the logic to determine
                    # which category each image belongs to
                    # For now, we'll just print the files we find
                    print(f"Found image: {os.path.join(dir_path, img_name)}")

def main():
    # Create the organized dataset directory
    organize_dataset()
    print("Dataset organization complete!")

if __name__ == "__main__":
    main() 