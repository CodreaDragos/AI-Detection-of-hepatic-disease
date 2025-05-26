import os
import shutil
import pandas as pd
import pydicom
import numpy as np
from sklearn.model_selection import train_test_split

# Read the Excel file with categories
categories_df = pd.read_excel('hepatica_categories.xlsx')

# Create a mapping for text categories to numbers
category_mapping = {
    'ficat omogen': 0,
    'TU hepatic (sau determinari secundare)': 1,
    'hemangiom hepatic': 2,
    'chist hepatic': 3
}

# Function to convert category to number
def get_category_number(category):
    if pd.isna(category):
        return None
    if isinstance(category, (int, np.integer)):
        return category
    if isinstance(category, str):
        if ',' in category:
            # For cases like '2,3', take the first number
            return int(category.split(',')[0])
        return category_mapping.get(category)
    return None

# Print unique categories to understand the data
print("Unique categories in the Excel file:")
print(categories_df['tip'].unique())

# Create a dictionary mapping folder IDs to their categories
id_to_category = {}
for idx, row in categories_df.iterrows():
    category = get_category_number(row['tip'])
    if category is not None:
        id_to_category[row['id']] = category

# Define category names for better readability
category_names = {
    0: '0_homogeneous',
    1: '1_tumor',
    2: '2_hemangioma',
    3: '3_cyst'
}

def normalize_dicom_image(img_array):
    """Normalize the image array to 0-255 range."""
    if img_array.min() == img_array.max():
        return None
    img_array = img_array.astype(float)
    img_array = ((img_array - img_array.min()) * 255.0 / (img_array.max() - img_array.min())).astype(np.uint8)
    return img_array

# Function to process a DICOM file and save it as PNG
def process_dicom(dicom_path, output_path):
    try:
        dicom = pydicom.dcmread(dicom_path, force=True)
        if not hasattr(dicom, 'file_meta'):
            dicom.file_meta = pydicom.dataset.FileMetaDataset()
        # Check for pixel data
        if not hasattr(dicom, 'PixelData'):
            print(f"Skipping {dicom_path}: No pixel data found.")
            return False
        # Try different transfer syntaxes
        transfer_syntaxes = [
            pydicom.uid.ImplicitVRLittleEndian,
            pydicom.uid.ExplicitVRLittleEndian,
            pydicom.uid.DeflatedExplicitVRLittleEndian,
            pydicom.uid.ExplicitVRBigEndian,
            '1.2.840.10008.1.2.4.50',  # JPEG Baseline
            '1.2.840.10008.1.2.4.51',  # JPEG Extended
            '1.2.840.10008.1.2.4.57',  # JPEG Lossless
            '1.2.840.10008.1.2.4.70',  # JPEG Lossless First Order
            '1.2.840.10008.1.2.4.80',  # JPEG-LS Lossless
            '1.2.840.10008.1.2.4.81',  # JPEG-LS Lossy
            '1.2.840.10008.1.2.4.90',  # JPEG 2000 Lossless
            '1.2.840.10008.1.2.4.91',  # JPEG 2000 Lossy
        ]
        success = False
        last_error = None
        for syntax in transfer_syntaxes:
            try:
                dicom.file_meta.TransferSyntaxUID = syntax
                img_array = dicom.pixel_array
                success = True
                break
            except Exception as e:
                last_error = str(e)
                continue
        if not success:
            print(f"Failed to decode {dicom_path}: {last_error}")
            return False
        normalized_array = normalize_dicom_image(img_array)
        if normalized_array is not None:
            from PIL import Image
            img = Image.fromarray(normalized_array)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            return True
        else:
            print(f"Warning: Zero contrast in {dicom_path}")
            return False
    except Exception as e:
        print(f"Error reading {dicom_path}: {str(e)}")
        return False

def main():
    studies_dir = 'studies'
    # Build a list of (patient_id, category) for all patients with a known category
    patient_ids = []
    patient_categories = []
    for folder_id in os.listdir(studies_dir):
        folder_path = os.path.join(studies_dir, folder_id)
        if not os.path.isdir(folder_path):
            continue
        try:
            folder_id_int = int(folder_id)
            category = id_to_category.get(folder_id_int)
            if category is not None:
                patient_ids.append(folder_id)
                patient_categories.append(category)
        except ValueError:
            continue
    # Split patients into train/valid/test (70/15/15)
    train_ids, temp_ids, train_cats, temp_cats = train_test_split(
        patient_ids, patient_categories, test_size=0.3, random_state=42, stratify=patient_categories
    )
    valid_ids, test_ids, valid_cats, test_cats = train_test_split(
        temp_ids, temp_cats, test_size=0.5, random_state=42, stratify=temp_cats
    )
    split_map = {}
    for pid in train_ids:
        split_map[pid] = 'train'
    for pid in valid_ids:
        split_map[pid] = 'valid'
    for pid in test_ids:
        split_map[pid] = 'test'
    print(f"\nSplit: {len(train_ids)} train, {len(valid_ids)} valid, {len(test_ids)} test patients.")
    # For each patient, copy all images to the correct split/category
    for folder_id in os.listdir(studies_dir):
        folder_path = os.path.join(studies_dir, folder_id)
        if not os.path.isdir(folder_path):
            continue
        split = split_map.get(folder_id)
        if split is None:
            print(f"Skipping patient {folder_id}: not in split map.")
            continue
        try:
            folder_id_int = int(folder_id)
            category = id_to_category.get(folder_id_int)
            if category is None:
                print(f"Warning: No category for patient {folder_id}")
                continue
            category_name = category_names[category]
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_dir = os.path.join(split, category_name)
                    output_path = os.path.join(output_dir, f"{folder_id}_{base_name}.png")
                    process_dicom(file_path, output_path)
        except ValueError:
            print(f"Warning: Skipping folder {folder_id}")
            continue
    print("\nPatient-level data organization complete!")

if __name__ == "__main__":
    main() 