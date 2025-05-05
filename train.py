import os
import shutil
import zipfile
import subprocess
import pandas as pd
import json
import csv
from pathlib import Path

# Load Kaggle API key
with open('./kaggle.json', 'r') as f:
    kaggle_creds = json.load(f)
    API_KEY = kaggle_creds['key']  # Extract the API key

# Dataset URLs and paths
DATASET_CONFIG = [
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/rishianand/devanagari-character-set",
        "zip_path": "./data/devanagari_character_set.zip",
        "extract_dir": "./data/devanagari_character_set",
        "original_zip_file": "./data/devanagari_character_set/devanagari_character_set.zip"
    },
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/ashokpant/devanagari-character-dataset",
        "zip_path": "./data/devanagari_character_dataset.zip",
        "extract_dir": "./data/devanagari_character_dataset",
        "original_zip_file": "./data/devanagari_character_dataset/devanagari_character_dataset.zip"
    },
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/anurags397/hindi-mnist-data",
        "zip_path": "./data/hindi_mnist_data.zip",
        "extract_dir": "./data/hindi_mnist_data",
        "original_zip_file": "./data/hindi_mnist_data/hindi_mnist_data.zip"
    },
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/ashokpant/devanagari-character-dataset-large",
        "zip_path": "./data/devanagari_character_dataset_large.zip",
        "extract_dir": "./data/devanagari_character_dataset_large",
        "original_zip_file": "./data/devanagari_character_dataset_large/devanagari_character_dataset_large.zip"
    },
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/suvooo/hindi-character-recognition",
        "zip_path": "./data/hindi_character_recognition.zip",
        "extract_dir": "./data/hindi_character_recognition",
        "original_zip_file": "./data/hindi_character_recognition/hindi_character_recognition.zip"
    },
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/sabarinathan/handwritten-hindi-word-recognition",
        "zip_path": "./data/handwritten_hindi_word_recognition.zip",
        "extract_dir": "./data/handwritten_hindi_word_recognition",
        "original_zip_file": "./data/handwritten_hindi_word_recognition/handwritten_hindi_word_recognition.zip"
    }
]

def extract_nested_zips(zip_path, extract_dir):
    """Recursively extract nested ZIP files"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Check for nested ZIPs
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.zip'):
                nested_zip = os.path.join(root, file)
                nested_extract = os.path.splitext(nested_zip)[0]
                os.makedirs(nested_extract, exist_ok=True)
                extract_nested_zips(nested_zip, nested_extract)
                os.remove(nested_zip)  # Clean up nested ZIP after extraction

# Download and process datasets
for config in DATASET_CONFIG:
    try:
        if not os.path.exists(config["extract_dir"]):
            print(f"\nDownloading {config['url']}...")
            
            # Download with curl
            subprocess.run([
                "curl", "-L", "-o", config["zip_path"],
                config["url"],
                "-H", f"Authorization: Bearer {API_KEY}"
            ], check=True)
            
            # Handle extraction
            if zipfile.is_zipfile(config["zip_path"]):
                print("Extracting ZIP file (including nested ZIPs if any)...")
                extract_nested_zips(config["zip_path"], config["extract_dir"])
            else:
                print("Not a ZIP file. Moving as-is.")
                shutil.move(config["zip_path"], os.path.join(config["extract_dir"], os.path.basename(config["zip_path"])))
            
            print(f"Successfully processed {config['url']}")
            print(f"Contents saved to: {config['extract_dir']}")
        
    except subprocess.CalledProcessError as e:
        print(f"Download failed for {config['url']}: {e}")
    except zipfile.BadZipFile:
        print(f"Invalid ZIP file for {config['url']}")
    except Exception as e:
        print(f"Error processing {config['url']}: {e}")
    finally:
        # Cleanup downloaded ZIP
        if os.path.exists(config["zip_path"]):
            os.remove(config["zip_path"])


global_label_file = "./data/label.csv"
        
# Data Preprocessing










# # For First Dataset "Devanagari Character Dataset"
    
# # First Dataset
# # Define the mapping from your CSV
# numerals = {
#     0: ("०", "Śūn'ya"),
#     1: ("१", "ēka"),
#     2: ("२", "du'ī"),
#     3: ("३", "tīna"),
#     4: ("४", "cāra"),
#     5: ("५", "pām̐ca"),
#     6: ("६", "cha"),
#     7: ("७", "sāta"),
#     8: ("८", "āṭha"),
#     9: ("९", "nau")
# }

# vowels = {
#     1: ("अ", "a"),
#     2: ("आ", "ā"),
#     3: ("इ", "i"),
#     4: ("ई", "ī"),
#     5: ("उ", "u"),
#     6: ("ऊ", "ū"),
#     7: ("ए", "ē"),
#     8: ("ऐ", "ai"),
#     9: ("ओ", "ō"),
#     10: ("औ", "au"),
#     11: ("अं", "aṁ"),
#     12: ("अः", "aḥ")
# }

# consonants = {
#     1: ("क", "ka"),
#     2: ("ख", "kha"),
#     3: ("ग", "ga"),
#     4: ("घ", "gha"),
#     5: ("ङ", "ṅa"),
#     6: ("च", "ca"),
#     7: ("छ", "cha"),
#     8: ("ज", "ja"),
#     9: ("झ", "jha"),
#     10: ("ञ", "ña"),
#     11: ("ट", "ṭa"),
#     12: ("ठ", "ṭha"),
#     13: ("ड", "ḍa"),
#     14: ("ढ", "ḍha"),
#     15: ("ण", "ṇa"),
#     16: ("त", "ta"),
#     17: ("थ", "tha"),
#     18: ("द", "da"),
#     19: ("ध", "dha"),
#     20: ("न", "na"),
#     21: ("प", "pa"),
#     22: ("फ", "pha"),
#     23: ("ब", "ba"),
#     24: ("भ", "bha"),
#     25: ("म", "ma"),
#     26: ("य", "ya"),
#     27: ("र", "ra"),
#     28: ("ल", "la"),
#     29: ("व", "va"),
#     30: ("श", "śa"),
#     31: ("ष", "ṣa"),
#     32: ("स", "sa"),
#     33: ("ह", "ha"),
#     34: ("क्ष", "kṣa"),
#     35: ("त्र", "tra"),
#     36: ("ज्ञ", "jña")
# }

# def create_label_csv(dataset_root, output_csv):
#     data = []
    
#     # Process numerals
#     numerals_dir = os.path.join(dataset_root, "numerals")
#     print("The numerals directory is: ", numerals_dir)
#     for class_id, (devanagari, phonetics) in numerals.items():
#         class_dir = os.path.join(numerals_dir, str(class_id))
#         if os.path.exists(class_dir):
#             for img_file in os.listdir(class_dir):
#                 if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(class_dir, img_file)
#                     data.append({
#                         "image_address": img_path,
#                         "character": devanagari
#                     })
    
#     # Process vowels
#     vowels_dir = os.path.join(dataset_root, "vowels")
#     for class_id, (devanagari, phonetics) in vowels.items():
#         class_dir = os.path.join(vowels_dir, str(class_id))
#         if os.path.exists(class_dir):
#             for img_file in os.listdir(class_dir):
#                 if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(class_dir, img_file)
#                     data.append({
#                         "image_address": img_path,
#                         "character": devanagari
#                     })
    
#     # Process consonants
#     consonants_dir = os.path.join(dataset_root, "consonants")
#     for class_id, (devanagari, phonetics) in consonants.items():
#         class_dir = os.path.join(consonants_dir, str(class_id))
#         if os.path.exists(class_dir):
#             for img_file in os.listdir(class_dir):
#                 if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(class_dir, img_file)
#                     data.append({
#                         "image_address": img_path,
#                         "character": devanagari
#                     })
    
#     print("The data is: ", data[0:5])
#     # Create DataFrame and save to CSV
#     df = pd.DataFrame(data)
#     df.to_csv(output_csv, index=False)
#     print(f"Created label CSV at: {output_csv}")
#     print(f"Total images processed: {len(df)}")

# # Usage
# dataset_root = "./data/devanagari_character_dataset/nhcd/nhcd"  # Path to your nhcd folder
# output_csv = "./data/label_first_dataset.csv"
# create_label_csv(dataset_root, output_csv)
    








# # Second Dataset

# # Path configuration
# dataset_root = "./data/devanagari_character_dataset_large"
# output_csv = "./data/label_second_dataset.csv"

# # Load the labels from labels.csv (assuming it's in the dataset root)
# labels_csv = os.path.join(dataset_root, "labels.csv")

# # Create a mapping from class number to Devanagari character
# class_to_char = {}
# with open(labels_csv, mode='r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         class_to_char[int(row['Class'])] = row['Devanagari label']

# # Prepare to collect all image paths and characters
# data = []

# # Process both train and test folders
# for subset in ['dhcd/train', 'dhcd/test']:
#     subset_path = os.path.join(dataset_root, subset)
    
#     if not os.path.exists(subset_path):
#         print(f"Warning: {subset} directory not found at {subset_path}")
#         continue
    
#     # Iterate through each class folder
#     for class_dir in os.listdir(subset_path):
#         class_path = os.path.join(subset_path, class_dir)
        
#         if not os.path.isdir(class_path):
#             continue
            
#         try:
#             class_num = int(class_dir)
#             devanagari_char = class_to_char.get(class_num)
            
#             if devanagari_char is None:
#                 print(f"Warning: No character mapping for class {class_num}")
#                 continue
                
#             # Add all images in this class directory
#             for img_file in os.listdir(class_path):
#                 if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(class_path, img_file)
#                     data.append({
#                         'image_address': img_path,
#                         'character': devanagari_char
#                     })
                    
#         except ValueError:
#             print(f"Warning: {class_dir} is not a valid class number")
#             continue

# # Create DataFrame and save to CSV
# df = pd.DataFrame(data)
# df.to_csv(output_csv, index=False)
# print(f"The CSV file is created at: {output_csv}")
# print("The number of data processed is: ", len(df))
        





# Third dataset

# Path configuration
full_txt_path = "./data/HindiSeg/full.txt"  # Path to your full.txt file
output_csv = "./data/label_third_dataset.csv"

# Prepare to collect all image paths and characters
data = []

with open(full_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Split each line into image path and Devanagari word
        parts = line.strip().split(' ')
        if len(parts) < 2:
            continue  # Skip malformed lines
            
        image_path = parts[0]
        devanagari_word = parts[1] # Handle multi-word labels
        
        # Extract individual characters from the word
        # for char in devanagari_word:
        data.append({
            'image_address': image_path,
            'character': devanagari_word
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Successfully created {output_csv}")
print(f"Total character-image pairs: {len(df)}")
# print(f"Sample of the created CSV:")
# print(df.head())

