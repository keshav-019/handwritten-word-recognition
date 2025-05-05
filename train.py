import os
import shutil
import zipfile
import subprocess
import pandas as pd
import json
import csv
from pathlib import Path
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split
from PIL import Image

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
        "url": "https://www.kaggle.com/api/v1/datasets/download/sabarinathan/handwritten-hindi-word-recognition",
        "zip_path": "./data/handwritten_hindi_word_recognition.zip",
        "extract_dir": "./data/handwritten_hindi_word_recognition",
        "original_zip_file": "./data/handwritten_hindi_word_recognition/handwritten_hindi_word_recognition.zip"
    },
    {
        "url": "https://www.kaggle.com/api/v1/datasets/download/ashokpant/devanagari-character-dataset-large",
        "zip_path": "./data/devanagari_character_dataset_large.zip",
        "extract_dir": "./data/devanagari_character_dataset_large",
        "original_zip_file": "./data/devanagari_character_dataset_large/devanagari_character_dataset_large.zip"
    },
]

def extract_nested_archives(archive_path, extract_dir):
    """
    Recursively extract nested archives (ZIP, TAR.GZ, TAR.XZ)
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract contents to
    """
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract based on file type
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as archive_ref:
            archive_ref.extractall(extract_dir)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as archive_ref:
            archive_ref.extractall(extract_dir)
    elif archive_path.endswith(('.tar.xz', '.txz')):
        with tarfile.open(archive_path, 'r:xz') as archive_ref:
            archive_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    # Recursively process extracted files
    for root, _, files in os.walk(extract_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check for nested archives
            if any(file.lower().endswith(ext) for ext in ['.zip', '.tar.gz', '.tgz', '.tar.xz', '.txz']):
                nested_extract = os.path.splitext(file_path)[0]
                try:
                    extract_nested_archives(file_path, nested_extract)
                    os.remove(file_path)  # Clean up after extraction
                except Exception as e:
                    print(f"Failed to extract {file_path}: {str(e)}")
                    continue

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
                extract_nested_archives(config["zip_path"], config["extract_dir"])
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










# For First Dataset "Devanagari Character Dataset"
    
# First Dataset
# Define the mapping from your CSV
numerals = {
    0: ("०", "Śūn'ya"),
    1: ("१", "ēka"),
    2: ("२", "du'ī"),
    3: ("३", "tīna"),
    4: ("४", "cāra"),
    5: ("५", "pām̐ca"),
    6: ("६", "cha"),
    7: ("७", "sāta"),
    8: ("८", "āṭha"),
    9: ("९", "nau")
}

vowels = {
    1: ("अ", "a"),
    2: ("आ", "ā"),
    3: ("इ", "i"),
    4: ("ई", "ī"),
    5: ("उ", "u"),
    6: ("ऊ", "ū"),
    7: ("ए", "ē"),
    8: ("ऐ", "ai"),
    9: ("ओ", "ō"),
    10: ("औ", "au"),
    11: ("अं", "aṁ"),
    12: ("अः", "aḥ")
}

consonants = {
    1: ("क", "ka"),
    2: ("ख", "kha"),
    3: ("ग", "ga"),
    4: ("घ", "gha"),
    5: ("ङ", "ṅa"),
    6: ("च", "ca"),
    7: ("छ", "cha"),
    8: ("ज", "ja"),
    9: ("झ", "jha"),
    10: ("ञ", "ña"),
    11: ("ट", "ṭa"),
    12: ("ठ", "ṭha"),
    13: ("ड", "ḍa"),
    14: ("ढ", "ḍha"),
    15: ("ण", "ṇa"),
    16: ("त", "ta"),
    17: ("थ", "tha"),
    18: ("द", "da"),
    19: ("ध", "dha"),
    20: ("न", "na"),
    21: ("प", "pa"),
    22: ("फ", "pha"),
    23: ("ब", "ba"),
    24: ("भ", "bha"),
    25: ("म", "ma"),
    26: ("य", "ya"),
    27: ("र", "ra"),
    28: ("ल", "la"),
    29: ("व", "va"),
    30: ("श", "śa"),
    31: ("ष", "ṣa"),
    32: ("स", "sa"),
    33: ("ह", "ha"),
    34: ("क्ष", "kṣa"),
    35: ("त्र", "tra"),
    36: ("ज्ञ", "jña")
}

def create_label_csv(dataset_root, output_csv):
    data = []
    
    # Process numerals
    numerals_dir = os.path.join(dataset_root, "numerals")
    print("The numerals directory is: ", numerals_dir)
    for class_id, (devanagari, phonetics) in numerals.items():
        class_dir = os.path.join(numerals_dir, str(class_id))
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    data.append({
                        "image_address": img_path,
                        "character": devanagari
                    })
    
    # Process vowels
    vowels_dir = os.path.join(dataset_root, "vowels")
    for class_id, (devanagari, phonetics) in vowels.items():
        class_dir = os.path.join(vowels_dir, str(class_id))
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    data.append({
                        "image_address": img_path,
                        "character": devanagari
                    })
    
    # Process consonants
    consonants_dir = os.path.join(dataset_root, "consonants")
    for class_id, (devanagari, phonetics) in consonants.items():
        class_dir = os.path.join(consonants_dir, str(class_id))
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    data.append({
                        "image_address": img_path,
                        "character": devanagari
                    })
    
    print("The data is: ", data[0:5])
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created label CSV at: {output_csv}")
    print(f"Total images processed: {len(df)}")

# Usage
dataset_root = "./data/devanagari_character_dataset/nhcd/nhcd"  # Path to your nhcd folder
output_csv = "./data/label_first_dataset.csv"
create_label_csv(dataset_root, output_csv)
    








# Second Dataset

# Path configuration
dataset_root = "./data/devanagari_character_dataset_large"
output_csv = "./data/label_second_dataset.csv"

# Load the labels from labels.csv (assuming it's in the dataset root)
labels_csv = os.path.join(dataset_root, "labels.csv")

# Create a mapping from class number to Devanagari character
class_to_char = {}
with open(labels_csv, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_to_char[int(row['Class'])] = row['Devanagari label']

# Prepare to collect all image paths and characters
data = []

# Process both train and test folders
for subset in ['dhcd/train', 'dhcd/test']:
    subset_path = os.path.join(dataset_root, subset)
    
    if not os.path.exists(subset_path):
        print(f"Warning: {subset} directory not found at {subset_path}")
        continue
    
    # Iterate through each class folder
    for class_dir in os.listdir(subset_path):
        class_path = os.path.join(subset_path, class_dir)
        
        if not os.path.isdir(class_path):
            continue
            
        try:
            class_num = int(class_dir)
            devanagari_char = class_to_char.get(class_num)
            
            if devanagari_char is None:
                print(f"Warning: No character mapping for class {class_num}")
                continue
                
            # Add all images in this class directory
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    data.append({
                        'image_address': img_path,
                        'character': devanagari_char
                    })
                    
        except ValueError:
            print(f"Warning: {class_dir} is not a valid class number")
            continue

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"The CSV file is created at: {output_csv}")
print("The number of data processed is: ", len(df))
        





# Third dataset

# Path configuration
full_txt_path = "./data/handwritten_hindi_word_recognition/train.txt"  # Path to your full.txt file 
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
            'image_address': "./data/handwritten_hindi_word_recognition/HindiSeg/" + image_path,
            'character': devanagari_word
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"Successfully created {output_csv}")
print(f"Total character-image pairs: {len(df)}")
# print(f"Sample of the created CSV:")
# print(df.head())







# Fourth Dataset


fourth_dataset_variables = {
    "character_01_ka": "क",
    "character_02_kha": "ख",
    "character_03_ga": "ग",
    "character_04_gha": "घ",
    "character_05_kna": "ङ",
    "character_06_cha": "च",
    "character_07_chha": "छ",
    "character_08_ja": "ज",
    "character_09_jha": "झ",
    "character_10_yna": "ञ",
    "character_11_taamatar": "ट",
    "character_12_thaa": "ठ",
    "character_13_daa": "ड",
    "character_14_dhaa": "ढ",
    "character_15_adna": "ण",
    "character_16_tabala": "त",
    "character_17_tha": "थ",
    "character_18_da": "द",
    "character_19_dha": "ध",
    "character_20_na": "न",
    "character_21_pa": "प",
    "character_22_pha": "फ",
    "character_23_ba": "ब",
    "character_24_bha": "भ",
    "character_25_ma": "म",
    "character_26_yaw": "य",
    "character_27_ra": "र",
    "character_28_la": "ल",
    "character_29_waw": "व",
    "character_30_motosaw": "श",
    "character_31_petchiryakha": "ष",
    "character_32_patalosaw": "स",
    "character_33_ha": "ह",
    "character_34_chhya": "क्ष",
    "character_35_tra": "त्र",
    "character_36_gya": "ज्ञ",
    "digit_0": "०",
    "digit_1": "१",
    "digit_2": "२",
    "digit_3": "३",
    "digit_4": "४",
    "digit_5": "५",
    "digit_6": "६",
    "digit_7": "७",
    "digit_8": "८",
    "digit_9": "९"
}

base_path = "./data/devanagari_character_set/Images/Images"

data = []

# Iterate through each character folder
for folder_name, devanagari_char in fourth_dataset_variables.items():
    folder_path = os.path.join(base_path, folder_name)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found - {folder_path}")
        continue
    
    # Process each image in the folder
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, image_file)
            data.append({
                "image_address": image_path,
                "character": devanagari_char
            })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("./data/label_fourth_dataset.csv", index=False)

print(f"CSV file created with {len(df)} entries")
print("Sample of the first 5 entries:")
print(df.head())






# Set memory growth for GPU if available


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Combine all datasets
def combine_datasets():
    dfs = []
    for i in range(1, 5):
        try:
            df = pd.read_csv(f'./data/label_{"first" if i==1 else "second" if i==2 else "third" if i==3 else "fourth"}_dataset.csv')
            dfs.append(df[['image_address', 'character']])
        except FileNotFoundError:
            print(f"Warning: label_{i}_dataset.csv not found")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv('./data/combined_dataset.csv', index=False)
    return combined



# Preprocess images (updated for TF 2.16)
def preprocess_image(image_path, img_size=(64, 64)):
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        return img_array.astype(np.float32)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None



def prepare_data(df, batch_size=32):
    char_to_label = {char: i for i, char in enumerate(sorted(df['character'].unique()))}
    df['label'] = df['character'].map(char_to_label)
    
    def generator():
        for _, row in df.iterrows():
            img_array = preprocess_image(row['image_address'])
            if img_array is not None:
                # Add channel dimension if missing
                if img_array.ndim == 2:
                    img_array = np.expand_dims(img_array, axis=-1)  # Shape becomes (64, 64, 1)
                yield img_array, utils.to_categorical(row['label'], num_classes=len(char_to_label))
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(64, 64, 1), dtype=tf.float32),  # Now explicitly (64,64,1)
            tf.TensorSpec(shape=(len(char_to_label),), dtype=tf.float32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, char_to_label

# Build optimized CNN model
def build_model(num_classes):
    inputs = tf.keras.Input(shape=(64, 64, 1))
    
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Combine datasets
combined_df = combine_datasets()

# Prepare data
dataset, char_to_label = prepare_data(combined_df)
label_to_char = {v: k for k, v in char_to_label.items()}
np.save('label_mapping.npy', label_to_char)

# Build and train model
model = build_model(len(char_to_label))

# Add callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5)
]

# Calculate steps per epoch
steps_per_epoch = len(combined_df) // 32

# Train model
history = model.fit(
    dataset,
    epochs=20,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch
)

# Save final model
model.save('devanagari_cnn.keras')

