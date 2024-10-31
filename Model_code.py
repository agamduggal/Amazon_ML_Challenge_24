from PIL import Image
import cv2
import numpy as np
from pytesseract import pytesseract
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import re
import pandas as pd
import requests
import os

# Path to Tesseract executable application
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def download_image(image_url, save_dir='images/', image_name=None):
    """Downloads image from a given URL."""
    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Use the image name from the URL or a given one
    if image_name is None:
        image_name = image_url.split("/")[-1]

    # Full path to save the image
    image_path = os.path.join(save_dir, image_name)

    # Download the image
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(image_path, 'wb') as out_file:
            out_file.write(response.content)
        print(f"Image downloaded: {image_path}")
    else:
        print(f"Failed to download image: {image_url}") #Returns error if image download fails

    return image_path

def download_images_from_csv(csv_file, image_column='image_link'):
    """Downloads all images from a CSV file """
    # Read the CSV file
    data = pd.read_csv(csv_file)

    for index, row in data.iterrows():
        image_url = row[image_column]

        # Download each image
        download_image(image_url, save_dir='images/', image_name=f'image_{index}.jpg')


def preprocess_image(image_path):
    """Preprocess the image to make it compatible with the model."""
    # Load the image
    img = cv2.imread(image_path)

    # Resize the image (common resizing is 224x224 for models like ResNet)
    img_resized = cv2.resize(img, (224, 224))

    # Normalize the image (rescaling pixel values)
    img_normalized = img_resized / 255.0

    # Convert to a format compatible with the model (e.g., add a batch dimension)
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img_batch

# Load the ResNet model pre-trained on ImageNet
model = ResNet50(weights='imagenet', include_top=False)

def extract_image_features(img_batch):
    """Extract features from the image using ResNet."""
    # Preprocess the image (specific to ResNet)
    img_preprocessed = preprocess_input(img_batch)

    # Extract features
    features = model.predict(img_preprocessed)
    return features

def extract_text_tesseract(image_path):
    """Extract text from the image using Tesseract OCR."""
    # Load the image
    img = Image.open(image_path)

    # Perform OCR
    text = pytesseract.image_to_string(img)
    print(text)
    return text

def extract_entity(text):
    """Extract entities from the extracted text."""
    # Example pattern to find numbers and units
    pattern = r'(\d+\.?\d*)\s?(gram|kilogram|cm|mm|inch|ounce|watt|volt)'

    match = re.search(pattern, text.lower())

    if match:
        value = match.group(1)  # Extract the number
        unit = match.group(2)  # Extract the unit

        return f"{value} {unit}"
    else:
        return ""  # Return an empty string if no match is found

def process_image(image_path):
    """Process the image to extract entities using OCR."""
    # Step 1: Preprocess the image (for feature extraction, optional)
    img_batch = preprocess_image(image_path)

    # Step 2: Extract text using OCR
    text = extract_text_tesseract(image_path)

    # Step 3: Extract the entity (e.g., weight, volume) from the text
    entity_value = extract_entity(text)

    return entity_value

def generate_predictions(test_csv, output_csv):
    """Generate predictions for the entities found in images and save them to a CSV."""
    # Download the images from the CSV
    download_images_from_csv(test_csv)

    # Load the test CSV file
    test_data = pd.read_csv(test_csv)

    predictions = []

    for index, row in test_data.iterrows():
        image_link = row['image_link']

        # Use the downloaded image (assume images are saved in 'images/' folder)
        image_path = f'images/image_{index}.jpg'

        # Process the image and get the predicted entity value
        prediction = process_image(image_path)

        predictions.append({
            'index': row['index'],
            'prediction': prediction
        })

    # Convert the predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Save the predictions to a CSV file
    predictions_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# Run the code to generate predictions
generate_predictions(test_csv='C:/Users/Agam/OneDrive/Desktop/ML1/ML/Amn_mlc/test_out.csv', output_csv='testpredictfin.csv')