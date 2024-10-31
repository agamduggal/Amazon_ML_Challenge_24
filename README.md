# Image-Based Entity Extraction Model

## Overview
This project focuses on building a machine learning model to automatically extract specific entities, such as product dimensions (width, height, depth), weight, wattage, etc., from images. Using Optical Character Recognition (OCR) and deep learning techniques, the project processes images that contain product details, blueprints, or technical specifications, automating the process of extracting relevant information from text-based visuals.

> **Note:** This project was initially developed by our team as part of the Amazon ML Challenge. However, due to time constraints, we were unable to submit our work within the competition deadline. We have decided to continue developing and refining the model, sharing it here on GitHub to encourage collaboration and future improvements.

## Approach
The model employs a combination of two key techniques:
1. **OCR (Optical Character Recognition)**: Tesseract OCR is used to extract textual information from images, targeting numeric values and units (e.g., "15 cm," "2.5 kg").
2. **Deep Learning-Based Image Processing**: The ResNet50 model, pre-trained on ImageNet, preprocesses images to improve OCR accuracy by standardizing image dimensions and quality.

## Model Workflow
1. **Image Downloading and Preprocessing**  
   - Downloads images from URLs provided in a CSV file and saves them locally.
   - Preprocesses each image to 224x224 pixels, normalizes pixel values, and prepares them for ResNet50 input.

2. **Feature Extraction Using Pre-trained ResNet50**  
   - Extracts high-dimensional feature representations from images using ResNet50, configured as a feature extractor without its final layer.

3. **Optical Character Recognition (OCR) with Tesseract**  
   - Extracts textual content from images using Tesseract OCR, focusing on product measurements and specifications.

4. **Entity Extraction from Text**  
   - Extracts numeric values and units from OCR output using regular expressions, identifying relevant product attributes like weight and dimensions.

5. **Pipeline for Processing and Prediction**  
   - A complete pipeline processes each image, extracts text, identifies entities, and saves predictions in a structured CSV format for easy analysis.

![image](https://github.com/user-attachments/assets/3ffda3f6-4bc0-48a2-9a30-6615368bf866)

## Future Directions
Our team plans to continue validating and enhancing the model's performance by:
- **Validation on Train-Test Split**: Initially, we will evaluate the model’s performance using the train-test split on the existing dataset.
- **Using a Cache System**: Future updates will include a caching mechanism to avoid re-downloading or re-processing steps if they have already been completed, improving model efficiency and speed.
- **Batch Processing and Memory Management**: For memory efficiency, we will implement a threaded download and processing system, where images are processed in small batches (e.g., the first 5000 rows initially). Each image will be deleted from memory after processing, and garbage collection (`gc.collect()`) will be triggered to free up memory resources.
- **Concurrent Download and Processing**: Use threading to manage concurrent downloads and preprocessing of images, optimizing runtime.

