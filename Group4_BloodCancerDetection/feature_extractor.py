"""
Feature Extractor using Pre-trained CNN Model
Extracts 2048-dimensional feature vectors from blood cell images
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from pathlib import Path

class CNNFeatureExtractor:
    def __init__(self, model_name='ResNet50'):
        """
        Initialize the CNN feature extractor
        
        Args:
            model_name: Name of pretrained model (default: ResNet50)
        """
        print(f"Loading {model_name} model...")
        
        # Load pre-trained ResNet50 without top layers (for feature extraction)
        self.model = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg',  # Global average pooling
            input_shape=(224, 224, 3)
        )
        
        self.model_name = model_name
        self.feature_dim = 2048  # ResNet50 output dimension
        
        print(f"Model loaded successfully. Feature dimension: {self.feature_dim}")
    
    def preprocess_image(self, img_path):
        """
        Load and preprocess image for CNN
        
        Args:
            img_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    
    def extract_features(self, img_path):
        """
        Extract CNN features from a single image
        
        Args:
            img_path: Path to image file
            
        Returns:
            Feature vector (2048-dimensional)
        """
        preprocessed_img = self.preprocess_image(img_path)
        features = self.model.predict(preprocessed_img, verbose=0)
        return features.flatten()
    
    def extract_features_batch(self, img_paths, batch_size=32):
        """
        Extract CNN features from multiple images
        
        Args:
            img_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            Feature matrix (n_samples, 2048)
        """
        features_list = []
        
        print(f"Extracting features from {len(img_paths)} images...")
        
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            batch_images = np.vstack([
                self.preprocess_image(path) for path in batch_paths
            ])
            
            batch_features = self.model.predict(batch_images, verbose=0)
            features_list.append(batch_features)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + len(batch_paths)}/{len(img_paths)} images")
        
        features = np.vstack(features_list)
        print(f"Feature extraction complete. Shape: {features.shape}")
        
        return features
    
    def extract_features_from_directory(self, data_dir, class_names):
        """
        Extract features from organized directory structure
        
        Directory structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
                img2.jpg
        
        Args:
            data_dir: Root directory containing class subdirectories
            class_names: List of class names (subdirectory names)
            
        Returns:
            features: Feature matrix
            labels: Corresponding labels
            file_paths: List of file paths
        """
        features_list = []
        labels_list = []
        file_paths_list = []
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            print(f"\nProcessing class: {class_name}")
            
            # Get all image files
            image_files = [
                os.path.join(class_dir, f) 
                for f in os.listdir(class_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            
            print(f"Found {len(image_files)} images")
            
            # Extract features
            for img_path in image_files:
                try:
                    features = self.extract_features(img_path)
                    features_list.append(features)
                    labels_list.append(class_idx)
                    file_paths_list.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
        
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        print(f"\nTotal features extracted: {features.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        return features, labels, file_paths_list


if __name__ == "__main__":
    # Example usage
    extractor = CNNFeatureExtractor()
    
    # Example: Extract features from organized directory
    data_dir = "data/train"
    class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]
    
    features, labels, file_paths = extractor.extract_features_from_directory(
        data_dir, class_names
    )
    
    print(f"\nFeature extraction complete!")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
