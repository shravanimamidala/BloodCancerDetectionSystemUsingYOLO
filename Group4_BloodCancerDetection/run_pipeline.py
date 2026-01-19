"""
Complete Pipeline: Feature Extraction + Model Training
Run this script to process your dataset and train all models
"""

import os
import numpy as np
import argparse
from feature_extractor import CNNFeatureExtractor
from model_trainer import ModelTrainer


def main(data_dir, output_dir='models', viz_dir='visualizations'):
    """
    Complete training pipeline
    
    Args:
        data_dir: Root directory with train/test folders
        output_dir: Directory to save models
        viz_dir: Directory to save visualizations
    """
    print("\n" + "="*70)
    print("BLOOD CANCER DETECTION SYSTEM - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # Define class names
    class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]
    
    # Step 1: Feature Extraction
    print("\n" + "="*70)
    print("STEP 1: CNN FEATURE EXTRACTION")
    print("="*70 + "\n")
    
    extractor = CNNFeatureExtractor()
    
    # Extract features from training data
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    print("Extracting features from training data...")
    train_features, train_labels, _ = extractor.extract_features_from_directory(
        train_dir, class_names
    )
    
    print("\nExtracting features from test data...")
    test_features, test_labels, _ = extractor.extract_features_from_directory(
        test_dir, class_names
    )
    
    # Combine train and test for full dataset
    all_features = np.vstack([train_features, test_features])
    all_labels = np.concatenate([train_labels, test_labels])
    
    print(f"\nTotal dataset size: {all_features.shape}")
    
    # Save features
    np.save(os.path.join(output_dir, 'features.npy'), all_features)
    np.save(os.path.join(output_dir, 'labels.npy'), all_labels)
    print(f"Features saved to {output_dir}")
    
    # Step 2: Model Training
    print("\n" + "="*70)
    print("STEP 2: MODEL TRAINING")
    print("="*70 + "\n")
    
    trainer = ModelTrainer(output_dir=output_dir, viz_dir=viz_dir)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(all_features, all_labels)
    
    # Train all models
    trainer.train_all_models(X_train, X_test, y_train, y_test, class_names)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nTrained models saved in: {output_dir}")
    print(f"Visualizations saved in: {viz_dir}")
    print("\nYou can now run the Flask server with: python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Blood Cancer Detection - Complete Training Pipeline'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Root directory containing train/ and test/ folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--viz_dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    main(args.data_dir, args.output_dir, args.viz_dir)
