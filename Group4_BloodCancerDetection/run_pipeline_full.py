"""
Complete Integrated Pipeline with YOLO
Runs CNN feature extraction, ML model training, and YOLO object detection
"""

import os
import sys
import numpy as np
import argparse
from feature_extractor import CNNFeatureExtractor
from model_trainer_cv import ModelTrainer
from yolo_trainer import train_yolo_pipeline


def main(data_dir, output_dir='models', viz_dir='visualizations', 
         include_yolo=True, yolo_epochs=50):
    """
    Complete training pipeline with YOLO
    
    Args:
        data_dir: Root directory with train/test folders
        output_dir: Directory to save models
        viz_dir: Directory to save visualizations
        include_yolo: Whether to train YOLO (default: True)
        yolo_epochs: Number of YOLO training epochs
    """
    print("\n" + "="*70)
    print("BLOOD CANCER DETECTION - COMPLETE INTEGRATED PIPELINE")
    print("="*70 + "\n")
    
    # Define class names
    class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]
    
    # ========================================================================
    # PART 1: CNN FEATURE EXTRACTION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: CNN FEATURE EXTRACTION")
    print("="*70 + "\n")
    
    extractor = CNNFeatureExtractor()
    
    # Extract features from training data
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        print(f"❌ Error: Training directory not found: {train_dir}")
        print("Please run: python prepare_data.py organize <your_data> --output_dir data_split")
        return
    
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
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'features.npy'), all_features)
    np.save(os.path.join(output_dir, 'labels.npy'), all_labels)
    print(f"Features saved to {output_dir}")
    
    # ========================================================================
    # PART 2: TRADITIONAL ML MODEL TRAINING WITH CROSS-VALIDATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: ML MODEL TRAINING (with Cross-Validation)")
    print("="*70 + "\n")
    
    trainer = ModelTrainer(output_dir=output_dir, viz_dir=viz_dir)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(all_features, all_labels)
    
    # Train all ML models with cross-validation
    trainer.train_all_models(X_train, X_test, y_train, y_test, class_names, use_cv=True)
    
    print("\n✓ ML models training complete!")
    
    # ========================================================================
    # PART 3: YOLO OBJECT DETECTION TRAINING
    # ========================================================================
    if include_yolo:
        print("\n" + "="*70)
        print("STEP 3: YOLO OBJECT DETECTION TRAINING")
        print("="*70 + "\n")
        
        try:
            yolo_trainer, yolo_metrics = train_yolo_pipeline(
                data_dir=data_dir,
                epochs=yolo_epochs
            )
            
            print("\n✓ YOLO training complete!")
            print(f"\nYOLO Results:")
            print(f"  mAP@0.5:      {yolo_metrics['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {yolo_metrics['mAP50-95']:.4f}")
            print(f"  Precision:    {yolo_metrics['precision']:.4f}")
            print(f"  Recall:       {yolo_metrics['recall']:.4f}")
            
        except Exception as e:
            print(f"\n⚠️  YOLO training failed: {str(e)}")
            print("Continuing with ML models only...")
    else:
        print("\n⚠️  Skipping YOLO training (include_yolo=False)")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Output Directories:")
    print(f"  Models:         {output_dir}")
    print(f"  Visualizations: {viz_dir}")
    
    print(f"\n📊 Trained Models:")
    print(f"  ✓ Logistic Regression")
    print(f"  ✓ Support Vector Machine (SVM)")
    print(f"  ✓ Random Forest")
    print(f"  ✓ XGBoost")
    print(f"  ✓ K-Means Clustering")
    if include_yolo:
        print(f"  ✓ YOLOv8 Object Detection")
    
    print(f"\n📈 Evaluation Metrics Computed:")
    print(f"  ✓ Accuracy")
    print(f"  ✓ Precision")
    print(f"  ✓ Recall (Sensitivity)")
    print(f"  ✓ F1-Score")
    print(f"  ✓ Specificity")
    print(f"  ✓ ROC-AUC curves")
    print(f"  ✓ Precision-Recall curves")
    print(f"  ✓ Confusion matrices")
    print(f"  ✓ 5-Fold Cross-Validation scores")
    if include_yolo:
        print(f"  ✓ mAP (mean Average Precision)")
    
    print(f"\n📉 Visualizations Generated:")
    print(f"  ✓ Confusion matrices")
    print(f"  ✓ Training/validation curves")
    print(f"  ✓ ROC curves")
    print(f"  ✓ Precision-Recall curves")
    print(f"  ✓ Feature importance plots")
    print(f"  ✓ Classification clusters (PCA)")
    print(f"  ✓ Cross-validation score charts")
    if include_yolo:
        print(f"  ✓ YOLO detection visualizations")
        print(f"  ✓ YOLO training curves")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Review metrics in: models/metrics_summary.csv")
    print("  2. Review CV results in: models/cv_results.csv")
    if include_yolo:
        print("  3. Review YOLO metrics in: models/yolo/yolo_metrics.csv")
        print("  4. Check YOLO predictions in: visualizations/yolo/predictions/")
    print(f"  {4 if not include_yolo else 5}. View all visualizations in: {viz_dir}/")
    print(f"  {5 if not include_yolo else 6}. Run web app: python app.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Blood Cancer Detection - Complete Integrated Pipeline with YOLO'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data_split',
        help='Root directory containing train/ and test/ folders (default: data_split)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    parser.add_argument(
        '--viz_dir',
        type=str,
        default='visualizations',
        help='Directory to save visualizations (default: visualizations)'
    )
    parser.add_argument(
        '--include_yolo',
        action='store_true',
        default=True,
        help='Include YOLO object detection training (default: True)'
    )
    parser.add_argument(
        '--skip_yolo',
        action='store_true',
        help='Skip YOLO training (use if you only want ML models)'
    )
    parser.add_argument(
        '--yolo_epochs',
        type=int,
        default=50,
        help='Number of YOLO training epochs (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Handle skip_yolo flag
    if args.skip_yolo:
        args.include_yolo = False
    
    # Run pipeline
    main(
        args.data_dir,
        args.output_dir,
        args.viz_dir,
        args.include_yolo,
        args.yolo_epochs
    )
