"""
YOLO Object Detection Module for Blood Cancer Detection
Trains YOLOv8 model for detecting and classifying blood cells with bounding boxes
Calculates mAP (mean Average Precision) for object detection evaluation
"""

import os
import numpy as np
import cv2
from pathlib import Path
import yaml
import shutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


class YOLOTrainer:
    def __init__(self, output_dir='models/yolo', viz_dir='visualizations/yolo'):
        """
        Initialize YOLO trainer
        
        Args:
            output_dir: Directory to save YOLO models
            viz_dir: Directory to save YOLO visualizations
        """
        self.output_dir = output_dir
        self.viz_dir = viz_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        self.model = None
        self.results = None
        self.class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]
        
        print(f"YOLO Trainer initialized")
        print(f"Models will be saved to: {output_dir}")
        print(f"Visualizations will be saved to: {viz_dir}")
    
    def prepare_yolo_dataset(self, data_dir, output_yolo_dir='data_yolo'):
        """
        Convert classification dataset to YOLO format
        Since we don't have bounding boxes, we'll create synthetic ones
        (In real scenarios, you need actual bounding box annotations)
        
        Args:
            data_dir: Directory with train/test folders
            output_yolo_dir: Output directory for YOLO formatted data
        """
        print("\n" + "="*70)
        print("PREPARING YOLO DATASET")
        print("="*70)
        print("\n⚠️  Note: Creating synthetic bounding boxes for classification images")
        print("   In production, use properly annotated data with real bounding boxes\n")
        
        # Create YOLO directory structure
        yolo_train = os.path.join(output_yolo_dir, 'train', 'images')
        yolo_train_labels = os.path.join(output_yolo_dir, 'train', 'labels')
        yolo_val = os.path.join(output_yolo_dir, 'val', 'images')
        yolo_val_labels = os.path.join(output_yolo_dir, 'val', 'labels')
        
        for dir_path in [yolo_train, yolo_train_labels, yolo_val, yolo_val_labels]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Process train and test (val) sets
        for split, yolo_split in [('train', 'train'), ('test', 'val')]:
            split_dir = os.path.join(data_dir, split)
            
            if not os.path.exists(split_dir):
                print(f"⚠️  Warning: {split_dir} not found")
                continue
            
            print(f"\nProcessing {split} set...")
            
            img_count = 0
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(split_dir, class_name)
                
                if not os.path.exists(class_dir):
                    continue
                
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                for img_file in tqdm(images, desc=f"  {class_name}"):
                    img_path = os.path.join(class_dir, img_file)
                    
                    # Read image to get dimensions
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Copy image
                    new_img_name = f"{class_name}_{img_count:05d}.jpg"
                    dst_img = os.path.join(output_yolo_dir, yolo_split, 'images', new_img_name)
                    shutil.copy2(img_path, dst_img)
                    
                    # Create synthetic bounding box (centered, 80% of image)
                    # YOLO format: <class> <x_center> <y_center> <width> <height> (normalized 0-1)
                    x_center = 0.5
                    y_center = 0.5
                    box_width = 0.8
                    box_height = 0.8
                    
                    # Write label file
                    label_file = os.path.join(output_yolo_dir, yolo_split, 'labels', 
                                             new_img_name.replace('.jpg', '.txt'))
                    with open(label_file, 'w') as f:
                        f.write(f"{class_idx} {x_center} {y_center} {box_width} {box_height}\n")
                    
                    img_count += 1
            
            print(f"  ✓ Processed {img_count} images")
        
        # Create data.yaml for YOLO
        data_yaml = {
            'path': os.path.abspath(output_yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        yaml_path = os.path.join(output_yolo_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\n✓ YOLO dataset prepared at: {output_yolo_dir}")
        print(f"✓ Configuration saved to: {yaml_path}")
        
        return yaml_path
    
    def train_yolo(self, data_yaml, epochs=50, img_size=640, batch_size=16):
        """
        Train YOLOv8 model
        
        Args:
            data_yaml: Path to YOLO data.yaml file
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size
        """
        print("\n" + "="*70)
        print("TRAINING YOLO MODEL")
        print("="*70)
        
        # Load YOLOv8 model (nano version for faster training)
        print("\nLoading YOLOv8n model...")
        self.model = YOLO('yolov8n.pt')
        
        # Train
        print(f"\nTraining for {epochs} epochs...")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Batch size: {batch_size}\n")
        
        self.results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=10,
            save=True,
            project=self.output_dir,
            name='train',
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            verbose=True,
            device='cpu'  # Change to 'cuda' or '0' if GPU available
        )
        
        # Save best model
        best_model_path = os.path.join(self.output_dir, 'train', 'weights', 'best.pt')
        final_model_path = os.path.join(self.output_dir, 'yolo_best.pt')
        
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, final_model_path)
            print(f"\n✓ Best model saved to: {final_model_path}")
        
        print("\n✓ YOLO training complete!")
    
    def validate_yolo(self, data_yaml):
        """
        Validate YOLO model and calculate mAP
        
        Args:
            data_yaml: Path to YOLO data.yaml file
            
        Returns:
            Dictionary with validation metrics including mAP
        """
        print("\n" + "="*70)
        print("VALIDATING YOLO MODEL")
        print("="*70)
        
        if self.model is None:
            # Load best model
            model_path = os.path.join(self.output_dir, 'yolo_best.pt')
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                print("❌ No trained model found!")
                return None
        
        # Validate
        print("\nRunning validation...")
        metrics = self.model.val(data=data_yaml, split='val')
        
        # Extract metrics
        results = {
            'mAP50': float(metrics.box.map50),  # mAP at IoU=0.50
            'mAP50-95': float(metrics.box.map),  # mAP at IoU=0.50:0.95
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
        
        # Per-class AP
        if hasattr(metrics.box, 'ap_class_index'):
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = self.class_names[int(class_idx)]
                results[f'AP50_{class_name}'] = float(metrics.box.ap50[i])
        
        print("\n" + "="*70)
        print("YOLO VALIDATION RESULTS")
        print("="*70)
        print(f"\nmAP@0.5:      {results['mAP50']:.4f}")
        print(f"mAP@0.5:0.95: {results['mAP50-95']:.4f}")
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:       {results['recall']:.4f}")
        
        print("\nPer-class AP@0.5:")
        for class_name in self.class_names:
            key = f'AP50_{class_name}'
            if key in results:
                print(f"  {class_name:20s}: {results[key]:.4f}")
        
        # Save metrics
        metrics_df = pd.DataFrame([results])
        metrics_path = os.path.join(self.output_dir, 'yolo_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n✓ Metrics saved to: {metrics_path}")
        
        return results
    
    def visualize_predictions(self, data_yaml, num_samples=10):
        """
        Visualize YOLO predictions on validation set
        
        Args:
            data_yaml: Path to YOLO data.yaml file
            num_samples: Number of samples to visualize
        """
        print("\n" + "="*70)
        print("GENERATING PREDICTION VISUALIZATIONS")
        print("="*70)
        
        if self.model is None:
            model_path = os.path.join(self.output_dir, 'yolo_best.pt')
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                print("❌ No trained model found!")
                return
        
        # Load data.yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        val_img_dir = os.path.join(data_config['path'], data_config['val'])
        
        if not os.path.exists(val_img_dir):
            print(f"❌ Validation directory not found: {val_img_dir}")
            return
        
        # Get random validation images
        val_images = [f for f in os.listdir(val_img_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(val_images) == 0:
            print("❌ No validation images found!")
            return
        
        sample_images = np.random.choice(val_images, 
                                        min(num_samples, len(val_images)), 
                                        replace=False)
        
        print(f"\nGenerating visualizations for {len(sample_images)} samples...")
        
        # Create visualization directory
        pred_viz_dir = os.path.join(self.viz_dir, 'predictions')
        os.makedirs(pred_viz_dir, exist_ok=True)
        
        for img_file in tqdm(sample_images):
            img_path = os.path.join(val_img_dir, img_file)
            
            # Predict
            results = self.model.predict(img_path, save=False, verbose=False)
            
            # Plot results
            if len(results) > 0:
                result = results[0]
                
                # Get annotated image
                annotated_img = result.plot()
                
                # Save
                save_path = os.path.join(pred_viz_dir, f'pred_{img_file}')
                cv2.imwrite(save_path, annotated_img)
        
        print(f"✓ Visualizations saved to: {pred_viz_dir}")
    
    def plot_training_curves(self):
        """
        Plot training curves from YOLO results
        """
        results_csv = os.path.join(self.output_dir, 'train', 'results.csv')
        
        if not os.path.exists(results_csv):
            print("⚠️  Training results not found")
            return
        
        print("\nPlotting training curves...")
        
        # Load results
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: mAP
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 0].plot(df['epoch'], df['metrics/mAP50(B)'], 
                           label='mAP@0.5', linewidth=2)
            axes[0, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], 
                           label='mAP@0.5:0.95', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('mAP')
            axes[0, 0].set_title('Mean Average Precision')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Precision & Recall
        if 'metrics/precision(B)' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/precision(B)'], 
                           label='Precision', linewidth=2)
            axes[0, 1].plot(df['epoch'], df['metrics/recall(B)'], 
                           label='Recall', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Precision & Recall')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss
        if 'train/box_loss' in df.columns:
            axes[1, 0].plot(df['epoch'], df['train/box_loss'], 
                           label='Box Loss', linewidth=2)
            axes[1, 0].plot(df['epoch'], df['train/cls_loss'], 
                           label='Class Loss', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Validation Loss
        if 'val/box_loss' in df.columns:
            axes[1, 1].plot(df['epoch'], df['val/box_loss'], 
                           label='Box Loss', linewidth=2)
            axes[1, 1].plot(df['epoch'], df['val/cls_loss'], 
                           label='Class Loss', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Validation Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, 'yolo_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to: {save_path}")


def train_yolo_pipeline(data_dir='data_split', epochs=50):
    """
    Complete YOLO training pipeline
    
    Args:
        data_dir: Directory with train/test folders
        epochs: Number of training epochs
    """
    print("\n" + "="*70)
    print("YOLO OBJECT DETECTION PIPELINE")
    print("="*70)
    
    # Initialize trainer
    trainer = YOLOTrainer()
    
    # Prepare dataset
    data_yaml = trainer.prepare_yolo_dataset(data_dir)
    
    # Train YOLO
    trainer.train_yolo(data_yaml, epochs=epochs)
    
    # Validate and get mAP
    metrics = trainer.validate_yolo(data_yaml)
    
    # Visualize predictions
    trainer.visualize_predictions(data_yaml, num_samples=20)
    
    # Plot training curves
    trainer.plot_training_curves()
    
    print("\n" + "="*70)
    print("YOLO PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFinal mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"Final mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    
    return trainer, metrics


if __name__ == "__main__":
    # Run YOLO pipeline
    trainer, metrics = train_yolo_pipeline(data_dir='data_split', epochs=50)
