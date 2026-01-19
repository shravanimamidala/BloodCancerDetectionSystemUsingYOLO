"""
Data Preparation Script
Automatically organizes and splits your dataset into train/test folders
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict
import argparse


def organize_dataset(source_dir, output_dir='data_split', test_split=0.2, random_seed=42):
    """
    Organize unsorted blood cell images into train/test structure
    
    Args:
        source_dir: Directory containing your images (organized by class or mixed)
        output_dir: Output directory (will create train/ and test/ subdirs)
        test_split: Proportion of data for testing (default: 0.2 = 20%)
        random_seed: Random seed for reproducibility
    """
    print("\n" + "="*70)
    print("BLOOD CANCER DETECTION - DATA PREPARATION")
    print("="*70 + "\n")
    
    random.seed(random_seed)
    
    # Define expected class names
    class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Create output directory structure
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    
    print(f"Creating directory structure in: {output_dir}")
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    print("✓ Directory structure created\n")
    
    # Collect images by class
    class_images = defaultdict(list)
    
    print("Scanning for images...")
    
    # Check if source_dir has class subdirectories
    subdirs = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    has_class_structure = any(subdir.lower() in [c.lower() for c in class_names] 
                               for subdir in subdirs)
    
    if has_class_structure:
        print("✓ Detected class-based folder structure")
        
        # Images are already organized by class
        for class_name in class_names:
            # Check for case-insensitive matches
            for subdir in subdirs:
                if subdir.lower() == class_name.lower():
                    class_dir = os.path.join(source_dir, subdir)
                    
                    # Get all image files
                    for file in os.listdir(class_dir):
                        file_path = os.path.join(class_dir, file)
                        if os.path.isfile(file_path):
                            ext = os.path.splitext(file)[1].lower()
                            if ext in image_extensions:
                                class_images[class_name].append(file_path)
                    
                    print(f"  {class_name}: {len(class_images[class_name])} images")
    
    else:
        print("⚠ No class structure detected")
        print("Please organize your images in folders by class name:")
        print("\nExpected structure:")
        print(f"{source_dir}/")
        for class_name in class_names:
            print(f"  ├── {class_name}/")
            print(f"  │   ├── image1.jpg")
            print(f"  │   ├── image2.jpg")
            print(f"  │   └── ...")
        print("\nClass names (case-insensitive):")
        for i, class_name in enumerate(class_names, 1):
            print(f"  {i}. {class_name}")
        return
    
    # Check if any images were found
    total_images = sum(len(images) for images in class_images.values())
    
    if total_images == 0:
        print("\n❌ No images found!")
        print(f"Please check that {source_dir} contains image files")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return
    
    print(f"\n✓ Total images found: {total_images}")
    
    # Split and copy images
    print("\nSplitting dataset into train and test...")
    print(f"Train/Test split: {int((1-test_split)*100)}% / {int(test_split*100)}%\n")
    
    train_count = 0
    test_count = 0
    
    for class_name, image_paths in class_images.items():
        if not image_paths:
            print(f"⚠ Warning: No images found for {class_name}")
            continue
        
        # Shuffle images
        random.shuffle(image_paths)
        
        # Calculate split
        n_test = max(1, int(len(image_paths) * test_split))
        n_train = len(image_paths) - n_test
        
        # Split images
        train_images = image_paths[:n_train]
        test_images = image_paths[n_train:]
        
        # Copy train images
        for i, src_path in enumerate(train_images, 1):
            ext = os.path.splitext(src_path)[1]
            dst_filename = f"{class_name}_{i:04d}{ext}"
            dst_path = os.path.join(train_dir, class_name, dst_filename)
            shutil.copy2(src_path, dst_path)
            train_count += 1
        
        # Copy test images
        for i, src_path in enumerate(test_images, 1):
            ext = os.path.splitext(src_path)[1]
            dst_filename = f"{class_name}_{i:04d}{ext}"
            dst_path = os.path.join(test_dir, class_name, dst_filename)
            shutil.copy2(src_path, dst_path)
            test_count += 1
        
        print(f"✓ {class_name}:")
        print(f"    Train: {len(train_images)} images")
        print(f"    Test:  {len(test_images)} images")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nTotal images processed: {total_images}")
    print(f"  Training set: {train_count} images ({train_count/total_images*100:.1f}%)")
    print(f"  Test set:     {test_count} images ({test_count/total_images*100:.1f}%)")
    print(f"\nData saved to: {output_dir}")
    print(f"  Train: {train_dir}")
    print(f"  Test:  {test_dir}")
    print("\nYou can now run: python run_pipeline.py --data_dir", output_dir)


def verify_dataset(data_dir):
    """
    Verify the dataset structure and count images
    """
    print("\n" + "="*70)
    print("DATASET VERIFICATION")
    print("="*70 + "\n")
    
    class_names = ["basophil", "erythroblast", "monocyte", "myeloblast", "seg_neutrophil"]
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"❌ {split.upper()} directory not found: {split_dir}")
            continue
        
        print(f"\n{split.upper()} SET:")
        print("-" * 40)
        
        total = 0
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"  ❌ {class_name}: Directory not found")
                continue
            
            # Count images
            images = [f for f in os.listdir(class_dir) 
                     if os.path.splitext(f)[1].lower() in image_extensions]
            
            count = len(images)
            total += count
            
            status = "✓" if count > 0 else "⚠"
            print(f"  {status} {class_name:20s}: {count:4d} images")
        
        print(f"  {'─'*20}")
        print(f"  {'TOTAL':20s}: {total:4d} images")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Prepare blood cell dataset for training'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Organize and split dataset')
    organize_parser.add_argument(
        'source_dir',
        type=str,
        help='Directory containing your images (organized by class)'
    )
    organize_parser.add_argument(
        '--output_dir',
        type=str,
        default='data_split',
        help='Output directory (default: data_split)'
    )
    organize_parser.add_argument(
        '--test_split',
        type=float,
        default=0.2,
        help='Proportion for test set (default: 0.2)'
    )
    organize_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset structure')
    verify_parser.add_argument(
        'data_dir',
        type=str,
        default='data_split',
        nargs='?',
        help='Data directory to verify (default: data_split)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'organize':
        organize_dataset(
            args.source_dir,
            args.output_dir,
            args.test_split,
            args.seed
        )
    elif args.command == 'verify':
        verify_dataset(args.data_dir)
    else:
        parser.print_help()
        print("\n" + "="*70)
        print("USAGE EXAMPLES")
        print("="*70)
        print("\n1. Split your data into train/test:")
        print("   python prepare_data.py organize data --output_dir data_split")
        print("\n2. Verify the split:")
        print("   python prepare_data.py verify data_split")
        print("\n3. Custom split ratio (e.g., 70/30):")
        print("   python prepare_data.py organize data --output_dir data_split --test_split 0.3")
        print("\n" + "="*70)
