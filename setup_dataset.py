"""
Dataset Setup Helper Script

This script helps you set up the Chest X-Ray Pneumonia dataset.
It verifies the dataset structure and provides download instructions.
"""
import os
from pathlib import Path
from src.config.settings import (
    DATA_DIR,
    DATASET_PATH,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    CLASS_LABELS,
)


def create_directory_structure():
    """Create the required directory structure."""
    print("Creating directory structure...")
    
    # Create base directories
    directories = [
        DATASET_PATH,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
    ]
    
    # Create class subdirectories
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for class_name in CLASS_LABELS:
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] Created: {class_dir}")
    
    print("\n[SUCCESS] Directory structure created successfully!\n")


def verify_dataset():
    """Verify that the dataset is properly set up."""
    print("Verifying dataset structure...\n")
    
    issues = []
    total_images = 0
    
    for split_name, split_dir in [
        ("Training", TRAIN_DIR),
        ("Validation", VAL_DIR),
        ("Test", TEST_DIR),
    ]:
        print(f"{split_name} set:")
        for class_name in CLASS_LABELS:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                issues.append(f"Missing directory: {class_dir}")
                print(f"  [ERROR] {class_name}: Directory not found")
                continue
            
            # Count images
            image_extensions = ["*.jpeg", "*.jpg", "*.png"]
            images = []
            for ext in image_extensions:
                images.extend(list(class_dir.glob(ext)))
            
            image_count = len(images)
            total_images += image_count
            
            if image_count == 0:
                issues.append(f"No images found in: {class_dir}")
                print(f"  [WARNING] {class_name}: No images found")
            else:
                print(f"  [OK] {class_name}: {image_count} images")
    
    print(f"\n[INFO] Total images: {total_images}")
    
    if issues:
        print("\n[WARNING] Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n[INFO] Please download the dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("\n[INSTRUCTIONS]:")
        print("   1. Download the dataset zip file from Kaggle")
        print("   2. Extract it to:", DATASET_PATH)
        print("   3. Ensure the structure matches:")
        print("      data/chest_xray/")
        print("      +-- train/")
        print("      |   +-- NORMAL/")
        print("      |   +-- PNEUMONIA/")
        print("      +-- test/")
        print("      |   +-- NORMAL/")
        print("      |   +-- PNEUMONIA/")
        print("      +-- val/")
        print("          +-- NORMAL/")
        print("          +-- PNEUMONIA/")
        return False
    else:
        print("\n[SUCCESS] Dataset verification passed! Ready for training.")
        return True


def main():
    """Main function."""
    print("=" * 60)
    print("Chest X-Ray Pneumonia Dataset Setup")
    print("=" * 60)
    print()
    
    # Create directory structure
    create_directory_structure()
    
    # Verify dataset
    is_ready = verify_dataset()
    
    if not is_ready:
        print("\n" + "=" * 60)
        print("[NEXT STEPS]:")
        print("=" * 60)
        print("1. Download the dataset from Kaggle")
        print("2. Extract the zip file to:", DATASET_PATH)
        print("3. Run this script again to verify the setup")
        print("\nKaggle Dataset URL:")
        print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    else:
        print("\n" + "=" * 60)
        print("[SUCCESS] Dataset is ready for training!")
        print("=" * 60)
        print("\nYou can now proceed to train the model:")
        print("  python src/training/train.py --epochs 10 --save-model")


if __name__ == "__main__":
    main()

