"""
Dataset Download Script

This script helps download the Chest X-Ray Pneumonia dataset from Kaggle.
Supports both kagglehub and kaggle API methods.
"""
import os
import sys
from pathlib import Path
import zipfile
import shutil

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = DATA_DIR / "chest_xray"


def download_with_kagglehub():
    """Download dataset using kagglehub library."""
    try:
        import kagglehub
        
        print("Downloading dataset using kagglehub...")
        print("Dataset: paultimothymooney/chest-xray-pneumonia")
        print()
        
        # Download latest version
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        
        print(f"Dataset downloaded to: {path}")
        print()
        
        # Move to project directory
        print("Moving dataset to project directory...")
        if Path(path).exists():
            # Find the chest_xray folder in the downloaded path
            chest_xray_path = None
            for root, dirs, files in os.walk(path):
                if 'chest_xray' in dirs:
                    chest_xray_path = Path(root) / 'chest_xray'
                    break
            
            if chest_xray_path and chest_xray_path.exists():
                # Remove existing directory if it exists
                if DATASET_DIR.exists():
                    shutil.rmtree(DATASET_DIR)
                
                # Move to project directory
                shutil.move(str(chest_xray_path), str(DATASET_DIR))
                print(f"Dataset moved to: {DATASET_DIR}")
                return True
            else:
                print("Warning: Could not find chest_xray folder in downloaded dataset")
                print(f"Please manually move the dataset from: {path}")
                return False
        else:
            print(f"Error: Download path does not exist: {path}")
            return False
            
    except ImportError:
        print("kagglehub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kagglehub")
        return download_with_kagglehub()
    except Exception as e:
        print(f"Error downloading with kagglehub: {e}")
        return False


def download_with_kaggle_api():
    """Download dataset using Kaggle API."""
    try:
        import kaggle
        
        print("Downloading dataset using Kaggle API...")
        print("Dataset: paultimothymooney/chest-xray-pneumonia")
        print()
        
        # Create data directory
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        kaggle.api.dataset_download_files(
            "paultimothymooney/chest-xray-pneumonia",
            path=str(DATA_DIR),
            unzip=True
        )
        
        print(f"Dataset downloaded to: {DATA_DIR}")
        
        # Verify structure
        if (DATA_DIR / "chest_xray").exists():
            print("Dataset structure verified!")
            return True
        else:
            print("Warning: Dataset structure may be different than expected")
            return False
            
    except ImportError:
        print("kaggle package not installed.")
        print("Install with: pip install kaggle")
        print("Then set up API credentials: https://www.kaggle.com/docs/api")
        return False
    except Exception as e:
        print(f"Error downloading with Kaggle API: {e}")
        print("\nMake sure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Set up API credentials at: https://www.kaggle.com/account")
        print("3. Placed kaggle.json in ~/.kaggle/ (or C:/Users/YourUsername/.kaggle/ on Windows)")
        return False


def extract_zip(zip_path: Path):
    """Extract downloaded zip file."""
    print(f"Extracting {zip_path}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to data directory
            zip_ref.extractall(DATA_DIR)
        
        print(f"Extracted to: {DATA_DIR}")
        
        # Check if chest_xray folder exists
        if (DATA_DIR / "chest_xray").exists():
            print("Dataset structure verified!")
            return True
        else:
            # Look for chest_xray in subdirectories
            for item in DATA_DIR.iterdir():
                if item.is_dir() and 'chest' in item.name.lower():
                    print(f"Found dataset folder: {item}")
                    # Rename if needed
                    if item.name != "chest_xray":
                        target = DATA_DIR / "chest_xray"
                        if target.exists():
                            shutil.rmtree(target)
                        item.rename(target)
                        print(f"Renamed to: {target}")
                    return True
        
        print("Warning: Could not find chest_xray folder after extraction")
        print(f"Please check: {DATA_DIR}")
        return False
        
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return False


def main():
    """Main function to download dataset."""
    print("=" * 60)
    print("Chest X-Ray Pneumonia Dataset Downloader")
    print("=" * 60)
    print()
    
    # Check if dataset already exists
    if DATASET_DIR.exists():
        # Check if it has images
        train_normal = DATASET_DIR / "train" / "NORMAL"
        if train_normal.exists() and any(train_normal.glob("*.jpeg")):
            print(f"Dataset already exists at: {DATASET_DIR}")
            print("Skipping download.")
            print()
            print("To re-download, delete the dataset folder first:")
            print(f"  {DATASET_DIR}")
            return
    
    print("Choose download method:")
    print("1. kagglehub (Recommended - easiest)")
    print("2. Kaggle API (requires API setup)")
    print("3. Manual download (you download zip file manually)")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        success = download_with_kagglehub()
    elif choice == "2":
        success = download_with_kaggle_api()
    elif choice == "3":
        print()
        print("Manual Download Instructions:")
        print("1. Click 'Download dataset as zip' on Kaggle")
        print("2. Save the zip file to this directory")
        print("3. Run this script again and choose option 3")
        print()
        zip_file = input("Enter path to downloaded zip file: ").strip().strip('"')
        zip_path = Path(zip_file)
        if zip_path.exists():
            success = extract_zip(zip_path)
        else:
            print(f"Error: File not found: {zip_path}")
            success = False
    else:
        print("Invalid choice. Using kagglehub method...")
        success = download_with_kagglehub()
    
    if success:
        print()
        print("=" * 60)
        print("[SUCCESS] Dataset download complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Verify dataset: python setup_dataset.py")
        print("2. Train model: python src/training/train.py --epochs 10 --save-model")
    else:
        print()
        print("=" * 60)
        print("[ERROR] Dataset download failed")
        print("=" * 60)
        print()
        print("Please try:")
        print("1. Manual download from Kaggle website")
        print("2. Check your internet connection")
        print("3. Verify Kaggle API credentials (if using API method)")


if __name__ == "__main__":
    main()

