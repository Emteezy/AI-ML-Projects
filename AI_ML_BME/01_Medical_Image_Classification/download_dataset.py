"""
Script to download the Chest X-Ray Pneumonia dataset.
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "chest_xray"
ZIP_PATH = DATA_DIR / "chest_xray.zip"

# Dataset URL (public mirror - if Kaggle API is not available)
# Note: This is a fallback. Primary method uses Kaggle API
DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/download"


def download_with_progress(url: str, destination: Path):
    """Download file with progress bar."""
    def reporthook(count, block_size, total_size):
        downloaded = count * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
    
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, destination, reporthook=reporthook)
    print("\nDownload complete!")


def download_with_kaggle_api():
    """Download dataset using Kaggle API."""
    try:
        import kaggle
        print("Using Kaggle API to download dataset...")
        
        # Kaggle dataset identifier
        dataset = "paultimothymooney/chest-xray-pneumonia"
        
        # Download to data directory
        kaggle.api.dataset_download_files(
            dataset,
            path=str(DATA_DIR),
            unzip=True
        )
        
        print("Dataset downloaded successfully using Kaggle API!")
        return True
    except ImportError:
        print("Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error using Kaggle API: {e}")
        print("Make sure you have:")
        print("1. Installed kaggle: pip install kaggle")
        print("2. Set up Kaggle API credentials:")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Click 'Create New API Token'")
        print("   - Save kaggle.json to ~/.kaggle/kaggle.json (or C:/Users/<username>/.kaggle/kaggle.json on Windows)")
        return False


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total number of files for progress
        total_files = len(zip_ref.namelist())
        
        # Extract with progress
        for i, member in enumerate(zip_ref.namelist()):
            zip_ref.extract(member, extract_to)
            if (i + 1) % 100 == 0:
                print(f"Extracted {i + 1}/{total_files} files...")
        
        print(f"Extraction complete! Extracted {total_files} files.")


def organize_dataset():
    """Organize the dataset into the expected structure."""
    print("\nOrganizing dataset structure...")
    
    # Check if dataset is already in the correct structure
    if (DATASET_PATH / "train" / "NORMAL").exists() and (DATASET_PATH / "train" / "PNEUMONIA").exists():
        print("Dataset is already organized correctly!")
        return
    
    # Look for extracted folders in data directory
    extracted_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name != "chest_xray"]
    
    for extracted_dir in extracted_dirs:
        # Check if this looks like the dataset
        if (extracted_dir / "train").exists():
            print(f"Found dataset in {extracted_dir}, moving to {DATASET_PATH}...")
            if DATASET_PATH.exists():
                shutil.rmtree(DATASET_PATH)
            shutil.move(str(extracted_dir), str(DATASET_PATH))
            print("Dataset organized!")
            return
    
    # If we have a chest_xray folder already, check its structure
    if DATASET_PATH.exists():
        # Check if it needs reorganization
        if (DATASET_PATH / "chest_xray" / "train").exists():
            # Nested structure, need to flatten
            nested = DATASET_PATH / "chest_xray"
            temp_path = DATA_DIR / "temp_chest_xray"
            shutil.move(str(nested), str(temp_path))
            shutil.rmtree(DATASET_PATH)
            shutil.move(str(temp_path), str(DATASET_PATH))
            print("Reorganized nested structure!")
        else:
            print("Dataset structure looks correct!")


def verify_dataset():
    """Verify that the dataset is properly organized."""
    print("\nVerifying dataset structure...")
    
    required_dirs = [
        DATASET_PATH / "train" / "NORMAL",
        DATASET_PATH / "train" / "PNEUMONIA",
        DATASET_PATH / "val" / "NORMAL",
        DATASET_PATH / "val" / "PNEUMONIA",
        DATASET_PATH / "test" / "NORMAL",
        DATASET_PATH / "test" / "PNEUMONIA",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"[X] Missing: {dir_path}")
            all_exist = False
        else:
            # Count images
            image_count = len(list(dir_path.glob("*.jpeg"))) + \
                         len(list(dir_path.glob("*.jpg"))) + \
                         len(list(dir_path.glob("*.png")))
            print(f"[OK] {dir_path.name}: {image_count} images")
    
    if all_exist:
        print("\n[OK] Dataset verification successful!")
        return True
    else:
        print("\n[X] Dataset structure is incomplete!")
        return False


def main():
    """Main download function."""
    print("=" * 60)
    print("Chest X-Ray Pneumonia Dataset Downloader")
    print("=" * 60)
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    if verify_dataset():
        print("\nDataset already exists and is properly organized!")
        response = input("Do you want to re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    # Try Kaggle API first
    print("\nAttempting to download using Kaggle API...")
    if download_with_kaggle_api():
        organize_dataset()
        if verify_dataset():
            # Clean up zip file if it exists
            if ZIP_PATH.exists():
                ZIP_PATH.unlink()
            print("\n[OK] Dataset download and setup complete!")
            return
    
    # Fallback: Manual download instructions
    print("\n" + "=" * 60)
    print("Kaggle API download failed. Please download manually:")
    print("=" * 60)
    print("\nOPTION 1 - Manual Download (Easiest):")
    print("1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Sign in to Kaggle (create free account if needed)")
    print("3. Click 'Download' button (top right)")
    print(f"4. Save the zip file to: {DATA_DIR}")
    print("5. Run this script again - it will automatically extract and organize")
    print("\nOPTION 2 - Set up Kaggle API (for automated downloads):")
    print("1. Go to: https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Save kaggle.json to: C:\\Users\\ASUS\\.kaggle\\kaggle.json")
    print("5. Run this script again")
    print(f"\nAfter downloading, expected structure: {DATASET_PATH}/train/NORMAL, etc.")
    print("=" * 60)
    
    # Check if zip file exists (user may have downloaded manually)
    zip_files = list(DATA_DIR.glob("*.zip"))
    if zip_files:
        print(f"\nFound zip file(s): {zip_files}")
        for zip_file in zip_files:
            print(f"\nProcessing {zip_file.name}...")
            extract_zip(zip_file, DATA_DIR)
            organize_dataset()
            if verify_dataset():
                zip_file.unlink()  # Remove zip after successful extraction
                print("\n[OK] Dataset setup complete!")
                return
        print("\n[X] Dataset extraction completed but structure verification failed.")
        print("Please check the dataset structure manually.")
    else:
        print("\nNo zip file found. Please download the dataset first (see instructions above).")


if __name__ == "__main__":
    main()

