"""Setup script for downloading and preparing MIT-BIH Arrhythmia Database."""

import os
import sys
from pathlib import Path
import subprocess
import shutil


def find_python_with_pip():
    """
    Find a Python executable that has pip available.
    Tries py launcher first, then python, then python3.
    """
    # Try py launcher first (Windows)
    for py_version in ['3.10', '3.11', '3.9', '3.13']:
        py_cmd = shutil.which('py')
        if py_cmd:
            try:
                result = subprocess.run(
                    [py_cmd, f'-{py_version}', '-m', 'pip', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return [py_cmd, f'-{py_version}']
            except:
                continue
    
    # Try standard python commands
    for cmd in ['python', 'python3']:
        python_cmd = shutil.which(cmd)
        if python_cmd:
            try:
                result = subprocess.run(
                    [python_cmd, '-m', 'pip', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return [python_cmd]
            except:
                continue
    
    return None


def check_wfdb_installed():
    """Check if wfdb package is installed."""
    try:
        import wfdb
        return True
    except ImportError:
        return False


def download_mitdb():
    """
    Download MIT-BIH Arrhythmia Database using wfdb.
    
    This function uses the wfdb package to download the dataset from PhysioNet.
    Note: You need to have a PhysioNet account and be logged in.
    """
    try:
        import wfdb
    except ImportError:
        print("Error: wfdb package not found. Please install it:")
        print("  pip install wfdb")
        sys.exit(1)
    
    # Target directory
    data_dir = Path(__file__).parent / "data" / "mitdb"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading MIT-BIH Arrhythmia Database...")
    print("Note: This requires a PhysioNet account.")
    print("If prompted, enter your PhysioNet username and password.")
    print()
    
    try:
        # Download the dataset
        wfdb.dl_database(
            db_dir='mitdb',
            dl_dir=str(data_dir.parent),
            records='all',
            annotators='atr'
        )
        
        print(f"\n[SUCCESS] Dataset downloaded successfully to {data_dir}")
        print(f"\nTo verify the download, check if files exist in: {data_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] Error downloading dataset: {e}")
        print("\nAlternative method:")
        print("1. Visit https://physionet.org/content/mitdb/1.0.0/")
        print("2. Download the dataset manually")
        print(f"3. Extract it to: {data_dir}")
        sys.exit(1)


def verify_dataset():
    """Verify that the dataset is properly set up."""
    data_dir = Path(__file__).parent / "data" / "mitdb"
    
    if not data_dir.exists():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        return False
    
    # Check for some expected files
    expected_files = ["100.atr", "100.dat", "100.hea"]
    files_found = sum(1 for f in expected_files if (data_dir / f).exists())
    
    if files_found == 0:
        print(f"[ERROR] No dataset files found in {data_dir}")
        print("Please download the dataset first.")
        return False
    elif files_found < len(expected_files):
        print(f"[WARNING] Only {files_found}/{len(expected_files)} expected files found.")
        print("Dataset may be incomplete.")
        return False
    else:
        print(f"[SUCCESS] Dataset verified. Found expected files in {data_dir}")
        return True


def main():
    """Main function."""
    print("=" * 60)
    print("MIT-BIH Arrhythmia Database Setup")
    print("=" * 60)
    print()
    
    # Check if dataset already exists
    if verify_dataset():
        print("\nDataset is already set up!")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Check if wfdb is installed
    if not check_wfdb_installed():
        print("\n[INFO] wfdb package not found.")
        print("Attempting to install wfdb...")
        
        # Find a Python with pip
        python_cmd = find_python_with_pip()
        if not python_cmd:
            print("\n[ERROR] Could not find a Python installation with pip.")
            print("\nPlease install dependencies manually:")
            print("  py -3.10 -m pip install -r requirements.txt")
            print("\nOr install wfdb directly:")
            print("  py -3.10 -m pip install wfdb")
            sys.exit(1)
        
        try:
            print(f"Using: {' '.join(python_cmd)}")
            subprocess.check_call(python_cmd + ["-m", "pip", "install", "wfdb"])
            print("[SUCCESS] wfdb installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Could not install wfdb automatically (exit code {e.returncode}).")
            print(f"\nPlease install it manually:")
            print(f"  {' '.join(python_cmd)} -m pip install wfdb")
            sys.exit(1)
    
    # Download dataset
    download_mitdb()
    
    # Verify
    print("\nVerifying dataset...")
    if verify_dataset():
        print("\n[SUCCESS] Setup complete!")
    else:
        print("\n[WARNING] Setup may be incomplete. Please check the dataset manually.")


if __name__ == "__main__":
    main()

