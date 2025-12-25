# PowerShell script to set up the Python environment for ECG Arrhythmia Detection
# This script helps avoid MSYS Python issues by using the Python Launcher

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ECG Project Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if py launcher is available
$pyAvailable = Get-Command py -ErrorAction SilentlyContinue

if (-not $pyAvailable) {
    Write-Host "[ERROR] Python Launcher (py) not found!" -ForegroundColor Red
    Write-Host "Please install Python from python.org" -ForegroundColor Yellow
    exit 1
}

# List available Python versions
Write-Host "Available Python versions:" -ForegroundColor Green
py --list
Write-Host ""

# Use Python 3.10 (or latest available)
$pythonVersion = "3.10"
Write-Host "Using Python $pythonVersion" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "[INFO] Virtual environment already exists" -ForegroundColor Yellow
    $response = Read-Host "Recreate it? (y/n)"
    if ($response -eq "y") {
        Remove-Item -Recurse -Force venv
        Write-Host "[INFO] Removed old virtual environment" -ForegroundColor Yellow
    } else {
        Write-Host "[INFO] Using existing virtual environment" -ForegroundColor Green
    }
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    py -$pythonVersion -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "[SUCCESS] Environment setup complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Activate the virtual environment:" -ForegroundColor White
    Write-Host "   venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Download the dataset:" -ForegroundColor White
    Write-Host "   python setup_dataset.py" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "[ERROR] Failed to install requirements" -ForegroundColor Red
    exit 1
}

