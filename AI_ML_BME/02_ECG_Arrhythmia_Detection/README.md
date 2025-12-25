# ECG Arrhythmia Detection

Real-time ECG signal processing and arrhythmia classification using deep learning.

## Overview

Automated ECG analysis system for detecting cardiac arrhythmias:
- Classification of multiple arrhythmia types (Normal, AFib, PVC, etc.)
- LSTM and Transformer models for time-series analysis
- Digital signal processing pipeline (filtering, feature extraction)
- REST API for real-time inference
- Web interface for signal visualization
- Docker containerization

## Architecture

```
Client (Browser)
       │
       ├─> Streamlit Frontend (Port 8501)
       │   - Signal upload and visualization
       │   - Real-time plotting
       │   - Prediction display
       │
       ├─> FastAPI Backend (Port 8000)
       │   - /predict, /analyze, /health
       │   - Swagger UI docs
       │
       └─> PyTorch Model (LSTM/Transformer)
           Time-series classification
```

## Features

- Time-series deep learning (LSTM, Transformer)
- ECG signal preprocessing (filtering, denoising, feature extraction)
- Real-time signal analysis
- FastAPI with async support
- Interactive Streamlit dashboard
- Multiple arrhythmia type classification
- Signal visualization with annotations

## 📁 Project Structure

```
02_ECG_Arrhythmia_Detection/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Docker Compose configuration
├── .gitignore              # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py    # LSTM time-series classifier
│   │   ├── transformer_model.py  # Transformer model
│   │   └── model_utils.py   # Model loading/saving utilities
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py          # FastAPI application
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py         # Training script
│   │   └── dataset.py       # ECG dataset loader
│   ├── signal_processing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # ECG signal preprocessing
│   │   ├── filtering.py     # Signal filtering and denoising
│   │   └── features.py      # Feature extraction
│   └── config/
│       └── settings.py       # Configuration settings
├── app/
│   └── streamlit_app.py     # Streamlit web interface
├── notebooks/
│   ├── data_exploration.ipynb
│   └── signal_analysis.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # API tests
│   ├── test_model.py        # Model tests
│   └── test_signal_processing.py
├── data/                    # Data directory (created automatically)
├── models/                  # Trained models (created automatically)
└── results/                 # Training results and visualizations
```

## Quick Start

**Prerequisites:** Python 3.8+, Docker (optional), GPU (recommended for training)

**Install:**
```bash
cd 02_ECG_Arrhythmia_Detection
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

**Dataset:** MIT-BIH Arrhythmia Database from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- Download and extract to `data/mitdb/`
- Run `python setup_dataset.py` to verify

**Train:**
```bash
python -m src.training.train --epochs 20 --batch-size 32 --model lstm
```

**Run API:**
```bash
python -m uvicorn src.api.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**Run Dashboard:**
```bash
streamlit run app/streamlit_app.py
```
Dashboard: http://localhost:8501

## API Endpoints

**GET /health** - Health check
```json
{"status": "healthy", "model_loaded": true}
```

**POST /predict** - Classify ECG signal
```json
{
  "prediction": "Atrial Fibrillation",
  "confidence": 0.92,
  "class_probabilities": {...}
}
```

**POST /analyze** - Detailed ECG analysis
```json
{
  "prediction": "Atrial Fibrillation",
  "confidence": 0.92,
  "features": {
    "heart_rate": 95,
    "qrs_duration": 0.08,
    "rr_interval": 0.63
  }
}
```

## Models

**Architectures:**
- LSTM - Long Short-Term Memory for sequential patterns
- Transformer - Attention-based for long-range dependencies
- CNN-LSTM - Hybrid convolutional + recurrent

**Signal Processing:**
1. Baseline correction and noise removal
2. Bandpass filter (0.5-40 Hz), notch filter (50/60 Hz)
3. QRS detection, heart rate, RR intervals
4. Signal standardization
5. Sliding window for real-time processing

**Performance (20 epochs):**
- Training: 92-96%
- Validation: 88-93%
- Test: 85-90%

## Testing

```bash
pytest tests/
```

## Important Notes

- Research/educational project - not for clinical use
- Consult medical professionals for diagnosis
- Ensure HIPAA compliance with real patient data

## License

MIT