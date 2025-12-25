# ECG Arrhythmia Detection System 💓

A **production-ready** biosignal processing system for real-time ECG arrhythmia classification. This project demonstrates time-series analysis, signal processing, and deep learning for biomedical applications.

## 🎯 Overview

This project provides a complete solution for ECG signal analysis, specifically designed for:
- **ECG Arrhythmia Classification** (Normal, Atrial Fibrillation, Premature Ventricular Contraction, etc.)
- **Time-Series Deep Learning** using LSTM and Transformer models
- **Real-time Signal Processing** with filtering and feature extraction
- **Production API** with FastAPI for real-time inference
- **Interactive Web Interface** with Streamlit for signal visualization
- **Docker Deployment** for easy containerization

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │
       │ HTTP/REST
       │
┌──────▼──────────────────┐
│   Streamlit Frontend    │
│   (Port 8501)           │
│   - Signal Upload       │
│   - Real-time Plot      │
│   - Prediction Display  │
└──────┬──────────────────┘
       │
       │ API Calls
       │
┌──────▼──────────────────┐
│   FastAPI Backend       │
│   (Port 8000)           │
│   - /predict            │
│   - /analyze            │
│   - /health             │
│   - /docs (Swagger UI)  │
└──────┬──────────────────┘
       │
       │ Signal Processing
       │
┌──────▼──────────────────┐
│   PyTorch Model         │
│   (LSTM/Transformer)    │
│   Time-Series Classifier │
└─────────────────────────┘
```

## ✨ Key Features

- ✅ **Time-Series Deep Learning**: LSTM and Transformer architectures
- ✅ **Signal Processing**: ECG filtering, denoising, feature extraction
- ✅ **Real-time Analysis**: Process ECG signals in real-time
- ✅ **Production API**: FastAPI with async support
- ✅ **Interactive UI**: Streamlit with signal visualization
- ✅ **Docker Deployment**: Containerized application
- ✅ **Multiple Arrhythmia Types**: Classify various cardiac conditions
- ✅ **Signal Visualization**: Plot ECG waveforms with annotations

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

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- Docker (optional, for containerized deployment)
- GPU (optional, but recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd 02_ECG_Arrhythmia_Detection
   ```

2. **Create and activate virtual environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows PowerShell)
   venv\Scripts\Activate.ps1
   
   # Activate (Linux/Mac)
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

The project uses the **MIT-BIH Arrhythmia Database** from PhysioNet.

1. **Download the dataset** from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
   - Create a free PhysioNet account if needed
   - Download the MIT-BIH Arrhythmia Database

2. **Extract to `data/mitdb/` directory**

3. **Verify dataset setup:**
   ```bash
   python setup_dataset.py
   ```

### Training the Model

Train the model with time-series data:

```bash
# Activate virtual environment first
venv\Scripts\Activate.ps1  # Windows PowerShell

# Run training
python -m src.training.train \
    --epochs 20 \
    --batch-size 32 \
    --model lstm \
    --learning-rate 0.001 \
    --save-model
```

**Training time:** ~1-2 hours on GPU, ~4-6 hours on CPU

### Running the API

Start the FastAPI server:

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Running the Web Interface

Start the Streamlit app (in a new terminal):

```bash
streamlit run app/streamlit_app.py
```

The web interface will open at: http://localhost:8501

## 📖 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "lstm_v1"
}
```

#### `POST /predict`
Classify ECG signal (arrhythmia type).

**Request:**
- Content-Type: `application/json`
- Body: ECG signal array (list of floats)

**Response:**
```json
{
  "prediction": "Atrial Fibrillation",
  "confidence": 0.92,
  "class_probabilities": {
    "Normal": 0.05,
    "Atrial Fibrillation": 0.92,
    "Premature Ventricular Contraction": 0.03
  },
  "model_version": "lstm_v1"
}
```

#### `POST /analyze`
Get detailed ECG analysis with signal features.

**Request:**
- Content-Type: `application/json`
- Body: ECG signal array

**Response:**
```json
{
  "prediction": "Atrial Fibrillation",
  "confidence": 0.92,
  "features": {
    "heart_rate": 95,
    "qrs_duration": 0.08,
    "rr_interval": 0.63
  },
  "signal_quality": "good"
}
```

## 🎓 Model Details

### Architecture

The system supports multiple time-series architectures:
- **LSTM**: Long Short-Term Memory networks for sequential patterns
- **Transformer**: Attention-based models for long-range dependencies
- **CNN-LSTM**: Hybrid architecture combining convolutional and recurrent layers

### Signal Processing Pipeline

1. **Preprocessing**: Baseline correction, noise removal
2. **Filtering**: Bandpass filter (0.5-40 Hz), notch filter (50/60 Hz)
3. **Feature Extraction**: QRS detection, heart rate, RR intervals
4. **Normalization**: Signal standardization
5. **Windowing**: Sliding window for real-time processing

### Performance

After training for 20 epochs, typical results:
- **Training Accuracy**: ~92-96%
- **Validation Accuracy**: ~88-93%
- **Test Accuracy**: ~85-90%
- **Model Size**: ~5-10MB (LSTM), ~15-20MB (Transformer)

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines:

```bash
black src/ tests/ app/
flake8 src/ tests/ app/
```

## 📊 Results

Training results and visualizations are saved in the `results/` directory:
- Training/validation loss curves
- Accuracy curves
- Confusion matrix
- Sample ECG signals with predictions
- Feature importance analysis

## 🎯 Use Cases

- **Clinical Decision Support**: Assist cardiologists in arrhythmia detection
- **Remote Monitoring**: Real-time ECG analysis for telemedicine
- **Research**: Baseline for biosignal processing research
- **Education**: Teaching tool for biomedical signal processing

## ⚠️ Important Notes

- **This is a research/educational project** - Not for clinical use
- **Always consult medical professionals** for actual diagnosis
- **Model performance** may vary with different datasets
- **Data privacy** - Ensure HIPAA compliance if using real patient data

## 🚧 Future Enhancements

- [ ] Support for multi-lead ECG signals
- [ ] Real-time streaming ECG analysis
- [ ] Integration with wearable devices
- [ ] Mobile app for ECG monitoring
- [ ] Model ensemble capabilities
- [ ] Cloud deployment guides (AWS, GCP, Azure)

## 📄 License

This project is open source and available under the MIT License.

---

**Project Status**: 🚧 In Development  
**Last Updated**: 2024-12-21

