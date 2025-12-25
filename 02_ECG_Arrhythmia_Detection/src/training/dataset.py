"""Dataset loader for ECG arrhythmia detection."""

import numpy as np
import wfdb
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset
import torch

from ..config.settings import (
    MITDB_DIR,
    SIGNAL_CONFIG,
    ANNOTATION_MAP,
    ARRHYTHMIA_CLASSES,
)
from ..signal_processing import preprocess_ecg_signal, denoise_signal


def load_mitdb_data(
    data_dir: Path = None,
    record_names: Optional[List[str]] = None,
    lead: int = 0
) -> List[Dict]:
    """
    Load MIT-BIH Arrhythmia Database records.
    
    Args:
        data_dir: Path to MIT-BIH database directory
        record_names: List of record names to load (if None, loads all)
        lead: Lead index to use (0 for MLII, 1 for V1)
        
    Returns:
        List of dictionaries containing signal data and annotations
    """
    if data_dir is None:
        data_dir = MITDB_DIR
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"MIT-BIH database directory not found at {data_dir}. "
            "Please download the dataset from PhysioNet."
        )
    
    # Get list of record files
    if record_names is None:
        # Standard MIT-BIH record names (101-234, but not all exist)
        record_names = [f"{i:03d}" for i in range(100, 235)]
    
    records = []
    
    for record_name in record_names:
        record_path = data_dir / record_name
        
        try:
            # Read signal
            record = wfdb.rdrecord(str(record_path))
            
            # Read annotations
            annotation = wfdb.rdann(str(record_path), "atr")
            
            # Get signal from specified lead
            signal_data = record.p_signal[:, lead] if record.n_sig > lead else record.p_signal[:, 0]
            sampling_rate = record.fs
            
            records.append({
                "name": record_name,
                "signal": signal_data,
                "sampling_rate": sampling_rate,
                "annotations": annotation.symbol,
                "annotation_indices": annotation.sample,
            })
            
        except FileNotFoundError:
            # Record doesn't exist, skip
            continue
        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
            continue
    
    return records


def create_segments_from_records(
    records: List[Dict],
    window_size: int = None,
    overlap: float = 0.0,
    min_segments_per_record: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labeled segments from records.
    
    Args:
        records: List of record dictionaries
        window_size: Size of each segment in samples
        overlap: Overlap ratio between segments
        min_segments_per_record: Minimum segments per record
        
    Returns:
        Tuple of (segments, labels) as numpy arrays
    """
    if window_size is None:
        window_size = SIGNAL_CONFIG["window_size"]
    
    all_segments = []
    all_labels = []
    
    for record in records:
        signal = record["signal"]
        annotations = record["annotations"]
        annotation_indices = record["annotation_indices"]
        sampling_rate = record["sampling_rate"]
        
        # Preprocess signal
        processed_signal = preprocess_ecg_signal(signal, sampling_rate)
        processed_signal = denoise_signal(processed_signal, sampling_rate)
        
        # Create segments
        step_size = int(window_size * (1 - overlap))
        segments_from_record = []
        labels_from_record = []
        
        for start_idx in range(0, len(processed_signal) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            segment = processed_signal[start_idx:end_idx]
            
            # Find dominant annotation in this segment
            segment_center = start_idx + window_size // 2
            closest_ann_idx = np.argmin(np.abs(annotation_indices - segment_center))
            ann_symbol = annotations[closest_ann_idx]
            
            # Map annotation to class
            class_name = ANNOTATION_MAP.get(ann_symbol, "Normal")
            if class_name not in ARRHYTHMIA_CLASSES:
                class_name = "Normal"
            
            class_idx = ARRHYTHMIA_CLASSES.index(class_name)
            
            segments_from_record.append(segment)
            labels_from_record.append(class_idx)
        
        # If we have enough segments, add them
        if len(segments_from_record) >= min_segments_per_record:
            all_segments.extend(segments_from_record)
            all_labels.extend(labels_from_record)
    
    return np.array(all_segments), np.array(all_labels)


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG arrhythmia detection.
    """
    
    def __init__(
        self,
        segments: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            segments: Array of ECG signal segments (N, window_size)
            labels: Array of class labels (N,)
            transform: Optional transform to apply to segments
        """
        self.segments = segments
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (signal, label)
        """
        segment = self.segments[idx].astype(np.float32)
        label = self.labels[idx]
        
        # Apply transform if provided
        if self.transform:
            segment = self.transform(segment)
        
        # Convert to tensor and add channel dimension
        # Shape: (window_size,) -> (1, window_size) -> (window_size, 1) for LSTM
        segment = torch.from_numpy(segment).unsqueeze(-1)  # (window_size, 1)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return segment, label

