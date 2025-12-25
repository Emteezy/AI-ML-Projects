"""Edge ML inference using TensorFlow Lite."""

import numpy as np
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import tensorflow as tf
from src.config.settings import MODELS_DIR


class EdgeMLInference:
    """TensorFlow Lite inference engine for edge devices."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize edge ML inference engine.
        
        Args:
            model_path: Path to TFLite model file. If None, uses default path.
        """
        self.model_path = model_path or MODELS_DIR / "health_classifier.tflite"
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.is_loaded = False
        
        if self.model_path.exists():
            self._load_model()
        else:
            print(f"Warning: Model not found at {self.model_path}")
            print("Running without ML inference. Train a model first.")
    
    def _load_model(self):
        """Load TensorFlow Lite model."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
            
            self.is_loaded = True
            print(f"Edge ML model loaded: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False
    
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on input data.
        
        Args:
            input_data: Input data array (should match model input shape)
        
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded:
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": "Model not loaded",
            }
        
        try:
            # Ensure input data matches expected shape
            if input_data.shape != tuple(self.input_shape[1:]):  # Skip batch dimension
                input_data = self._preprocess_input(input_data)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data.astype(np.float32))
            
            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process output (assuming classification)
            if len(output[0]) > 1:  # Multi-class classification
                prediction_idx = np.argmax(output[0])
                confidence = float(output[0][prediction_idx])
                predictions = {
                    "class": int(prediction_idx),
                    "confidence": confidence,
                    "probabilities": output[0].tolist(),
                }
            else:  # Binary classification or regression
                predictions = {
                    "value": float(output[0][0]),
                    "confidence": 1.0,
                }
            
            return {
                "prediction": predictions,
                "inference_time_ms": inference_time * 1000,
                "timestamp": time.time(),
            }
        
        except Exception as e:
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": str(e),
            }
    
    def _preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data to match model requirements.
        
        Args:
            input_data: Input data array
        
        Returns:
            Preprocessed data array
        """
        # Reshape if needed
        expected_shape = self.input_shape[1:]  # Remove batch dimension
        
        # If input is 1D but model expects 2D, add batch dimension handling
        if len(input_data.shape) == 1 and len(expected_shape) == 1:
            if input_data.shape[0] != expected_shape[0]:
                # Resize or pad as needed
                if input_data.shape[0] < expected_shape[0]:
                    # Pad with zeros
                    padding = expected_shape[0] - input_data.shape[0]
                    input_data = np.pad(input_data, (0, padding), mode='constant')
                else:
                    # Truncate
                    input_data = input_data[:expected_shape[0]]
        
        # Add batch dimension if needed
        if len(input_data.shape) == len(expected_shape):
            input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def predict_health_status(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict health status from sensor data.
        
        Args:
            sensor_data: List of sensor readings
        
        Returns:
            Health status prediction
        """
        if not self.is_loaded or len(sensor_data) == 0:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "message": "Model not available or no data",
            }
        
        # Extract features from sensor data
        # This should match the features used during training
        features = self._extract_features_for_prediction(sensor_data)
        
        if features is None:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "message": "Failed to extract features",
            }
        
        # Run prediction
        result = self.predict(features)
        
        if result.get("error"):
            return {
                "status": "error",
                "confidence": 0.0,
                "message": result["error"],
            }
        
        # Map prediction to health status
        prediction = result["prediction"]
        if isinstance(prediction, dict) and "class" in prediction:
            class_labels = ["normal", "warning", "critical"]
            status = class_labels[prediction["class"]] if prediction["class"] < len(class_labels) else "unknown"
        else:
            status = "unknown"
        
        return {
            "status": status,
            "confidence": prediction.get("confidence", 0.0) if isinstance(prediction, dict) else 0.0,
            "inference_time_ms": result.get("inference_time_ms", 0),
            "timestamp": result.get("timestamp", time.time()),
        }
    
    def _extract_features_for_prediction(self, sensor_data: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        Extract features from sensor data for model prediction.
        
        Args:
            sensor_data: List of sensor readings
        
        Returns:
            Feature array or None if extraction fails
        """
        if len(sensor_data) == 0:
            return None
        
        # Extract features (simplified - should match training pipeline)
        features = []
        
        for reading in sensor_data:
            feature_vector = []
            
            # Heart rate feature
            if "heart_rate" in reading:
                feature_vector.append(reading["heart_rate"])
            
            # SpO2 feature
            if "spo2" in reading:
                feature_vector.append(reading["spo2"])
            
            # Accelerometer features
            if "acceleration" in reading:
                accel = reading["acceleration"]
                feature_vector.extend([
                    accel.get("x", 0),
                    accel.get("y", 0),
                    accel.get("z", 0),
                    accel.get("magnitude", 0),
                ])
            
            if feature_vector:
                features.append(feature_vector)
        
        if not features:
            return None
        
        # Convert to numpy array
        features_array = np.array(features)
        
        # If we need a single feature vector (e.g., mean of recent readings)
        if len(features_array.shape) == 2 and features_array.shape[0] > 1:
            # Use mean of recent readings
            features_array = np.mean(features_array, axis=0)
        
        return features_array

