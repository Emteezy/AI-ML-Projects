"""Convert TensorFlow models to TensorFlow Lite format."""

import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, Any
from src.config.settings import MODELS_DIR


def convert_keras_model_to_tflite(
    model: tf.keras.Model,
    output_path: Path,
    quantize: bool = False,
    optimization: str = "default"
) -> Path:
    """
    Convert a Keras model to TensorFlow Lite format.
    
    Args:
        model: Trained Keras model
        output_path: Path to save the TFLite model
        quantize: If True, apply quantization for smaller model size
        optimization: Optimization type ('default', 'float16', 'int8')
    
    Returns:
        Path to saved TFLite model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    if optimization == "float16" or quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif optimization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    # Convert model
    try:
        tflite_model = converter.convert()
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model converted successfully: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return output_path
    
    except Exception as e:
        raise Exception(f"Error converting model to TFLite: {e}")


def convert_saved_model_to_tflite(
    saved_model_path: Path,
    output_path: Path,
    quantize: bool = False
) -> Path:
    """
    Convert a saved TensorFlow model to TFLite format.
    
    Args:
        saved_model_path: Path to saved model directory
        output_path: Path to save the TFLite model
        quantize: If True, apply quantization
    
    Returns:
        Path to saved TFLite model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted successfully: {output_path}")
    return output_path


def get_model_info(tflite_model_path: Path) -> Dict[str, Any]:
    """
    Get information about a TFLite model.
    
    Args:
        tflite_model_path: Path to TFLite model file
    
    Returns:
        Dictionary with model information
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    model_size = tflite_model_path.stat().st_size / 1024  # Size in KB
    
    info = {
        "model_path": str(tflite_model_path),
        "model_size_kb": model_size,
        "inputs": [
            {
                "name": detail.get("name", "unknown"),
                "shape": detail["shape"].tolist(),
                "dtype": str(detail["dtype"]),
            }
            for detail in input_details
        ],
        "outputs": [
            {
                "name": detail.get("name", "unknown"),
                "shape": detail["shape"].tolist(),
                "dtype": str(detail["dtype"]),
            }
            for detail in output_details
        ],
    }
    
    return info

