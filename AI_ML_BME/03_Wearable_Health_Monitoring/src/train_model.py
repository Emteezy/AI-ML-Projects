"""Script to train health classification models for edge deployment."""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import argparse

from src.config.settings import MODELS_DIR
from src.edge_ml.model_converter import convert_keras_model_to_tflite


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic health sensor data for training.
    
    In production, this would load real sensor data from the database.
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with synthetic sensor data and labels
    """
    np.random.seed(42)
    
    # Generate synthetic data
    data = []
    
    for i in range(n_samples):
        # Normal health (class 0)
        if i < n_samples * 0.7:
            hr = np.random.normal(75, 10)
            spo2 = np.random.normal(98, 1.5)
            accel_mag = np.random.normal(9.8, 0.5)
            label = 0  # normal
        
        # Warning health (class 1)
        elif i < n_samples * 0.9:
            hr = np.random.normal(105, 15)  # Elevated heart rate
            spo2 = np.random.normal(93, 2)  # Lower SpO2
            accel_mag = np.random.normal(12, 2)  # Higher activity
            label = 1  # warning
        
        # Critical health (class 2)
        else:
            hr = np.random.normal(130, 20)  # Very high heart rate
            spo2 = np.random.normal(88, 3)  # Low SpO2
            accel_mag = np.random.normal(15, 3)  # Very high activity
            label = 2  # critical
        
        # Clip values to realistic ranges
        hr = np.clip(hr, 40, 200)
        spo2 = np.clip(spo2, 70, 100)
        accel_mag = np.clip(accel_mag, 5, 25)
        
        data.append({
            "heart_rate": hr,
            "spo2": spo2,
            "acceleration_magnitude": accel_mag,
            "label": label,
        })
    
    return pd.DataFrame(data)


def create_model(input_dim: int, num_classes: int = 3) -> tf.keras.Model:
    """
    Create a neural network model for health classification.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


def train_model(
    data_path: Path = None,
    model_name: str = "health_classifier",
    epochs: int = 50,
    batch_size: int = 32,
    use_synthetic: bool = True,
    convert_to_tflite: bool = True,
):
    """
    Train health classification model.
    
    Args:
        data_path: Path to training data CSV. If None and use_synthetic=False, raises error.
        model_name: Name for the saved model
        epochs: Number of training epochs
        batch_size: Batch size for training
        use_synthetic: If True, generate synthetic data. If False, load from data_path.
        convert_to_tflite: If True, convert model to TFLite format
    """
    # Load or generate data
    if use_synthetic:
        print("Generating synthetic training data...")
        df = generate_synthetic_data(n_samples=10000)
    else:
        if data_path is None or not data_path.exists():
            raise ValueError("data_path must be provided when use_synthetic=False")
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
    
    # Prepare features and labels
    feature_cols = ["heart_rate", "spo2", "acceleration_magnitude"]
    X = df[feature_cols].values
    y = df["label"].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    import pickle
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Create model
    model = create_model(input_dim=X_train_scaled.shape[1], num_classes=3)
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=["normal", "warning", "critical"]))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))
    
    # Save model
    model_path = MODELS_DIR / f"{model_name}.h5"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Convert to TFLite
    if convert_to_tflite:
        print("\nConverting to TensorFlow Lite...")
        tflite_path = MODELS_DIR / f"{model_name}.tflite"
        convert_keras_model_to_tflite(
            model,
            tflite_path,
            quantize=False,  # Can enable quantization for smaller model
        )
        print(f"TFLite model saved to {tflite_path}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train health classification model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to training data CSV (optional, uses synthetic data if not provided)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="health_classifier",
        help="Name for the saved model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Disable synthetic data generation (requires --data-path)",
    )
    parser.add_argument(
        "--no-tflite",
        action="store_true",
        help="Skip TFLite conversion",
    )
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_synthetic=not args.no_synthetic,
        convert_to_tflite=not args.no_tflite,
    )

