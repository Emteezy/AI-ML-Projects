"""Main script to run the edge device (sensor reading and streaming)."""

import time
import logging
import signal
import sys
from typing import Optional
from src.sensors import HeartRateSensor, PulseOximeterSensor, AccelerometerSensor
from src.iot import DataStreamer
from src.edge_ml import EdgeMLInference
from src.processing import AnomalyDetector
from src.config.settings import SENSOR_CONFIG, DEVICE_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EdgeDevice:
    """Edge device for health monitoring."""
    
    def __init__(self):
        """Initialize edge device with sensors and services."""
        logger.info("Initializing edge device...")
        logger.info(f"Device ID: {DEVICE_CONFIG['device_id']}")
        logger.info(f"Simulation Mode: {SENSOR_CONFIG['simulation_mode']}")
        
        # Initialize sensors
        self.heart_rate_sensor = HeartRateSensor()
        self.spo2_sensor = PulseOximeterSensor()
        self.accelerometer = AccelerometerSensor()
        
        # Initialize data streamer
        self.data_streamer = DataStreamer()
        
        # Initialize edge ML inference
        self.ml_inference = EdgeMLInference()
        
        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector()
        
        # Running state
        self.running = False
        
        # Statistics
        self.readings_count = 0
        self.predictions_count = 0
    
    def connect(self) -> bool:
        """Connect to MQTT broker."""
        logger.info("Connecting to MQTT broker...")
        return self.data_streamer.connect()
    
    def read_sensors(self) -> list:
        """
        Read all sensors.
        
        Returns:
            List of sensor readings
        """
        readings = []
        
        # Read heart rate
        if self.heart_rate_sensor.is_available():
            hr_reading = self.heart_rate_sensor.read()
            readings.append(hr_reading)
        
        # Read SpO2
        if self.spo2_sensor.is_available():
            spo2_reading = self.spo2_sensor.read()
            readings.append(spo2_reading)
        
        # Read accelerometer
        if self.accelerometer.is_available():
            accel_reading = self.accelerometer.read()
            readings.append(accel_reading)
        
        return readings
    
    def process_readings(self, readings: list):
        """
        Process sensor readings (ML inference, anomaly detection, streaming).
        
        Args:
            readings: List of sensor readings
        """
        if not readings:
            return
        
        # Stream readings to MQTT
        for reading in readings:
            self.data_streamer.publish_sensor_reading(reading)
            self.readings_count += 1
        
        # Run ML inference if model is available
        if self.ml_inference.is_loaded:
            try:
                health_status = self.ml_inference.predict_health_status(readings)
                if health_status.get("status") != "unknown":
                    self.data_streamer.publish_health_status(health_status)
                    self.predictions_count += 1
                    logger.debug(f"Health status: {health_status['status']} (confidence: {health_status['confidence']:.2%})")
            except Exception as e:
                logger.error(f"Error in ML inference: {e}")
        
        # Check for anomalies
        for reading in readings:
            sensor_type = reading.get("sensor_type")
            try:
                is_anomaly, severity, message = self.anomaly_detector.detect_using_thresholds(reading)
                if is_anomaly:
                    anomaly = {
                        "timestamp": reading.get("timestamp", time.time()),
                        "severity": severity,
                        "message": message,
                        "sensor_type": sensor_type,
                        "reading": reading,
                    }
                    self.data_streamer.publish_anomaly_alert(anomaly)
                    logger.warning(f"Anomaly detected: {message}")
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
    
    def run(self, interval: float = 1.0):
        """
        Run the edge device (read sensors and stream data).
        
        Args:
            interval: Time interval between sensor readings in seconds
        """
        logger.info("Starting edge device...")
        
        if not self.connect():
            logger.error("Failed to connect to MQTT broker. Exiting.")
            return
        
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                
                # Read sensors
                readings = self.read_sensors()
                
                # Process readings
                if readings:
                    self.process_readings(readings)
                    logger.debug(f"Processed {len(readings)} sensor readings")
                
                # Sleep to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Log statistics periodically
                if self.readings_count % 100 == 0 and self.readings_count > 0:
                    stats = self.data_streamer.get_statistics()
                    logger.info(
                        f"Statistics - Readings: {self.readings_count}, "
                        f"Predictions: {self.predictions_count}, "
                        f"Messages sent: {stats['messages_sent']}"
                    )
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the edge device."""
        logger.info("Stopping edge device...")
        self.running = False
        self.data_streamer.disconnect()
        logger.info("Edge device stopped.")


def main():
    """Main function."""
    device = EdgeDevice()
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        device.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run device
    # Adjust interval based on sensor sampling rates
    device.run(interval=1.0)  # Read sensors every second


if __name__ == "__main__":
    main()

