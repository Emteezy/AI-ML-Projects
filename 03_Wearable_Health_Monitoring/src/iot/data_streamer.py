"""Data streaming module for sensor data transmission."""

import time
import logging
from typing import Dict, Any, Optional, List
from src.iot.mqtt_client import MQTTClient
from src.config.settings import DEVICE_CONFIG

logger = logging.getLogger(__name__)


class DataStreamer:
    """Stream sensor data to MQTT broker."""
    
    def __init__(self, mqtt_client: Optional[MQTTClient] = None):
        """
        Initialize data streamer.
        
        Args:
            mqtt_client: MQTT client instance. If None, creates a new one.
        """
        self.mqtt_client = mqtt_client or MQTTClient()
        self.device_id = DEVICE_CONFIG["device_id"]
        self.is_streaming = False
        self.streaming_thread = None
        
        # Statistics
        self.messages_sent = 0
        self.last_sent_time = None
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            True if connected successfully
        """
        return self.mqtt_client.connect()
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        self.stop_streaming()
        self.mqtt_client.disconnect()
    
    def publish_sensor_reading(self, reading: Dict[str, Any]) -> bool:
        """
        Publish a single sensor reading.
        
        Args:
            reading: Sensor reading dictionary
        
        Returns:
            True if published successfully
        """
        sensor_type = reading.get("sensor_type", "unknown")
        topic_suffix = f"sensors/{sensor_type}"
        
        payload = {
            "device_id": self.device_id,
            "timestamp": reading.get("timestamp", time.time()),
            "sensor_type": sensor_type,
            "data": reading,
        }
        
        success = self.mqtt_client.publish(topic_suffix, payload)
        if success:
            self.messages_sent += 1
            self.last_sent_time = time.time()
        
        return success
    
    def publish_health_status(self, status: Dict[str, Any]) -> bool:
        """
        Publish health status (from ML inference).
        
        Args:
            status: Health status dictionary
        
        Returns:
            True if published successfully
        """
        topic_suffix = "health/status"
        
        payload = {
            "device_id": self.device_id,
            "timestamp": status.get("timestamp", time.time()),
            "status": status.get("status", "unknown"),
            "confidence": status.get("confidence", 0.0),
            "inference_time_ms": status.get("inference_time_ms", 0),
        }
        
        return self.mqtt_client.publish(topic_suffix, payload)
    
    def publish_anomaly_alert(self, anomaly: Dict[str, Any]) -> bool:
        """
        Publish anomaly alert.
        
        Args:
            anomaly: Anomaly detection result
        
        Returns:
            True if published successfully
        """
        topic_suffix = "alerts/anomaly"
        
        payload = {
            "device_id": self.device_id,
            "timestamp": anomaly.get("timestamp", time.time()),
            "severity": anomaly.get("severity", "warning"),
            "message": anomaly.get("message", ""),
            "sensor_type": anomaly.get("sensor_type", "unknown"),
            "reading": anomaly.get("reading", {}),
        }
        
        # Use retain flag for alerts so they're not missed
        return self.mqtt_client.publish(topic_suffix, payload, retain=True)
    
    def publish_batch_readings(self, readings: List[Dict[str, Any]]) -> int:
        """
        Publish multiple sensor readings in a batch.
        
        Args:
            readings: List of sensor reading dictionaries
        
        Returns:
            Number of successfully published messages
        """
        published_count = 0
        
        for reading in readings:
            if self.publish_sensor_reading(reading):
                published_count += 1
        
        return published_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get streaming statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "messages_sent": self.messages_sent,
            "last_sent_time": self.last_sent_time,
            "is_connected": self.mqtt_client.is_connected,
            "is_streaming": self.is_streaming,
        }
    
    def stop_streaming(self):
        """Stop streaming (if using background thread)."""
        self.is_streaming = False
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=1.0)

