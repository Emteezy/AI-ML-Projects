"""MQTT subscriber service for receiving sensor data in the backend."""

import json
import logging
from datetime import datetime
from typing import Dict, Any
from sqlalchemy.orm import Session
from src.iot.mqtt_client import MQTTClient
from src.api.database import SensorReading, HealthStatus, AnomalyAlert, SessionLocal
from src.processing.anomaly_detection import AnomalyDetector

logger = logging.getLogger(__name__)


class MQTTSubscriberService:
    """Service to subscribe to MQTT topics and store data in database."""
    
    def __init__(self):
        """Initialize MQTT subscriber service."""
        self.mqtt_client = MQTTClient(client_id="backend_subscriber")
        self.anomaly_detector = AnomalyDetector()
        self.is_running = False
    
    def _handle_sensor_reading(self, topic: str, payload: Dict[str, Any]):
        """
        Handle sensor reading from MQTT.
        
        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            device_id = payload.get("device_id")
            sensor_data = payload.get("data", {})
            timestamp = payload.get("timestamp")
            
            if timestamp:
                timestamp = datetime.fromtimestamp(timestamp)
            else:
                timestamp = datetime.utcnow()
            
            sensor_type = sensor_data.get("sensor_type", "")
            
            # Create database session
            db = SessionLocal()
            try:
                # Create sensor reading record
                db_reading = SensorReading(
                    device_id=device_id,
                    sensor_type=sensor_type,
                    timestamp=timestamp,
                    raw_data=sensor_data,
                )
                
                # Extract sensor-specific fields
                if sensor_type == "heart_rate":
                    db_reading.heart_rate = sensor_data.get("heart_rate")
                elif sensor_type == "pulse_oximeter":
                    db_reading.spo2 = sensor_data.get("spo2")
                elif sensor_type == "accelerometer":
                    accel = sensor_data.get("acceleration", {})
                    db_reading.acceleration_x = accel.get("x")
                    db_reading.acceleration_y = accel.get("y")
                    db_reading.acceleration_z = accel.get("z")
                    db_reading.acceleration_magnitude = accel.get("magnitude")
                
                db.add(db_reading)
                db.commit()
                db.refresh(db_reading)
                
                logger.debug(f"Saved sensor reading: {sensor_type} from {device_id}")
                
                # Check for anomalies
                self._check_anomalies(db_reading, db)
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"Error handling sensor reading: {e}", exc_info=True)
    
    def _handle_health_status(self, topic: str, payload: Dict[str, Any]):
        """
        Handle health status from MQTT.
        
        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            device_id = payload.get("device_id")
            timestamp = payload.get("timestamp")
            
            if timestamp:
                timestamp = datetime.fromtimestamp(timestamp)
            else:
                timestamp = datetime.utcnow()
            
            db = SessionLocal()
            try:
                db_status = HealthStatus(
                    device_id=device_id,
                    timestamp=timestamp,
                    status=payload.get("status", "unknown"),
                    confidence=payload.get("confidence", 0.0),
                    inference_time_ms=payload.get("inference_time_ms"),
                )
                
                db.add(db_status)
                db.commit()
                
                logger.debug(f"Saved health status: {payload.get('status')} from {device_id}")
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"Error handling health status: {e}", exc_info=True)
    
    def _handle_anomaly_alert(self, topic: str, payload: Dict[str, Any]):
        """
        Handle anomaly alert from MQTT.
        
        Args:
            topic: MQTT topic
            payload: Message payload
        """
        try:
            device_id = payload.get("device_id")
            timestamp = payload.get("timestamp")
            
            if timestamp:
                timestamp = datetime.fromtimestamp(timestamp)
            else:
                timestamp = datetime.utcnow()
            
            db = SessionLocal()
            try:
                db_alert = AnomalyAlert(
                    device_id=device_id,
                    timestamp=timestamp,
                    severity=payload.get("severity", "warning"),
                    message=payload.get("message", ""),
                    sensor_type=payload.get("sensor_type", "unknown"),
                    alert_data=payload.get("reading", {}),
                )
                
                db.add(db_alert)
                db.commit()
                
                logger.warning(f"Saved anomaly alert: {payload.get('message')} from {device_id}")
            
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"Error handling anomaly alert: {e}", exc_info=True)
    
    def _check_anomalies(self, reading: SensorReading, db: Session):
        """Check for anomalies in a sensor reading."""
        try:
            reading_dict = {
                "sensor_type": reading.sensor_type,
                "timestamp": reading.timestamp.timestamp(),
            }
            
            if reading.sensor_type == "heart_rate":
                reading_dict["heart_rate"] = reading.heart_rate
            elif reading.sensor_type == "pulse_oximeter":
                reading_dict["spo2"] = reading.spo2
            elif reading.sensor_type == "accelerometer":
                reading_dict["acceleration"] = {
                    "magnitude": reading.acceleration_magnitude,
                }
            
            is_anomaly, severity, message = self.anomaly_detector.detect_using_thresholds(reading_dict)
            
            if is_anomaly:
                alert = AnomalyAlert(
                    device_id=reading.device_id,
                    timestamp=reading.timestamp,
                    severity=severity,
                    message=message,
                    sensor_type=reading.sensor_type,
                    alert_data=reading_dict,
                )
                db.add(alert)
                db.commit()
                logger.warning(f"Anomaly detected: {message}")
        
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")
    
    def start(self):
        """Start MQTT subscriber service."""
        if self.is_running:
            logger.warning("MQTT subscriber is already running")
            return
        
        logger.info("Starting MQTT subscriber service...")
        
        # Connect to MQTT broker
        if not self.mqtt_client.connect():
            logger.error("Failed to connect to MQTT broker")
            return False
        
        # Subscribe to sensor readings topic (wildcard for all devices)
        # Topic pattern: wearable/health/{device_id}/sensors/{sensor_type}
        self.mqtt_client.subscribe("+/sensors/+", self._handle_sensor_reading)
        
        # Subscribe to health status topic
        # Topic pattern: wearable/health/{device_id}/health/status
        self.mqtt_client.subscribe("+/health/status", self._handle_health_status)
        
        # Subscribe to anomaly alerts topic
        # Topic pattern: wearable/health/{device_id}/alerts/anomaly
        self.mqtt_client.subscribe("+/alerts/anomaly", self._handle_anomaly_alert)
        
        self.is_running = True
        logger.info("MQTT subscriber service started successfully")
        logger.info("Subscribed to topics: +/sensors/+, +/health/status, +/alerts/anomaly")
        
        return True
    
    def stop(self):
        """Stop MQTT subscriber service."""
        if not self.is_running:
            return
        
        logger.info("Stopping MQTT subscriber service...")
        self.mqtt_client.disconnect()
        self.is_running = False
        logger.info("MQTT subscriber service stopped")


# Global instance
mqtt_subscriber = MQTTSubscriberService()

