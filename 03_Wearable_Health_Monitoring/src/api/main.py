"""FastAPI backend for wearable health monitoring system."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from src.api.database import (
    get_db,
    init_db,
    SensorReading,
    HealthStatus,
    AnomalyAlert,
)
from src.api.models import (
    SensorReadingResponse,
    HealthStatusResponse,
    AnomalyAlertResponse,
    StatisticsResponse,
    TimeRangeQuery,
    SensorReadingBase,
)
from src.config.settings import API_CONFIG
from src.api.mqtt_subscriber import mqtt_subscriber
from src.processing.anomaly_detection import AnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    version=API_CONFIG["version"],
    description="API for Wearable Health Monitoring System",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")
    
    # Start MQTT subscriber to receive data from edge devices
    logger.info("Starting MQTT subscriber...")
    mqtt_subscriber.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down API...")
    mqtt_subscriber.stop()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# Sensor readings endpoints
@app.post("/api/readings", response_model=SensorReadingResponse)
async def create_reading(
    reading: SensorReadingBase,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new sensor reading.
    """
    # Create database record
    db_reading = SensorReading(
        device_id=reading.device_id,
        sensor_type=reading.sensor_type,
        timestamp=reading.timestamp or datetime.utcnow(),
        raw_data=reading.raw_data,
    )
    
    # Extract sensor-specific fields from raw_data
    if reading.raw_data:
        if reading.sensor_type == "heart_rate":
            db_reading.heart_rate = reading.raw_data.get("heart_rate")
        elif reading.sensor_type == "pulse_oximeter":
            db_reading.spo2 = reading.raw_data.get("spo2")
        elif reading.sensor_type == "accelerometer":
            accel = reading.raw_data.get("acceleration", {})
            db_reading.acceleration_x = accel.get("x")
            db_reading.acceleration_y = accel.get("y")
            db_reading.acceleration_z = accel.get("z")
            db_reading.acceleration_magnitude = accel.get("magnitude")
    
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    
    # Check for anomalies in background
    background_tasks.add_task(check_anomalies, db_reading.id, db)
    
    return db_reading


@app.get("/api/readings", response_model=List[SensorReadingResponse])
async def get_readings(
    query: TimeRangeQuery = Depends(),
    db: Session = Depends(get_db)
):
    """
    Get sensor readings with optional filters.
    """
    query_obj = db.query(SensorReading)
    
    if query.device_id:
        query_obj = query_obj.filter(SensorReading.device_id == query.device_id)
    
    if query.sensor_type:
        query_obj = query_obj.filter(SensorReading.sensor_type == query.sensor_type)
    
    if query.start_time:
        query_obj = query_obj.filter(SensorReading.timestamp >= query.start_time)
    
    if query.end_time:
        query_obj = query_obj.filter(SensorReading.timestamp <= query.end_time)
    
    readings = query_obj.order_by(desc(SensorReading.timestamp)).limit(query.limit).all()
    return readings


@app.get("/api/readings/{reading_id}", response_model=SensorReadingResponse)
async def get_reading(reading_id: int, db: Session = Depends(get_db)):
    """
    Get a specific sensor reading by ID.
    """
    reading = db.query(SensorReading).filter(SensorReading.id == reading_id).first()
    if not reading:
        raise HTTPException(status_code=404, detail="Reading not found")
    return reading


@app.get("/api/readings/latest/{device_id}", response_model=List[SensorReadingResponse])
async def get_latest_readings(
    device_id: str,
    sensor_type: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get latest sensor readings for a device.
    """
    query_obj = db.query(SensorReading).filter(SensorReading.device_id == device_id)
    
    if sensor_type:
        query_obj = query_obj.filter(SensorReading.sensor_type == sensor_type)
    
    readings = query_obj.order_by(desc(SensorReading.timestamp)).limit(limit).all()
    return readings


# Health status endpoints
@app.post("/api/health-status", response_model=HealthStatusResponse)
async def create_health_status(
    status: HealthStatusResponse,
    db: Session = Depends(get_db)
):
    """
    Create a new health status record.
    """
    db_status = HealthStatus(
        device_id=status.device_id,
        timestamp=status.timestamp or datetime.utcnow(),
        status=status.status,
        confidence=status.confidence,
        inference_time_ms=status.inference_time_ms,
    )
    
    db.add(db_status)
    db.commit()
    db.refresh(db_status)
    
    return db_status


@app.get("/api/health-status", response_model=List[HealthStatusResponse])
async def get_health_status(
    device_id: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get health status records.
    """
    query_obj = db.query(HealthStatus)
    
    if device_id:
        query_obj = query_obj.filter(HealthStatus.device_id == device_id)
    
    statuses = query_obj.order_by(desc(HealthStatus.timestamp)).limit(limit).all()
    return statuses


@app.get("/api/health-status/latest/{device_id}", response_model=HealthStatusResponse)
async def get_latest_health_status(device_id: str, db: Session = Depends(get_db)):
    """
    Get latest health status for a device.
    """
    status = (
        db.query(HealthStatus)
        .filter(HealthStatus.device_id == device_id)
        .order_by(desc(HealthStatus.timestamp))
        .first()
    )
    
    if not status:
        raise HTTPException(status_code=404, detail="Health status not found")
    
    return status


# Anomaly alerts endpoints
@app.get("/api/alerts", response_model=List[AnomalyAlertResponse])
async def get_alerts(
    device_id: Optional[str] = None,
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get anomaly alerts.
    """
    query_obj = db.query(AnomalyAlert)
    
    if device_id:
        query_obj = query_obj.filter(AnomalyAlert.device_id == device_id)
    
    if severity:
        query_obj = query_obj.filter(AnomalyAlert.severity == severity)
    
    if acknowledged is not None:
        query_obj = query_obj.filter(AnomalyAlert.acknowledged == acknowledged)
    
    alerts = query_obj.order_by(desc(AnomalyAlert.timestamp)).limit(limit).all()
    return alerts


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """
    Acknowledge an anomaly alert.
    """
    alert = db.query(AnomalyAlert).filter(AnomalyAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Alert acknowledged", "alert_id": alert_id}


# Statistics endpoint
@app.get("/api/statistics", response_model=StatisticsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """
    Get system statistics.
    """
    total_readings = db.query(func.count(SensorReading.id)).scalar()
    device_count = db.query(func.count(func.distinct(SensorReading.device_id))).scalar()
    latest_reading = (
        db.query(SensorReading)
        .order_by(desc(SensorReading.timestamp))
        .first()
    )
    alerts_count = db.query(func.count(AnomalyAlert.id)).scalar()
    unacknowledged_alerts = (
        db.query(func.count(AnomalyAlert.id))
        .filter(AnomalyAlert.acknowledged == False)
        .scalar()
    )
    
    return StatisticsResponse(
        total_readings=total_readings or 0,
        device_count=device_count or 0,
        latest_reading_time=latest_reading.timestamp if latest_reading else None,
        alerts_count=alerts_count or 0,
        unacknowledged_alerts=unacknowledged_alerts or 0,
    )


# Helper function for anomaly detection
def check_anomalies(reading_id: int, db: Session):
    """
    Check for anomalies in a sensor reading (background task).
    """
    try:
        reading = db.query(SensorReading).filter(SensorReading.id == reading_id).first()
        if not reading:
            return
        
        # Convert reading to dict for anomaly detection
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
        
        # Detect anomalies
        detector = AnomalyDetector()
        is_anomaly, severity, message = detector.detect_using_thresholds(reading_dict)
        
        if is_anomaly:
            # Create alert
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
            logger.info(f"Anomaly detected: {message}")
    
    except Exception as e:
        logger.error(f"Error checking anomalies: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
    )

