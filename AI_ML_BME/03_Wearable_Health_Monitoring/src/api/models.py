"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class SensorReadingBase(BaseModel):
    """Base sensor reading model."""
    device_id: str
    sensor_type: str
    timestamp: Optional[datetime] = None
    raw_data: Optional[Dict[str, Any]] = None


class HeartRateReading(SensorReadingBase):
    """Heart rate sensor reading."""
    heart_rate: float = Field(..., ge=0, le=250, description="Heart rate in bpm")
    sensor_type: str = "heart_rate"


class SpO2Reading(SensorReadingBase):
    """SpO2 sensor reading."""
    spo2: float = Field(..., ge=0, le=100, description="SpO2 percentage")
    sensor_type: str = "pulse_oximeter"


class AccelerometerReading(SensorReadingBase):
    """Accelerometer sensor reading."""
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float
    acceleration_magnitude: Optional[float] = None
    sensor_type: str = "accelerometer"


class SensorReadingResponse(BaseModel):
    """Sensor reading response model."""
    id: int
    device_id: str
    sensor_type: str
    timestamp: datetime
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    acceleration_x: Optional[float] = None
    acceleration_y: Optional[float] = None
    acceleration_z: Optional[float] = None
    acceleration_magnitude: Optional[float] = None
    raw_data: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class HealthStatusRequest(BaseModel):
    """Health status prediction request."""
    device_id: str
    sensor_readings: List[Dict[str, Any]]


class HealthStatusResponse(BaseModel):
    """Health status response model."""
    id: Optional[int] = None
    device_id: str
    timestamp: datetime
    status: str  # normal, warning, critical
    confidence: float = Field(..., ge=0, le=1)
    inference_time_ms: Optional[float] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AnomalyAlertResponse(BaseModel):
    """Anomaly alert response model."""
    id: int
    device_id: str
    timestamp: datetime
    severity: str  # warning, critical
    message: str
    sensor_type: str
    alert_data: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    created_at: datetime
    
    class Config:
        from_attributes = True


class StatisticsResponse(BaseModel):
    """Statistics response model."""
    total_readings: int
    device_count: int
    latest_reading_time: Optional[datetime] = None
    alerts_count: int
    unacknowledged_alerts: int


class TimeRangeQuery(BaseModel):
    """Time range query parameters."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    device_id: Optional[str] = None
    sensor_type: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)

