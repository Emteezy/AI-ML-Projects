"""Database models and session management."""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Generator
from src.config.settings import DATABASE_CONFIG

Base = declarative_base()


class SensorReading(Base):
    """Sensor reading database model."""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    sensor_type = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Sensor-specific fields
    heart_rate = Column(Float, nullable=True)
    spo2 = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    acceleration_x = Column(Float, nullable=True)
    acceleration_y = Column(Float, nullable=True)
    acceleration_z = Column(Float, nullable=True)
    acceleration_magnitude = Column(Float, nullable=True)
    
    # Raw data as JSON for flexibility
    raw_data = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class HealthStatus(Base):
    """Health status prediction database model."""
    __tablename__ = "health_status"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    status = Column(String, nullable=False)  # normal, warning, critical
    confidence = Column(Float, nullable=False)
    inference_time_ms = Column(Float, nullable=True)
    
    # Associated sensor readings IDs (JSON array)
    sensor_reading_ids = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyAlert(Base):
    """Anomaly alert database model."""
    __tablename__ = "anomaly_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    severity = Column(String, nullable=False)  # warning, critical
    message = Column(String, nullable=False)
    sensor_type = Column(String, nullable=False)
    
    # Alert data
    alert_data = Column(JSON, nullable=True)
    
    # Acknowledgment
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# Database setup
engine = create_engine(
    DATABASE_CONFIG["url"],
    pool_size=DATABASE_CONFIG["pool_size"],
    max_overflow=DATABASE_CONFIG["max_overflow"],
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database (create tables)."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Database dependency for FastAPI.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

