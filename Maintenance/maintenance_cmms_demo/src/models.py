"""
Database Models for CMMS
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Asset(Base):
    """Asset/Equipment model"""
    __tablename__ = "assets"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_tag = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    category = Column(String(100))  # Motor, Pump, Conveyor, etc.
    location = Column(String(200))
    manufacturer = Column(String(200))
    model = Column(String(200))
    serial_number = Column(String(200))
    install_date = Column(DateTime)
    criticality = Column(String(20))  # LOW, MEDIUM, HIGH, CRITICAL
    status = Column(String(50), default="OPERATIONAL")  # OPERATIONAL, DOWN, MAINTENANCE
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    work_orders = relationship("WorkOrder", back_populates="asset")
    pm_plans = relationship("PMPlan", back_populates="asset")
    downtime_incidents = relationship("DowntimeIncident", back_populates="asset")


class PMPlan(Base):
    """Preventive Maintenance Plan model"""
    __tablename__ = "pm_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    plan_name = Column(String(200), nullable=False)
    description = Column(Text)
    frequency_days = Column(Integer, nullable=False)  # Days between PM
    estimated_hours = Column(Float)
    last_completed = Column(DateTime)
    next_due = Column(DateTime)
    active = Column(Boolean, default=True)
    checklist_items = Column(Text)  # JSON string of checklist items
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    asset = relationship("Asset", back_populates="pm_plans")


class WorkOrder(Base):
    """Work Order model"""
    __tablename__ = "work_orders"
    
    id = Column(Integer, primary_key=True, index=True)
    wo_number = Column(String(50), unique=True, index=True, nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    wo_type = Column(String(20), nullable=False)  # PM, CM (Corrective Maintenance)
    priority = Column(String(20))  # LOW, MEDIUM, HIGH, URGENT
    status = Column(String(50), default="OPEN")  # OPEN, ASSIGNED, IN_PROGRESS, COMPLETED, CLOSED
    
    title = Column(String(200), nullable=False)
    description = Column(Text)
    
    assigned_to = Column(String(100))
    created_by = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.now)
    scheduled_date = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    closed_at = Column(DateTime)
    
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    
    resolution_notes = Column(Text)
    
    # Relationships
    asset = relationship("Asset", back_populates="work_orders")
    spare_parts_used = relationship("SparePartUsage", back_populates="work_order")
    downtime_incident_id = Column(Integer, ForeignKey("downtime_incidents.id"))
    downtime_incident = relationship("DowntimeIncident", back_populates="work_orders")


class SparePart(Base):
    """Spare Parts Inventory model"""
    __tablename__ = "spare_parts"
    
    id = Column(Integer, primary_key=True, index=True)
    part_number = Column(String(100), unique=True, index=True, nullable=False)
    description = Column(String(300), nullable=False)
    category = Column(String(100))
    quantity_on_hand = Column(Integer, default=0)
    min_quantity = Column(Integer, default=0)
    unit_cost = Column(Float)
    location = Column(String(200))
    supplier = Column(String(200))
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    usage_records = relationship("SparePartUsage", back_populates="spare_part")


class SparePartUsage(Base):
    """Spare Parts Usage tracking"""
    __tablename__ = "spare_part_usage"
    
    id = Column(Integer, primary_key=True, index=True)
    work_order_id = Column(Integer, ForeignKey("work_orders.id"), nullable=False)
    spare_part_id = Column(Integer, ForeignKey("spare_parts.id"), nullable=False)
    quantity_used = Column(Integer, nullable=False)
    used_at = Column(DateTime, default=datetime.now)
    notes = Column(Text)
    
    # Relationships
    work_order = relationship("WorkOrder", back_populates="spare_parts_used")
    spare_part = relationship("SparePart", back_populates="usage_records")


class DowntimeIncident(Base):
    """Downtime/Failure Incident model"""
    __tablename__ = "downtime_incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    incident_number = Column(String(50), unique=True, index=True, nullable=False)
    
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    downtime_hours = Column(Float)
    
    failure_mode = Column(String(200))  # e.g., "Motor Failure", "Belt Breakage"
    reason_code = Column(String(100))  # Standardized codes
    severity = Column(String(20))  # MINOR, MAJOR, CRITICAL
    
    description = Column(Text)
    immediate_action = Column(Text)
    
    production_impact = Column(Float)  # Units lost, revenue impact, etc.
    
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    asset = relationship("Asset", back_populates="downtime_incidents")
    work_orders = relationship("WorkOrder", back_populates="downtime_incident")
    rca = relationship("RootCauseAnalysis", back_populates="incident", uselist=False)


class RootCauseAnalysis(Base):
    """Root Cause Analysis model"""
    __tablename__ = "root_cause_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(Integer, ForeignKey("downtime_incidents.id"), unique=True, nullable=False)
    
    # 5-Why Analysis
    why_1 = Column(Text)
    why_2 = Column(Text)
    why_3 = Column(Text)
    why_4 = Column(Text)
    why_5 = Column(Text)
    root_cause = Column(Text)
    
    # Fishbone/Ishikawa categories
    people_factors = Column(Text)
    process_factors = Column(Text)
    equipment_factors = Column(Text)
    materials_factors = Column(Text)
    environment_factors = Column(Text)
    management_factors = Column(Text)
    
    # Corrective Actions
    corrective_actions = Column(Text)
    preventive_actions = Column(Text)
    
    completed_by = Column(String(100))
    completed_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    incident = relationship("DowntimeIncident", back_populates="rca")

