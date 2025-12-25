"""
CMMS API Service
FastAPI backend for maintenance management system
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager
import logging

from .database import init_db, get_session
from .models import Asset, PMPlan, WorkOrder, SparePart, SparePartUsage, DowntimeIncident, RootCauseAnalysis
from .kpi_calculator import get_all_kpis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    logger.info("Initializing CMMS database...")
    await init_db()
    logger.info("CMMS API started")
    yield
    logger.info("CMMS API shutdown")


app = FastAPI(
    title="CMMS API",
    description="Computerized Maintenance Management System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class AssetCreate(BaseModel):
    asset_tag: str
    name: str
    category: Optional[str] = None
    location: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    install_date: Optional[datetime] = None
    criticality: str = "MEDIUM"
    notes: Optional[str] = None


class PMPlanCreate(BaseModel):
    asset_id: int
    plan_name: str
    description: Optional[str] = None
    frequency_days: int
    estimated_hours: Optional[float] = None
    checklist_items: Optional[str] = None


class WorkOrderCreate(BaseModel):
    asset_id: int
    wo_type: str  # PM or CM
    priority: str = "MEDIUM"
    title: str
    description: Optional[str] = None
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    scheduled_date: Optional[datetime] = None
    estimated_hours: Optional[float] = None


class WorkOrderUpdate(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_hours: Optional[float] = None
    resolution_notes: Optional[str] = None


class SparePartCreate(BaseModel):
    part_number: str
    description: str
    category: Optional[str] = None
    quantity_on_hand: int = 0
    min_quantity: int = 0
    unit_cost: Optional[float] = None
    location: Optional[str] = None
    supplier: Optional[str] = None


class DowntimeIncidentCreate(BaseModel):
    asset_id: int
    incident_number: str
    start_time: datetime
    end_time: Optional[datetime] = None
    failure_mode: Optional[str] = None
    reason_code: Optional[str] = None
    severity: str = "MINOR"
    description: Optional[str] = None
    immediate_action: Optional[str] = None


class RCACreate(BaseModel):
    incident_id: int
    why_1: Optional[str] = None
    why_2: Optional[str] = None
    why_3: Optional[str] = None
    why_4: Optional[str] = None
    why_5: Optional[str] = None
    root_cause: Optional[str] = None
    people_factors: Optional[str] = None
    process_factors: Optional[str] = None
    equipment_factors: Optional[str] = None
    materials_factors: Optional[str] = None
    environment_factors: Optional[str] = None
    corrective_actions: Optional[str] = None
    preventive_actions: Optional[str] = None
    completed_by: Optional[str] = None


# API Endpoints
@app.get("/")
def root():
    return {"service": "CMMS API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Assets
@app.post("/assets")
async def create_asset(asset: AssetCreate, session: AsyncSession = Depends(get_session)):
    """Create new asset"""
    db_asset = Asset(**asset.dict())
    session.add(db_asset)
    await session.commit()
    await session.refresh(db_asset)
    return db_asset


@app.get("/assets")
async def list_assets(session: AsyncSession = Depends(get_session)):
    """List all assets"""
    result = await session.execute(select(Asset))
    assets = result.scalars().all()
    return assets


@app.get("/assets/{asset_id}")
async def get_asset(asset_id: int, session: AsyncSession = Depends(get_session)):
    """Get asset by ID"""
    result = await session.execute(select(Asset).where(Asset.id == asset_id))
    asset = result.scalar_one_or_none()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


# PM Plans
@app.post("/pm-plans")
async def create_pm_plan(plan: PMPlanCreate, session: AsyncSession = Depends(get_session)):
    """Create PM plan"""
    db_plan = PMPlan(**plan.dict())
    # Set next due date
    if plan.frequency_days:
        from datetime import timedelta
        db_plan.next_due = datetime.now() + timedelta(days=plan.frequency_days)
    
    session.add(db_plan)
    await session.commit()
    await session.refresh(db_plan)
    return db_plan


@app.get("/pm-plans")
async def list_pm_plans(session: AsyncSession = Depends(get_session)):
    """List all PM plans"""
    result = await session.execute(select(PMPlan))
    plans = result.scalars().all()
    return plans


# Work Orders
@app.post("/work-orders")
async def create_work_order(wo: WorkOrderCreate, session: AsyncSession = Depends(get_session)):
    """Create work order"""
    # Generate WO number
    result = await session.execute(select(WorkOrder))
    existing_wos = result.scalars().all()
    wo_number = f"WO-{len(existing_wos) + 1:04d}"
    
    db_wo = WorkOrder(**wo.dict(), wo_number=wo_number)
    session.add(db_wo)
    await session.commit()
    await session.refresh(db_wo)
    return db_wo


@app.get("/work-orders")
async def list_work_orders(
    status: Optional[str] = None,
    wo_type: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
):
    """List work orders with optional filters"""
    query = select(WorkOrder)
    
    if status:
        query = query.where(WorkOrder.status == status)
    if wo_type:
        query = query.where(WorkOrder.wo_type == wo_type)
    
    result = await session.execute(query)
    work_orders = result.scalars().all()
    return work_orders


@app.get("/work-orders/{wo_id}")
async def get_work_order(wo_id: int, session: AsyncSession = Depends(get_session)):
    """Get work order by ID"""
    result = await session.execute(select(WorkOrder).where(WorkOrder.id == wo_id))
    wo = result.scalar_one_or_none()
    if not wo:
        raise HTTPException(status_code=404, detail="Work order not found")
    return wo


@app.patch("/work-orders/{wo_id}")
async def update_work_order(
    wo_id: int,
    update: WorkOrderUpdate,
    session: AsyncSession = Depends(get_session)
):
    """Update work order"""
    result = await session.execute(select(WorkOrder).where(WorkOrder.id == wo_id))
    wo = result.scalar_one_or_none()
    if not wo:
        raise HTTPException(status_code=404, detail="Work order not found")
    
    for key, value in update.dict(exclude_unset=True).items():
        setattr(wo, key, value)
    
    # Auto-set closed_at when status changes to CLOSED
    if update.status == "CLOSED" and not wo.closed_at:
        wo.closed_at = datetime.now()
    
    await session.commit()
    await session.refresh(wo)
    return wo


# Spare Parts
@app.post("/spare-parts")
async def create_spare_part(part: SparePartCreate, session: AsyncSession = Depends(get_session)):
    """Create spare part"""
    db_part = SparePart(**part.dict())
    session.add(db_part)
    await session.commit()
    await session.refresh(db_part)
    return db_part


@app.get("/spare-parts")
async def list_spare_parts(session: AsyncSession = Depends(get_session)):
    """List all spare parts"""
    result = await session.execute(select(SparePart))
    parts = result.scalars().all()
    return parts


@app.get("/spare-parts/low-stock")
async def get_low_stock_parts(session: AsyncSession = Depends(get_session)):
    """Get parts below minimum quantity"""
    result = await session.execute(
        select(SparePart).where(SparePart.quantity_on_hand <= SparePart.min_quantity)
    )
    parts = result.scalars().all()
    return parts


# Downtime Incidents
@app.post("/downtime")
async def create_downtime_incident(
    incident: DowntimeIncidentCreate,
    session: AsyncSession = Depends(get_session)
):
    """Create downtime incident"""
    db_incident = DowntimeIncident(**incident.dict())
    
    # Calculate downtime hours if end_time provided
    if incident.end_time and incident.start_time:
        delta = incident.end_time - incident.start_time
        db_incident.downtime_hours = delta.total_seconds() / 3600
    
    session.add(db_incident)
    await session.commit()
    await session.refresh(db_incident)
    return db_incident


@app.get("/downtime")
async def list_downtime_incidents(session: AsyncSession = Depends(get_session)):
    """List all downtime incidents"""
    result = await session.execute(select(DowntimeIncident))
    incidents = result.scalars().all()
    return incidents


@app.patch("/downtime/{incident_id}")
async def update_downtime_incident(
    incident_id: int,
    end_time: datetime,
    session: AsyncSession = Depends(get_session)
):
    """Update downtime incident with end time"""
    result = await session.execute(
        select(DowntimeIncident).where(DowntimeIncident.id == incident_id)
    )
    incident = result.scalar_one_or_none()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    incident.end_time = end_time
    delta = end_time - incident.start_time
    incident.downtime_hours = delta.total_seconds() / 3600
    
    await session.commit()
    await session.refresh(incident)
    return incident


# Root Cause Analysis
@app.post("/rca")
async def create_rca(rca: RCACreate, session: AsyncSession = Depends(get_session)):
    """Create root cause analysis"""
    db_rca = RootCauseAnalysis(**rca.dict())
    session.add(db_rca)
    await session.commit()
    await session.refresh(db_rca)
    return db_rca


@app.get("/rca/{incident_id}")
async def get_rca(incident_id: int, session: AsyncSession = Depends(get_session)):
    """Get RCA for incident"""
    result = await session.execute(
        select(RootCauseAnalysis).where(RootCauseAnalysis.incident_id == incident_id)
    )
    rca = result.scalar_one_or_none()
    if not rca:
        raise HTTPException(status_code=404, detail="RCA not found")
    return rca


# KPIs
@app.get("/kpis")
async def get_kpis(days: int = 30, session: AsyncSession = Depends(get_session)):
    """Get all KPIs"""
    kpis = await get_all_kpis(session, days=days)
    return kpis


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

