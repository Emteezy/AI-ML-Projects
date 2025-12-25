"""
KPI Calculations Module
"""
from datetime import datetime, timedelta
from typing import Dict, List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Asset, WorkOrder, DowntimeIncident


async def calculate_mttr(session: AsyncSession, asset_id: int = None, days: int = 30) -> float:
    """
    Calculate Mean Time To Repair (MTTR)
    Average time to complete corrective maintenance work orders
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    query = select(WorkOrder).where(
        WorkOrder.wo_type == "CM",
        WorkOrder.status == "COMPLETED",
        WorkOrder.completed_at >= cutoff_date
    )
    
    if asset_id:
        query = query.where(WorkOrder.asset_id == asset_id)
    
    result = await session.execute(query)
    work_orders = result.scalars().all()
    
    if not work_orders:
        return 0.0
    
    total_hours = sum(wo.actual_hours or 0 for wo in work_orders)
    return total_hours / len(work_orders)


async def calculate_mtbf(session: AsyncSession, asset_id: int = None, days: int = 30) -> float:
    """
    Calculate Mean Time Between Failures (MTBF)
    Average time between downtime incidents
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    query = select(DowntimeIncident).where(
        DowntimeIncident.start_time >= cutoff_date,
        DowntimeIncident.end_time.isnot(None)
    ).order_by(DowntimeIncident.start_time)
    
    if asset_id:
        query = query.where(DowntimeIncident.asset_id == asset_id)
    
    result = await session.execute(query)
    incidents = result.scalars().all()
    
    if len(incidents) < 2:
        return 0.0
    
    # Calculate time between incidents
    total_time = 0
    for i in range(1, len(incidents)):
        time_between = (incidents[i].start_time - incidents[i-1].end_time).total_seconds() / 3600
        total_time += time_between
    
    return total_time / (len(incidents) - 1)


async def calculate_downtime_hours(session: AsyncSession, asset_id: int = None, days: int = 30) -> float:
    """Calculate total downtime hours"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    query = select(func.sum(DowntimeIncident.downtime_hours)).where(
        DowntimeIncident.start_time >= cutoff_date
    )
    
    if asset_id:
        query = query.where(DowntimeIncident.asset_id == asset_id)
    
    result = await session.execute(query)
    total = result.scalar()
    
    return total or 0.0


async def get_top_failure_modes(session: AsyncSession, limit: int = 5, days: int = 30) -> List[Dict]:
    """Get most common failure modes"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    query = select(
        DowntimeIncident.failure_mode,
        func.count(DowntimeIncident.id).label('count'),
        func.sum(DowntimeIncident.downtime_hours).label('total_hours')
    ).where(
        DowntimeIncident.start_time >= cutoff_date,
        DowntimeIncident.failure_mode.isnot(None)
    ).group_by(
        DowntimeIncident.failure_mode
    ).order_by(
        func.count(DowntimeIncident.id).desc()
    ).limit(limit)
    
    result = await session.execute(query)
    rows = result.all()
    
    return [
        {
            "failure_mode": row[0],
            "count": row[1],
            "total_hours": row[2] or 0.0
        }
        for row in rows
    ]


async def calculate_pm_compliance(session: AsyncSession, days: int = 30) -> float:
    """Calculate PM compliance rate"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Count PM work orders due in period
    query_due = select(func.count(WorkOrder.id)).where(
        WorkOrder.wo_type == "PM",
        WorkOrder.scheduled_date >= cutoff_date,
        WorkOrder.scheduled_date <= datetime.now()
    )
    
    result = await session.execute(query_due)
    total_due = result.scalar() or 0
    
    if total_due == 0:
        return 100.0
    
    # Count completed PM work orders
    query_completed = select(func.count(WorkOrder.id)).where(
        WorkOrder.wo_type == "PM",
        WorkOrder.scheduled_date >= cutoff_date,
        WorkOrder.scheduled_date <= datetime.now(),
        WorkOrder.status == "COMPLETED"
    )
    
    result = await session.execute(query_completed)
    completed = result.scalar() or 0
    
    return (completed / total_due) * 100.0


async def get_asset_availability(session: AsyncSession, asset_id: int, days: int = 30) -> float:
    """
    Calculate asset availability percentage
    Availability = (Total Time - Downtime) / Total Time * 100
    """
    total_hours = days * 24
    downtime = await calculate_downtime_hours(session, asset_id, days)
    
    if total_hours == 0:
        return 0.0
    
    availability = ((total_hours - downtime) / total_hours) * 100.0
    return max(0.0, min(100.0, availability))


async def get_all_kpis(session: AsyncSession, days: int = 30) -> Dict:
    """Get all KPIs"""
    mttr = await calculate_mttr(session, days=days)
    mtbf = await calculate_mtbf(session, days=days)
    downtime = await calculate_downtime_hours(session, days=days)
    failure_modes = await get_top_failure_modes(session, days=days)
    pm_compliance = await calculate_pm_compliance(session, days=days)
    
    return {
        "mttr_hours": round(mttr, 2),
        "mtbf_hours": round(mtbf, 2),
        "total_downtime_hours": round(downtime, 2),
        "pm_compliance_percent": round(pm_compliance, 2),
        "top_failure_modes": failure_modes,
    }

