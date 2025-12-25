"""
Seed database with sample data
"""
import asyncio
from datetime import datetime, timedelta
import random
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import async_session_maker, init_db
from src.models import Asset, PMPlan, WorkOrder, SparePart, DowntimeIncident, RootCauseAnalysis


async def seed_data():
    """Seed the database with sample data"""
    await init_db()
    
    async with async_session_maker() as session:
        print("Seeding assets...")
        assets = [
            Asset(
                asset_tag="CONV-001",
                name="Main Conveyor Belt",
                category="Conveyor",
                location="Production Line A",
                manufacturer="ConveyorCo",
                model="CB-2000",
                serial_number="SN12345",
                install_date=datetime(2020, 1, 15),
                criticality="CRITICAL",
                status="OPERATIONAL"
            ),
            Asset(
                asset_tag="PUMP-001",
                name="Hydraulic Pump #1",
                category="Pump",
                location="Pump Room",
                manufacturer="PumpTech",
                model="HP-500",
                serial_number="SN23456",
                install_date=datetime(2019, 6, 20),
                criticality="HIGH",
                status="OPERATIONAL"
            ),
            Asset(
                asset_tag="MOTOR-001",
                name="Main Drive Motor",
                category="Motor",
                location="Production Line A",
                manufacturer="MotorWorks",
                model="MDM-100",
                serial_number="SN34567",
                install_date=datetime(2018, 3, 10),
                criticality="CRITICAL",
                status="OPERATIONAL"
            ),
            Asset(
                asset_tag="COMP-001",
                name="Air Compressor",
                category="Compressor",
                location="Utility Room",
                manufacturer="AirSystems",
                model="AC-750",
                serial_number="SN45678",
                install_date=datetime(2021, 2, 5),
                criticality="HIGH",
                status="OPERATIONAL"
            ),
            Asset(
                asset_tag="ROBOT-001",
                name="Robotic Arm",
                category="Robot",
                location="Assembly Station",
                manufacturer="RoboTech",
                model="RA-6000",
                serial_number="SN56789",
                install_date=datetime(2022, 8, 1),
                criticality="MEDIUM",
                status="OPERATIONAL"
            ),
        ]
        
        session.add_all(assets)
        await session.commit()
        
        print("Seeding PM plans...")
        pm_plans = [
            PMPlan(
                asset_id=1,
                plan_name="Monthly Conveyor Inspection",
                description="Inspect belt tension, alignment, and bearings",
                frequency_days=30,
                estimated_hours=2.0,
                last_completed=datetime.now() - timedelta(days=15),
                next_due=datetime.now() + timedelta(days=15),
                checklist_items="Check belt tension;Inspect bearings;Lubricate chains;Check alignment"
            ),
            PMPlan(
                asset_id=2,
                plan_name="Pump Maintenance",
                description="Check seals, oil level, and performance",
                frequency_days=90,
                estimated_hours=3.0,
                last_completed=datetime.now() - timedelta(days=45),
                next_due=datetime.now() + timedelta(days=45),
                checklist_items="Check oil level;Inspect seals;Test pressure;Check for leaks"
            ),
            PMPlan(
                asset_id=3,
                plan_name="Motor Servicing",
                description="Electrical checks and bearing lubrication",
                frequency_days=180,
                estimated_hours=4.0,
                last_completed=datetime.now() - timedelta(days=90),
                next_due=datetime.now() + timedelta(days=90),
                checklist_items="Check electrical connections;Measure insulation resistance;Lubricate bearings;Check vibration"
            ),
        ]
        
        session.add_all(pm_plans)
        await session.commit()
        
        print("Seeding spare parts...")
        spare_parts = [
            SparePart(
                part_number="BELT-2000-20M",
                description="Conveyor Belt 20m",
                category="Belts",
                quantity_on_hand=2,
                min_quantity=1,
                unit_cost=450.00,
                location="Warehouse A-12",
                supplier="ConveyorCo"
            ),
            SparePart(
                part_number="BEARING-6205",
                description="Ball Bearing 6205",
                category="Bearings",
                quantity_on_hand=15,
                min_quantity=10,
                unit_cost=12.50,
                location="Warehouse B-05",
                supplier="BearingSupply"
            ),
            SparePart(
                part_number="SEAL-HP500",
                description="Pump Seal Kit HP-500",
                category="Seals",
                quantity_on_hand=3,
                min_quantity=2,
                unit_cost=85.00,
                location="Warehouse C-08",
                supplier="PumpTech"
            ),
            SparePart(
                part_number="OIL-HYD-20L",
                description="Hydraulic Oil 20L",
                category="Lubricants",
                quantity_on_hand=8,
                min_quantity=5,
                unit_cost=65.00,
                location="Chemical Storage",
                supplier="LubricantsInc"
            ),
            SparePart(
                part_number="FILTER-AIR-AC750",
                description="Air Filter for AC-750",
                category="Filters",
                quantity_on_hand=4,
                min_quantity=3,
                unit_cost=35.00,
                location="Warehouse B-12",
                supplier="AirSystems"
            ),
        ]
        
        session.add_all(spare_parts)
        await session.commit()
        
        print("Seeding work orders...")
        work_orders = []
        for i in range(10):
            asset_id = random.randint(1, 5)
            wo_type = random.choice(["PM", "CM"])
            status = random.choice(["OPEN", "ASSIGNED", "IN_PROGRESS", "COMPLETED", "COMPLETED"])
            
            wo = WorkOrder(
                wo_number=f"WO-{i+1:04d}",
                asset_id=asset_id,
                wo_type=wo_type,
                priority=random.choice(["LOW", "MEDIUM", "HIGH"]),
                status=status,
                title=f"{'Preventive' if wo_type == 'PM' else 'Corrective'} Maintenance",
                description=f"Maintenance work on asset {asset_id}",
                assigned_to=random.choice(["John Smith", "Jane Doe", "Bob Johnson"]),
                created_by="System",
                created_at=datetime.now() - timedelta(days=random.randint(1, 30)),
                scheduled_date=datetime.now() - timedelta(days=random.randint(0, 7)),
                estimated_hours=random.uniform(1.0, 8.0)
            )
            
            if status == "COMPLETED":
                wo.started_at = wo.scheduled_date
                wo.completed_at = wo.started_at + timedelta(hours=random.uniform(1, 6))
                wo.actual_hours = random.uniform(1.0, 8.0)
                wo.resolution_notes = "Work completed successfully"
            
            work_orders.append(wo)
        
        session.add_all(work_orders)
        await session.commit()
        
        print("Seeding downtime incidents...")
        incidents = []
        failure_modes = [
            "Motor Failure", "Belt Breakage", "Bearing Seizure",
            "Hydraulic Leak", "Electrical Fault", "Sensor Malfunction"
        ]
        
        for i in range(5):
            start = datetime.now() - timedelta(days=random.randint(1, 30))
            end = start + timedelta(hours=random.uniform(0.5, 8.0))
            
            incident = DowntimeIncident(
                asset_id=random.randint(1, 5),
                incident_number=f"INC-{i+1:04d}",
                start_time=start,
                end_time=end,
                downtime_hours=(end - start).total_seconds() / 3600,
                failure_mode=random.choice(failure_modes),
                reason_code=f"RC-{random.randint(100, 999)}",
                severity=random.choice(["MINOR", "MAJOR", "CRITICAL"]),
                description="Equipment failure incident",
                production_impact=random.uniform(100, 5000)
            )
            incidents.append(incident)
        
        session.add_all(incidents)
        await session.commit()
        
        print("Seeding RCA for some incidents...")
        rca1 = RootCauseAnalysis(
            incident_id=1,
            why_1="Motor stopped running",
            why_2="Thermal overload tripped",
            why_3="Motor was overheating",
            why_4="Cooling fan was not working",
            why_5="Fan belt was broken",
            root_cause="Lack of preventive maintenance on cooling system",
            equipment_factors="Worn fan belt not replaced",
            process_factors="PM schedule not followed",
            corrective_actions="Replace fan belt, reset thermal overload",
            preventive_actions="Add fan belt to PM checklist, increase inspection frequency",
            completed_by="John Smith"
        )
        
        session.add(rca1)
        await session.commit()
        
        print("✓ Database seeded successfully!")


if __name__ == "__main__":
    asyncio.run(seed_data())

