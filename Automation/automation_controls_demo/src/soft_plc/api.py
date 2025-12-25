"""
Soft PLC API Service
FastAPI endpoints for PLC control and monitoring
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from contextlib import asynccontextmanager

from .plc_core import SoftPLC, PLCMode, AlarmState
from .historian import Historian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global PLC instance
plc: Optional[SoftPLC] = None
historian: Optional[Historian] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global plc, historian
    
    # Startup
    logger.info("Starting Soft PLC service...")
    plc = SoftPLC(scan_rate_ms=100)
    historian = Historian(db_path="./data/plc_data.db")
    await historian.initialize()
    
    plc.start_scan()
    historian.start_logging(plc)
    
    logger.info("Soft PLC service started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Soft PLC service...")
    await plc.stop_scan()
    await historian.stop_logging()
    await historian.close()
    logger.info("Soft PLC service stopped")


app = FastAPI(
    title="Soft PLC API",
    description="Industrial PLC simulation with scan cycle, modes, and alarms",
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


# Request/Response Models
class ModeChangeRequest(BaseModel):
    mode: PLCMode


class InputUpdateRequest(BaseModel):
    tag: str
    value: bool | float


class OutputUpdateRequest(BaseModel):
    tag: str
    value: bool | float


class AlarmAckRequest(BaseModel):
    tag: str


# API Endpoints
@app.get("/")
def root():
    return {
        "service": "Soft PLC API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/status")
def get_status():
    """Get PLC status"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    return plc.get_status()


@app.get("/io")
def get_io_image():
    """Get IO image table"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    return plc.get_io_image_dict()


@app.post("/mode")
def set_mode(request: ModeChangeRequest):
    """Set PLC mode"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    plc.set_mode(request.mode)
    return {"success": True, "mode": request.mode}


@app.post("/command/start")
def command_start():
    """Execute start command"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    success = plc.command_start()
    if not success:
        raise HTTPException(status_code=400, detail="Start permissive not met")
    
    return {"success": True, "running": plc.state.running}


@app.post("/command/stop")
def command_stop():
    """Execute stop command"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    plc.command_stop()
    return {"success": True, "running": plc.state.running}


@app.post("/command/reset")
def command_reset():
    """Execute reset command"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    success = plc.command_reset()
    if not success:
        raise HTTPException(status_code=400, detail="Reset conditions not met")
    
    return {"success": True, "mode": plc.state.mode}


@app.post("/input")
def update_input(request: InputUpdateRequest):
    """Update input value (for plant simulator)"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    try:
        # Update IO image
        if hasattr(plc.io_image, request.tag):
            setattr(plc.io_image, request.tag, request.value)
            
            # Track sensor changes for jam detection
            if request.tag.startswith("DI_ConveyorSensor"):
                plc.update_sensor_state()
            
            return {"success": True, "tag": request.tag, "value": request.value}
        else:
            raise HTTPException(status_code=404, detail=f"Tag {request.tag} not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/output")
def update_output(request: OutputUpdateRequest):
    """Update output value (manual mode only)"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    if plc.state.mode != PLCMode.MANUAL:
        raise HTTPException(status_code=400, detail="Manual mode required")
    
    try:
        if hasattr(plc.io_image, request.tag):
            setattr(plc.io_image, request.tag, request.value)
            return {"success": True, "tag": request.tag, "value": request.value}
        else:
            raise HTTPException(status_code=404, detail=f"Tag {request.tag} not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/alarms")
def get_alarms(state: Optional[AlarmState] = None):
    """Get alarms, optionally filtered by state"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    alarms = plc.alarms.values()
    
    if state:
        alarms = [a for a in alarms if a.state == state]
    
    return {
        "alarms": [
            {
                "tag": a.tag,
                "message": a.message,
                "severity": a.severity,
                "state": a.state,
                "timestamp": a.timestamp.isoformat(),
                "ack_timestamp": a.ack_timestamp.isoformat() if a.ack_timestamp else None,
                "clear_timestamp": a.clear_timestamp.isoformat() if a.clear_timestamp else None,
            }
            for a in alarms
        ],
        "count": len(list(alarms))
    }


@app.post("/alarms/acknowledge")
def acknowledge_alarm(request: AlarmAckRequest):
    """Acknowledge an alarm"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    if request.tag not in plc.alarms:
        raise HTTPException(status_code=404, detail=f"Alarm {request.tag} not found")
    
    plc.acknowledge_alarm(request.tag)
    return {"success": True, "tag": request.tag}


@app.get("/events")
def get_events(limit: int = 100):
    """Get event log"""
    if plc is None:
        raise HTTPException(status_code=503, detail="PLC not initialized")
    
    events = plc.event_log[-limit:]
    return {
        "events": events,
        "count": len(events)
    }


@app.get("/trends/{tag}")
async def get_trend(tag: str, hours: int = 1):
    """Get historical trend data for a tag"""
    if historian is None:
        raise HTTPException(status_code=503, detail="Historian not initialized")
    
    try:
        data = await historian.get_tag_history(tag, hours=hours)
        return {
            "tag": tag,
            "data": data,
            "count": len(data)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "plc_initialized": plc is not None,
        "plc_running": plc.running if plc else False
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

