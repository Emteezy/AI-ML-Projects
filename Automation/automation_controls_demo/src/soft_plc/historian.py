"""
Historian Module
Time-series data logging for PLC tags
"""
import asyncio
import aiosqlite
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Historian:
    """Time-series historian for PLC data"""
    
    def __init__(self, db_path: str = "./data/plc_data.db"):
        self.db_path = db_path
        self.db: Optional[aiosqlite.Connection] = None
        self.logging_task: Optional[asyncio.Task] = None
        self.running = False
        self.log_interval = 1.0  # seconds
    
    async def initialize(self):
        """Initialize database"""
        self.db = await aiosqlite.connect(self.db_path)
        
        # Create tables
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS tag_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                tag TEXT NOT NULL,
                value REAL NOT NULL
            )
        """)
        
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tag_timestamp 
            ON tag_history(tag, timestamp)
        """)
        
        await self.db.commit()
        logger.info(f"Historian database initialized: {self.db_path}")
    
    def start_logging(self, plc):
        """Start logging PLC data"""
        if not self.running:
            self.running = True
            self.logging_task = asyncio.create_task(self._logging_loop(plc))
            logger.info("Historian logging started")
    
    async def stop_logging(self):
        """Stop logging"""
        if self.running:
            self.running = False
            if self.logging_task:
                await self.logging_task
            logger.info("Historian logging stopped")
    
    async def _logging_loop(self, plc):
        """Main logging loop"""
        while self.running:
            try:
                timestamp = datetime.now().timestamp()
                
                # Log key analog values
                analog_tags = [
                    ("AI_ConveyorSpeed", plc.io_image.AI_ConveyorSpeed),
                    ("AI_MotorCurrent", plc.io_image.AI_MotorCurrent),
                    ("AO_ConveyorSpeedSetpoint", plc.io_image.AO_ConveyorSpeedSetpoint),
                ]
                
                # Log digital outputs as 0/1
                digital_tags = [
                    ("DO_ConveyorMotor", float(plc.io_image.DO_ConveyorMotor)),
                    ("DO_DiverterActuator", float(plc.io_image.DO_DiverterActuator)),
                    ("DO_AlarmLight", float(plc.io_image.DO_AlarmLight)),
                    ("DO_RunLight", float(plc.io_image.DO_RunLight)),
                ]
                
                # Insert all tags
                for tag, value in analog_tags + digital_tags:
                    await self.db.execute(
                        "INSERT INTO tag_history (timestamp, tag, value) VALUES (?, ?, ?)",
                        (timestamp, tag, value)
                    )
                
                await self.db.commit()
                
            except Exception as e:
                logger.error(f"Historian logging error: {e}")
            
            await asyncio.sleep(self.log_interval)
    
    async def get_tag_history(self, tag: str, hours: int = 1) -> List[Dict]:
        """Get historical data for a tag"""
        if self.db is None:
            raise RuntimeError("Historian not initialized")
        
        start_time = (datetime.now() - timedelta(hours=hours)).timestamp()
        
        cursor = await self.db.execute(
            """
            SELECT timestamp, value 
            FROM tag_history 
            WHERE tag = ? AND timestamp >= ?
            ORDER BY timestamp
            """,
            (tag, start_time)
        )
        
        rows = await cursor.fetchall()
        
        return [
            {
                "timestamp": datetime.fromtimestamp(row[0]).isoformat(),
                "value": row[1]
            }
            for row in rows
        ]
    
    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
            logger.info("Historian database closed")

