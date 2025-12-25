"""
Plant Simulator
Simulates an industrial conveyor cell with sensors and faults
"""
import asyncio
import random
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlantState:
    """Simulated plant state"""
    
    def __init__(self):
        # Conveyor state
        self.conveyor_position = 0.0  # Position along conveyor
        self.conveyor_speed = 0.0  # Current speed
        
        # Product positions
        self.products: list = []  # List of product positions
        
        # Fault injection
        self.jam_active = False
        self.estop_active = False
        self.sensor_stuck = False
        
        # Physical parameters
        self.conveyor_length = 100.0  # meters
        self.product_length = 2.0
        self.sensor_positions = [25.0, 50.0, 75.0]  # Sensor locations
        
    def add_product(self):
        """Add a new product to the conveyor"""
        if not self.products or self.products[-1] > 5.0:
            self.products.append(0.0)
            logger.info("New product added to conveyor")
    
    def update(self, dt: float, target_speed: float, motor_on: bool):
        """Update plant physics"""
        # Update speed based on motor state
        if motor_on and not self.jam_active:
            self.conveyor_speed = min(target_speed, self.conveyor_speed + 10.0 * dt)
        else:
            self.conveyor_speed = max(0.0, self.conveyor_speed - 20.0 * dt)
        
        # Move products
        if not self.jam_active:
            new_products = []
            for pos in self.products:
                new_pos = pos + self.conveyor_speed * dt
                if new_pos < self.conveyor_length:
                    new_products.append(new_pos)
            self.products = new_products
    
    def check_sensor(self, sensor_index: int) -> bool:
        """Check if a sensor detects a product"""
        if self.sensor_stuck and sensor_index == 1:
            return True  # Sensor 2 stuck
        
        sensor_pos = self.sensor_positions[sensor_index]
        
        for product_pos in self.products:
            if abs(product_pos - sensor_pos) < self.product_length:
                return True
        
        return False


class PlantSimulator:
    """Plant simulator service"""
    
    def __init__(self, plc_api_url: str = "http://localhost:8001"):
        self.plc_api_url = plc_api_url
        self.state = PlantState()
        self.running = False
        self.sim_task: Optional[asyncio.Task] = None
        self.product_gen_task: Optional[asyncio.Task] = None
        
        # Simulation parameters
        self.update_rate = 0.1  # seconds
        self.product_interval = 8.0  # seconds
    
    def start(self):
        """Start simulation"""
        if not self.running:
            self.running = True
            self.sim_task = asyncio.create_task(self._simulation_loop())
            self.product_gen_task = asyncio.create_task(self._product_generator())
            logger.info("Plant simulator started")
    
    async def stop(self):
        """Stop simulation"""
        if self.running:
            self.running = False
            if self.sim_task:
                await self.sim_task
            if self.product_gen_task:
                self.product_gen_task.cancel()
            logger.info("Plant simulator stopped")
    
    async def _simulation_loop(self):
        """Main simulation loop"""
        async with httpx.AsyncClient() as client:
            while self.running:
                try:
                    # Get PLC outputs
                    response = await client.get(f"{self.plc_api_url}/io")
                    if response.status_code == 200:
                        io_data = response.json()
                        outputs = io_data.get("outputs", {})
                        
                        motor_on = outputs.get("DO_ConveyorMotor", False)
                        target_speed = outputs.get("AO_ConveyorSpeedSetpoint", 0.0) / 10.0
                        
                        # Update plant physics
                        self.state.update(self.update_rate, target_speed, motor_on)
                        
                        # Update sensor states
                        sensor1 = self.state.check_sensor(0)
                        sensor2 = self.state.check_sensor(1)
                        sensor3 = self.state.check_sensor(2)
                        
                        # Send inputs to PLC
                        await self._update_plc_input(client, "DI_ConveyorSensor1", sensor1)
                        await self._update_plc_input(client, "DI_ConveyorSensor2", sensor2)
                        await self._update_plc_input(client, "DI_ConveyorSensor3", sensor3)
                        
                        # Update analog inputs
                        await self._update_plc_input(
                            client, "AI_ConveyorSpeed", self.state.conveyor_speed
                        )
                        
                        # Motor current proportional to speed
                        motor_current = self.state.conveyor_speed * 0.5 if motor_on else 0.0
                        await self._update_plc_input(client, "AI_MotorCurrent", motor_current)
                        
                        # E-Stop
                        await self._update_plc_input(client, "DI_EStop", self.state.estop_active)
                
                except Exception as e:
                    logger.error(f"Simulation loop error: {e}")
                
                await asyncio.sleep(self.update_rate)
    
    async def _product_generator(self):
        """Generate products periodically"""
        while self.running:
            await asyncio.sleep(self.product_interval)
            if not self.state.jam_active:
                self.state.add_product()
    
    async def _update_plc_input(self, client: httpx.AsyncClient, tag: str, value):
        """Update a PLC input"""
        try:
            await client.post(
                f"{self.plc_api_url}/input",
                json={"tag": tag, "value": value}
            )
        except Exception as e:
            logger.error(f"Failed to update PLC input {tag}: {e}")
    
    def inject_jam(self):
        """Inject a jam fault"""
        self.state.jam_active = True
        logger.warning("JAM FAULT INJECTED")
    
    def clear_jam(self):
        """Clear jam fault"""
        self.state.jam_active = False
        # Clear jammed product
        if self.state.products:
            self.state.products = self.state.products[:-1]
        logger.info("Jam cleared")
    
    def inject_estop(self):
        """Inject E-Stop"""
        self.state.estop_active = True
        logger.warning("E-STOP ACTIVATED")
    
    def clear_estop(self):
        """Clear E-Stop"""
        self.state.estop_active = False
        logger.info("E-Stop cleared")
    
    def inject_sensor_stuck(self):
        """Inject stuck sensor fault"""
        self.state.sensor_stuck = True
        logger.warning("SENSOR STUCK FAULT INJECTED")
    
    def clear_sensor_stuck(self):
        """Clear stuck sensor fault"""
        self.state.sensor_stuck = False
        logger.info("Sensor fault cleared")


# FastAPI app
app = FastAPI(title="Plant Simulator API", version="1.0.0")
simulator: Optional[PlantSimulator] = None


@app.on_event("startup")
async def startup():
    global simulator
    simulator = PlantSimulator()
    simulator.start()


@app.on_event("shutdown")
async def shutdown():
    if simulator:
        await simulator.stop()


@app.get("/")
def root():
    return {"service": "Plant Simulator", "status": "running"}


class FaultCommand(BaseModel):
    fault_type: str  # "jam", "estop", "sensor_stuck"
    active: bool


@app.post("/fault")
def inject_fault(cmd: FaultCommand):
    """Inject or clear a fault"""
    if simulator is None:
        return {"error": "Simulator not initialized"}
    
    if cmd.fault_type == "jam":
        if cmd.active:
            simulator.inject_jam()
        else:
            simulator.clear_jam()
    elif cmd.fault_type == "estop":
        if cmd.active:
            simulator.inject_estop()
        else:
            simulator.clear_estop()
    elif cmd.fault_type == "sensor_stuck":
        if cmd.active:
            simulator.inject_sensor_stuck()
        else:
            simulator.clear_sensor_stuck()
    else:
        return {"error": f"Unknown fault type: {cmd.fault_type}"}
    
    return {"success": True, "fault": cmd.fault_type, "active": cmd.active}


@app.get("/state")
def get_state():
    """Get simulator state"""
    if simulator is None:
        return {"error": "Simulator not initialized"}
    
    return {
        "conveyor_speed": round(simulator.state.conveyor_speed, 2),
        "products_count": len(simulator.state.products),
        "jam_active": simulator.state.jam_active,
        "estop_active": simulator.state.estop_active,
        "sensor_stuck": simulator.state.sensor_stuck,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

