"""
Modbus TCP Server
Exposes PLC tags via Modbus protocol
"""
import asyncio
import logging
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.device import ModbusDeviceIdentification
from typing import Optional

logger = logging.getLogger(__name__)


class ModbusGateway:
    """
    Modbus TCP gateway for PLC
    
    Register Mapping:
    Coils (0xxxx):
        0: DO_ConveyorMotor
        1: DO_DiverterActuator
        2: DO_AlarmLight
        3: DO_RunLight
    
    Discrete Inputs (1xxxx):
        0: DI_ConveyorSensor1
        1: DI_ConveyorSensor2
        2: DI_ConveyorSensor3
        3: DI_DiverterSensor
        4: DI_EStop
        5: DI_StartButton
        6: DI_StopButton
        7: DI_ResetButton
    
    Holding Registers (4xxxx):
        0: AI_ConveyorSpeed (scaled x10)
        1: AI_MotorCurrent (scaled x100)
        2: AO_ConveyorSpeedSetpoint (scaled x10)
        3: PLC Mode (0=MANUAL, 1=AUTO, 2=FAULT)
        4: Running state (0=stopped, 1=running)
    """
    
    def __init__(self, plc, host: str = "0.0.0.0", port: int = 5020):
        self.plc = plc
        self.host = host
        self.port = port
        self.server_task: Optional[asyncio.Task] = None
        self.update_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Create datastore
        self.store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * 10),  # Discrete inputs
            co=ModbusSequentialDataBlock(0, [0] * 10),  # Coils
            hr=ModbusSequentialDataBlock(0, [0] * 10),  # Holding registers
            ir=ModbusSequentialDataBlock(0, [0] * 10),  # Input registers
        )
        
        self.context = ModbusServerContext(slaves=self.store, single=True)
    
    async def start(self):
        """Start Modbus server"""
        if not self.running:
            self.running = True
            
            # Start update loop
            self.update_task = asyncio.create_task(self._update_loop())
            
            logger.info(f"Modbus TCP server starting on {self.host}:{self.port}")
            
            # Start server
            try:
                await StartAsyncTcpServer(
                    context=self.context,
                    address=(self.host, self.port)
                )
            except Exception as e:
                logger.error(f"Modbus server error: {e}")
    
    async def _update_loop(self):
        """Update Modbus registers from PLC"""
        while self.running:
            try:
                # Update discrete inputs
                di_values = [
                    self.plc.io_image.DI_ConveyorSensor1,
                    self.plc.io_image.DI_ConveyorSensor2,
                    self.plc.io_image.DI_ConveyorSensor3,
                    self.plc.io_image.DI_DiverterSensor,
                    self.plc.io_image.DI_EStop,
                    self.plc.io_image.DI_StartButton,
                    self.plc.io_image.DI_StopButton,
                    self.plc.io_image.DI_ResetButton,
                ]
                
                for i, value in enumerate(di_values):
                    self.context[0].setValues(2, i, [int(value)])
                
                # Update coils (outputs)
                co_values = [
                    self.plc.io_image.DO_ConveyorMotor,
                    self.plc.io_image.DO_DiverterActuator,
                    self.plc.io_image.DO_AlarmLight,
                    self.plc.io_image.DO_RunLight,
                ]
                
                for i, value in enumerate(co_values):
                    self.context[0].setValues(1, i, [int(value)])
                
                # Update holding registers
                mode_map = {"MANUAL": 0, "AUTO": 1, "FAULT": 2}
                
                hr_values = [
                    int(self.plc.io_image.AI_ConveyorSpeed * 10),
                    int(self.plc.io_image.AI_MotorCurrent * 100),
                    int(self.plc.io_image.AO_ConveyorSpeedSetpoint * 10),
                    mode_map.get(self.plc.state.mode, 0),
                    int(self.plc.state.running),
                ]
                
                for i, value in enumerate(hr_values):
                    self.context[0].setValues(3, i, [value])
                
            except Exception as e:
                logger.error(f"Modbus update error: {e}")
            
            await asyncio.sleep(0.1)  # Update at 10Hz
    
    async def stop(self):
        """Stop Modbus server"""
        if self.running:
            self.running = False
            if self.update_task:
                self.update_task.cancel()
            logger.info("Modbus TCP server stopped")

