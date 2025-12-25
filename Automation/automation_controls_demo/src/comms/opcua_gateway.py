"""
OPC UA Server
Exposes PLC tags via OPC UA protocol
"""
import asyncio
import logging
from asyncua import Server
from asyncua import ua
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class OPCUAGateway:
    """
    OPC UA gateway for PLC
    
    Namespace: http://plc.automation.demo
    
    Node Structure:
    - Inputs/
        - DI_ConveyorSensor1
        - DI_ConveyorSensor2
        - DI_ConveyorSensor3
        - DI_DiverterSensor
        - DI_EStop
        - DI_StartButton
        - DI_StopButton
        - DI_ResetButton
        - AI_ConveyorSpeed
        - AI_MotorCurrent
    - Outputs/
        - DO_ConveyorMotor
        - DO_DiverterActuator
        - DO_AlarmLight
        - DO_RunLight
        - AO_ConveyorSpeedSetpoint
    - Status/
        - Mode
        - Running
        - ScanCount
        - ScanTimeMs
    """
    
    def __init__(self, plc, host: str = "0.0.0.0", port: int = 4840):
        self.plc = plc
        self.host = host
        self.port = port
        self.server: Optional[Server] = None
        self.update_task: Optional[asyncio.Task] = None
        self.running = False
        self.nodes: Dict[str, any] = {}
    
    async def start(self):
        """Start OPC UA server"""
        if self.running:
            return
        
        self.running = True
        self.server = Server()
        
        await self.server.init()
        self.server.set_endpoint(f"opc.tcp://{self.host}:{self.port}/freeopcua/server/")
        
        # Setup namespace
        uri = "http://plc.automation.demo"
        idx = await self.server.register_namespace(uri)
        
        # Get root objects node
        objects = self.server.nodes.objects
        
        # Create folder structure
        plc_root = await objects.add_folder(idx, "PLC")
        inputs_folder = await plc_root.add_folder(idx, "Inputs")
        outputs_folder = await plc_root.add_folder(idx, "Outputs")
        status_folder = await plc_root.add_folder(idx, "Status")
        
        # Create input variables
        self.nodes["DI_ConveyorSensor1"] = await inputs_folder.add_variable(
            idx, "DI_ConveyorSensor1", False
        )
        self.nodes["DI_ConveyorSensor2"] = await inputs_folder.add_variable(
            idx, "DI_ConveyorSensor2", False
        )
        self.nodes["DI_ConveyorSensor3"] = await inputs_folder.add_variable(
            idx, "DI_ConveyorSensor3", False
        )
        self.nodes["DI_DiverterSensor"] = await inputs_folder.add_variable(
            idx, "DI_DiverterSensor", False
        )
        self.nodes["DI_EStop"] = await inputs_folder.add_variable(
            idx, "DI_EStop", False
        )
        self.nodes["DI_StartButton"] = await inputs_folder.add_variable(
            idx, "DI_StartButton", False
        )
        self.nodes["DI_StopButton"] = await inputs_folder.add_variable(
            idx, "DI_StopButton", False
        )
        self.nodes["DI_ResetButton"] = await inputs_folder.add_variable(
            idx, "DI_ResetButton", False
        )
        self.nodes["AI_ConveyorSpeed"] = await inputs_folder.add_variable(
            idx, "AI_ConveyorSpeed", 0.0
        )
        self.nodes["AI_MotorCurrent"] = await inputs_folder.add_variable(
            idx, "AI_MotorCurrent", 0.0
        )
        
        # Create output variables
        self.nodes["DO_ConveyorMotor"] = await outputs_folder.add_variable(
            idx, "DO_ConveyorMotor", False
        )
        self.nodes["DO_DiverterActuator"] = await outputs_folder.add_variable(
            idx, "DO_DiverterActuator", False
        )
        self.nodes["DO_AlarmLight"] = await outputs_folder.add_variable(
            idx, "DO_AlarmLight", False
        )
        self.nodes["DO_RunLight"] = await outputs_folder.add_variable(
            idx, "DO_RunLight", False
        )
        self.nodes["AO_ConveyorSpeedSetpoint"] = await outputs_folder.add_variable(
            idx, "AO_ConveyorSpeedSetpoint", 0.0
        )
        
        # Create status variables
        self.nodes["Mode"] = await status_folder.add_variable(idx, "Mode", "MANUAL")
        self.nodes["Running"] = await status_folder.add_variable(idx, "Running", False)
        self.nodes["ScanCount"] = await status_folder.add_variable(idx, "ScanCount", 0)
        self.nodes["ScanTimeMs"] = await status_folder.add_variable(idx, "ScanTimeMs", 0.0)
        
        # Make all nodes writable
        for node in self.nodes.values():
            await node.set_writable()
        
        logger.info(f"OPC UA server starting on opc.tcp://{self.host}:{self.port}")
        
        # Start server
        async with self.server:
            # Start update loop
            await self._update_loop()
    
    async def _update_loop(self):
        """Update OPC UA nodes from PLC"""
        while self.running:
            try:
                # Update inputs
                await self.nodes["DI_ConveyorSensor1"].write_value(
                    self.plc.io_image.DI_ConveyorSensor1
                )
                await self.nodes["DI_ConveyorSensor2"].write_value(
                    self.plc.io_image.DI_ConveyorSensor2
                )
                await self.nodes["DI_ConveyorSensor3"].write_value(
                    self.plc.io_image.DI_ConveyorSensor3
                )
                await self.nodes["DI_DiverterSensor"].write_value(
                    self.plc.io_image.DI_DiverterSensor
                )
                await self.nodes["DI_EStop"].write_value(
                    self.plc.io_image.DI_EStop
                )
                await self.nodes["DI_StartButton"].write_value(
                    self.plc.io_image.DI_StartButton
                )
                await self.nodes["DI_StopButton"].write_value(
                    self.plc.io_image.DI_StopButton
                )
                await self.nodes["DI_ResetButton"].write_value(
                    self.plc.io_image.DI_ResetButton
                )
                await self.nodes["AI_ConveyorSpeed"].write_value(
                    self.plc.io_image.AI_ConveyorSpeed
                )
                await self.nodes["AI_MotorCurrent"].write_value(
                    self.plc.io_image.AI_MotorCurrent
                )
                
                # Update outputs
                await self.nodes["DO_ConveyorMotor"].write_value(
                    self.plc.io_image.DO_ConveyorMotor
                )
                await self.nodes["DO_DiverterActuator"].write_value(
                    self.plc.io_image.DO_DiverterActuator
                )
                await self.nodes["DO_AlarmLight"].write_value(
                    self.plc.io_image.DO_AlarmLight
                )
                await self.nodes["DO_RunLight"].write_value(
                    self.plc.io_image.DO_RunLight
                )
                await self.nodes["AO_ConveyorSpeedSetpoint"].write_value(
                    self.plc.io_image.AO_ConveyorSpeedSetpoint
                )
                
                # Update status
                await self.nodes["Mode"].write_value(str(self.plc.state.mode))
                await self.nodes["Running"].write_value(self.plc.state.running)
                await self.nodes["ScanCount"].write_value(self.plc.state.scan_count)
                await self.nodes["ScanTimeMs"].write_value(self.plc.state.scan_time_ms)
                
            except Exception as e:
                logger.error(f"OPC UA update error: {e}")
            
            await asyncio.sleep(0.1)  # Update at 10Hz
    
    async def stop(self):
        """Stop OPC UA server"""
        if self.running:
            self.running = False
            logger.info("OPC UA server stopped")


async def run_opcua_server(plc, host: str = "0.0.0.0", port: int = 4840):
    """Run OPC UA server"""
    gateway = OPCUAGateway(plc, host, port)
    await gateway.start()

