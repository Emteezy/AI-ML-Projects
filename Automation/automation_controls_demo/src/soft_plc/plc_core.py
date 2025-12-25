"""
Soft PLC Core Module
Implements PLC-like scan cycle with modes, interlocks, and alarms.
"""
import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class PLCMode(str, Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"
    FAULT = "FAULT"


class AlarmSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlarmState(str, Enum):
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CLEARED = "CLEARED"


@dataclass
class Alarm:
    """Represents an alarm instance"""
    tag: str
    message: str
    severity: AlarmSeverity
    state: AlarmState
    timestamp: datetime
    ack_timestamp: Optional[datetime] = None
    clear_timestamp: Optional[datetime] = None


@dataclass
class IOImage:
    """Process IO Image Table"""
    # Digital Inputs
    DI_ConveyorSensor1: bool = False
    DI_ConveyorSensor2: bool = False
    DI_ConveyorSensor3: bool = False
    DI_DiverterSensor: bool = False
    DI_EStop: bool = False
    DI_StartButton: bool = False
    DI_StopButton: bool = False
    DI_ResetButton: bool = False
    
    # Analog Inputs
    AI_ConveyorSpeed: float = 0.0  # RPM
    AI_MotorCurrent: float = 0.0  # Amps
    
    # Digital Outputs
    DO_ConveyorMotor: bool = False
    DO_DiverterActuator: bool = False
    DO_AlarmLight: bool = False
    DO_RunLight: bool = False
    
    # Analog Outputs
    AO_ConveyorSpeedSetpoint: float = 0.0  # RPM


@dataclass
class PLCState:
    """PLC State Machine"""
    mode: PLCMode = PLCMode.MANUAL
    running: bool = False
    estop_active: bool = False
    jam_detected: bool = False
    scan_count: int = 0
    scan_time_ms: float = 0.0
    last_scan_timestamp: datetime = field(default_factory=datetime.now)
    
    # Permissives
    start_permissive: bool = True
    run_permissive: bool = True


class SoftPLC:
    """
    Soft PLC implementation with scan cycle, modes, and alarms.
    """
    
    def __init__(self, scan_rate_ms: int = 100):
        self.scan_rate_ms = scan_rate_ms
        self.io_image = IOImage()
        self.state = PLCState()
        self.alarms: Dict[str, Alarm] = {}
        self.event_log: List[Dict] = []
        self.running = False
        self._scan_task: Optional[asyncio.Task] = None
        
        # Jam detection parameters
        self.jam_detection_timeout = 5.0  # seconds
        self.last_sensor_change = time.time()
        
        logger.info(f"SoftPLC initialized with scan rate: {scan_rate_ms}ms")
    
    def start_scan(self):
        """Start PLC scan cycle"""
        if not self.running:
            self.running = True
            self._scan_task = asyncio.create_task(self._scan_loop())
            logger.info("PLC scan started")
            self._log_event("PLC_START", "PLC scan loop started")
    
    async def stop_scan(self):
        """Stop PLC scan cycle"""
        if self.running:
            self.running = False
            if self._scan_task:
                await self._scan_task
            logger.info("PLC scan stopped")
            self._log_event("PLC_STOP", "PLC scan loop stopped")
    
    async def _scan_loop(self):
        """Main PLC scan loop"""
        while self.running:
            scan_start = time.time()
            
            try:
                # 1. Read Inputs
                await self._read_inputs()
                
                # 2. Execute Logic
                await self._execute_logic()
                
                # 3. Update Outputs
                await self._update_outputs()
                
                # 4. Update scan statistics
                self.state.scan_count += 1
                self.state.scan_time_ms = (time.time() - scan_start) * 1000
                self.state.last_scan_timestamp = datetime.now()
                
            except Exception as e:
                logger.error(f"Scan error: {e}")
                self._raise_alarm("ALM_ScanError", f"Scan error: {str(e)}", AlarmSeverity.CRITICAL)
            
            # Sleep to maintain scan rate
            elapsed = time.time() - scan_start
            sleep_time = max(0, (self.scan_rate_ms / 1000.0) - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def _read_inputs(self):
        """Read inputs phase - would interface with real I/O in production"""
        # This is updated by the plant simulator via API
        pass
    
    async def _execute_logic(self):
        """Main logic execution phase"""
        # Check E-Stop
        if self.io_image.DI_EStop:
            if not self.state.estop_active:
                self.state.estop_active = True
                self.state.mode = PLCMode.FAULT
                self.state.running = False
                self._raise_alarm("ALM_EStop", "Emergency Stop Activated", AlarmSeverity.CRITICAL)
                self._log_event("ESTOP", "Emergency stop activated")
        else:
            if self.state.estop_active:
                self.state.estop_active = False
                self._clear_alarm("ALM_EStop")
                self._log_event("ESTOP_CLEAR", "Emergency stop cleared")
        
        # Update permissives
        self._update_permissives()
        
        # Check for jam in AUTO mode
        if self.state.mode == PLCMode.AUTO and self.state.running:
            await self._check_jam_detection()
        
        # Handle mode-specific logic
        if self.state.mode == PLCMode.AUTO:
            await self._auto_mode_logic()
        elif self.state.mode == PLCMode.MANUAL:
            await self._manual_mode_logic()
        elif self.state.mode == PLCMode.FAULT:
            await self._fault_mode_logic()
    
    def _update_permissives(self):
        """Update interlocks and permissives"""
        # Start permissive: no E-Stop, not in fault
        self.state.start_permissive = (
            not self.state.estop_active and 
            self.state.mode != PLCMode.FAULT
        )
        
        # Run permissive: all safety conditions met
        self.state.run_permissive = (
            self.state.start_permissive and
            not self.state.jam_detected
        )
        
        # If run permissive lost during operation, stop
        if self.state.running and not self.state.run_permissive:
            self.state.running = False
            self._log_event("AUTO_STOP", "Automatic stop due to permissive loss")
    
    async def _check_jam_detection(self):
        """Detect conveyor jam based on sensor activity"""
        # Simple jam detection: if conveyor running but no sensor changes
        current_time = time.time()
        
        # Check if any sensor is active (product detected)
        any_sensor_active = (
            self.io_image.DI_ConveyorSensor1 or
            self.io_image.DI_ConveyorSensor2 or
            self.io_image.DI_ConveyorSensor3
        )
        
        # If sensors stuck on for too long, detect jam
        if any_sensor_active and (current_time - self.last_sensor_change) > self.jam_detection_timeout:
            if not self.state.jam_detected:
                self.state.jam_detected = True
                self.state.mode = PLCMode.FAULT
                self.state.running = False
                self._raise_alarm("ALM_JamDetected", "Conveyor jam detected", AlarmSeverity.HIGH)
                self._log_event("JAM", "Conveyor jam detected")
    
    async def _auto_mode_logic(self):
        """AUTO mode logic"""
        if self.state.running:
            # Run conveyor at setpoint speed
            self.io_image.AO_ConveyorSpeedSetpoint = 100.0
            
            # Control diverter based on sensor 2
            if self.io_image.DI_ConveyorSensor2:
                self.io_image.DO_DiverterActuator = True
            else:
                self.io_image.DO_DiverterActuator = False
    
    async def _manual_mode_logic(self):
        """MANUAL mode logic - operator control"""
        # In manual mode, outputs controlled via HMI/API
        pass
    
    async def _fault_mode_logic(self):
        """FAULT mode logic - stop all"""
        self.io_image.DO_ConveyorMotor = False
        self.io_image.DO_DiverterActuator = False
        self.io_image.AO_ConveyorSpeedSetpoint = 0.0
    
    async def _update_outputs(self):
        """Update outputs phase"""
        # Run light follows running state
        self.io_image.DO_RunLight = self.state.running
        
        # Alarm light if any active unacknowledged alarms
        self.io_image.DO_AlarmLight = any(
            a.state == AlarmState.ACTIVE for a in self.alarms.values()
        )
        
        # Conveyor motor follows running state in AUTO
        if self.state.mode == PLCMode.AUTO and self.state.running:
            self.io_image.DO_ConveyorMotor = True
        elif self.state.mode != PLCMode.MANUAL:
            self.io_image.DO_ConveyorMotor = False
    
    def _raise_alarm(self, tag: str, message: str, severity: AlarmSeverity):
        """Raise an alarm"""
        if tag not in self.alarms or self.alarms[tag].state == AlarmState.CLEARED:
            alarm = Alarm(
                tag=tag,
                message=message,
                severity=severity,
                state=AlarmState.ACTIVE,
                timestamp=datetime.now()
            )
            self.alarms[tag] = alarm
            logger.warning(f"Alarm raised: {tag} - {message}")
    
    def _clear_alarm(self, tag: str):
        """Clear an alarm"""
        if tag in self.alarms and self.alarms[tag].state != AlarmState.CLEARED:
            self.alarms[tag].state = AlarmState.CLEARED
            self.alarms[tag].clear_timestamp = datetime.now()
            logger.info(f"Alarm cleared: {tag}")
    
    def acknowledge_alarm(self, tag: str):
        """Acknowledge an alarm"""
        if tag in self.alarms and self.alarms[tag].state == AlarmState.ACTIVE:
            self.alarms[tag].state = AlarmState.ACKNOWLEDGED
            self.alarms[tag].ack_timestamp = datetime.now()
            logger.info(f"Alarm acknowledged: {tag}")
            self._log_event("ALARM_ACK", f"Alarm {tag} acknowledged by operator")
    
    def _log_event(self, event_type: str, description: str):
        """Log an event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "description": description,
            "mode": self.state.mode,
            "scan_count": self.state.scan_count
        }
        self.event_log.append(event)
        # Keep only last 1000 events
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def set_mode(self, mode: PLCMode):
        """Set PLC mode"""
        if mode != self.state.mode:
            old_mode = self.state.mode
            self.state.mode = mode
            self._log_event("MODE_CHANGE", f"Mode changed from {old_mode} to {mode}")
            logger.info(f"Mode changed: {old_mode} -> {mode}")
    
    def command_start(self) -> bool:
        """Execute start command"""
        if self.state.start_permissive and not self.state.running:
            self.state.running = True
            self._log_event("START_CMD", "Start command executed")
            logger.info("Start command executed")
            return True
        return False
    
    def command_stop(self):
        """Execute stop command"""
        if self.state.running:
            self.state.running = False
            self._log_event("STOP_CMD", "Stop command executed")
            logger.info("Stop command executed")
    
    def command_reset(self) -> bool:
        """Execute reset command"""
        if self.state.mode == PLCMode.FAULT:
            # Clear fault condition if safe
            if not self.state.estop_active:
                self.state.jam_detected = False
                self.state.mode = PLCMode.MANUAL
                self._log_event("RESET_CMD", "Reset command executed")
                logger.info("Reset command executed")
                return True
        return False
    
    def update_sensor_state(self):
        """Track sensor state changes for jam detection"""
        self.last_sensor_change = time.time()
    
    def get_status(self) -> Dict:
        """Get PLC status"""
        return {
            "mode": self.state.mode,
            "running": self.state.running,
            "scan_count": self.state.scan_count,
            "scan_time_ms": round(self.state.scan_time_ms, 2),
            "estop_active": self.state.estop_active,
            "jam_detected": self.state.jam_detected,
            "start_permissive": self.state.start_permissive,
            "run_permissive": self.state.run_permissive,
            "active_alarms": len([a for a in self.alarms.values() if a.state == AlarmState.ACTIVE])
        }
    
    def get_io_image_dict(self) -> Dict:
        """Get IO image as dictionary"""
        return {
            "inputs": {
                "DI_ConveyorSensor1": self.io_image.DI_ConveyorSensor1,
                "DI_ConveyorSensor2": self.io_image.DI_ConveyorSensor2,
                "DI_ConveyorSensor3": self.io_image.DI_ConveyorSensor3,
                "DI_DiverterSensor": self.io_image.DI_DiverterSensor,
                "DI_EStop": self.io_image.DI_EStop,
                "DI_StartButton": self.io_image.DI_StartButton,
                "DI_StopButton": self.io_image.DI_StopButton,
                "DI_ResetButton": self.io_image.DI_ResetButton,
                "AI_ConveyorSpeed": round(self.io_image.AI_ConveyorSpeed, 2),
                "AI_MotorCurrent": round(self.io_image.AI_MotorCurrent, 2),
            },
            "outputs": {
                "DO_ConveyorMotor": self.io_image.DO_ConveyorMotor,
                "DO_DiverterActuator": self.io_image.DO_DiverterActuator,
                "DO_AlarmLight": self.io_image.DO_AlarmLight,
                "DO_RunLight": self.io_image.DO_RunLight,
                "AO_ConveyorSpeedSetpoint": round(self.io_image.AO_ConveyorSpeedSetpoint, 2),
            }
        }

