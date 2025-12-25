"""
Documentation Generator
Generates commissioning documentation (I/O lists, alarm lists, C&E matrix, checklists)
"""
import csv
import os
from pathlib import Path
from datetime import datetime


def ensure_docs_dir():
    """Ensure docs directory exists"""
    docs_dir = Path("./docs")
    docs_dir.mkdir(exist_ok=True)
    return docs_dir


def generate_io_list():
    """Generate I/O list CSV"""
    docs_dir = ensure_docs_dir()
    
    io_list = [
        # Digital Inputs
        ["DI", "DI_ConveyorSensor1", "Conveyor Sensor 1", "Digital", "Input", "Product detection at position 1"],
        ["DI", "DI_ConveyorSensor2", "Conveyor Sensor 2", "Digital", "Input", "Product detection at position 2"],
        ["DI", "DI_ConveyorSensor3", "Conveyor Sensor 3", "Digital", "Input", "Product detection at position 3"],
        ["DI", "DI_DiverterSensor", "Diverter Sensor", "Digital", "Input", "Diverter position feedback"],
        ["DI", "DI_EStop", "Emergency Stop", "Digital", "Input", "Emergency stop button"],
        ["DI", "DI_StartButton", "Start Button", "Digital", "Input", "Operator start button"],
        ["DI", "DI_StopButton", "Stop Button", "Digital", "Input", "Operator stop button"],
        ["DI", "DI_ResetButton", "Reset Button", "Digital", "Input", "Fault reset button"],
        
        # Analog Inputs
        ["AI", "AI_ConveyorSpeed", "Conveyor Speed", "Analog", "Input", "Actual conveyor speed (RPM)"],
        ["AI", "AI_MotorCurrent", "Motor Current", "Analog", "Input", "Motor current draw (Amps)"],
        
        # Digital Outputs
        ["DO", "DO_ConveyorMotor", "Conveyor Motor", "Digital", "Output", "Conveyor motor contactor"],
        ["DO", "DO_DiverterActuator", "Diverter Actuator", "Digital", "Output", "Diverter solenoid valve"],
        ["DO", "DO_AlarmLight", "Alarm Light", "Digital", "Output", "Alarm indication light"],
        ["DO", "DO_RunLight", "Run Light", "Digital", "Output", "System running indication"],
        
        # Analog Outputs
        ["AO", "AO_ConveyorSpeedSetpoint", "Speed Setpoint", "Analog", "Output", "Conveyor speed command (RPM)"],
    ]
    
    filepath = docs_dir / "IO_List.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Tag", "Description", "Signal Type", "Direction", "Notes"])
        writer.writerows(io_list)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_alarm_list():
    """Generate alarm list CSV"""
    docs_dir = ensure_docs_dir()
    
    alarms = [
        ["ALM_EStop", "Emergency Stop Activated", "CRITICAL", "Safety", "E-Stop button pressed"],
        ["ALM_JamDetected", "Conveyor Jam Detected", "HIGH", "Process", "Product stuck on conveyor"],
        ["ALM_ScanError", "PLC Scan Error", "CRITICAL", "System", "Error in PLC scan cycle"],
        ["ALM_MotorOvercurrent", "Motor Overcurrent", "HIGH", "Equipment", "Motor current exceeds limit"],
        ["ALM_SensorFault", "Sensor Communication Fault", "MEDIUM", "Equipment", "Sensor not responding"],
    ]
    
    filepath = docs_dir / "Alarm_List.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Alarm Tag", "Message", "Severity", "Category", "Description"])
        writer.writerows(alarms)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_cause_effect_matrix():
    """Generate Cause & Effect matrix CSV"""
    docs_dir = ensure_docs_dir()
    
    # Causes (rows) vs Effects (columns)
    causes = [
        "DI_EStop",
        "Jam Detected",
        "Start Command",
        "Stop Command",
        "Mode = AUTO",
        "DI_ConveyorSensor2",
    ]
    
    effects = [
        "DO_ConveyorMotor",
        "DO_DiverterActuator",
        "DO_AlarmLight",
        "ALM_EStop",
        "ALM_JamDetected",
        "Fault Mode"
    ]
    
    # Matrix (X = effect occurs)
    matrix = [
        ["DI_EStop", "", "", "X", "X", "", "X"],
        ["Jam Detected", "X", "", "X", "", "X", "X"],
        ["Start Command", "X", "", "", "", "", ""],
        ["Stop Command", "X", "", "", "", "", ""],
        ["Mode = AUTO", "X", "", "", "", "", ""],
        ["DI_ConveyorSensor2", "", "X", "", "", "", ""],
    ]
    
    filepath = docs_dir / "Cause_Effect_Matrix.csv"
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Cause / Effect"] + effects)
        writer.writerows(matrix)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_fat_checklist():
    """Generate Factory Acceptance Test checklist"""
    docs_dir = ensure_docs_dir()
    
    content = f"""# Factory Acceptance Test (FAT) Checklist
**Project:** Automation Controls Demo  
**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Location:** Factory Floor  
**Tested By:** _______________

## 1. Documentation Review
- [ ] P&ID reviewed and approved
- [ ] I/O list complete and accurate
- [ ] Alarm list reviewed
- [ ] Cause & Effect matrix verified
- [ ] Control philosophy document reviewed

## 2. Hardware Inspection
- [ ] PLC installed and powered
- [ ] All I/O modules present and addressed
- [ ] Wiring inspection complete
- [ ] Panel labeling complete
- [ ] E-Stop circuits verified

## 3. Software Verification
- [ ] PLC program loaded
- [ ] Program version documented
- [ ] Backup created
- [ ] HMI screens functional
- [ ] Communication gateways operational (Modbus, OPC UA)

## 4. Functional Testing

### 4.1 Mode Changes
- [ ] MANUAL mode entry
- [ ] AUTO mode entry
- [ ] FAULT mode behavior verified
- [ ] Mode interlocks functional

### 4.2 Start/Stop Sequences
- [ ] Start permissives verified
- [ ] Start command successful
- [ ] Stop command successful
- [ ] Emergency stop functional

### 4.3 Sensor Testing
- [ ] Sensor 1 detection verified
- [ ] Sensor 2 detection verified
- [ ] Sensor 3 detection verified
- [ ] Diverter sensor verified

### 4.4 Interlock Testing
- [ ] E-Stop interlock verified
- [ ] Jam detection interlock verified
- [ ] Start permissive logic verified
- [ ] Run permissive logic verified

### 4.5 Alarm Testing
- [ ] E-Stop alarm activates
- [ ] Jam alarm activates
- [ ] Alarm acknowledgment functional
- [ ] Alarm clear logic verified
- [ ] Alarm light operates correctly

### 4.6 Control Logic
- [ ] Conveyor motor control verified
- [ ] Speed setpoint control verified
- [ ] Diverter actuator control verified
- [ ] Product tracking functional

## 5. Communication Testing
- [ ] Modbus TCP server responding
- [ ] OPC UA server responding
- [ ] Register mapping verified
- [ ] HMI communication stable

## 6. Performance Testing
- [ ] PLC scan time within limits (<200ms)
- [ ] HMI response time acceptable
- [ ] Historian logging functional
- [ ] Trend data accuracy verified

## 7. Safety Verification
- [ ] E-Stop stops all motion
- [ ] E-Stop cannot be bypassed
- [ ] Fault mode stops outputs
- [ ] Safety interlocks functional

## Sign-Off
**Passed:** [ ] Yes [ ] No  
**Comments:** _______________________________________________  
**Signature:** _________________ **Date:** _________________
"""
    
    filepath = docs_dir / "FAT_Checklist.md"
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_sat_checklist():
    """Generate Site Acceptance Test checklist"""
    docs_dir = ensure_docs_dir()
    
    content = f"""# Site Acceptance Test (SAT) Checklist
**Project:** Automation Controls Demo  
**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Location:** Customer Site  
**Tested By:** _______________

## 1. Installation Verification
- [ ] System installed per drawings
- [ ] Field wiring complete
- [ ] Power supply verified
- [ ] Grounding verified
- [ ] Environmental conditions acceptable

## 2. Field Device Checkout
- [ ] All sensors calibrated
- [ ] Sensor alignment verified
- [ ] Actuators operate smoothly
- [ ] Motor rotation correct
- [ ] E-Stop button accessible

## 3. System Integration
- [ ] Communication to field devices verified
- [ ] HMI accessible from control room
- [ ] Remote access functional (if applicable)
- [ ] Integration with existing systems verified

## 4. Production Run Testing
- [ ] System operates with actual products
- [ ] Throughput meets specifications
- [ ] Product tracking accurate
- [ ] Diverter timing correct
- [ ] No false alarms observed

## 5. Operator Training
- [ ] HMI operation demonstrated
- [ ] Start/stop procedures trained
- [ ] Mode change procedures trained
- [ ] Alarm handling trained
- [ ] Fault reset procedures trained
- [ ] Manual control procedures trained

## 6. Performance Verification
- [ ] Cycle time meets target
- [ ] System uptime > 95% during test period
- [ ] No unexpected stops
- [ ] Trend data reviewed
- [ ] Event log reviewed

## 7. Safety Verification (On-Site)
- [ ] E-Stop accessible from all positions
- [ ] E-Stop tested with production load
- [ ] Guards and interlocks functional
- [ ] Safety signage in place
- [ ] LOTO procedures verified

## 8. Documentation Handover
- [ ] As-built drawings provided
- [ ] Operator manual provided
- [ ] Maintenance manual provided
- [ ] Spare parts list provided
- [ ] Training records completed

## 9. Punch List
| Item | Description | Responsible | Target Date | Status |
|------|-------------|-------------|-------------|--------|
| 1    |             |             |             |        |
| 2    |             |             |             |        |
| 3    |             |             |             |        |

## Sign-Off
**Passed:** [ ] Yes [ ] No  
**Customer Approval:** _________________  
**Date:** _________________  
**Comments:** _______________________________________________
"""
    
    filepath = docs_dir / "SAT_Checklist.md"
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_control_philosophy():
    """Generate control philosophy document"""
    docs_dir = ensure_docs_dir()
    
    content = f"""# Control Philosophy Document
**Project:** Automation Controls Demo - Conveyor Cell  
**Document Version:** 1.0  
**Date:** {datetime.now().strftime('%Y-%m-%d')}

## 1. System Overview
This document describes the control philosophy for a simulated industrial conveyor cell consisting of:
- Main conveyor with variable speed drive
- Three product detection sensors
- Diverter actuator for product routing
- Emergency stop system
- HMI/SCADA interface

## 2. Control Modes

### 2.1 MANUAL Mode
- Operator has direct control of outputs via HMI
- Conveyor motor can be controlled manually
- Diverter can be positioned manually
- Speed setpoint adjustable
- Start/stop interlocks still active
- Safety functions remain active

### 2.2 AUTO Mode
- System operates automatically based on sensor inputs
- Conveyor runs at programmed speed
- Diverter actuates based on product detection at Sensor 2
- Product tracking active
- Jam detection active
- Automatic recovery sequences enabled

### 2.3 FAULT Mode
- Entered when safety or operational fault occurs
- All outputs de-energized
- System requires RESET before operation can resume
- Fault causes logged in event log
- Alarm generated

## 3. Operating Sequences

### 3.1 Start Sequence (AUTO Mode)
1. Verify all start permissives met:
   - No E-Stop active
   - Not in FAULT mode
   - No active jams
2. Operator presses START
3. Run light illuminates
4. Conveyor motor energizes
5. Speed ramps to setpoint
6. Product detection active
7. System enters RUN state

### 3.2 Normal Operation
- Products enter at beginning of conveyor
- Sensor 1 detects product entry
- Product tracked along conveyor
- Sensor 2 triggers diverter decision
  - If divert required: energize diverter actuator
  - If pass through: diverter remains inactive
- Sensor 3 confirms product exit

### 3.3 Stop Sequence
1. Operator presses STOP (or auto-stop triggered)
2. Speed setpoint ramped to zero
3. Conveyor motor de-energizes when speed < 5 RPM
4. Run light extinguishes
5. System enters STOP state

### 3.4 Emergency Stop
1. E-Stop button pressed
2. All motion stops immediately
3. System enters FAULT mode
4. E-Stop alarm raised
5. Operator must clear E-Stop and press RESET

## 4. Interlocks and Permissives

### 4.1 Start Permissives
- E-Stop not active
- System not in FAULT mode
- No jam detected
- (Manual mode: no additional checks)
- (Auto mode: all sensors healthy)

### 4.2 Run Permissives
- All start permissives maintained
- Motor current within limits
- Speed feedback within tolerance

### 4.3 Safety Interlocks
- E-Stop overrides all other commands
- Fault mode disables all outputs
- Jam condition stops conveyor

## 5. Alarm Management

### 5.1 Alarm Priorities
**CRITICAL:** System safety compromised, immediate action required
- E-Stop
- PLC scan error

**HIGH:** Production stopped, operator attention required
- Conveyor jam
- Motor overcurrent

**MEDIUM:** Reduced functionality, attention needed soon
- Sensor fault
- Communication warning

**LOW:** Informational, monitor condition
- Maintenance reminder

### 5.2 Alarm Handling
1. Alarm activates
2. Alarm light flashes
3. Alarm logged in HMI
4. Event recorded with timestamp
5. Operator acknowledges alarm
6. Alarm light steady (if not cleared)
7. Fault condition cleared
8. Alarm auto-clears

## 6. Jam Detection Logic
- Monitor Sensor 1, 2, and 3
- If any sensor remains active > 5 seconds continuously:
  - Jam condition detected
  - Stop conveyor
  - Enter FAULT mode
  - Raise jam alarm
- Operator must:
  1. Physically clear jam
  2. Acknowledge alarm
  3. Press RESET
  4. Restart system

## 7. Data Historian
- Key process values logged at 1 Hz:
  - Conveyor speed (actual)
  - Motor current
  - Speed setpoint
  - Digital output states
- Historian data retained for minimum 7 days
- Trend data accessible via HMI

## 8. Communications

### 8.1 Modbus TCP
- Server on port 5020
- Register map documented separately
- 100ms update rate
- Supports SCADA integration

### 8.2 OPC UA
- Server on port 4840
- Namespace: http://plc.automation.demo
- Node structure documented separately
- 100ms update rate

## 9. HMI Functionality

### 9.1 Overview Screen
- System status display
- Mode indication
- Run/stop state
- Active alarm count
- Control buttons

### 9.2 Manual Control Screen
- Direct output control (MANUAL mode only)
- Speed setpoint adjustment
- Safety interlocks displayed

### 9.3 Alarm Screen
- Active alarm list
- Alarm history
- Acknowledge button
- Alarm details

### 9.4 Trends Screen
- Real-time trends
- Historical data retrieval
- Multi-tag display
- Zoom and pan

### 9.5 Events Screen
- Event log display
- Filterable by type
- Export to CSV

## 10. Maintenance and Diagnostics
- Scan time monitoring (target < 100ms, alarm > 200ms)
- Communication health monitoring
- I/O status indicators
- Fault injection for testing (diagnostics screen)

## Document Approval
**Prepared By:** System Engineer  
**Reviewed By:** _______________  
**Approved By:** _______________  
**Date:** {datetime.now().strftime('%Y-%m-%d')}
"""
    
    filepath = docs_dir / "Control_Philosophy.md"
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_modbus_register_map():
    """Generate Modbus register mapping documentation"""
    docs_dir = ensure_docs_dir()
    
    content = """# Modbus TCP Register Map

## Connection Details
- **Protocol:** Modbus TCP
- **Port:** 5020
- **Unit ID:** 1
- **Update Rate:** 100ms

## Discrete Inputs (Function Code 2)
| Address | Tag | Description |
|---------|-----|-------------|
| 0 | DI_ConveyorSensor1 | Conveyor Sensor 1 |
| 1 | DI_ConveyorSensor2 | Conveyor Sensor 2 |
| 2 | DI_ConveyorSensor3 | Conveyor Sensor 3 |
| 3 | DI_DiverterSensor | Diverter Sensor |
| 4 | DI_EStop | Emergency Stop |
| 5 | DI_StartButton | Start Button |
| 6 | DI_StopButton | Stop Button |
| 7 | DI_ResetButton | Reset Button |

## Coils (Function Code 1, 5, 15)
| Address | Tag | Description | Writable |
|---------|-----|-------------|----------|
| 0 | DO_ConveyorMotor | Conveyor Motor | Read Only |
| 1 | DO_DiverterActuator | Diverter Actuator | Read Only |
| 2 | DO_AlarmLight | Alarm Light | Read Only |
| 3 | DO_RunLight | Run Light | Read Only |

## Holding Registers (Function Code 3, 6, 16)
| Address | Tag | Description | Units | Scale | Writable |
|---------|-----|-------------|-------|-------|----------|
| 0 | AI_ConveyorSpeed | Actual Conveyor Speed | RPM | x10 | Read Only |
| 1 | AI_MotorCurrent | Motor Current | Amps | x100 | Read Only |
| 2 | AO_ConveyorSpeedSetpoint | Speed Setpoint | RPM | x10 | Read Only |
| 3 | PLC_Mode | PLC Mode | Enum | 1 | Read Only |
| 4 | PLC_Running | Running State | Bool | 1 | Read Only |

## PLC Mode Encoding
- 0 = MANUAL
- 1 = AUTO
- 2 = FAULT

## Scaling Notes
- Speed values are multiplied by 10 (e.g., 1000 = 100.0 RPM)
- Current values are multiplied by 100 (e.g., 250 = 2.50 Amps)

## Example Client Code (Python)
```python
from pymodbus.client import ModbusTcpClient

client = ModbusTcpClient('localhost', port=5020)
client.connect()

# Read discrete inputs
result = client.read_discrete_inputs(0, 8)
print(f"Sensors: {result.bits}")

# Read holding registers
result = client.read_holding_registers(0, 5)
speed = result.registers[0] / 10.0
current = result.registers[1] / 100.0
print(f"Speed: {speed} RPM, Current: {current} A")

client.close()
```
"""
    
    filepath = docs_dir / "Modbus_Register_Map.md"
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_opcua_namespace():
    """Generate OPC UA namespace documentation"""
    docs_dir = ensure_docs_dir()
    
    content = """# OPC UA Namespace Documentation

## Connection Details
- **Endpoint:** opc.tcp://localhost:4840/freeopcua/server/
- **Namespace URI:** http://plc.automation.demo
- **Namespace Index:** 2
- **Update Rate:** 100ms
- **Security:** None (demo purposes only)

## Node Structure

```
Objects/
└── PLC/
    ├── Inputs/
    │   ├── DI_ConveyorSensor1 (Boolean)
    │   ├── DI_ConveyorSensor2 (Boolean)
    │   ├── DI_ConveyorSensor3 (Boolean)
    │   ├── DI_DiverterSensor (Boolean)
    │   ├── DI_EStop (Boolean)
    │   ├── DI_StartButton (Boolean)
    │   ├── DI_StopButton (Boolean)
    │   ├── DI_ResetButton (Boolean)
    │   ├── AI_ConveyorSpeed (Double, RPM)
    │   └── AI_MotorCurrent (Double, Amps)
    ├── Outputs/
    │   ├── DO_ConveyorMotor (Boolean)
    │   ├── DO_DiverterActuator (Boolean)
    │   ├── DO_AlarmLight (Boolean)
    │   ├── DO_RunLight (Boolean)
    │   └── AO_ConveyorSpeedSetpoint (Double, RPM)
    └── Status/
        ├── Mode (String)
        ├── Running (Boolean)
        ├── ScanCount (Int32)
        └── ScanTimeMs (Double)
```

## Data Types
- **Boolean:** True/False
- **Double:** Floating point values
- **String:** Text values
- **Int32:** Integer values

## Example Client Code (Python)
```python
from asyncua import Client
import asyncio

async def main():
    client = Client("opc.tcp://localhost:4840/freeopcua/server/")
    await client.connect()
    
    # Get namespace index
    nsidx = await client.get_namespace_index("http://plc.automation.demo")
    
    # Read a variable
    node = client.get_node(f"ns={nsidx};s=PLC.Status.Mode")
    mode = await node.read_value()
    print(f"Mode: {mode}")
    
    # Subscribe to changes
    handler = SubscriptionHandler()
    sub = await client.create_subscription(100, handler)
    await sub.subscribe_data_change(node)
    
    await asyncio.sleep(10)
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Browse Path Examples
- Mode: `Objects/PLC/Status/Mode`
- Conveyor Speed: `Objects/PLC/Inputs/AI_ConveyorSpeed`
- Conveyor Motor: `Objects/PLC/Outputs/DO_ConveyorMotor`
"""
    
    filepath = docs_dir / "OPCUA_Namespace.md"
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Generated: {filepath}")
    return filepath


def generate_all_docs():
    """Generate all commissioning documentation"""
    print("Generating commissioning documentation...")
    
    generate_io_list()
    generate_alarm_list()
    generate_cause_effect_matrix()
    generate_fat_checklist()
    generate_sat_checklist()
    generate_control_philosophy()
    generate_modbus_register_map()
    generate_opcua_namespace()
    
    print("\nAll documentation generated successfully!")
    print("Documentation files created in ./docs/")


if __name__ == "__main__":
    generate_all_docs()

