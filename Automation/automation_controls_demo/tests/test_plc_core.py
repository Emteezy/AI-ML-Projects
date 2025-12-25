"""
Tests for Soft PLC Core
"""
import pytest
import asyncio
from src.soft_plc.plc_core import SoftPLC, PLCMode, AlarmSeverity, AlarmState


@pytest.fixture
def plc():
    """Create PLC instance for testing"""
    return SoftPLC(scan_rate_ms=100)


def test_plc_initialization(plc):
    """Test PLC initializes correctly"""
    assert plc.scan_rate_ms == 100
    assert plc.state.mode == PLCMode.MANUAL
    assert not plc.state.running
    assert plc.state.scan_count == 0


def test_mode_change(plc):
    """Test mode changes"""
    plc.set_mode(PLCMode.AUTO)
    assert plc.state.mode == PLCMode.AUTO
    assert len(plc.event_log) > 0


def test_start_permissive(plc):
    """Test start permissive logic"""
    # Should be true initially (no faults)
    assert plc.state.start_permissive
    
    # E-Stop should block start
    plc.io_image.DI_EStop = True
    plc.state.estop_active = True
    plc.state.mode = PLCMode.FAULT
    plc._update_permissives()
    assert not plc.state.start_permissive


def test_command_start(plc):
    """Test start command"""
    plc.set_mode(PLCMode.AUTO)
    success = plc.command_start()
    assert success
    assert plc.state.running


def test_command_stop(plc):
    """Test stop command"""
    plc.set_mode(PLCMode.AUTO)
    plc.command_start()
    assert plc.state.running
    
    plc.command_stop()
    assert not plc.state.running


def test_command_reset(plc):
    """Test reset from fault"""
    plc.state.mode = PLCMode.FAULT
    plc.state.jam_detected = True
    plc.state.estop_active = False
    
    success = plc.command_reset()
    assert success
    assert plc.state.mode == PLCMode.MANUAL
    assert not plc.state.jam_detected


def test_alarm_raise_and_acknowledge(plc):
    """Test alarm management"""
    # Raise alarm
    plc._raise_alarm("TEST_ALM", "Test alarm", AlarmSeverity.HIGH)
    assert "TEST_ALM" in plc.alarms
    assert plc.alarms["TEST_ALM"].state == AlarmState.ACTIVE
    
    # Acknowledge alarm
    plc.acknowledge_alarm("TEST_ALM")
    assert plc.alarms["TEST_ALM"].state == AlarmState.ACKNOWLEDGED


def test_alarm_clear(plc):
    """Test alarm clearing"""
    plc._raise_alarm("TEST_ALM", "Test alarm", AlarmSeverity.LOW)
    plc._clear_alarm("TEST_ALM")
    assert plc.alarms["TEST_ALM"].state == AlarmState.CLEARED


@pytest.mark.asyncio
async def test_scan_loop(plc):
    """Test PLC scan loop runs"""
    plc.start_scan()
    await asyncio.sleep(0.5)
    
    # Should have executed several scans
    assert plc.state.scan_count > 0
    assert plc.state.scan_time_ms > 0
    
    await plc.stop_scan()


@pytest.mark.asyncio
async def test_output_control_in_auto(plc):
    """Test outputs update correctly in AUTO mode"""
    plc.set_mode(PLCMode.AUTO)
    plc.command_start()
    
    plc.start_scan()
    await asyncio.sleep(0.3)
    
    # Conveyor motor should be on
    assert plc.io_image.DO_ConveyorMotor
    
    await plc.stop_scan()


@pytest.mark.asyncio
async def test_estop_behavior(plc):
    """Test E-Stop triggers fault mode"""
    plc.set_mode(PLCMode.AUTO)
    plc.command_start()
    
    plc.start_scan()
    await asyncio.sleep(0.2)
    
    # Trigger E-Stop
    plc.io_image.DI_EStop = True
    await asyncio.sleep(0.3)
    
    # Should enter fault mode
    assert plc.state.mode == PLCMode.FAULT
    assert plc.state.estop_active
    assert "ALM_EStop" in plc.alarms
    
    await plc.stop_scan()


def test_event_logging(plc):
    """Test event log captures events"""
    initial_count = len(plc.event_log)
    
    plc.set_mode(PLCMode.AUTO)
    plc.command_start()
    plc.command_stop()
    
    # Should have logged events
    assert len(plc.event_log) > initial_count


def test_io_image_dict(plc):
    """Test IO image dictionary export"""
    io_dict = plc.get_io_image_dict()
    
    assert "inputs" in io_dict
    assert "outputs" in io_dict
    assert "DI_ConveyorSensor1" in io_dict["inputs"]
    assert "DO_ConveyorMotor" in io_dict["outputs"]


def test_status_dict(plc):
    """Test status dictionary export"""
    status = plc.get_status()
    
    assert "mode" in status
    assert "running" in status
    assert "scan_count" in status
    assert "estop_active" in status

