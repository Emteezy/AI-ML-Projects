"""
HMI/SCADA Dashboard
Streamlit interface for PLC monitoring and control
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import httpx
from typing import Dict, List

# Configuration
PLC_API_URL = "http://localhost:8001"
PLANT_SIM_URL = "http://localhost:8002"


# Helper functions
def call_api(method: str, endpoint: str, data: dict = None):
    """Call PLC API"""
    try:
        url = f"{PLC_API_URL}{endpoint}"
        if method == "GET":
            response = httpx.get(url, timeout=2.0)
        elif method == "POST":
            response = httpx.post(url, json=data, timeout=2.0)
        else:
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def get_status() -> Dict:
    """Get PLC status"""
    return call_api("GET", "/status") or {}


def get_io_image() -> Dict:
    """Get IO image"""
    return call_api("GET", "/io") or {"inputs": {}, "outputs": {}}


def get_alarms() -> List:
    """Get active alarms"""
    result = call_api("GET", "/alarms")
    return result.get("alarms", []) if result else []


def get_events(limit: int = 50) -> List:
    """Get event log"""
    result = call_api("GET", f"/events?limit={limit}")
    return result.get("events", []) if result else []


def get_trend_data(tag: str, hours: int = 1) -> List:
    """Get trend data"""
    result = call_api("GET", f"/trends/{tag}?hours={hours}")
    return result.get("data", []) if result else []


# Page configuration
st.set_page_config(
    page_title="PLC HMI/SCADA",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-indicator {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    margin: 10px;
}
.running { background-color: #90EE90; color: #006400; }
.stopped { background-color: #FFB6C1; color: #8B0000; }
.fault { background-color: #FFA500; color: #8B4513; }
.alarm-critical { background-color: #FF4444; color: white; }
.alarm-high { background-color: #FFA500; color: white; }
.alarm-medium { background-color: #FFD700; color: black; }
.alarm-low { background-color: #87CEEB; color: black; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏭 PLC HMI/SCADA")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Manual Control", "Alarms", "Trends", "Events", "Diagnostics"]
)

st.sidebar.markdown("---")

# Get current status
status = get_status()
io_image = get_io_image()

# Status indicators in sidebar
if status:
    mode = status.get("mode", "UNKNOWN")
    running = status.get("running", False)
    active_alarms = status.get("active_alarms", 0)
    
    st.sidebar.metric("Mode", mode)
    st.sidebar.metric("Running", "YES" if running else "NO")
    st.sidebar.metric("Active Alarms", active_alarms)
    st.sidebar.metric("Scan Time", f"{status.get('scan_time_ms', 0):.2f} ms")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto Refresh (2s)", value=True)
if auto_refresh:
    time.sleep(2)
    st.rerun()

# Main content
st.title("PLC HMI/SCADA Dashboard")

# ========== OVERVIEW PAGE ==========
if page == "Overview":
    st.header("System Overview")
    
    if status:
        # Top row - Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mode = status.get("mode", "UNKNOWN")
            mode_class = "running" if mode == "AUTO" else ("fault" if mode == "FAULT" else "stopped")
            st.markdown(f'<div class="big-indicator {mode_class}">{mode}</div>', 
                       unsafe_allow_html=True)
            st.caption("PLC Mode")
        
        with col2:
            running = status.get("running", False)
            run_class = "running" if running else "stopped"
            run_text = "RUNNING" if running else "STOPPED"
            st.markdown(f'<div class="big-indicator {run_class}">{run_text}</div>', 
                       unsafe_allow_html=True)
            st.caption("System State")
        
        with col3:
            estop = status.get("estop_active", False)
            estop_class = "fault" if estop else "running"
            estop_text = "ACTIVE" if estop else "NORMAL"
            st.markdown(f'<div class="big-indicator {estop_class}">{estop_text}</div>', 
                       unsafe_allow_html=True)
            st.caption("E-Stop")
        
        with col4:
            alarms = status.get("active_alarms", 0)
            alarm_class = "fault" if alarms > 0 else "running"
            st.markdown(f'<div class="big-indicator {alarm_class}">{alarms}</div>', 
                       unsafe_allow_html=True)
            st.caption("Active Alarms")
        
        st.markdown("---")
        
        # Control buttons
        st.subheader("Control Commands")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("▶️ START", use_container_width=True, type="primary"):
                result = call_api("POST", "/command/start")
                if result:
                    st.success("Start command sent")
                    time.sleep(0.5)
                    st.rerun()
        
        with col2:
            if st.button("⏹️ STOP", use_container_width=True):
                result = call_api("POST", "/command/stop")
                if result:
                    st.success("Stop command sent")
                    time.sleep(0.5)
                    st.rerun()
        
        with col3:
            if st.button("🔄 RESET", use_container_width=True):
                result = call_api("POST", "/command/reset")
                if result:
                    st.success("Reset command sent")
                    time.sleep(0.5)
                    st.rerun()
        
        with col4:
            mode_change = st.selectbox(
                "Change Mode",
                ["MANUAL", "AUTO", "FAULT"],
                index=["MANUAL", "AUTO", "FAULT"].index(status.get("mode", "MANUAL"))
            )
            if st.button("Set Mode", use_container_width=True):
                result = call_api("POST", "/mode", {"mode": mode_change})
                if result:
                    st.success(f"Mode changed to {mode_change}")
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown("---")
        
        # IO Status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Inputs")
            inputs = io_image.get("inputs", {})
            
            # Digital inputs
            st.markdown("**Digital Inputs**")
            for key, value in inputs.items():
                if key.startswith("DI_"):
                    status_icon = "🟢" if value else "⚫"
                    st.text(f"{status_icon} {key}: {value}")
            
            # Analog inputs
            st.markdown("**Analog Inputs**")
            for key, value in inputs.items():
                if key.startswith("AI_"):
                    st.metric(key, f"{value:.2f}")
        
        with col2:
            st.subheader("📤 Outputs")
            outputs = io_image.get("outputs", {})
            
            # Digital outputs
            st.markdown("**Digital Outputs**")
            for key, value in outputs.items():
                if key.startswith("DO_"):
                    status_icon = "🟢" if value else "⚫"
                    st.text(f"{status_icon} {key}: {value}")
            
            # Analog outputs
            st.markdown("**Analog Outputs**")
            for key, value in outputs.items():
                if key.startswith("AO_"):
                    st.metric(key, f"{value:.2f}")

# ========== MANUAL CONTROL PAGE ==========
elif page == "Manual Control":
    st.header("Manual Control")
    
    if status.get("mode") != "MANUAL":
        st.warning("⚠️ PLC must be in MANUAL mode for manual control")
    else:
        st.success("✅ Manual mode active")
        
        outputs = io_image.get("outputs", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Digital Outputs")
            
            do_conveyor = st.checkbox(
                "Conveyor Motor",
                value=outputs.get("DO_ConveyorMotor", False)
            )
            if st.button("Update Conveyor Motor"):
                call_api("POST", "/output", {"tag": "DO_ConveyorMotor", "value": do_conveyor})
                st.success("Updated")
                time.sleep(0.5)
                st.rerun()
            
            do_diverter = st.checkbox(
                "Diverter Actuator",
                value=outputs.get("DO_DiverterActuator", False)
            )
            if st.button("Update Diverter"):
                call_api("POST", "/output", {"tag": "DO_DiverterActuator", "value": do_diverter})
                st.success("Updated")
                time.sleep(0.5)
                st.rerun()
        
        with col2:
            st.subheader("Analog Outputs")
            
            ao_speed = st.slider(
                "Conveyor Speed Setpoint (RPM)",
                0.0, 150.0,
                value=float(outputs.get("AO_ConveyorSpeedSetpoint", 0.0))
            )
            if st.button("Update Speed Setpoint"):
                call_api("POST", "/output", {"tag": "AO_ConveyorSpeedSetpoint", "value": ao_speed})
                st.success("Updated")
                time.sleep(0.5)
                st.rerun()

# ========== ALARMS PAGE ==========
elif page == "Alarms":
    st.header("Alarm Management")
    
    alarms = get_alarms()
    
    if not alarms:
        st.success("✅ No active alarms")
    else:
        st.warning(f"⚠️ {len(alarms)} alarm(s) active")
        
        for alarm in alarms:
            severity = alarm.get("severity", "LOW")
            state = alarm.get("state", "ACTIVE")
            
            severity_class = f"alarm-{severity.lower()}"
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f'<div class="{severity_class}" style="padding:10px; border-radius:5px;">'
                              f'<b>{alarm.get("tag")}</b>: {alarm.get("message")}<br>'
                              f'<small>Time: {alarm.get("timestamp")}</small>'
                              f'</div>', unsafe_allow_html=True)
                
                with col2:
                    st.text(f"State: {state}")
                    st.text(f"Severity: {severity}")
                
                with col3:
                    if state == "ACTIVE":
                        if st.button(f"Acknowledge", key=alarm.get("tag")):
                            result = call_api("POST", "/alarms/acknowledge", 
                                            {"tag": alarm.get("tag")})
                            if result:
                                st.success("Acknowledged")
                                time.sleep(0.5)
                                st.rerun()
                
                st.markdown("---")

# ========== TRENDS PAGE ==========
elif page == "Trends":
    st.header("Historical Trends")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        tag = st.selectbox(
            "Select Tag",
            ["AI_ConveyorSpeed", "AI_MotorCurrent", "AO_ConveyorSpeedSetpoint",
             "DO_ConveyorMotor", "DO_AlarmLight"]
        )
    
    with col2:
        hours = st.selectbox("Time Range", [1, 2, 4, 8, 24], index=0)
    
    # Get trend data
    trend_data = get_trend_data(tag, hours)
    
    if trend_data:
        df = pd.DataFrame(trend_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines',
            name=tag,
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title=f"Trend: {tag}",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current", f"{df['value'].iloc[-1]:.2f}")
        col2.metric("Average", f"{df['value'].mean():.2f}")
        col3.metric("Min", f"{df['value'].min():.2f}")
        col4.metric("Max", f"{df['value'].max():.2f}")
    else:
        st.info("No trend data available")

# ========== EVENTS PAGE ==========
elif page == "Events":
    st.header("Event Log")
    
    limit = st.slider("Number of events", 10, 200, 50)
    events = get_events(limit)
    
    if events:
        df = pd.DataFrame(events)
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            df[['timestamp', 'type', 'description', 'mode']],
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Event Log",
            data=csv,
            file_name=f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No events logged")

# ========== DIAGNOSTICS PAGE ==========
elif page == "Diagnostics":
    st.header("System Diagnostics")
    
    if status:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PLC Status")
            st.json(status)
        
        with col2:
            st.subheader("Permissives")
            st.metric("Start Permissive", "✅ OK" if status.get("start_permissive") else "❌ NOT MET")
            st.metric("Run Permissive", "✅ OK" if status.get("run_permissive") else "❌ NOT MET")
            st.metric("E-Stop Active", "❌ YES" if status.get("estop_active") else "✅ NO")
            st.metric("Jam Detected", "❌ YES" if status.get("jam_detected") else "✅ NO")
        
        st.markdown("---")
        
        # Plant simulator control
        st.subheader("Plant Simulator (Fault Injection)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Conveyor Jam**")
            if st.button("Inject Jam"):
                try:
                    httpx.post(f"{PLANT_SIM_URL}/fault", 
                             json={"fault_type": "jam", "active": True},
                             timeout=2.0)
                    st.success("Jam injected")
                except:
                    st.error("Failed to inject jam")
            
            if st.button("Clear Jam"):
                try:
                    httpx.post(f"{PLANT_SIM_URL}/fault",
                             json={"fault_type": "jam", "active": False},
                             timeout=2.0)
                    st.success("Jam cleared")
                except:
                    st.error("Failed to clear jam")
        
        with col2:
            st.markdown("**E-Stop**")
            if st.button("Activate E-Stop"):
                try:
                    httpx.post(f"{PLANT_SIM_URL}/fault",
                             json={"fault_type": "estop", "active": True},
                             timeout=2.0)
                    st.success("E-Stop activated")
                except:
                    st.error("Failed")
            
            if st.button("Clear E-Stop"):
                try:
                    httpx.post(f"{PLANT_SIM_URL}/fault",
                             json={"fault_type": "estop", "active": False},
                             timeout=2.0)
                    st.success("E-Stop cleared")
                except:
                    st.error("Failed")
        
        with col3:
            st.markdown("**Sensor Stuck**")
            if st.button("Inject Sensor Fault"):
                try:
                    httpx.post(f"{PLANT_SIM_URL}/fault",
                             json={"fault_type": "sensor_stuck", "active": True},
                             timeout=2.0)
                    st.success("Sensor fault injected")
                except:
                    st.error("Failed")
            
            if st.button("Clear Sensor Fault"):
                try:
                    httpx.post(f"{PLANT_SIM_URL}/fault",
                             json={"fault_type": "sensor_stuck", "active": False},
                             timeout=2.0)
                    st.success("Sensor fault cleared")
                except:
                    st.error("Failed")

