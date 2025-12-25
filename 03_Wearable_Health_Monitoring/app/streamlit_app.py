"""Streamlit dashboard for wearable health monitoring."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from typing import Dict, Any, Optional
import time

# Configuration
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Wearable Health Monitoring Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=5)
def fetch_data(endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
    """Fetch data from API with caching."""
    try:
        response = requests.get(f"{API_URL}{endpoint}", params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data from API: {e}")
        return None


def fetch_readings(device_id: Optional[str] = None, sensor_type: Optional[str] = None, limit: int = 1000):
    """Fetch sensor readings."""
    params = {}
    if device_id:
        params["device_id"] = device_id
    if sensor_type:
        params["sensor_type"] = sensor_type
    params["limit"] = limit
    
    data = fetch_data("/api/readings", params)
    return data if data else []


def fetch_health_status(device_id: Optional[str] = None):
    """Fetch health status."""
    endpoint = f"/api/health-status/latest/{device_id}" if device_id else "/api/health-status"
    data = fetch_data(endpoint)
    return data


def fetch_alerts(device_id: Optional[str] = None, acknowledged: bool = False):
    """Fetch anomaly alerts."""
    params = {"acknowledged": acknowledged, "limit": 50}
    if device_id:
        params["device_id"] = device_id
    
    data = fetch_data("/api/alerts", params)
    return data if data else []


def fetch_statistics():
    """Fetch system statistics."""
    data = fetch_data("/api/statistics")
    return data


def create_time_series_chart(df: pd.DataFrame, y_column: str, title: str, color: str = "blue"):
    """Create a time series chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[y_column],
        mode="lines+markers",
        name=y_column,
        line=dict(color=color),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_column,
        hovermode="x unified",
        height=400,
    )
    return fig


def main():
    """Main dashboard function."""
    st.title("🏥 Wearable Health Monitoring Dashboard")
    st.markdown("Real-time health monitoring and analytics")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        
        # Device filter
        device_id = st.text_input("Device ID (optional)", value="")
        
        # Time range filter
        time_range = st.selectbox("Time Range", ["Last hour", "Last 6 hours", "Last 24 hours", "All"])
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # Fetch statistics
    stats = fetch_statistics()
    
    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Readings", stats.get("total_readings", 0))
        
        with col2:
            st.metric("Devices", stats.get("device_count", 0))
        
        with col3:
            st.metric("Total Alerts", stats.get("alerts_count", 0))
        
        with col4:
            unack_alerts = stats.get("unacknowledged_alerts", 0)
            st.metric("Unacknowledged Alerts", unack_alerts, delta=None if unack_alerts == 0 else "Action needed")
    
    # Health Status Section
    st.header("📊 Current Health Status")
    
    device_filter = device_id if device_id else None
    health_status = fetch_health_status(device_filter)
    
    if health_status:
        if isinstance(health_status, list):
            health_status = health_status[0] if health_status else None
        
        if health_status:
            status = health_status.get("status", "unknown")
            confidence = health_status.get("confidence", 0.0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Status indicator
                if status == "normal":
                    st.success(f"✅ Status: {status.upper()} (Confidence: {confidence:.1%})")
                elif status == "warning":
                    st.warning(f"⚠️ Status: {status.upper()} (Confidence: {confidence:.1%})")
                else:
                    st.error(f"🚨 Status: {status.upper()} (Confidence: {confidence:.1%})")
            
            with col2:
                inference_time = health_status.get("inference_time_ms", 0)
                st.metric("Inference Time", f"{inference_time:.2f} ms")
    
    # Alerts Section
    st.header("🚨 Recent Alerts")
    
    alerts = fetch_alerts(device_id=device_filter, acknowledged=False)
    
    if alerts:
        for alert in alerts[:10]:  # Show latest 10
            severity = alert.get("severity", "warning")
            message = alert.get("message", "")
            timestamp = alert.get("timestamp", "")
            alert_id = alert.get("id")
            
            if severity == "critical":
                st.error(f"**CRITICAL** - {message} ({timestamp})")
            else:
                st.warning(f"**WARNING** - {message} ({timestamp})")
            
            if st.button(f"Acknowledge Alert #{alert_id}", key=f"ack_{alert_id}"):
                try:
                    response = requests.post(f"{API_URL}/api/alerts/{alert_id}/acknowledge")
                    if response.status_code == 200:
                        st.success("Alert acknowledged")
                        st.cache_data.clear()
                        time.sleep(0.5)
                        st.rerun()
                except Exception as e:
                    st.error(f"Error acknowledging alert: {e}")
    else:
        st.info("No unacknowledged alerts")
    
    # Sensor Data Visualization
    st.header("📈 Sensor Data")
    
    # Time range calculation
    if time_range == "Last hour":
        start_time = datetime.utcnow() - timedelta(hours=1)
    elif time_range == "Last 6 hours":
        start_time = datetime.utcnow() - timedelta(hours=6)
    elif time_range == "Last 24 hours":
        start_time = datetime.utcnow() - timedelta(hours=24)
    else:
        start_time = None
    
    # Fetch readings
    readings = fetch_readings(device_id=device_filter, limit=1000)
    
    if readings:
        df = pd.DataFrame(readings)
        
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by time range
        if start_time:
            df = df[df["timestamp"] >= start_time]
        
        # Heart Rate Chart
        hr_readings = df[df["sensor_type"] == "heart_rate"]
        if not hr_readings.empty and "heart_rate" in hr_readings.columns:
            st.subheader("Heart Rate")
            fig_hr = create_time_series_chart(
                hr_readings,
                "heart_rate",
                "Heart Rate Over Time",
                color="red"
            )
            # Add threshold lines
            fig_hr.add_hline(y=60, line_dash="dash", line_color="green", annotation_text="Min Normal")
            fig_hr.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Max Normal")
            st.plotly_chart(fig_hr, use_container_width=True)
        
        # SpO2 Chart
        spo2_readings = df[df["sensor_type"] == "pulse_oximeter"]
        if not spo2_readings.empty and "spo2" in spo2_readings.columns:
            st.subheader("SpO2 (Oxygen Saturation)")
            fig_spo2 = create_time_series_chart(
                spo2_readings,
                "spo2",
                "SpO2 Over Time",
                color="blue"
            )
            # Add threshold line
            fig_spo2.add_hline(y=95, line_dash="dash", line_color="orange", annotation_text="Normal Threshold")
            st.plotly_chart(fig_spo2, use_container_width=True)
        
        # Accelerometer Chart
        accel_readings = df[df["sensor_type"] == "accelerometer"]
        if not accel_readings.empty and "acceleration_magnitude" in accel_readings.columns:
            st.subheader("Activity (Accelerometer Magnitude)")
            fig_accel = create_time_series_chart(
                accel_readings,
                "acceleration_magnitude",
                "Activity Level Over Time",
                color="green"
            )
            st.plotly_chart(fig_accel, use_container_width=True)
        
        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(df)
    
    else:
        st.info("No sensor data available. Make sure the API is running and sensors are streaming data.")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()

