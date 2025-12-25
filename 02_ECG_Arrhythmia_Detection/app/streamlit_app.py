"""Streamlit web interface for ECG Arrhythmia Detection."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests
from typing import Optional
import io

from src.config.settings import STREAMLIT_CONFIG, SIGNAL_CONFIG, ARRHYTHMIA_CLASSES


# Page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL
API_URL = STREAMLIT_CONFIG["api_url"]


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def predict_ecg(signal: np.ndarray, model_name: str = "lstm_best.pth") -> Optional[dict]:
    """Send ECG signal to API for prediction."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "signal": signal.tolist(),
                "model_name": model_name
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def analyze_ecg(signal: np.ndarray, model_name: str = "lstm_best.pth") -> Optional[dict]:
    """Send ECG signal to API for detailed analysis."""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={
                "signal": signal.tolist(),
                "model_name": model_name
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def generate_sample_signal(signal_type: str = "normal", duration: float = 10.0) -> np.ndarray:
    """Generate sample ECG signal for demonstration."""
    sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    if signal_type == "normal":
        # Normal sinus rhythm - regular QRS complexes
        signal = np.zeros_like(t)
        heart_rate = 70  # BPM
        rr_interval = 60.0 / heart_rate
        
        for i in range(int(duration / rr_interval)):
            qrs_time = i * rr_interval
            qrs_idx = int(qrs_time * sampling_rate)
            if qrs_idx < len(signal):
                # Simulate QRS complex
                qrs_width = int(0.1 * sampling_rate)  # 100ms
                qrs_signal = np.exp(-0.5 * ((t[qrs_idx:qrs_idx+qrs_width] - qrs_time) / 0.02) ** 2)
                if len(qrs_signal) == qrs_width:
                    signal[qrs_idx:qrs_idx+qrs_width] += qrs_signal * 2.0
                # P wave
                p_idx = max(0, qrs_idx - int(0.2 * sampling_rate))
                if p_idx < len(signal):
                    p_signal = np.exp(-0.5 * ((t[p_idx:qrs_idx] - (qrs_time - 0.2)) / 0.05) ** 2)
                    signal[p_idx:qrs_idx] += p_signal * 0.3
        
        # Add noise
        noise = np.random.normal(0, 0.1, len(signal))
        signal = signal + noise
    
    elif signal_type == "afib":
        # Atrial fibrillation - irregular rhythm
        signal = np.zeros_like(t)
        base_heart_rate = 100  # BPM
        rr_intervals = []
        current_time = 0
        
        while current_time < duration:
            # Variable RR interval (irregular)
            rr_interval = 60.0 / base_heart_rate + np.random.uniform(-0.2, 0.2)
            rr_intervals.append(rr_interval)
            qrs_idx = int(current_time * sampling_rate)
            
            if qrs_idx < len(signal):
                qrs_width = int(0.1 * sampling_rate)
                qrs_signal = np.exp(-0.5 * ((t[qrs_idx:qrs_idx+qrs_width] - current_time) / 0.02) ** 2)
                if len(qrs_signal) == qrs_width:
                    signal[qrs_idx:qrs_idx+qrs_width] += qrs_signal * 2.0
            
            current_time += rr_interval
        
        noise = np.random.normal(0, 0.15, len(signal))
        signal = signal + noise
    
    else:
        # Simple sine wave as fallback
        signal = np.sin(2 * np.pi * 1 * t) + 0.1 * np.sin(2 * np.pi * 60 * t)
    
    return signal


def plot_ecg_signal(signal: np.ndarray, title: str = "ECG Signal", show_prediction: bool = False, 
                    prediction: Optional[str] = None, confidence: Optional[float] = None):
    """Plot ECG signal using Plotly."""
    sampling_rate = SIGNAL_CONFIG["sampling_rate"]
    duration = len(signal) / sampling_rate
    time = np.linspace(0, duration, len(signal))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (mV)",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    if show_prediction and prediction:
        fig.add_annotation(
            text=f"Prediction: {prediction}<br>Confidence: {confidence:.2%}" if confidence else f"Prediction: {prediction}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=14, color="green" if prediction == "Normal" else "red")
        )
    
    st.plotly_chart(fig, use_container_width=True)


# Main app
def main():
    """Main Streamlit application."""
    st.title("💓 ECG Arrhythmia Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API health check
        api_available = check_api_health()
        if api_available:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Not Available")
            st.warning(f"Please ensure the API is running at {API_URL}")
        
        st.markdown("---")
        
        # Model selection
        model_name = st.selectbox(
            "Model",
            ["lstm_best.pth", "transformer_best.pth"],
            index=0
        )
        
        st.markdown("---")
        
        # Signal input method
        input_method = st.radio(
            "Signal Input Method",
            ["Upload File", "Generate Sample", "Manual Input"],
            index=1
        )
    
    # Main content
    if not api_available:
        st.warning(
            "⚠️ **API not available.** Please start the FastAPI server:\n\n"
            "```bash\npython -m uvicorn src.api.main:app --reload --port 8000\n```"
        )
        return
    
    # Signal input
    signal: Optional[np.ndarray] = None
    
    if input_method == "Upload File":
        st.subheader("📁 Upload ECG Signal")
        uploaded_file = st.file_uploader(
            "Upload CSV or NPY file",
            type=["csv", "npy", "txt"],
            help="Upload a file containing ECG signal data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = np.loadtxt(uploaded_file, delimiter=',')
                elif uploaded_file.name.endswith('.npy'):
                    data = np.load(uploaded_file)
                else:
                    data = np.loadtxt(uploaded_file)
                
                # Handle 2D arrays (take first column if needed)
                if len(data.shape) > 1:
                    data = data[:, 0]
                
                signal = data.astype(np.float32)
                st.success(f"✅ Loaded signal with {len(signal)} samples")
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    elif input_method == "Generate Sample":
        st.subheader("🔬 Generate Sample Signal")
        col1, col2 = st.columns(2)
        
        with col1:
            signal_type = st.selectbox(
                "Signal Type",
                ["normal", "afib"],
                index=0,
                help="Type of ECG signal to generate"
            )
        
        with col2:
            duration = st.slider(
                "Duration (seconds)",
                min_value=1.0,
                max_value=30.0,
                value=10.0,
                step=0.5
            )
        
        if st.button("Generate Signal", type="primary"):
            signal = generate_sample_signal(signal_type, duration)
            st.success(f"✅ Generated {signal_type} signal ({len(signal)} samples)")
    
    elif input_method == "Manual Input":
        st.subheader("✍️ Manual Signal Input")
        signal_text = st.text_area(
            "Enter signal values (comma or space separated)",
            height=100,
            help="Enter ECG signal values separated by commas or spaces"
        )
        
        if signal_text:
            try:
                # Parse signal values
                values = signal_text.replace(',', ' ').split()
                signal = np.array([float(v) for v in values], dtype=np.float32)
                st.success(f"✅ Loaded signal with {len(signal)} samples")
            except Exception as e:
                st.error(f"Error parsing signal: {str(e)}")
    
    # Display signal and prediction
    if signal is not None:
        st.markdown("---")
        
        # Check signal length
        min_length = SIGNAL_CONFIG["window_size"]
        if len(signal) < min_length:
            st.warning(
                f"⚠️ Signal is too short ({len(signal)} samples). "
                f"Minimum required: {min_length} samples. "
                "Please provide a longer signal."
            )
        else:
            # Plot signal
            col1, col2 = st.columns([2, 1])
            
            with col1:
                plot_ecg_signal(signal, title="Input ECG Signal")
            
            with col2:
                st.subheader("📊 Signal Info")
                st.metric("Length", f"{len(signal)} samples")
                st.metric("Duration", f"{len(signal) / SIGNAL_CONFIG['sampling_rate']:.2f} s")
                st.metric("Mean", f"{np.mean(signal):.4f}")
                st.metric("Std", f"{np.std(signal):.4f}")
            
            # Prediction buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔮 Predict", type="primary", use_container_width=True):
                    with st.spinner("Analyzing ECG signal..."):
                        result = predict_ecg(signal, model_name)
                        
                        if result:
                            st.success("✅ Prediction complete!")
                            
                            # Display prediction
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Prediction",
                                    result["prediction"],
                                    help="Predicted arrhythmia class"
                                )
                            
                            with col2:
                                st.metric(
                                    "Confidence",
                                    f"{result['confidence']:.2%}",
                                    help="Model confidence in prediction"
                                )
                            
                            with col3:
                                st.metric(
                                    "Model",
                                    result["model_version"],
                                    help="Model used for prediction"
                                )
                            
                            # Class probabilities
                            st.subheader("📈 Class Probabilities")
                            prob_data = result["class_probabilities"]
                            
                            # Create bar chart
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(prob_data.keys()),
                                    y=list(prob_data.values()),
                                    marker_color=['green' if k == result["prediction"] else 'lightblue' 
                                                 for k in prob_data.keys()]
                                )
                            ])
                            fig.update_layout(
                                title="Prediction Probabilities",
                                xaxis_title="Class",
                                yaxis_title="Probability",
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Plot signal with prediction
                            plot_ecg_signal(
                                signal,
                                title="ECG Signal with Prediction",
                                show_prediction=True,
                                prediction=result["prediction"],
                                confidence=result["confidence"]
                            )
            
            with col2:
                if st.button("🔍 Detailed Analysis", use_container_width=True):
                    with st.spinner("Performing detailed analysis..."):
                        result = analyze_ecg(signal, model_name)
                        
                        if result:
                            st.success("✅ Analysis complete!")
                            
                            # Display analysis results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Prediction", result["prediction"])
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                                st.metric("Signal Quality", result["signal_quality"].upper())
                            
                            with col2:
                                st.subheader("📊 Features")
                                features = result["features"]
                                st.metric("Heart Rate", f"{features['heart_rate']:.1f} BPM")
                                st.metric("QRS Duration", f"{features['qrs_duration']*1000:.1f} ms")
                                st.metric("RR Interval", f"{features['rr_mean']:.3f} s")
    
    else:
        # Welcome message
        st.info(
            "👋 **Welcome to the ECG Arrhythmia Detection System!**\n\n"
            "1. Choose a signal input method from the sidebar\n"
            "2. Upload, generate, or input an ECG signal\n"
            "3. Click 'Predict' to classify the signal\n"
            "4. Use 'Detailed Analysis' for comprehensive ECG features\n\n"
            "💡 **Tip:** Start with 'Generate Sample' to see a demonstration!"
        )


if __name__ == "__main__":
    main()

