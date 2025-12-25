"""
Streamlit web interface for medical image classification.
"""
import streamlit as st
import requests
import io
from PIL import Image
import base64
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Medical Image Classification",
    page_icon="🏥",
    layout="wide",
)

# API endpoint
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

# Title
st.title("🏥 Medical Image Classification")
st.markdown("### Chest X-Ray Pneumonia Detection using Deep Learning")

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
This application uses a deep learning model (ResNet) trained on chest X-ray images 
to detect pneumonia. The model uses transfer learning and provides explainable AI 
visualizations using Grad-CAM.
""")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a chest X-ray image (JPEG or PNG)
2. Click "Classify Image" to get prediction
3. View the Grad-CAM explanation to see which regions the model focuses on
""")

# Check API health
@st.cache_data(ttl=60)
def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

# Main content
api_healthy, health_data = check_api_health()

if not api_healthy:
    st.error(f"⚠️ API is not available at {API_URL}")
    st.info("Please make sure the API server is running:\n```bash\npython -m uvicorn src.api.main:app --reload --port 8000\n```")
    st.stop()

if health_data:
    if health_data.get("model_loaded"):
        st.success("✅ API is healthy and model is loaded")
        if "model_info" in health_data:
            with st.expander("Model Information"):
                st.json(health_data["model_info"])
    else:
        st.warning("⚠️ API is running but model is not loaded")

# File upload
uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray image in JPEG or PNG format"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Input Image", use_container_width=True)
    
    # Prediction button
    if st.button("🔍 Classify Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                # Prepare file for API
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Get prediction
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display prediction
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    
                    # Color based on prediction
                    if prediction == "PNEUMONIA":
                        st.error(f"⚠️ **Prediction: {prediction}**")
                        st.warning("⚠️ This is a medical prediction tool. Please consult a healthcare professional for actual diagnosis.")
                    else:
                        st.success(f"✅ **Prediction: {prediction}**")
                    
                    # Confidence score
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Class probabilities
                    st.subheader("Class Probabilities")
                    probs = result["class_probabilities"]
                    
                    # Progress bars for probabilities
                    for class_name, prob in probs.items():
                        label = "🔴 PNEUMONIA" if class_name == "PNEUMONIA" else "🟢 NORMAL"
                        st.progress(prob, text=f"{label}: {prob:.2%}")
                    
                    # Get explanation
                    st.subheader("🔬 Model Explanation (Grad-CAM)")
                    explain_response = requests.post(f"{API_URL}/explain", files=files, timeout=30)
                    
                    if explain_response.status_code == 200:
                        explain_result = explain_response.json()
                        
                        # Display explanation image
                        explanation_img = base64.b64decode(explain_result["explanation_image"])
                        explanation_pil = Image.open(io.BytesIO(explanation_img))
                        
                        with col2:
                            st.subheader("Explanation Visualization")
                            st.image(explanation_pil, caption="Grad-CAM Heatmap", use_container_width=True)
                        
                        st.info("""
                        **Grad-CAM Visualization:**
                        - The heatmap shows which regions of the image the model focuses on
                        - Red/yellow areas indicate regions important for the prediction
                        - This helps understand the model's decision-making process
                        """)
                    else:
                        st.warning("Could not generate explanation")
                
                else:
                    st.error(f"Prediction failed: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>⚠️ <strong>Disclaimer:</strong> This is a research/educational tool. 
    Not for clinical use. Always consult medical professionals for diagnosis.</p>
    <p>Medical Image Classification API | Built with PyTorch, FastAPI, and Streamlit</p>
</div>
""", unsafe_allow_html=True)

