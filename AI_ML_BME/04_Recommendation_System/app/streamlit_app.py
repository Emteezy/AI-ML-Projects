"""
Streamlit Dashboard for Movie Recommendation System
"""
import streamlit as st
import pandas as pd
import requests
from typing import List, Dict
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.markdown("Get personalized movie recommendations using multiple AI algorithms")

# Sidebar
st.sidebar.header("Settings")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Recommendation Algorithm",
    options=["hybrid", "user_cf", "item_cf", "svd", "nmf", "content_based", "neural_cf"],
    help="Choose the recommendation algorithm"
)

n_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

# Check API health
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code == 200:
        health_data = response.json()
        st.sidebar.success(f"✅ API Connected")
        st.sidebar.info(f"Models loaded: {', '.join(health_data.get('models_loaded', []))}")
    else:
        st.sidebar.error("❌ API Error")
except Exception as e:
    st.sidebar.error(f"❌ Cannot connect to API: {e}")
    st.sidebar.info(f"API URL: {API_URL}")

# Main content
tab1, tab2, tab3 = st.tabs(["Get Recommendations", "Predict Rating", "About"])

with tab1:
    st.header("Get Movie Recommendations")
    
    user_id = st.number_input(
        "User ID",
        min_value=1,
        value=1,
        help="Enter the user ID to get recommendations"
    )
    
    if st.button("Get Recommendations", type="primary"):
        try:
            with st.spinner("Generating recommendations..."):
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={
                        "user_id": int(user_id),
                        "n_recommendations": n_recommendations,
                        "algorithm": algorithm
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data["recommendations"]
                    
                    st.success(f"✅ Generated {len(recommendations)} recommendations using {algorithm}")
                    
                    # Display recommendations
                    if recommendations:
                        rec_df = pd.DataFrame(recommendations)
                        rec_df.columns = ["Movie ID", "Predicted Rating"]
                        rec_df["Rank"] = range(1, len(rec_df) + 1)
                        rec_df = rec_df[["Rank", "Movie ID", "Predicted Rating"]]
                        
                        st.dataframe(rec_df, use_container_width=True)
                        
                        # Visualization
                        st.subheader("Recommendation Scores")
                        st.bar_chart(rec_df.set_index("Rank")["Predicted Rating"])
                    else:
                        st.warning("No recommendations generated")
                else:
                    error_data = response.json()
                    st.error(f"Error: {error_data.get('detail', 'Unknown error')}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("Predict Movie Rating")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_id_pred = st.number_input(
            "User ID",
            min_value=1,
            value=1,
            key="pred_user"
        )
    
    with col2:
        item_id = st.number_input(
            "Movie ID",
            min_value=1,
            value=1,
            key="pred_item"
        )
    
    if st.button("Predict Rating", type="primary"):
        try:
            with st.spinner("Predicting rating..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "user_id": int(user_id_pred),
                        "item_id": int(item_id),
                        "algorithm": algorithm
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predicted_rating = data["predicted_rating"]
                    
                    st.success(f"✅ Predicted Rating: **{predicted_rating:.2f}** / 5.0")
                    
                    # Visualize rating
                    st.progress(predicted_rating / 5.0)
                    st.metric("Predicted Rating", f"{predicted_rating:.2f}", "out of 5.0")
                else:
                    error_data = response.json()
                    st.error(f"Error: {error_data.get('detail', 'Unknown error')}")
        
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.header("About")
    
    st.markdown("""
    ### Movie Recommendation System
    
    This application demonstrates a production-ready recommendation system with multiple algorithms:
    
    #### Algorithms:
    - **Hybrid**: Combines multiple algorithms for best results
    - **User-based CF**: Finds similar users and recommends items they liked
    - **Item-based CF**: Finds similar items based on user preferences
    - **SVD**: Matrix factorization using Singular Value Decomposition
    - **NMF**: Non-negative Matrix Factorization
    - **Content-based**: Uses movie features (genres, year, etc.)
    - **Neural CF**: Deep learning model for recommendations
    
    #### Features:
    - Multiple recommendation algorithms
    - Real-time API (FastAPI)
    - Interactive dashboard (Streamlit)
    - Evaluation metrics (Precision@K, Recall@K, NDCG, MAP)
    - Production-ready deployment
    
    #### API Endpoints:
    - `GET /health` - Health check
    - `POST /recommend` - Get recommendations
    - `POST /predict` - Predict rating
    - `GET /models` - List available models
    """)
    
    st.info("💡 **Tip**: Make sure the API server is running on the configured URL")

