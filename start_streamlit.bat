@echo off
REM Start Streamlit web interface for Medical Image Classification
echo Starting Streamlit Web Interface...
echo.
echo Streamlit will open in your browser at: http://localhost:8501
echo.
echo Make sure the API is running first (run start_api.bat)
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run app/streamlit_app.py

