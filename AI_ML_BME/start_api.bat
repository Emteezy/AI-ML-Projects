@echo off
REM Start FastAPI server for Medical Image Classification
echo Starting Medical Image Classification API...
echo.
echo API will be available at: http://localhost:8000
echo Interactive docs at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
call venv\Scripts\activate.bat
python -m uvicorn src.api.main:app --reload --port 8000

