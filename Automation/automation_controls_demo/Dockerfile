FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY demo.py .

# Create data directory
RUN mkdir -p ./data

# Expose ports
EXPOSE 8001 8002 8501 5020 4840

CMD ["python", "-m", "uvicorn", "src.soft_plc.api:app", "--host", "0.0.0.0", "--port", "8001"]

