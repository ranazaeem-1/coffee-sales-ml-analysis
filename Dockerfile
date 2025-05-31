
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for MLflow and models
RUN mkdir -p mlruns models

# Copy model to backend directory if it exists in ml directory
RUN if [ -f ml/model.pkl ]; then cp ml/model.pkl backend/model.pkl; fi

# Ensure model file exists or create placeholder
RUN if [ ! -f backend/model.pkl ]; then \
    echo "Warning: Model file not found. Please run training notebook first." && \
    touch backend/model.pkl; \
    fi

# Expose ports for FastAPI (8000), Streamlit (8501), and MLflow (5000)
EXPOSE 8000 8501 5000

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Coffee Sales ML Application ==="\n\
echo "Starting services..."\n\
\n\
# Check if model exists\n\
if [ ! -s backend/model.pkl ]; then\n\
    echo "WARNING: Model file is empty or missing!"\n\
    echo "Please run the training notebook first to generate model.pkl"\n\
    echo "Training: jupyter notebook ml/training.ipynb"\n\
fi\n\
\n\
# Start MLflow UI in background\n\
echo "Starting MLflow tracking server on port 5000..."\n\
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns &\n\
MLflow_PID=$!\n\
\n\
# Start FastAPI backend in background\n\
echo "Starting FastAPI backend on port 8000..."\n\
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
API_PID=$!\n\
cd ..\n\
\n\
# Wait a moment for backend to start\n\
sleep 5\n\
\n\
# Test API health\n\
echo "Testing API health..."\n\
curl -f http://localhost:8000/health || echo "API health check failed - model may not be loaded"\n\
\n\
# Start Streamlit frontend (blocking)\n\
echo "Starting Streamlit frontend on port 8501..."\n\
echo ""\n\
echo "=== Application URLs ==="\n\
echo "  Frontend Dashboard: http://localhost:8501"\n\
echo "  API Documentation:  http://localhost:8000/docs"\n\
echo "  MLflow Tracking:    http://localhost:5000"\n\
echo ""\n\
echo "If model predictions fail, run training notebook first:"\n\
echo "  jupyter notebook ml/training.ipynb"\n\
echo ""\n\
cd frontend && streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Set default command
CMD ["./start.sh"]

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Labels for documentation
LABEL maintainer="BDA Assignment 4 Team"
LABEL description="Coffee Sales ML Prediction Application with FastAPI, Streamlit, and MLflow"
LABEL version="1.0.0"
LABEL ports="8000,8501,5000"
LABEL features="ML Prediction API, Interactive Dashboard, Experiment Tracking"
