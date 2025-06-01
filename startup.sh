#!/bin/bash

echo "🚀 Starting Coffee Sales ML Application on Azure App Service..."

# Set working directory
cd /home/site/wwwroot

# Set Python path
export PYTHONPATH="/home/site/wwwroot:$PYTHONPATH"

# Create logs directory
mkdir -p /home/site/wwwroot/logs

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r /home/site/wwwroot/requirements.txt

# Create necessary directories
mkdir -p /home/site/wwwroot/mlruns
mkdir -p /home/site/wwwroot/models

# Copy model to backend if needed
if [ ! -f "/home/site/wwwroot/backend/model.pkl" ] && [ -f "/home/site/wwwroot/ml/model.pkl" ]; then
    echo "📁 Copying model file to backend directory..."
    cp /home/site/wwwroot/ml/model.pkl /home/site/wwwroot/backend/model.pkl
fi

# Start Streamlit frontend in background
echo "🧠 Launching Streamlit frontend..."
nohup streamlit run /home/site/wwwroot/frontend/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true > /home/site/wwwroot/logs/streamlit.log 2>&1 &

# Give Streamlit time to start
sleep 5

# Start FastAPI backend
echo "🌐 Starting FastAPI backend..."
exec gunicorn --bind=0.0.0.0:8000 --workers=1 --timeout=600 --access-logfile=- --error-logfile=- backend.main:app
