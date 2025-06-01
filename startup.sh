#!/bin/bash

echo "🚀 Starting Coffee Sales ML Application on Azure App Service..."

# Set Python path
export PYTHONPATH="/home/site/wwwroot:$PYTHONPATH"

# Create necessary directories
mkdir -p /home/site/wwwroot/mlruns
mkdir -p /home/site/wwwroot/models

# Check if model file exists
if [ ! -f "/home/site/wwwroot/backend/model.pkl" ] && [ -f "/home/site/wwwroot/ml/model.pkl" ]; then
    echo "📁 Copying model file to backend directory..."
    cp /home/site/wwwroot/ml/model.pkl /home/site/wwwroot/backend/model.pkl
fi

# Start the FastAPI application
echo "🌐 Starting FastAPI backend..."
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0:8000 --workers=1 --timeout=600 --access-logfile=- --error-logfile=- backend.main:app
