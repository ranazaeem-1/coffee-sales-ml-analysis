#!/bin/bash

echo "🚀 Starting Coffee Sales ML Application on Azure App Service..."

# Debug information
echo "Current directory: $(pwd)"
echo "Listing directory contents:"
ls -la
echo "Python version: $(python --version)"

# Set working directory
cd /home/site/wwwroot || echo "Failed to change to /home/site/wwwroot, continuing anyway"
echo "Current working directory: $(pwd)"

# Set Python path
export PYTHONPATH="/home/site/wwwroot:$PYTHONPATH"

# Create logs directory
mkdir -p /home/site/wwwroot/logs

# Debug: Check if requirements.txt exists
if [ -f "/home/site/wwwroot/requirements.txt" ]; then
    echo "📋 Found requirements.txt at /home/site/wwwroot/"
else
    echo "⚠️ requirements.txt not found at /home/site/wwwroot/, checking other locations..."
    if [ -f "requirements.txt" ]; then
        echo "📋 Found requirements.txt in current directory, copying..."
        cp requirements.txt /home/site/wwwroot/
    elif [ -f "../requirements.txt" ]; then
        echo "📋 Found requirements.txt in parent directory, copying..."
        cp ../requirements.txt /home/site/wwwroot/
    else
        echo "❌ requirements.txt not found in any expected location"
        ls -la /home/site
        ls -la /home
    fi
fi

# Install Python dependencies
echo "📦 Installing dependencies..."
if [ -f "/home/site/wwwroot/requirements.txt" ]; then
    pip install -r /home/site/wwwroot/requirements.txt
else
    echo "❌ Could not install dependencies, requirements.txt not found"
fi

# Create necessary directories
mkdir -p /home/site/wwwroot/mlruns
mkdir -p /home/site/wwwroot/models

# Debug: Check backend and frontend directories
echo "Checking for backend directory:"
ls -la /home/site/wwwroot/backend || echo "Backend directory not found"
echo "Checking for frontend directory:"
ls -la /home/site/wwwroot/frontend || echo "Frontend directory not found"

# Copy model to backend if needed
if [ ! -f "/home/site/wwwroot/backend/model.pkl" ] && [ -f "/home/site/wwwroot/ml/model.pkl" ]; then
    echo "📁 Copying model file to backend directory..."
    cp /home/site/wwwroot/ml/model.pkl /home/site/wwwroot/backend/model.pkl
fi

# Check if the Streamlit app exists
if [ ! -f "/home/site/wwwroot/frontend/streamlit_app.py" ]; then
    echo "❌ Streamlit app not found at expected location"
    # Search for it
    find /home/site/wwwroot -name "streamlit_app.py" || echo "Could not find streamlit_app.py"
    # Try to find any Python files that might be the Streamlit app
    find /home/site/wwwroot -name "*.py" | grep -i stream || echo "No files with 'stream' in the name found"
else
    echo "✅ Found Streamlit app at expected location"
fi

# Start Streamlit frontend in background
echo "🧠 Launching Streamlit frontend..."
STREAMLIT_PATH="/home/site/wwwroot/frontend/streamlit_app.py"
if [ -f "$STREAMLIT_PATH" ]; then
    nohup streamlit run "$STREAMLIT_PATH" --server.port=8501 --server.address=0.0.0.0 --server.headless=true > /home/site/wwwroot/logs/streamlit.log 2>&1 &
else
    echo "❌ Could not start Streamlit: $STREAMLIT_PATH not found"
fi

# Give Streamlit time to start
sleep 5

# Check if the FastAPI backend exists
if [ ! -f "/home/site/wwwroot/backend/main.py" ]; then
    echo "❌ FastAPI main.py not found at expected location"
    # Search for it
    find /home/site/wwwroot -name "main.py" || echo "Could not find main.py"
else
    echo "✅ Found FastAPI backend at expected location"
fi

# Start FastAPI backend
echo "🌐 Starting FastAPI backend..."
if [ -f "/home/site/wwwroot/backend/main.py" ]; then
    exec gunicorn --bind=0.0.0.0:8000 --workers=1 --timeout=600 --access-logfile=- --error-logfile=- backend.main:app
else
    echo "❌ Cannot start FastAPI backend: main.py not found"
    # Try to find any app module that might work
    find /home/site/wwwroot -name "*.py" | xargs grep -l "app = FastAPI" || echo "No FastAPI app definition found"
    exit 1
fi
