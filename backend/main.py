from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coffee Sales ML API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = None
model_data = None
try:
    # Try multiple paths for the model
    model_paths = [
        "model.pkl",
        "ml/model.pkl", 
        "../ml/model.pkl",
        os.path.join(os.path.dirname(__file__), "model.pkl"),
        os.path.join(os.path.dirname(__file__), "../ml/model.pkl")
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Load the complete model data
                model_data = joblib.load(path)
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    logger.info(f"Model loaded successfully from {path}")
                    logger.info(f"Model type: {type(model).__name__}")
                    if 'feature_columns' in model_data:
                        logger.info(f"Feature columns: {model_data['feature_columns']}")
                    break
                else:
                    # Assume it's just the model
                    model = model_data
                    model_data = {'model': model}
                    logger.info(f"Model loaded (legacy format) from {path}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
                continue
    
    if model is None:
        logger.error("Could not load model from any path")
        
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class PredictionRequest(BaseModel):
    unit_price: float
    quantity: float
    month: int            # 1-12 for months
    day: int              # 1-31 for day of month
    day_of_week: int      # 0-6 for days of week (0=Monday, 6=Sunday)
    city_encoded: int     # 0-9 for different cities
    product_encoded: int  # 0-4 for different coffee types
    discount_amount: float = 0.0

class PredictionResponse(BaseModel):
    prediction: float
    status: str

class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]

@app.get("/")
async def root():
    return {"message": "Coffee Sales ML API", "status": "running"}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Calculate derived features exactly as in training
        # From training notebook: Used_Discount_Encoded is based on discount amount > 0
        used_discount_encoded = 1 if request.discount_amount > 0 else 0
        
        # Calculate discount rate: discount_amount / sales_amount (unit_price * quantity)
        sales_amount = request.unit_price * request.quantity
        discount_rate = request.discount_amount / sales_amount if sales_amount > 0 else 0.0
        
        # IsWeekend: 1 if day_of_week is 5 (Saturday) or 6 (Sunday), 0 otherwise
        is_weekend = 1 if request.day_of_week in [5, 6] else 0
        
        # Prepare input data in the exact order the model expects:
        # ['Unit Price', 'Quantity', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'City_Encoded', 'Product_Encoded', 'Used_Discount_Encoded', 'Discount_Rate']
        features = np.array([
            [
                request.unit_price,           # Unit Price
                request.quantity,             # Quantity  
                request.month,                # Month
                request.day,                  # Day
                request.day_of_week,         # DayOfWeek
                is_weekend,                  # IsWeekend
                request.city_encoded,        # City_Encoded
                request.product_encoded,     # Product_Encoded
                used_discount_encoded,       # Used_Discount_Encoded
                discount_rate                # Discount_Rate
            ]
        ])
        
        logger.info(f"Making prediction with features: {features[0]}")
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        logger.info(f"Prediction result: {prediction}")
        
        return PredictionResponse(
            prediction=float(prediction),
            status="success"
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Request data: {request}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare batch data
        features_list = []
        for item in request.data:
            # Calculate derived features exactly as in training
            used_discount_encoded = 1 if item.discount_amount > 0 else 0
            sales_amount = item.unit_price * item.quantity
            discount_rate = item.discount_amount / sales_amount if sales_amount > 0 else 0.0
            is_weekend = 1 if item.day_of_week in [5, 6] else 0
            
            features_list.append([
                item.unit_price,           # Unit Price
                item.quantity,             # Quantity
                item.month,                # Month
                item.day,                  # Day
                item.day_of_week,         # DayOfWeek
                is_weekend,               # IsWeekend
                item.city_encoded,        # City_Encoded
                item.product_encoded,     # Product_Encoded
                used_discount_encoded,    # Used_Discount_Encoded
                discount_rate             # Discount_Rate
            ])
        
        features = np.array(features_list)
        predictions = model.predict(features)
        
        # Ensure non-negative predictions
        predictions = np.maximum(0, predictions)
        
        return {
            "predictions": [float(pred) for pred in predictions],
            "status": "success",
            "count": len(predictions)
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get model information
        model_type = type(model).__name__
        
        # Expected feature names based on training notebook
        feature_names = [
            'Unit Price', 'Quantity', 'Month', 'Day', 'DayOfWeek', 
            'IsWeekend', 'City_Encoded', 'Product_Encoded', 
            'Used_Discount_Encoded', 'Discount_Rate'
        ]
        
        # Try to get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Get performance metrics if available from loaded model data
        performance_metrics = None
        if model_data and 'performance_metrics' in model_data:
            performance_metrics = model_data['performance_metrics']
        
        # Get encoders mapping if available
        encoders_info = {}
        if model_data:
            if 'city_encoder' in model_data:
                city_encoder = model_data['city_encoder']
                encoders_info['cities'] = {i: city for i, city in enumerate(city_encoder.classes_)}
            if 'product_encoder' in model_data:
                product_encoder = model_data['product_encoder']
                encoders_info['products'] = {i: product for i, product in enumerate(product_encoder.classes_)}
        
        return {
            "model_type": model_type,
            "feature_importance": feature_importance,
            "performance_metrics": performance_metrics,
            "expected_features": feature_names,
            "feature_count": len(feature_names),
            "encoders_mapping": encoders_info,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to get model info: {str(e)}")

@app.get("/encoders")
async def get_encoders():
    """Get the label encoder mappings for cities and products"""
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model data not loaded")
    
    try:
        encoders = {}
        
        if 'city_encoder' in model_data:
            city_encoder = model_data['city_encoder']
            encoders['cities'] = {
                'classes': city_encoder.classes_.tolist(),
                'mapping': {i: city for i, city in enumerate(city_encoder.classes_)}
            }
        
        if 'product_encoder' in model_data:
            product_encoder = model_data['product_encoder']
            encoders['products'] = {
                'classes': product_encoder.classes_.tolist(),
                'mapping': {i: product for i, product in enumerate(product_encoder.classes_)}
            }
            
        return {
            "encoders": encoders,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Encoders error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to get encoders: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
