import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

# Configure page
st.set_page_config(
    page_title="Coffee Sales ML Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Default mappings (will be updated from API if available)
PRODUCT_MAPPING = {
    "Brazilian": 0,
    "Colombian": 1, 
    "Costa Rica": 2,
    "Ethiopian": 3,
    "Guatemala": 4
}

CITY_MAPPING = {
    "Abha": 0,
    "Buraidah": 1,
    "Dammam": 2,
    "Hail": 3,
    "Jeddah": 4,
    "Khobar": 5,
    "Mecca": 6,
    "Medina": 7,
    "Riyadh": 8,
    "Tabuk": 9
}

DAY_MAPPING = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except:
        return False

def get_encoders():
    """Get encoder mappings from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/encoders")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def update_mappings():
    """Update mappings from API encoders"""
    global PRODUCT_MAPPING, CITY_MAPPING
    
    encoders_data = get_encoders()
    if encoders_data and 'encoders' in encoders_data:
        encoders = encoders_data['encoders']
        
        # Update product mapping
        if 'products' in encoders:
            PRODUCT_MAPPING = {product: i for i, product in encoders['products']['mapping'].items()}
            
        # Update city mapping  
        if 'cities' in encoders:
            CITY_MAPPING = {city: i for i, city in encoders['cities']['mapping'].items()}

def make_prediction(data):
    """Make single prediction"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def make_batch_prediction(data_list):
    """Make batch predictions"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict_batch", json={"data": data_list})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model_info")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

def main():
    st.title("☕ Coffee Sales ML Dashboard")
    st.markdown("### Predict coffee sales using machine learning")
    
    # Check API status
    if not check_api_health():
        st.error("🔴 Backend API is not running. Please start the FastAPI server.")
        st.code("uvicorn backend.main:app --reload")
        return
    
    st.success("🟢 Backend API is running")
    
    # Update mappings from API
    update_mappings()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Single Prediction", 
        "Batch Prediction", 
        "Model Information",
        "Analytics Dashboard"
    ])
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Model Information":
        model_info_page()
    elif page == "Analytics Dashboard":
        analytics_dashboard_page()

def single_prediction_page():
    st.header("Single Sales Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Product Information")
        product = st.selectbox("Coffee Product", list(PRODUCT_MAPPING.keys()))
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=10)
        unit_price = st.number_input("Unit Price (SAR)", min_value=1.0, max_value=500.0, value=35.0, step=1.0)
        discount_amount = st.number_input("Discount Amount (SAR)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    
    with col2:
        st.subheader("Location & Time")
        city = st.selectbox("City", list(CITY_MAPPING.keys()))
        day_of_week = st.selectbox("Day of Week", list(DAY_MAPPING.keys()))
        month = st.selectbox("Month", list(range(1, 13)), index=0)
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=1)
    
    if st.button("Predict Sales", type="primary"):
        # Prepare data matching the exact feature order from the trained model
        prediction_data = {
            "unit_price": float(unit_price),
            "quantity": float(quantity),
            "month": int(month),
            "day": int(day),
            "day_of_week": DAY_MAPPING[day_of_week],
            "city_encoded": CITY_MAPPING[city],
            "product_encoded": PRODUCT_MAPPING[product],
            "discount_amount": float(discount_amount)
        }
        
        # Make prediction
        result = make_prediction(prediction_data)
        
        if result:
            st.success("Prediction Complete!")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Sales", f"SAR {result['prediction']:.2f}")
            with col2:
                revenue = quantity * unit_price - discount_amount
                st.metric("Expected Revenue", f"SAR {revenue:.2f}")
            with col3:
                if result['prediction'] > revenue:
                    profit_margin = ((result['prediction'] - revenue) / revenue * 100) if revenue > 0 else 0
                    st.metric("Profit Margin", f"{profit_margin:.1f}%")
                else:
                    st.metric("Note", "Check pricing strategy")
            
            # Display input summary
            st.subheader("Prediction Summary")
            summary_df = pd.DataFrame([{
                "Product": product,
                "City": city,
                "Quantity": quantity,
                "Unit Price": f"SAR {unit_price:.2f}",
                "Discount": f"SAR {discount_amount:.2f}",
                "Day": day_of_week,
                "Month": month,
                "Day of Month": day,
                "Predicted Sales": f"SAR {result['prediction']:.2f}"
            }])
            st.dataframe(summary_df, use_container_width=True)

def batch_prediction_page():
    st.header("Batch Sales Prediction")
    
    # Option to upload CSV or manual entry
    option = st.radio("Choose input method:", ["Upload CSV", "Manual Entry"])
    
    if option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                if st.button("Generate Predictions"):
                    # Process the dataframe and make predictions
                    predictions_list = []
                    
                    for _, row in df.iterrows():
                        prediction_data = {
                            "unit_price": float(row.get('unit_price', 35)),
                            "quantity": float(row.get('quantity', 10)),
                            "month": int(row.get('month', 1)),
                            "day": int(row.get('day', 1)),
                            "day_of_week": int(row.get('day_of_week', 0)),
                            "city_encoded": int(row.get('city_encoded', 0)),
                            "product_encoded": int(row.get('product_encoded', 0)),
                            "discount_amount": float(row.get('discount_amount', 0))
                        }
                        predictions_list.append(prediction_data)
                    
                    result = make_batch_prediction(predictions_list)
                    
                    if result:
                        df['predicted_sales'] = result['predictions']
                        st.success(f"Generated {result['count']} predictions!")
                        st.dataframe(df)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download predictions as CSV",
                            data=csv,
                            file_name="coffee_sales_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    else:  # Manual Entry
        st.write("Add multiple predictions manually:")
        
        if 'predictions_data' not in st.session_state:
            st.session_state.predictions_data = []
        
        col1, col2, col3 = st.columns(3)
        with col1:
            product = st.selectbox("Product", list(PRODUCT_MAPPING.keys()), key="batch_product")
            quantity = st.number_input("Quantity", min_value=1, value=10, key="batch_quantity")
        with col2:
            unit_price = st.number_input("Unit Price", min_value=1.0, value=35.0, key="batch_price")
            city = st.selectbox("City", list(CITY_MAPPING.keys()), key="batch_city")
        with col3:
            day_of_week = st.selectbox("Day", list(DAY_MAPPING.keys()), key="batch_day")
            month = st.selectbox("Month", list(range(1, 13)), key="batch_month")
            day = st.number_input("Day of Month", min_value=1, max_value=31, value=1, key="batch_day_num")
        
        if st.button("Add to Batch"):
            entry = {
                "product": product,
                "quantity": quantity,
                "unit_price": unit_price,
                "city": city,
                "day_of_week": day_of_week,
                "month": month,
                "day": day
            }
            st.session_state.predictions_data.append(entry)
            st.success("Entry added to batch!")
        
        if st.session_state.predictions_data:
            st.write(f"Current batch size: {len(st.session_state.predictions_data)}")
            
            if st.button("Generate Batch Predictions"):
                predictions_list = []
                for entry in st.session_state.predictions_data:
                    prediction_data = {
                        "unit_price": float(entry['unit_price']),
                        "quantity": float(entry['quantity']),
                        "month": int(entry['month']),
                        "day": int(entry['day']),
                        "day_of_week": DAY_MAPPING[entry['day_of_week']],
                        "city_encoded": CITY_MAPPING[entry['city']],
                        "product_encoded": PRODUCT_MAPPING[entry['product']],
                        "discount_amount": 0.0
                    }
                    predictions_list.append(prediction_data)
                
                result = make_batch_prediction(predictions_list)
                
                if result:
                    # Create results dataframe
                    results_df = pd.DataFrame(st.session_state.predictions_data)
                    results_df['predicted_sales'] = result['predictions']
                    
                    st.success(f"Generated {result['count']} predictions!")
                    st.dataframe(results_df)
                    
                    # Clear batch
                    if st.button("Clear Batch"):
                        st.session_state.predictions_data = []
                        st.experimental_rerun()

def model_info_page():
    st.header("Model Information")
    
    model_info = get_model_info()
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Details")
            st.write(f"**Model Type:** {model_info['model_type']}")
            st.write("**Status:** ✅ Loaded and Ready")
            st.write(f"**Features:** {model_info['feature_count']}")
            
            if model_info.get('performance_metrics'):
                st.subheader("Performance Metrics")
                metrics = model_info['performance_metrics']
                st.write(f"**R² Score:** {metrics.get('test_r2', 'N/A'):.4f}")
                st.write(f"**RMSE:** {metrics.get('test_rmse', 'N/A'):.2f}")
                st.write(f"**MAE:** {metrics.get('test_mae', 'N/A'):.2f}")
                st.write(f"**MAPE:** {metrics.get('mape', 'N/A'):.2f}%")
        
        with col2:
            st.subheader("API Status")
            st.write("**Health:** 🟢 Healthy")
            st.write("**Endpoint:** http://localhost:8000")
            
            if model_info.get('encoders_mapping'):
                st.subheader("Encoders")
                encoders = model_info['encoders_mapping']
                if 'cities' in encoders:
                    st.write("**Cities:**")
                    for code, city in encoders['cities'].items():
                        st.write(f"  {code}: {city}")
                if 'products' in encoders:
                    st.write("**Products:**")
                    for code, product in encoders['products'].items():
                        st.write(f"  {code}: {product}")
        
        if model_info.get('feature_importance'):
            st.subheader("Feature Importance")
            
            importance_df = pd.DataFrame(
                list(model_info['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Feature Importance in Sales Prediction"
            )
            st.plotly_chart(fig, use_container_width=True)

def analytics_dashboard_page():
    st.header("Analytics Dashboard")
    
    # Sample data for demonstration
    st.subheader("Sales Analytics Overview")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='D')
    sample_sales = np.random.normal(1000, 200, len(dates))
    
    sales_df = pd.DataFrame({
        'Date': dates,
        'Sales': sample_sales
    })
    
    # Time series plot
    fig = px.line(sales_df, x='Date', y='Sales', title='Daily Sales Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    # Product performance
    products = list(PRODUCT_MAPPING.keys())
    product_sales = np.random.normal(5000, 1000, len(products))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            x=products, 
            y=product_sales,
            title='Sales by Coffee Product'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            values=product_sales,
            names=products,
            title='Market Share by Product'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()
