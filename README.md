# Coffee Sales Machine Learning Analysis


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.5%2B-orange)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning project analyzing coffee sales data from Saudi Arabia (January-June 2023) to extract business insights and build predictive models with a complete web application stack.

## 📊 Project Overview

This project performs end-to-end machine learning analysis on coffee sales data including:
- **Data Exploration & Preprocessing**
- **Feature Engineering & Model Training**
- **Random Forest Regressor for Sales Prediction**
- **Interactive Web Dashboard with Streamlit**
- **RESTful API with FastAPI**
- **MLflow Experiment Tracking**

### Dataset Information
- **Source**: Coffee sales transactions from 10 Saudi Arabian cities
- **Period**: January 2023 - December 2023 (730 records)
- **Products**: 5 coffee varieties (Brazilian, Colombian, Costa Rica, Ethiopian, Guatemala)
- **Cities**: Abha, Buraidah, Dammam, Hail, Jeddah, Khobar, Mecca, Medina, Riyadh, Tabuk
- **Features**: Date, Customer ID, City, Product, Pricing, Discounts, Sales metrics

## 🏗️ Project Structure

```
BDA_Assignment_4/
├── ml/
│   └── training.ipynb          # Main ML analysis notebook
├── backend/
│   └── main.py                 # FastAPI backend service
├── frontend/
│   └── streamlit_app.py        # Interactive dashboard
├── mlflow_tracking/
│   └── tracking_setup.md       # MLflow configuration
├── DatasetForCoffeeSales2.csv  # Raw dataset
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone "https://github.com/ranazaeem-1/coffee-sales-ml-analysis.git"
   cd coffee-sales-ml-analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook ml/training.ipynb
   ```

## 📈 Analysis Components

### 1. Data Exploration & Preprocessing
- Data quality assessment
- Missing value analysis
- Feature creation and engineering
- Date/time feature extraction

### 2. Exploratory Data Analysis (EDA)
- Sales trend analysis
- Product performance comparison
- Geographic sales distribution
- Seasonal pattern identification
- Correlation analysis

### 3. Machine Learning Models
- **Linear Regression**
- **Ridge & Lasso Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regression**

### 4. Model Evaluation
- Cross-validation
- Hyperparameter tuning
- Performance metrics (R², RMSE, MAE)
- Feature importance analysis

### 5. Customer Segmentation
- K-means clustering
- Customer behavior analysis
- Segment profiling
- Revenue contribution analysis

### 6. Time Series Analysis
- Daily sales trend analysis
- Seasonal decomposition
- Forecasting models
- Business trend identification

## 🎯 Key Findings

### Model Performance
- **Best Model**: Random Forest Regressor
- **R² Score**: 0.95+ (varies by run)
- **Key Features**: Quantity, Unit Price, Product Type

### Customer Insights
- **4 Customer Segments** identified
- **Top 20% customers** contribute to 60%+ of revenue
- **Seasonal patterns** in purchasing behavior

### Product Performance
- **Colombian coffee** shows highest revenue
- **Ethiopian coffee** has premium pricing
- **Discount strategy** effectiveness varies by product

## 🔧 Running the Applications

### Jupyter Notebook Analysis
```bash
jupyter notebook ml/training.ipynb
```

### Streamlit Dashboard
```bash
streamlit run frontend/streamlit_app.py
```

### FastAPI Backend
```bash
uvicorn backend.main:app --reload
```

### Docker Deployment
```bash
docker build -t coffee-sales-ml-analysis .
docker run -p 8000:8000 coffee-sales-ml-analysis
```

## 📊 Results & Visualizations

The notebook generates comprehensive visualizations including:
- Sales trend charts
- Product performance comparisons
- Customer segmentation plots
- Correlation heatmaps
- Prediction vs actual plots
- Feature importance charts

## 🔍 Business Recommendations

Based on the analysis, key recommendations include:
1. **Product Focus**: Enhance Colombian coffee marketing
2. **Customer Retention**: Implement loyalty programs for high-value segments
3. **Seasonal Strategy**: Adjust inventory based on monthly patterns
4. **Geographic Expansion**: Replicate successful city strategies
5. **Pricing Optimization**: Implement dynamic pricing strategies

## 📚 Technologies Used

- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Time Series**: Statsmodels
- **Web Framework**: FastAPI, Streamlit
- **Development**: Jupyter, Docker

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or collaboration opportunities, please reach out through:
- GitHub Issues
- Email: [zaeemrana168@gmail.com]

## 🙏 Acknowledgments

- Saudi Arabia coffee market data providers
- Open source community for excellent ML libraries
- Big Data Analytics course instructors and peers

---

**Note**: This is an academic project. The dataset and analysis are for educational purposes.
