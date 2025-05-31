# MLflow Tracking Setup for Coffee Sales ML Project

## Quick Start Commands

### 1. Start MLflow Server
```bash
# Option 1: Using batch file (Windows)
start_mlflow.bat

# Option 2: Manual command
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns
```

### 2. Access MLflow UI
- Open browser: `http://localhost:5000`
- View experiments, runs, and metrics
- Compare model performance
- Download artifacts

## What Gets Logged

### Parameters
- Model hyperparameters (n_estimators, max_depth, etc.)
- Feature selection choices
- Dataset information
- Split configurations

### Metrics
- R² Score (train/test)
- RMSE (train/test)
- MAE (train/test)
- MAPE
- Cross-validation scores
- Feature importance values
- Feature correlations

### Artifacts
- Trained models (registered)
- Plots and visualizations
- Feature importance charts
- Model configuration files
- Encoder mappings
- Performance comparison results

### Model Registry
- Registered models for deployment
- Model versioning
- Stage management (Staging/Production)

## Experiment Structure
