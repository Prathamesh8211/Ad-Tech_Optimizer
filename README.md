# Ad-Tech Optimizer Dashboard

An end-to-end advertising technology optimization platform that processes ad performance data, trains machine learning models for click-through rate (CTR) and conversion rate prediction, and provides an interactive unified dashboard to optimize ad spend and campaign performance.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Data Flow](#data-flow)
- [Model Information](#model-information)
- [Dashboard Walkthrough](#dashboard-walkthrough)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Ad-Tech Optimizer is a comprehensive solution designed to help advertisers and agencies maximize their return on ad spend (ROAS) by:

- Processing raw advertising performance data through a multi-layer ETL pipeline
- Training and evaluating machine learning models to predict ad performance
- Providing an interactive, unified dashboard with actionable insights
- Enabling data-driven decisions for budget allocation, creative optimization, and campaign strategy

## 🏗️ Architecture

The system follows a modern data lakehouse architecture with three distinct layers:

```
Raw Data → [Bronze Layer] → [Silver Layer] → [Gold Layer] → [ML Models] → [Streamlit Dashboard]
```

**Layer Breakdown:**

- **Bronze Layer:** Raw data ingestion and initial validation
- **Silver Layer:** Data cleaning, deduplication, and standardization
- **Gold Layer:** Feature engineering, aggregation, and preparation for ML
- **ML Modeling Layer:** Model training, evaluation, and explainability
- **Presentation Layer:** Streamlit-based interactive dashboard

## ⭐ Key Features

### Data Pipeline
- Automated ETL processes for ad performance data
- Data quality validation and anomaly detection
- Feature engineering for CTR and conversion prediction
- S3 integration for scalable data storage

### Machine Learning
- Classification models for click prediction
- Regression models for conversion value forecasting
- Model comparison and selection framework
- SHAP-based model explainability
- MLflow integration for experiment tracking

### Dashboard Capabilities
- **Unified Workspace:** A single combined perspective merging business-owner KPIs and agency/campaign metrics — no separate logins or selections
- **Three Focused Tabs:** Overview, Analytics, and Budget Optimizer
- **Clean Light Theme:** High-contrast, readability-first UI
- **Interactive Filters:** Global filters for ad category and date ranges
- **Performance Analytics:** Real-time KPI tracking and trend analysis
- **Predictive Insights:** ML-powered forecasting and recommendations
- **Budget Optimization:** ROAS-weighted budget allocation models
- **Data Export:** CSV/JSON export of filtered datasets

## 📁 Project Structure

```
Ad-Tech_Optimizer/
├── README.md
├── .gitignore
│
├── dashboard/                     # Streamlit dashboard (application root)
│   ├── app.py                     # Main application entry point
│   ├── requirements.txt           # Dashboard-specific dependencies
│   ├── sample_ad_performance.csv  # Sample data for demo mode
│   ├── .env                       # Environment variables (S3 / AWS)
│   ├── mlflow.db                  # MLflow tracking database
│   ├── .streamlit/                # Streamlit theme configuration
│   ├── pages/                     # Dashboard tab modules
│   │   ├── _01_dashboard_home.py  #   → Overview tab
│   │   ├── _02_predictions.py     #   → Analytics (Strategic Predictions)
│   │   ├── _03_performance_insights.py  # → Analytics (Performance Insights)
│   │   ├── _04_recommendations.py #   → Budget Optimizer tab
│   │   └── _05_ai_copilot.py      #   → Optional AI copilot module
│   ├── utils/                     # Helper modules
│   │   ├── data_loader.py
│   │   └── model_loader.py
│   └── llm/
│       └── copilot.py
│
├── databricks_pipeline/           # Production pipeline + notebook versions
│   ├── Bronze_layer/              # Raw data ingestion
│   ├── Silver_layer/              # Data cleaning and transformation
│   ├── Gold_layer/                # Feature engineering and modeling prep
│   ├── ML_Modeling_layer/         # Model training and evaluation
│   ├── Verification/              # SQL verification scripts per layer
│   └── *.ipynb                    # Jupyter notebooks mirroring the pipeline
│
├── yaml_config/                   # Pipeline configuration files
│   ├── data_quality_rules.yaml
│   ├── llm_semantic_mapping.yaml
│   └── pipeline_manifest.yaml
│
├── docs/                          # Project documentation
│   └── ad-tech-optimizer-interview-plan.md
│
└── Synthetic_Data/                # Data generation and raw datasets
    ├── raw_synthetic_ad_click_data.csv
    ├── ad_catalog_raw.py
    └── raw_synthetic_ad_click_data.py
```

> **Note:** The `venv/` virtual environment folder is local-only and is excluded from GitHub via `.gitignore`.

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- AWS account (optional — only for live S3 integration; local demo data works out of the box)
- Databricks workspace (optional — for production deployment)

### Step-by-Step Installation

**1. Clone the repository**

```
git clone https://github.com/Prathamesh8211/Ad-Tech_Optimizer.git
cd Ad-Tech_Optimizer
```

**2. Create and activate a virtual environment**

```
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies**

```
pip install -r dashboard/requirements.txt
```

**4. Configure environment variables (optional)**

Edit `dashboard/.env` with your AWS credentials only if you want to connect to live S3 data:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your-region
S3_BUCKET=your-ad-tech-bucket
```

### Data Setup Options

**Option 1: Use Demo Data (Recommended)**
- The dashboard automatically loads `dashboard/sample_ad_performance.csv`
- No configuration needed — works out of the box

**Option 2: Connect to Production S3**
- Configure `dashboard/.env` as shown above
- Ensure your S3 bucket contains properly formatted parquet/csv files matching the expected schema

## ▶️ Usage

### Running the Data Pipeline (Optional)

```
# Bronze layer ingestion
python databricks_pipeline/Bronze_layer/01_LOAD_TO_BRONZE.py

# Silver layer cleaning
python databricks_pipeline/Silver_layer/03_CLEAN_TO_SILVER.py

# Gold layer feature engineering
python databricks_pipeline/Gold_layer/04_FEATURE_ENGINEERING.py

# Train models
python databricks_pipeline/ML_Modeling_layer/02_Baseline_Models.py
python databricks_pipeline/ML_Modeling_layer/03_Train_Regression_Models.py
python databricks_pipeline/ML_Modeling_layer/04_Train_Classification_Models.py
```

### Launching the Dashboard

```
# Activate the virtual environment first (see Installation)
cd dashboard
streamlit run app.py
```

The dashboard opens in your default browser at `http://localhost:8501`.

## 🔄 Data Flow

1. **Data Ingestion (Bronze)** — Raw clickstream data ingested from S3 or local CSV, schema validation, stored as partitioned Parquet files
2. **Data Processing (Silver)** — Deduplication, anomaly detection, type standardization, missing value handling and outlier capping
3. **Feature Engineering (Gold)** — Aggregation by ad/campaign/time period, CTR/CVR/RPM/engagement metrics, temporal features, encoding and normalization
4. **Model Training & Evaluation** — Baseline and advanced models, hyperparameter tuning, cross-validation, SHAP explainability
5. **Dashboard Consumption** — Loads processed data and trained models, interactive visualizations, real-time predictions and recommendations

## 🤖 Model Information

### Prediction Tasks

**Click-Through Rate (CTR) Prediction**
- Binary classification: Will the user click on the ad?
- Evaluation: AUC-ROC, Precision@K, Log Loss

**Conversion Value Prediction**
- Regression: Predict expected conversion value
- Evaluation: RMSE, MAE, R² Score

### Model Registry
- All models tracked using MLflow
- Model versions stored with performance metrics
- Feature importance and SHAP values for interpretability

### Currently Supported Models
- Logistic Regression (baseline for CTR)
- Random Forest Classifier / Regressor
- XGBoost Classifier / Regressor
- Neural Networks (MLP)
- Linear Regression (baseline for conversion)

## 📊 Dashboard Walkthrough

The dashboard uses a **single unified workspace** — no separate role selection. It opens directly into three tabs:

### Overview
- Combined financial and campaign KPIs (ROAS, total spend, revenue, CTR/CPC indicators)
- Budget pacing and utilization tracking
- Profitability analysis by campaign and campaign type

### Analytics
A compact selector switches between two views:
- **Strategic Predictions** — ML-powered forecasts, advanced model factor sliders, scenario planning
- **Performance Insights** — Top-performing ads and campaigns, creative performance analysis, audience segmentation insights

### Budget Optimizer
- ROAS-weighted budget reallocation
- Marginal ROI analysis
- Optimal spend distribution recommendations

### Sidebar
- Dataset export (CSV / JSON) of the currently filtered data

## 📦 Dependencies

- `streamlit>=1.28.0`: Interactive web application framework
- `pandas>=2.0.0`: Data manipulation and analysis
- `numpy>=1.24.0`: Numerical computing
- `altair>=5.0.0`: Declarative statistical visualization
- `plotly>=5.14.0`: Interactive plotting library
- `s3fs>=2023.6.0`: S3 filesystem interface for Python
- `mlflow>=2.5.0`: Machine learning lifecycle management
- `databricks-sdk>=0.12.0`: Databricks platform SDK
- `python-dotenv>=1.0.0`: Environment variable management

##  Contributing

We welcome contributions to improve the Ad-Tech Optimizer!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## Acknowledgments

- Synthetic data generation based on realistic advertising patterns
- MLflow for experiment tracking and model management
- Streamlit community for innovative UI components
- Open-source ML libraries (scikit-learn, XGBoost) for modeling capabilities
