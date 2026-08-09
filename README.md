# Ad-Tech Optimizer Dashboard

An end-to-end advertising technology optimization platform that processes ad performance data, trains machine learning models for click-through rate (CTR) and conversion rate prediction, and provides interactive dashboards for business owners and advertising agencies to optimize ad spend and campaign performance.

## �� 📋 Table of Contents
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

## �� 🎯 Overview

The Ad-Tech Optimizer is a comprehensive solution designed to help advertisers and agencies maximize their return on ad spend (ROAS) by:
- Processing raw advertising performance data through a multi-layer ETL pipeline
- Training and evaluating machine learning models to predict ad performance
- Providing interactive, role-based dashboards with actionable insights
- Enabling data-driven decisions for budget allocation, creative optimization, and campaign strategy

## �� 🏗��️ Architecture

The system follows a modern data lakehouse architecture with three distinct layers:

```
Raw Data → [Bronze Layer] → [Silver Layer] → [Gold Layer] → [ML Models] → [Streamlit Dashboard]
```

### Layer Breakdown:
- **Bronze Layer**: Raw data ingestion and initial validation
- **Silver Layer**: Data cleaning, deduplication, and standardization  
- **Gold Layer**: Feature engineering, aggregation, and preparation for ML
- **ML Modeling Layer**: Model training, evaluation, and explainability
- **Presentation Layer**: Streamlit-based interactive dashboards

## �� ⭐ Key Features

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
- **Role-Based Views**: Separate dashboards for Business Owners and Agency/Campaign managers
- **Interactive Filters**: Global filters for ad category, device type, and date ranges
- **Performance Analytics**: Real-time KPI tracking and trend analysis
- **Predictive Insights**: ML-powered forecasting and recommendations
- **Budget Optimization**: ROAS-weighted budget allocation models
- **Creative Recommendations**: AI-powered ad format and targeting suggestions
- **Data Export**: CSV/JSON export of filtered datasets
- **AI Copilot**: Natural language interface for data exploration

## �� 📁 Project Structure

```
final_cdac_project/
├── README.md
├── requirements.txt
├── manifest.mf
│
├── Synthetic_Data/                    # Data generation and raw datasets
│   ├── ad_catalog_raw.csv             # Ad catalog reference data
│   ├── raw_synthetic_ad_click_data.csv # Raw clickstream data
│   ├── data_quality.py                # Data validation utilities
│   └── *.py                           # Data generation scripts
│
├── Ad-Tech Optimizer/                 # Production data pipeline
│   ├── Bronze_layer/                  # Raw data ingestion (01_LOAD_TO_BRONZE.py, etc.)
│   ├── Silver_layer/                  # Data cleaning and transformation
│   ├── Gold_layer/                    # Feature engineering and modeling prep
│   ├── ML_Modeling_layer/             # Model training and evaluation
│   └── Verification/                  # SQL verification scripts for each layer
│
├── ad_click_project/                  # Jupyter notebook versions
│   └── ad_click_project/              # Notebooks mirroring the Python pipeline
│
�└── ad_click_project/ad_click_project/ad-tech-optimizer-dashboard/ # Streamlit Dashboard
    ├── app.py                         # Main application entry point
    ├── requirements.txt               # Dashboard-specific dependencies
    ├── sample_ad_performance.csv      # Sample data for demo
    ├── .env                           # Environment variables
    ├── mlflow.db                      # MLflow tracking database
    │
    ├── pages/                         # Dashboard pages
    │   ├── _01_dashboard_home.py
    │   ├── _02_predictions.py
    │   ├── _03_performance_insights.py
    │   ├── _04_recommendations.py
    │   └── _05_ai_copilot.py
    │
    └── utils/                         # Helper modules
        ├── data_loader.py
        ├── model_loader.py
        └── test_mlflow.py
```

## �� 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Git
- AWS account (for S3 integration) or local demo data
- Databricks workspace (optional, for production deployment)

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd final_cdac_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r ad-click-project/ad-click-project/ad-tech-optimizer-dashboard/requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp ad-click-project/ad-click-project/ad-tech-optimizer-dashboard/.env.example \
      ad-click-project/ad-click-project/ad-tech-optimizer-dashboard/.env
   # Edit .env with your AWS credentials and S3 bucket information
   ```

5. **Initialize MLflow tracking**
   ```bash
   mlflow ui --backend-store-uri ./ad-click-project/ad-click-project/ad-tech-optimizer-dashboard/mlflow.db
   ```

### Data Setup Options

**Option 1: Use Demo Data (Recommended for initial exploration)**
- The system automatically falls back to local synthetic data if S3 credentials aren't configured
- Sample data is included in `Synthetic_Data/raw_synthetic_ad_click_data.csv`

**Option 2: Connect to Production S3**
1. Update `.env` with your AWS credentials:
   ```env
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=your-region
   S3_BUCKET=your-ad-tech-bucket
   ```
2. Ensure your S3 bucket contains properly formatted parquet/csv files matching the expected schema

## � ▶��️ Usage

### Running the Data Pipeline (Optional)
To process data through the ETL layers:
```bash
# Run Bronze layer ingestion
python Ad-Tech\ Optimizer/Bronze_layer/01_LOAD_TO_BRONZE.py

# Run Silver layer cleaning
python Ad-Tech\ Optimizer/Silver_layer/03_CLEAN_TO_SILVER.py

# Run Gold layer feature engineering
python Ad-Tech\ Optimizer/Gold_layer/04_FEATURE_ENGINEERING.py

# Train models
python Ad-Tech\ Optimizer/ML_Modeling_layer/02_Baseline_Models.py
python Ad-Tech\ Optimizer/ML_Modeling_layer/03_Train_Regression_Models.py
python Ad-Tech\ Optimizer/ML_Modeling_layer/04_Train_Classification_Models.py
```

### Launching the Dashboard
```bash
# Activate virtual environment if not already activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Navigate to dashboard directory
cd ad-click-project/ad-click-project/ad-tech-optimizer-dashboard

# Run Streamlit app
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## �� 🔄 Data Flow

1. **Data Ingestion (Bronze)**
   - Raw clickstream data ingested from S3 or local CSV
   - Initial schema validation and partitioning
   - Stored as partitioned Parquet files

2. **Data Processing (Silver)**
   - Deduplication and anomaly detection
   - Data type standardization and enrichment
   - Missing value handling and outlier capping
   - Cleaned data prepared for feature engineering

3. **Feature Engineering (Gold)**
   - Aggregation by ad, campaign, time periods
   - Feature creation: CTR, CVR, RPM, engagement metrics
   - Temporal features (hour/day/week patterns)
   - Categorical encoding and normalization
   - Final feature store for ML models

4. **Model Training & Evaluation**
   - Baseline models (Logistic Regression, Linear Regression)
   - Advanced models (Random Forest, XGBoost, Neural Networks)
   - Hyperparameter tuning and cross-validation
   - Model performance comparison and selection
   - SHAP explainability analysis

5. **Dashboard Consumption**
   - Loads processed Gold layer data and trained models
   - Provides interactive visualizations and filters
   - Generates real-time predictions and recommendations
   - Enables data export and scenario planning

## �� 🤖 Model Information

### Prediction Tasks
1. **Click-Through Rate (CTR) Prediction**
   - Binary classification: Will user click on ad?
   - Features: Ad properties, user context, historical performance
   - Evaluation: AUC-ROC, Precision@K, Log Loss

2. **Conversion Value Prediction**
   - Regression: Predict expected conversion value
   - Features: Click features, landing page properties, offer details
   - Evaluation: RMSE, MAE, R² Score

### Model Registry
- All models tracked using MLflow
- Model versions stored with performance metrics
- A/B testing framework for champion/challenger models
- Feature importance and SHAP values for interpretability

### Currently Supported Models
- Logistic Regression (baseline for CTR)
- Random Forest Classifier
- XGBoost Classifier
- Neural Networks (MLP)
- Linear Regression (baseline for conversion)
- Random Forest Regressor
- XGBoost Regressor

## �� 📊 Dashboard Walkthrough

### Landing Page
- Workspace selection: Business Owner or Agency & Campaign
- Clear descriptions of each workspace's focus areas

### Business Owner Workspace
1. **Financial Overview**
   - ROAS trends, total spend, revenue metrics
   - Budget pacing and utilization tracking
   - Profitability analysis by campaign/campaign type

2. **Strategic Predictions**
   - Forecasted performance for upcoming periods
   - Budget allocation recommendations
   - Scenario planning for spend changes

3. **Performance Insights**
   - Top-performing ads and campaigns
   - Creative performance analysis
   - Audience segmentation insights

4. **Budget Optimizer**
   - ROAS-weighted budget reallocation
   - Marginal ROI analysis
   - Optimal spend distribution recommendations

5. **AI Analytics Copilot**
   - Natural language querying of performance data
   - Automated insight generation
   - Custom report creation assistant

### Agency & Campaign Workspace
1. **Campaign Analytics**
   - Granular CTR/CPC/CPM metrics
   - Device and geographic performance breakdown
   - Ad format and placement analysis

2. **ML Performance Forecaster**
   - Individual ad performance predictions
   - Confidence intervals for forecasts
   - Under/over-performing asset identification

3. **Granular Performance Insights**
   - Hourly/daypart performance patterns
   - Creative element analysis (images, copy, CTAs)
   - Audience response characteristics

4. **Creative Recommender**
   - A/B test suggestions for creative elements
   - Best-performing template recommendations
   - Fatigue detection and refresh timing

5. **AI Analytics Copilot**
   - Same natural language capabilities as Business version
   - Agency-specific query examples and templates

## �� 📦 Dependencies

### Core Dependencies
- `streamlit>=1.28.0`: Interactive web application framework
- `pandas>=2.0.0`: Data manipulation and analysis
- `numpy>=1.24.0`: Numerical computing
- `altair>=5.0.0`: Declarative statistical visualization
- `plotly>=5.14.0`: Interactive plotting library
- `st-files-connection>=0.1.0`: Streamlit file connection component
- `s3fs>=2023.6.0`: S3 filesystem interface for Python
- `mlflow>=2.5.0`: Machine learning lifecycle management
- `databricks-sdk>=0.12.0`: Databricks platform SDK
- `python-dotenv>=1.0.0`: Environment variable management

### Development Dependencies
- Jupyter notebooks for exploratory analysis
- Various data validation and testing utilities

## �� 🤝 Contributing

We welcome contributions to improve the Ad-Tech Optimizer! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Contribution Areas
- New ML model implementations
- Additional dashboard visualizations
- Data source connectors (other cloud providers, APIs)
- Performance optimizations
- Documentation improvements
- Bug fixes

Please ensure your code follows existing styles and includes appropriate tests.

## �� 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## �� 🙏 Acknowledgments

- Synthetic data generation based on realistic advertising patterns
- MLflow for experiment tracking and model management
- Streamlit community for innovative UI components
- Open-source ML libraries (scikit-learn, XGBoost) for modeling capabilities

## �� 📞 Support

For questions, issues, or feature requests:
1. Check the [Issue Tracker](https://github.com/your-username/ad-tech-optimizer/issues)
2. Contact the maintainers through GitHub Discussions
3. Refer to the inline code documentation and comments

---

*Built with �� ❤��️ for data-driven advertising optimization*