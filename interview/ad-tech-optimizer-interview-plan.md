# Ad-Tech Optimizer Project - Interview Preparation Plan

## Context
This plan documents a comprehensive understanding of the Ad-Tech Optimizer project, covering the end-to-end data pipeline, ML model training, Streamlit dashboard architecture, data quality mechanisms, and key architectural decisions. This knowledge will enable tackling any interview question about the project.

## End-to-End Data Pipeline (Bronze -> Silver -> Gold)

### 1. Bronze Layer (Raw Ingestion)
- **Files**: `01_LOAD_TO_BRONZE.py`
- **Purpose**: Ingest raw synthetic data from CSV files into Delta tables
- **Key Features**:
  - Idempotency checks using `DESCRIBE` table commands
  - Metadata tracking: ingestion timestamp, batch ID, environment, git commit
  - Partitioning by `ingestion_date` for performance
  - Version tracking: `batch_log` (OVERWRITE - latest only) and `version_history` (APPEND - all runs)
  - Monitoring tables for anomaly detection and data quality tracking

### 2. Silver Layer (Data Cleaning)
- **Files**: `03_CLEAN_TO_SILVER.py`
- **Purpose**: Clean all anomalies from bronze data and create standardized silver tables
- **Key Cleaning Operations**:
  - Timestamp format fixing (handles both `yyyy-MM-dd HH:mm:ss` and `dd/MM/yyyy HH:mm`)
  - Device corruption fix (extracts clean device from corrupted values like "47 DESKTOP")
  - Ad Type standardization (fixes typos like "Vedeo"→"Video", "Carusel"→"Carousel")
  - Watch Duration cleaning (negative values → 0, >180 → 180 seconds)
  - User Age cleaning (<18 → 25, >100 → 35, nulls → 30)
  - Platform source typo fixes ("gogle"→"google", "facebok"→"facebook")
  - User Gender identity shifting resolution (uses most common gender per user)
  - Logical conflict resolution (clicked=1 with watch_duration=0 → clicked=0)
  - Deduplication on business keys (User_ID, Click_Timestamp, Ad_Reference_ID)
- **Output**: Standardized columns with `_cleaned` or `_standard` suffixes

### 3. Gold Layer (Feature Engineering)
- **Files**: `04_FEATURE_ENGINEERING.py`
- **Purpose**: Create business metrics and ML features from silver data
- **Key Features Created** (60+ total):
  - **Temporal Features** (5): hour_of_day, day_of_week, month, is_weekend, time_of_day_category
  - **Session Features** (2): time_since_last_action, session_id (30-min inactivity threshold)
  - **User Engagement Features** (3): user_total_clicks, user_total_impressions, user_ctr
  - **Ad Performance Features** (4): ad_total_clicks, ad_total_impressions, ad_ctr, ad_avg_watch_duration
  - **Watch Duration Features** (2): watch_duration_ratio, watch_duration_ratio_clipped
  - **Demographic Features** (3): age_category, category_performance_by_age, demographic_engagement_delta (DED Score)
  - **Financial Metrics** (4): total_revenue, total_cost, return_on_ad_spend (ROAS), profit_margin
  - **Platform Features** (3): platform_avg_roas, platform_total_spend, platform_total_revenue
  - **Ad-Level Aggregates** (25+): CTR, ROAS, conversion_rate, etc.
  - **Derived Features** (11): engagement_efficiency, profit_margin, cost_per_conversion, high_performance (ROAS>2.0), cost_efficiency_score, engagement_score, audience_alignment_score, ad_age_days, ad_lifecycle_stage, location_type, season
  - **Date Dimension**: processing_date for partitioning

### 4. Data Quality Enforcement & Verification
- **Preventive**: Schema definitions in YAML files with required fields, data types, valid ranges/values
- **Validation**: YAML-based data quality rules with severity levels (critical/high/medium/low) and actions (alert/fail/fix)
- **Correction**: Silver layer scripts that fix identified anomalies
- **Verification**: SQL scripts (`00_VERIFY_SETUP.sql` to `03_VERIFY_GOLD.sql`) that validate:
  - Row counts match expectations
  - Schema compliance
  - Anomaly detection effectiveness
  - Monitoring table verification
  - Version history tracking
- **Monitoring**: 
  - `adtech_catalog.monitoring.anomaly_report` - tracks all detected anomalies
  - `adtech_catalog.monitoring.batch_log` - latest run metadata (OVERWRITE with mergeSchema)
  - `adtech_catalog.monitoring.version_history` - all pipeline runs (APPEND)

### 5. ML Pipeline Configuration
- **Files**: `pipeline_manifest.yaml` (ml_pipeline section)
- **Configuration**:
  - Random seed: 42
  - Test split ratio: 0.2
  - Cross-validation folds: 5
  - Use cases:
    - CTR Prediction (regression)
    - ROAS Prediction (regression)
    - Conversion Rate Prediction (regression)
    - High Performance Classification (binary, ROAS>2.0)
  - Hyperparameter tuning: enabled, max 20 trials, random search

### 6. ML Model Training Process
- **Files**: ML_Modeling_layer/*.py
- **Process**:
  1. **EDA and Data Preparation** (`01_EDA_and_Data_Preparation.py`):
     - Loads gold data
     - Creates stratified train/test split guaranteeing both classes in training for classification
     - Saves split indices as CSV files for reproducibility
     - Performs correlation analysis and categorical feature analysis
  2. **Baseline Models** (`02_Baseline_Models.py`):
     - Trains simple Linear and Logistic Regression models
     - Establishes performance baseline
  3. **Advanced Regression Models** (`03_Train_Regression_Models.py`):
     - Targets: CTR, ROAS, Conversion Rate, DED Score, Cost Efficiency Score
     - Models: RandomForestRegressor, GradientBoostingRegressor, XGBoost, LightGBM, Voting Ensemble
     - Features: 8 pre-launch features (cost_per_click, ad_video_length, ad_category, ad_device, ad_type, ad_location, avg_ded_score, category_age_affinity) + interaction features
     - Evaluation: RMSE, MAE, R² metrics
     - Cross-validation for robustness
  4. **Classification Models** (`04_Train_Classification_Models.py`):
     - Target: High Performance (ROAS > 2.0)
     - Handles class imbalance with SMOTE when ratio > 3:1
     - Models: LogisticRegression, RandomForestClassifier, XGBoost, LightGBM, VotingClassifier
     - Feature selection using RandomForest importance
     - Hyperparameter tuning with RandomizedSearchCV
     - Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC
  5. **Model Explainability** (`05_Model_Explainability.py`):
     - Uses SHAP (SHapley Additive exPlanations) for model interpretation
     - Explains top 3 models: LightGBM (High Performance), LightGBM (Conversion Rate), RandomForest (Cost Efficiency)
     - Generates: summary plots, dependence plots, force plots, waterfall plots
     - Saves SHAP values and consolidated summary for stakeholder presentation
  6. **Model Comparison and Selection** (`06_Model_Comparison_and_Selection.py`):
     - Compares all models and selects best performer for each target
     - Uses R² for regression, AUC for classification
     - Creates final selection table for deployment decisions

### 7. Streamlit Dashboard Architecture
- **Main File**: `ad-click-project/ad-click-project/ad-tech-optimizer-dashboard/app.py`
- **Key Components**:
  - **Workspace Selection**: Business Owner vs Agency & Campaign views with different KPIs and navigation
  - **Data Source Toggle**: Live Production Data (S3) vs Demo Synthetic Data (Local)
  - **Global Filters**: Category and Device filters that persist across pages
  - **Tabbed Navigation**: 5 pages per workspace using st.radio with custom CSS
  - **Custom Premium Styling**: Plus Jakarta Sans font, metric cards with hover effects, responsive design
  - **Cursor Fixes**: Ensures proper cursor states (pointer for interactive elements, default for static content)
  - **Export Functionality**: CSV/JSON export of filtered dataset
  - **Workspace Switching**: Ability to change workspaces without losing state

- **Page Modules**:
  - `_01_dashboard_home.py`: KPIs, charts, performance overview
  - `_02_predictions.py`: ML model prediction interface with form inputs and dynamic recommendations
  - `_03_performance_insights.py`: Detailed performance analysis
  - `_04_recommendations.py`: Budget optimization and ad format recommendations
  - `_05_ai_copilot.py`: LLM-powered insights using Ollama Llama 3.2

- **Utilities**:
  - `utils/data_loader.py`: Loads Gold data from S3 with caching, applies 10x attribution correction for iOS 14+ tracking loss
  - `utils/model_loader.py`: Loads ML models from MLflow Registry (Unity Catalog), prepares features, applies dynamic factors for realistic predictions
  - `llm/copilot.py`: Ollama-powered AI copilot for natural language insights
  - `llm/llm.py`: Utility functions for LLM integration

### 8. Configuration Management
- **YAML Files**:
  - `pipeline_manifest.yaml`: Complete pipeline configuration (data sources, table definitions, data quality rules, feature engineering, ML pipeline, deployment, monitoring, Git/CI/CD)
  - `data_quality_rules.yaml`: Detailed validation rules for bronze and silver layers with expected anomaly counts
  - `llm_semantic_mapping.yaml`: Semantic data dictionary for LLM/AI understanding with business meaning, LLM context, and examples

### 9. Key Architectural Decisions & Trade-offs
- **Medallion Architecture**: Bronze (raw) → Silver (cleaned) → Gold (aggregated/features) for clear separation of concerns
- **Idempotent Operations**: All layer scripts can be rerun safely with overwrite modes where appropriate
- **Version Tracking**: 
  - `batch_log`: OVERWRITE with mergeSchema (keeps only latest run)
  - `version_history`: APPEND (maintains full history of all pipeline runs)
- **Monitoring-First Approach**: Anomaly detection and logging built into each layer
- **Feature Store Pattern**: Gold layer serves as feature store for ML models with 60+ business-relevant features
- **Pre-Launch Features**: Only 8 features available before ad launch (cost_per_click, ad_video_length, ad_category, ad_device, ad_type, ad_location, avg_ded_score, category_age_affinity) for realistic prediction scenarios
- **Dynamic Factors in Dashboard**: Applies calibrated adjustments (±18% max) to raw model outputs so predictions vary visibly across inputs without hitting display caps
- **Attribution Correction**: 10x multiplier on S3 revenue to compensate for iOS 14+ tracking loss (identified during EDA)
- **Class Balance Guarantee**: Ensures both classes appear in training data for classification models through stratified sampling with fallback manual adjustment
- **Schema Evolution Handling**: Uses mergeSchema and overwriteSchema options to handle schema changes gracefully
- **Resource Management**: 
  - Spark session reuse across notebooks
  - Caching in Streamlit (@st.cache_data, @st.cache_resource) for performance
  - External model loading via MLflow to avoid bundling large models with dashboard

### 10. Deployment & Scheduling
- **Cron Schedule** (from pipeline_manifest.yaml):
  - Data ingestion: 0 2 * * * (2:00 AM daily)
  - Data cleansing: 0 3 * * * (3:00 AM daily)
  - Feature engineering: 0 4 * * * (4:00 AM daily)
  - ML training: 0 5 * * 0 (5:00 AM weekly on Sunday)
  - Model deployment: 0 6 * * 1 (6:00 AM weekly on Monday)
- **Deployment Target**: Databricks Unity Catalog with MLflow model registry
- **Model Staging**: Uses @Production stage for promoted models
- **Environment Separation**: development/staging/production configurations

### 11. Verification & Quality Assurance
- **Automated Verification**: SQL scripts at each layer validate:
  - Row counts within expected ranges
  - Zero tolerance for critical anomalies after cleaning
  - Schema compliance
  - Referential integrity between tables
- **Anomaly Reporting**: Detailed tracking of:
  - What anomalies were detected
  - How many of each type
  - Which pipeline run detected them
  - Trend analysis over time
- **Testing Approach**:
  - Unit tests implied in verification scripts
  - Integration tested through full pipeline execution
  - Model performance tracked via monitoring tables
  - Data drift detection for model retraining triggers

### 12. Technologies & Tools
- **Data Processing**: Apache Spark (Delta Lake) on Databricks
- **Orchestration**: Custom Python scripts with Databricks notebooks
- **ML Framework**: scikit-learn, XGBoost, LightGBM with MLflow tracking
- **Dashboard**: Streamlit with custom CSS and JavaScript enhancements
- **Data Storage**: 
  - Bronze/Silver/Gold: Delta Lake in Unity Catalog
  - Models: MLflow Model Registry
  - Logs/Metrics: Monitoring tables in Unity Catalog
  - Raw Data: CSV files in DBFS / Volumes
  - Sample Data: Local CSV fallback
- **Configuration**: YAML files for pipeline manifest, data quality rules, semantic mapping
- **Environment Management**: Python-dotenv for environment variables
- **Version Tracking**: Git commit hashes, custom versioning scheme (YYYYMMDD_HHMMSS)

### 13. Business Metrics & KPIs
- **Primary Financial Metric**: ROAS (Return on Ad Spend) > 2.0 = profitable
- **Engagement Metrics**: 
  - CTR (Click-Through Rate): target 2-5%
  - Conversion Rate: target 5-15%
  - Engagement Score: weighted combination of signals
  - DED Score: Demographic Engagement Delta (0-1 range)
- **Efficiency Metrics**:
  - Cost Per Click (CPC): lower is better
  - Cost Per Conversion: spend / conversions
  - Cost Efficiency Score: (CTR × Conversion Rate) / CPC
  - Profit Margin: (Revenue - Cost) / Cost
- **Diagnostic Metrics**:
  - Ad Lifecycle Stage: New (<7 days), Growing (<30 days), Mature (<60 days), Declining (>60 days)
  - Location Type: Urban/Semi-Urban/Rural
  - Season: Winter/Spring/Summer/Fall
  - Platform Performance: Google/Facebook/Instagram/Other
  - Audience Alignment: category_age_affinity × avg_ded_score

### 14. Potential Interview Questions & Answers

#### Data Pipeline Questions
**Q: How does the pipeline handle data quality issues?**
A: The pipeline uses a multi-layered approach:
1. Preventive: Schema definitions in YAML with required fields, types, ranges
2. Detective: YAML-based anomaly detection rules with expected counts
3. Corrective: Silver layer scripts that fix identified issues (timestamp formats, device corruption, etc.)
4. Verification: SQL scripts that validate zero critical anomalies remain after cleaning
5. Monitoring: Anomaly_report table tracks all detected issues over time

**Q: Explain the version tracking strategy.**
A: Two tables serve different purposes:
- `batch_log`: Uses OVERWRITE with mergeSchema to keep only the latest run's metadata (for operational dashboards)
- `version_history`: Uses APPEND to maintain immutable history of all pipeline runs (for audit/compliance)
Both capture: version_id (timestamp), environment, git commit, batch counts, status

**Q: How do you ensure reproducible train/test splits?**
A: The EDA script uses stratified sampling with guaranteed class presence:
1. Attempts stratified splits with increasing random seeds until both classes appear in training
2. If unsuccessful after 20 attempts, applies manual fix by moving one minority class sample from test to train
3. Saves split indices as CSV files for exact reproducibility across model training runs
4. Used consistently across all ML models to ensure fair comparison

#### ML Questions
**Q: How do you handle class imbalance in the high performance classification model?**
A: Three-tiered approach:
1. First, use algorithm-level solutions: class_weight='balanced' in Logistic Regression, scale_pos_weight in XGBoost
2. Second, apply SMOTE oversampling when imbalance ratio > 3:1 (synthetic minority over-sampling)
3. Third, use proper evaluation metrics: AUC-ROC, F1-score, Precision-Recall instead of accuracy alone
4. Validate with cross-validation to ensure generalization

**Q: What features are available before an ad launch for prediction?**
A: Exactly 8 pre-launch features that don't depend on historical performance:
1. cost_per_click (bid amount)
2. ad_video_length (creative attribute)
3. ad_category (industry vertical)
4. ad_device (target device)
5. ad_type (creative format)
6. ad_location (geographic target)
7. avg_ded_score (audience alignment score from similar ads)
8. category_age_affinity (historical performance of this age-category combination)
These enable cold-start predictions for new campaigns.

**Q: How do you interpret SHAP values for model explainability?**
A: SHAP (SHapley Additive exPlanations) shows:
- Feature importance: mean absolute SHAP value across all predictions
- Direction and magnitude: SHAP value indicates how much a feature increases/decreases the prediction
- Interaction effects: dependence plots show how feature effects change based on other feature values
- Individual predictions: force/waterfall plots show baseline prediction plus cumulative feature contributions
In our models, top features for high performance were typically: ad_category, cost_per_click, ad_video_length, showing that placement, bidding, and creative format drive profitability.

#### Dashboard Questions
**Q: How does the dashboard ensure models are loaded efficiently?**
A: Three-layer optimization:
1. MLflow Model Registry: Centralized model storage and versioning
2. Streamlit Caching: @st.cache_resource for model loading (persists across reruns)
3. Lazy Loading: Models loaded only when first needed, not on app startup
4. Fallback Mechanism: Dynamic heuristic predictions when MLflow unreachable
5. Signature Adjustment: Automatic feature matrix resizing to match model expectations

**Q: How are predictions made actionable for business users?**
A: The predictions page provides:
1. Clear recommendation status based on ROAS thresholds (≥2.95x = approve, ≥2.75x = approve with caution, <2.75x = hold)
2. Specific optimization tips based on input values (video length, CPC, CTR, conversion rate)
3. Side-by-side comparison of all three key metrics (CTR, Conversion Rate, ROAS)
4. Business context explanation of what each metric means
5. Recommendation text that explains why the decision was made

**Q: What global filtering capabilities exist in the dashboard?**
A: Two persistent filters:
1. Category Filter: Single or multiple selection from 6 categories (Electronics, Fashion, Travel, Health, Gaming, Food)
2. Device Filter: Single selection from 4 options (All-Devices, Mobile, Desktop, Tablet)
These filters:
- Are set in the Campaign Analytics page (Agency workspace)
- Automatically inherited by Predictions, Performance Insights, Recommendations, and AI Copilot pages
- Display in an Active Context Banner showing current selection
- Affect all data visualizations and model predictions throughout the dashboard

## Verification of Understanding
This plan covers all major components explored:
- Bronze layer ingestion with metadata tracking
- Silver layer cleaning with comprehensive anomaly fixes  
- Gold layer feature engineering (60+ features)
- ML model training (regression & classification)
- Model explainability with SHAP
- Model comparison and selection
- Streamlit dashboard architecture and navigation
- Data quality enforcement and verification
- Configuration management via YAML files
- Deployment scheduling and monitoring
- Business metrics and KPI definitions

This comprehensive understanding enables confident discussion of any aspect of the Ad-Tech Optimizer project in an interview setting.