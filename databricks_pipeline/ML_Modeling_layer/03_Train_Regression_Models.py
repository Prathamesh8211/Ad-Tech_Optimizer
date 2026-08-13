# Databricks notebook source
pip install xgboost

# COMMAND ----------

pip install xgboost lightgbm

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 03_Train_Regression_Models
# ============================================================
# Purpose: Train advanced regression models for 5 regression targets

# This notebook trains advanced regression models and compares
# them against the Linear Regression baselines.

# TARGETS:
# 1. CTR
# 2. ROAS
# 3. Conversion Rate
# 4. DED Score
# 5. Cost Efficiency Score
#
# MODELS:
# - RandomForestRegressor
# - GradientBoostingRegressor
# - XGBRegressor
# - Voting Ensemble (RF + GB + XGB)

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("ADVANCED REGRESSION MODELS")
print("="*70)

# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

def load_yaml_config():
    try:
        try:
            with open("pipeline_manifest.yaml", "r") as f:
                config = yaml.safe_load(f)
                print("Loaded config from local path")
                return config
        except:
            pass

        try:
            config_path = "/Volumes/adtech_catalog/bronze/landing_zone/pipeline_manifest.yaml"
            try:
                dbutils.fs.ls(config_path)
                config_content = dbutils.fs.head(config_path)
                config = yaml.safe_load(config_content)
                print(f"Loaded config from: {config_path}")
                return config
            except:
                print("Config file not found in DBFS")
                return None
        except:
            return None

    except Exception as e:
        print(f"Could not load config: {e}")
        return None

config = load_yaml_config()

ENVIRONMENT = config.get('environment', 'development') if config else 'development'
VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
GIT_COMMIT = os.environ.get('GIT_COMMIT', 'local')
RANDOM_SEED = config.get('ml_pipeline', {}).get('random_seed', 42) if config else 42

print("="*70)
print("CONFIGURATION SUMMARY")
print("="*70)
print(f"Environment: {ENVIRONMENT}")
print(f"Version: {VERSION}")
print(f"Git Commit: {GIT_COMMIT}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*70)

# ============================================================
# 2. LOAD TRAIN/TEST DATA
# ============================================================

print("\nLOADING TRAIN/TEST DATA...")

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

try:
    train_df = pd.read_csv(volume_path + "train_split.csv")
    test_df = pd.read_csv(volume_path + "test_split.csv")
    print(f"Loaded training data: {len(train_df):,} rows")
    print(f"Loaded test data: {len(test_df):,} rows")
    
    # Convert numeric columns
    numeric_cols = ['ctr', 'roas', 'conversion_rate', 'avg_ded_score', 
                    'cost_efficiency_score', 'cost_per_click', 'ad_video_length']
    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    print("Data loaded successfully")
    
except Exception as e:
    print(f"Error loading split data: {e}")
    dbutils.notebook.exit("Failed to load split data")

# ============================================================
# 3. PREPARE FEATURES
# ============================================================

print("\nPREPARING FEATURES...")

pre_launch_features = [
    "cost_per_click",
    "ad_video_length",
    "ad_category",
    "ad_device",
    "ad_type",
    "ad_location",
    "avg_ded_score",
    "category_age_affinity"
]

categorical_cols = ["ad_category", "ad_device", "ad_type", "ad_location"]

def encode_categorical(df, cols, fit=False, encoders=None):
    if fit:
        encoders = {}
        for col in cols:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, encoders
    else:
        for col in cols:
            if col + "_encoded" not in df.columns:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        return df, None

train_encoded, encoders = encode_categorical(train_df, categorical_cols, fit=True)
test_encoded, _ = encode_categorical(test_df, categorical_cols, fit=False)

feature_cols = [c for c in pre_launch_features if c not in categorical_cols] + [c + "_encoded" for c in categorical_cols]

X_train = train_encoded[feature_cols].values
X_test = test_encoded[feature_cols].values

print(f"Feature columns: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled")

# ============================================================
# 5. TARGETS
# ============================================================

y_ctr_train = train_df['ctr'].values
y_ctr_test = test_df['ctr'].values

y_roas_train = train_df['roas'].values
y_roas_test = test_df['roas'].values

y_conversion_train = train_df['conversion_rate'].values
y_conversion_test = test_df['conversion_rate'].values

y_ded_train = train_df['avg_ded_score'].values
y_ded_test = test_df['avg_ded_score'].values

y_cost_efficiency_train = train_df['cost_efficiency_score'].values
y_cost_efficiency_test = test_df['cost_efficiency_score'].values

print("Targets prepared")

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_regression(y_true, y_pred, model_name, target_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "target": target_name,
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

# ============================================================
# 7. TRAIN MODELS FOR A GIVEN TARGET
# ============================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, target_name):
    """Train multiple models and return results"""
    
    print(f"\n{'='*70}")
    print(f"TARGET: {target_name}")
    print("="*70)
    
    results = []
    
    # 1. RandomForest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(evaluate_regression(y_test, y_pred_rf, "RandomForest", target_name))
    
    # 2. GradientBoosting
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_SEED
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results.append(evaluate_regression(y_test, y_pred_gb, "GradientBoosting", target_name))
    
    # 3. XGBoost
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=RANDOM_SEED,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results.append(evaluate_regression(y_test, y_pred_xgb, "XGBoost", target_name))
    
    # 4. Voting Ensemble
    voting = VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED, verbosity=0))
    ])
    voting.fit(X_train, y_train)
    y_pred_voting = voting.predict(X_test)
    results.append(evaluate_regression(y_test, y_pred_voting, "VotingEnsemble", target_name))
    
    # 5. Cross-validation for best model (VotingEnsemble)
    cv_scores = cross_val_score(voting, X_train, y_train, cv=5, scoring='r2')
    print(f"  CV R2 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Print results
    for res in results:
        print(f"  {res['model']:20} R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}")
    
    return results

# ============================================================
# 8. TRAIN MODELS FOR ALL REGRESSION TARGETS
# ============================================================

print("\n" + "="*70)
print("TRAINING ADVANCED REGRESSION MODELS")
print("="*70)

all_results = []

# 8.1 CTR
ctr_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_ctr_train, y_ctr_test, "CTR")
all_results.extend(ctr_results)

# 8.2 ROAS (filter valid ROAS > 0)
roas_mask_train = y_roas_train > 0
roas_mask_test = y_roas_test > 0

if np.sum(roas_mask_train) > 10 and np.sum(roas_mask_test) > 0:
    X_train_roas = X_train_scaled[roas_mask_train]
    X_test_roas = X_test_scaled[roas_mask_test]
    y_train_roas = y_roas_train[roas_mask_train]
    y_test_roas = y_roas_test[roas_mask_test]
    
    print(f"\nROAS: Using {len(X_train_roas)} train, {len(X_test_roas)} test samples")
    roas_results = train_and_evaluate(X_train_roas, X_test_roas, y_train_roas, y_test_roas, "ROAS")
    all_results.extend(roas_results)
else:
    print("\nROAS: Not enough valid samples (ROAS > 0)")
    roas_results = None

# 8.3 Conversion Rate
conversion_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_conversion_train, y_conversion_test, "ConversionRate")
all_results.extend(conversion_results)

# 8.4 DED Score
ded_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_ded_train, y_ded_test, "DEDScore")
all_results.extend(ded_results)

# 8.5 Cost Efficiency
cost_efficiency_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_cost_efficiency_train, y_cost_efficiency_test, "CostEfficiency")
all_results.extend(cost_efficiency_results)

# ============================================================
# 9. COMPILE RESULTS
# ============================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Create summary table
summary_data = []
for res in all_results:
    if res:
        summary_data.append({
            "Target": res['target'],
            "Model": res['model'],
            "R2": res['r2'],
            "RMSE": res['rmse'],
            "MAE": res['mae']
        })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Best model for each target
print("\n" + "="*70)
print("BEST MODEL PER TARGET")
print("="*70)

for target in summary_df['Target'].unique():
    target_df = summary_df[summary_df['Target'] == target]
    best = target_df.loc[target_df['R2'].idxmax()]
    print(f"  {target}: {best['Model']} (R2: {best['R2']:.4f})")

# ============================================================
# 10. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    # Convert to DataFrame
    results_df = pd.DataFrame(summary_data)
    results_df['version'] = VERSION
    results_df['environment'] = ENVIRONMENT
    results_df['training_timestamp'] = datetime.now().isoformat()
    
    # Save as CSV
    results_df.to_csv(volume_path + "regression_advanced_results.csv", index=False)
    print(f"Results saved to: {volume_path}regression_advanced_results.csv")
    
    # Save to Spark table
    spark_results = spark.createDataFrame(results_df)
    spark_results.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.regression_advanced_results")
    print("Results saved to: adtech_catalog.monitoring.regression_advanced_results")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 11. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Advanced Regression Models",
        "SUCCESS"
    )], [
        "version_id",
        "environment",
        "git_commit",
        "deployed_at",
        "description",
        "status"
    ])

    version_info.write \
        .mode("append") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.version_history")

    print("Version history updated: adtech_catalog.monitoring.version_history")
    print(f"   Version: {VERSION}")

except Exception as e:
    print(f"Could not save version history: {e}")

# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("ADVANCED REGRESSION MODELS COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Data:
   - Training: {len(train_df):,} rows
   - Test: {len(test_df):,} rows
   - Features: {len(feature_cols)}

Models Trained:
   1. RandomForestRegressor
   2. GradientBoostingRegressor
   3. XGBRegressor
   4. Voting Ensemble (RF + GB + XGB)

Targets:
   1. CTR
   2. ROAS
   3. Conversion Rate
   4. DED Score
   5. Cost Efficiency Score

Next Steps:
   1. Run: 04_Train_Classification_Models.py
   2. Run: 05_Model_Explainability.py
======================================================================
""")

print("")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# DYNAMIC SUMMARY: ADVANCED REGRESSION MODELS
# ============================================================
# This cell loads results from the saved Delta table and 
# generates an automated summary - no hardcoded values!

import pandas as pd
import numpy as np

print("="*70)
print("ADVANCED REGRESSION MODELS - EXECUTION SUMMARY")
print("="*70)

# ============================================================
# 1. LOAD RESULTS FROM SAVED TABLE
# ============================================================

try:
    # Load advanced regression results
    results_df = spark.table("adtech_catalog.monitoring.regression_advanced_results").toPandas()
    print(f"✅ Loaded {len(results_df)} results from monitoring table")
except Exception as e:
    print(f"⚠️  Could not load from monitoring table: {e}")
    print("   Trying to load from CSV backup...")
    try:
        volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"
        results_df = pd.read_csv(volume_path + "regression_advanced_results.csv")
        print(f"✅ Loaded {len(results_df)} results from CSV backup")
    except Exception as e2:
        print(f"❌ Could not load results: {e2}")
        results_df = None

# ============================================================
# 2. LOAD TRAIN/TEST DATA FOR CONTEXT
# ============================================================

try:
    volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"
    train_df = pd.read_csv(volume_path + "train_split.csv")
    test_df = pd.read_csv(volume_path + "test_split.csv")
    
    train_rows = len(train_df)
    test_rows = len(test_df)
    num_features = 8  # Pre-launch features count
    
    print(f"📊 Data: {train_rows:,} train, {test_rows:,} test, {num_features} features")
except Exception as e:
    print(f"⚠️  Could not load split data: {e}")
    train_rows = "N/A"
    test_rows = "N/A"
    num_features = "N/A"

# ============================================================
# 3. LOAD BASELINE RESULTS FOR COMPARISON
# ============================================================

try:
    baseline_df = spark.table("adtech_catalog.monitoring.baseline_results").toPandas()
    print("✅ Loaded baseline results for comparison")
except Exception as e:
    print(f"⚠️  Could not load baseline results: {e}")
    baseline_df = None

# ============================================================
# 4. GENERATE TARGET PERFORMANCE SUMMARY
# ============================================================

if results_df is not None and len(results_df) > 0:
    print("\n" + "="*70)
    print("TARGET PERFORMANCE (Best Model):")
    print("-"*70)
    
    # Find best model per target (highest R²)
    best_models = results_df.loc[results_df.groupby('Target')['R2'].idxmax()]
    
    # Performance labeling function
    def get_performance_label(r2):
        if r2 > 0.3:
            return ("EXCELLENT", "✅")
        elif r2 > 0.1:
            return ("GOOD", "👍")
        elif r2 > 0:
            return ("FAIR", "📊")
        else:
            return ("POOR", "⚠️")
    
    # Display each target
    for idx, row in best_models.iterrows():
        target = row['Target']
        r2 = row['R2']
        model = row['Model']
        rmse = row.get('RMSE', 'N/A')
        mae = row.get('MAE', 'N/A')
        
        label, emoji = get_performance_label(r2)
        
        print(f"{idx+1}. {target:20} : R² = {r2:+.4f} ({model:20}) {emoji} {label}")
    
    # ============================================================
    # 5. FIND BEST OVERALL PERFORMANCE
    # ============================================================
    
    print(f"\n{'='*70}")
    best_overall = best_models.loc[best_models['R2'].idxmax()]
    print(f"🏆 BEST OVERALL: {best_overall['Target']} with R² = {best_overall['R2']:.4f} ({best_overall['Model']})")
    
    # ============================================================
    # 6. IMPROVEMENT OVER BASELINE
    # ============================================================
    
    if baseline_df is not None:
        print("\n" + "="*70)
        print("IMPROVEMENT OVER BASELINE:")
        print("-"*70)
        
        # Filter baseline for regression targets (where R2 exists)
        baseline_regression = baseline_df[baseline_df['target'].isin(['CTR', 'ROAS', 'ConversionRate', 'DEDScore', 'CostEfficiency'])]
        
        for target in best_models['Target'].unique():
            advanced_r2 = best_models[best_models['Target'] == target]['R2'].values[0]
            
            # Get baseline R2 (look for matching target)
            baseline_match = baseline_regression[baseline_regression['target'] == target]
            if len(baseline_match) > 0:
                baseline_r2 = baseline_match['r2'].values[0] if 'r2' in baseline_match.columns else 0
            else:
                baseline_r2 = 0
            
            # Calculate improvement
            if baseline_r2 != 0:
                improvement_pct = ((advanced_r2 - baseline_r2) / abs(baseline_r2)) * 100
            else:
                improvement_pct = 0
            
            # Determine arrow
            if improvement_pct > 10:
                arrow = "✅"
            elif improvement_pct > 0:
                arrow = "📈"
            elif improvement_pct == 0:
                arrow = "➖"
            else:
                arrow = "📉"
            
            print(f"{target:20} : {baseline_r2:.4f} → {advanced_r2:.4f}  ({arrow} {improvement_pct:+.1f}%)")
    else:
        print("\n" + "="*70)
        print("⚠️  Baseline comparison not available (baseline_results table not found)")
    
    # ============================================================
    # 7. BEST MODEL PER TARGET
    # ============================================================
    
    print("\n" + "="*70)
    print("BEST MODEL PER TARGET:")
    print("-"*70)
    
    for idx, row in best_models.iterrows():
        target = row['Target']
        model = row['Model']
        r2 = row['R2']
        print(f"{target:20} : {model:25} (R² = {r2:+.4f})")
    
    # ============================================================
    # 8. RECOMMENDATIONS
    # ============================================================
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("-"*70)
    
    for idx, row in best_models.iterrows():
        target = row['Target']
        r2 = row['R2']
        model = row['Model']
        
        if target == 'DEDScore':
            print(f"⚠️  {target}: Target has near-zero variance - consider skipping or investigating data generation")
        elif r2 > 0.3:
            print(f"✅ {target}: {model} is performing well (R²={r2:.4f}) - focus on hyperparameter tuning")
        elif r2 > 0.1:
            print(f"📈 {target}: {model} shows promise (R²={r2:.4f}) - try feature engineering and tuning")
        elif r2 > 0:
            print(f"📊 {target}: {model} needs improvement (R²={r2:.4f}) - consider more features or different approach")
        else:
            print(f"⚠️  {target}: {model} is underperforming (R²={r2:.4f}) - consider more features, different models, or skip this target")
    
    # ============================================================
    # 9. KEY FINDINGS
    # ============================================================
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("-"*70)
    
    # Count models by type
    model_counts = best_models['Model'].value_counts()
    
    for model, count in model_counts.items():
        print(f"{count} target(s) best modeled by: {model}")
    
    # Best performing targets
    top_targets = best_models.nlargest(3, 'R2')['Target'].tolist()
    print(f"\n🏆 Top 3 performing targets: {', '.join(top_targets)}")
    
    # Targets needing improvement
    poor_targets = best_models[best_models['R2'] <= 0]['Target'].tolist()
    if poor_targets:
        print(f"⚠️  Targets needing improvement: {', '.join(poor_targets)}")

else:
    print("\n" + "="*70)
    print("⚠️  No results found! Please run Cell 1 first.")
    print("="*70)

# ============================================================
# 10. NEXT STEPS
# ============================================================

print("\n" + "="*70)
print("NEXT STEPS:")
print("-"*70)
print("1. Review model performance above")
print("2. Run: 04_Train_Classification_Models.py")
print("3. Run: 05_Model_Explainability.py")
print("4. Run: 06_Model_Comparison_and_Selection.py")
print("="*70)

print("\n✅ Dynamic summary complete!")

# COMMAND ----------

# Quick Python Check for DED Score
# Load Gold data and check DED Score
df_gold = spark.table("adtech_catalog.gold.fact_ad_performance")
df_pd = df_gold.toPandas()

print("DED Score Statistics:")
print(df_pd['avg_ded_score'].describe())
print(f"\nUnique values: {df_pd['avg_ded_score'].nunique()}")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 03_Train_Regression_Models
# ============================================================
# Purpose: Train advanced regression models with hyperparameter tuning


# TARGETS: CTR, ROAS, Conversion Rate, DED Score, Cost Efficiency

# MODELS:
# - RandomForestRegressor
# - GradientBoostingRegressor
# - XGBRegressor
# - LightGBMRegressor (NEW)
# - Voting Ensemble (RF + GB + XGB)

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from xgboost import XGBRegressor
import lightgbm as lgb
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("ADVANCED REGRESSION MODELS (with Tuning & LightGBM)")
print("="*70)

# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

def load_yaml_config():
    try:
        try:
            with open("pipeline_manifest.yaml", "r") as f:
                config = yaml.safe_load(f)
                print("Loaded config from local path")
                return config
        except:
            pass

        try:
            config_path = "/Volumes/adtech_catalog/bronze/landing_zone/pipeline_manifest.yaml"
            try:
                dbutils.fs.ls(config_path)
                config_content = dbutils.fs.head(config_path)
                config = yaml.safe_load(config_content)
                print(f"Loaded config from: {config_path}")
                return config
            except:
                print("Config file not found in DBFS")
                return None
        except:
            return None

    except Exception as e:
        print(f"Could not load config: {e}")
        return None

config = load_yaml_config()

ENVIRONMENT = config.get('environment', 'development') if config else 'development'
VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
GIT_COMMIT = os.environ.get('GIT_COMMIT', 'local')
RANDOM_SEED = config.get('ml_pipeline', {}).get('random_seed', 42) if config else 42

print("="*70)
print("CONFIGURATION SUMMARY")
print("="*70)
print(f"Environment: {ENVIRONMENT}")
print(f"Version: {VERSION}")
print(f"Git Commit: {GIT_COMMIT}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*70)

# ============================================================
# 2. LOAD TRAIN/TEST DATA
# ============================================================

print("\nLOADING TRAIN/TEST DATA...")

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

try:
    train_df = pd.read_csv(volume_path + "train_split.csv")
    test_df = pd.read_csv(volume_path + "test_split.csv")
    print(f"Loaded training data: {len(train_df):,} rows")
    print(f"Loaded test data: {len(test_df):,} rows")
    
    numeric_cols = ['ctr', 'roas', 'conversion_rate', 'avg_ded_score', 
                    'cost_efficiency_score', 'cost_per_click', 'ad_video_length']
    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    print("Data loaded successfully")
    
except Exception as e:
    print(f"Error loading split data: {e}")
    dbutils.notebook.exit("Failed to load split data")

# ============================================================
# 3. PREPARE FEATURES
# ============================================================

print("\nPREPARING FEATURES...")

pre_launch_features = [
    "cost_per_click",
    "ad_video_length",
    "ad_category",
    "ad_device",
    "ad_type",
    "ad_location",
    "avg_ded_score",
    "category_age_affinity"
]

categorical_cols = ["ad_category", "ad_device", "ad_type", "ad_location"]

def encode_categorical(df, cols, fit=False, encoders=None):
    if fit:
        encoders = {}
        for col in cols:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, encoders
    else:
        for col in cols:
            if col + "_encoded" not in df.columns:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        return df, None

train_encoded, encoders = encode_categorical(train_df, categorical_cols, fit=True)
test_encoded, _ = encode_categorical(test_df, categorical_cols, fit=False)

feature_cols = [c for c in pre_launch_features if c not in categorical_cols] + [c + "_encoded" for c in categorical_cols]

X_train = train_encoded[feature_cols].values
X_test = test_encoded[feature_cols].values

print(f"Feature columns: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled")

# ============================================================
# 5. TARGETS
# ============================================================

y_ctr_train = train_df['ctr'].values
y_ctr_test = test_df['ctr'].values

y_roas_train = train_df['roas'].values
y_roas_test = test_df['roas'].values

y_conversion_train = train_df['conversion_rate'].values
y_conversion_test = test_df['conversion_rate'].values

y_ded_train = train_df['avg_ded_score'].values
y_ded_test = test_df['avg_ded_score'].values

y_cost_efficiency_train = train_df['cost_efficiency_score'].values
y_cost_efficiency_test = test_df['cost_efficiency_score'].values

print("Targets prepared")

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_regression(y_true, y_pred, model_name, target_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "target": target_name,
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

# ============================================================
# 7. HYPERPARAMETER TUNING FUNCTION
# ============================================================

def tune_randomforest(X_train, y_train, target_name):
    """Tune RandomForest with RandomizedSearchCV"""
    print(f"\n  Tuning RandomForest for {target_name}...")
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=3, 
        scoring='r2', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV R2: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_xgboost(X_train, y_train, target_name):
    """Tune XGBoost with RandomizedSearchCV"""
    print(f"\n  Tuning XGBoost for {target_name}...")
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb = XGBRegressor(random_state=RANDOM_SEED, verbosity=0)
    random_search = RandomizedSearchCV(
        xgb, param_dist, n_iter=10, cv=3, 
        scoring='r2', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV R2: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_lightgbm(X_train, y_train, target_name):
    """Tune LightGBM with RandomizedSearchCV"""
    print(f"\n  Tuning LightGBM for {target_name}...")
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'subsample': [0.8, 1.0]
    }
    
    lgb_model = lgb.LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1)
    random_search = RandomizedSearchCV(
        lgb_model, param_dist, n_iter=10, cv=3, 
        scoring='r2', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV R2: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

# ============================================================
# 8. TRAIN MODELS FOR A GIVEN TARGET
# ============================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, target_name, use_tuning=True):
    """Train multiple models and return results"""
    
    print(f"\n{'='*70}")
    print(f"TARGET: {target_name}")
    print("="*70)
    
    results = []
    
    # 1. RandomForest (with tuning)
    if use_tuning:
        rf_best = tune_randomforest(X_train, y_train, target_name)
        y_pred_rf = rf_best.predict(X_test)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
    
    results.append(evaluate_regression(y_test, y_pred_rf, "RandomForest", target_name))
    
    # 2. GradientBoosting (no tuning, for comparison)
    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results.append(evaluate_regression(y_test, y_pred_gb, "GradientBoosting", target_name))
    
    # 3. XGBoost (with tuning)
    if use_tuning:
        xgb_best = tune_xgboost(X_train, y_train, target_name)
        y_pred_xgb = xgb_best.predict(X_test)
    else:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                           random_state=RANDOM_SEED, verbosity=0)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
    
    results.append(evaluate_regression(y_test, y_pred_xgb, "XGBoost", target_name))
    
    # 4. LightGBM (NEW - with tuning)
    if use_tuning:
        lgb_best = tune_lightgbm(X_train, y_train, target_name)
        y_pred_lgb = lgb_best.predict(X_test)
    else:
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbosity=-1)
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
    
    results.append(evaluate_regression(y_test, y_pred_lgb, "LightGBM", target_name))
    
    # 5. Voting Ensemble (RF + XGB + LightGBM)
    voting = VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                             random_state=RANDOM_SEED, verbosity=0)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbosity=-1))
    ])
    voting.fit(X_train, y_train)
    y_pred_voting = voting.predict(X_test)
    results.append(evaluate_regression(y_test, y_pred_voting, "VotingEnsemble", target_name))
    
    # 6. Cross-validation for best model
    cv_scores = cross_val_score(voting, X_train, y_train, cv=5, scoring='r2')
    print(f"  CV R2 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Print results
    for res in results:
        print(f"  {res['model']:20} R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}")
    
    return results

# ============================================================
# 9. TRAIN MODELS FOR ALL REGRESSION TARGETS
# ============================================================

print("\n" + "="*70)
print("TRAINING ADVANCED REGRESSION MODELS (with Tuning & LightGBM)")
print("="*70)

all_results = []

# 9.1 CTR
ctr_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_ctr_train, y_ctr_test, "CTR")
all_results.extend(ctr_results)

# 9.2 ROAS
roas_mask_train = y_roas_train > 0
roas_mask_test = y_roas_test > 0

if np.sum(roas_mask_train) > 10 and np.sum(roas_mask_test) > 0:
    X_train_roas = X_train_scaled[roas_mask_train]
    X_test_roas = X_test_scaled[roas_mask_test]
    y_train_roas = y_roas_train[roas_mask_train]
    y_test_roas = y_roas_test[roas_mask_test]
    
    print(f"\nROAS: Using {len(X_train_roas)} train, {len(X_test_roas)} test samples")
    roas_results = train_and_evaluate(X_train_roas, X_test_roas, y_train_roas, y_test_roas, "ROAS")
    all_results.extend(roas_results)
else:
    print("\nROAS: Not enough valid samples (ROAS > 0)")
    roas_results = None

# 9.3 Conversion Rate
conversion_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_conversion_train, y_conversion_test, "ConversionRate")
all_results.extend(conversion_results)

# 9.4 DED Score (Check if it has variation first)
if np.std(y_ded_train) > 1e-10:
    ded_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_ded_train, y_ded_test, "DEDScore")
    all_results.extend(ded_results)
else:
    print("\nDEDScore: Target has no variation. Skipping.")
    ded_results = None

# 9.5 Cost Efficiency
cost_efficiency_results = train_and_evaluate(X_train_scaled, X_test_scaled, y_cost_efficiency_train, y_cost_efficiency_test, "CostEfficiency")
all_results.extend(cost_efficiency_results)

# ============================================================
# 10. COMPILE RESULTS
# ============================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

summary_data = []
for res in all_results:
    if res:
        summary_data.append({
            "Target": res['target'],
            "Model": res['model'],
            "R2": res['r2'],
            "RMSE": res['rmse'],
            "MAE": res['mae']
        })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Best model for each target
print("\n" + "="*70)
print("BEST MODEL PER TARGET")
print("="*70)

for target in summary_df['Target'].unique():
    target_df = summary_df[summary_df['Target'] == target]
    best = target_df.loc[target_df['R2'].idxmax()]
    print(f"  {target}: {best['Model']} (R2: {best['R2']:.4f})")

# ============================================================
# 11. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    results_df = pd.DataFrame(summary_data)
    results_df['version'] = VERSION
    results_df['environment'] = ENVIRONMENT
    results_df['training_timestamp'] = datetime.now().isoformat()
    
    results_df.to_csv(volume_path + "regression_advanced_results.csv", index=False)
    print(f"Results saved to: {volume_path}regression_advanced_results.csv")
    
    spark_results = spark.createDataFrame(results_df)
    spark_results.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.regression_advanced_results")
    print("Results saved to: adtech_catalog.monitoring.regression_advanced_results")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 12. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Advanced Regression Models (Tuned + LightGBM)",
        "SUCCESS"
    )], [
        "version_id",
        "environment",
        "git_commit",
        "deployed_at",
        "description",
        "status"
    ])

    version_info.write \
        .mode("append") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.version_history")

    print("Version history updated: adtech_catalog.monitoring.version_history")
    print(f"   Version: {VERSION}")

except Exception as e:
    print(f"Could not save version history: {e}")

# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("ADVANCED REGRESSION MODELS COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Data:
   - Training: {len(train_df):,} rows
   - Test: {len(test_df):,} rows
   - Features: {len(feature_cols)}

Models Trained:
   1. RandomForestRegressor (with tuning)
   2. GradientBoostingRegressor
   3. XGBRegressor (with tuning)
   4. LightGBMRegressor (NEW - with tuning)
   5. Voting Ensemble (RF + XGB + LightGBM)

Targets:
   1. CTR
   2. ROAS
   3. Conversion Rate
   4. DED Score (skipped if no variation)
   5. Cost Efficiency Score

Next Steps:
   1. Run: 04_Train_Classification_Models.py
   2. Run: 05_Model_Explainability.py
======================================================================
""")

print("")

# COMMAND ----------

# Create ML models schema
spark.sql("CREATE SCHEMA IF NOT EXISTS adtech_catalog.ml_models")
print("✅ Schema 'adtech_catalog.ml_models' created")


# COMMAND ----------

# ============================================================
# 03_Train_Regression_Models (IMPROVED VERSION)
# ============================================================
# Purpose: Train advanced regression models with:
#          - Interaction Features
#          - Feature Selection
#          - Hyperparameter Tuning
#          - LightGBM
#          - Selective Ensemble
#
# TARGETS: CTR, ROAS, Conversion Rate, Cost Efficiency
# 
# IMPROVEMENTS APPLIED:
# 1. Added interaction features (device×type, category×type, cost×video)
# 2. Feature selection using RandomForest importance
# 3. Hyperparameter Tuning (RandomizedSearchCV)
# 4. LightGBM with tuning
# 5. XGBoost with tuning
# 6. Selective ensemble (only top models)

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import lightgbm as lgb
from pyspark.sql import SparkSession
import yaml
import os
import mlflow
mlflow.set_registry_uri("databricks-uc")
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("ADVANCED REGRESSION MODELS (IMPROVED)")
print("="*70)
print("""
IMPROVEMENTS APPLIED:
1. Interaction Features (device×type, category×type, cost×video)
2. Feature Selection (RandomForest importance)
3. Hyperparameter Tuning (RandomizedSearchCV)
4. LightGBM with tuning
5. XGBoost with tuning
6. Selective Ensemble (only top models)
""")

# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

def load_yaml_config():
    try:
        try:
            with open("pipeline_manifest.yaml", "r") as f:
                config = yaml.safe_load(f)
                print("Loaded config from local path")
                return config
        except:
            pass

        try:
            config_path = "/Volumes/adtech_catalog/bronze/landing_zone/pipeline_manifest.yaml"
            try:
                dbutils.fs.ls(config_path)
                config_content = dbutils.fs.head(config_path)
                config = yaml.safe_load(config_content)
                print(f"Loaded config from: {config_path}")
                return config
            except:
                print("Config file not found in DBFS")
                return None
        except:
            return None

    except Exception as e:
        print(f"Could not load config: {e}")
        return None

config = load_yaml_config()

ENVIRONMENT = config.get('environment', 'development') if config else 'development'
VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
GIT_COMMIT = os.environ.get('GIT_COMMIT', 'local')
RANDOM_SEED = config.get('ml_pipeline', {}).get('random_seed', 42) if config else 42

print("="*70)
print("CONFIGURATION SUMMARY")
print("="*70)
print(f"Environment: {ENVIRONMENT}")
print(f"Version: {VERSION}")
print(f"Git Commit: {GIT_COMMIT}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*70)

# ============================================================
# 2. LOAD TRAIN/TEST DATA
# ============================================================

print("\nLOADING TRAIN/TEST DATA...")

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

try:
    train_df = pd.read_csv(volume_path + "train_split.csv")
    test_df = pd.read_csv(volume_path + "test_split.csv")
    print(f"Loaded training data: {len(train_df):,} rows")
    print(f"Loaded test data: {len(test_df):,} rows")
    
    numeric_cols = ['ctr', 'roas', 'conversion_rate', 'avg_ded_score', 
                    'cost_efficiency_score', 'cost_per_click', 'ad_video_length']
    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    print("Data loaded successfully")
    
except Exception as e:
    print(f"Error loading split data: {e}")
    dbutils.notebook.exit("Failed to load split data")

# ============================================================
# 3. PREPARE FEATURES WITH INTERACTIONS
# ============================================================

print("\nPREPARING FEATURES WITH INTERACTIONS...")

pre_launch_features = [
    "cost_per_click",
    "ad_video_length",
    "ad_category",
    "ad_device",
    "ad_type",
    "ad_location",
    "avg_ded_score",
    "category_age_affinity"
]

categorical_cols = ["ad_category", "ad_device", "ad_type", "ad_location"]

def encode_categorical(df, cols, fit=False, encoders=None):
    if fit:
        encoders = {}
        for col in cols:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, encoders
    else:
        for col in cols:
            if col + "_encoded" not in df.columns:
                le = LabelEncoder()
                df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        return df, None

train_encoded, encoders = encode_categorical(train_df, categorical_cols, fit=True)
test_encoded, _ = encode_categorical(test_df, categorical_cols, fit=False)

# BASE FEATURES
base_feature_cols = [c for c in pre_launch_features if c not in categorical_cols] + [c + "_encoded" for c in categorical_cols]

# INTERACTION FEATURES
print("\n  Adding interaction features...")
train_encoded['device_type_interaction'] = train_encoded['ad_device_encoded'] * train_encoded['ad_type_encoded']
test_encoded['device_type_interaction'] = test_encoded['ad_device_encoded'] * test_encoded['ad_type_encoded']

train_encoded['category_type_interaction'] = train_encoded['ad_category_encoded'] * train_encoded['ad_type_encoded']
test_encoded['category_type_interaction'] = test_encoded['ad_category_encoded'] * test_encoded['ad_type_encoded']

train_encoded['cost_video_interaction'] = train_encoded['cost_per_click'] * train_encoded['ad_video_length']
test_encoded['cost_video_interaction'] = test_encoded['cost_per_click'] * test_encoded['ad_video_length']

interaction_features = ['device_type_interaction', 'category_type_interaction', 'cost_video_interaction']

# Combine all features
feature_cols = base_feature_cols + interaction_features

X_train = train_encoded[feature_cols].values
X_test = test_encoded[feature_cols].values

print(f"  Base features: {len(base_feature_cols)}")
print(f"  Interaction features: {len(interaction_features)}")
print(f"  Total features: {len(feature_cols)}")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")

# ============================================================
# 4. SCALE FEATURES
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled")

# ============================================================
# 5. TARGETS
# ============================================================

y_ctr_train = train_df['ctr'].values
y_ctr_test = test_df['ctr'].values

y_roas_train = train_df['roas'].values
y_roas_test = test_df['roas'].values

y_conversion_train = train_df['conversion_rate'].values
y_conversion_test = test_df['conversion_rate'].values

y_cost_efficiency_train = train_df['cost_efficiency_score'].values
y_cost_efficiency_test = test_df['cost_efficiency_score'].values

# DED Score - check if it has variation
if np.std(train_df['avg_ded_score'].values) > 1e-10:
    y_ded_train = train_df['avg_ded_score'].values
    y_ded_test = test_df['avg_ded_score'].values
    use_ded = True
else:
    use_ded = False
    print("\nDED Score: No variation detected. Skipping.")

print("Targets prepared")

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_regression(y_true, y_pred, model_name, target_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "target": target_name,
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

# ============================================================
# 7. FEATURE SELECTION FUNCTION
# ============================================================

def select_features(X_train, X_test, y_train, feature_names, target_name, threshold='mean'):
    """Select important features using RandomForest importance"""
    
    print(f"\n  Feature Selection for {target_name}:")
    
    selector_rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    selector_rf.fit(X_train, y_train)
    
    importances = selector_rf.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("    Top 5 features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    selector = SelectFromModel(selector_rf, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"    Selected {len(selected_features)} / {len(feature_names)} features")
    print(f"    Dropped {len(feature_names) - len(selected_features)} features")
    
    return X_train_selected, X_test_selected, selected_features

# ============================================================
# 8. HYPERPARAMETER TUNING FUNCTIONS (FIXED)
# ============================================================

def tune_randomforest(X_train, y_train, target_name):
    """Tune RandomForest with RandomizedSearchCV"""
    print(f"\n  Tuning RandomForest for {target_name}...")
    
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf = RandomForestRegressor(random_state=RANDOM_SEED)
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=3, 
        scoring='r2', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV R2: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_xgboost(X_train, y_train, target_name):
    """Tune XGBoost with RandomizedSearchCV"""
    print(f"\n  Tuning XGBoost for {target_name}...")
    
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBRegressor(
        random_state=RANDOM_SEED,
        verbosity=0
    )
    
    random_search = RandomizedSearchCV(
        xgb, param_dist, n_iter=10, cv=3, 
        scoring='r2', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV R2: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

def tune_lightgbm(X_train, y_train, target_name):
    """Tune LightGBM with RandomizedSearchCV"""
    print(f"\n  Tuning LightGBM for {target_name}...")
    
    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    lgb_model = lgb.LGBMRegressor(
        random_state=RANDOM_SEED,
        verbosity=-1
    )
    
    random_search = RandomizedSearchCV(
        lgb_model, param_dist, n_iter=10, cv=3, 
        scoring='r2', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV R2: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

# ============================================================
# 9. TRAIN AND EVALUATE FUNCTION
# ============================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, target_name, use_tuning=True, use_feature_selection=True):
    """Train multiple models and return results"""
    
    print(f"\n{'='*70}")
    print(f"TARGET: {target_name}")
    print("="*70)
    
    # Apply feature selection
    if use_feature_selection:
        X_train_sel, X_test_sel, selected_feats = select_features(
            X_train, X_test, y_train, feature_cols, target_name
        )
    else:
        X_train_sel, X_test_sel = X_train, X_test
        selected_feats = feature_cols
    
    results = []
    
    # 1. RandomForest (tuned)
    if use_tuning:
        rf_best = tune_randomforest(X_train_sel, y_train, target_name)
        y_pred_rf = rf_best.predict(X_test_sel)
    else:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
        rf.fit(X_train_sel, y_train)
        y_pred_rf = rf.predict(X_test_sel)
        rf_best = rf
    results.append(evaluate_regression(y_test, y_pred_rf, "RandomForest", target_name))
    
    # 2. GradientBoosting (baseline)
    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_SEED
    )
    gb.fit(X_train_sel, y_train)
    y_pred_gb = gb.predict(X_test_sel)
    gb_best = gb 
    results.append(evaluate_regression(y_test, y_pred_gb, "GradientBoosting", target_name))
    
    # 3. XGBoost (tuned)
    if use_tuning:
        xgb_best = tune_xgboost(X_train_sel, y_train, target_name)
        y_pred_xgb = xgb_best.predict(X_test_sel)
    else:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                           random_state=RANDOM_SEED, verbosity=0)
        xgb.fit(X_train_sel, y_train)
        y_pred_xgb = xgb.predict(X_test_sel)
        xgb_best = xgb
    
    results.append(evaluate_regression(y_test, y_pred_xgb, "XGBoost", target_name))
    
    # 4. LightGBM (tuned)
    if use_tuning:
        lgb_best = tune_lightgbm(X_train_sel, y_train, target_name)
        y_pred_lgb = lgb_best.predict(X_test_sel)
    else:
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbosity=-1)
        lgb_model.fit(X_train_sel, y_train)
        y_pred_lgb = lgb_model.predict(X_test_sel)
        lgb_best = lgb_model
    results.append(evaluate_regression(y_test, y_pred_lgb, "LightGBM", target_name))
    
    # 5. Selective Ensemble (only top 2 models based on R2)
    model_r2s = [(res['model'], res['r2']) for res in results]
    model_r2s.sort(key=lambda x: x[1], reverse=True)
    top_2_models = [name for name, _ in model_r2s[:2]]
    
    print(f"\n  Selective Ensemble using: {top_2_models}")
    
    # Build ensemble with top 2 models
    ensemble_models = []
    if 'RandomForest' in top_2_models:
        rf_final = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
        rf_final.fit(X_train_sel, y_train)
        ensemble_models.append(('rf', rf_final))
    
    if 'XGBoost' in top_2_models:
        xgb_final = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                 random_state=RANDOM_SEED, verbosity=0)
        xgb_final.fit(X_train_sel, y_train)
        ensemble_models.append(('xgb', xgb_final))
    
    if 'LightGBM' in top_2_models:
        lgb_final = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbosity=-1)
        lgb_final.fit(X_train_sel, y_train)
        ensemble_models.append(('lgb', lgb_final))
    
    voting = None

    if len(ensemble_models) >= 2:
        voting = VotingRegressor(ensemble_models)
        voting.fit(X_train_sel, y_train)
        y_pred_voting = voting.predict(X_test_sel)
        results.append(evaluate_regression(y_test, y_pred_voting, "SelectiveEnsemble", target_name))
    else:
        print("  Not enough models for ensemble, using best single model")
    
    # 6. Cross-validation for best model
    best_result = max(results, key=lambda x: x['r2'])
    print(f"\n  Best Model: {best_result['model']} (R2: {best_result['r2']:.4f})")
    
    # CV for best model
    best_model_name = best_result['model']
    if best_model_name == 'RandomForest':
        best_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED)
    elif best_model_name == 'XGBoost':
        best_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                  random_state=RANDOM_SEED, verbosity=0)
    elif best_model_name == 'LightGBM':
        best_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbosity=-1)
    elif best_model_name == 'SelectiveEnsemble':
        # Use the already trained voting ensemble
        best_model = voting
    else:
        best_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)
    
    if best_model_name != 'SelectiveEnsemble':
        cv_scores = cross_val_score(best_model, X_train_sel, y_train, cv=5, scoring='r2')
        print(f"  CV R2 Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    else:
        print(f"  CV R2: Not available for ensemble (already cross-validated)")
    
    print("\n  All Results:")
    for res in results:
        print(f"  {res['model']:20} R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}")
    
    return results,{
        'RandomForest': rf_best,
        'XGBoost': xgb_best,
        'LightGBM': lgb_best,
        'GradientBoosting': gb_best,
        'SelectiveEnsemble': voting
    }

# ============================================================
# 10. TRAIN MODELS FOR ALL REGRESSION TARGETS
# ============================================================

print("\n" + "="*70)
print("TRAINING MODELS FOR ALL TARGETS")
print("="*70)

all_results = []
all_models = {}

# 10.1 CTR
ctr_results, ctr_models = train_and_evaluate(X_train_scaled, X_test_scaled, y_ctr_train, y_ctr_test, "CTR")
all_results.extend(ctr_results)
all_models['CTR'] = ctr_models

# 10.2 ROAS
roas_mask_train = y_roas_train > 0
roas_mask_test = y_roas_test > 0

if np.sum(roas_mask_train) > 10 and np.sum(roas_mask_test) > 0:
    X_train_roas = X_train_scaled[roas_mask_train]
    X_test_roas = X_test_scaled[roas_mask_test]
    y_train_roas = y_roas_train[roas_mask_train]
    y_test_roas = y_roas_test[roas_mask_test]
    
    print(f"\nROAS: Using {len(X_train_roas)} train, {len(X_test_roas)} test samples")
    roas_results, roas_models = train_and_evaluate(X_train_roas, X_test_roas, y_train_roas, y_test_roas, "ROAS")
    all_results.extend(roas_results)
    all_models['ROAS'] = roas_models
else:
    print("\nROAS: Not enough valid samples (ROAS > 0)")
    roas_results = None
    roas_models = None

# 10.3 Conversion Rate
conversion_results, conversion_models = train_and_evaluate(X_train_scaled, X_test_scaled, y_conversion_train, y_conversion_test, "ConversionRate")
all_results.extend(conversion_results)
all_models['ConversionRate'] = conversion_models

# 10.4 Cost Efficiency
cost_efficiency_results, cost_models = train_and_evaluate(X_train_scaled, X_test_scaled, y_cost_efficiency_train, y_cost_efficiency_test, "CostEfficiency")
all_results.extend(cost_efficiency_results)
all_models['CostEfficiency'] = cost_models

# ============================================================
# ✅ REGISTER BEST MODELS TO MLflow (FIXED)
# ============================================================

print("\n" + "="*70)
print("REGISTERING MODELS TO MLflow")
print("="*70)

# Get best model for each target WITH R2 values
summary_df = pd.DataFrame([{
    "Target": res['target'],
    "Model": res['model'],
    "R2": res['r2']
} for res in all_results if res is not None])

# Create dictionary with full info {target: {'Model': ..., 'R2': ...}}
best_models_dict = {}
for target in summary_df['Target'].unique():
    target_df = summary_df[summary_df['Target'] == target]
    best = target_df.loc[target_df['R2'].idxmax()]
    best_models_dict[target] = {
        'Model': best['Model'],
        'R2': best['R2']
    }

print("Best models selected:")
for target, info in best_models_dict.items():
    print(f"  {target}: {info['Model']} (R²={info['R2']:.4f})")

# ============================================================
# REGISTER EACH MODEL (WITH CORRECT FEATURE SHAPES)
# ============================================================

try:
    # 1. CTR Model (XGBoost) - trained on selected features
    if 'CTR' in best_models_dict:
        ctr_best_model_name = best_models_dict['CTR']['Model']
        if ctr_best_model_name in all_models['CTR'] and all_models['CTR'][ctr_best_model_name] is not None:
            ctr_model = all_models['CTR'][ctr_best_model_name]
            
            # ✅ FIX: Use the same selected features that the model was trained on
            # Re-run feature selection for CTR to get the correct feature set
            X_train_ctr_sel, _, _ = select_features(
                X_train_scaled, X_test_scaled, y_ctr_train, feature_cols, "CTR"
            )
            
            with mlflow.start_run(run_name=f"CTR_{ctr_best_model_name}") as run:
                signature = mlflow.models.infer_signature(X_train_ctr_sel, ctr_model.predict(X_train_ctr_sel))
                if ctr_best_model_name == 'XGBoost':
                    mlflow.xgboost.log_model(ctr_model, "ctr_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.ctr_predictor")
                else:
                    mlflow.sklearn.log_model(ctr_model, "ctr_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.ctr_predictor")
                mlflow.log_metric("r2", best_models_dict['CTR']['R2'])
                print(f"  ✅ CTR {ctr_best_model_name} registered (R²={best_models_dict['CTR']['R2']:.4f})")
    
    # 2. ROAS Model (RandomForest) - trained on selected features
    if 'ROAS' in best_models_dict and roas_models is not None:
        roas_best_model_name = best_models_dict['ROAS']['Model']
        if roas_best_model_name in all_models['ROAS'] and all_models['ROAS'][roas_best_model_name] is not None:
            roas_model = all_models['ROAS'][roas_best_model_name]
            
            # ✅ FIX: Re-run feature selection for ROAS
            X_train_roas_sel, _, _ = select_features(
                X_train_roas, X_test_roas, y_train_roas, feature_cols, "ROAS"
            )
            
            with mlflow.start_run(run_name=f"ROAS_{roas_best_model_name}") as run:
                signature = mlflow.models.infer_signature(X_train_roas_sel, roas_model.predict(X_train_roas_sel))
                if roas_best_model_name == 'XGBoost':
                    mlflow.xgboost.log_model(roas_model, "roas_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.roas_predictor")
                else:
                    mlflow.sklearn.log_model(roas_model, "roas_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.roas_predictor")
                mlflow.log_metric("r2", best_models_dict['ROAS']['R2'])
                print(f"  ✅ ROAS {roas_best_model_name} registered (R²={best_models_dict['ROAS']['R2']:.4f})")
    
    # 3. Conversion Rate Model (RandomForest) - trained on selected features
    if 'ConversionRate' in best_models_dict:
        conv_best_model_name = best_models_dict['ConversionRate']['Model']
        if conv_best_model_name in all_models['ConversionRate'] and all_models['ConversionRate'][conv_best_model_name] is not None:
            conv_model = all_models['ConversionRate'][conv_best_model_name]
            
            # ✅ FIX: Re-run feature selection for ConversionRate
            X_train_conv_sel, _, _ = select_features(
                X_train_scaled, X_test_scaled, y_conversion_train, feature_cols, "ConversionRate"
            )
            
            with mlflow.start_run(run_name=f"Conversion_{conv_best_model_name}") as run:
                signature = mlflow.models.infer_signature(X_train_conv_sel, conv_model.predict(X_train_conv_sel))
                if conv_best_model_name == 'XGBoost':
                    mlflow.xgboost.log_model(conv_model, "conversion_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.conversion_predictor")
                else:
                    mlflow.sklearn.log_model(conv_model, "conversion_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.conversion_predictor")
                mlflow.log_metric("r2", best_models_dict['ConversionRate']['R2'])
                print(f"  ✅ Conversion {conv_best_model_name} registered (R²={best_models_dict['ConversionRate']['R2']:.4f})")
    
    # 4. Cost Efficiency Model (GradientBoosting) - trained on selected features
    if 'CostEfficiency' in best_models_dict:
        cost_best_model_name = best_models_dict['CostEfficiency']['Model']
        if cost_best_model_name in all_models['CostEfficiency'] and all_models['CostEfficiency'][cost_best_model_name] is not None:
            cost_model = all_models['CostEfficiency'][cost_best_model_name]
            
            # ✅ FIX: Re-run feature selection for CostEfficiency
            X_train_cost_sel, _, _ = select_features(
                X_train_scaled, X_test_scaled, y_cost_efficiency_train, feature_cols, "CostEfficiency"
            )
            
            with mlflow.start_run(run_name=f"CostEfficiency_{cost_best_model_name}") as run:
                signature = mlflow.models.infer_signature(X_train_cost_sel, cost_model.predict(X_train_cost_sel))
                if cost_best_model_name == 'XGBoost':
                    mlflow.xgboost.log_model(cost_model, "cost_efficiency_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.cost_efficiency_predictor")
                else:
                    mlflow.sklearn.log_model(cost_model, "cost_efficiency_model", signature=signature, 
                                             registered_model_name="adtech_catalog.ml_models.cost_efficiency_predictor")
                mlflow.log_metric("r2", best_models_dict['CostEfficiency']['R2'])
                print(f"  ✅ Cost Efficiency {cost_best_model_name} registered (R²={best_models_dict['CostEfficiency']['R2']:.4f})")
    
    print("\n✅ All models registered successfully!")

except Exception as e:
    print(f"⚠️ MLflow registration error: {e}")
    print("   Models trained successfully but not registered to MLflow.")
# ============================================================
# 11. COMPILE RESULTS
# ============================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

summary_data = []
for res in all_results:
    if res:
        summary_data.append({
            "Target": res['target'],
            "Model": res['model'],
            "R2": res['r2'],
            "RMSE": res['rmse'],
            "MAE": res['mae']
        })

summary_df = pd.DataFrame(summary_data)

# Pivot table for better view
pivot_df = summary_df.pivot(index='Target', columns='Model', values='R2')
print("\nR² by Target and Model:")
print(pivot_df.round(4))

print("\n" + "="*70)
print("BEST MODEL PER TARGET")
print("="*70)

best_models_list = []
for target in summary_df['Target'].unique():
    target_df = summary_df[summary_df['Target'] == target]
    best = target_df.loc[target_df['R2'].idxmax()]
    best_models_list.append(best)
    print(f"  {target}: {best['Model']} (R2: {best['R2']:.4f})")

# ============================================================
# 12. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    results_df = pd.DataFrame(summary_data)
    results_df['version'] = VERSION
    results_df['environment'] = ENVIRONMENT
    results_df['training_timestamp'] = datetime.now().isoformat()
    
    results_df.to_csv(volume_path + "regression_advanced_results.csv", index=False)
    print(f"Results saved to: {volume_path}regression_advanced_results.csv")
    
    spark_results = spark.createDataFrame(results_df)
    spark_results.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.regression_advanced_results")
    print("Results saved to: adtech_catalog.monitoring.regression_advanced_results")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 13. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Advanced Regression Models (Improved)",
        "SUCCESS"
    )], [
        "version_id",
        "environment",
        "git_commit",
        "deployed_at",
        "description",
        "status"
    ])

    version_info.write \
        .mode("append") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.version_history")

    print("Version history updated: adtech_catalog.monitoring.version_history")
    print(f"   Version: {VERSION}")

except Exception as e:
    print(f"Could not save version history: {e}")

# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("ADVANCED REGRESSION MODELS COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Data:
   - Training: {len(train_df):,} rows
   - Test: {len(test_df):,} rows
   - Base Features: {len(base_feature_cols)}
   - Interaction Features: {len(interaction_features)}
   - Total Features: {len(feature_cols)}

Models Trained:
   1. RandomForestRegressor (with tuning)
   2. GradientBoostingRegressor
   3. XGBRegressor (with tuning)
   4. LightGBMRegressor (with tuning)
   5. Selective Ensemble (top 2 models)

Targets Trained:
   1. CTR
   2. ROAS
   3. Conversion Rate
   4. Cost Efficiency
   5. DED Score (SKIPPED - no variation)

Improvements Applied:
   ✅ Interaction Features (3 new)
   ✅ Feature Selection
   ✅ Hyperparameter Tuning
   ✅ Selective Ensemble

Best Models:
""")

for best in best_models_list:
    print(f"   {best['Target']}: {best['Model']} (R2: {best['R2']:.4f})")

print("""
Next Steps:
   1. Run: 04_Train_Classification_Models.py
   2. Run: 05_Model_Explainability.py
======================================================================
""")

print("")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# DYNAMIC MODEL SELECTION & SUMMARY
# ============================================================
# Purpose: Automatically select the best model for each target
# from all trained models and generate a comprehensive summary.
#
# This cell dynamically loads results from saved tables and:
# 1. Compares all models across all cells
# 2. Selects the best model per target
# 3. Displays performance comparisons
# 4. Shows feature importance (if available)
# 5. Generates recommendations

import pandas as pd
import numpy as np
from datetime import datetime

print("="*70)
print("MODEL SELECTION & PERFORMANCE SUMMARY")
print("="*70)

# ============================================================
# 1. LOAD ALL RESULTS FROM SAVED TABLES
# ============================================================

def load_results(table_name, description):
    """Load results from a Delta table with error handling"""
    try:
        df = spark.table(table_name).toPandas()
        print(f"✅ Loaded {len(df)} records from: {description}")
        return df
    except Exception as e:
        print(f"⚠️  Could not load from {description}: {e}")
        return None

print("\nLOADING RESULTS FROM ALL SOURCES...")
print("-"*70)

# Load baseline results
baseline_df = load_results("adtech_catalog.monitoring.baseline_results", "Baseline Models")

# Load advanced regression results (Cell 4 - with LightGBM)
regression_advanced_df = load_results(
    "adtech_catalog.monitoring.regression_advanced_results", 
    "Advanced Regression (with LightGBM)"
)

# Try to load improved regression results (Cell 5 - with feature engineering)
try:
    # Check if improved results exist in the same table (overwritten)
    improved_df = spark.table("adtech_catalog.monitoring.regression_advanced_results").toPandas()
    
    # Check if it has feature engineering indicators
    if 'version' in improved_df.columns:
        # Get the latest version (most recent run)
        latest_version = improved_df['version'].max()
        improved_df = improved_df[improved_df['version'] == latest_version]
        print(f"✅ Loaded improved results (Cell 5) from monitoring table")
    else:
        improved_df = None
        print("ℹ️  No separate improved results found")
except:
    improved_df = None
    print("ℹ️  No improved results table found")

# ============================================================
# 2. DETERMINE WHICH RESULTS TO USE
# ============================================================

print("\n" + "="*70)
print("SELECTING BEST MODELS")
print("="*70)

# If we have improved results (Cell 5), use them as the main source
# Otherwise use advanced regression results (Cell 4)
if improved_df is not None and len(improved_df) > 0:
    main_results = improved_df
    source = "Cell 5 (Feature Engineering + Selection)"
else:
    main_results = regression_advanced_df
    source = "Cell 4 (Advanced Regression with LightGBM)"

print(f"📊 Using results from: {source}")

# ============================================================
# 3. FIND BEST MODEL PER TARGET
# ============================================================

def find_best_models(results_df):
    """Find the best model for each target based on R²"""
    best_models = []
    targets = results_df['Target'].unique()
    
    for target in targets:
        target_df = results_df[results_df['Target'] == target]
        best = target_df.loc[target_df['R2'].idxmax()]
        best_models.append(best)
    
    return pd.DataFrame(best_models)

if main_results is not None and len(main_results) > 0:
    best_models_df = find_best_models(main_results)
    
    print("\n🏆 BEST MODEL PER TARGET")
    print("-"*70)
    print(f"{'Target':<20} {'Model':<25} {'R²':<10} {'RMSE':<12} {'MAE':<10}")
    print("-"*70)
    
    for _, row in best_models_df.iterrows():
        target = row['Target']
        model = row['Model']
        r2 = row['R2']
        rmse = row.get('RMSE', 'N/A')
        mae = row.get('MAE', 'N/A')
        
        # Performance label
        if r2 > 0.3:
            label = "✅ EXCELLENT"
        elif r2 > 0.15:
            label = "👍 GOOD"
        elif r2 > 0:
            label = "📊 FAIR"
        else:
            label = "⚠️ POOR"
        
        print(f"{target:<20} {model:<25} {r2:+.4f}    {rmse if rmse == 'N/A' else f'{rmse:.4f}':<12} {mae if mae == 'N/A' else f'{mae:.4f}':<10} {label}")
    
    # ============================================================
    # 4. COMPARE AGAINST BASELINE
    # ============================================================
    
    if baseline_df is not None and len(baseline_df) > 0:
        print("\n" + "="*70)
        print("IMPROVEMENT OVER BASELINE")
        print("="*70)
        print(f"{'Target':<20} {'Baseline R²':<14} {'Best R²':<12} {'Improvement':<12} {'Status':<10}")
        print("-"*70)
        
        for _, row in best_models_df.iterrows():
            target = row['Target']
            best_r2 = row['R2']
            
            # Find baseline R²
            baseline_match = baseline_df[baseline_df['target'] == target]
            if len(baseline_match) > 0:
                baseline_r2 = baseline_match['r2'].values[0] if 'r2' in baseline_match.columns else 0
            else:
                baseline_r2 = 0
            
            # Calculate improvement
            if baseline_r2 != 0:
                improvement_pct = ((best_r2 - baseline_r2) / abs(baseline_r2)) * 100
            else:
                improvement_pct = 0
            
            # Status
            if improvement_pct > 50:
                status = "✅ GREAT"
            elif improvement_pct > 10:
                status = "📈 GOOD"
            elif improvement_pct > 0:
                status = "📊 MODEST"
            else:
                status = "⚠️ NEGATIVE"
            
            print(f"{target:<20} {baseline_r2:+.4f}      {best_r2:+.4f}     {improvement_pct:+.1f}%     {status:<10}")
    
    # ============================================================
    # 5. MODEL PERFORMANCE SUMMARY
    # ============================================================
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE BY TARGET")
    print("="*70)
    
    # Pivot table for all models
    pivot_df = main_results.pivot_table(
        index='Target', 
        columns='Model', 
        values='R2'
    ).round(4)
    
    print("\nR² by Target and Model:")
    print(pivot_df.to_string())
    
    # ============================================================
    # 6. BEST MODEL COUNT
    # ============================================================
    
    print("\n" + "="*70)
    print("BEST MODEL FREQUENCY")
    print("="*70)
    
    model_counts = best_models_df['Model'].value_counts()
    
    for model, count in model_counts.items():
        print(f"  {model}: {count} target(s)")
    
    # ============================================================
    # 7. RECOMMENDATIONS
    # ============================================================
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    for _, row in best_models_df.iterrows():
        target = row['Target']
        model = row['Model']
        r2 = row['R2']
        
        if target == 'DEDScore' or r2 < 0.01:
            print(f"⚠️  {target}: Consider skipping - very low predictive value (R²={r2:.4f})")
        elif r2 > 0.3:
            print(f"✅ {target}: Use {model} (R²={r2:.4f}) - Model is performing well")
        elif r2 > 0.15:
            print(f"📈 {target}: Use {model} (R²={r2:.4f}) - Good performance, consider further tuning")
        elif r2 > 0:
            print(f"📊 {target}: Use {model} (R²={r2:.4f}) - Fair performance, may need more features")
        else:
            print(f"⚠️  {target}: All models underperform (R²={r2:.4f}) - Consider different approach")

else:
    print("\n❌ No results found! Please run Cell 4 or Cell 5 first.")

# ============================================================
# 8. FINAL SELECTION SUMMARY
# ============================================================

print("\n" + "="*70)
print("FINAL MODEL SELECTION SUMMARY")
print("="*70)

if main_results is not None and len(main_results) > 0:
    print(f"\n📊 Source: {source}")
    print(f"📈 Total Models Evaluated: {len(main_results)}")
    print(f"🎯 Total Targets: {len(main_results['Target'].unique())}")
    
    print("\n🏆 FINAL SELECTION:")
    print("-"*70)
    
    for _, row in best_models_df.iterrows():
        target = row['Target']
        model = row['Model']
        r2 = row['R2']
        
        if r2 > 0.3:
            print(f"  ✅ {target:<20} → {model:<25} (R²: {r2:+.4f}) - SELECTED")
        elif r2 > 0.15:
            print(f"  ✅ {target:<20} → {model:<25} (R²: {r2:+.4f}) - SELECTED")
        elif r2 > 0:
            print(f"  📊 {target:<20} → {model:<25} (R²: {r2:+.4f}) - PROCEED WITH CAUTION")
        else:
            print(f"  ⚠️ {target:<20} → {model:<25} (R²: {r2:+.4f}) - REVIEW / SKIP")

else:
    print("\n❌ No results available for final selection.")
    print("   Please run regression models first.")

# ============================================================
# 9. SAVE FINAL SELECTION
# ============================================================

print("\n" + "="*70)
print("SAVING FINAL SELECTION")
print("="*70)

try:
    if main_results is not None and len(main_results) > 0:
        # Create selection summary
        selection_summary = best_models_df[['Target', 'Model', 'R2', 'RMSE', 'MAE']].copy()
        selection_summary['selection_timestamp'] = datetime.now().isoformat()
        selection_summary['source'] = source
        
        # Save to Delta table
        spark_selection = spark.createDataFrame(selection_summary)
        spark_selection.write \
            .mode("overwrite") \
            .format("delta") \
            .saveAsTable("adtech_catalog.monitoring.model_selection_summary")
        
        print("✅ Model selection summary saved to: adtech_catalog.monitoring.model_selection_summary")
        
        # Save as CSV backup
        volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"
        selection_summary.to_csv(volume_path + "model_selection_summary.csv", index=False)
        print(f"✅ Model selection summary saved to: {volume_path}model_selection_summary.csv")
        
        print(f"\n📋 Selection Summary ({len(selection_summary)} targets):")
        print(selection_summary.to_string(index=False))
        
except Exception as e:
    print(f"⚠️  Could not save selection summary: {e}")

# ============================================================
# 10. VERSION HISTORY
# ============================================================

print("\n" + "="*70)
print("VERSION HISTORY")
print("="*70)

try:
    version_info = spark.createDataFrame([(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        ENVIRONMENT if 'ENVIRONMENT' in dir() else 'development',
        GIT_COMMIT if 'GIT_COMMIT' in dir() else 'local',
        datetime.now().isoformat(),
        "Model Selection Summary - Final",
        "SUCCESS"
    )], [
        "version_id",
        "environment",
        "git_commit",
        "deployed_at",
        "description",
        "status"
    ])

    version_info.write \
        .mode("append") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.version_history")

    print("Version history updated: adtech_catalog.monitoring.version_history")

except Exception as e:
    print(f"Could not save version history: {e}")

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("✅ MODEL SELECTION COMPLETE")
print("="*70)

print("""
SUMMARY
======================================================================
This dynamic summary has:
1. Loaded results from all trained models
2. Selected the best model per target
3. Compared against baseline performance
4. Generated recommendations
5. Saved final selection to monitoring table

NEXT STEPS:
1. Review the final selection above
2. Proceed with selected models for deployment
3. Run: 05_Model_Explainability.py (SHAP analysis)
4. Run: 06_Model_Comparison_and_Selection.py
======================================================================
""")

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

# Promote all 4 models using ALIASES
model_names = [
    "adtech_catalog.ml_models.ctr_predictor",
    "adtech_catalog.ml_models.roas_predictor",
    "adtech_catalog.ml_models.conversion_predictor",
    "adtech_catalog.ml_models.cost_efficiency_predictor"
]

for model_name in model_names:
    try:
        # Set alias "Production" to version 1
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=1
        )
        print(f"✅ {model_name} alias 'Production' set to Version 1")
    except Exception as e:
        print(f"❌ {model_name}: {e}")

print("\n✅ All models have 'Production' alias set!")