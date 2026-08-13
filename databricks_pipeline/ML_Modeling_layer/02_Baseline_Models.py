# Databricks notebook source
# Databricks notebook source
# ============================================================
# 02_Baseline_Models
# ============================================================
# Purpose: Establish baseline performance for all 7 ML targets

# This notebook trains simple baseline models for all 7 ML targets.
# These baselines establish the "floor" performance before any
# advanced modeling. If advanced models don't beat these baselines
# meaningfully, that's an honest finding to report.
#
# TARGETS:
# 1. CTR (Regression)
# 2. ROAS (Regression)
# 3. Conversion Rate (Regression)
# 4. High Performance (Binary Classification)
# 5. DED Score (Regression)
# 6. Ad Lifecycle Stage (Multi-class Classification)
# 7. Cost Efficiency Score (Regression)

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("BASELINE MODELS")
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
# 2. LOAD TRAIN/TEST DATA WITH TYPE CONVERSION
# ============================================================

print("\nLOADING TRAIN/TEST DATA...")

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

try:
    train_df = pd.read_csv(volume_path + "train_split.csv")
    test_df = pd.read_csv(volume_path + "test_split.csv")
    print(f"Loaded training data: {len(train_df):,} rows")
    print(f"Loaded test data: {len(test_df):,} rows")
    
    # Convert high_performance to int (not string)
    if 'high_performance' in train_df.columns:
        train_df['high_performance'] = train_df['high_performance'].astype(int)
        test_df['high_performance'] = test_df['high_performance'].astype(int)
        print("  high_performance converted to int")
    
    # Convert other numeric columns
    numeric_cols = ['ctr', 'roas', 'conversion_rate', 'avg_ded_score', 
                    'cost_efficiency_score', 'cost_per_click', 'ad_video_length']
    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # Fill any NaN from conversion
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # Verify class distribution
    if 'high_performance' in train_df.columns:
        train_hp_counts = train_df['high_performance'].value_counts()
        test_hp_counts = test_df['high_performance'].value_counts()
        
        print(f"\nClass Distribution Check:")
        print(f"  Training - Class 0: {train_hp_counts.get(0, 0)}")
        print(f"  Training - Class 1: {train_hp_counts.get(1, 0)}")
        print(f"  Test - Class 0: {test_hp_counts.get(0, 0)}")
        print(f"  Test - Class 1: {test_hp_counts.get(1, 0)}")
        
        if train_hp_counts.get(1, 0) == 0:
            print("\n  ERROR: Training set has NO profitable ads (class 1)!")
            dbutils.notebook.exit("Invalid split: no class 1 in training")
        else:
            print("\n  VALID SPLIT: Both classes present in training")
    
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

y_hp_train = train_df['high_performance'].values
y_hp_test = test_df['high_performance'].values

# Ad Lifecycle Stage
y_lifecycle_train = train_df['ad_lifecycle_stage'].values
y_lifecycle_test = test_df['ad_lifecycle_stage'].values

lifecycle_labels = ['New', 'Growing', 'Mature', 'Declining']
le_lifecycle = LabelEncoder()
y_lifecycle_train_encoded = le_lifecycle.fit_transform(y_lifecycle_train)
y_lifecycle_test_encoded = le_lifecycle.transform(y_lifecycle_test)

print("Targets prepared")

# ============================================================
# 6. EVALUATION FUNCTIONS
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

def evaluate_classification_binary(y_true, y_pred, y_prob, model_name, target_name):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "target": target_name,
        "model": model_name,
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# ============================================================
# 7. MODEL 1: CTR PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 1: CTR PREDICTION (Baseline)")
print("="*70)

ctr_model = LinearRegression()
ctr_model.fit(X_train_scaled, y_ctr_train)
y_ctr_pred = ctr_model.predict(X_test_scaled)
ctr_results = evaluate_regression(y_ctr_test, y_ctr_pred, "LinearRegression", "CTR")
print(f"  RMSE: {ctr_results['rmse']:.4f}")
print(f"  MAE:  {ctr_results['mae']:.4f}")
print(f"  R2:   {ctr_results['r2']:.4f}")

# ============================================================
# 8. MODEL 2: ROAS PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 2: ROAS PREDICTION (Baseline)")
print("="*70)

roas_mask_train = y_roas_train > 0
roas_mask_test = y_roas_test > 0

if np.sum(roas_mask_train) > 10 and np.sum(roas_mask_test) > 0:
    X_roas_train = X_train_scaled[roas_mask_train]
    X_roas_test = X_test_scaled[roas_mask_test]
    y_roas_train_filtered = y_roas_train[roas_mask_train]
    y_roas_test_filtered = y_roas_test[roas_mask_test]
    
    roas_model = LinearRegression()
    roas_model.fit(X_roas_train, y_roas_train_filtered)
    y_roas_pred = roas_model.predict(X_roas_test)
    roas_results = evaluate_regression(y_roas_test_filtered, y_roas_pred, "LinearRegression", "ROAS")
    
    print(f"  RMSE: {roas_results['rmse']:.4f}")
    print(f"  MAE:  {roas_results['mae']:.4f}")
    print(f"  R2:   {roas_results['r2']:.4f}")
    print(f"  Samples used: {len(X_roas_train)} train, {len(X_roas_test)} test")
else:
    print("  Not enough valid ROAS samples for baseline")
    roas_results = None

# ============================================================
# 9. MODEL 3: CONVERSION RATE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 3: CONVERSION RATE PREDICTION (Baseline)")
print("="*70)

conversion_model = LinearRegression()
conversion_model.fit(X_train_scaled, y_conversion_train)
y_conversion_pred = conversion_model.predict(X_test_scaled)
conversion_results = evaluate_regression(y_conversion_test, y_conversion_pred, "LinearRegression", "ConversionRate")
print(f"  RMSE: {conversion_results['rmse']:.4f}")
print(f"  MAE:  {conversion_results['mae']:.4f}")
print(f"  R2:   {conversion_results['r2']:.4f}")

# ============================================================
# 10. MODEL 4: DED SCORE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 4: DED SCORE PREDICTION (Baseline)")
print("="*70)

ded_model = LinearRegression()
ded_model.fit(X_train_scaled, y_ded_train)
y_ded_pred = ded_model.predict(X_test_scaled)
ded_results = evaluate_regression(y_ded_test, y_ded_pred, "LinearRegression", "DEDScore")
print(f"  RMSE: {ded_results['rmse']:.4f}")
print(f"  MAE:  {ded_results['mae']:.4f}")
print(f"  R2:   {ded_results['r2']:.4f}")

# ============================================================
# 11. MODEL 5: COST EFFICIENCY SCORE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 5: COST EFFICIENCY SCORE PREDICTION (Baseline)")
print("="*70)

cost_efficiency_model = LinearRegression()
cost_efficiency_model.fit(X_train_scaled, y_cost_efficiency_train)
y_cost_efficiency_pred = cost_efficiency_model.predict(X_test_scaled)
cost_efficiency_results = evaluate_regression(y_cost_efficiency_test, y_cost_efficiency_pred, "LinearRegression", "CostEfficiency")
print(f"  RMSE: {cost_efficiency_results['rmse']:.4f}")
print(f"  MAE:  {cost_efficiency_results['mae']:.4f}")
print(f"  R2:   {cost_efficiency_results['r2']:.4f}")

# ============================================================
# 12. MODEL 6: HIGH PERFORMANCE CLASSIFICATION
# ============================================================

print("\n" + "="*70)
print("MODEL 6: HIGH PERFORMANCE CLASSIFICATION (Baseline)")
print("="*70)

# DEBUG: Double-check class distribution before training
unique_train = np.unique(y_hp_train)
unique_test = np.unique(y_hp_test)

print(f"  Unique classes in training: {unique_train}")
print(f"  Unique classes in test: {unique_test}")

if len(unique_train) < 2:
    print("  ERROR: Training data has only one class!")
    print("  The split files are still invalid.")
    print("  Please re-run 01_EDA_and_Data_Preparation.py")
    print("  and ensure the split files are overwritten.")
    print("\n  Using Dummy Baseline to continue.")
    
    y_hp_pred = np.zeros_like(y_hp_test)
    y_hp_prob = np.ones_like(y_hp_test, dtype=float) * 0.5
    
    hp_results = {
        "target": "HighPerformance",
        "model": "DummyBaseline",
        "auc": 0.5,
        "accuracy": accuracy_score(y_hp_test, y_hp_pred),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    
    print(f"\n  Dummy Baseline Results:")
    print(f"  Accuracy: {hp_results['accuracy']:.4f}")
    print(f"  AUC: 0.5")
    
else:
    hp_model = LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, max_iter=1000)
    hp_model.fit(X_train_scaled, y_hp_train)

    y_hp_pred = hp_model.predict(X_test_scaled)
    y_hp_prob = hp_model.predict_proba(X_test_scaled)[:, 1]

    hp_results = evaluate_classification_binary(
        y_hp_test, y_hp_pred, y_hp_prob, 
        "LogisticRegression", "HighPerformance"
    )

    print(f"  AUC:      {hp_results['auc']:.4f}")
    print(f"  Accuracy: {hp_results['accuracy']:.4f}")
    print(f"  Precision:{hp_results['precision']:.4f}")
    print(f"  Recall:   {hp_results['recall']:.4f}")
    print(f"  F1:       {hp_results['f1']:.4f}")

# ============================================================
# 13. MODEL 7: AD LIFECYCLE STAGE
# ============================================================

print("\n" + "="*70)
print("MODEL 7: AD LIFECYCLE STAGE (Baseline)")
print("="*70)

lifecycle_model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=RANDOM_SEED)
lifecycle_model.fit(X_train_scaled, y_lifecycle_train_encoded)

y_lifecycle_pred_encoded = lifecycle_model.predict(X_test_scaled)
y_lifecycle_pred = le_lifecycle.inverse_transform(y_lifecycle_pred_encoded)

lifecycle_acc = accuracy_score(y_lifecycle_test, y_lifecycle_pred)
lifecycle_f1 = f1_score(y_lifecycle_test_encoded, y_lifecycle_pred_encoded, average='weighted', zero_division=0)

print(f"  Accuracy: {lifecycle_acc:.4f}")
print(f"  F1 (weighted): {lifecycle_f1:.4f}")

print("\n  Confusion Matrix:")
cm = confusion_matrix(y_lifecycle_test_encoded, y_lifecycle_pred_encoded)
print(cm)


# ============================================================
# 14. COMPILE RESULTS
# ============================================================

print("\n" + "="*70)
print("BASELINE RESULTS SUMMARY")
print("="*70)

all_results = []

regression_results = [
    ctr_results,
    conversion_results,
    ded_results,
    cost_efficiency_results
]

if roas_results:
    regression_results.append(roas_results)

for res in regression_results:
    if res:
        all_results.append(res)
        print(f"\n{res['target']} ({res['model']}):")
        print(f"  R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}")

if hp_results:
    all_results.append(hp_results)
    print(f"\n{hp_results['target']} ({hp_results['model']}):")
    print(f"  AUC: {hp_results['auc']:.4f}, Accuracy: {hp_results['accuracy']:.4f}")
    print(f"  Precision: {hp_results['precision']:.4f}, Recall: {hp_results['recall']:.4f}, F1: {hp_results['f1']:.4f}")
else:
    print("\nHigh Performance Classification: SKIPPED")

print(f"\nLifecycle Stage (LogisticRegression):")
print(f"  Accuracy: {lifecycle_acc:.4f}, F1 (weighted): {lifecycle_f1:.4f}")

# ============================================================
# 15. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    valid_results = [r for r in all_results if r is not None]
    
    if valid_results:
        results_df = pd.DataFrame(valid_results)
        results_df['version'] = VERSION
        results_df['environment'] = ENVIRONMENT
        results_df['training_timestamp'] = datetime.now().isoformat()
        
        results_df.to_csv(volume_path + "baseline_results.csv", index=False)
        print(f"Results saved to: {volume_path}baseline_results.csv")
        
        spark_results = spark.createDataFrame(results_df)
        spark_results.write \
            .mode("overwrite") \
            .format("delta") \
            .saveAsTable("adtech_catalog.monitoring.baseline_results")
        print("Results saved to: adtech_catalog.monitoring.baseline_results")
    else:
        print("No valid results to save")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 16. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Baseline Models - All 7 Targets",
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
# 17. SUMMARY
# ============================================================

print("\n" + "="*70)
print("BASELINE MODELS COMPLETE")
print("="*70)

summary_text = f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Data:
   - Training: {len(train_df):,} rows
   - Test: {len(test_df):,} rows
   - Features: {len(feature_cols)}

Key Baseline Metrics:
   - CTR R2: {ctr_results['r2']:.4f}
   - Conversion R2: {conversion_results['r2']:.4f}
   - DED Score R2: {ded_results['r2']:.4f}
   - Cost Efficiency R2: {cost_efficiency_results['r2']:.4f}
"""

if hp_results:
    summary_text += f"   - High Performance AUC: {hp_results['auc']:.4f}\n"
else:
    summary_text += "   - High Performance: DUMMY BASELINE (no class 1 in training)\n"

if roas_results:
    summary_text += f"   - ROAS R2: {roas_results['r2']:.4f}\n"
else:
    summary_text += "   - ROAS: SKIPPED (not enough valid samples)\n"

summary_text += f"""   - Lifecycle Accuracy: {lifecycle_acc:.4f}

Next Steps:
   1. Run: 03_Train_Regression_Models.py
   2. Run: 04_Train_Classification_Models.py
======================================================================
"""

print(summary_text)

print("")

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 02_Baseline_Models
# ============================================================
# Purpose: Establish baseline performance for all 7 ML targets

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("BASELINE MODELS")
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
    
    # Convert high_performance to int
    if 'high_performance' in train_df.columns:
        train_df['high_performance'] = train_df['high_performance'].astype(int)
        test_df['high_performance'] = test_df['high_performance'].astype(int)
    
    # Convert other numeric columns
    numeric_cols = ['ctr', 'roas', 'conversion_rate', 'avg_ded_score', 
                    'cost_efficiency_score', 'cost_per_click', 'ad_video_length']
    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # Verify class distribution
    train_hp = train_df['high_performance'].value_counts()
    print(f"\nTraining class distribution:")
    print(f"  Class 0: {train_hp.get(0, 0)}")
    print(f"  Class 1: {train_hp.get(1, 0)}")
    
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

y_hp_train = train_df['high_performance'].values
y_hp_test = test_df['high_performance'].values

print("Targets prepared")

# ============================================================
# 6. EVALUATION FUNCTIONS
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

def evaluate_dummy_classification(y_true, y_pred, model_name, target_name):
    acc = accuracy_score(y_true, y_pred)
    return {
        "target": target_name,
        "model": model_name,
        "accuracy": acc,
        "auc": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }

# ============================================================
# 7. MODEL 1: CTR PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 1: CTR PREDICTION (Baseline)")
print("="*70)

ctr_model = LinearRegression()
ctr_model.fit(X_train_scaled, y_ctr_train)
y_ctr_pred = ctr_model.predict(X_test_scaled)
ctr_results = evaluate_regression(y_ctr_test, y_ctr_pred, "LinearRegression", "CTR")
print(f"  R2: {ctr_results['r2']:.4f}, RMSE: {ctr_results['rmse']:.4f}, MAE: {ctr_results['mae']:.4f}")

# ============================================================
# 8. MODEL 2: ROAS PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 2: ROAS PREDICTION (Baseline)")
print("="*70)

roas_mask_train = y_roas_train > 0
roas_mask_test = y_roas_test > 0

if np.sum(roas_mask_train) > 10 and np.sum(roas_mask_test) > 0:
    X_roas_train = X_train_scaled[roas_mask_train]
    X_roas_test = X_test_scaled[roas_mask_test]
    y_roas_train_filtered = y_roas_train[roas_mask_train]
    y_roas_test_filtered = y_roas_test[roas_mask_test]
    
    roas_model = LinearRegression()
    roas_model.fit(X_roas_train, y_roas_train_filtered)
    y_roas_pred = roas_model.predict(X_roas_test)
    roas_results = evaluate_regression(y_roas_test_filtered, y_roas_pred, "LinearRegression", "ROAS")
    
    print(f"  R2: {roas_results['r2']:.4f}, RMSE: {roas_results['rmse']:.4f}, MAE: {roas_results['mae']:.4f}")
else:
    print("  Not enough valid ROAS samples for baseline")
    roas_results = None

# ============================================================
# 9. MODEL 3: CONVERSION RATE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 3: CONVERSION RATE PREDICTION (Baseline)")
print("="*70)

conversion_model = LinearRegression()
conversion_model.fit(X_train_scaled, y_conversion_train)
y_conversion_pred = conversion_model.predict(X_test_scaled)
conversion_results = evaluate_regression(y_conversion_test, y_conversion_pred, "LinearRegression", "ConversionRate")
print(f"  R2: {conversion_results['r2']:.4f}, RMSE: {conversion_results['rmse']:.4f}, MAE: {conversion_results['mae']:.4f}")

# ============================================================
# 10. MODEL 4: DED SCORE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 4: DED SCORE PREDICTION (Baseline)")
print("="*70)

ded_model = LinearRegression()
ded_model.fit(X_train_scaled, y_ded_train)
y_ded_pred = ded_model.predict(X_test_scaled)
ded_results = evaluate_regression(y_ded_test, y_ded_pred, "LinearRegression", "DEDScore")
print(f"  R2: {ded_results['r2']:.4f}, RMSE: {ded_results['rmse']:.4f}, MAE: {ded_results['mae']:.4f}")

# ============================================================
# 11. MODEL 5: COST EFFICIENCY SCORE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 5: COST EFFICIENCY SCORE PREDICTION (Baseline)")
print("="*70)

cost_efficiency_model = LinearRegression()
cost_efficiency_model.fit(X_train_scaled, y_cost_efficiency_train)
y_cost_efficiency_pred = cost_efficiency_model.predict(X_test_scaled)
cost_efficiency_results = evaluate_regression(y_cost_efficiency_test, y_cost_efficiency_pred, "LinearRegression", "CostEfficiency")
print(f"  R2: {cost_efficiency_results['r2']:.4f}, RMSE: {cost_efficiency_results['rmse']:.4f}, MAE: {cost_efficiency_results['mae']:.4f}")

# ============================================================
# 12. MODEL 6: HIGH PERFORMANCE - DUMMY BASELINE
# ============================================================

print("\n" + "="*70)
print("MODEL 6: HIGH PERFORMANCE CLASSIFICATION (Dummy Baseline)")
print("="*70)

# Since there are only 26 profitable ads in training, Logistic Regression can fail.
# Use a simple Dummy Classifier that predicts the majority class (0)
dummy_model = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
dummy_model.fit(X_train_scaled, y_hp_train)

y_hp_pred = dummy_model.predict(X_test_scaled)
y_hp_prob = dummy_model.predict_proba(X_test_scaled)[:, 1]

hp_results = evaluate_dummy_classification(y_hp_test, y_hp_pred, "DummyClassifier", "HighPerformance")

print(f"  Accuracy: {hp_results['accuracy']:.4f}")
print(f"  AUC: 0.5 (random - dummy model)")
print(f"  Note: Logistic Regression skipped due to class imbalance (only 26 profitable ads in training)")

# ============================================================
# 13. COMPILE RESULTS
# ============================================================

print("\n" + "="*70)
print("BASELINE RESULTS SUMMARY")
print("="*70)

all_results = [ctr_results, conversion_results, ded_results, cost_efficiency_results, hp_results]

if roas_results:
    all_results.append(roas_results)

for res in all_results:
    if res:
        print(f"\n{res['target']} ({res['model']}):")
        if 'r2' in res:
            print(f"  R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}")
        elif 'accuracy' in res:
            print(f"  Accuracy: {res['accuracy']:.4f}, AUC: {res['auc']:.4f}")


# ============================================================
# 13.5 FORMATTED SUMMARY TABLE (ADD THIS)
# ============================================================

print("\n" + "="*70)
print("BASELINE PERFORMANCE SUMMARY TABLE")
print("="*70)

summary_data = []

for res in all_results:
    if res:
        row = {
            "Target": res.get("target", "N/A"),
            "Model": res.get("model", "N/A"),
            "Type": "Regression" if "r2" in res else "Classification",
        }
        
        if "r2" in res:
            row["R2"] = f"{res.get('r2', 0):.4f}"
            row["RMSE"] = f"{res.get('rmse', 0):.4f}"
            row["MAE"] = f"{res.get('mae', 0):.4f}"
            row["AUC"] = "-"
            row["Accuracy"] = "-"
        else:
            row["R2"] = "-"
            row["RMSE"] = "-"
            row["MAE"] = "-"
            row["AUC"] = f"{res.get('auc', 0):.4f}"
            row["Accuracy"] = f"{res.get('accuracy', 0):.4f}"
        
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ============================================================
# 14. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    valid_results = [r for r in all_results if r is not None]
    
    if valid_results:
        results_df = pd.DataFrame(valid_results)
        results_df['version'] = VERSION
        results_df['environment'] = ENVIRONMENT
        results_df['training_timestamp'] = datetime.now().isoformat()
        
        results_df.to_csv(volume_path + "baseline_results.csv", index=False)
        print(f"Results saved to: {volume_path}baseline_results.csv")
        
        spark_results = spark.createDataFrame(results_df)
        spark_results.write \
            .mode("overwrite") \
            .format("delta") \
            .saveAsTable("adtech_catalog.monitoring.baseline_results")
        print("Results saved to: adtech_catalog.monitoring.baseline_results")
    else:
        print("No valid results to save")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 15. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Baseline Models - All Targets",
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
# 16. SUMMARY
# ============================================================

print("\n" + "="*70)
print("BASELINE MODELS COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Data:
   - Training: {len(train_df):,} rows
   - Test: {len(test_df):,} rows

Key Baseline Metrics:
   - CTR R2: {ctr_results['r2']:.4f}
   - Conversion R2: {conversion_results['r2']:.4f}
   - DED Score R2: {ded_results['r2']:.4f}
   - Cost Efficiency R2: {cost_efficiency_results['r2']:.4f}
""")

if roas_results:
    print(f"   - ROAS R2: {roas_results['r2']:.4f}")
else:
    print("   - ROAS: SKIPPED (not enough valid samples)")

print(f"   - High Performance Accuracy: {hp_results['accuracy']:.4f} (Dummy Baseline)")

print("""
Next Steps:
   1. Run: 03_Train_Regression_Models.py
   2. Run: 04_Train_Classification_Models.py
======================================================================
""")

print("")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 02_Baseline_Models
# ============================================================
# Purpose: Establish baseline performance for all 7 ML targets

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("BASELINE MODELS")
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
    
    # Convert high_performance to int
    if 'high_performance' in train_df.columns:
        train_df['high_performance'] = train_df['high_performance'].astype(int)
        test_df['high_performance'] = test_df['high_performance'].astype(int)
    
    # Convert other numeric columns
    numeric_cols = ['ctr', 'roas', 'conversion_rate', 'avg_ded_score', 
                    'cost_efficiency_score', 'cost_per_click', 'ad_video_length']
    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # Verify class distribution
    train_hp = train_df['high_performance'].value_counts()
    print(f"\nTraining class distribution:")
    print(f"  Class 0: {train_hp.get(0, 0)}")
    print(f"  Class 1: {train_hp.get(1, 0)}")
    
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

y_hp_train = train_df['high_performance'].values
y_hp_test = test_df['high_performance'].values

print("Targets prepared")

# ============================================================
# 6. EVALUATION FUNCTIONS
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

# ADDED: Missing function that was causing the error
def evaluate_classification_binary(y_true, y_pred, y_prob, model_name, target_name):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "target": target_name,
        "model": model_name,
        "auc": auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# ============================================================
# 7. MODEL 1: CTR PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 1: CTR PREDICTION (Baseline)")
print("="*70)

ctr_model = LinearRegression()
ctr_model.fit(X_train_scaled, y_ctr_train)
y_ctr_pred = ctr_model.predict(X_test_scaled)
ctr_results = evaluate_regression(y_ctr_test, y_ctr_pred, "LinearRegression", "CTR")
print(f"  R2: {ctr_results['r2']:.4f}, RMSE: {ctr_results['rmse']:.4f}, MAE: {ctr_results['mae']:.4f}")

# ============================================================
# 8. MODEL 2: ROAS PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 2: ROAS PREDICTION (Baseline)")
print("="*70)

roas_mask_train = y_roas_train > 0
roas_mask_test = y_roas_test > 0

if np.sum(roas_mask_train) > 10 and np.sum(roas_mask_test) > 0:
    X_roas_train = X_train_scaled[roas_mask_train]
    X_roas_test = X_test_scaled[roas_mask_test]
    y_roas_train_filtered = y_roas_train[roas_mask_train]
    y_roas_test_filtered = y_roas_test[roas_mask_test]
    
    roas_model = LinearRegression()
    roas_model.fit(X_roas_train, y_roas_train_filtered)
    y_roas_pred = roas_model.predict(X_roas_test)
    roas_results = evaluate_regression(y_roas_test_filtered, y_roas_pred, "LinearRegression", "ROAS")
    
    print(f"  R2: {roas_results['r2']:.4f}, RMSE: {roas_results['rmse']:.4f}, MAE: {roas_results['mae']:.4f}")
else:
    print("  Not enough valid ROAS samples for baseline")
    roas_results = None

# ============================================================
# 9. MODEL 3: CONVERSION RATE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 3: CONVERSION RATE PREDICTION (Baseline)")
print("="*70)

conversion_model = LinearRegression()
conversion_model.fit(X_train_scaled, y_conversion_train)
y_conversion_pred = conversion_model.predict(X_test_scaled)
conversion_results = evaluate_regression(y_conversion_test, y_conversion_pred, "LinearRegression", "ConversionRate")
print(f"  R2: {conversion_results['r2']:.4f}, RMSE: {conversion_results['rmse']:.4f}, MAE: {conversion_results['mae']:.4f}")

# ============================================================
# 10. MODEL 4: DED SCORE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 4: DED SCORE PREDICTION (Baseline)")
print("="*70)

ded_model = LinearRegression()
ded_model.fit(X_train_scaled, y_ded_train)
y_ded_pred = ded_model.predict(X_test_scaled)
ded_results = evaluate_regression(y_ded_test, y_ded_pred, "LinearRegression", "DEDScore")
print(f"  R2: {ded_results['r2']:.4f}, RMSE: {ded_results['rmse']:.4f}, MAE: {ded_results['mae']:.4f}")

# ============================================================
# 11. MODEL 5: COST EFFICIENCY SCORE PREDICTION
# ============================================================

print("\n" + "="*70)
print("MODEL 5: COST EFFICIENCY SCORE PREDICTION (Baseline)")
print("="*70)

cost_efficiency_model = LinearRegression()
cost_efficiency_model.fit(X_train_scaled, y_cost_efficiency_train)
y_cost_efficiency_pred = cost_efficiency_model.predict(X_test_scaled)
cost_efficiency_results = evaluate_regression(y_cost_efficiency_test, y_cost_efficiency_pred, "LinearRegression", "CostEfficiency")
print(f"  R2: {cost_efficiency_results['r2']:.4f}, RMSE: {cost_efficiency_results['rmse']:.4f}, MAE: {cost_efficiency_results['mae']:.4f}")

# ============================================================
# 12. MODEL 6: HIGH PERFORMANCE CLASSIFICATION (Baseline)
# ============================================================

print("\n" + "="*70)
print("MODEL 6: HIGH PERFORMANCE CLASSIFICATION (Baseline)")
print("="*70)

# Check class distribution
unique_train = np.unique(y_hp_train)
unique_test = np.unique(y_hp_test)

print(f"  Unique classes in training: {unique_train}")
print(f"  Unique classes in test: {unique_test}")

if len(unique_train) < 2:
    print("  ERROR: Training data has only one class!")
    print("  Using Dummy Classifier as fallback.")
    
    dummy_model = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
    dummy_model.fit(X_train_scaled, y_hp_train)
    y_hp_pred = dummy_model.predict(X_test_scaled)
    y_hp_prob = dummy_model.predict_proba(X_test_scaled)[:, 1]
    
    hp_results = {
        "target": "HighPerformance",
        "model": "DummyClassifier",
        "auc": 0.5,
        "accuracy": accuracy_score(y_hp_test, y_hp_pred),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    
    print(f"\n  Dummy Classifier Results (Fallback):")
    print(f"  Accuracy: {hp_results['accuracy']:.4f}")
    
else:
    print(f"  Training - Class 0: {np.sum(y_hp_train == 0)}")
    print(f"  Training - Class 1: {np.sum(y_hp_train == 1)}")
    print(f"  Test - Class 0: {np.sum(y_hp_test == 0)}")
    print(f"  Test - Class 1: {np.sum(y_hp_test == 1)}")
    
    # RandomForest with class_weight='balanced'
    hp_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=5,
        random_state=RANDOM_SEED
    )
    hp_model.fit(X_train_scaled, y_hp_train)
    
    y_hp_pred = hp_model.predict(X_test_scaled)
    y_hp_prob = hp_model.predict_proba(X_test_scaled)[:, 1]
    
    # Now this function exists!
    hp_results = evaluate_classification_binary(
        y_hp_test, y_hp_pred, y_hp_prob,
        "RandomForest_balanced", "HighPerformance"
    )
    
    print(f"\n  Results:")
    print(f"  AUC:      {hp_results['auc']:.4f}")
    print(f"  Accuracy: {hp_results['accuracy']:.4f}")
    print(f"  Precision:{hp_results['precision']:.4f}")
    print(f"  Recall:   {hp_results['recall']:.4f}")
    print(f"  F1:       {hp_results['f1']:.4f}")
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': hp_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 3 Features:")
    for i, row in importance.head(3).iterrows():
        print(f"    {row['feature']}: {row['importance']:.3f}")

# ============================================================
# 13. COMPILE RESULTS
# ============================================================

print("\n" + "="*70)
print("BASELINE RESULTS SUMMARY")
print("="*70)

all_results = [ctr_results, conversion_results, ded_results, cost_efficiency_results, hp_results]

if roas_results:
    all_results.append(roas_results)

for res in all_results:
    if res:
        print(f"\n{res['target']} ({res['model']}):")
        if 'r2' in res:
            print(f"  R2: {res['r2']:.4f}, RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}")
        elif 'accuracy' in res:
            print(f"  Accuracy: {res['accuracy']:.4f}, AUC: {res['auc']:.4f}")

# ============================================================
# 13.5 FORMATTED SUMMARY TABLE (ADD THIS)
# ============================================================

print("\n" + "="*70)
print("BASELINE PERFORMANCE SUMMARY TABLE")
print("="*70)

summary_data = []

for res in all_results:
    if res:
        row = {
            "Target": res.get("target", "N/A"),
            "Model": res.get("model", "N/A"),
            "Type": "Regression" if "r2" in res else "Classification",
        }
        
        if "r2" in res:
            row["R2"] = f"{res.get('r2', 0):.4f}"
            row["RMSE"] = f"{res.get('rmse', 0):.4f}"
            row["MAE"] = f"{res.get('mae', 0):.4f}"
            row["AUC"] = "-"
            row["Accuracy"] = "-"
        else:
            row["R2"] = "-"
            row["RMSE"] = "-"
            row["MAE"] = "-"
            row["AUC"] = f"{res.get('auc', 0):.4f}"
            row["Accuracy"] = f"{res.get('accuracy', 0):.4f}"
        
        summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))


# ============================================================
# 14. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    valid_results = [r for r in all_results if r is not None]
    
    if valid_results:
        results_df = pd.DataFrame(valid_results)
        results_df['version'] = VERSION
        results_df['environment'] = ENVIRONMENT
        results_df['training_timestamp'] = datetime.now().isoformat()
        
        results_df.to_csv(volume_path + "baseline_results.csv", index=False)
        print(f"Results saved to: {volume_path}baseline_results.csv")
        
        spark_results = spark.createDataFrame(results_df)
        spark_results.write \
            .mode("overwrite") \
            .format("delta") \
            .saveAsTable("adtech_catalog.monitoring.baseline_results")
        print("Results saved to: adtech_catalog.monitoring.baseline_results")
    else:
        print("No valid results to save")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 15. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Baseline Models - All Targets",
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
# 16. SUMMARY
# ============================================================

print("\n" + "="*70)
print("BASELINE MODELS COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Data:
   - Training: {len(train_df):,} rows
   - Test: {len(test_df):,} rows

Key Baseline Metrics:
   - CTR R2: {ctr_results['r2']:.4f}
   - Conversion R2: {conversion_results['r2']:.4f}
   - DED Score R2: {ded_results['r2']:.4f}
   - Cost Efficiency R2: {cost_efficiency_results['r2']:.4f}
""")

if roas_results:
    print(f"   - ROAS R2: {roas_results['r2']:.4f}")
else:
    print("   - ROAS: SKIPPED (not enough valid samples)")

if hp_results:
    print(f"   - High Performance AUC: {hp_results['auc']:.4f}")
    print(f"   - High Performance Accuracy: {hp_results['accuracy']:.4f}")
else:
    print("   - High Performance: SKIPPED (only one class in training)")

print("""
Next Steps:
   1. Run: 03_Train_Regression_Models.py
   2. Run: 04_Train_Classification_Models.py
======================================================================
""")

print("")