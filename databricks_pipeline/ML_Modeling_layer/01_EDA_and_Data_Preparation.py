# Databricks notebook source
# ============================================================
# DELETE OLD SPLIT FILES
# ============================================================

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

try:
    dbutils.fs.rm(volume_path + "train_split.csv")
    print("Deleted: train_split.csv")
except Exception as e:
    print(f"train_split.csv not found or already deleted: {e}")

try:
    dbutils.fs.rm(volume_path + "test_split.csv")
    print("Deleted: test_split.csv")
except Exception as e:
    print(f"test_split.csv not found or already deleted: {e}")

try:
    dbutils.fs.rm(volume_path + "split_metadata.csv")
    print("Deleted: split_metadata.csv")
except Exception as e:
    print(f"split_metadata.csv not found or already deleted: {e}")

print("\nOld split files deleted. Now re-run 01_EDA_and_Data_Preparation.py")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 01_EDA_and_Data_Preparation
# ============================================================
# Purpose: Exploratory data analysis and train/test split
# Author: Sanju
# Team: 5 Members
# Date: 2026-07-31
#
# This notebook performs EDA and creates a fixed train/test split
# that will be reused across all ML models.
#


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("EDA AND DATA PREPARATION")
print("="*70)

# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

def load_yaml_config():
    """Load pipeline configuration from YAML file"""
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

# Split ratio from config or default
SPLIT_RATIO = config.get('ml_pipeline', {}).get('test_split_ratio', 0.2) if config else 0.2
RANDOM_SEED = config.get('ml_pipeline', {}).get('random_seed', 42) if config else 42

print("="*70)
print("CONFIGURATION SUMMARY")
print("="*70)
print(f"Environment: {ENVIRONMENT}")
print(f"Version: {VERSION}")
print(f"Git Commit: {GIT_COMMIT}")
print(f"Split Ratio: {SPLIT_RATIO}")
print(f"Random Seed: {RANDOM_SEED}")
print("="*70)

# ============================================================
# 2. LOAD GOLD DATA WITH IDEMPOTENCY CHECK
# ============================================================

print("\nLOADING GOLD DATA...")

def check_table_exists(table_name):
    try:
        spark.sql(f"DESCRIBE {table_name}")
        return True
    except:
        return False

def check_s3_path(s3_path):
    try:
        dbutils.fs.ls(s3_path)
        return True
    except:
        return False

GOLD_TABLE = "adtech_catalog.gold.fact_ad_performance"
S3_PATH = "s3://adtech-optimizer-data/gold/fact_ad_performance/"

df_gold = None

# Try S3 first (if available)
if check_s3_path(S3_PATH):
    print(f"Loading from S3: {S3_PATH}")
    df_spark = spark.read.parquet(S3_PATH)
    df_gold = df_spark.toPandas()
    print(f"Loaded {len(df_gold):,} rows from S3")
else:
    print(f"S3 path not found, trying table: {GOLD_TABLE}")
    if check_table_exists(GOLD_TABLE):
        df_spark = spark.table(GOLD_TABLE)
        df_gold = df_spark.toPandas()
        print(f"Loaded {len(df_gold):,} rows from table")
    else:
        print(f"ERROR: Table {GOLD_TABLE} does not exist.")
        print("Please run 04_FEATURE_ENGINEERING.py first.")
        dbutils.notebook.exit("Gold table not found")

if df_gold is None or len(df_gold) == 0:
    print("ERROR: No data loaded.")
    dbutils.notebook.exit("No data available")

# Convert timestamp columns to string to avoid Arrow conversion issues
timestamp_cols = ['ingestion_timestamp', 'processing_date']
for col in timestamp_cols:
    if col in df_gold.columns:
        df_gold[col] = df_gold[col].astype(str) if col == 'ingestion_timestamp' else pd.to_datetime(df_gold[col])

# Fill nulls
df_gold = df_gold.fillna(0)

print(f"Data shape: {df_gold.shape}")
print(f"Columns: {len(df_gold.columns)}")

# ============================================================
# 3. DATA OVERVIEW
# ============================================================

print("\n" + "="*70)
print("DATA OVERVIEW")
print("="*70)

print("\nFirst 5 rows:")
display(df_gold.head(5))

print("\nData types:")
print(df_gold.dtypes.value_counts())

print("\nBasic statistics:")
numeric_cols_for_stats = df_gold.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols_for_stats:
    display(df_gold[numeric_cols_for_stats].describe())

# ============================================================
# 4. TARGET DISTRIBUTIONS
# ============================================================

print("\n" + "="*70)
print("TARGET DISTRIBUTIONS")
print("="*70)

targets = ['ctr', 'roas', 'conversion_rate', 'high_performance']

for target in targets:
    if target in df_gold.columns:
        print(f"\n{target.upper()} Distribution:")
        print(f"  Count: {df_gold[target].count():,}")
        print(f"  Mean: {df_gold[target].mean():.4f}")
        print(f"  Min: {df_gold[target].min():.4f}")
        print(f"  Max: {df_gold[target].max():.4f}")
        print(f"  Std: {df_gold[target].std():.4f}")

# Check class balance for high_performance
if 'high_performance' in df_gold.columns:
    hp_counts = df_gold['high_performance'].value_counts()
    total = len(df_gold)
    print(f"\nHigh Performance Class Balance:")
    print(f"  0 (Not Profitable): {hp_counts.get(0, 0):,} ({hp_counts.get(0, 0)/total*100:.1f}%)")
    print(f"  1 (Profitable):     {hp_counts.get(1, 0):,} ({hp_counts.get(1, 0)/total*100:.1f}%)")
    
    # Class Imbalance Recommendations
    print("\n" + "-"*50)
    print("CLASS IMBALANCE RECOMMENDATIONS")
    print("-"*50)
    imbalance_ratio = hp_counts.get(0, 0) / hp_counts.get(1, 1) if hp_counts.get(1, 0) > 0 else 0
    print(f"  Imbalance Ratio: {imbalance_ratio:.1f}:1")
    print("  Recommended approaches for classification models:")
    print("    1. Use class_weight='balanced' in RandomForest/GradientBoosting")
    print("    2. Use scale_pos_weight in XGBoost")
    print("    3. Use SMOTE for oversampling (with caution on 1,000 rows)")
    print("    4. Use StratifiedKFold for cross-validation")

# ============================================================
# 5. TRAIN/TEST SPLIT (STRATIFIED WITH GUARANTEE)
# ============================================================

print("\n" + "="*70)
print("TRAIN/TEST SPLIT")
print("="*70)

# Check class distribution before split
hp_counts = df_gold['high_performance'].value_counts()
print(f"Class distribution in full dataset:")
print(f"  0 (Not Profitable): {hp_counts.get(0, 0):,}")
print(f"  1 (Profitable):     {hp_counts.get(1, 0):,}")

def ensure_both_classes_in_train(df, test_size=0.2, max_attempts=20):
    """
    Find a random split that ensures both classes appear in training.
    If not found after max_attempts, manually move one sample from test to train.
    """
    stratify_col = df['high_performance']
    
    for attempt in range(max_attempts):
        seed = RANDOM_SEED + attempt
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_col
        )
        
        # Check if both classes exist in training
        if len(np.unique(train_df['high_performance'])) == 2:
            print(f"  Valid split found with random_seed={seed}")
            return train_df, test_df, seed
    
    # If we reach here, no valid split was found
    print("  WARNING: Could not find a valid split with both classes.")
    print("  Manually ensuring both classes in training...")
    
    # Try one more split without stratify
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED
    )
    
    # Check if both classes exist in training
    if len(np.unique(train_df['high_performance'])) == 2:
        print(f"  Valid split found without stratify")
        return train_df, test_df, RANDOM_SEED
    
    # Manual fix: Move one profitable sample from test to train
    profitable_indices = df[df['high_performance'] == 1].index.tolist()
    if len(profitable_indices) > 0:
        # Take one profitable ad
        move_idx = profitable_indices[0]
        train_df = pd.concat([train_df, df.loc[[move_idx]]])
        test_df = test_df[test_df.index != move_idx]
        print(f"  Manually moved 1 profitable ad from test to training.")
        print(f"  New training size: {len(train_df):,}")
        print(f"  New test size: {len(test_df):,}")
    
    return train_df, test_df, RANDOM_SEED

train_df, test_df, used_seed = ensure_both_classes_in_train(df_gold, SPLIT_RATIO)

print(f"\nSplit Summary:")
print(f"  Used Random Seed: {used_seed}")
print(f"  Training rows: {len(train_df):,} ({len(train_df)/len(df_gold)*100:.1f}%)")
print(f"  Test rows: {len(test_df):,} ({len(test_df)/len(df_gold)*100:.1f}%)")

# Verify class balance in split
if 'high_performance' in df_gold.columns:
    train_hp = train_df['high_performance'].value_counts()
    test_hp = test_df['high_performance'].value_counts()
    
    print(f"\nClass balance in Training:")
    print(f"  0 (Not Profitable): {train_hp.get(0, 0):,} ({train_hp.get(0, 0)/len(train_df)*100:.1f}%)")
    print(f"  1 (Profitable):     {train_hp.get(1, 0):,} ({train_hp.get(1, 0)/len(train_df)*100:.1f}%)")
    
    print(f"\nClass balance in Test:")
    print(f"  0 (Not Profitable): {test_hp.get(0, 0):,} ({test_hp.get(0, 0)/len(test_df)*100:.1f}%)")
    print(f"  1 (Profitable):     {test_hp.get(1, 0):,} ({test_hp.get(1, 0)/len(test_df)*100:.1f}%)")
    
    # Verify both classes exist in training
    if len(np.unique(train_df['high_performance'])) < 2:
        print("\n  ERROR: Training set still has only one class!")
        print("  Manual intervention required.")
    else:
        print("\n  BOTH CLASSES PRESENT IN TRAINING SET")

# ============================================================
# 6. SAVE SPLIT INDICES
# ============================================================

print("\nSAVING SPLIT INDICES...")

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

# DELETE OLD FILES FIRST
try:
    dbutils.fs.rm(volume_path + "train_split.csv")
    print("Removed old train_split.csv")
except:
    pass

try:
    dbutils.fs.rm(volume_path + "test_split.csv")
    print("Removed old test_split.csv")
except:
    pass

# Create a copy without timestamp columns for saving
train_save = train_df.copy()
test_save = test_df.copy()

# Drop problematic timestamp columns before saving
for col in ['ingestion_timestamp', 'processing_date']:
    if col in train_save.columns:
        train_save = train_save.drop(columns=[col])
    if col in test_save.columns:
        test_save = test_save.drop(columns=[col])

# Add split indicator column
train_save['split'] = 'train'
test_save['split'] = 'test'

try:
    train_save.to_csv(volume_path + "train_split.csv", index=False)
    test_save.to_csv(volume_path + "test_split.csv", index=False)
    print(f"Training split saved to: {volume_path}train_split.csv")
    print(f"Test split saved to: {volume_path}test_split.csv")
except Exception as e:
    print(f"Error saving to Volume: {e}")

# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

print("\n" + "="*70)
print("CORRELATION ANALYSIS")
print("="*70)

# Select numeric columns for correlation
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Limit to key columns for readability
key_features = ['ctr', 'roas', 'conversion_rate', 'high_performance',
                'cost_per_click', 'total_impressions', 'total_clicks',
                'avg_watch_ratio', 'avg_ded_score']

available_features = [col for col in key_features if col in numeric_cols]

if len(available_features) > 1:
    corr_matrix = train_df[available_features].corr()
    print("\nCorrelation Matrix:")
    display(corr_matrix)

    # Print top correlations with target
    for target in ['ctr', 'roas', 'conversion_rate']:
        if target in corr_matrix.columns:
            print(f"\nTop correlations with {target}:")
            corr_target = corr_matrix[target].sort_values(ascending=False)
            for feature, corr in corr_target.items():
                if feature != target:
                    print(f"  {feature}: {corr:.3f}")

# ============================================================
# 8. CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n" + "="*70)
print("CATEGORICAL FEATURE ANALYSIS")
print("="*70)

categorical_cols = ['ad_category', 'ad_device', 'ad_type', 'ad_location']

for col in categorical_cols:
    if col in train_df.columns:
        print(f"\n{col.upper()} Distribution:")
        value_counts = train_df[col].value_counts()
        print(value_counts.to_string())

# ============================================================
# 9. PRE-LAUNCH FEATURES (FOR ML)
# ============================================================

print("\n" + "="*70)
print("PRE-LAUNCH FEATURES")
print("="*70)

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

print("Features available before ad launch:")
for i, feat in enumerate(pre_launch_features, 1):
    print(f"  {i}. {feat}")

print(f"\nTotal pre-launch features: {len(pre_launch_features)}")

# ============================================================
# 10. SAVE VERSION HISTORY
# ============================================================

print("\nSAVING VERSION HISTORY...")

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "EDA and Data Preparation",
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
# 11. SUMMARY
# ============================================================

print("\n" + "="*70)
print("EDA AND DATA PREPARATION COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Data:
   - Total Rows: {len(df_gold):,}
   - Total Columns: {len(df_gold.columns)}
   - Pre-launch Features: {len(pre_launch_features)}

Split:
   - Method: Random Stratified (guaranteed both classes in training)
   - Used Random Seed: {used_seed}
   - Training: {len(train_df):,} rows ({len(train_df)/len(df_gold)*100:.1f}%)
   - Test: {len(test_df):,} rows ({len(test_df)/len(df_gold)*100:.1f}%)

Class Balance:
   - Profitable Ads (Train): {train_df['high_performance'].mean()*100:.1f}%
   - Profitable Ads (Test): {test_df['high_performance'].mean()*100:.1f}%

Saved Files:
   - {volume_path}train_split.csv
   - {volume_path}test_split.csv
   - {volume_path}split_metadata.csv

Next Steps:
   1. Run: 02_Baseline_Models.py
   2. Run: 09_Analytics_Only.py
======================================================================
""")

print("")

# COMMAND ----------

# ============================================================
# VERIFY NEW SPLIT FILES
# ============================================================

import pandas as pd

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

train_df = pd.read_csv(volume_path + "train_split.csv")
test_df = pd.read_csv(volume_path + "test_split.csv")

print("="*70)
print("VERIFYING SPLIT FILES")
print("="*70)

print(f"Training rows: {len(train_df):,}")
print(f"Test rows: {len(test_df):,}")

train_hp = train_df['high_performance'].value_counts()
test_hp = test_df['high_performance'].value_counts()

print(f"\nTraining class distribution:")
print(f"  Class 0: {train_hp.get(0, 0)}")
print(f"  Class 1: {train_hp.get(1, 0)}")

print(f"\nTest class distribution:")
print(f"  Class 0: {test_hp.get(0, 0)}")
print(f"  Class 1: {test_hp.get(1, 0)}")

if train_hp.get(1, 0) > 0:
    print("\n  BOTH CLASSES PRESENT IN TRAINING SET ")
else:
    print("\n  ERROR: Training set has ONLY class 0 ")