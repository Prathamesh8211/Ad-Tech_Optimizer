# Databricks notebook source
# Databricks notebook source
# ============================================================
# 06_Model_Comparison_and_Selection
# ============================================================
# Purpose: Compare all models and select best for each target


import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("MODEL COMPARISON AND SELECTION")
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

print("="*70)
print("CONFIGURATION SUMMARY")
print("="*70)
print(f"Environment: {ENVIRONMENT}")
print(f"Version: {VERSION}")
print(f"Git Commit: {GIT_COMMIT}")
print("="*70)

# ============================================================
# 2. LOAD REGRESSION RESULTS
# ============================================================

print("\nLOADING REGRESSION RESULTS...")

volume_path = "/Volumes/adtech_catalog/bronze/landing_zone/"

try:
    reg_results = pd.read_csv(volume_path + "regression_advanced_results.csv")
    print(f"  Loaded {len(reg_results)} regression results")
    print(f"  Targets: {reg_results['Target'].unique().tolist()}")
except Exception as e:
    print(f"  Error loading regression results: {e}")
    reg_results = None

# ============================================================
# 3. LOAD CLASSIFICATION RESULTS
# ============================================================

print("\nLOADING CLASSIFICATION RESULTS...")

try:
    cls_results = pd.read_csv(volume_path + "classification_binary_results.csv")
    print(f"  Loaded {len(cls_results)} classification results")
    print(f"  Target: {cls_results['target'].unique().tolist() if 'target' in cls_results.columns else 'HighPerformance'}")
except Exception as e:
    print(f"  Error loading classification results: {e}")
    cls_results = None

# ============================================================
# 4. PROCESS REGRESSION RESULTS
# ============================================================

print("\n" + "="*70)
print("REGRESSION MODEL COMPARISON")
print("="*70)

if reg_results is not None:
    # Find best model for each target
    best_reg_models = {}
    for target in reg_results['Target'].unique():
        target_df = reg_results[reg_results['Target'] == target]
        best_idx = target_df['R2'].idxmax()
        best_model = target_df.loc[best_idx]
        best_reg_models[target] = {
            'model': best_model['Model'],
            'r2': best_model['R2'],
            'rmse': best_model['RMSE'],
            'mae': best_model['MAE']
        }

    # Print regression summary
    print("\n  REGRESSION SUMMARY:")
    print("-"*70)
    print(f"  {'Target':20} {'Best Model':20} {'R2':10} {'RMSE':10} {'MAE':10}")
    print("-"*70)
    for target, info in best_reg_models.items():
        print(f"  {target:20} {info['model']:20} {info['r2']:.4f}   {info['rmse']:.4f}   {info['mae']:.4f}")

# ============================================================
# 5. PROCESS CLASSIFICATION RESULTS
# ============================================================

print("\n" + "="*70)
print("CLASSIFICATION MODEL COMPARISON")
print("="*70)

if cls_results is not None:
    # Find best model
    if 'auc' in cls_results.columns:
        best_idx = cls_results['auc'].idxmax()
        best_model = cls_results.loc[best_idx]
        best_cls_model = {
            'model': best_model['model'],
            'auc': best_model['auc'],
            'accuracy': best_model['accuracy'],
            'precision': best_model['precision'],
            'recall': best_model['recall'],
            'f1': best_model['f1']
        }

        print("\n  CLASSIFICATION SUMMARY:")
        print("-"*70)
        print(f"  {'Target':20} {'Best Model':20} {'AUC':10} {'Accuracy':10} {'F1':10}")
        print("-"*70)
        print(f"  {'HighPerformance':20} {best_cls_model['model']:20} {best_cls_model['auc']:.4f}   {best_cls_model['accuracy']:.4f}   {best_cls_model['f1']:.4f}")
    else:
        print("  No classification results found")

# ============================================================
# 6. CREATE FINAL SELECTION TABLE
# ============================================================

print("\n" + "="*70)
print("FINAL MODEL SELECTION")
print("="*70)

selection_data = []

# Add regression models
if reg_results is not None:
    for target, info in best_reg_models.items():
        selection_data.append({
            'Target': target,
            'Model': info['model'],
            'Metric': 'R2',
            'Value': info['r2']
        })

# Add classification models
if cls_results is not None and 'auc' in cls_results.columns:
    selection_data.append({
        'Target': 'HighPerformance',
        'Model': best_cls_model['model'],
        'Metric': 'AUC',
        'Value': best_cls_model['auc']
    })

selection_df = pd.DataFrame(selection_data)

if not selection_df.empty:
    print("\n  SELECTED MODELS:")
    print("-"*60)
    print(f"  {'Target':20} {'Model':25} {'Metric':10} {'Value':10}")
    print("-"*60)
    for _, row in selection_df.iterrows():
        print(f"  {row['Target']:20} {row['Model']:25} {row['Metric']:10} {row['Value']:.4f}")

# ============================================================
# 7. SAVE SELECTION RESULTS
# ============================================================

print("\nSAVING SELECTION RESULTS...")

try:
    selection_df.to_csv(volume_path + "model_selection_results.csv", index=False)
    print(f"  Results saved to: {volume_path}model_selection_results.csv")
    
    spark_df = spark.createDataFrame(selection_df)
    spark_df.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.model_selection_results")
    print("  Results saved to: adtech_catalog.monitoring.model_selection_results")
    
except Exception as e:
    print(f"  Error saving results: {e}")

# ============================================================
# 8. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Model Comparison and Selection",
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
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("MODEL COMPARISON COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Selected Models for Deployment:
""")

if not selection_df.empty:
    for _, row in selection_df.iterrows():
        status = "GOOD" if row['Value'] > 0 else "POOR"
        print(f"   {row['Target']}: {row['Model']} ({row['Metric']}: {row['Value']:.4f}) - {status}")

print("""
Next Steps:
   1. Deploy selected models to production
   2. Build Streamlit Dashboard
   3. Monitor model performance
======================================================================
""")

print("")