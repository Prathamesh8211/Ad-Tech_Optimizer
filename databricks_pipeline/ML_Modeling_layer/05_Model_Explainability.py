# Databricks notebook source
pip install xgboost lightgbm imbalanced-learn

# COMMAND ----------

pip install shap

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 05_Model_Explainability
# ============================================================
# Purpose: Explain model predictions using SHAP

# This notebook explains the best models using SHAP:
# 1. LightGBM (Best for High Performance Classification)
# 2. LightGBM (Best for Conversion Rate Regression)
# 3. RandomForest (Best for Cost Efficiency Regression)

# SHAP (SHapley Additive exPlanations) shows:
# - Which features drive predictions
# - How each feature affects the prediction
# - Feature importance for each model

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("MODEL EXPLAINABILITY - SHAP ANALYSIS")
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

base_feature_cols = [c for c in pre_launch_features if c not in categorical_cols] + [c + "_encoded" for c in categorical_cols]

train_encoded['device_type_interaction'] = train_encoded['ad_device_encoded'] * train_encoded['ad_type_encoded']
test_encoded['device_type_interaction'] = test_encoded['ad_device_encoded'] * test_encoded['ad_type_encoded']

train_encoded['category_type_interaction'] = train_encoded['ad_category_encoded'] * train_encoded['ad_type_encoded']
test_encoded['category_type_interaction'] = test_encoded['ad_category_encoded'] * test_encoded['ad_type_encoded']

train_encoded['cost_video_interaction'] = train_encoded['cost_per_click'] * train_encoded['ad_video_length']
test_encoded['cost_video_interaction'] = test_encoded['cost_per_click'] * test_encoded['ad_video_length']

interaction_features = ['device_type_interaction', 'category_type_interaction', 'cost_video_interaction']
feature_cols = base_feature_cols + interaction_features

X_train = train_encoded[feature_cols].values
X_test = test_encoded[feature_cols].values

print(f"Total features: {len(feature_cols)}")
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

# High Performance (Binary Classification)
y_hp_train = train_df['high_performance'].values
y_hp_test = test_df['high_performance'].values

# Conversion Rate (Regression)
y_conversion_train = train_df['conversion_rate'].values
y_conversion_test = test_df['conversion_rate'].values

# Cost Efficiency (Regression)
y_cost_efficiency_train = train_df['cost_efficiency_score'].values
y_cost_efficiency_test = test_df['cost_efficiency_score'].values

print("Targets prepared")

# ============================================================
# 6. TRAIN BEST MODELS
# ============================================================

print("\nTRAINING BEST MODELS...")

# 6.1 High Performance Classification (LightGBM)
print("\n  Training LightGBM for High Performance...")
imbalance_ratio = np.sum(y_hp_train == 0) / np.sum(y_hp_train == 1) if np.sum(y_hp_train == 1) > 0 else 1

hp_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    scale_pos_weight=imbalance_ratio,
    random_state=RANDOM_SEED,
    verbosity=-1
)
hp_model.fit(X_train_scaled, y_hp_train)
print("    High Performance model trained")

# 6.2 Conversion Rate Regression (LightGBM)
print("\n  Training LightGBM for Conversion Rate...")
conversion_model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=RANDOM_SEED,
    verbosity=-1
)
conversion_model.fit(X_train_scaled, y_conversion_train)
print("    Conversion Rate model trained")

# 6.3 Cost Efficiency Regression (RandomForest)
print("\n  Training RandomForest for Cost Efficiency...")
cost_efficiency_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=RANDOM_SEED
)
cost_efficiency_model.fit(X_train_scaled, y_cost_efficiency_train)
print("    Cost Efficiency model trained")

# ============================================================
# 7. SHAP ANALYSIS - HIGH PERFORMANCE CLASSIFICATION
# ============================================================

print("\n" + "="*70)
print("SHAP ANALYSIS: HIGH PERFORMANCE CLASSIFICATION")
print("="*70)

# Create SHAP explainer for LightGBM
explainer_hp = shap.TreeExplainer(hp_model)
shap_values_hp = explainer_hp.shap_values(X_test_scaled)

# Feature importance plot
print("\n  Feature Importance (Mean |SHAP|):")
feature_importance_hp = np.abs(shap_values_hp).mean(axis=0)
importance_df_hp = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance_hp
}).sort_values('importance', ascending=False)

for i, row in importance_df_hp.head(10).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# Summary plot
print("\n  Generating SHAP Summary Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_hp, X_test_scaled, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot - High Performance Classification")
plt.tight_layout()

# Save to Volume
plt.savefig(volume_path + "shap_summary_hp.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved to: {volume_path}shap_summary_hp.png")

# ============================================================
# 8. SHAP ANALYSIS - CONVERSION RATE
# ============================================================

print("\n" + "="*70)
print("SHAP ANALYSIS: CONVERSION RATE PREDICTION")
print("="*70)

# Create SHAP explainer for LightGBM
explainer_conversion = shap.TreeExplainer(conversion_model)
shap_values_conversion = explainer_conversion.shap_values(X_test_scaled)

# Feature importance plot
print("\n  Feature Importance (Mean |SHAP|):")
feature_importance_conversion = np.abs(shap_values_conversion).mean(axis=0)
importance_df_conversion = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance_conversion
}).sort_values('importance', ascending=False)

for i, row in importance_df_conversion.head(10).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# Summary plot
print("\n  Generating SHAP Summary Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_conversion, X_test_scaled, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot - Conversion Rate Prediction")
plt.tight_layout()

plt.savefig(volume_path + "shap_summary_conversion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved to: {volume_path}shap_summary_conversion.png")

# ============================================================
# 9. SHAP ANALYSIS - COST EFFICIENCY
# ============================================================

print("\n" + "="*70)
print("SHAP ANALYSIS: COST EFFICIENCY PREDICTION")
print("="*70)

# Create SHAP explainer for RandomForest
explainer_cost = shap.TreeExplainer(cost_efficiency_model)
shap_values_cost = explainer_cost.shap_values(X_test_scaled)

# Feature importance plot
print("\n  Feature Importance (Mean |SHAP|):")
feature_importance_cost = np.abs(shap_values_cost).mean(axis=0)
importance_df_cost = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance_cost
}).sort_values('importance', ascending=False)

for i, row in importance_df_cost.head(10).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# Summary plot
print("\n  Generating SHAP Summary Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_cost, X_test_scaled, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot - Cost Efficiency Prediction")
plt.tight_layout()

plt.savefig(volume_path + "shap_summary_cost.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved to: {volume_path}shap_summary_cost.png")

# ============================================================
# 10. ADDITIONAL SHAP VISUALIZATIONS
# ============================================================

print("\n" + "="*70)
print("ADDITIONAL SHAP VISUALIZATIONS")
print("="*70)

# 10.1 SHAP Dependence Plots
print("\n  Generating SHAP Dependence Plots...")

for feature in importance_df_hp.head(3)['feature']:
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature, shap_values_hp, X_test_scaled, 
                         feature_names=feature_cols, show=False)
    plt.title(f"SHAP Dependence Plot - {feature}")
    plt.tight_layout()
    plt.savefig(volume_path + f"shap_dependence_{feature}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: shap_dependence_{feature}.png")

# 10.2 SHAP Force Plot for a profitable ad
print("\n  Generating SHAP Force Plot...")

profitable_idx = np.where(y_hp_test == 1)[0][0] if np.sum(y_hp_test == 1) > 0 else 0

plt.figure(figsize=(12, 3))
shap.force_plot(explainer_hp.expected_value, shap_values_hp[profitable_idx], 
                X_test_scaled[profitable_idx], feature_names=feature_cols, 
                matplotlib=True, show=False)
plt.title("SHAP Force Plot - Profitable Ad Prediction")
plt.tight_layout()
plt.savefig(volume_path + "shap_force_profitable.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: shap_force_profitable.png")

# 10.3 SHAP Waterfall Plot
print("\n  Generating SHAP Waterfall Plot...")

plt.figure(figsize=(10, 8))
shap.waterfall_plot(shap.Explanation(values=shap_values_hp[profitable_idx], 
                                      base_values=explainer_hp.expected_value,
                                      data=X_test_scaled[profitable_idx],
                                      feature_names=feature_cols), show=False)
plt.title("SHAP Waterfall Plot - Feature Contributions")
plt.tight_layout()
plt.savefig(volume_path + "shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: shap_waterfall.png")

# 10.4 Consolidated SHAP Summary Table
print("\n  Generating Consolidated SHAP Summary...")

shap_summary = pd.DataFrame({
    'Feature': feature_cols,
    'High Performance': feature_importance_hp,
    'Conversion Rate': feature_importance_conversion,
    'Cost Efficiency': feature_importance_cost
})
shap_summary = shap_summary.sort_values('High Performance', ascending=False)

print("\n  SHAP Importance by Model:")
print(shap_summary.to_string(index=False))

shap_summary.to_csv(volume_path + "shap_summary_consolidated.csv", index=False)
print("\n  Saved: shap_summary_consolidated.csv")

# ============================================================
# 11. SAVE SHAP VALUES
# ============================================================

print("\nSAVING SHAP VALUES...")

try:
    shap_df_hp = pd.DataFrame(shap_values_hp, columns=[f"SHAP_{f}" for f in feature_cols])
    shap_df_hp.to_csv(volume_path + "shap_values_hp.csv", index=False)
    print(f"  SHAP values (HP) saved to: {volume_path}shap_values_hp.csv")
    
    shap_df_conversion = pd.DataFrame(shap_values_conversion, columns=[f"SHAP_{f}" for f in feature_cols])
    shap_df_conversion.to_csv(volume_path + "shap_values_conversion.csv", index=False)
    print(f"  SHAP values (Conversion) saved to: {volume_path}shap_values_conversion.csv")
    
    shap_df_cost = pd.DataFrame(shap_values_cost, columns=[f"SHAP_{f}" for f in feature_cols])
    shap_df_cost.to_csv(volume_path + "shap_values_cost.csv", index=False)
    print(f"  SHAP values (Cost) saved to: {volume_path}shap_values_cost.csv")
    
except Exception as e:
    print(f"  Error saving SHAP values: {e}")

# ============================================================
# 12. SAVE VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Model Explainability - SHAP Analysis",
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
print("MODEL EXPLAINABILITY COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Models Explained:
   1. High Performance Classification (LightGBM)
   2. Conversion Rate Prediction (LightGBM)
   3. Cost Efficiency Prediction (RandomForest)

SHAP Outputs:
   - Summary plots saved to Volume
   - Dependence plots saved to Volume
   - Force plot saved to Volume
   - Waterfall plot saved to Volume
   - SHAP values saved to CSV
   - Consolidated SHAP summary saved to CSV

Key Insights:
""")

print("  High Performance Classification - Top 3 Features:")
for i, row in importance_df_hp.head(3).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

print("\n  Conversion Rate Prediction - Top 3 Features:")
for i, row in importance_df_conversion.head(3).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

print("\n  Cost Efficiency Prediction - Top 3 Features:")
for i, row in importance_df_cost.head(3).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

print("""
Next Steps:
   1. Review SHAP plots in Volume
   2. Present findings to business stakeholders
   3. Deploy models to production
======================================================================
""")

print("")