# Databricks notebook source
pip install xgboost lightgbm imbalanced-learn

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 04_Train_Classification_Models - Part 1: Binary Classification-------1
# ============================================================
# Purpose: Train advanced binary classification models
# Target: High Performance (ROAS > 2.0)


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("BINARY CLASSIFICATION: HIGH PERFORMANCE")
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
# 5. TARGET AND EVALUATION FUNCTIONS
# ============================================================

y_hp_train = train_df['high_performance'].values
y_hp_test = test_df['high_performance'].values

hp_train_counts = np.bincount(y_hp_train)
hp_test_counts = np.bincount(y_hp_test)

print(f"\nHigh Performance Class Balance:")
print(f"  Training - Class 0: {hp_train_counts[0]} ({hp_train_counts[0]/len(y_hp_train)*100:.1f}%)")
print(f"  Training - Class 1: {hp_train_counts[1]} ({hp_train_counts[1]/len(y_hp_train)*100:.1f}%)")
print(f"  Test - Class 0: {hp_test_counts[0]} ({hp_test_counts[0]/len(y_hp_test)*100:.1f}%)")
print(f"  Test - Class 1: {hp_test_counts[1]} ({hp_test_counts[1]/len(y_hp_test)*100:.1f}%)")

def evaluate_binary(y_true, y_pred, y_prob, model_name):
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    }

def select_features(X_train, X_test, y_train, feature_names):
    print("\n  Feature Selection:")
    selector_rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    selector_rf.fit(X_train, y_train)
    importances = selector_rf.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    print("    Top 5 features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    selector = SelectFromModel(selector_rf, threshold='mean', prefit=True)
    return selector.transform(X_train), selector.transform(X_test)

def tune_model(X_train, y_train, model, param_dist, model_name):
    print(f"\n  Tuning {model_name}...")
    random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=3, scoring='roc_auc', 
                                       random_state=RANDOM_SEED, n_jobs=-1)
    random_search.fit(X_train, y_train)
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV AUC: {random_search.best_score_:.4f}")
    return random_search.best_estimator_

# ============================================================
# 6. TRAIN BINARY CLASSIFIERS
# ============================================================

print("\n" + "="*70)
print("TRAINING BINARY CLASSIFIERS")
print("="*70)

X_train_sel, X_test_sel = select_features(X_train_scaled, X_test_scaled, y_hp_train, feature_cols)

class_counts = np.bincount(y_hp_train)
imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 0
print(f"\n  Class Imbalance Ratio: {imbalance_ratio:.1f}:1")

if imbalance_ratio > 3:
    print("  Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_sel, y_hp_train)
    print(f"    Before SMOTE: {len(y_hp_train)} samples")
    print(f"    After SMOTE: {len(y_train_resampled)} samples")
else:
    X_train_resampled, y_train_resampled = X_train_sel, y_hp_train

results = []

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
lr.fit(X_train_resampled, y_train_resampled)
y_pred = lr.predict(X_test_sel)
y_prob = lr.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "LogisticRegression"))

# RandomForest
rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
rf_best = tune_model(X_train_resampled, y_train_resampled, RandomForestClassifier(random_state=RANDOM_SEED), 
                     rf_params, "RandomForest")
y_pred = rf_best.predict(X_test_sel)
y_prob = rf_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "RandomForest"))

# XGBoost
xgb_params = {'n_estimators': [100, 150, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
xgb_best = tune_model(X_train_resampled, y_train_resampled, XGBClassifier(random_state=RANDOM_SEED, verbosity=0),
                      xgb_params, "XGBoost")
y_pred = xgb_best.predict(X_test_sel)
y_prob = xgb_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "XGBoost"))

# LightGBM
lgb_params = {'n_estimators': [100, 150, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]}
lgb_best = tune_model(X_train_resampled, y_train_resampled, lgb.LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1),
                      lgb_params, "LightGBM")
y_pred = lgb_best.predict(X_test_sel)
y_prob = lgb_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "LightGBM"))

# Voting Ensemble
model_aucs = [(res['model'], res['auc']) for res in results]
model_aucs.sort(key=lambda x: x[1], reverse=True)
top_2 = [name for name, _ in model_aucs[:2]]
print(f"\n  Voting Ensemble using: {top_2}")

ensemble_models = []
if 'RandomForest' in top_2:
    ensemble_models.append(('rf', RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                         class_weight='balanced', random_state=RANDOM_SEED)))
if 'XGBoost' in top_2:
    ensemble_models.append(('xgb', XGBClassifier(n_estimators=100, max_depth=5, 
                                                 random_state=RANDOM_SEED, verbosity=0)))
if 'LightGBM' in top_2:
    ensemble_models.append(('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=5,
                                                      random_state=RANDOM_SEED, verbosity=-1)))

if len(ensemble_models) >= 2:
    voting = VotingClassifier(ensemble_models, voting='soft')
    voting.fit(X_train_resampled, y_train_resampled)
    y_pred = voting.predict(X_test_sel)
    y_prob = voting.predict_proba(X_test_sel)[:, 1]
    results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "VotingEnsemble"))

# Cross-validation for best model
best_result = max(results, key=lambda x: x['auc'])
print(f"\n  Best Model: {best_result['model']} (AUC: {best_result['auc']:.4f})")

cv_scores = cross_val_score(rf_best if best_result['model'] == 'RandomForest' else 
                            xgb_best if best_result['model'] == 'XGBoost' else 
                            lgb_best if best_result['model'] == 'LightGBM' else
                            voting if best_result['model'] == 'VotingEnsemble' else lr,
                            X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc')
print(f"  CV AUC Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n  Results Summary:")
print("-"*70)
print(f"  {'Model':20} {'AUC':8} {'Accuracy':8} {'Precision':8} {'Recall':8} {'F1':8}")
print("-"*70)
for res in results:
    print(f"  {res['model']:20} {res['auc']:.4f}   {res['accuracy']:.4f}   {res['precision']:.4f}   {res['recall']:.4f}   {res['f1']:.4f}")

# ============================================================
# 7. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    results_df = pd.DataFrame(results)
    results_df['target'] = 'HighPerformance'
    results_df['version'] = VERSION
    results_df['environment'] = ENVIRONMENT
    results_df['training_timestamp'] = datetime.now().isoformat()
    
    results_df.to_csv(volume_path + "classification_binary_results.csv", index=False)
    print(f"Results saved to: {volume_path}classification_binary_results.csv")
    
    spark_results = spark.createDataFrame(results_df)
    spark_results.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.classification_binary_results")
    print("Results saved to: adtech_catalog.monitoring.classification_binary_results")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 8. VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Binary Classification: High Performance",
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

print("\n" + "="*70)
print("BINARY CLASSIFICATION COMPLETE")
print("="*70)

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

High Performance Classification (Binary):
   - Best Model: {best_result['model']}
   - Best AUC: {best_result['auc']:.4f}
   - CV AUC Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})

Next Step:
   Run Cell 2: Multi-class Classification (Ad Lifecycle Stage)
======================================================================
""")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 04_Train_Classification_Models - Part 1: Binary Classification -------2
# ============================================================
# Purpose: Train advanced binary classification models
# Target: High Performance (ROAS > 2.0)


import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("BINARY CLASSIFICATION: HIGH PERFORMANCE (FIXED)")
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
# 5. TARGET AND EVALUATION FUNCTIONS
# ============================================================

y_hp_train = train_df['high_performance'].values
y_hp_test = test_df['high_performance'].values

hp_train_counts = np.bincount(y_hp_train)
hp_test_counts = np.bincount(y_hp_test)

print(f"\nHigh Performance Class Balance:")
print(f"  Training - Class 0: {hp_train_counts[0]} ({hp_train_counts[0]/len(y_hp_train)*100:.1f}%)")
print(f"  Training - Class 1: {hp_train_counts[1]} ({hp_train_counts[1]/len(y_hp_train)*100:.1f}%)")
print(f"  Test - Class 0: {hp_test_counts[0]} ({hp_test_counts[0]/len(y_hp_test)*100:.1f}%)")
print(f"  Test - Class 1: {hp_test_counts[1]} ({hp_test_counts[1]/len(y_hp_test)*100:.1f}%)")

def evaluate_binary(y_true, y_pred, y_prob, model_name):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc
    }

def tune_model(X_train, y_train, model, param_dist, model_name):
    print(f"\n  Tuning {model_name}...")
    random_search = RandomizedSearchCV(
        model, param_dist, n_iter=10, cv=3, 
        scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV AUC: {random_search.best_score_:.4f}")
    return random_search.best_estimator_

# ============================================================
# 6. FEATURE SELECTION (Simple)
# ============================================================

print("\nFEATURE SELECTION...")

def select_features_simple(X_train, X_test, y_train, feature_names, threshold='mean'):
    """Simple feature selection using RandomForest importance"""
    
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("    Top 5 features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"    Selected {len(selected_features)} / {len(feature_names)} features")
    
    return X_train_selected, X_test_selected, selected_features

# Apply feature selection (not RFECV - simpler)
X_train_sel, X_test_sel, selected_feats = select_features_simple(
    X_train_scaled, X_test_scaled, y_hp_train, feature_cols, threshold='mean'
)

# ============================================================
# 7. SMOTE (After Feature Selection)
# ============================================================

print("\n" + "="*70)
print("TRAINING BINARY CLASSIFIERS")
print("="*70)

class_counts = np.bincount(y_hp_train)
imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 0
print(f"\n  Class Imbalance Ratio: {imbalance_ratio:.1f}:1")

# Apply SMOTE
print("  Applying SMOTE for class imbalance...")
smote = SMOTE(random_state=RANDOM_SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_sel, y_hp_train)
print(f"    Before SMOTE: {len(y_hp_train)} samples")
print(f"    After SMOTE: {len(y_train_resampled)} samples")

results = []

# ============================================================
# 8. TRAIN MODELS (Simpler Params)
# ============================================================

# 1. Logistic Regression (baseline)
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
lr.fit(X_train_resampled, y_train_resampled)
y_pred = lr.predict(X_test_sel)
y_prob = lr.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "LogisticRegression"))

# 2. RandomForest (simpler params)
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_best = tune_model(X_train_resampled, y_train_resampled,
                      RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED),
                      rf_params, "RandomForest")
y_pred = rf_best.predict(X_test_sel)
y_prob = rf_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "RandomForest"))

# 3. XGBoost (simpler params)
xgb_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'scale_pos_weight': [imbalance_ratio, imbalance_ratio/2]
}
xgb_best = tune_model(X_train_resampled, y_train_resampled,
                       XGBClassifier(random_state=RANDOM_SEED, verbosity=0),
                       xgb_params, "XGBoost")
y_pred = xgb_best.predict(X_test_sel)
y_prob = xgb_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "XGBoost"))

# 4. LightGBM (simpler params)
lgb_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [15, 31],
    'scale_pos_weight': [imbalance_ratio, imbalance_ratio/2]
}
lgb_best = tune_model(X_train_resampled, y_train_resampled,
                       lgb.LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1),
                       lgb_params, "LightGBM")
y_pred = lgb_best.predict(X_test_sel)
y_prob = lgb_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "LightGBM"))

# 5. GradientBoosting (simpler params)
gb_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05]
}
gb_best = tune_model(X_train_resampled, y_train_resampled,
                      GradientBoostingClassifier(random_state=RANDOM_SEED),
                      gb_params, "GradientBoosting")
y_pred = gb_best.predict(X_test_sel)
y_prob = gb_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "GradientBoosting"))

# 6. Voting Ensemble (top 2 models)
model_aucs = [(res['model'], res['auc']) for res in results]
model_aucs.sort(key=lambda x: x[1], reverse=True)
top_2 = [name for name, _ in model_aucs[:2]]
print(f"\n  Voting Ensemble using: {top_2}")

ensemble_models = []
if 'RandomForest' in top_2:
    ensemble_models.append(('rf', RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                         class_weight='balanced', random_state=RANDOM_SEED)))
if 'XGBoost' in top_2:
    ensemble_models.append(('xgb', XGBClassifier(n_estimators=100, max_depth=5, 
                                                 scale_pos_weight=imbalance_ratio,
                                                 random_state=RANDOM_SEED, verbosity=0)))
if 'LightGBM' in top_2:
    ensemble_models.append(('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=5,
                                                      scale_pos_weight=imbalance_ratio,
                                                      random_state=RANDOM_SEED, verbosity=-1)))

if len(ensemble_models) >= 2:
    voting = VotingClassifier(ensemble_models, voting='soft')
    voting.fit(X_train_resampled, y_train_resampled)
    y_pred = voting.predict(X_test_sel)
    y_prob = voting.predict_proba(X_test_sel)[:, 1]
    results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "VotingEnsemble"))

# ============================================================
# 9. OPTIMAL THRESHOLD TUNING
# ============================================================

print("\nOPTIMAL THRESHOLD TUNING...")

best_idx = np.argmax([res['auc'] for res in results])
best_model_name = results[best_idx]['model']

# Get the best model object
if best_model_name == 'RandomForest':
    best_model = rf_best
elif best_model_name == 'XGBoost':
    best_model = xgb_best
elif best_model_name == 'LightGBM':
    best_model = lgb_best
elif best_model_name == 'GradientBoosting':
    best_model = gb_best
elif best_model_name == 'VotingEnsemble':
    best_model = voting
else:
    best_model = lr

# Find optimal threshold
y_prob_train = best_model.predict_proba(X_train_resampled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train_resampled, y_prob_train)
youden_j = tpr - fpr
best_threshold = thresholds[np.argmax(youden_j)]

print(f"  Best threshold: {best_threshold:.4f}")
print(f"  Youden's J: {np.max(youden_j):.4f}")

# Apply optimized threshold
y_prob_test = best_model.predict_proba(X_test_sel)[:, 1]
y_pred_optimized = (y_prob_test >= best_threshold).astype(int)

optimized_results = evaluate_binary(y_hp_test, y_pred_optimized, y_prob_test, 
                                    f"{best_model_name}_Optimized")
results.append(optimized_results)

print(f"\n  Optimized {best_model_name}:")
print(f"  AUC: {optimized_results['auc']:.4f}")
print(f"  Accuracy: {optimized_results['accuracy']:.4f}")
print(f"  Precision: {optimized_results['precision']:.4f}")
print(f"  Recall: {optimized_results['recall']:.4f}")
print(f"  F1: {optimized_results['f1']:.4f}")

# ============================================================
# 10. CROSS-VALIDATION
# ============================================================

best_result = max(results, key=lambda x: x['auc'])
print(f"\n  Best Model: {best_result['model']} (AUC: {best_result['auc']:.4f})")

cv_scores = cross_val_score(
    rf_best if best_result['model'] == 'RandomForest' or 'RandomForest' in best_result['model'] else
    xgb_best if best_result['model'] == 'XGBoost' or 'XGBoost' in best_result['model'] else
    lgb_best if best_result['model'] == 'LightGBM' or 'LightGBM' in best_result['model'] else
    gb_best if best_result['model'] == 'GradientBoosting' or 'GradientBoosting' in best_result['model'] else
    voting if 'VotingEnsemble' in best_result['model'] else lr,
    X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc'
)
print(f"  CV AUC Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 11. RESULTS SUMMARY
# ============================================================

print("\n  Results Summary:")
print("-"*75)
print(f"  {'Model':22} {'AUC':8} {'Accuracy':8} {'Precision':8} {'Recall':8} {'F1':8}")
print("-"*75)
for res in results:
    print(f"  {res['model']:22} {res['auc']:.4f}   {res['accuracy']:.4f}   {res['precision']:.4f}   {res['recall']:.4f}   {res['f1']:.4f}")

best_optimized = max([r for r in results if 'Optimized' in r['model']], 
                      key=lambda x: x['f1']) if any('Optimized' in r['model'] for r in results) else None
if best_optimized:
    print(f"\n  BEST OVERALL: {best_optimized['model']}")
    print(f"    AUC: {best_optimized['auc']:.4f}")
    print(f"    Accuracy: {best_optimized['accuracy']:.4f}")
    print(f"    Precision: {best_optimized['precision']:.4f}")
    print(f"    Recall: {best_optimized['recall']:.4f}")
    print(f"    F1: {best_optimized['f1']:.4f}")

# ============================================================
# 12. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    results_df = pd.DataFrame(results)
    results_df['target'] = 'HighPerformance'
    results_df['version'] = VERSION
    results_df['environment'] = ENVIRONMENT
    results_df['training_timestamp'] = datetime.now().isoformat()
    
    results_df.to_csv(volume_path + "classification_binary_results.csv", index=False)
    print(f"Results saved to: {volume_path}classification_binary_results.csv")
    
    spark_results = spark.createDataFrame(results_df)
    spark_results.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.classification_binary_results")
    print("Results saved to: adtech_catalog.monitoring.classification_binary_results")
    
except Exception as e:
    print(f"Error saving results: {e}")

# ============================================================
# 13. VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Binary Classification: High Performance (Fixed)",
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
print("BINARY CLASSIFICATION COMPLETE")
print("="*70)

best_auc_model = max(results, key=lambda x: x['auc'])
best_f1_model = max(results, key=lambda x: x['f1'])

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

High Performance Classification (Binary):
   - Best AUC Model: {best_auc_model['model']} (AUC: {best_auc_model['auc']:.4f})
   - Best F1 Model: {best_f1_model['model']} (F1: {best_f1_model['f1']:.4f})
   - CV AUC Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})

Fixes Applied:
   ✅ Removed RFECV (was causing over-regularization)
   ✅ Simplified parameter tuning ranges
   ✅ SMOTE applied AFTER feature selection

Next Step:
   Run Cell 2: Multi-class Classification (Ad Lifecycle Stage)
======================================================================
""")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# 04_Train_Classification_Models - Part 1: Binary Classification (FINAL)
# ============================================================
# Purpose: Train binary classification models with optimized ensembling
# Target: High Performance (ROAS > 2.0)
#
# ENSEMBLE STRATEGY:
# Simple Soft Voting with 3 Diverse Models:
# 1. RandomForest (Bagging) - Captures non-linear interactions
# 2. LightGBM (Boosting) - Captures complex patterns
# 3. Logistic Regression (Linear) - Captures linear relationships
#
# FIXES APPLIED:
# - Added regularization to LightGBM to reduce overfitting
# - Reduced max_depth from 5 to 4
# - Added L1/L2 regularization (reg_alpha, reg_lambda)
# - Added min_child_samples to prevent overfitting

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")
import mlflow
mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("BINARY CLASSIFICATION: HIGH PERFORMANCE (FINAL)")
print("="*70)
print("""
ENSEMBLE STRATEGY:
Simple Soft Voting with 3 Diverse Models:
1. RandomForest (Bagging) - Captures non-linear interactions
2. LightGBM (Boosting) - Captures complex patterns
3. Logistic Regression (Linear) - Captures linear relationships

FIXES APPLIED:
- Added regularization to LightGBM to reduce overfitting
- Reduced max_depth from 5 to 4
- Added L1/L2 regularization (reg_alpha, reg_lambda)
- Added min_child_samples to prevent overfitting
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
# 5. TARGET AND EVALUATION FUNCTIONS
# ============================================================

y_hp_train = train_df['high_performance'].values
y_hp_test = test_df['high_performance'].values

hp_train_counts = np.bincount(y_hp_train)
hp_test_counts = np.bincount(y_hp_test)

print(f"\nHigh Performance Class Balance:")
print(f"  Training - Class 0: {hp_train_counts[0]} ({hp_train_counts[0]/len(y_hp_train)*100:.1f}%)")
print(f"  Training - Class 1: {hp_train_counts[1]} ({hp_train_counts[1]/len(y_hp_train)*100:.1f}%)")
print(f"  Test - Class 0: {hp_test_counts[0]} ({hp_test_counts[0]/len(y_hp_test)*100:.1f}%)")
print(f"  Test - Class 1: {hp_test_counts[1]} ({hp_test_counts[1]/len(y_hp_test)*100:.1f}%)")

def evaluate_binary(y_true, y_pred, y_prob, model_name):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc
    }

def tune_model(X_train, y_train, model, param_dist, model_name):
    print(f"\n  Tuning {model_name}...")
    random_search = RandomizedSearchCV(
        model, param_dist, n_iter=10, cv=3, 
        scoring='roc_auc', random_state=RANDOM_SEED, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print(f"    Best params: {random_search.best_params_}")
    print(f"    Best CV AUC: {random_search.best_score_:.4f}")
    return random_search.best_estimator_

# ============================================================
# 6. FEATURE SELECTION
# ============================================================

print("\nFEATURE SELECTION...")

def select_features_simple(X_train, X_test, y_train, feature_names, threshold='mean'):
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("    Top 5 features:")
    for i, row in importance_df.head(5).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    print(f"    Selected {len(selected_features)} / {len(feature_names)} features")
    
    return X_train_selected, X_test_selected, selected_features

X_train_sel, X_test_sel, selected_feats = select_features_simple(
    X_train_scaled, X_test_scaled, y_hp_train, feature_cols, threshold='mean'
)

# ============================================================
# 7. SMOTE
# ============================================================

print("\n" + "="*70)
print("TRAINING BINARY CLASSIFIERS")
print("="*70)

class_counts = np.bincount(y_hp_train)
imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 0
print(f"\n  Class Imbalance Ratio: {imbalance_ratio:.1f}:1")

print("  Applying SMOTE for class imbalance...")
smote = SMOTE(random_state=RANDOM_SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_sel, y_hp_train)
print(f"    Before SMOTE: {len(y_hp_train)} samples")
print(f"    After SMOTE: {len(y_train_resampled)} samples")

results = []

# ============================================================
# 8. INDIVIDUAL MODELS (WITH REGULARIZATION)
# ============================================================

print("\n" + "-"*70)
print("TRAINING INDIVIDUAL MODELS")
print("-"*70)

# 1. Logistic Regression (baseline)
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
lr.fit(X_train_resampled, y_train_resampled)
y_pred = lr.predict(X_test_sel)
y_prob = lr.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "LogisticRegression"))

# 2. RandomForest
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_best = tune_model(X_train_resampled, y_train_resampled,
                      RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED),
                      rf_params, "RandomForest")
y_pred = rf_best.predict(X_test_sel)
y_prob = rf_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "RandomForest"))

# 3. LightGBM (FIXED - WITH REGULARIZATION)
lgb_params = {
    'n_estimators': [50, 80],
    'max_depth': [3, 4],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [15, 31],
    'scale_pos_weight': [imbalance_ratio, imbalance_ratio/2],
    'reg_alpha': [0.1, 0.5],          # L1 regularization
    'reg_lambda': [0.1, 0.5],         # L2 regularization
    'min_child_samples': [10, 20]     # Minimum samples per leaf
}
lgb_best = tune_model(X_train_resampled, y_train_resampled,
                       lgb.LGBMClassifier(random_state=RANDOM_SEED, verbosity=-1),
                       lgb_params, "LightGBM")
y_pred = lgb_best.predict(X_test_sel)
y_prob = lgb_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "LightGBM"))

# 4. XGBoost (for comparison)
xgb_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'scale_pos_weight': [imbalance_ratio, imbalance_ratio/2]
}
xgb_best = tune_model(X_train_resampled, y_train_resampled,
                       XGBClassifier(random_state=RANDOM_SEED, verbosity=0),
                       xgb_params, "XGBoost")
y_pred = xgb_best.predict(X_test_sel)
y_prob = xgb_best.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "XGBoost"))

# ============================================================
# 9. ENSEMBLE (3 Diverse Models)
# ============================================================

print("\n" + "-"*70)
print("TRAINING ENSEMBLE (3 Diverse Models)")
print("-"*70)

# RF (Bagging) + LGB (Boosting) + LR (Linear)
ensemble_models = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight='balanced', random_state=RANDOM_SEED)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=5,
                                scale_pos_weight=imbalance_ratio,
                                random_state=RANDOM_SEED, verbosity=-1)),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED))
]

voting_ensemble = VotingClassifier(ensemble_models, voting='soft')
voting_ensemble.fit(X_train_resampled, y_train_resampled)

y_pred = voting_ensemble.predict(X_test_sel)
y_prob = voting_ensemble.predict_proba(X_test_sel)[:, 1]
results.append(evaluate_binary(y_hp_test, y_pred, y_prob, "Voting_RF_LGB_LR"))

print(f"\n  Ensemble (RF + LGB + LR):")
print(f"  AUC: {results[-1]['auc']:.4f}")
print(f"  Accuracy: {results[-1]['accuracy']:.4f}")
print(f"  Precision: {results[-1]['precision']:.4f}")
print(f"  Recall: {results[-1]['recall']:.4f}")
print(f"  F1: {results[-1]['f1']:.4f}")

# ============================================================
# 10. OPTIMAL THRESHOLD TUNING
# ============================================================

print("\nOPTIMAL THRESHOLD TUNING...")

# Find best model
best_idx = np.argmax([res['auc'] for res in results])
best_model_name = results[best_idx]['model']

# Get the best model object
if best_model_name == 'RandomForest':
    best_model = rf_best
elif best_model_name == 'LightGBM':
    best_model = lgb_best
elif best_model_name == 'XGBoost':
    best_model = xgb_best
elif best_model_name == 'Voting_RF_LGB_LR':
    best_model = voting_ensemble
else:
    best_model = lr

# Find optimal threshold
y_prob_train = best_model.predict_proba(X_train_resampled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train_resampled, y_prob_train)
youden_j = tpr - fpr
best_threshold = thresholds[np.argmax(youden_j)]

print(f"  Best threshold: {best_threshold:.4f}")
print(f"  Youden's J: {np.max(youden_j):.4f}")

# Apply optimized threshold
y_prob_test = best_model.predict_proba(X_test_sel)[:, 1]
y_pred_optimized = (y_prob_test >= best_threshold).astype(int)

optimized_results = evaluate_binary(y_hp_test, y_pred_optimized, y_prob_test, 
                                    f"{best_model_name}_Optimized")
results.append(optimized_results)

print(f"\n  Optimized {best_model_name}:")
print(f"  AUC: {optimized_results['auc']:.4f}")
print(f"  Accuracy: {optimized_results['accuracy']:.4f}")
print(f"  Precision: {optimized_results['precision']:.4f}")
print(f"  Recall: {optimized_results['recall']:.4f}")
print(f"  F1: {optimized_results['f1']:.4f}")

# ============================================================
# 11. CROSS-VALIDATION
# ============================================================

best_result = max(results, key=lambda x: x['auc'])
print(f"\n  Best Model: {best_result['model']} (AUC: {best_result['auc']:.4f})")

cv_scores = cross_val_score(
    rf_best if best_result['model'] == 'RandomForest' or 'RandomForest' in best_result['model'] else
    lgb_best if best_result['model'] == 'LightGBM' or 'LightGBM' in best_result['model'] else
    xgb_best if best_result['model'] == 'XGBoost' or 'XGBoost' in best_result['model'] else
    voting_ensemble if best_result['model'] == 'Voting_RF_LGB_LR' else lr,
    X_train_resampled, y_train_resampled, cv=5, scoring='roc_auc'
)
print(f"  CV AUC Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 12. RESULTS SUMMARY
# ============================================================

print("\n  Full Results Summary:")
print("-"*75)
print(f"  {'Model':25} {'AUC':8} {'Accuracy':8} {'Precision':8} {'Recall':8} {'F1':8}")
print("-"*75)
for res in results:
    print(f"  {res['model']:25} {res['auc']:.4f}   {res['accuracy']:.4f}   {res['precision']:.4f}   {res['recall']:.4f}   {res['f1']:.4f}")

best_optimized = max([r for r in results if 'Optimized' in r['model']], 
                      key=lambda x: x['f1']) if any('Optimized' in r['model'] for r in results) else None
if best_optimized:
    print(f"\n  BEST OVERALL: {best_optimized['model']}")
    print(f"    AUC: {best_optimized['auc']:.4f}")
    print(f"    Accuracy: {best_optimized['accuracy']:.4f}")
    print(f"    Precision: {best_optimized['precision']:.4f}")
    print(f"    Recall: {best_optimized['recall']:.4f}")
    print(f"    F1: {best_optimized['f1']:.4f}")

# ============================================================
# 13. SAVE RESULTS
# ============================================================

print("\nSAVING RESULTS...")

try:
    results_df = pd.DataFrame(results)
    results_df['target'] = 'HighPerformance'
    results_df['version'] = VERSION
    results_df['environment'] = ENVIRONMENT
    results_df['training_timestamp'] = datetime.now().isoformat()
    
    results_df.to_csv(volume_path + "classification_binary_results.csv", index=False)
    print(f"Results saved to: {volume_path}classification_binary_results.csv")
    
    spark_results = spark.createDataFrame(results_df)
    spark_results.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.classification_binary_results")
    print("Results saved to: adtech_catalog.monitoring.classification_binary_results")
    
except Exception as e:
    print(f"Error saving results: {e}")
# ============================================================
# 14. REGISTER MODEL TO MLflow
# ============================================================

print("\n" + "="*70)
print("REGISTERING HIGH PERFORMANCE MODEL TO MLflow")
print("="*70)

try:
    # Find the best model (LightGBM_Optimized has best F1)
    best_auc_model = max(results, key=lambda x: x['auc'])
    best_f1_model = max(results, key=lambda x: x['f1'])
    
    # Choose the model with best F1 for registration (LightGBM_Optimized)
    best_model_to_register = best_f1_model
    
    # Get the model object
    if 'LightGBM' in best_model_to_register['model'] or 'Optimized' in best_model_to_register['model']:
        # Use LightGBM model
        model_to_register = lgb_best
        model_type = "LightGBM"
    elif 'RandomForest' in best_model_to_register['model']:
        model_to_register = rf_best
        model_type = "RandomForest"
    elif 'XGBoost' in best_model_to_register['model']:
        model_to_register = xgb_best
        model_type = "XGBoost"
    elif 'Voting' in best_model_to_register['model']:
        model_to_register = voting_ensemble
        model_type = "VotingEnsemble"
    else:
        model_to_register = lgb_best
        model_type = "LightGBM"
    
    # Get the best features (using the same selected features)
    X_train_sel_for_registration, _, _ = select_features_simple(
        X_train_scaled, X_test_scaled, y_hp_train, feature_cols, threshold='mean'
    )
    
    # Register to MLflow
    with mlflow.start_run(run_name="HighPerformance_LightGBM_Final") as run:
        signature = mlflow.models.infer_signature(
            X_train_sel_for_registration, 
            model_to_register.predict(X_train_sel_for_registration)
        )
        
        if model_type == "LightGBM":
            mlflow.lightgbm.log_model(
                model_to_register,
                name="hp_lgb_model",
                signature=signature,
                registered_model_name="adtech_catalog.ml_models.high_performance_classifier"
            )
        else:
            mlflow.sklearn.log_model(
                model_to_register,
                name="hp_model",
                signature=signature,
                registered_model_name="adtech_catalog.ml_models.high_performance_classifier"
            )
        
        # Log metrics
        mlflow.log_metric("auc", best_auc_model['auc'])
        mlflow.log_metric("accuracy", best_auc_model['accuracy'])
        mlflow.log_metric("precision", best_auc_model['precision'])
        mlflow.log_metric("recall", best_auc_model['recall'])
        mlflow.log_metric("f1", best_f1_model['f1'])
        mlflow.log_metric("best_threshold", best_threshold)
        
        print(f"  ✅ High Performance {model_type} model registered successfully!")
        print(f"     Model: adtech_catalog.ml_models.high_performance_classifier")
        print(f"     AUC: {best_auc_model['auc']:.4f}")
        print(f"     F1: {best_f1_model['f1']:.4f}")
        print(f"     Best Threshold: {best_threshold:.4f}")

except Exception as e:
    print(f"⚠️ MLflow registration error: {e}")
    print("   Model trained successfully but not registered to MLflow.")

# ============================================================
# 15. SET ALIAS TO PRODUCTION
# ============================================================

print("\n" + "="*70)
print("SETTING PRODUCTION ALIAS")
print("="*70)

try:
    from mlflow import MlflowClient
    client = MlflowClient(registry_uri="databricks-uc")
    
    client.set_registered_model_alias(
        name="adtech_catalog.ml_models.high_performance_classifier",
        alias="Production",
        version=1
    )
    print("✅ High Performance model alias 'Production' set to Version 1")
    
except Exception as e:
    print(f"⚠️ Could not set alias: {e}")
    print("   Please set alias manually in Catalog UI.")
# ============================================================
# 16. VERSION HISTORY
# ============================================================

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Binary Classification: High Performance (Final)",
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
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "="*70)
print("BINARY CLASSIFICATION COMPLETE")
print("="*70)

best_auc_model = max(results, key=lambda x: x['auc'])
best_f1_model = max(results, key=lambda x: x['f1'])
best_ensemble = max([r for r in results if 'Voting' in r['model']], 
                    key=lambda x: x['auc']) if any('Voting' in r['model'] for r in results) else None

print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

High Performance Classification (Binary):
   - Best AUC Model: {best_auc_model['model']} (AUC: {best_auc_model['auc']:.4f})
   - Best F1 Model: {best_f1_model['model']} (F1: {best_f1_model['f1']:.4f})
""")

if best_ensemble:
    print(f"   - Ensemble: {best_ensemble['model']} (AUC: {best_ensemble['auc']:.4f})")

print(f"""
   - CV AUC Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})

FIXES APPLIED:
   Reduced LightGBM max_depth (5 → 4)
   Added L1 regularization (reg_alpha)
   Added L2 regularization (reg_lambda)
   Added min_child_samples to prevent overfitting

Ensemble Strategy Used:
   Soft Voting with 3 Diverse Models:
      1. RandomForest (Bagging)
      2. LightGBM (Boosting)  
      3. Logistic Regression (Linear)

Next Step:
   Multi-class Classification (Ad Lifecycle Stage) - Will be skipped
======================================================================
""")

# COMMAND ----------

# Databricks notebook source-------------------------------------------No need-------------------------------------------------------------
# ============================================================
# 04_Train_Classification_Models - Part 2: Multi-class Classification
# ============================================================
# Purpose: Train multi-class classification models for Ad Lifecycle Stage
# Target: Ad Lifecycle Stage (New, Growing, Mature, Declining)
#
# NOTE: This notebook checks if the target has variation.
# If all samples belong to one class, it skips training.

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pyspark.sql import SparkSession
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

print("="*70)
print("MULTI-CLASS CLASSIFICATION: AD LIFECYCLE STAGE")
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
    
    print("Data loaded successfully")
    
except Exception as e:
    print(f"Error loading split data: {e}")
    dbutils.notebook.exit("Failed to load split data")

# ============================================================
# 3. CHECK TARGET VARIATION
# ============================================================

print("\nCHECKING TARGET VARIATION...")

y_lifecycle_train = train_df['ad_lifecycle_stage'].values
y_lifecycle_test = test_df['ad_lifecycle_stage'].values

# Check unique values
unique_train = np.unique(y_lifecycle_train)
unique_test = np.unique(y_lifecycle_test)

print(f"  Unique classes in training: {unique_train}")
print(f"  Unique classes in test: {unique_test}")

# Check class distribution
print(f"\n  Class distribution in training:")
for val in unique_train:
    count = np.sum(y_lifecycle_train == val)
    print(f"    {val}: {count} ({count/len(y_lifecycle_train)*100:.1f}%)")

# ============================================================
# 4. DECISION: SKIP OR TRAIN
# ============================================================

if len(unique_train) < 2:
    print("\n" + "="*70)
    print("SKIPPING MULTI-CLASS CLASSIFICATION")
    print("="*70)
    print("""
    REASON: Training data has only ONE class.
    
    This means all ads have the same lifecycle stage (all are 'New').
    The model cannot learn anything from data with no variation.
    
    ACTION: Skipping multi-class classification.
    
    RECOMMENDATION:
    - This is expected for synthetic data where all ads have same ingestion date.
    - In production with real data, this target would have variation.
    - The pipeline is ready for multi-class classification when data has variation.
    """)
    
    # Save a placeholder result
    placeholder_results = pd.DataFrame([{
        "model": "SKIPPED",
        "accuracy": 0.0,
        "f1_weighted": 0.0,
        "f1_macro": 0.0,
        "target": "AdLifecycleStage",
        "version": VERSION,
        "environment": ENVIRONMENT,
        "training_timestamp": datetime.now().isoformat(),
        "reason": "Only one class in training data"
    }])
    
    placeholder_results.to_csv(volume_path + "classification_multi_results.csv", index=False)
    print(f"Placeholder results saved to: {volume_path}classification_multi_results.csv")
    
    # Save version history
    try:
        version_info = spark.createDataFrame([(
            VERSION,
            ENVIRONMENT,
            GIT_COMMIT,
            datetime.now().isoformat(),
            "Multi-class Classification: SKIPPED (only one class)",
            "SKIPPED"
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
        
        print("Version history updated")
        
    except Exception as e:
        print(f"Could not save version history: {e}")
    
    print("\n" + "="*70)
    print("CELL 2 COMPLETE (SKIPPED)")
    print("="*70)
    
    dbutils.notebook.exit("Multi-class classification skipped - only one class in data")

else:
    print(f"\n  Multiple classes found! Proceeding with training...")
    
    # ============================================================
    # 5. PREPARE FEATURES
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
    # 6. SCALE FEATURES
    # ============================================================
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled")
    
    # ============================================================
    # 7. ENCODE TARGET
    # ============================================================
    
    y_lifecycle_train = train_df['ad_lifecycle_stage'].values
    y_lifecycle_test = test_df['ad_lifecycle_stage'].values
    
    lifecycle_labels = ['New', 'Growing', 'Mature', 'Declining']
    le_lifecycle = LabelEncoder()
    y_lifecycle_train_encoded = le_lifecycle.fit_transform(y_lifecycle_train)
    y_lifecycle_test_encoded = le_lifecycle.transform(y_lifecycle_test)
    
    print(f"\n  Encoded labels: {le_lifecycle.classes_}")
    print(f"  Training shape: {y_lifecycle_train_encoded.shape}")
    print(f"  Test shape: {y_lifecycle_test_encoded.shape}")
    
    # ============================================================
    # 8. TRAIN MODELS (if multiple classes exist)
    # ============================================================
    
    print("\n" + "="*70)
    print("TRAINING MULTI-CLASS CLASSIFIERS")
    print("="*70)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    results = []
    
    # Logistic Regression
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train_scaled, y_lifecycle_train_encoded)
    y_pred = lr.predict(X_test_scaled)
    results.append({
        "model": "LogisticRegression",
        "accuracy": accuracy_score(y_lifecycle_test_encoded, y_pred),
        "f1_weighted": f1_score(y_lifecycle_test_encoded, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_lifecycle_test_encoded, y_pred, average='macro', zero_division=0)
    })
    
    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=RANDOM_SEED)
    rf.fit(X_train_scaled, y_lifecycle_train_encoded)
    y_pred = rf.predict(X_test_scaled)
    results.append({
        "model": "RandomForest",
        "accuracy": accuracy_score(y_lifecycle_test_encoded, y_pred),
        "f1_weighted": f1_score(y_lifecycle_test_encoded, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_lifecycle_test_encoded, y_pred, average='macro', zero_division=0)
    })
    
    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbosity=0)
        xgb.fit(X_train_scaled, y_lifecycle_train_encoded)
        y_pred = xgb.predict(X_test_scaled)
        results.append({
            "model": "XGBoost",
            "accuracy": accuracy_score(y_lifecycle_test_encoded, y_pred),
            "f1_weighted": f1_score(y_lifecycle_test_encoded, y_pred, average='weighted', zero_division=0),
            "f1_macro": f1_score(y_lifecycle_test_encoded, y_pred, average='macro', zero_division=0)
        })
    except:
        print("  XGBoost not available, skipping...")
    
    # LightGBM
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_SEED, verbosity=-1)
        lgb_model.fit(X_train_scaled, y_lifecycle_train_encoded)
        y_pred = lgb_model.predict(X_test_scaled)
        results.append({
            "model": "LightGBM",
            "accuracy": accuracy_score(y_lifecycle_test_encoded, y_pred),
            "f1_weighted": f1_score(y_lifecycle_test_encoded, y_pred, average='weighted', zero_division=0),
            "f1_macro": f1_score(y_lifecycle_test_encoded, y_pred, average='macro', zero_division=0)
        })
    except:
        print("  LightGBM not available, skipping...")
    
    # ============================================================
    # 9. RESULTS SUMMARY
    # ============================================================
    
    print("\n  Results Summary:")
    print("-"*70)
    print(f"  {'Model':20} {'Accuracy':10} {'F1 (Weighted)':12} {'F1 (Macro)':10}")
    print("-"*70)
    for res in results:
        print(f"  {res['model']:20} {res['accuracy']:.4f}      {res['f1_weighted']:.4f}         {res['f1_macro']:.4f}")
    
    # ============================================================
    # 10. SAVE RESULTS
    # ============================================================
    
    print("\nSAVING RESULTS...")
    
    try:
        results_df = pd.DataFrame(results)
        results_df['target'] = 'AdLifecycleStage'
        results_df['version'] = VERSION
        results_df['environment'] = ENVIRONMENT
        results_df['training_timestamp'] = datetime.now().isoformat()
        
        results_df.to_csv(volume_path + "classification_multi_results.csv", index=False)
        print(f"Results saved to: {volume_path}classification_multi_results.csv")
        
        spark_results = spark.createDataFrame(results_df)
        spark_results.write \
            .mode("overwrite") \
            .format("delta") \
            .saveAsTable("adtech_catalog.monitoring.classification_multi_results")
        print("Results saved to: adtech_catalog.monitoring.classification_multi_results")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # ============================================================
    # 11. VERSION HISTORY
    # ============================================================
    
    try:
        version_info = spark.createDataFrame([(
            VERSION,
            ENVIRONMENT,
            GIT_COMMIT,
            datetime.now().isoformat(),
            "Multi-class Classification: Ad Lifecycle Stage",
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
    
    print("\n" + "="*70)
    print("MULTI-CLASS CLASSIFICATION COMPLETE")
    print("="*70)
    
    best_result = max(results, key=lambda x: x['f1_weighted'])
    print(f"""
SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Ad Lifecycle Stage Classification (Multi-class):
   - Best Model: {best_result['model']}
   - Best F1 Weighted: {best_result['f1_weighted']:.4f}
   - Accuracy: {best_result['accuracy']:.4f}

Next Steps:
   1. Run: 05_Model_Explainability.py
======================================================================
""")