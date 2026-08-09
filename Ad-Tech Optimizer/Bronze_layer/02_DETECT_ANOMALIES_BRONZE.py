# Databricks notebook source
# Databricks notebook source
# ============================================================
# ANOMALY DETECTION ON BRONZE LAYER DATA
# ============================================================
# Purpose: Detect and log all injected anomalies in bronze data
# Author: Sanju
# Team: 5 Members
# Date: 2026-07-01
# 
# IMPROVEMENTS APPLIED:
# 1. OVERWRITE for monitoring tables (keeps only latest)
# 2. Idempotency checks (table existence before loading)
# 3. Parameterized runs (config-driven paths)
# 4. Version tracking (batch_id + environment + git_commit)
# 5. Enhanced error handling and user feedback

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime
import yaml
import os

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

# Get environment from config or default
ENVIRONMENT = config.get('environment', 'development') if config else 'development'

# Generate version
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
# 2. LOAD BRONZE TABLES WITH IDEMPOTENCY CHECKS
# ============================================================

print("")
print("="*70)
print("LOADING BRONZE TABLES")
print("="*70)

def check_table_exists(table_name):
    """Check if a Delta table exists"""
    try:
        spark.sql(f"DESCRIBE {table_name}")
        return True
    except:
        return False

TABLE_CLICKS = "adtech_catalog.bronze.ad_click_logs"
TABLE_CATALOG = "adtech_catalog.bronze.ad_metadata_catalog"

# Check if tables exist before loading
clicks_exists = check_table_exists(TABLE_CLICKS)
catalog_exists = check_table_exists(TABLE_CATALOG)

if not clicks_exists:
    print(f"ERROR: Table {TABLE_CLICKS} does not exist.")
    print("Please run 01_LOAD_TO_BRONZE.py first.")
    dbutils.notebook.exit("Bronze table not found: ad_click_logs")

if not catalog_exists:
    print(f"ERROR: Table {TABLE_CATALOG} does not exist.")
    print("Please run 01_LOAD_TO_BRONZE.py first.")
    dbutils.notebook.exit("Bronze table not found: ad_metadata_catalog")

df_clicks = spark.table(TABLE_CLICKS)
df_catalog = spark.table(TABLE_CATALOG)

print(f"Loaded User Events: {df_clicks.count():,} rows")
print(f"Loaded Ad Catalog: {df_catalog.count():,} rows")

print("")
print("User Events Sample:")
display(df_clicks.limit(5))

print("")
print("Ad Catalog Sample:")
display(df_catalog.limit(5))

# ============================================================
# 3. USER EVENTS ANOMALY DETECTION
# ============================================================

print("")
print("="*70)
print("USER EVENTS ANOMALY DETECTION")
print("="*70)

anomaly_results = {}

# ------------------------------------------------------------
# 3.1: Timestamp Format Anomalies
# ------------------------------------------------------------
print("")
print("3.1: Timestamp Format Anomalies")

slash_format_count = df_clicks.filter(
    col("Click_Timestamp").rlike(r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$')
).count()

future_dates_count = df_clicks.filter(
    col("Click_Timestamp") == "2035-12-25 12:00:00"
).count()

print(f"   Slash Format (DD/MM/YYYY): {slash_format_count:,} rows")
print(f"   Future Dates (2035): {future_dates_count:,} rows")

anomaly_results["timestamp_slash_format"] = slash_format_count
anomaly_results["timestamp_future_dates"] = future_dates_count

print("")
print("   Sample Timestamp Anomalies:")
display(
    df_clicks.filter(
        col("Click_Timestamp").rlike(r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$') |
        (col("Click_Timestamp") == "2035-12-25 12:00:00")
    ).select("Click_Timestamp").limit(10)
)

# ------------------------------------------------------------
# 3.2: Ad_Type Anomalies
# ------------------------------------------------------------
print("")
print("3.2: Ad_Type Anomalies")

null_ad_type = df_clicks.filter(col("Ad_Type").isNull()).count()
lower_video = df_clicks.filter(col("Ad_Type") == "video").count()
upper_image = df_clicks.filter(col("Ad_Type") == "IMAGE").count()

print(f"   Null Values: {null_ad_type:,}")
print(f"   Lowercase video: {lower_video:,}")
print(f"   Uppercase IMAGE: {upper_image:,}")

anomaly_results["ad_type_null"] = null_ad_type
anomaly_results["ad_type_lowercase"] = lower_video
anomaly_results["ad_type_uppercase"] = upper_image

print("")
print("   Sample Ad_Type Anomalies:")
display(
    df_clicks.filter(
        col("Ad_Type").isNull() |
        (col("Ad_Type") == "video") |
        (col("Ad_Type") == "IMAGE")
    ).select("Ad_Type").limit(10)
)

# ------------------------------------------------------------
# 3.3: Watch_Duration Anomalies
# ------------------------------------------------------------
print("")
print("3.3: Watch_Duration Anomalies")

neg_duration = df_clicks.filter(col("Watch_Duration") < 0).count()
overflow_duration = df_clicks.filter(col("Watch_Duration") > 1000).count()
logical_conflict = df_clicks.filter(
    (col("user_clicked") == 1) & (col("Watch_Duration") == 0)
).count()

print(f"   Negative Duration: {neg_duration:,}")
print(f"   Overflow (>1000s): {overflow_duration:,}")
print(f"   Clicked but 0s Duration (Logical Conflict): {logical_conflict:,}")

anomaly_results["watch_duration_negative"] = neg_duration
anomaly_results["watch_duration_overflow"] = overflow_duration
anomaly_results["watch_duration_logical_conflict"] = logical_conflict

print("")
print("   Sample Watch_Duration Anomalies:")
display(
    df_clicks.filter(
        (col("Watch_Duration") < 0) |
        (col("Watch_Duration") > 1000) |
        ((col("user_clicked") == 1) & (col("Watch_Duration") == 0))
    ).select("Watch_Duration", "user_clicked").limit(10)
)

# ------------------------------------------------------------
# 3.4: User_Age Anomalies
# ------------------------------------------------------------
print("")
print("3.4: User_Age Anomalies")

neg_age = df_clicks.filter(col("user_age") < 0).count()
old_age = df_clicks.filter(col("user_age") > 100).count()
null_age = df_clicks.filter(col("user_age").isNull()).count()

print(f"   Negative Age: {neg_age:,}")
print(f"   Age > 100: {old_age:,}")
print(f"   Null Age: {null_age:,}")

anomaly_results["user_age_negative"] = neg_age
anomaly_results["user_age_old"] = old_age
anomaly_results["user_age_null"] = null_age

print("")
print("   Sample User_Age Anomalies:")
display(
    df_clicks.filter(
        (col("user_age") < 0) |
        (col("user_age") > 100) |
        col("user_age").isNull()
    ).select("user_age").limit(10)
)

# ------------------------------------------------------------
# 3.5: Device Anomalies
# ------------------------------------------------------------
print("")
print("3.5: Device Anomalies")

typo_moble = df_clicks.filter(col("device") == "moble").count()
caps_desktop = df_clicks.filter(col("device") == "DESKTOP").count()
null_device = df_clicks.filter(col("device").isNull()).count()
empty_device = df_clicks.filter(col("device") == "").count()

device_corruption = df_clicks.filter(
    col("device").rlike(r'^\d+\s+')
).count()

print(f"   Typo moble: {typo_moble:,}")
print(f"   Uppercase DESKTOP: {caps_desktop:,}")
print(f"   Null Device: {null_device:,}")
print(f"   Empty String: {empty_device:,}")
print(f"   CORRUPTED age prefix like 47 DESKTOP: {device_corruption:,}")

anomaly_results["device_typo_moble"] = typo_moble
anomaly_results["device_uppercase"] = caps_desktop
anomaly_results["device_null"] = null_device
anomaly_results["device_empty"] = empty_device
anomaly_results["device_corruption"] = device_corruption

if device_corruption > 0:
    print("")
    print("   Sample Device Corruption:")
    display(
        df_clicks.filter(col("device").rlike(r'^\d+\s+'))
        .select("device")
        .limit(10)
    )

print("")
print("   Device Issue Summary:")
device_summary = df_clicks.groupBy("device").count().orderBy(col("count").desc()).limit(20)
display(device_summary)

# ------------------------------------------------------------
# 3.6: Platform Source Anomalies
# ------------------------------------------------------------
print("")
print("3.6: Platform Source Anomalies")

if "platform_source" in df_clicks.columns:
    gogle_typo = df_clicks.filter(col("platform_source") == "gogle").count()
    facebok_typo = df_clicks.filter(col("platform_source") == "facebok").count()
    unknown_platform = df_clicks.filter(col("platform_source") == "Unknown").count()
    null_platform = df_clicks.filter(col("platform_source").isNull()).count()
    
    print(f"   Typo gogle: {gogle_typo:,}")
    print(f"   Typo facebok: {facebok_typo:,}")
    print(f"   Unknown Platform: {unknown_platform:,}")
    print(f"   Null Platform: {null_platform:,}")
    
    anomaly_results["platform_gogle"] = gogle_typo
    anomaly_results["platform_facebok"] = facebok_typo
    anomaly_results["platform_unknown"] = unknown_platform
    anomaly_results["platform_null"] = null_platform
    
    print("")
    print("   Sample Platform Anomalies:")
    display(
        df_clicks.filter(
            (col("platform_source") == "gogle") |
            (col("platform_source") == "facebok") |
            (col("platform_source") == "Unknown")
        ).select("platform_source").limit(10)
    )
else:
    print("   platform_source column not found")

# ------------------------------------------------------------
# 3.7: Gender Identity Shifting
# ------------------------------------------------------------
print("")
print("3.7: Gender Identity Shifting Anomaly")

gender_shift_users = df_clicks.groupBy("User_ID").agg(
    countDistinct("user_gender").alias("gender_count"),
    collect_set("user_gender").alias("gender_values")
).filter(col("gender_count") > 1)

gender_shift = gender_shift_users.count()

print(f"   Users with multiple genders: {gender_shift:,}")

anomaly_results["gender_identity_shifting"] = gender_shift

if gender_shift > 0:
    print("")
    print("   Sample Gender Identity Shifting:")
    display(
        gender_shift_users.select(
            "User_ID",
            "gender_count",
            "gender_values"
        ).limit(10)
    )
    
    print("")
    print("   Gender Shift Details:")
    display(
        gender_shift_users.select(
            "User_ID",
            "gender_count",
            "gender_values"
        ).limit(5)
    )

# ------------------------------------------------------------
# 3.8: Duplicate Rows
# ------------------------------------------------------------
print("")
print("3.8: Duplicate Rows")

total_rows = df_clicks.count()
duplicate_rows = df_clicks.count() - df_clicks.dropDuplicates().count()
duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0

print(f"   Total Rows: {total_rows:,}")
print(f"   Duplicate Rows: {duplicate_rows:,}")
print(f"   Duplicate Percentage: {duplicate_percentage:.2f}%")

anomaly_results["total_rows"] = total_rows
anomaly_results["duplicate_rows"] = duplicate_rows
anomaly_results["duplicate_percentage"] = duplicate_percentage

# ============================================================
# 4. AD CATALOG ANOMALY DETECTION
# ============================================================

print("")
print("="*70)
print("AD CATALOG ANOMALY DETECTION")
print("="*70)

# Create a temporary DataFrame with safe string conversions for all columns
df_catalog_temp = df_catalog.withColumn(
    "ad_video_length_str",
    col("ad_video_length").cast("string")
).withColumn(
    "ad_type_str",
    col("Ad_Type").cast("string")
).withColumn(
    "cpc_str",
    col("Cost_Per_Click").cast("string")
)

# ------------------------------------------------------------
# 4.1: Primary Key Duplicates
# ------------------------------------------------------------
print("")
print("4.1: Primary Key Duplicates")

total_catalog = df_catalog.count()
duplicate_ads = df_catalog.groupBy("Ad_Reference_ID").count().filter(col("count") > 1).count()
corrupt_categories = df_catalog.filter(col("Ad_Category") == "Corrupted_Category").count()

print(f"   Total Catalog Rows: {total_catalog:,}")
print(f"   Duplicate Ad_Reference_IDs: {duplicate_ads:,}")
print(f"   Corrupted Categories: {corrupt_categories:,}")

anomaly_results["catalog_total_rows"] = total_catalog
anomaly_results["catalog_duplicates"] = duplicate_ads
anomaly_results["catalog_corrupted"] = corrupt_categories

if duplicate_ads > 0:
    print("")
    print("   Sample Duplicate Ad_Reference_IDs:")
    display(
        df_catalog.groupBy("Ad_Reference_ID").count()
        .filter(col("count") > 1)
        .orderBy(col("count").desc())
        .limit(10)
    )

# ------------------------------------------------------------
# 4.2: Category Anomalies
# ------------------------------------------------------------
print("")
print("4.2: Category Anomalies")

lower_cat = df_catalog.filter(lower(col("Ad_Category")) == col("Ad_Category")).count()
typo_electronics = df_catalog.filter(col("Ad_Category") == "Eletronics").count()
null_category = df_catalog.filter(col("Ad_Category").isNull()).count()

print(f"   Lowercase Categories: {lower_cat:,}")
print(f"   Typo Eletronics: {typo_electronics:,}")
print(f"   Null Category: {null_category:,}")

anomaly_results["category_lowercase"] = lower_cat
anomaly_results["category_typo"] = typo_electronics
anomaly_results["category_null"] = null_category

print("")
print("   Sample Category Anomalies:")
display(
    df_catalog.filter(
        (lower(col("Ad_Category")) == col("Ad_Category")) |
        (col("Ad_Category") == "Eletronics") |
        (col("Ad_Category") == "Corrupted_Category")
    ).select("Ad_Category").limit(10)
)

# ------------------------------------------------------------
# 4.3: Device Anomalies
# ------------------------------------------------------------
print("")
print("4.3: Device Anomalies")

null_device_cat = df_catalog.filter(col("Ad_Device").isNull()).count()
padded_mobile = df_catalog.filter(col("Ad_Device") == "  Mobile  ").count()
typo_moble_cat = df_catalog.filter(col("Ad_Device") == "moble").count()
empty_device_cat = df_catalog.filter(col("Ad_Device") == "").count()

print(f"   Null Device: {null_device_cat:,}")
print(f"   Padded Mobile: {padded_mobile:,}")
print(f"   Typo moble: {typo_moble_cat:,}")
print(f"   Empty String: {empty_device_cat:,}")

anomaly_results["catalog_device_null"] = null_device_cat
anomaly_results["catalog_device_padded"] = padded_mobile
anomaly_results["catalog_device_typo"] = typo_moble_cat
anomaly_results["catalog_device_empty"] = empty_device_cat

# ------------------------------------------------------------
# 4.4: Location Anomalies
# ------------------------------------------------------------
print("")
print("4.4: Location Anomalies")

trailing_spaces = df_catalog.filter(col("Ad_Location").endswith(" ")).count()
null_location = df_catalog.filter(col("Ad_Location").isNull()).count()

print(f"   Trailing Spaces: {trailing_spaces:,}")
print(f"   Null Location: {null_location:,}")

anomaly_results["location_trailing_spaces"] = trailing_spaces
anomaly_results["location_null"] = null_location

# ------------------------------------------------------------
# 4.5: Cost Per Click Anomalies
# ------------------------------------------------------------
print("")
print("4.5: Cost Per Click Anomalies")

cpc_strings = df_catalog_temp.filter(
    col("cpc_str").contains("$")
).count()

df_catalog_temp = df_catalog_temp.withColumn(
    "cpc_numeric",
    when(
        col("cpc_str").contains("$"),
        regexp_replace(col("cpc_str"), "\\$", "").cast("double")
    ).otherwise(col("cpc_str").cast("double"))
)

negative_cpc = df_catalog_temp.filter(col("cpc_numeric") < 0).count()

print(f"   String Format $: {cpc_strings:,}")
print(f"   Negative Values: {negative_cpc:,}")

anomaly_results["cpc_string_format"] = cpc_strings
anomaly_results["cpc_negative"] = negative_cpc

print("")
print("   Sample Cost Per Click Anomalies:")
display(
    df_catalog_temp.filter(
        col("cpc_str").contains("$") |
        (col("cpc_numeric") < 0)
    ).select("Cost_Per_Click", "cpc_str", "cpc_numeric").limit(10)
)

# ------------------------------------------------------------
# 4.6: Ad Type Anomalies
# ------------------------------------------------------------
print("")
print("4.6: Ad Type Anomalies")

lowercase_vedeo = df_catalog.filter(col("Ad_Type") == "video").count()
typo_vedeo = df_catalog.filter(col("Ad_Type") == "Vedeo").count()
uppercase_image = df_catalog.filter(col("Ad_Type") == "IMAGE").count()
typo_carusel = df_catalog.filter(col("Ad_Type") == "Carusel").count()
null_ad_type_cat = df_catalog.filter(col("Ad_Type").isNull()).count()

print(f"   Lowercase video: {lowercase_vedeo:,}")
print(f"   Typo Vedeo: {typo_vedeo:,}")
print(f"   Uppercase IMAGE: {uppercase_image:,}")
print(f"   Typo Carusel: {typo_carusel:,}")
print(f"   Null Ad Type: {null_ad_type_cat:,}")

anomaly_results["catalog_ad_type_lowercase"] = lowercase_vedeo
anomaly_results["catalog_ad_type_typo_vedeo"] = typo_vedeo
anomaly_results["catalog_ad_type_uppercase"] = uppercase_image
anomaly_results["catalog_ad_type_typo_carusel"] = typo_carusel
anomaly_results["catalog_ad_type_null"] = null_ad_type_cat

# ------------------------------------------------------------
# 4.7: Video Length Anomalies
# ------------------------------------------------------------
print("")
print("4.7: Video Length Anomalies")

string_length = df_catalog_temp.filter(
    col("ad_video_length_str") == "45s"
).count()

conflict_video_len = df_catalog_temp.filter(
    (col("ad_type_str") != "Video") & 
    (col("ad_type_str") != "video") &
    (col("ad_type_str") != "Vedeo") &
    (col("ad_video_length_str").isNotNull()) &
    (col("ad_video_length_str") != "0") &
    (col("ad_video_length_str") != "0.0") &
    (col("ad_video_length_str") != "null") &
    (col("ad_video_length_str") != "")
).count()

print(f"   String Format 45s: {string_length:,}")
print(f"   Logical Conflict Non-Video with Length > 0: {conflict_video_len:,}")

anomaly_results["video_length_string"] = string_length
anomaly_results["video_length_conflict"] = conflict_video_len

print("")
print("   Sample Video Length Anomalies:")
display(
    df_catalog_temp.filter(
        (col("ad_video_length_str") == "45s") |
        ((col("ad_type_str") != "Video") & 
         (col("ad_type_str") != "video") & 
         (col("ad_video_length_str").isNotNull()) &
         (col("ad_video_length_str") != "0") &
         (col("ad_video_length_str") != "0.0"))
    ).select("Ad_Type", "ad_video_length", "ad_video_length_str").limit(10)
)

# ============================================================
# 5. ANOMALY SUMMARY REPORT
# ============================================================

print("")
print("="*70)
print("ANOMALY DETECTION SUMMARY REPORT")
print("="*70)

anomaly_summary = spark.createDataFrame([
    ("User Events", "Timestamp - Slash Format", anomaly_results.get("timestamp_slash_format", 0)),
    ("User Events", "Timestamp - Future Dates", anomaly_results.get("timestamp_future_dates", 0)),
    ("User Events", "Ad_Type - Null", anomaly_results.get("ad_type_null", 0)),
    ("User Events", "Ad_Type - Lowercase video", anomaly_results.get("ad_type_lowercase", 0)),
    ("User Events", "Ad_Type - Uppercase IMAGE", anomaly_results.get("ad_type_uppercase", 0)),
    ("User Events", "Watch_Duration - Negative", anomaly_results.get("watch_duration_negative", 0)),
    ("User Events", "Watch_Duration - Overflow", anomaly_results.get("watch_duration_overflow", 0)),
    ("User Events", "Watch_Duration - Logical Conflict", anomaly_results.get("watch_duration_logical_conflict", 0)),
    ("User Events", "User_Age - Negative", anomaly_results.get("user_age_negative", 0)),
    ("User Events", "User_Age - Old >100", anomaly_results.get("user_age_old", 0)),
    ("User Events", "User_Age - Null", anomaly_results.get("user_age_null", 0)),
    ("User Events", "Device - Typo moble", anomaly_results.get("device_typo_moble", 0)),
    ("User Events", "Device - Uppercase DESKTOP", anomaly_results.get("device_uppercase", 0)),
    ("User Events", "Device - Null", anomaly_results.get("device_null", 0)),
    ("User Events", "Device - Empty", anomaly_results.get("device_empty", 0)),
    ("User Events", "Device - Corruption CRITICAL", anomaly_results.get("device_corruption", 0)),
    ("User Events", "Platform - Typo gogle", anomaly_results.get("platform_gogle", 0)),
    ("User Events", "Platform - Typo facebok", anomaly_results.get("platform_facebok", 0)),
    ("User Events", "Platform - Unknown", anomaly_results.get("platform_unknown", 0)),
    ("User Events", "Gender - Identity Shifting", anomaly_results.get("gender_identity_shifting", 0)),
    ("User Events", "Duplicate Rows", anomaly_results.get("duplicate_rows", 0)),
    ("Ad Catalog", "Total Rows", anomaly_results.get("catalog_total_rows", 0)),
    ("Ad Catalog", "Duplicate Ad_Reference_IDs", anomaly_results.get("catalog_duplicates", 0)),
    ("Ad Catalog", "Corrupted Categories", anomaly_results.get("catalog_corrupted", 0)),
    ("Ad Catalog", "Category - Lowercase", anomaly_results.get("category_lowercase", 0)),
    ("Ad Catalog", "Category - Typo Eletronics", anomaly_results.get("category_typo", 0)),
    ("Ad Catalog", "Category - Null", anomaly_results.get("category_null", 0)),
    ("Ad Catalog", "Device - Null", anomaly_results.get("catalog_device_null", 0)),
    ("Ad Catalog", "Device - Padded Spaces", anomaly_results.get("catalog_device_padded", 0)),
    ("Ad Catalog", "Device - Typo moble", anomaly_results.get("catalog_device_typo", 0)),
    ("Ad Catalog", "Device - Empty", anomaly_results.get("catalog_device_empty", 0)),
    ("Ad Catalog", "Location - Trailing Spaces", anomaly_results.get("location_trailing_spaces", 0)),
    ("Ad Catalog", "Location - Null", anomaly_results.get("location_null", 0)),
    ("Ad Catalog", "CPC - String Format $", anomaly_results.get("cpc_string_format", 0)),
    ("Ad Catalog", "CPC - Negative Values", anomaly_results.get("cpc_negative", 0)),
    ("Ad Catalog", "Ad_Type - Lowercase video", anomaly_results.get("catalog_ad_type_lowercase", 0)),
    ("Ad Catalog", "Ad_Type - Typo Vedeo", anomaly_results.get("catalog_ad_type_typo_vedeo", 0)),
    ("Ad Catalog", "Ad_Type - Uppercase IMAGE", anomaly_results.get("catalog_ad_type_uppercase", 0)),
    ("Ad Catalog", "Ad_Type - Typo Carusel", anomaly_results.get("catalog_ad_type_typo_carusel", 0)),
    ("Ad Catalog", "Ad_Type - Null", anomaly_results.get("catalog_ad_type_null", 0)),
    ("Ad Catalog", "Video Length - String 45s", anomaly_results.get("video_length_string", 0)),
    ("Ad Catalog", "Video Length - Logical Conflict", anomaly_results.get("video_length_conflict", 0))
], ["table", "anomaly_type", "count"])

print("")
print("Anomaly Summary Table:")
display(anomaly_summary)

# ============================================================
# 6. SAVE ANOMALY REPORT WITH VERSION TRACKING
# ============================================================

print("")
print("="*70)
print("SAVING ANOMALY REPORT")
print("="*70)

current_ts = datetime.now()

anomaly_report = anomaly_summary.withColumn(
    "detection_timestamp", lit(current_ts)
).withColumn(
    "detection_date", lit(current_ts.date())
).withColumn(
    "batch_id", lit(VERSION)
).withColumn(
    "environment", lit(ENVIRONMENT)
).withColumn(
    "git_commit", lit(GIT_COMMIT)
).withColumn(
    "status", 
    when(col("count") > 0, "ANOMALY_DETECTED")
    .otherwise("NO_ANOMALY")
)

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS adtech_catalog.monitoring")
    
    anomaly_report.write \
        .mode("overwrite") \
        .format("delta") \
        .partitionBy("detection_date") \
        .saveAsTable("adtech_catalog.monitoring.anomaly_report")
    
    print(f"Anomaly report saved to: adtech_catalog.monitoring.anomaly_report")
    print(f"   Records: {anomaly_report.count():,}")
    print(f"   Batch ID: {VERSION}")
    print(f"   Environment: {ENVIRONMENT}")
   
except Exception as err:
    print("Could not save anomaly report: {}".format(str(err)))
    anomaly_report.createOrReplaceTempView("temp_anomaly_report")
    print("Anomaly report available as temp view: temp_anomaly_report")

# ============================================================
# 7. SAVE TO VERSION HISTORY (FIXED - 6 COLUMNS)
# ============================================================

print("")
print("="*70)
print("SAVING VERSION HISTORY")
print("="*70)

try:
    # Match the existing schema from 01_LOAD_TO_BRONZE.py (6 columns)
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Anomaly Detection - Bronze Layer",
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
# 8. CRITICAL ANOMALIES ALERT
# ============================================================

print("")
print("="*70)
print("CRITICAL ANOMALIES CHECK")
print("="*70)

critical_anomalies = anomaly_summary.filter(
    (col("anomaly_type").contains("CRITICAL")) |
    (col("anomaly_type").contains("Corruption")) |
    (col("anomaly_type").contains("Logical Conflict"))
).filter(col("count") > 0)

critical_count = critical_anomalies.count()

if critical_count > 0:
    print(f"Found {critical_count} critical anomalies that need attention:")
    display(critical_anomalies)
    print("")
    print("CRITICAL ISSUES DETECTED - REQUIRES CLEANING IN SILVER LAYER")
else:
    print("No critical anomalies detected")

# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================

print("")
print("="*70)
print("FINAL SUMMARY")
print("="*70)

total_anomalies = anomaly_summary.filter(col("count") > 0).count()
total_anomaly_count = anomaly_summary.agg(sum("count")).collect()[0][0] if anomaly_summary.count() > 0 else 0

user_events_anomalies = anomaly_summary.filter(col("table") == "User Events").filter(col("count") > 0).count()
user_events_total = anomaly_summary.filter(col("table") == "User Events").agg(sum("count")).collect()[0][0] if anomaly_summary.filter(col("table") == "User Events").count() > 0 else 0

ad_catalog_anomalies = anomaly_summary.filter(col("table") == "Ad Catalog").filter(col("count") > 0).count()
ad_catalog_total = anomaly_summary.filter(col("table") == "Ad Catalog").agg(sum("count")).collect()[0][0] if anomaly_summary.filter(col("table") == "Ad Catalog").count() > 0 else 0

print(f"""
ANOMALY DETECTION COMPLETE
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}
Total Anomaly Types Detected: {total_anomalies}
Total Anomaly Count: {total_anomaly_count:,}
Critical Anomalies: {critical_count}

User Events:
   - Total Rows: {df_clicks.count():,}
   - Anomaly Types: {user_events_anomalies}
   - Total Anomalies: {user_events_total:,}

Ad Catalog:
   - Total Rows: {df_catalog.count():,}
   - Anomaly Types: {ad_catalog_anomalies}
   - Total Anomalies: {ad_catalog_total:,}

Monitoring Tables Updated:
   - anomaly_report: adtech_catalog.monitoring.anomaly_report
   - version_history: adtech_catalog.monitoring.version_history

Next Steps:
   1. Review critical anomalies above
   2. Run: 03_CLEAN_TO_SILVER.py
   3. Run: 04_FEATURE_ENGINEERING.py
======================================================================
""")

print("")