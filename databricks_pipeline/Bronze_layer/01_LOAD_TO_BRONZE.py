# Databricks notebook source
# Databricks notebook source
# ============================================================
# LOAD RAW DATA TO BRONZE TABLES WITH METADATA
# ============================================================
# This notebook reads CSV files and creates Delta tables in Bronze layer
# 
# IMPROVEMENTS APPLIED:
# 1.  OVERWRITE for Monitoring Tables (clean, no duplicates)
# 2.  Idempotency Checks (file + table existence)
# 3.  Parameterized Runs (config-driven paths)
# 4.  Version Tracking (batch_id + environment + git_commit)

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime
import yaml
import os

# ============================================================
# 1. LOAD YAML CONFIGURATION (Parameterized)
# ============================================================

def load_yaml_config():
    """Load pipeline configuration from YAML file"""
    try:
        # Try local path first
        try:
            with open("pipeline_manifest.yaml", "r") as f:
                config = yaml.safe_load(f)
                print(" Loaded config from local path")
                return config
        except:
            pass
        
        # Try DBFS path
        try:
            config_path = "/Volumes/adtech_catalog/bronze/landing_zone/pipeline_manifest.yaml"
            config_content = dbutils.fs.head(config_path)
            config = yaml.safe_load(config_content)
            print(f" Loaded config from: {config_path}")
            return config
        except:
            pass
        
        print(" Config not found. Using default values.")
        return None
            
    except Exception as e:
        print(f" Could not load config: {e}")
        print("Using default values...")
        return None

config = load_yaml_config()

# ============================================================
# 2. DEFINE PATHS (Parameterized with fallbacks)
# ============================================================

# Get paths from config with fallbacks
if config:
    USER_EVENTS_PATH = config.get('paths', {}).get('user_events', 
        "/Volumes/adtech_catalog/bronze/landing_zone/raw_synthetic_ad_click_data.csv")
    AD_CATALOG_PATH = config.get('paths', {}).get('ad_catalog',
        "/Volumes/adtech_catalog/bronze/landing_zone/ad_catalog_raw.csv")
else:
    # Default paths
    USER_EVENTS_PATH = "/Volumes/adtech_catalog/bronze/landing_zone/raw_synthetic_ad_click_data.csv"
    AD_CATALOG_PATH = "/Volumes/adtech_catalog/bronze/landing_zone/ad_catalog_raw.csv"

# Get environment from config or default
ENVIRONMENT = config.get('environment', 'development') if config else 'development'

# Generate version (Version Tracking)
VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
GIT_COMMIT = os.environ.get('GIT_COMMIT', 'local')

print(f"""
📋 CONFIGURATION SUMMARY
======================================================================
Environment: {ENVIRONMENT}
Version: {VERSION}
Git Commit: {GIT_COMMIT}
User Events Path: {USER_EVENTS_PATH}
Ad Catalog Path: {AD_CATALOG_PATH}
======================================================================
""")

# ============================================================
# 3. CHECK FILE AVAILABILITY (Idempotency Check)
# ============================================================

print("\n" + "="*60)
print("CHECKING FILE AVAILABILITY")
print("="*60)

def check_file_exists(file_path, file_description):
    """Check if a file exists, exit gracefully if not"""
    try:
        dbutils.fs.ls(file_path)
        print(f" {file_description} found: {file_path}")
        return True
    except Exception as e:
        print(f" {file_description} NOT found at: {file_path}")
        print(f"   Error: {e}")
        print("   Please upload the file manually.")
        dbutils.notebook.exit(f"File not found: {file_description}")
        return False

# Check both files
user_events_exists = check_file_exists(USER_EVENTS_PATH, "User Events CSV")
ad_catalog_exists = check_file_exists(AD_CATALOG_PATH, "Ad Catalog CSV")

# ============================================================
# 4. READ CSV FILES
# ============================================================

print("\n" + "="*60)
print("READING CSV FILES")
print("="*60)

# Read user events with options to handle anomalies
df_user_events_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "true") \
    .option("escape", '"') \
    .option("mode", "PERMISSIVE") \
    .csv(USER_EVENTS_PATH)

# Read ad catalog
df_ad_catalog_raw = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("multiLine", "true") \
    .option("escape", '"') \
    .option("mode", "PERMISSIVE") \
    .csv(AD_CATALOG_PATH)

user_count = df_user_events_raw.count()
catalog_count = df_ad_catalog_raw.count()

print(f" User Events Loaded: {user_count:,} rows")
print(f" Ad Catalog Loaded: {catalog_count:,} rows")

# Validate data is not empty (Idempotency Check)
if user_count == 0:
    print(" User Events file is empty!")
    dbutils.notebook.exit("Empty file: User Events")
if catalog_count == 0:
    print(" Ad Catalog file is empty!")
    dbutils.notebook.exit("Empty file: Ad Catalog")

# Display sample data
print("\n User Events Sample:")
display(df_user_events_raw.limit(5))

print("\n Ad Catalog Sample:")
display(df_ad_catalog_raw.limit(5))

# Show schema
print("\n User Events Schema:")
df_user_events_raw.printSchema()

print("\n Ad Catalog Schema:")
df_ad_catalog_raw.printSchema()

# ============================================================
# 5. ADD INGESTION METADATA (Version Tracking)
# ============================================================

print("\n" + "="*60)
print("ADDING INGESTION METADATA")
print("="*60)

# Add metadata columns to user events
df_user_events_bronze = df_user_events_raw \
    .withColumn("ingestion_timestamp", current_timestamp()) \
    .withColumn("ingestion_date", current_date()) \
    .withColumn("source_file", lit("raw_synthetic_ad_click_data.csv")) \
    .withColumn("ingestion_batch_id", lit(VERSION)) \
    .withColumn("environment", lit(ENVIRONMENT)) \
    .withColumn("git_commit", lit(GIT_COMMIT)) \
    .withColumn("processing_status", lit("RAW"))

# Add metadata columns to ad catalog
df_ad_catalog_bronze = df_ad_catalog_raw \
    .withColumn("ingestion_timestamp", current_timestamp()) \
    .withColumn("ingestion_date", current_date()) \
    .withColumn("source_file", lit("ad_catalog_raw.csv")) \
    .withColumn("ingestion_batch_id", lit(VERSION)) \
    .withColumn("environment", lit(ENVIRONMENT)) \
    .withColumn("git_commit", lit(GIT_COMMIT)) \
    .withColumn("processing_status", lit("RAW"))

print(f" Added ingestion metadata to both datasets")
print(f"   Batch ID: {VERSION}")
print(f"   Environment: {ENVIRONMENT}")

# ============================================================
# 6. CHECK IF TABLES ALREADY EXIST (Idempotency Check)
# ============================================================

print("\n" + "="*60)
print("CHECKING EXISTING TABLES")
print("="*60)

def check_table_exists(table_name):
    """Check if a Delta table exists"""
    try:
        spark.sql(f"DESCRIBE {table_name}")
        return True
    except:
        return False

table_click_logs = "adtech_catalog.bronze.ad_click_logs"
table_catalog = "adtech_catalog.bronze.ad_metadata_catalog"

click_logs_exists = check_table_exists(table_click_logs)
catalog_exists = check_table_exists(table_catalog)

if click_logs_exists:
    existing_count = spark.table(table_click_logs).count()
    print(f" Table {table_click_logs} exists with {existing_count:,} rows")
    print(f"   Will be OVERWRITTEN with new data")
else:
    print(f" Table {table_click_logs} does not exist. Will be created.")

if catalog_exists:
    existing_count = spark.table(table_catalog).count()
    print(f" Table {table_catalog} exists with {existing_count:,} rows")
    print(f"   Will be OVERWRITTEN with new data")
else:
    print(f" Table {table_catalog} does not exist. Will be created.")

# ============================================================
# 7. WRITE TO BRONZE DELTA TABLES
# ============================================================

print("\n" + "="*60)
print("WRITING TO BRONZE TABLES")
print("="*60)

# Write user events to bronze
df_user_events_bronze.write \
    .mode("overwrite") \
    .format("delta") \
    .option("mergeSchema", "true") \
    .partitionBy("ingestion_date") \
    .saveAsTable(table_click_logs)

print(f" Created/Updated: {table_click_logs}")
print(f"   Rows: {df_user_events_bronze.count():,}")

# Write ad catalog to bronze
df_ad_catalog_bronze.write \
    .mode("overwrite") \
    .format("delta") \
    .option("mergeSchema", "true") \
    .partitionBy("ingestion_date") \
    .saveAsTable(table_catalog)

print(f" Created/Updated: {table_catalog}")
print(f"   Rows: {df_ad_catalog_bronze.count():,}")

# ============================================================
# 8. VERIFY TABLES
# ============================================================

print("\n" + "="*60)
print("VERIFYING BRONZE TABLES")
print("="*60)

def verify_table(table_name):
    """Verify table exists and has data"""
    try:
        df_check = spark.table(table_name)
        row_count = df_check.count()
        col_count = len(df_check.columns)
        print(f" Table: {table_name}")
        print(f"   Rows: {row_count:,}")
        print(f"   Columns: {col_count}")
        
        if row_count == 0:
            print(f"    WARNING: Table has 0 rows!")
            return False
        return True
    except Exception as e:
        print(f" Error reading {table_name}: {e}")
        return False

# Verify both tables
table1_ok = verify_table(table_click_logs)
table2_ok = verify_table(table_catalog)

if not (table1_ok and table2_ok):
    print("\n WARNING: Some tables failed verification!")
else:
    print("\n All tables verified successfully!")

# Show sample
print("\n Sample from bronze tables:")
display(spark.table(table_click_logs).select(
    "User_ID", "Ad_Reference_ID", "ingestion_date", "environment"
).limit(3))

# ============================================================
# 9. SAVE BATCH METADATA (FIXED - OVERWRITE)
# ============================================================

print("\n" + "="*60)
print("SAVING BATCH METADATA")
print("="*60)

try:
    # Create monitoring schema if not exists
    spark.sql("CREATE SCHEMA IF NOT EXISTS adtech_catalog.monitoring")
    
    # Create metadata DataFrame with extended columns
    batch_metadata = spark.createDataFrame([(
        datetime.now().isoformat(),
        "BRONZE_LOAD",
        "adtech_catalog",
        "ad_click_logs",
        int(df_user_events_bronze.count()),
        int(df_ad_catalog_bronze.count()),
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        "SUCCESS",
        "Bronze layer load from CSV files"
    )], [
        "batch_timestamp",
        "batch_type",
        "catalog_name",
        "table_name",
        "user_events_count",
        "ad_catalog_count",
        "batch_id",
        "environment",
        "git_commit",
        "status",
        "remarks"
    ])
    
    # FIXED: Use OVERWRITE instead of DROP TABLE + APPEND
    # This keeps only the latest run and prevents duplicates
    batch_metadata.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.batch_log")
    
    print(" Batch metadata saved to: adtech_catalog.monitoring.batch_log")
    print(f"   Batch ID: {VERSION}")
    print(f"   Environment: {ENVIRONMENT}")
    
except Exception as e:
    print(f" Could not save metadata: {e}")
    print("   Metadata displayed above only")

# ============================================================
# 10. SAVE VERSION INFO (Version Tracking)
# ============================================================

print("\n" + "="*60)
print("SAVING VERSION INFO")
print("="*60)

try:
    # Create version tracking table if not exists
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Bronze Layer - Initial Load",
        "SUCCESS"
    )], [
        "version_id",
        "environment",
        "git_commit",
        "deployed_at",
        "description",
        "status"
    ])
    
    # Append to version history (keep all versions)
    version_info.write \
        .mode("append") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.version_history")
    
    print(" Version info saved to: adtech_catalog.monitoring.version_history")
    print(f"   Version: {VERSION}")
    
except Exception as e:
    print(f" Could not save version info: {e}")

# ============================================================
# 11. SUMMARY REPORT
# ============================================================

print("\n" + "="*60)
print("BRONZE LAYER LOAD COMPLETE")
print("="*60)

print(f"""
 LOAD SUMMARY
======================================================================
User Events:
   - Source: {USER_EVENTS_PATH}
   - Rows: {df_user_events_bronze.count():,}
   - Columns: {len(df_user_events_bronze.columns)}
   - Table: {table_click_logs}

Ad Catalog:
   - Source: {AD_CATALOG_PATH}
   - Rows: {df_ad_catalog_bronze.count():,}
   - Columns: {len(df_ad_catalog_bronze.columns)}
   - Table: {table_catalog}

Metadata:
   - Batch ID: {VERSION}
   - Environment: {ENVIRONMENT}
   - Git Commit: {GIT_COMMIT}
   - Timestamp: {datetime.now().isoformat()}

Monitoring:
   - Batch Log: adtech_catalog.monitoring.batch_log (OVERWRITE - latest only)
   - Version History: adtech_catalog.monitoring.version_history (APPEND - all versions)

Status:  SUCCESS
======================================================================

NEXT STEPS:
   1.  Run: 02_DETECT_ANOMALIES_BRONZE.py
   2.  Run: 03_CLEAN_TO_SILVER.py
   3.  Run: 04_FEATURE_ENGINEERING.py
======================================================================
""")

# COMMAND ----------

# Databricks notebook source
# DATABRICKS NOTEBOOK: VERIFY_BRONZE_TABLES
# ============================================================
# BRONZE TABLES VERIFICATION
# ============================================================
# Purpose: Verify that Bronze tables were loaded correctly

from pyspark.sql.functions import col

print("="*70)
print("BRONZE TABLES VERIFICATION")
print("="*70)

print("\n1. TABLES IN BRONZE SCHEMA:")
display(spark.sql("SHOW TABLES IN adtech_catalog.bronze"))

print("\n2. ALL SCHEMAS IN ADTECH_CATALOG:")
display(spark.sql("SHOW SCHEMAS IN adtech_catalog"))

print("\n3. VERIFYING USER EVENTS TABLE")
print("-"*50)

try:
    df_clicks = spark.table("adtech_catalog.bronze.ad_click_logs")
    print("Table: adtech_catalog.bronze.ad_click_logs")
    print("Total Rows: {:,}".format(df_clicks.count()))
    print("")
    print("Schema:")
    df_clicks.printSchema()
    print("\nData Sample:")
    display(df_clicks.limit(3))
    
    print("\nDevice Corruption Check:")
    device_issues = df_clicks.filter(col("device").rlike(r'^\d+\s+')).count()
    print("Rows with device corruption: {:,}".format(device_issues))
    
    if device_issues > 0:
        print("Device corruption detected")
        print("\nSample corrupted rows:")
        display(df_clicks.filter(col("device").rlike(r'^\d+\s+')).select("device").limit(3))
    
except Exception as err:
    print("Error checking user events: {}".format(str(err)))

print("\n4. VERIFYING AD CATALOG TABLE")
print("-"*50)

try:
    df_catalog = spark.table("adtech_catalog.bronze.ad_metadata_catalog")
    print("Table: adtech_catalog.bronze.ad_metadata_catalog")
    print("Total Rows: {:,}".format(df_catalog.count()))
    print("")
    print("Schema:")
    df_catalog.printSchema()
    print("\nData Sample:")
    display(df_catalog.limit(3))
    
    print("\nDuplicate Check:")
    duplicate_ads = df_catalog.groupBy("Ad_Reference_ID").count().filter(col("count") > 1).count()
    print("Duplicate Ad_Reference_IDs: {:,}".format(duplicate_ads))
    
    if duplicate_ads > 0:
        print("Duplicates detected")
        print("\nSample duplicates:")
        display(df_catalog.groupBy("Ad_Reference_ID").count().filter(col("count") > 1).limit(3))
    
except Exception as err:
    print("Error checking ad catalog: {}".format(str(err)))

print("\n5. CHECKING MONITORING TABLES (NEW)")
print("-"*50)

try:
    # Check batch log
    batch_log = spark.table("adtech_catalog.monitoring.batch_log")
    print(" Batch Log exists!")
    print("   Rows: {:,}".format(batch_log.count()))
    print("\n   Latest entry:")
    display(batch_log.orderBy(col("batch_timestamp").desc()).limit(1))
except Exception as err:
    print("Batch Log not found: {}".format(str(err)))

try:
    # Check version history
    version_history = spark.table("adtech_catalog.monitoring.version_history")
    print(" Version History exists!")
    print("   Rows: {:,}".format(version_history.count()))
    print("\n   Latest version:")
    display(version_history.orderBy(col("deployed_at").desc()).limit(1))
except Exception as err:
    print("Version History not found: {}".format(str(err)))

print("\n6. VERIFY NEW METADATA COLUMNS")
print("-"*50)

try:
    df_clicks = spark.table("adtech_catalog.bronze.ad_click_logs")
    columns = df_clicks.columns
    new_columns = ["environment", "git_commit"]
    for col_name in new_columns:
        if col_name in columns:
            print(f" Column '{col_name}' exists")
        else:
            print(f" Column '{col_name}' NOT found")
except Exception as err:
    print("Error checking columns: {}".format(str(err)))

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

print("""
 SUMMARY
======================================================================
 Bronze tables loaded successfully
 Batch log saved (OVERWRITE - latest only)
 Version history saved (APPEND - all versions)
 New metadata columns added: environment, git_commit

NEXT STEPS:
   1. 02_DETECT_ANOMALIES_BRONZE.py
   2. 03_CLEAN_TO_SILVER.py
   3. 04_FEATURE_ENGINEERING.py
======================================================================
""")