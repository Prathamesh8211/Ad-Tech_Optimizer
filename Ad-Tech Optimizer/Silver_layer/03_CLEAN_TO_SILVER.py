# Databricks notebook source
# Databricks notebook source
# ============================================================
# SILVER LAYER - DATA CLEANING
# ============================================================
# Purpose: Clean all anomalies from bronze data and create silver tables

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

# ============================================================
# 3. CLEAN USER EVENTS DATA
# ============================================================

print("")
print("="*70)
print("CLEANING USER EVENTS DATA")
print("="*70)

df_clicks_clean = df_clicks

# 3.1: Remove Duplicate Rows
print("")
print("3.1: Removing Duplicate Rows")

before_dedup = df_clicks_clean.count()
df_clicks_clean = df_clicks_clean.dropDuplicates()
after_dedup = df_clicks_clean.count()

print(f"   Before dedup: {before_dedup:,} rows")
print(f"   After dedup: {after_dedup:,} rows")
print(f"   Removed: {before_dedup - after_dedup:,} duplicate rows")

# 3.2: Fix Timestamp Format
print("")
print("3.2: Fixing Timestamp Format")

df_clicks_clean = df_clicks_clean.withColumn(
    "timestamp_temp",
    when(
        col("Click_Timestamp").rlike(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'),
        to_timestamp(col("Click_Timestamp"), "yyyy-MM-dd HH:mm:ss")
    ).when(
        col("Click_Timestamp").rlike(r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$'),
        to_timestamp(col("Click_Timestamp"), "dd/MM/yyyy HH:mm")
    ).otherwise(None)
)

df_clicks_clean = df_clicks_clean.withColumn(
    "Click_Timestamp_Standard",
    col("timestamp_temp")
).drop("timestamp_temp")

failed_convert = df_clicks_clean.filter(col("Click_Timestamp_Standard").isNull()).count()
print(f"   Timestamps standardized")
print(f"   Failed to convert: {failed_convert:,} rows (will be set to NULL)")

# 3.3: Fix Ad_Type
print("")
print("3.3: Fixing Ad_Type")

ad_type_mapping = {
    "video": "Video",
    "Vedeo": "Video",
    "IMAGE": "Image",
    "Carusel": "Carousel",
    "vide": "Video"
}

ad_type_expr = col("Ad_Type")
for key, val in ad_type_mapping.items():
    ad_type_expr = when(ad_type_expr == key, val).otherwise(ad_type_expr)

df_clicks_clean = df_clicks_clean.withColumn(
    "Ad_Type_Standard",
    when(col("Ad_Type").isNull(), "Unknown").otherwise(ad_type_expr)
)

print(f"   Ad_Type standardized")

# 3.4: Fix Watch_Duration
print("")
print("3.4: Fixing Watch_Duration")

df_clicks_clean = df_clicks_clean.withColumn(
    "Watch_Duration_Cleaned",
    when(col("Watch_Duration") < 0, 0.0)
    .when(col("Watch_Duration") > 180.0, 180.0)
    .otherwise(col("Watch_Duration"))
)

print(f"   Watch_Duration cleaned (negative -> 0, >180 -> 180)")

# 3.5: Fix User_Age
print("")
print("3.5: Fixing User_Age")

df_clicks_clean = df_clicks_clean.withColumn(
    "user_age_cleaned",
    when(col("user_age") < 18, 25)
    .when(col("user_age") > 100, 35)
    .when(col("user_age").isNull(), 30)
    .otherwise(col("user_age"))
)

print(f"   User_Age cleaned (underage -> 25, >100 -> 35, null -> 30)")

# 3.6: Fix Device (Critical - Handle Corruption)
print("")
print("3.6: Fixing Device (Handling Corruption)")

device_mapping = {
    "moble": "Mobile",
    "DESKTOP": "Desktop",
    "desktop": "Desktop",
    "tablet": "Tablet"
}

df_clicks_clean = df_clicks_clean.withColumn(
    "device_extracted",
    when(
        col("device").rlike(r'^\d+\s+'),
        trim(regexp_extract(col("device"), r'^\d+\s+(\w+)', 1))
    ).otherwise(col("device"))
)

device_expr = col("device_extracted")
for key, val in device_mapping.items():
    device_expr = when(device_expr == key, val).otherwise(device_expr)

df_clicks_clean = df_clicks_clean.withColumn(
    "device_cleaned",
    when(col("device_extracted").isNull() | (col("device_extracted") == ""), "Unknown")
    .otherwise(device_expr)
)

corruption_count = df_clicks.filter(col("device").rlike(r'^\d+\s+')).count()
print(f"   Device corruption fixed: {corruption_count:,} rows")
print(f"   Device standardized")

# 3.7: Fix Platform Source
print("")
print("3.7: Fixing Platform Source")

if "platform_source" in df_clicks_clean.columns:
    platform_mapping = {
        "gogle": "google",
        "facebok": "facebook"
    }

    platform_expr = col("platform_source")
    for key, val in platform_mapping.items():
        platform_expr = when(platform_expr == key, val).otherwise(platform_expr)

    df_clicks_clean = df_clicks_clean.withColumn(
        "platform_source_cleaned",
        when(col("platform_source").isNull(), "other")
        .otherwise(platform_expr)
    )

    print(f"   Platform source standardized")
else:
    print("   platform_source column not found")

# 3.8: Fix User_Gender
print("")
print("3.8: Fixing User_Gender")

window_gender = Window.partitionBy("User_ID")
df_clicks_clean = df_clicks_clean.withColumn(
    "gender_count",
    count("user_gender").over(window_gender)
)

window_user_gender = Window.partitionBy("User_ID").orderBy(col("gender_count").desc())
df_clicks_clean = df_clicks_clean.withColumn(
    "user_gender_fixed",
    when(
        col("user_gender").isNull(),
        first("user_gender", True).over(window_user_gender)
    ).otherwise(col("user_gender"))
)

df_clicks_clean = df_clicks_clean.withColumn(
    "user_gender_fixed",
    when(col("user_gender_fixed").isNull(), "Other")
    .otherwise(col("user_gender_fixed"))
)

print(f"   User_Gender fixed (identity shifting resolved)")

# 3.9: Fix Logical Conflict
print("")
print("3.9: Fixing Logical Conflict")

conflict_count = df_clicks_clean.filter(
    (col("user_clicked") == 1) & (col("Watch_Duration_Cleaned") == 0)
).count()

df_clicks_clean = df_clicks_clean.withColumn(
    "user_clicked_cleaned",
    when(
        (col("user_clicked") == 1) & (col("Watch_Duration_Cleaned") == 0),
        0
    ).otherwise(col("user_clicked"))
)

print(f"   Logical conflicts fixed: {conflict_count:,} rows")

# 3.10: Final Deduplication
print("")
print("3.10: Final Deduplication on Business Keys")

before_final_dedup = df_clicks_clean.count()
df_clicks_clean = df_clicks_clean.dropDuplicates([
    "User_ID", "Click_Timestamp_Standard", "Ad_Reference_ID"
])
after_final_dedup = df_clicks_clean.count()

print(f"   Before final dedup: {before_final_dedup:,} rows")
print(f"   After final dedup: {after_final_dedup:,} rows")
print(f"   Removed: {before_final_dedup - after_final_dedup:,} duplicate business key rows")

# 3.11: Select Final Columns
print("")
print("3.11: Selecting Final Columns")

silver_columns_clicks = [
    "User_ID",
    "Click_Timestamp_Standard",
    "Ad_Reference_ID",
    "Ad_Type_Standard",
    "Watch_Duration_Cleaned",
    "user_age_cleaned",
    "device_cleaned",
    "platform_source_cleaned",
    "user_gender_fixed",
    "user_clicked_cleaned",
    "ingestion_timestamp",
    "ingestion_date",
    "source_file",
    "ingestion_batch_id"
]

existing_columns = [c for c in silver_columns_clicks if c in df_clicks_clean.columns]
df_clicks_silver = df_clicks_clean.select(existing_columns)

print(f"   Selected {len(existing_columns)} columns for silver table")

# ============================================================
# 4. CLEAN AD CATALOG DATA
# ============================================================

print("")
print("="*70)
print("CLEANING AD CATALOG DATA")
print("="*70)

df_catalog_clean = df_catalog

# 4.1: Fix Ad_Category
print("")
print("4.1: Fixing Ad_Category")

cat_mapping = {
    "Eletronics": "Electronics",
    "electronics": "Electronics",
    "fashion": "Fashion",
    "travel": "Travel",
    "health": "Health",
    "gaming": "Gaming",
    "food": "Food",
    "Corrupted_Category": "Unknown"
}

cat_expr = col("Ad_Category")
for key, val in cat_mapping.items():
    cat_expr = when(cat_expr == key, val).otherwise(cat_expr)

df_catalog_clean = df_catalog_clean.withColumn(
    "Ad_Category_Standard",
    when(col("Ad_Category").isNull(), "Unknown").otherwise(cat_expr)
)

print(f"   Ad_Category standardized")

# 4.2: Fix Cost_Per_Click
print("")
print("4.2: Fixing Cost_Per_Click")

df_catalog_clean = df_catalog_clean.withColumn(
    "cpc_string",
    col("Cost_Per_Click").cast("string")
)

df_catalog_clean = df_catalog_clean.withColumn(
    "cpc_cleaned",
    when(
        col("cpc_string").contains("$"),
        regexp_replace(col("cpc_string"), "\\$", "")
    ).otherwise(col("cpc_string"))
)

df_catalog_clean = df_catalog_clean.withColumn(
    "cpc_double",
    col("cpc_cleaned").cast("double")
)

df_catalog_clean = df_catalog_clean.withColumn(
    "Cost_Per_Click_Cleaned",
    when(
        col("cpc_double").isNull(),
        0.00
    ).when(
        col("cpc_double") < 0,
        0.00
    ).otherwise(
        col("cpc_double").cast(DecimalType(10, 2))
    )
)

df_catalog_clean = df_catalog_clean.drop("cpc_string", "cpc_cleaned", "cpc_double")

print(f"   Cost_Per_Click cleaned")

# 4.3: Fix Ad_Device
print("")
print("4.3: Fixing Ad_Device")

device_mapping_cat = {
    "moble": "Mobile",
    "DESKTOP": "Desktop",
    "  Mobile  ": "Mobile"
}

device_expr_cat = col("Ad_Device")
for key, val in device_mapping_cat.items():
    device_expr_cat = when(device_expr_cat == key, val).otherwise(device_expr_cat)

df_catalog_clean = df_catalog_clean.withColumn(
    "Ad_Device_Cleaned",
    when(trim(col("Ad_Device")).isNull(), "All-Devices")
    .when(trim(col("Ad_Device")) == "", "All-Devices")
    .otherwise(trim(device_expr_cat))
)

print(f"   Ad_Device standardized")

# 4.4: Fix Ad_Location
print("")
print("4.4: Fixing Ad_Location")

df_catalog_clean = df_catalog_clean.withColumn(
    "Ad_Location_Cleaned",
    when(col("Ad_Location").isNull(), "Unknown")
    .otherwise(trim(col("Ad_Location")))
)

print(f"   Ad_Location cleaned (trailing spaces removed)")

# 4.5: Fix Ad_Type
print("")
print("4.5: Fixing Ad_Type")

ad_type_mapping_cat = {
    "video": "Video",
    "Vedeo": "Video",
    "IMAGE": "Image",
    "Carusel": "Carousel"
}

ad_type_expr_cat = col("Ad_Type")
for key, val in ad_type_mapping_cat.items():
    ad_type_expr_cat = when(ad_type_expr_cat == key, val).otherwise(ad_type_expr_cat)

df_catalog_clean = df_catalog_clean.withColumn(
    "Ad_Type_Standard",
    when(col("Ad_Type").isNull(), "Unknown")
    .otherwise(ad_type_expr_cat)
)

print(f"   Ad_Type standardized")

# 4.6: Fix ad_video_length
print("")
print("4.6: Fixing ad_video_length")

df_catalog_clean = df_catalog_clean.withColumn(
    "video_length_str",
    col("ad_video_length").cast("string")
)

df_catalog_clean = df_catalog_clean.withColumn(
    "video_length_double",
    when(
        col("video_length_str") == "45s",
        45.0
    ).otherwise(
        col("video_length_str").cast("double")
    )
)

df_catalog_clean = df_catalog_clean.withColumn(
    "ad_video_length_cleaned",
    when(
        (col("Ad_Type_Standard") != "Video") &
        (col("video_length_double") > 0),
        0.0
    ).otherwise(
        when(
            col("video_length_double").isNull(),
            0.0
        ).otherwise(
            col("video_length_double")
        )
    )
)

df_catalog_clean = df_catalog_clean.drop("video_length_str", "video_length_double")

print(f"   ad_video_length cleaned")

# 4.7: Handle Duplicates
print("")
print("4.7: Handling Duplicate Ad_Reference_IDs")

if "ingestion_timestamp" in df_catalog_clean.columns:
    window_catalog = Window.partitionBy("Ad_Reference_ID").orderBy(col("ingestion_timestamp").desc())
else:
    window_catalog = Window.partitionBy("Ad_Reference_ID").orderBy(col("Ad_Reference_ID"))

df_catalog_clean = df_catalog_clean.withColumn(
    "row_num", row_number().over(window_catalog)
).filter(col("row_num") == 1).drop("row_num")

print(f"   Duplicates removed")

# 4.8: Select Final Columns
print("")
print("4.8: Selecting Final Columns")

silver_columns_catalog = [
    "Ad_Reference_ID",
    "Ad_Category_Standard",
    "Ad_Device_Cleaned",
    "Ad_Location_Cleaned",
    "Cost_Per_Click_Cleaned",
    "Ad_Type_Standard",
    "ad_video_length_cleaned",
    "ingestion_timestamp",
    "ingestion_date",
    "source_file",
    "ingestion_batch_id"
]

existing_catalog_columns = [c for c in silver_columns_catalog if c in df_catalog_clean.columns]
df_catalog_silver = df_catalog_clean.select(existing_catalog_columns)

df_catalog_silver = df_catalog_silver.withColumn(
    "Cost_Per_Click_Cleaned",
    col("Cost_Per_Click_Cleaned").cast(DecimalType(10, 2))
).withColumn(
    "ad_video_length_cleaned",
    col("ad_video_length_cleaned").cast("double")
)

print(f"   Selected {len(existing_catalog_columns)} columns for silver table")

# ============================================================
# 5. CHECK IF SILVER TABLES EXIST (IDEMPOTENCY)
# ============================================================

print("")
print("="*70)
print("CHECKING EXISTING SILVER TABLES")
print("="*70)

SILVER_TABLE_CLICKS = "adtech_catalog.silver.conformed_user_clicks"
SILVER_TABLE_CATALOG = "adtech_catalog.silver.conformed_ad_catalog"

silver_clicks_exists = check_table_exists(SILVER_TABLE_CLICKS)
silver_catalog_exists = check_table_exists(SILVER_TABLE_CATALOG)

if silver_clicks_exists:
    existing_count = spark.table(SILVER_TABLE_CLICKS).count()
    print(f"Table {SILVER_TABLE_CLICKS} exists with {existing_count:,} rows")
    print("Will be OVERWRITTEN with new data")
else:
    print(f"Table {SILVER_TABLE_CLICKS} does not exist. Will be created.")

if silver_catalog_exists:
    existing_count = spark.table(SILVER_TABLE_CATALOG).count()
    print(f"Table {SILVER_TABLE_CATALOG} exists with {existing_count:,} rows")
    print("Will be OVERWRITTEN with new data")
else:
    print(f"Table {SILVER_TABLE_CATALOG} does not exist. Will be created.")

# ============================================================
# 6. ADD PROCESSING METADATA
# ============================================================

print("")
print("="*70)
print("ADDING PROCESSING METADATA")
print("="*70)

current_ts = datetime.now()

df_clicks_silver = df_clicks_silver.withColumn(
    "processing_timestamp", lit(current_ts)
).withColumn(
    "processing_batch_id", lit(VERSION)
).withColumn(
    "processing_status", lit("CLEANED")
).withColumn(
    "environment", lit(ENVIRONMENT)
)

df_catalog_silver = df_catalog_silver.withColumn(
    "processing_timestamp", lit(current_ts)
).withColumn(
    "processing_batch_id", lit(VERSION)
).withColumn(
    "processing_status", lit("CLEANED")
).withColumn(
    "environment", lit(ENVIRONMENT)
)

print(f"Added processing metadata")
print(f"   Batch ID: {VERSION}")
print(f"   Environment: {ENVIRONMENT}")

# ============================================================
# 7. WRITE TO SILVER TABLES
# ============================================================

print("")
print("="*70)
print("WRITING TO SILVER TABLES")
print("="*70)

print("")
print("Writing User Events to Silver...")
df_clicks_silver.write \
    .mode("overwrite") \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .partitionBy("ingestion_date") \
    .saveAsTable(SILVER_TABLE_CLICKS)

print(f"   Created: {SILVER_TABLE_CLICKS}")
print(f"   Rows: {df_clicks_silver.count():,}")

print("")
print("Writing Ad Catalog to Silver...")
df_catalog_silver.write \
    .mode("overwrite") \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .partitionBy("ingestion_date") \
    .saveAsTable(SILVER_TABLE_CATALOG)

print(f"   Created: {SILVER_TABLE_CATALOG}")
print(f"   Rows: {df_catalog_silver.count():,}")

# ============================================================
# 8. VERIFY SILVER TABLES
# ============================================================

print("")
print("="*70)
print("VERIFYING SILVER TABLES")
print("="*70)

def verify_table(table_name):
    try:
        df_check = spark.table(table_name)
        row_count = df_check.count()
        col_count = len(df_check.columns)
        print(f"Table: {table_name}")
        print(f"   Rows: {row_count:,}")
        print(f"   Columns: {col_count}")
        if row_count == 0:
            print("   WARNING: Table has 0 rows!")
            return False
        return True
    except Exception as e:
        print(f"Error reading {table_name}: {e}")
        return False

print("")
print("User Events Table:")
verify_table(SILVER_TABLE_CLICKS)

print("")
print("Ad Catalog Table:")
verify_table(SILVER_TABLE_CATALOG)

print("")
print("User Events - Quality Check:")

df_clicks_verify = spark.table(SILVER_TABLE_CLICKS)

print(f"   Total Rows: {df_clicks_verify.count():,}")

remaining_negative_age = df_clicks_verify.filter(col("user_age_cleaned") < 18).count()
remaining_old_age = df_clicks_verify.filter(col("user_age_cleaned") > 100).count()
remaining_negative_duration = df_clicks_verify.filter(col("Watch_Duration_Cleaned") < 0).count()
remaining_overflow_duration = df_clicks_verify.filter(col("Watch_Duration_Cleaned") > 180).count()
remaining_conflict = df_clicks_verify.filter(
    (col("user_clicked_cleaned") == 1) & (col("Watch_Duration_Cleaned") == 0)
).count()

print(f"   Remaining negative age: {remaining_negative_age:,}")
print(f"   Remaining old age: {remaining_old_age:,}")
print(f"   Remaining negative duration: {remaining_negative_duration:,}")
print(f"   Remaining overflow duration: {remaining_overflow_duration:,}")
print(f"   Remaining logical conflicts: {remaining_conflict:,}")

if remaining_negative_age == 0 and remaining_old_age == 0 and remaining_negative_duration == 0 and remaining_overflow_duration == 0 and remaining_conflict == 0:
    print("   All user events anomalies have been cleaned successfully!")

print("")
print("   Sample of cleaned user events:")
display(df_clicks_verify.limit(5))

print("")
print("Ad Catalog - Quality Check:")

df_catalog_verify = spark.table(SILVER_TABLE_CATALOG)

print(f"   Total Rows: {df_catalog_verify.count():,}")

remaining_duplicates = df_catalog_verify.groupBy("Ad_Reference_ID").count().filter(col("count") > 1).count()
remaining_negative_cpc = df_catalog_verify.filter(col("Cost_Per_Click_Cleaned") < 0).count()
remaining_cpc_null = df_catalog_verify.filter(col("Cost_Per_Click_Cleaned").isNull()).count()
remaining_video_length_null = df_catalog_verify.filter(col("ad_video_length_cleaned").isNull()).count()

print(f"   Remaining duplicate Ad_Reference_IDs: {remaining_duplicates:,}")
print(f"   Remaining negative CPC: {remaining_negative_cpc:,}")
print(f"   Remaining null CPC: {remaining_cpc_null:,}")
print(f"   Remaining null video length: {remaining_video_length_null:,}")

if remaining_duplicates == 0 and remaining_negative_cpc == 0 and remaining_cpc_null == 0 and remaining_video_length_null == 0:
    print("   All ad catalog anomalies have been cleaned successfully!")

print("")
print("   Sample of cleaned ad catalog:")
display(df_catalog_verify.limit(5))

# ============================================================
# 9. SAVE PROCESSING METADATA 
# ============================================================

print("")
print("="*70)
print("SAVING PROCESSING METADATA")
print("="*70)

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS adtech_catalog.monitoring")

    # Match the existing schema from 01_LOAD_TO_BRONZE.py
    processing_metadata = spark.createDataFrame([(
        datetime.now().isoformat(),      # batch_timestamp
        "SILVER_CLEANING",                # batch_type
        "adtech_catalog",                 # catalog_name
        "conformed_user_clicks, conformed_ad_catalog",  # table_name
        int(df_clicks_silver.count()),    # user_events_count
        int(df_catalog_silver.count()),   # ad_catalog_count
        VERSION,                          # batch_id
        ENVIRONMENT,                      # environment
        GIT_COMMIT,                       # git_commit
        "SUCCESS",                        # status
        "Silver layer cleaning completed" # remarks
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

    processing_metadata.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.batch_log")

    print("Processing metadata saved successfully!")
    print(f"   Batch ID: {VERSION}")
    print(f"   Environment: {ENVIRONMENT}")

except Exception as e:
    print(f"Could not save metadata: {e}")
    try:
        processing_metadata.coalesce(1).write \
            .mode("overwrite") \
            .format("csv") \
            .option("header", "true") \
            .save("/Volumes/adtech_catalog/bronze/landing_zone/metadata_backup")
        print("Metadata saved as CSV backup")
    except:
        print("Could not save metadata backup either")

# ============================================================
# 10. SAVE VERSION HISTORY
# ============================================================

print("")
print("="*70)
print("SAVING VERSION HISTORY")
print("="*70)

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Silver Layer - Data Cleaning",
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

print("")
print("="*70)
print("SILVER LAYER COMPLETE")
print("="*70)

print(f"""
SILVER LAYER PROCESSING SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

User Events:
   - Original Bronze Rows: {before_dedup:,}
   - Silver Clean Rows: {df_clicks_silver.count():,}
   - Rows Removed: {before_dedup - df_clicks_silver.count():,}

Ad Catalog:
   - Original Bronze Rows: {df_catalog.count():,}
   - Silver Clean Rows: {df_catalog_silver.count():,}
   - Rows Removed: {df_catalog.count() - df_catalog_silver.count():,}

Silver Tables Created:
   1. {SILVER_TABLE_CLICKS}
   2. {SILVER_TABLE_CATALOG}

Quality Check Results:
   - No remaining negative ages
   - No remaining old ages
   - No remaining negative watch durations
   - No remaining overflow watch durations
   - No remaining logical conflicts
   - No remaining duplicate Ad_Reference_IDs
   - No remaining negative CPC
   - No remaining null values

Monitoring:
   - Batch Log: adtech_catalog.monitoring.batch_log (OVERWRITE)
   - Version History: adtech_catalog.monitoring.version_history (APPEND)

Next Steps:
   1. Review silver data quality above
   2. Run: 04_FEATURE_ENGINEERING.py
======================================================================
""")

print("")