# Databricks notebook source
# Test basic S3 access
try:
    dbutils.fs.ls("s3://adtech-optimizer-data/")
    print("✅ S3 access successful!")
except Exception as e:
    print(f"❌ Error: {e}")

# COMMAND ----------

# Databricks notebook source
# ============================================================
# EXPORT GOLD TABLE TO AWS S3
# ============================================================
# Purpose: Export Gold table to S3 as Parquet with validation

from pyspark.sql.functions import *
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
# 2. LOAD GOLD DATA WITH IDEMPOTENCY CHECK
# ============================================================

print("="*70)
print("EXPORTING GOLD DATA TO S3")
print("="*70)

def check_table_exists(table_name):
    try:
        spark.sql(f"DESCRIBE {table_name}")
        return True
    except:
        return False

GOLD_TABLE = "adtech_catalog.gold.fact_ad_performance"

if not check_table_exists(GOLD_TABLE):
    print(f"ERROR: Table {GOLD_TABLE} does not exist.")
    print("Please run 04_FEATURE_ENGINEERING.py first.")
    dbutils.notebook.exit("Gold table not found: fact_ad_performance")

df_gold = spark.table(GOLD_TABLE)

total_rows = df_gold.count()
total_cols = len(df_gold.columns)

print(f"""
GOLD DATA SUMMARY
======================================================================
Table: {GOLD_TABLE}
Total Rows: {total_rows:,}
Total Columns: {total_cols}
======================================================================
""")

# ============================================================
# 3. ADD PARTITION COLUMNS
# ============================================================

print("")
print("Adding partition columns...")

current_ts = datetime.now()

df_export = df_gold.withColumn(
    "year", year("processing_date")
).withColumn(
    "month", month("processing_date")
).withColumn(
    "day", dayofmonth("processing_date")
).withColumn(
    "export_timestamp", lit(current_ts.isoformat())
).withColumn(
    "export_version", lit(VERSION)
).withColumn(
    "environment", lit(ENVIRONMENT)
)

print("Partition columns: year, month, day")
print(f"Export Version: {VERSION}")
print(f"Environment: {ENVIRONMENT}")

# ============================================================
# 4. CONFIGURE S3 PATHS
# ============================================================

print("")
print("Configuring S3 paths...")

# New bucket name - more professional and descriptive
BUCKET_NAME = "adtech-optimizer-data"
S3_PATH = f"s3://{BUCKET_NAME}/gold/fact_ad_performance/"

print(f"S3 Path: {S3_PATH}")
print(f"Bucket: {BUCKET_NAME}")

# ============================================================
# 5. CHECK IF DATA ALREADY EXISTS
# ============================================================

print("")
print("Checking if data already exists...")

try:
    existing_files = dbutils.fs.ls(S3_PATH)
    print(f"Existing data found at: {S3_PATH}")
    print("It will be OVERWRITTEN")
except Exception as e:
    print("No existing data found. Creating new.")

# ============================================================
# 6. WRITE TO S3
# ============================================================

print("")
print("Writing to S3...")

try:
    df_export.write \
        .mode("overwrite") \
        .format("parquet") \
        .option("compression", "snappy") \
        .partitionBy("year", "month", "day") \
        .save(S3_PATH)

    print("Data exported successfully!")
    print(f"   Path: {S3_PATH}")
    print(f"   Rows: {total_rows:,}")

except Exception as e:
    print(f"Export failed: {e}")
    print("")
    print("Troubleshooting:")
    print("1. Check IAM role permissions")
    print("2. Check bucket exists: adtech-optimizer-data")
    print("3. Check External Location configuration")
    raise

# ============================================================
# 7. VERIFY EXPORT
# ============================================================

print("")
print("Verifying export...")

try:
    df_verify = spark.read.parquet(S3_PATH)
    verify_count = df_verify.count()
    verify_cols = len(df_verify.columns)

    print("Verification successful!")
    print(f"   Rows in S3: {verify_count:,}")
    print(f"   Columns in S3: {verify_cols}")

    if verify_count == total_rows:
        print("   Row count matches Gold table!")
    else:
        print(f"   Row count mismatch. Expected: {total_rows}, Found: {verify_count}")

except Exception as e:
    print(f"Verification failed: {e}")

# ============================================================
# 8. LIST FILES IN S3
# ============================================================

print("")
print("Files in S3:")

try:
    files = dbutils.fs.ls(S3_PATH)
    print(f"   Total items: {len(files)}")

    parquet_files = [f for f in files if f.name.endswith('.parquet')]
    print(f"   Parquet files: {len(parquet_files)}")

    if parquet_files:
        print("")
        print("   Sample files:")
        for f in parquet_files[:5]:
            size_mb = f.size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")

    if len(parquet_files) > 5:
        print(f"   ... and {len(parquet_files) - 5} more files")

except Exception as e:
    print(f"Could not list files: {e}")

# ============================================================
# 9. SAMPLE DATA FROM S3
# ============================================================

print("")
print("Sample data from S3:")

try:
    df_sample = spark.read.parquet(S3_PATH).limit(5)
    display(df_sample)
except Exception as e:
    print(f"Could not display sample: {e}")

# ============================================================
# 10. SAVE EXPORT METADATA
# ============================================================

print("")
print("Saving export metadata...")

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS adtech_catalog.monitoring")

    export_metadata = spark.createDataFrame([(
        current_ts.isoformat(),
        S3_PATH,
        total_rows,
        total_cols,
        "fact_ad_performance",
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        "SUCCESS"
    )], [
        "export_timestamp",
        "s3_path",
        "row_count",
        "column_count",
        "table_name",
        "export_version",
        "environment",
        "git_commit",
        "status"
    ])

    export_metadata.write \
        .mode("append") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.s3_export_log")

    print("Export metadata saved to: adtech_catalog.monitoring.s3_export_log")
    print(f"   Export Version: {VERSION}")
    print(f"   Environment: {ENVIRONMENT}")

except Exception as e:
    print(f"Could not save export metadata: {e}")

# ============================================================
# 11. SAVE VERSION HISTORY
# ============================================================

print("")
print("Saving version history...")

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Export Gold to S3",
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
# 12. SUMMARY
# ============================================================

print("")
print("="*70)
print("EXPORT COMPLETE")
print("="*70)

print(f"""
EXPORT SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

S3 Path: {S3_PATH}
Rows: {total_rows:,}
Columns: {total_cols}
Format: Parquet (Snappy compression)
Partitions: year/month/day
Export Version: {VERSION}

S3 Bucket: {BUCKET_NAME}

Folder Structure:
   s3://{BUCKET_NAME}/
   └── gold/
       └── fact_ad_performance/
           ├── year=2026/
           │   └── month=07/
           │       └── day=31/
           │           └── part-*.snappy.parquet
           └── _SUCCESS

Monitoring:
   - Export Log: adtech_catalog.monitoring.s3_export_log
   - Version History: adtech_catalog.monitoring.version_history

Next Steps:
   1. Verify data in AWS S3 Console
   2. Run: ML Training (01_CTR_PREDICTION.py)
   3. Build: Streamlit Dashboard
======================================================================
""")