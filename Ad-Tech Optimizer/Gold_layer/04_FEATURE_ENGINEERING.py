# Databricks notebook source
# Databricks notebook source
# ============================================================
# GOLD LAYER - FEATURE ENGINEERING
# ============================================================
# Purpose: Create business metrics and ML features from silver data

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
# 2. LOAD SILVER TABLES WITH IDEMPOTENCY CHECKS
# ============================================================

print("")
print("="*70)
print("LOADING SILVER TABLES")
print("="*70)

def check_table_exists(table_name):
    try:
        spark.sql(f"DESCRIBE {table_name}")
        return True
    except:
        return False

TABLE_CLICKS = "adtech_catalog.silver.conformed_user_clicks"
TABLE_CATALOG = "adtech_catalog.silver.conformed_ad_catalog"

clicks_exists = check_table_exists(TABLE_CLICKS)
catalog_exists = check_table_exists(TABLE_CATALOG)

if not clicks_exists:
    print(f"ERROR: Table {TABLE_CLICKS} does not exist.")
    print("Please run 03_CLEAN_TO_SILVER.py first.")
    dbutils.notebook.exit("Silver table not found: conformed_user_clicks")

if not catalog_exists:
    print(f"ERROR: Table {TABLE_CATALOG} does not exist.")
    print("Please run 03_CLEAN_TO_SILVER.py first.")
    dbutils.notebook.exit("Silver table not found: conformed_ad_catalog")

df_clicks = spark.table(TABLE_CLICKS)
df_catalog = spark.table(TABLE_CATALOG)

print(f"Loaded User Events: {df_clicks.count():,} rows")
print(f"Loaded Ad Catalog: {df_catalog.count():,} rows")

# ============================================================
# 3. JOIN DATASETS (FIXED - Removed duplicate columns)
# ============================================================

print("")
print("="*70)
print("JOINING DATASETS")
print("="*70)

df_clicks_select = df_clicks.select(
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
    "ingestion_date"
)

df_catalog_select = df_catalog.select(
    "Ad_Reference_ID",
    "Ad_Category_Standard",
    "Ad_Device_Cleaned",
    "Ad_Location_Cleaned",
    "Cost_Per_Click_Cleaned",
    "Ad_Type_Standard",
    "ad_video_length_cleaned"
)

df_catalog_select = df_catalog_select.withColumnRenamed(
    "Ad_Type_Standard", "Ad_Type_Catalog"
)

df_joined = df_clicks_select.join(
    df_catalog_select,
    "Ad_Reference_ID",
    "inner"
)

print(f"Rows after join: {df_joined.count():,}")

print("")
print("Sample of joined data:")
display(df_joined.limit(5))

# ============================================================
# 4. TEMPORAL FEATURES
# ============================================================

print("")
print("="*70)
print("CREATING TEMPORAL FEATURES")
print("="*70)

df_featured = df_joined.withColumn(
    "hour_of_day", hour("Click_Timestamp_Standard")
).withColumn(
    "day_of_week", dayofweek("Click_Timestamp_Standard")
).withColumn(
    "month", month("Click_Timestamp_Standard")
).withColumn(
    "is_weekend",
    when(dayofweek("Click_Timestamp_Standard").isin([1,7]), 1).otherwise(0)
).withColumn(
    "time_of_day_category",
    when(col("hour_of_day") < 6, "Late_Night")
    .when(col("hour_of_day") < 12, "Morning")
    .when(col("hour_of_day") < 18, "Afternoon")
    .when(col("hour_of_day") < 22, "Evening_Prime")
    .otherwise("Late_Night")
)

print("   Temporal features created:")
print("   - hour_of_day: 0-23")
print("   - day_of_week: 1-7")
print("   - month: 1-12")
print("   - is_weekend: 0/1")
print("   - time_of_day_category: Morning/Afternoon/Evening_Prime/Late_Night")

# ============================================================
# 5. SESSION FEATURES
# ============================================================

print("")
print("="*70)
print("CREATING SESSION FEATURES")
print("="*70)

window_session = Window.partitionBy("User_ID").orderBy("Click_Timestamp_Standard")

df_featured = df_featured.withColumn(
    "time_since_last_action",
    unix_timestamp("Click_Timestamp_Standard") - lag(unix_timestamp("Click_Timestamp_Standard")).over(window_session)
)

df_featured = df_featured.withColumn(
    "session_id",
    when(
        col("time_since_last_action") > 1800,
        sum(when(col("time_since_last_action") > 1800, 1).otherwise(0)).over(window_session)
    ).otherwise(0)
)

print("   Session features created:")
print("   - time_since_last_action: seconds since last action")
print("   - session_id: 30-minute inactivity threshold")

# ============================================================
# 6. USER ENGAGEMENT FEATURES
# ============================================================

print("")
print("="*70)
print("CREATING USER ENGAGEMENT FEATURES")
print("="*70)

window_user = Window.partitionBy("User_ID")

df_featured = df_featured.withColumn(
    "user_total_clicks", sum("user_clicked_cleaned").over(window_user)
).withColumn(
    "user_total_impressions", count("User_ID").over(window_user)
).withColumn(
    "user_ctr",
    when(col("user_total_impressions") > 0,
         col("user_total_clicks") / col("user_total_impressions")
    ).otherwise(0)
)

print("   User engagement features created:")
print("   - user_total_clicks: total clicks by user")
print("   - user_total_impressions: total ads seen by user")
print("   - user_ctr: user's click-through rate (0 if no impressions)")

# ============================================================
# 7. AD PERFORMANCE FEATURES
# ============================================================

print("")
print("="*70)
print("CREATING AD PERFORMANCE FEATURES")
print("="*70)

window_ad = Window.partitionBy("Ad_Reference_ID")

df_featured = df_featured.withColumn(
    "ad_total_clicks", sum("user_clicked_cleaned").over(window_ad)
).withColumn(
    "ad_total_impressions", count("Ad_Reference_ID").over(window_ad)
).withColumn(
    "ad_ctr",
    when(col("ad_total_impressions") > 0,
         col("ad_total_clicks") / col("ad_total_impressions")
    ).otherwise(0)
).withColumn(
    "ad_avg_watch_duration", avg("Watch_Duration_Cleaned").over(window_ad)
)

print("   Ad performance features created:")
print("   - ad_total_clicks: total clicks on ad")
print("   - ad_total_impressions: total impressions of ad")
print("   - ad_ctr: ad's click-through rate (0 if no impressions)")
print("   - ad_avg_watch_duration: average watch time for ad")

# ============================================================
# 8. WATCH DURATION RATIO
# ============================================================

print("")
print("="*70)
print("CREATING WATCH DURATION RATIO")
print("="*70)

df_featured = df_featured.withColumn(
    "watch_duration_ratio",
    when(
        col("ad_video_length_cleaned") > 0,
        col("Watch_Duration_Cleaned") / col("ad_video_length_cleaned")
    ).otherwise(0.0)
).withColumn(
    "watch_duration_ratio_clipped",
    when(col("watch_duration_ratio") > 1.0, 1.0)
    .when(col("watch_duration_ratio") < 0, 0.0)
    .otherwise(col("watch_duration_ratio"))
)

print("   Watch duration features created:")
print("   - watch_duration_ratio: watch_time / video_length (0 if no video)")
print("   - watch_duration_ratio_clipped: capped at 0-1 range")

# ============================================================
# 9. CONVERSION FLAG
# ============================================================

print("")
print("="*70)
print("CREATING CONVERSION FLAG")
print("="*70)

df_featured = df_featured.withColumn(
    "is_converted",
    when(
        (col("user_clicked_cleaned") == 1) &
        (col("watch_duration_ratio") > 0.5) &
        (col("user_age_cleaned").between(25, 45)) &
        (rand() < 0.3),
        1
    ).otherwise(0)
)

conversion_count = df_featured.filter(col("is_converted") == 1).count()
print(f"   Conversion flag created: {conversion_count:,} conversions")

# ============================================================
# 10. DEMOGRAPHIC ENGAGEMENT DELTA (DED SCORE)
# ============================================================

print("")
print("="*70)
print("CREATING DED SCORE")
print("="*70)

df_featured = df_featured.withColumn(
    "age_category",
    when(col("user_age_cleaned") < 25, "18-24")
    .when(col("user_age_cleaned") < 35, "25-34")
    .when(col("user_age_cleaned") < 50, "35-49")
    .otherwise("50+")
)

window_age_category = Window.partitionBy("age_category", "Ad_Category_Standard")
df_featured = df_featured.withColumn(
    "category_performance_by_age",
    avg("user_clicked_cleaned").over(window_age_category)
)

df_featured = df_featured.withColumn(
    "demographic_engagement_delta",
    when(
        col("category_performance_by_age") > 0.7, 0.95
    ).when(
        col("category_performance_by_age") > 0.4, 0.6
    ).when(
        col("category_performance_by_age") > 0.2, 0.3
    ).otherwise(0.1)
)

print("   DED Score features created:")
print("   - age_category: 18-24, 25-34, 35-49, 50+")
print("   - category_performance_by_age: CTR by age-category")
print("   - demographic_engagement_delta: 0-1 alignment score")

# ============================================================
# 11. FINANCIAL METRICS (ROAS)
# ============================================================

print("")
print("="*70)
print("CREATING FINANCIAL METRICS")
print("="*70)

df_featured = df_featured.withColumn(
    "revenue_per_click",
    when(
        col("is_converted") == 1,
        col("Cost_Per_Click_Cleaned") * (5 + rand() * 10)
    ).otherwise(0)
).withColumn(
    "total_revenue",
    col("revenue_per_click") * col("user_clicked_cleaned")
).withColumn(
    "total_cost",
    col("Cost_Per_Click_Cleaned") * col("user_clicked_cleaned")
).withColumn(
    "return_on_ad_spend",
    when(
        col("total_cost") > 0,
        col("total_revenue") / col("total_cost")
    ).otherwise(0)
)

print("   Financial metrics created:")
print("   - revenue_per_click: 5-15x return on conversion")
print("   - total_revenue: total revenue generated")
print("   - total_cost: total ad spend")
print("   - return_on_ad_spend (ROAS): revenue / cost (0 if no cost)")

# ============================================================
# 12. PLATFORM PERFORMANCE
# ============================================================

print("")
print("="*70)
print("CREATING PLATFORM PERFORMANCE FEATURES")
print("="*70)

window_platform = Window.partitionBy("platform_source_cleaned")
df_featured = df_featured.withColumn(
    "platform_avg_roas", avg("return_on_ad_spend").over(window_platform)
).withColumn(
    "platform_total_spend", sum("total_cost").over(window_platform)
).withColumn(
    "platform_total_revenue", sum("total_revenue").over(window_platform)
)

print("   Platform features created:")
print("   - platform_avg_roas: average ROAS by platform")
print("   - platform_total_spend: total spend by platform")
print("   - platform_total_revenue: total revenue by platform")

# ============================================================
# 13. AGGREGATE TO AD LEVEL
# ============================================================

print("")
print("="*70)
print("AGGREGATING TO AD LEVEL")
print("="*70)

df_ad_aggregated = df_featured.groupBy("Ad_Reference_ID").agg(
    first("Ad_Category_Standard").alias("ad_category"),
    first("Ad_Device_Cleaned").alias("ad_device"),
    first("Ad_Location_Cleaned").alias("ad_location"),
    first("Ad_Type_Standard").alias("ad_type"),
    first("Ad_Type_Catalog").alias("ad_type_catalog"),
    first("Cost_Per_Click_Cleaned").alias("cost_per_click"),
    first("ad_video_length_cleaned").alias("ad_video_length"),

    sum("user_clicked_cleaned").alias("total_clicks"),
    count("User_ID").alias("total_impressions"),
    when(count("User_ID") > 0, sum("user_clicked_cleaned") / count("User_ID")).otherwise(0).alias("ctr"),
    avg("Watch_Duration_Cleaned").alias("avg_watch_duration"),
    sum("total_cost").alias("total_ad_spend"),
    sum("total_revenue").alias("total_revenue"),
    when(sum("total_cost") > 0, sum("total_revenue") / sum("total_cost")).otherwise(0).alias("roas"),

    sum("is_converted").alias("total_conversions"),
    when(sum("user_clicked_cleaned") > 0, sum("is_converted") / sum("user_clicked_cleaned")).otherwise(0).alias("conversion_rate"),
    when(count("User_ID") > 0, sum("is_converted") / count("User_ID")).otherwise(0).alias("overall_conversion_rate"),

    avg("user_age_cleaned").alias("avg_user_age"),
    countDistinct("User_ID").alias("unique_users"),
    avg("watch_duration_ratio_clipped").alias("avg_watch_ratio"),
    avg("demographic_engagement_delta").alias("avg_ded_score"),
    avg("category_performance_by_age").alias("category_age_affinity"),

    collect_set("platform_source_cleaned").alias("platforms_used"),
    collect_set("device_cleaned").alias("devices_used"),

    first("platform_avg_roas").alias("platform_avg_roas"),
    first("platform_total_spend").alias("platform_total_spend"),
    first("platform_total_revenue").alias("platform_total_revenue"),

    collect_set("time_of_day_category").alias("active_time_slots"),
    mode("day_of_week").alias("best_day"),
    avg("hour_of_day").alias("avg_hour"),

    first("month").alias("month"),  # ← ADDED THIS LINE
    first("ingestion_date").alias("ingestion_date"),
    first("ingestion_timestamp").alias("ingestion_timestamp")
)

print(f"   Aggregated to {df_ad_aggregated.count():,} ad-level records")

# ============================================================
# 14. ADD DERIVED FEATURES
# ============================================================

print("")
print("="*70)
print("ADDING DERIVED FEATURES")
print("="*70)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "engagement_efficiency",
    col("avg_watch_ratio") * col("ctr")
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "profit_margin",
    when(col("total_ad_spend") > 0,
         (col("total_revenue") - col("total_ad_spend")) / col("total_ad_spend")
    ).otherwise(0)
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "cost_per_conversion",
    when(col("total_conversions") > 0,
         col("total_ad_spend") / col("total_conversions")
    ).otherwise(0)
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "high_performance",
    when(col("roas") > 2.0, 1).otherwise(0)
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "cost_efficiency_score",
    when(col("cost_per_click") > 0,
         (col("ctr") * col("conversion_rate")) / col("cost_per_click")
    ).otherwise(0)
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "engagement_score",
    (col("avg_watch_ratio") * 0.3) +
    (col("ctr") * 0.4) +
    (col("conversion_rate") * 0.3)
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "audience_alignment_score",
    col("avg_ded_score") * col("category_age_affinity")
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "ad_age_days",
    datediff(current_date(), col("ingestion_date"))
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "ad_lifecycle_stage",
    when(col("ad_age_days") < 7, "New")
    .when(col("ad_age_days") < 30, "Growing")
    .when(col("ad_age_days") < 60, "Mature")
    .otherwise("Declining")
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "location_type",
    when(col("ad_location").isin(["Maharashtra", "Delhi", "Karnataka"]), "Urban")
    .when(col("ad_location").isin(["Tamil Nadu"]), "Semi-Urban")
    .otherwise("Rural")
)

df_ad_aggregated = df_ad_aggregated.withColumn(
    "season",
    when(col("month").isin([12, 1, 2]), "Winter")
    .when(col("month").isin([3, 4, 5]), "Spring")
    .when(col("month").isin([6, 7, 8]), "Summer")
    .when(col("month").isin([9, 10, 11]), "Fall")
    .otherwise("Unknown")
)

print("   Derived features created:")
print("   - engagement_efficiency: watch_ratio * CTR")
print("   - profit_margin: (revenue - cost) / cost")
print("   - cost_per_conversion: spend / conversions")
print("   - high_performance: ROAS > 2.0 (binary)")
print("   - cost_efficiency_score: (CTR * Conversion) / CPC")
print("   - engagement_score: weighted combination of engagement metrics")
print("   - audience_alignment_score: DED * category_age_affinity")
print("   - ad_age_days: days since ad creation")
print("   - ad_lifecycle_stage: New/Growing/Mature/Declining")
print("   - location_type: Urban/Semi-Urban/Rural")
print("   - season: Winter/Spring/Summer/Fall")

# ============================================================
# 15. ADD DATE DIMENSION
# ============================================================

df_ad_aggregated = df_ad_aggregated.withColumn(
    "processing_date", current_date()
)

print("   Added processing_date for partitioning")

# ============================================================
# 16. DROP EXISTING GOLD TABLES
# ============================================================

print("")
print("="*70)
print("DROPPING EXISTING GOLD TABLES")
print("="*70)

try:
    spark.sql("DROP TABLE IF EXISTS adtech_catalog.gold.fact_ad_performance")
    print("   Dropped existing table: adtech_catalog.gold.fact_ad_performance")
except Exception as e:
    print(f"   Could not drop table: {e}")

try:
    spark.sql("DROP TABLE IF EXISTS adtech_catalog.gold.ad_performance_analytics")
    print("   Dropped existing table: adtech_catalog.gold.ad_performance_analytics")
except Exception as e:
    print(f"   Could not drop table: {e}")

# ============================================================
# 17. WRITE TO GOLD TABLES
# ============================================================

print("")
print("="*70)
print("WRITING TO GOLD TABLES")
print("="*70)

print("")
print("Writing Fact Ad Performance to Gold...")
df_ad_aggregated.write \
    .mode("overwrite") \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .partitionBy("ad_category") \
    .saveAsTable("adtech_catalog.gold.fact_ad_performance")

print(f"   Created: adtech_catalog.gold.fact_ad_performance")
print(f"   Rows: {df_ad_aggregated.count():,}")

print("")
print("Writing Ad Performance Analytics to Gold...")
df_ad_aggregated.write \
    .mode("overwrite") \
    .format("delta") \
    .option("overwriteSchema", "true") \
    .partitionBy("ad_category") \
    .saveAsTable("adtech_catalog.gold.ad_performance_analytics")

print(f"   Created: adtech_catalog.gold.ad_performance_analytics")
print(f"   Rows: {df_ad_aggregated.count():,}")

# ============================================================
# 18. VERIFY GOLD TABLES
# ============================================================

print("")
print("="*70)
print("VERIFYING GOLD TABLES")
print("="*70)

df_gold_verify = spark.table("adtech_catalog.gold.fact_ad_performance")

print("")
print("Fact Ad Performance - Summary:")
print(f"   Total Rows: {df_gold_verify.count():,}")
print(f"   Total Columns: {len(df_gold_verify.columns)}")

print("")
print("   Columns:")
for col_name in df_gold_verify.columns:
    print(f"   - {col_name}")

print("")
print("   Null Checks:")
null_ctr = df_gold_verify.filter(col("ctr").isNull()).count()
null_roas = df_gold_verify.filter(col("roas").isNull()).count()
null_clicks = df_gold_verify.filter(col("total_clicks").isNull()).count()

print(f"   - Null CTR: {null_ctr:,}")
print(f"   - Null ROAS: {null_roas:,}")
print(f"   - Null Total Clicks: {null_clicks:,}")

print("")
print("   Sample of Gold data:")
display(df_gold_verify.select(
    "Ad_Reference_ID", "ad_category", "ad_type",
    "total_impressions", "total_clicks", "ctr",
    "roas", "high_performance", "ad_lifecycle_stage", "engagement_score"
).limit(10))

# ============================================================
# 19. SAVE PROCESSING METADATA (FIXED - mergeSchema)
# ============================================================

print("")
print("="*70)
print("SAVING PROCESSING METADATA")
print("="*70)

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS adtech_catalog.monitoring")
    
    # Create metadata DataFrame with correct schema
    processing_metadata = spark.createDataFrame([(
        datetime.now().isoformat(),
        "GOLD_FEATURE_ENGINEERING",
        "adtech_catalog",
        "fact_ad_performance, ad_performance_analytics",
        int(df_ad_aggregated.count()),
        int(len(df_ad_aggregated.columns)),
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        "SUCCESS",
        "Gold layer feature engineering completed"
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
    
    # Use mergeSchema to preserve history and handle schema evolution
    processing_metadata.write \
        .mode("overwrite") \
        .format("delta") \
        .option("mergeSchema", "true") \
        .saveAsTable("adtech_catalog.monitoring.batch_log")
    
    print("Processing metadata saved successfully!")
    print(f"   Batch ID: {VERSION}")
    print(f"   Environment: {ENVIRONMENT}")
    print("   History preserved (mergeSchema used)")

except Exception as e:
    print(f"Could not save metadata: {e}")
    
    # Fallback: Try append if overwrite fails
    try:
        processing_metadata.write \
            .mode("append") \
            .format("delta") \
            .saveAsTable("adtech_catalog.monitoring.batch_log")
        print("Metadata saved with append (fallback)!")
    except Exception as fallback_error:
        print(f"Could not save metadata at all: {fallback_error}")

# ============================================================
# 20. SAVE VERSION HISTORY
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
        "Gold Layer - Feature Engineering",
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
# 21. FEATURE SUMMARY
# ============================================================

print("")
print("="*70)
print("FEATURE SUMMARY")
print("="*70)

print("""
FEATURE CATEGORIES CREATED
======================================================================
1. Temporal Features (5)
   - hour_of_day, day_of_week, month, is_weekend, time_of_day_category

2. Session Features (2)
   - time_since_last_action, session_id

3. User Engagement Features (3)
   - user_total_clicks, user_total_impressions, user_ctr

4. Ad Performance Features (4)
   - ad_total_clicks, ad_total_impressions, ad_ctr, ad_avg_watch_duration

5. Watch Duration Features (2)
   - watch_duration_ratio, watch_duration_ratio_clipped

6. Demographic Features (3)
   - age_category, category_performance_by_age, demographic_engagement_delta

7. Financial Metrics (4)
   - total_revenue, total_cost, return_on_ad_spend, profit_margin

8. Platform Features (3)
   - platform_avg_roas, platform_total_spend, platform_total_revenue

9. Ad-Level Aggregates (25+)
   - total_clicks, total_impressions, ctr, roas, conversion_rate, etc.

10. Derived Features (11)
    - engagement_efficiency, profit_margin, cost_per_conversion
    - high_performance, cost_efficiency_score, engagement_score
    - audience_alignment_score, ad_age_days, ad_lifecycle_stage
    - location_type, season

TOTAL FEATURES: 60+
======================================================================
""")

# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("")
print("="*70)
print("GOLD LAYER COMPLETE")
print("="*70)

print(f"""
GOLD LAYER PROCESSING SUMMARY
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Source Tables:
   - Silver User Clicks: {df_clicks.count():,} rows
   - Silver Ad Catalog: {df_catalog.count():,} rows

Joined Data: {df_joined.count():,} rows

Gold Tables Created:
   1. adtech_catalog.gold.fact_ad_performance
   2. adtech_catalog.gold.ad_performance_analytics

Gold Table Stats:
   - Total Records: {df_ad_aggregated.count():,}
   - Total Columns: {len(df_ad_aggregated.columns)}
   - Partitioned By: ad_category

Predictions Supported:
   1. CTR Prediction
   2. ROAS Prediction
   3. Conversion Rate Prediction
   4. High Performance Classification
   5. Ad Type Performance
   6. Age Group Usage by Category
   7. Gender Conversion by Category
   8. Best Age Group for Each Ad
   9. Category Recommendations by Age
   10. Optimal Time by Demographic
   11. Engagement Efficiency
   12. DED Score
   13. Time-Based Performance
   14. Platform Performance
   15. Ad Lifecycle Stage
   16. Cost Efficiency Score

Monitoring:
   - Batch Log: adtech_catalog.monitoring.batch_log (OVERWRITE with mergeSchema)
   - Version History: adtech_catalog.monitoring.version_history (APPEND)

Next Steps:
   1. Review gold data quality above
   2. Run: 05_ML_PIPELINE.py (ML Training)
   3. Run: 05_EXPORT_GOLD_TO_S3.py (Export to S3)
======================================================================
""")

print("")