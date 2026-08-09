# Databricks notebook source
# Databricks notebook source
# ============================================================
# GOLD DATA QUALITY REVIEW
# ============================================================
# Purpose: Review and validate Gold layer data quality before ML

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
print("GOLD DATA QUALITY REVIEW")
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
GOLD TABLE SUMMARY
======================================================================
Table: {GOLD_TABLE}
Total Rows: {total_rows:,}
Total Columns: {total_cols}
Partitioned By: ad_category
======================================================================
""")

# ============================================================
# 3. SCHEMA INFORMATION
# ============================================================

print("SCHEMA INFORMATION")
print("-"*70)
df_gold.printSchema()

# ============================================================
# 4. NULL VALUE ANALYSIS
# ============================================================

print("")
print("NULL VALUE ANALYSIS")
print("-"*70)

null_report = []
for col_name in df_gold.columns:
    null_count = df_gold.filter(col(col_name).isNull()).count()
    null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
    null_report.append((col_name, null_count, null_pct))

null_df = spark.createDataFrame(null_report, ["column", "null_count", "null_percentage"])
null_df = null_df.orderBy(col("null_percentage").desc())

print("Columns with null values:")
display(null_df.filter(col("null_count") > 0))

if null_df.filter(col("null_count") > 0).count() == 0:
    print("No null values found in any column!")
else:
    print("Some columns have null values. Review above.")

# ============================================================
# 5. DUPLICATE CHECK
# ============================================================

print("")
print("DUPLICATE CHECK")
print("-"*70)

duplicate_count = df_gold.groupBy("Ad_Reference_ID").count().filter(col("count") > 1).count()

if duplicate_count == 0:
    print(f"No duplicate Ad_Reference_ID found ({total_rows} unique ads)")
else:
    print(f"Found {duplicate_count} duplicate Ad_Reference_IDs")

# ============================================================
# 6. CATEGORICAL COLUMN DISTRIBUTIONS
# ============================================================

print("")
print("CATEGORICAL COLUMN DISTRIBUTIONS")
print("-"*70)

cat_cols = ["ad_category", "ad_device", "ad_location", "ad_type", "ad_lifecycle_stage", "season", "location_type"]

for col_name in cat_cols:
    if col_name in df_gold.columns:
        print(f"\n{col_name.upper()} Distribution:")
        dist = df_gold.groupBy(col_name).count().orderBy(col("count").desc())
        display(dist)

# ============================================================
# 7. BUSINESS METRIC VALIDATION
# ============================================================

print("")
print("BUSINESS METRIC VALIDATION")
print("-"*70)

# CTR Statistics
print("CTR (Click-Through Rate) Statistics:")
df_gold.select("ctr").describe().show()

# ROAS Statistics
print("\nROAS (Return on Ad Spend) Statistics:")
df_gold.select("roas").describe().show()

# Conversion Rate Statistics
print("\nConversion Rate Statistics:")
df_gold.select("conversion_rate").describe().show()

# Engagement Score Statistics (New)
print("\nEngagement Score Statistics:")
df_gold.select("engagement_score").describe().show()

# Cost Efficiency Score Statistics (New)
print("\nCost Efficiency Score Statistics:")
df_gold.select("cost_efficiency_score").describe().show()

# High Performance
high_perf_count = df_gold.filter(col("high_performance") == 1).count()
high_perf_pct = (high_perf_count / total_rows) * 100 if total_rows > 0 else 0

print(f"\nHigh Performing Ads (ROAS > 2.0):")
print(f"   Count: {high_perf_count:,}")
print(f"   Percentage: {high_perf_pct:.1f}%")

# ============================================================
# 8. KEY METRICS SUMMARY
# ============================================================

print("")
print("KEY METRICS SUMMARY")
print("-"*70)

avg_metrics = df_gold.select(
    avg("ctr").alias("avg_ctr"),
    avg("roas").alias("avg_roas"),
    avg("conversion_rate").alias("avg_conversion_rate"),
    avg("engagement_score").alias("avg_engagement_score"),
    avg("cost_efficiency_score").alias("avg_cost_efficiency")
).collect()[0]

avg_ctr = avg_metrics.avg_ctr if avg_metrics.avg_ctr is not None else 0
avg_roas = avg_metrics.avg_roas if avg_metrics.avg_roas is not None else 0
avg_conversion = avg_metrics.avg_conversion_rate if avg_metrics.avg_conversion_rate is not None else 0
avg_engagement = avg_metrics.avg_engagement_score if avg_metrics.avg_engagement_score is not None else 0
avg_cost_eff = avg_metrics.avg_cost_efficiency if avg_metrics.avg_cost_efficiency is not None else 0

print(f"Average CTR: {avg_ctr:.4f}")
print(f"Average ROAS: {avg_roas:.2f}")
print(f"Average Conversion Rate: {avg_conversion:.4f}")
print(f"Average Engagement Score: {avg_engagement:.4f}")
print(f"Average Cost Efficiency Score: {avg_cost_eff:.4f}")

# ============================================================
# 9. DISTRIBUTION ANALYSIS (NEW)
# ============================================================

print("")
print("DISTRIBUTION ANALYSIS")
print("-"*70)

# Lifecycle stage distribution
print("Ad Lifecycle Stage Distribution:")
df_gold.groupBy("ad_lifecycle_stage").count().orderBy("count", ascending=False).show()

# Season distribution
print("\nSeason Distribution:")
df_gold.groupBy("season").count().orderBy("count", ascending=False).show()

# Location type distribution
print("\nLocation Type Distribution:")
df_gold.groupBy("location_type").count().orderBy("count", ascending=False).show()

# Engagement score distribution
print("\nEngagement Score Distribution:")
df_gold.select(
    when(col("engagement_score") > 0.7, "High (>0.7)")
    .when(col("engagement_score") > 0.5, "Medium (0.5-0.7)")
    .otherwise("Low (<0.5)").alias("engagement_level")
).groupBy("engagement_level").count().orderBy("count", ascending=False).show()

# Cost efficiency distribution
print("\nCost Efficiency Distribution:")
df_gold.select(
    when(col("cost_efficiency_score") > 0.4, "Excellent (>0.4)")
    .when(col("cost_efficiency_score") > 0.25, "Good (0.25-0.4)")
    .when(col("cost_efficiency_score") > 0.15, "Average (0.15-0.25)")
    .otherwise("Poor (<0.15)").alias("efficiency_level")
).groupBy("efficiency_level").count().orderBy("count", ascending=False).show()

# ============================================================
# 10. QUALITY CHECKS PASS/FAIL
# ============================================================

print("")
print("QUALITY CHECK RESULTS")
print("-"*70)

all_passed = True
check_results = []

# Check 1: No nulls in critical columns
critical_cols = ["Ad_Reference_ID", "ad_category", "ctr", "roas", "conversion_rate", "high_performance"]
for col_name in critical_cols:
    if col_name in df_gold.columns:
        null_count = df_gold.filter(col(col_name).isNull()).count()
        status = "PASS" if null_count == 0 else "FAIL"
        check_results.append((f"No nulls in {col_name}", status))
        if null_count > 0:
            all_passed = False

# Check 2: CTR range (0-1)
if "ctr" in df_gold.columns:
    invalid_ctr = df_gold.filter((col("ctr") < 0) | (col("ctr") > 1)).count()
    status = "PASS" if invalid_ctr == 0 else "FAIL"
    check_results.append(("CTR in valid range (0-1)", status))
    if invalid_ctr > 0:
        all_passed = False

# Check 3: ROAS non-negative
if "roas" in df_gold.columns:
    invalid_roas = df_gold.filter(col("roas") < 0).count()
    status = "PASS" if invalid_roas == 0 else "FAIL"
    check_results.append(("ROAS non-negative", status))
    if invalid_roas > 0:
        all_passed = False

# Check 4: Conversion rate range (0-1)
if "conversion_rate" in df_gold.columns:
    invalid_conversion = df_gold.filter((col("conversion_rate") < 0) | (col("conversion_rate") > 1)).count()
    status = "PASS" if invalid_conversion == 0 else "FAIL"
    check_results.append(("Conversion rate in valid range (0-1)", status))
    if invalid_conversion > 0:
        all_passed = False

# Check 5: Engagement score range (0-1)
if "engagement_score" in df_gold.columns:
    invalid_engagement = df_gold.filter((col("engagement_score") < 0) | (col("engagement_score") > 1)).count()
    status = "PASS" if invalid_engagement == 0 else "FAIL"
    check_results.append(("Engagement score in valid range (0-1)", status))
    if invalid_engagement > 0:
        all_passed = False

# Check 6: No duplicates
status = "PASS" if duplicate_count == 0 else "FAIL"
check_results.append(("No duplicate Ad_Reference_ID", status))
if duplicate_count > 0:
    all_passed = False

# Check 7: Row count
status = "PASS" if 800 <= total_rows <= 1200 else "WARN"
check_results.append(("Row count in expected range (800-1200)", status))

# Check 8: High performance rate (should be between 10-40%)
if total_rows > 0:
    hp_rate = (high_perf_count / total_rows) * 100
    status = "PASS" if 10 <= hp_rate <= 40 else "WARN"
    check_results.append((f"High performance rate ({hp_rate:.1f}%) in expected range (10-40%)", status))

# Display results
for check, status in check_results:
    if status == "PASS":
        print(f"PASS: {check}")
    elif status == "WARN":
        print(f"WARN: {check}")
    else:
        print(f"FAIL: {check}")

# ============================================================
# 11. QUALITY SCORE
# ============================================================

print("")
print("DATA QUALITY SCORE")
print("-"*70)

score = 100

# Penalty for nulls
null_cols = null_df.filter(col("null_count") > 0).count()
if null_cols > 0:
    penalty = null_cols * 5
    if penalty > 30:
        penalty = 30
    score = score - penalty

# Penalty for duplicates
if duplicate_count > 0:
    penalty = duplicate_count * 0.5
    if penalty > 10:
        penalty = 10
    score = score - penalty

# Penalty for invalid CTR
if "ctr" in df_gold.columns:
    invalid_ctr = df_gold.filter((col("ctr") < 0) | (col("ctr") > 1)).count()
    if invalid_ctr > 0:
        score = score - 10

# Penalty for invalid ROAS
if "roas" in df_gold.columns:
    invalid_roas = df_gold.filter(col("roas") < 0).count()
    if invalid_roas > 0:
        score = score - 10

# Ensure score not negative
if score < 0:
    score = 0

# Determine grade
if score >= 90:
    grade = "A (Excellent)"
elif score >= 80:
    grade = "B (Good)"
elif score >= 70:
    grade = "C (Fair)"
else:
    grade = "D (Poor)"

print(f"""
======================================================================
QUALITY SCORE: {score:.1f}% - {grade}

Grading Scale:
A (90-100) - Excellent, ready for ML
B (80-89)  - Good, minor issues to review
C (70-79)  - Fair, should fix before ML
D (0-69)   - Poor, must fix before ML
======================================================================
""")

# ============================================================
# 12. SUMMARY
# ============================================================

print("="*70)
print("SUMMARY")
print("="*70)

null_cols_count = null_df.filter(col("null_count") > 0).count()

print(f"""
GOLD DATA QUALITY REVIEW COMPLETE
======================================================================
Version: {VERSION}
Environment: {ENVIRONMENT}

Overview:
   - Total Rows: {total_rows:,}
   - Total Columns: {total_cols}
   - Null Values: {null_cols_count} columns have nulls
   - Duplicates: {duplicate_count}

Key Metrics:
   - Avg CTR: {avg_ctr:.4f}
   - Avg ROAS: {avg_roas:.2f}
   - Avg Conversion Rate: {avg_conversion:.4f}
   - Avg Engagement Score: {avg_engagement:.4f}
   - Avg Cost Efficiency: {avg_cost_eff:.4f}
   - High Performing Ads: {high_perf_count:,} ({high_perf_pct:.1f}%)

Quality Score: {score:.1f}% ({grade})

Overall Status: {'READY FOR ML PIPELINE' if score >= 80 else 'FIX ISSUES BEFORE ML'}
======================================================================
""")

# ============================================================
# 13. SAVE QUALITY REPORT
# ============================================================

print("SAVING QUALITY REPORT")
print("-"*70)

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS adtech_catalog.monitoring")

    quality_report = spark.createDataFrame([(
        datetime.now().isoformat(),
        total_rows,
        total_cols,
        null_cols_count,
        duplicate_count,
        float(avg_ctr),
        float(avg_roas),
        float(avg_conversion),
        float(avg_engagement),
        float(avg_cost_eff),
        high_perf_count,
        score,
        grade,
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        "PASS" if score >= 80 else "FAIL"
    )], [
        "review_timestamp",
        "total_rows",
        "total_columns",
        "columns_with_nulls",
        "duplicate_count",
        "avg_ctr",
        "avg_roas",
        "avg_conversion_rate",
        "avg_engagement_score",
        "avg_cost_efficiency",
        "high_performance_count",
        "quality_score",
        "quality_grade",
        "batch_id",
        "environment",
        "git_commit",
        "status"
    ])

    quality_report.write \
        .mode("overwrite") \
        .format("delta") \
        .saveAsTable("adtech_catalog.monitoring.gold_quality_report")

    print("Quality report saved to: adtech_catalog.monitoring.gold_quality_report")
    print(f"   Batch ID: {VERSION}")
    print(f"   Environment: {ENVIRONMENT}")

except Exception as e:
    print(f"Could not save quality report: {e}")

# ============================================================
# 14. SAVE VERSION HISTORY
# ============================================================

print("")
print("SAVING VERSION HISTORY")
print("-"*70)

try:
    version_info = spark.createDataFrame([(
        VERSION,
        ENVIRONMENT,
        GIT_COMMIT,
        datetime.now().isoformat(),
        "Gold Quality Review",
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
# 15. NEXT STEPS
# ============================================================

print("")
print("NEXT STEPS")
print("="*70)

if score >= 80:
    print("""
    Gold data quality is good.

    Next Steps:
    1. Run: 05_ML_PIPELINE.py
    2. Train 7 ML models
    3. Evaluate model performance
    4. Proceed to optimization and dashboard
    """)
else:
    print("""
    Issues found in gold data.

    Next Steps:
    1. Review the issues above
    2. Fix issues in 04_FEATURE_ENGINEERING.py
    3. Re-run 04_FEATURE_ENGINEERING.py
    4. Re-run this quality review
    5. Proceed to ML pipeline only after passing
    """)

print("="*70)
print("GOLD QUALITY REVIEW COMPLETE")
print("="*70)