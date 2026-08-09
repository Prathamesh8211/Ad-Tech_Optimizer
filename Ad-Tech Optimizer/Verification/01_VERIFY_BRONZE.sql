-- Databricks notebook source
-- Count rows in ad_click_logs
-- Expected: ~505,000 rows
SELECT COUNT(*) as ad_click_logs_count
FROM adtech_catalog.bronze.ad_click_logs;

-- COMMAND ----------

-- Count rows in ad_metadata_catalog
-- Expected: ~1,050 rows
SELECT COUNT(*) as ad_metadata_catalog_count
FROM adtech_catalog.bronze.ad_metadata_catalog;

-- COMMAND ----------

-- Summary of both tables
SELECT 
    'ad_click_logs' as table_name,
    COUNT(*) as row_count
FROM adtech_catalog.bronze.ad_click_logs
UNION ALL
SELECT 
    'ad_metadata_catalog' as table_name,
    COUNT(*) as row_count
FROM adtech_catalog.bronze.ad_metadata_catalog;

-- COMMAND ----------

-- Sample data from ad_click_logs (check for anomalies)
SELECT * 
FROM adtech_catalog.bronze.ad_click_logs 
LIMIT 5;

-- COMMAND ----------

-- View all columns in anomaly_report
DESCRIBE adtech_catalog.monitoring.anomaly_report;

-- COMMAND ----------

-- All anomalies with their counts (sorted by count)
-- Shows what data quality issues were found
SELECT 
    table,
    anomaly_type,
    count,
    detection_timestamp
FROM adtech_catalog.monitoring.anomaly_report
ORDER BY count DESC;

-- COMMAND ----------

-- Summary of anomalies by pipeline run
-- Shows total anomalies found in each run
SELECT 
    batch_id,
    MAX(detection_timestamp) as detection_time,
    SUM(count) as total_anomalies
FROM adtech_catalog.monitoring.anomaly_report
GROUP BY batch_id
ORDER BY detection_time DESC;

-- COMMAND ----------

-- Critical anomalies only (need immediate attention)
SELECT 
    anomaly_type,
    count,
    detection_timestamp
FROM adtech_catalog.monitoring.anomaly_report
WHERE anomaly_type LIKE '%CRITICAL%' 
   OR anomaly_type LIKE '%Corruption%'
   OR anomaly_type LIKE '%Logical Conflict%'
ORDER BY count DESC;

-- COMMAND ----------

-- View complete version history
SELECT * 
FROM adtech_catalog.monitoring.version_history 
ORDER BY deployed_at DESC;

-- COMMAND ----------

-- Show only anomaly detection runs
SELECT 
    version_id,
    deployed_at,
    description,
    status
FROM adtech_catalog.monitoring.version_history
WHERE description LIKE '%Anomaly%'
ORDER BY deployed_at DESC;

-- COMMAND ----------

-- Show only bronze load runs
SELECT 
    version_id,
    deployed_at,
    description,
    status
FROM adtech_catalog.monitoring.version_history
WHERE description LIKE '%Bronze%'
ORDER BY deployed_at DESC;

-- COMMAND ----------

-- Pipeline health summary
SELECT 
    'Bronze Layer' as layer,
    CASE 
        WHEN COUNT(*) > 0 THEN ' EXISTS'
        ELSE ' MISSING'
    END as status,
    COUNT(*) as row_count
FROM adtech_catalog.bronze.ad_click_logs
UNION ALL
SELECT 
    'Anomaly Report' as layer,
    CASE 
        WHEN COUNT(*) > 0 THEN ' EXISTS'
        ELSE ' MISSING'
    END as status,
    COUNT(*) as row_count
FROM adtech_catalog.monitoring.anomaly_report
UNION ALL
SELECT 
    'Version History' as layer,
    CASE 
        WHEN COUNT(*) > 0 THEN ' EXISTS'
        ELSE ' MISSING'
    END as status,
    COUNT(*) as row_count
FROM adtech_catalog.monitoring.version_history;
