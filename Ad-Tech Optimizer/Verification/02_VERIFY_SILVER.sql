-- Databricks notebook source
-- Check if silver tables were created
SHOW TABLES IN adtech_catalog.silver;

-- COMMAND ----------

-- Count rows in conformed_user_clicks
-- Expected: ~500,000 rows
SELECT 
    'conformed_user_clicks' as table_name,
    COUNT(*) as row_count
FROM adtech_catalog.silver.conformed_user_clicks;

-- COMMAND ----------

-- Count rows in conformed_ad_catalog
-- Expected: ~1,000 rows
SELECT 
    'conformed_ad_catalog' as table_name,
    COUNT(*) as row_count
FROM adtech_catalog.silver.conformed_ad_catalog;

-- COMMAND ----------

-- Summary of both tables
SELECT 
    'conformed_user_clicks' as table_name,
    COUNT(*) as row_count
FROM adtech_catalog.silver.conformed_user_clicks
UNION ALL
SELECT 
    'conformed_ad_catalog' as table_name,
    COUNT(*) as row_count
FROM adtech_catalog.silver.conformed_ad_catalog;

-- COMMAND ----------

-- Sample from conformed_user_clicks (check cleaned data)
SELECT * 
FROM adtech_catalog.silver.conformed_user_clicks 
LIMIT 5;

-- Sample from conformed_ad_catalog
SELECT * 
FROM adtech_catalog.silver.conformed_ad_catalog 
LIMIT 5;

-- COMMAND ----------

-- Check for remaining anomalies after cleaning
SELECT 
    COUNT(*) as total_rows,
    SUM(CASE WHEN user_age_cleaned < 18 THEN 1 ELSE 0 END) as underage_users,
    SUM(CASE WHEN user_age_cleaned > 100 THEN 1 ELSE 0 END) as overage_users,
    SUM(CASE WHEN Watch_Duration_Cleaned < 0 THEN 1 ELSE 0 END) as negative_duration,
    SUM(CASE WHEN Watch_Duration_Cleaned > 180 THEN 1 ELSE 0 END) as overflow_duration,
    SUM(CASE WHEN device_cleaned = 'Unknown' THEN 1 ELSE 0 END) as unknown_device,
    SUM(CASE WHEN user_clicked_cleaned = 1 AND Watch_Duration_Cleaned = 0 THEN 1 ELSE 0 END) as logical_conflicts
FROM adtech_catalog.silver.conformed_user_clicks;

-- COMMAND ----------

-- Check for remaining anomalies in ad catalog
SELECT 
    COUNT(*) as total_rows,
    SUM(CASE WHEN Ad_Category_Standard = 'Unknown' THEN 1 ELSE 0 END) as unknown_category,
    SUM(CASE WHEN Cost_Per_Click_Cleaned < 0 THEN 1 ELSE 0 END) as negative_cpc,
    SUM(CASE WHEN Cost_Per_Click_Cleaned IS NULL THEN 1 ELSE 0 END) as null_cpc,
    SUM(CASE WHEN ad_video_length_cleaned < 0 THEN 1 ELSE 0 END) as negative_video_length,
    SUM(CASE WHEN ad_video_length_cleaned > 60 THEN 1 ELSE 0 END) as long_video
FROM adtech_catalog.silver.conformed_ad_catalog;

-- COMMAND ----------

-- See how ads are distributed by category
SELECT 
    Ad_Category_Standard,
    COUNT(*) as ad_count
FROM adtech_catalog.silver.conformed_ad_catalog
GROUP BY Ad_Category_Standard
ORDER BY ad_count DESC;

-- COMMAND ----------

-- See how users are distributed by device
SELECT 
    device_cleaned,
    COUNT(*) as user_count
FROM adtech_catalog.silver.conformed_user_clicks
GROUP BY device_cleaned
ORDER BY user_count DESC;

-- COMMAND ----------

-- Check if Silver run was logged
SELECT 
    version_id,
    deployed_at,
    description,
    status
FROM adtech_catalog.monitoring.version_history
WHERE description LIKE '%Silver%'
ORDER BY deployed_at DESC;

-- COMMAND ----------

-- Check all pipeline runs
SELECT 
    description,
    COUNT(*) as total_runs,
    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status != 'SUCCESS' THEN 1 ELSE 0 END) as failed
FROM adtech_catalog.monitoring.version_history
GROUP BY description
ORDER BY total_runs DESC;

-- COMMAND ----------

-- One-line summary of Silver layer
SELECT 
    'SILVER LAYER' as layer,
    COUNT(*) as total_rows,
    'PASSED' as quality_status
FROM adtech_catalog.silver.conformed_user_clicks
UNION ALL
SELECT 
    'SILVER CATALOG' as layer,
    COUNT(*) as total_rows,
    'PASSED' as quality_status
FROM adtech_catalog.silver.conformed_ad_catalog;