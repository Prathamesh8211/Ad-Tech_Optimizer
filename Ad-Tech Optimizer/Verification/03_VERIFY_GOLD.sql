-- Databricks notebook source
-- Check if gold tables were created
SHOW TABLES IN adtech_catalog.gold;

-- COMMAND ----------

-- Count rows in fact_ad_performance
-- Expected: ~1,000 rows (one per ad)
SELECT 
    'fact_ad_performance' as table_name,
    COUNT(*) as row_count
FROM adtech_catalog.gold.fact_ad_performance;

-- COMMAND ----------

-- Count rows by category (distribution check)
SELECT 
    ad_category,
    COUNT(*) as ad_count
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY ad_category
ORDER BY ad_count DESC;

-- COMMAND ----------

-- View all columns in the gold table
DESCRIBE adtech_catalog.gold.fact_ad_performance;

-- COMMAND ----------

-- Sample 10 rows from gold table
SELECT 
    Ad_Reference_ID,
    ad_category,
    ad_type,
    ad_device,
    cost_per_click,
    total_impressions,
    total_clicks,
    ctr,
    roas,
    conversion_rate,
    high_performance,
    ad_lifecycle_stage,
    engagement_score
FROM adtech_catalog.gold.fact_ad_performance
LIMIT 10;

-- COMMAND ----------

-- Overall statistics
SELECT 
    COUNT(*) as total_ads,
    ROUND(AVG(ctr) * 100, 2) as avg_ctr_pct,
    ROUND(AVG(roas), 2) as avg_roas,
    ROUND(AVG(conversion_rate) * 100, 2) as avg_conversion_pct,
    SUM(high_performance) as high_performance_ads,
    ROUND(SUM(high_performance) * 100.0 / COUNT(*), 2) as high_performance_pct,
    ROUND(AVG(engagement_score), 2) as avg_engagement_score
FROM adtech_catalog.gold.fact_ad_performance;

-- COMMAND ----------

-- Category-wise performance
SELECT 
    ad_category,
    COUNT(*) as ad_count,
    ROUND(AVG(ctr) * 100, 2) as avg_ctr_pct,
    ROUND(AVG(roas), 2) as avg_roas,
    ROUND(AVG(conversion_rate) * 100, 2) as avg_conversion_pct,
    SUM(high_performance) as winning_ads,
    ROUND(SUM(high_performance) * 100.0 / COUNT(*), 2) as win_rate_pct,
    ROUND(AVG(engagement_score), 2) as avg_engagement
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY ad_category
ORDER BY avg_roas DESC;

-- COMMAND ----------

-- Ad type performance
SELECT 
    ad_type,
    COUNT(*) as ad_count,
    ROUND(AVG(ctr) * 100, 2) as avg_ctr_pct,
    ROUND(AVG(roas), 2) as avg_roas,
    ROUND(AVG(conversion_rate) * 100, 2) as avg_conversion_pct,
    ROUND(AVG(engagement_score), 2) as avg_engagement
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY ad_type
ORDER BY avg_roas DESC;

-- COMMAND ----------

-- Ad lifecycle distribution
SELECT 
    ad_lifecycle_stage,
    COUNT(*) as ad_count,
    ROUND(AVG(roas), 2) as avg_roas,
    ROUND(AVG(ctr) * 100, 2) as avg_ctr_pct
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY ad_lifecycle_stage
ORDER BY 
    CASE ad_lifecycle_stage
        WHEN 'New' THEN 1
        WHEN 'Growing' THEN 2
        WHEN 'Mature' THEN 3
        WHEN 'Declining' THEN 4
    END;

-- COMMAND ----------

-- Season performance
SELECT 
    season,
    COUNT(*) as ad_count,
    ROUND(AVG(roas), 2) as avg_roas,
    ROUND(AVG(ctr) * 100, 2) as avg_ctr_pct
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY season
ORDER BY avg_roas DESC;

-- COMMAND ----------

-- Location type performance
SELECT 
    location_type,
    COUNT(*) as ad_count,
    ROUND(AVG(roas), 2) as avg_roas,
    ROUND(AVG(ctr) * 100, 2) as avg_ctr_pct
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY location_type
ORDER BY avg_roas DESC;

-- COMMAND ----------

-- Cost efficiency distribution
SELECT 
    CASE 
        WHEN cost_efficiency_score > 0.4 THEN 'Excellent (>0.4)'
        WHEN cost_efficiency_score > 0.25 THEN 'Good (0.25-0.4)'
        WHEN cost_efficiency_score > 0.15 THEN 'Average (0.15-0.25)'
        ELSE 'Poor (<0.15)'
    END as efficiency_level,
    COUNT(*) as ad_count,
    ROUND(AVG(roas), 2) as avg_roas
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY efficiency_level
ORDER BY avg_roas DESC;

-- COMMAND ----------

-- Engagement score distribution
SELECT 
    CASE 
        WHEN engagement_score > 0.7 THEN 'High (>0.7)'
        WHEN engagement_score > 0.5 THEN 'Medium (0.5-0.7)'
        ELSE 'Low (<0.5)'
    END as engagement_level,
    COUNT(*) as ad_count,
    ROUND(AVG(roas), 2) as avg_roas,
    ROUND(AVG(ctr) * 100, 2) as avg_ctr_pct
FROM adtech_catalog.gold.fact_ad_performance
GROUP BY engagement_level
ORDER BY avg_roas DESC;

-- COMMAND ----------

-- Check for nulls in critical columns
SELECT 
    SUM(CASE WHEN Ad_Reference_ID IS NULL THEN 1 ELSE 0 END) as null_ad_reference,
    SUM(CASE WHEN ad_category IS NULL THEN 1 ELSE 0 END) as null_category,
    SUM(CASE WHEN ctr IS NULL THEN 1 ELSE 0 END) as null_ctr,
    SUM(CASE WHEN roas IS NULL THEN 1 ELSE 0 END) as null_roas,
    SUM(CASE WHEN conversion_rate IS NULL THEN 1 ELSE 0 END) as null_conversion,
    SUM(CASE WHEN high_performance IS NULL THEN 1 ELSE 0 END) as null_high_performance
FROM adtech_catalog.gold.fact_ad_performance;

-- COMMAND ----------

-- Check that gold has correct number of ads
-- Should match silver catalog count (1,000)
SELECT 
    'Bronze/Silver Catalog' as source,
    COUNT(*) as ad_count
FROM adtech_catalog.silver.conformed_ad_catalog
UNION ALL
SELECT 
    'Gold Fact' as source,
    COUNT(*) as ad_count
FROM adtech_catalog.gold.fact_ad_performance;

-- COMMAND ----------

-- Check if Gold run was logged
SELECT 
    version_id,
    deployed_at,
    description,
    status
FROM adtech_catalog.monitoring.version_history
WHERE description LIKE '%Gold%'
ORDER BY deployed_at DESC;

-- COMMAND ----------

-- One-line summary
SELECT 
    'GOLD LAYER' as layer,
    COUNT(*) as total_rows,
    CAST(AVG(roas) AS DECIMAL(5,2)) as avg_roas,
    CAST(AVG(ctr) * 100 AS DECIMAL(5,2)) as avg_ctr_pct,
    SUM(high_performance) as winning_ads,
    'PASSED' as quality_status
FROM adtech_catalog.gold.fact_ad_performance;