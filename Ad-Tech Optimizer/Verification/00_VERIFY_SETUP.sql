-- Databricks notebook source

-- Check Catalogs
SHOW CATALOGS;



-- COMMAND ----------

-- Check Schemas in adtech_catalog
SHOW SCHEMAS IN adtech_catalog;


-- COMMAND ----------


-- Check Bronze Tables (will show empty before running 01_LOAD_TO_BRONZE.py)
SHOW TABLES IN adtech_catalog.bronze;
