"""
Test MLflow Connection
Run this script to verify Databricks authentication
"""

import os
from dotenv import load_dotenv
import mlflow

# Load .env file
load_dotenv()

print("="*60)
print("MLflow Connection Test")
print("="*60)

# Check if environment variables are loaded
host = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")

print(f"\n1. Checking Environment Variables:")
print(f"   DATABRICKS_HOST: {host[:30] if host else '❌ NOT SET'}...")
print(f"   DATABRICKS_TOKEN: {'✅ SET' if token else '❌ NOT SET'}")

if not host or not token:
    print("\n❌ Environment variables not found!")
    print("   Please check your .env file:")
    print("   DATABRICKS_HOST=https://your-workspace.cloud.databricks.com")
    print("   DATABRICKS_TOKEN=dapi...")
    exit()

# Configure MLflow
print("\n2. Configuring MLflow...")
try:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    print("   ✅ MLflow configured")
except Exception as e:
    print(f"   ❌ MLflow configuration failed: {e}")
    exit()

# Test connection
print("\n3. Testing connection to Databricks...")
try:
    client = mlflow.tracking.MlflowClient()
    # Try to list registered models (only first 5)
    models = client.search_registered_models(max_results=5)
    print(f"   ✅ Connected successfully! Found {len(models)} models")
    for model in models[:3]:
        print(f"      - {model.name}")
except Exception as e:
    print(f"   ❌ Connection failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error details: {e}")

print("\n4. Testing model loading by alias...")
try:
    model_uri = "models:/adtech_catalog.ml_models.ctr_predictor@Production"
    model = mlflow.pyfunc.load_model(model_uri)
    print("   ✅ Successfully loaded ctr_predictor@Production")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")

print("\n" + "="*60)
print("Test Complete")