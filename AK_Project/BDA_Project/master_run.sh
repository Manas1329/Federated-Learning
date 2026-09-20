#!/bin/bash

echo "Starting Student Retention Pipeline Orchestration..."

# 1. Start Infrastructure
echo "Step 1: Spinning up Docker containers..."
docker-compose up -d
sleep 30 # Wait for DBs to initialize

# 2. Generate Synthetic Data
echo "Step 2: Generating synthetic data..."
python3 scripts/generate_synthetic_data.py

# 3. Ingest Data to HDFS
echo "Step 3: Ingesting MySQL data via Sqoop..."
docker exec hadoop_master /home/scripts/sqoop_import.sh

# Note: In a real scenario, we'd also export MongoDB to HDFS here
# For the prototype, we assume the data is moved to /user/hdfs/raw_lms_data/

# 4. Batch Processing
echo "Step 4: Running Pig cleaning script..."
docker exec hadoop_master pig -f /home/scripts/clean_logs.pig

echo "Step 5: Running Hive aggregations..."
docker exec hadoop_master hive -f /home/scripts/hive_warehouse.sql

# 5. ML Analytics
echo "Step 6: Running R ML engine..."
docker exec r_analytics Rscript /home/rstudio/scripts/predict_retention.R

# 6. Launch Dashboard
echo "Step 7: Dashboard is ready at http://localhost:3838"
echo "Pipeline execution complete."
