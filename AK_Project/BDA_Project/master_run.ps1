Write-Host "Starting Student Retention Pipeline Orchestration..." -ForegroundColor Cyan

# 1. Start Infrastructure
Write-Host "Step 1: Spinning up Docker containers..." -ForegroundColor Yellow
docker compose up -d
Write-Host "Waiting 30 seconds for databases to initialize..."
Start-Sleep -Seconds 30

# 2. Generate Synthetic Data
Write-Host "Step 2: Generating synthetic data..." -ForegroundColor Yellow
py scripts/generate_synthetic_data.py

# 3. Ingest Data to HDFS
Write-Host "Step 3: Uploading logs to HDFS..." -ForegroundColor Yellow
docker exec namenode hdfs dfs -mkdir -p /user/hdfs/raw_lms_data/
docker exec namenode hdfs dfs -put -f /home/data/lms_logs.csv /user/hdfs/raw_lms_data/

# 4. Batch Processing
Write-Host "Step 4: Running Hive aggregations..." -ForegroundColor Yellow
docker exec hive-server hive -f /home/scripts/hive_warehouse.sql

# 5. ML Analytics
Write-Host "Step 5: Running R ML engine..." -ForegroundColor Yellow
docker exec r_analytics Rscript /home/rstudio/scripts/predict_retention.R

# 6. Launch Dashboard
Write-Host "Step 6: Dashboard is ready at http://localhost:3838" -ForegroundColor Green
Write-Host "Pipeline execution complete." -ForegroundColor Green
