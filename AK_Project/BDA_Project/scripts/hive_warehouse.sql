-- Create Database
CREATE DATABASE IF NOT EXISTS edm_warehouse;
USE edm_warehouse;

-- External Table for SIS Data (Imported via Sqoop)
CREATE EXTERNAL TABLE IF NOT EXISTS student_master_ext (
    student_id STRING,
    name STRING,
    roll_no STRING,
    enrollment_status STRING,
    historical_gpa FLOAT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
LOCATION '/user/hdfs/sis_data/';

-- External Table for Cleaned LMS Logs (Processed by Pig)
CREATE EXTERNAL TABLE IF NOT EXISTS lms_logs_ext (
    student_id STRING,
    ts STRING,
    event_type STRING,
    duration INT,
    assignment_id STRING,
    submission_delay INT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
LOCATION '/user/hdfs/cleaned_lms_data/';

-- Create Feature Matrix Table
CREATE TABLE IF NOT EXISTS student_features AS
SELECT
    s.student_id,
    s.historical_gpa,
    COUNT(CASE WHEN l.event_type = 'login' THEN 1 END) AS total_logins,
    AVG(CASE WHEN l.event_type = 'assignment_submission' THEN l.submission_delay END) AS avg_delay_days,
    SUM(CASE WHEN l.event_type = 'lecture_video' THEN l.duration ELSE 0 END) / 3600.0 AS video_hours,
    COUNT(CASE WHEN l.event_type = 'forum_post' THEN 1 END) AS forum_activity_score,
    CASE WHEN s.enrollment_status = 'Withdrawn' THEN 1 ELSE 0 END AS label_dropped
FROM student_master_ext s
LEFT JOIN lms_logs_ext l ON s.student_id = l.student_id
GROUP BY s.student_id, s.historical_gpa, s.enrollment_status;

-- Export to CSV for R (This would typically be done via 'INSERT OVERWRITE DIRECTORY')
-- For this prototype, we assume the resulting table is exported to 'student_features.csv'
INSERT OVERWRITE LOCAL DIRECTORY '/home/data/export'
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
SELECT * FROM student_features;
