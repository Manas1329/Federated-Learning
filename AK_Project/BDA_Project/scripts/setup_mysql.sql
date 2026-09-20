CREATE DATABASE IF NOT EXISTS sis_db;
USE sis_db;

CREATE TABLE IF NOT EXISTS student_master (
    student_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100),
    roll_no VARCHAR(20),
    enrollment_status VARCHAR(20), -- 'Active', 'Withdrawn', 'Graduated'
    historical_gpa DECIMAL(3, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
