-- Load raw clickstream logs from HDFS
-- Schema: student_id, timestamp, event_type, duration, assignment_id, submission_delay
raw_logs = LOAD '/user/hdfs/raw_lms_data/' USING PigStorage(',')
           AS (student_id:chararray, timestamp:chararray, event_type:chararray,
               duration:int, assignment_id:chararray, submission_delay:int);

-- Filter out records with missing student_id
filtered_logs = FILTER raw_logs BY student_id IS NOT NULL;

-- Compute basic cleaning: handle null durations and delays
cleaned_logs = FOREACH filtered_logs GENERATE
    student_id,
    timestamp,
    event_type,
    (duration IS NULL ? 0 : duration) AS duration,
    (assignment_id IS NULL ? 'N/A' : assignment_id) AS assignment_id,
    (submission_delay IS NULL ? 0 : submission_delay) AS submission_delay;

-- Store cleaned data back to HDFS for Hive
STORE cleaned_logs INTO '/user/hdfs/cleaned_lms_data/' USING PigStorage(',');
