#!/bin/bash

# Sqoop script to import MySQL student_master to HDFS
# Assuming run from within the Hadoop container or where Sqoop is installed

sqoop import \
  --connect jdbc:mysql://sis_mysql:3306/sis_db \
  --username root \
  --password rootpassword \
  --table student_master \
  --target-dir /user/hdfs/sis_data/ \
  --m 1 \
  --delete-target-dir \
  --fields-terminated-by ','
