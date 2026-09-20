import mysql.connector
from pymongo import MongoClient
from faker import Faker
import random
import datetime
import json

fake = Faker()

# Configuration
MYSQL_CONFIG = {
    'user': 'root',
    'password': 'rootpassword',
    'host': 'localhost', # Use 'mysql' if running inside docker network
    'database': 'sis_db'
}

MONGO_URI = "mongodb://localhost:27017/" # Use 'mongodb' if running inside docker network

def generate_mysql_data(n=1000):
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()

    student_ids = []
    for i in range(n):
        s_id = f"STU{1000 + i}"
        student_ids.append(s_id)
        name = fake.name()
        roll_no = f"R{2023000 + i}"
        status = random.choices(['Active', 'Withdrawn'], weights=[0.9, 0.1])[0]
        gpa = round(random.uniform(2.0, 4.0), 2)

        cursor.execute(
            "INSERT INTO student_master (student_id, name, roll_no, enrollment_status, historical_gpa) VALUES (%s, %s, %s, %s, %s)",
            (s_id, name, roll_no, status, gpa)
        )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Generated {n} student records in MySQL.")
    return student_ids

def generate_mongodb_data(student_ids, n=50000):
    client = MongoClient(MONGO_URI)
    db = client['lms_db']
    collection = db['lms_clickstreams']

    event_types = ['login', 'assignment_submission', 'lecture_video', 'forum_post']

    records = []
    for _ in range(n):
        student_id = random.choice(student_ids)
        event_type = random.choice(event_types)
        timestamp = fake.date_time_between(start_date='-60d', end_date='now')

        duration = random.randint(30, 3600) if event_type in ['lecture_video', 'forum_post'] else None
        assignment_id = f"ASN{random.randint(1, 10)}" if event_type == 'assignment_submission' else None
        submission_delay = random.randint(-2, 7) if event_type == 'assignment_submission' else None

        record = {
            "student_id": student_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "duration": duration,
            "assignment_id": assignment_id,
            "submission_delay": submission_delay
        }
        records.append(record)

        if len(records) >= 1000:
            collection.insert_many(records)
            records = []

    if records:
        collection.insert_many(records)

    print(f"Generated {n} clickstream records in MongoDB.")

if __name__ == "__main__":
    try:
        s_ids = generate_mysql_data(1000)
        generate_mongodb_data(s_ids, 50000)
    except Exception as e:
        print(f"Error: {e}. Make sure databases are running and accessible.")
