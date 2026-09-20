db = db.getSiblingDB('lms_db');

db.createCollection('lms_clickstreams');

// Optional: Create indexes for performance
db.lms_clickstreams.createIndex({ "student_id": 1 });
db.lms_clickstreams.createIndex({ "timestamp": -1 });
