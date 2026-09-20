import sys
import os
from pathlib import Path
import random
import uuid
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from federated_healthcare.webapp.database.connection import Base, engine, SessionLocal
from federated_healthcare.webapp.models.db_models import (
    User, UserRole, Hospital, TrustStatus, TrainingStatus,
    GlobalTrainingRound, TrainingRound, ModelUpdate,
    UnlearningRequest, AuditLog, AuditEventType, SystemMetrics
)
from federated_healthcare.webapp.core.security import get_password_hash
from federated_healthcare.webapp.core.config import settings


def init_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        print("[+] Seeding Federated Healthcare AI Platform Database...\n")

        admin = User(
            username=settings.ADMIN_USERNAME,
            email=settings.ADMIN_EMAIL,
            hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
            full_name="Super Administrator",
            role=UserRole.ADMIN,
            is_active=True,
            created_at=datetime.utcnow() - timedelta(days=90)
        )
        db.add(admin)
        db.flush()
        print(f"[OK] Created admin user: {settings.ADMIN_USERNAME} / {settings.ADMIN_PASSWORD}")

        hospitals_data = [
            {
                "name": "Hospital A (Mumbai)",
                "code": "HOSP-A",
                "location": "Mumbai, India",
                "contact_email": "admin@hospitala.org",
                "contact_phone": "+91-22-0000-0001",
                "trust_score": 0.95,
                "trust_status": TrustStatus.TRUSTED,
                "total_patients": 3200,
                "total_training_samples": 1250,
                "current_accuracy": 93.6,
                "dp_epsilon": 1.0,
                "quantization_enabled": True,
                "username": "hospital_a",
                "password": "hospital123",
                "full_name": "Dr. Sharma",
                "doctor_email": "dr.sharma@hospitala.org"
            },
            {
                "name": "Hospital B (Delhi)",
                "code": "HOSP-B",
                "location": "New Delhi, India",
                "contact_email": "admin@hospitalb.org",
                "contact_phone": "+91-11-0000-0002",
                "trust_score": 0.89,
                "trust_status": TrustStatus.TRUSTED,
                "total_patients": 4500,
                "total_training_samples": 1750,
                "current_accuracy": 92.1,
                "dp_epsilon": 1.0,
                "quantization_enabled": True,
                "username": "hospital_b",
                "password": "hospital123",
                "full_name": "Dr. Verma",
                "doctor_email": "dr.verma@hospitalb.org"
            },
            {
                "name": "Hospital C (Bangalore)",
                "code": "HOSP-C",
                "location": "Bangalore, India",
                "contact_email": "admin@hospitalc.org",
                "contact_phone": "+91-80-0000-0003",
                "trust_score": 0.42,
                "trust_status": TrustStatus.SUSPICIOUS,
                "total_patients": 2800,
                "total_training_samples": 1100,
                "current_accuracy": 81.3,
                "dp_epsilon": 1.0,
                "quantization_enabled": True,
                "username": "hospital_c",
                "password": "hospital123",
                "full_name": "Dr. Iyer",
                "doctor_email": "dr.iyer@hospitalc.org"
            },
            {
                "name": "Hospital D (Chennai)",
                "code": "HOSP-D",
                "location": "Chennai, India",
                "contact_email": "admin@hospitald.org",
                "contact_phone": "+91-44-0000-0004",
                "trust_score": 0.31,
                "trust_status": TrustStatus.UNTRUSTED,
                "total_patients": 1900,
                "total_training_samples": 800,
                "current_accuracy": 89.7,
                "dp_epsilon": 1.0,
                "quantization_enabled": True,
                "username": "hospital_d",
                "password": "hospital123",
                "full_name": "Dr. Rajan",
                "doctor_email": "dr.rajan@hospitald.org"
            },
            {
                "name": "Hospital E (Hyderabad)",
                "code": "HOSP-E",
                "location": "Hyderabad, India",
                "contact_email": "admin@hospitale.org",
                "contact_phone": "+91-40-0000-0005",
                "trust_score": 0.97,
                "trust_status": TrustStatus.TRUSTED,
                "total_patients": 5100,
                "total_training_samples": 2200,
                "current_accuracy": 94.8,
                "dp_epsilon": 1.0,
                "quantization_enabled": True,
                "username": "hospital_e",
                "password": "hospital123",
                "full_name": "Dr. Reddy",
                "doctor_email": "dr.reddy@hospitale.org"
            },
            {
                "name": "Hospital F (Kolkata)",
                "code": "HOSP-F",
                "location": "Kolkata, India",
                "contact_email": "admin@hospitalf.org",
                "contact_phone": "+91-33-0000-0006",
                "trust_score": 0.00,
                "trust_status": TrustStatus.INACTIVE,
                "total_patients": 1200,
                "total_training_samples": 500,
                "current_accuracy": 0.0,
                "dp_epsilon": 1.0,
                "quantization_enabled": False,
                "username": "hospital_f",
                "password": "hospital123",
                "full_name": "Dr. Banerjee",
                "doctor_email": "dr.banerjee@hospitalf.org"
            }
        ]

        hospitals = []
        for hdata in hospitals_data:
            last_seen_val = datetime.utcnow() - timedelta(minutes=random.randint(0, 45))
            if hdata["trust_status"] == TrustStatus.INACTIVE:
                last_seen_val = datetime.utcnow() - timedelta(days=5)
            hospital = Hospital(
                name=hdata["name"],
                code=hdata["code"],
                location=hdata["location"],
                contact_email=hdata["contact_email"],
                contact_phone=hdata["contact_phone"],
                trust_score=hdata["trust_score"],
                trust_status=hdata["trust_status"],
                is_active=hdata["trust_status"] != TrustStatus.INACTIVE,
                total_patients=hdata["total_patients"],
                total_training_samples=hdata["total_training_samples"],
                current_accuracy=hdata["current_accuracy"],
                dp_epsilon=hdata["dp_epsilon"],
                quantization_enabled=hdata["quantization_enabled"],
                created_at=datetime.utcnow() - timedelta(days=random.randint(30, 90)),
                last_seen=last_seen_val
            )
            db.add(hospital)
            db.flush()
            hospitals.append(hospital)

            hosp_user = User(
                username=hdata["username"],
                email=hdata["doctor_email"],
                hashed_password=get_password_hash(hdata["password"]),
                full_name=hdata["full_name"],
                role=UserRole.HOSPITAL,
                is_active=hdata["trust_status"] != TrustStatus.INACTIVE,
                hospital_id=hospital.id,
                created_at=datetime.utcnow() - timedelta(days=random.randint(20, 60))
            )
            db.add(hosp_user)
            print(f"[OK] Created {hospital.name} user: {hdata['username']} / {hdata['password']}")

        db.flush()

        total_rounds = 18
        for rn in range(1, total_rounds + 1):
            round_started = datetime.utcnow() - timedelta(minutes=(total_rounds - rn) * 180 + 30)
            round_completed = None
            if rn < total_rounds:
                round_completed = round_started + timedelta(minutes=random.randint(60, 120))
            gr = GlobalTrainingRound(
                round_number=rn,
                started_at=round_started,
                completed_at=round_completed,
                total_clients=6,
                participating_clients=random.randint(4, 5) if rn < total_rounds else 5,
                global_accuracy=round(72.0 + rn * 1.25 + random.uniform(-0.5, 0.5), 2),
                global_loss=round(max(0.10, 0.85 - rn * 0.038), 4),
                best_accuracy=0,
                avg_update_size=random.randint(2000, 3000),
                status=TrainingStatus.COMPLETED if rn < total_rounds else TrainingStatus.IN_PROGRESS
            )
            db.add(gr)
            db.flush()

            active_hospitals = [h for h in hospitals if h.trust_status != TrustStatus.INACTIVE]
            for hospital in active_hospitals:
                seed_val = hospital.id * 1000 + rn
                rng = random.Random(seed_val)
                train_acc = round(70 + rn * rng.uniform(1.8, 2.3) + rng.uniform(-1, 1), 2)
                val_acc = round(train_acc - rng.uniform(0.2, 1.8), 2)
                val_acc = min(99.0, max(50.0, val_acc))
                train_acc = min(99.0, max(50.0, train_acc))
                tr = TrainingRound(
                    round_number=rn,
                    hospital_id=hospital.id,
                    started_at=round_started + timedelta(minutes=rng.randint(1, 15)),
                    completed_at=round_started + timedelta(minutes=rng.randint(40, 90)),
                    local_epochs=5,
                    training_loss=round(rng.uniform(0.10, 0.6), 4),
                    training_accuracy=train_acc,
                    validation_loss=round(rng.uniform(0.14, 0.7), 4),
                    validation_accuracy=val_acc,
                    num_samples=hospital.total_training_samples,
                    status=TrainingStatus.COMPLETED
                )
                db.add(tr)
                db.flush()

                is_quantized = hospital.quantization_enabled
                raw_size = rng.randint(2, 4) * 1024 * 1024
                update_size = raw_size // 4 if is_quantized else raw_size
                mu = ModelUpdate(
                    hospital_id=hospital.id,
                    global_round_id=gr.id,
                    round_number=rn,
                    received_at=round_started + timedelta(minutes=rng.randint(45, 100)),
                    update_size_bytes=update_size,
                    is_quantized=is_quantized,
                    has_dp_noise=True,
                    dp_epsilon_used=hospital.dp_epsilon,
                    update_quality_score=round(min(1.0, max(0.1, hospital.trust_score + rng.uniform(-0.15, 0.1))), 3),
                    contribution_weight=hospital.trust_score,
                    status="Aggregated" if rn < total_rounds else "Received"
                )
                db.add(mu)

        db.flush()

        hosp_d_id = [h.id for h in hospitals if h.code == "HOSP-D"][0]
        ul_req = UnlearningRequest(
            request_id="UL-9B2F4D1A",
            hospital_id=hosp_d_id,
            submitted_at=datetime.utcnow() - timedelta(hours=6),
            reason="Patient withdrew consent. Remove 12 patient records from our historical training data.",
            patient_identifiers="MRN-00112, MRN-00113, MRN-00234, MRN-00567, MRN-00890",
            current_step=3,
            total_steps=5,
            progress_percent=60.0,
            status="Under Process"
        )
        db.add(ul_req)
        print(f"[OK] Created unlearning request: {ul_req.request_id}")

        base_time = datetime.utcnow()
        admin_evts = [
            (AuditEventType.MODEL_UPLOAD, "Model update (2.4 MB) received from Hospital E for round 18", "Success", None, hospitals[4]),
            (AuditEventType.TRAINING_COMPLETE, "Federated Training round 18 phase completed", "Success", admin, None),
            (AuditEventType.MODEL_UPLOAD, "Model update received from Hospital B for round 18", "Success", None, hospitals[1]),
            (AuditEventType.TRUST_SCORE_UPDATE, "Hospital C marked as Suspicious - update quality degraded", "Success", admin, hospitals[2]),
            (AuditEventType.TRUST_SCORE_UPDATE, "Hospital D flagged as Untrusted - malicious pattern detected", "Success", admin, hospitals[3]),
            (AuditEventType.UNLEARNING_REQUEST, "Unlearning request UL-9B2F4D1A submitted by Hospital D", "Success", None, hospitals[3]),
            (AuditEventType.MODEL_UPLOAD, "Model update received from Hospital A for round 18", "Success", None, hospitals[0]),
        ]
        for i, (et, desc, status, usr, hosp) in enumerate(admin_evts):
            t = base_time - timedelta(minutes=(i + 1) * random.randint(5, 20))
            al = AuditLog(
                timestamp=t,
                event_type=et,
                description=desc,
                status=status,
                ip_address=f"192.168.1.{random.randint(1, 254)}",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                user_id=usr.id if usr else None,
                hospital_id=hosp.id if hosp else None
            )
            db.add(al)

        for hospital in hospitals:
            hosp_user = db.query(User).filter(User.hospital_id == hospital.id).first()
            if not hosp_user:
                continue
            local_events = [
                (AuditEventType.LOGIN, f"User '{hosp_user.username}' logged in at {hospital.name}", "Success"),
                (AuditEventType.TRAINING_START, f"Local training started at {hospital.name} for round 18", "Success"),
                (AuditEventType.PRIVACY_CHECK, f"Privacy check completed for {hospital.name}", "Success"),
                (AuditEventType.TRAINING_COMPLETE, f"Local training completed at {hospital.name}", "Success"),
                (AuditEventType.MODEL_UPLOAD, f"Model update (2.4 MB) sent from {hospital.name}", "Success"),
            ]
            for i, (et, desc, status) in enumerate(local_events):
                al = AuditLog(
                    timestamp=base_time - timedelta(hours=i + 1, minutes=random.randint(0, 59)),
                    event_type=et,
                    description=desc,
                    status=status,
                    ip_address=f"10.0.{hospital.id}.{random.randint(1, 254)}",
                    user_id=hosp_user.id,
                    hospital_id=hospital.id
                )
                db.add(al)

        metrics_data = [
            ("global_round_current", "18"),
            ("global_round_total", "20"),
            ("fl_server_status", "running"),
            ("secure_aggregation", "enabled"),
            ("dp_enabled", "enabled"),
            ("encryption_tls", "active")
        ]
        for name, val in metrics_data:
            sm = SystemMetrics(metric_name=name, metric_value=val, updated_at=datetime.utcnow())
            db.add(sm)

        db.commit()

        print("\n[DONE] Database seeding completed successfully!")
        print(f"    Admin: {settings.ADMIN_USERNAME} / {settings.ADMIN_PASSWORD}")
        print(f"    Hospitals: {len(hospitals)} nodes created")
        print(f"    Global Rounds: {total_rounds} rounds with metrics")
        print(f"    Audit Logs: Populated with system events")
        print("\nLaunch the platform: python -m federated_healthcare.webapp.main\n")

    except Exception as e:
        db.rollback()
        print(f"[ERR] Error during seeding: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
