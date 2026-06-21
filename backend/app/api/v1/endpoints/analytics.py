"""
Analytics and monitoring endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from typing import Optional
from datetime import datetime, timedelta

from app.db.session import get_db
from app.core.security import require_admin, require_doctor
from app.models.user import User
from app.models.patient import Patient, Measurement
from app.models.prediction import Prediction

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/dashboard")
async def get_dashboard_stats(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """Get dashboard analytics."""
    since = datetime.utcnow() - timedelta(days=days)

    # Patient stats
    total_patients = db.query(func.count(Patient.id)).filter(Patient.is_deleted == 0).scalar()
    new_patients = db.query(func.count(Patient.id)).filter(Patient.created_at >= since).scalar()

    # Prediction stats
    total_predictions = db.query(func.count(Prediction.id)).filter(Prediction.created_at >= since).scalar()

    # Severity distribution
    severity_counts = db.query(
        Prediction.overall_risk,
        func.count(Prediction.id)
    ).filter(Prediction.created_at >= since).group_by(Prediction.overall_risk).all()

    # By governorate
    governorate_stats = db.query(
        Patient.governorate,
        func.count(distinct(Patient.id)),
        func.count(Prediction.id)
    ).outerjoin(Prediction, Patient.id == Prediction.patient_id).filter(
        Patient.is_deleted == 0
    ).group_by(Patient.governorate).all()

    return {
        "period_days": days,
        "patients": {
            "total": total_patients,
            "new": new_patients,
        },
        "predictions": {
            "total": total_predictions,
        },
        "severity_distribution": {s: c for s, c in severity_counts},
        "governorate_stats": [
            {"governorate": g, "patients": p, "predictions": pr}
            for g, p, pr in governorate_stats
        ],
    }


@router.get("/model-performance")
async def get_model_performance(
    model_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Get AI model performance metrics."""
    from app.models.knowledge import ModelVersion

    query = db.query(ModelVersion)
    if model_type:
        query = query.filter(ModelVersion.model_type == model_type)

    models = query.order_by(ModelVersion.created_at.desc()).all()

    return {
        "models": [
            {
                "name": m.model_name,
                "version": m.version,
                "type": m.model_type,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "auc_roc": m.auc_roc,
                "status": m.status,
                "deployed_at": m.deployed_at,
            }
            for m in models
        ]
    }


@router.get("/system-health")
async def get_system_health(
    current_user: User = Depends(require_admin),
):
    """Get system health status."""
    import psutil

    return {
        "cpu": {
            "percent": psutil.cpu_percent(),
            "count": psutil.cpu_count(),
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
        },
    }
