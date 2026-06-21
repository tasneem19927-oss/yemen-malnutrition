"""
Patient management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.db.session import get_db
from app.core.security import require_nurse, require_doctor, get_current_user
from app.models.user import User
from app.models.patient import Patient, Measurement
from app.schemas.patient import (
    PatientCreate, PatientUpdate, PatientResponse, PatientDetailResponse,
    MeasurementCreate, MeasurementResponse, PatientSearch
)
from app.services.ml.zscore_calculator import calculate_all_zscores

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("", response_model=List[PatientResponse])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = None,
    governorate: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """List patients with filtering."""
    query = db.query(Patient).filter(Patient.is_deleted == 0)

    # Nurses can only see patients from their center
    if current_user.role.name == "nurse" and current_user.healthcare_center_id:
        query = query.filter(Patient.healthcare_center_id == current_user.healthcare_center_id)

    if search:
        query = query.filter(
            Patient.first_name.ilike(f"%{search}%") |
            Patient.last_name.ilike(f"%{search}%") |
            Patient.registration_number.ilike(f"%{search}%")
        )
    if governorate:
        query = query.filter(Patient.governorate == governorate)

    patients = query.order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()
    return [PatientResponse.model_validate(p) for p in patients]


@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient_data: PatientCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """Register a new patient."""
    # Generate registration number
    import uuid
    reg_num = f"YMN-{datetime.utcnow().strftime('%Y%m')}-{str(uuid.uuid4())[:6].upper()}"

    # Calculate age in months
    from dateutil.relativedelta import relativedelta
    age_months = relativedelta(datetime.utcnow(), patient_data.date_of_birth).months
    age_months += relativedelta(datetime.utcnow(), patient_data.date_of_birth).years * 12

    patient = Patient(
        registration_number=reg_num,
        first_name=patient_data.first_name,
        last_name=patient_data.last_name,
        first_name_ar=patient_data.first_name_ar,
        last_name_ar=patient_data.last_name_ar,
        date_of_birth=patient_data.date_of_birth,
        age_months=age_months,
        sex=patient_data.sex,
        caregiver_name=patient_data.caregiver_name,
        caregiver_phone=patient_data.caregiver_phone,
        caregiver_relation=patient_data.caregiver_relation,
        governorate=patient_data.governorate,
        district=patient_data.district,
        village=patient_data.village,
        residence_type=patient_data.residence_type,
        wealth_index=patient_data.wealth_index,
        maternal_education=patient_data.maternal_education,
        healthcare_center_id=patient_data.healthcare_center_id or current_user.healthcare_center_id,
        created_by=current_user.id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    return PatientResponse.model_validate(patient)


@router.get("/{patient_id}", response_model=PatientDetailResponse)
async def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """Get patient details with measurements and predictions."""
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.is_deleted == 0).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Check access
    if current_user.role.name == "nurse" and patient.healthcare_center_id != current_user.healthcare_center_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return PatientDetailResponse.model_validate(patient)


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: int,
    patient_data: PatientUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """Update patient information."""
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.is_deleted == 0).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    for field, value in patient_data.model_dump(exclude_unset=True).items():
        setattr(patient, field, value)

    db.commit()
    db.refresh(patient)
    return PatientResponse.model_validate(patient)


@router.delete("/{patient_id}")
async def delete_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """Soft delete patient."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    patient.is_deleted = 1
    patient.deleted_at = datetime.utcnow()
    db.commit()
    return {"message": "Patient deleted successfully"}


# Measurements
@router.post("/{patient_id}/measurements", response_model=MeasurementResponse)
async def add_measurement(
    patient_id: int,
    measurement_data: MeasurementCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """Add measurement for a patient."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Calculate z-scores
    zscores = calculate_all_zscores(
        age_months=patient.age_months,
        sex=patient.sex,
        weight_kg=measurement_data.weight_kg,
        height_cm=measurement_data.height_cm,
    )

    measurement = Measurement(
        patient_id=patient_id,
        weight_kg=measurement_data.weight_kg,
        height_cm=measurement_data.height_cm,
        muac_mm=measurement_data.muac_mm,
        head_circumference_cm=measurement_data.head_circumference_cm,
        oedema=measurement_data.oedema,
        oedema_severity=measurement_data.oedema_severity,
        diarrhea_recent=measurement_data.diarrhea_recent,
        fever_recent=measurement_data.fever_recent,
        cough_recent=measurement_data.cough_recent,
        breastfeeding=measurement_data.breastfeeding,
        exclusive_breastfeeding=measurement_data.exclusive_breastfeeding,
        vitamin_a=measurement_data.vitamin_a,
        haz=zscores.get("haz"),
        whz=zscores.get("whz"),
        waz=zscores.get("waz"),
        bmiz=zscores.get("bmiz"),
        measured_by=current_user.id,
        measurement_date=measurement_data.measurement_date,
        notes=measurement_data.notes,
    )
    db.add(measurement)
    db.commit()
    db.refresh(measurement)

    return MeasurementResponse.model_validate(measurement)


@router.get("/{patient_id}/measurements", response_model=List[MeasurementResponse])
async def get_measurements(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_nurse),
):
    """Get all measurements for a patient."""
    measurements = db.query(Measurement).filter(Measurement.patient_id == patient_id).order_by(Measurement.measurement_date.desc()).all()
    return [MeasurementResponse.model_validate(m) for m in measurements]
