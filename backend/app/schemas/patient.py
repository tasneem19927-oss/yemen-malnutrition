"""
Pydantic schemas for Patient and Measurement.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


class PatientBase(BaseModel):
    first_name: str
    last_name: str
    first_name_ar: Optional[str] = None
    last_name_ar: Optional[str] = None
    date_of_birth: date
    sex: str = Field(..., pattern="^(male|female)$")

    # Caregiver
    caregiver_name: Optional[str] = None
    caregiver_phone: Optional[str] = None
    caregiver_relation: Optional[str] = None

    # Location
    governorate: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    residence_type: Optional[str] = Field(None, pattern="^(urban|rural|camp)$")

    # Socioeconomic
    wealth_index: Optional[str] = None
    maternal_education: Optional[str] = None


class PatientCreate(PatientBase):
    healthcare_center_id: Optional[int] = None


class PatientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    caregiver_name: Optional[str] = None
    caregiver_phone: Optional[str] = None
    governorate: Optional[str] = None
    district: Optional[str] = None
    village: Optional[str] = None
    wealth_index: Optional[str] = None
    maternal_education: Optional[str] = None


class PatientResponse(PatientBase):
    id: int
    registration_number: str
    age_months: int
    healthcare_center_id: Optional[int]
    created_by: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PatientDetailResponse(PatientResponse):
    measurements: List["MeasurementResponse"] = []
    predictions: List["PredictionSummaryResponse"] = []


class MeasurementBase(BaseModel):
    weight_kg: float = Field(..., gt=0, le=50)
    height_cm: float = Field(..., gt=30, le=150)
    muac_mm: Optional[float] = Field(None, gt=0, le=300)
    head_circumference_cm: Optional[float] = None

    # Clinical
    oedema: bool = False
    oedema_severity: Optional[str] = None
    diarrhea_recent: bool = False
    fever_recent: bool = False
    cough_recent: bool = False

    # Feeding
    breastfeeding: bool = False
    exclusive_breastfeeding: bool = False
    vitamin_a: bool = False

    measurement_date: datetime
    notes: Optional[str] = None


class MeasurementCreate(MeasurementBase):
    patient_id: int


class MeasurementResponse(MeasurementBase):
    id: int
    patient_id: int
    measured_by: int

    # Calculated
    haz: Optional[float]
    whz: Optional[float]
    waz: Optional[float]
    bmiz: Optional[float]

    created_at: datetime

    class Config:
        from_attributes = True


class PatientSearch(BaseModel):
    query: Optional[str] = None
    governorate: Optional[str] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    sex: Optional[str] = None
    risk_level: Optional[str] = None
