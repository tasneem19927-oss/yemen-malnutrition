"""
Pydantic schemas for Prediction and Clinical Reasoning.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class SeverityResult(BaseModel):
    probability: float = Field(..., ge=0, le=1)
    risk_percent: float = Field(..., ge=0, le=100)
    severity: str = Field(..., pattern="^(normal|mild|moderate|severe)$")
    confidence: float = Field(..., ge=0, le=1)


class PredictionInput(BaseModel):
    patient_id: int
    measurement_id: int

    # Raw features
    age_months: int = Field(..., ge=0, le=59)
    sex: str = Field(..., pattern="^(male|female)$")
    weight_kg: float = Field(..., gt=0)
    height_cm: float = Field(..., gt=0)
    oedema: bool = False
    breastfeeding: bool = False
    vitamin_a: bool = False
    diarrhea_recent: bool = False
    fever_recent: bool = False
    cough_recent: bool = False
    maternal_education: Optional[str] = None
    wealth_index: Optional[str] = None
    residence_type: Optional[str] = None

    # Optional: pre-calculated z-scores
    haz: Optional[float] = None
    whz: Optional[float] = None
    waz: Optional[float] = None


class PredictionFeatures(BaseModel):
    age_months: int
    sex: int  # 0=male, 1=female
    weight: float
    height: float
    oedema: int
    breastfeeding: int
    vitamin_a: int
    diarrhea_recent: int
    fever_recent: int
    cough_recent: int
    maternal_education: int
    wealth_index: int
    residence_type: int

    # Engineered features
    haz: float
    whz: float
    waz: float
    bmi: float
    weight_height_ratio: float
    age_weight_interaction: float
    age_height_interaction: float
    health_risk_score: float
    nutrition_risk_score: float
    muac_zscore: Optional[float] = None
    head_circumference_zscore: Optional[float] = None
    growth_velocity: Optional[float] = None
    wasting_stunting_interaction: Optional[float] = None
    age_group: int
    season: int
    food_security_index: Optional[float] = None


class PredictionResult(BaseModel):
    id: int
    patient_id: int

    # Predictions
    stunting: SeverityResult
    wasting: SeverityResult
    underweight: SeverityResult

    # Overall
    overall_risk: str
    overall_recommendation: str

    # Clinical reasoning
    clinical_query: str
    rag_evidence: List[Dict[str, Any]]
    recommended_intervention: str
    referral_needed: bool
    referral_urgency: Optional[str]

    # Doctor review
    doctor_notes: Optional[str]
    doctor_approved: bool

    # Metadata
    model_version: str
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    patient_id: int
    measurement_id: int
    include_explanation: bool = True
    include_rag: bool = True
    language: str = Field(default="en", pattern="^(en|ar)$")


class DoctorReviewRequest(BaseModel):
    prediction_id: int
    doctor_notes: str
    approved: bool = True
    override_recommendation: Optional[str] = None


class ClinicalNoteCreate(BaseModel):
    patient_id: int
    note_type: str = Field(default="general", pattern="^(general|assessment|treatment|follow_up)$")
    content: str
    content_ar: Optional[str] = None


class ClinicalNoteResponse(ClinicalNoteCreate):
    id: int
    created_by: int
    extracted_entities: Optional[List[Dict[str, Any]]]
    created_at: datetime

    class Config:
        from_attributes = True


class ReportRequest(BaseModel):
    prediction_id: int
    language: str = Field(default="en", pattern="^(en|ar)$")
    include_evidence: bool = True
    include_recommendations: bool = True
    include_doctor_notes: bool = True
