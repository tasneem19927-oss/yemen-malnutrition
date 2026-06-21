"""
Prediction and Recommendation models.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from app.db.base import Base, TimestampMixin


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    measurement_id = Column(Integer, ForeignKey("measurements.id"))
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Input features (stored for reproducibility)
    input_features = Column(JSON)

    # Stunting Prediction
    stunting_probability = Column(Float)
    stunting_risk_percent = Column(Float)
    stunting_severity = Column(String(20))  # normal, mild, moderate, severe
    stunting_confidence = Column(Float)

    # Wasting Prediction
    wasting_probability = Column(Float)
    wasting_risk_percent = Column(Float)
    wasting_severity = Column(String(20))
    wasting_confidence = Column(Float)

    # Underweight Prediction
    underweight_probability = Column(Float)
    underweight_risk_percent = Column(Float)
    underweight_severity = Column(String(20))
    underweight_confidence = Column(Float)

    # Overall assessment
    overall_risk = Column(String(20))
    overall_recommendation = Column(Text)

    # Clinical reasoning
    clinical_query = Column(Text)
    rag_evidence = Column(JSON)
    recommended_intervention = Column(Text)
    referral_needed = Column(Boolean, default=False)
    referral_urgency = Column(String(20))  # routine, urgent, emergency

    # Doctor review
    doctor_notes = Column(Text)
    doctor_approved = Column(Boolean, default=False)
    approved_at = Column(DateTime)

    # Model version
    model_version = Column(String(50))

    # Offline sync
    synced = Column(Boolean, default=True)
    offline_id = Column(String(100))

    # Relationships
    patient = relationship("Patient", back_populates="predictions")
    created_by_user = relationship("User", back_populates="predictions")


class Recommendation(Base, TimestampMixin):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))

    # Recommendation details
    category = Column(String(50))  # nutrition, medical, referral, follow_up
    priority = Column(String(20))  # low, medium, high, critical
    title = Column(String(255))
    title_ar = Column(String(255))
    description = Column(Text)
    description_ar = Column(Text)

    # Evidence
    evidence_source = Column(String(255))
    who_reference = Column(String(255))
    confidence_score = Column(Float)

    # Action tracking
    status = Column(String(20), default="pending")  # pending, in_progress, completed, cancelled
    assigned_to = Column(Integer, ForeignKey("users.id"))
    completed_at = Column(DateTime)
    completion_notes = Column(Text)


class ClinicalNote(Base, TimestampMixin):
    __tablename__ = "clinical_notes"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    note_type = Column(String(50))  # general, assessment, treatment, follow_up
    content = Column(Text, nullable=False)
    content_ar = Column(Text)

    # NER extracted entities
    extracted_entities = Column(JSON)
