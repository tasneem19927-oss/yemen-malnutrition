"""
Patient and Measurement models.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Date, Enum
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum

from app.db.base import Base, TimestampMixin, SoftDeleteMixin


class SexEnum(str, PyEnum):
    MALE = "male"
    FEMALE = "female"


class ResidenceEnum(str, PyEnum):
    URBAN = "urban"
    RURAL = "rural"
    CAMP = "camp"


class Patient(Base, TimestampMixin, SoftDeleteMixin):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)

    # Identification
    registration_number = Column(String(100), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    first_name_ar = Column(String(100))
    last_name_ar = Column(String(100))

    # Demographics
    date_of_birth = Column(Date, nullable=False)
    age_months = Column(Integer, nullable=False)
    sex = Column(String(10), nullable=False)

    # Caregiver info
    caregiver_name = Column(String(255))
    caregiver_phone = Column(String(50))
    caregiver_relation = Column(String(50))

    # Location
    governorate = Column(String(100))
    district = Column(String(100))
    village = Column(String(100))
    residence_type = Column(String(20))

    # Socioeconomic
    wealth_index = Column(String(20))
    maternal_education = Column(String(50))

    # Metadata
    healthcare_center_id = Column(Integer, ForeignKey("healthcare_centers.id"))
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Relationships
    healthcare_center = relationship("HealthcareCenter", back_populates="patients")
    created_by_user = relationship("User", back_populates="patients")
    measurements = relationship("Measurement", back_populates="patient", order_by="Measurement.measurement_date.desc()")
    predictions = relationship("Prediction", back_populates="patient", order_by="Prediction.created_at.desc()")


class Measurement(Base, TimestampMixin):
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    # Anthropometric
    weight_kg = Column(Float, nullable=False)
    height_cm = Column(Float, nullable=False)
    muac_mm = Column(Float)  # Mid-upper arm circumference
    head_circumference_cm = Column(Float)

    # Clinical signs
    oedema = Column(Boolean, default=False)
    oedema_severity = Column(String(20))  # mild, moderate, severe

    # Health status
    diarrhea_recent = Column(Boolean, default=False)
    fever_recent = Column(Boolean, default=False)
    cough_recent = Column(Boolean, default=False)

    # Feeding
    breastfeeding = Column(Boolean, default=False)
    exclusive_breastfeeding = Column(Boolean, default=False)
    vitamin_a = Column(Boolean, default=False)

    # Calculated z-scores
    haz = Column(Float)  # Height-for-age Z-score
    whz = Column(Float)  # Weight-for-height Z-score
    waz = Column(Float)  # Weight-for-age Z-score
    bmiz = Column(Float)  # BMI-for-age Z-score

    # Metadata
    measured_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    measurement_date = Column(DateTime, nullable=False)
    notes = Column(Text)

    # Relationships
    patient = relationship("Patient", back_populates="measurements")
