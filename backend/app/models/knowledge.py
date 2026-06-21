"""
Knowledge Base and Audit models.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum

from app.db.base import Base, TimestampMixin


class KnowledgeSource(str, PyEnum):
    WHO = "who"
    UNICEF = "unicef"
    LANCET = "lancet"
    JME = "jme"
    OTHER = "other"


class KnowledgeBase(Base, TimestampMixin):
    __tablename__ = "knowledge_base"

    id = Column(Integer, primary_key=True, index=True)

    # Bibliographic info
    title = Column(String(500), nullable=False)
    title_ar = Column(String(500))
    authors = Column(Text)
    organization = Column(String(255))
    year = Column(Integer)

    # Content
    abstract = Column(Text)
    abstract_ar = Column(Text)
    clinical_summary = Column(Text)
    clinical_summary_ar = Column(Text)
    keywords = Column(JSON)
    keywords_ar = Column(JSON)

    # References
    citation = Column(Text)
    doi = Column(String(255))
    source_url = Column(String(500))
    source_type = Column(String(50))  # who_standard, guideline, research, report

    # Status
    status = Column(String(20), default="pending")  # pending, approved, rejected, archived
    approved_by = Column(Integer, ForeignKey("users.id"))
    approved_at = Column(DateTime)

    # Embedding
    embedding_id = Column(String(100))

    # Relationships
    approver = relationship("User", foreign_keys=[approved_by])


class AuditLog(Base, TimestampMixin):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    action = Column(String(50), nullable=False)  # create, read, update, delete, login, predict
    entity_type = Column(String(50))  # patient, prediction, user, knowledge
    entity_id = Column(Integer)

    details = Column(JSON)
    ip_address = Column(String(50))
    user_agent = Column(String(500))

    # Relationships
    user = relationship("User", back_populates="audit_logs")


class ModelVersion(Base, TimestampMixin):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)

    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50))  # xgboost_stunting, xgboost_wasting, xgboost_underweight, biomobilebert

    # Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)

    # Deployment
    file_path = Column(String(500))
    file_size = Column(Integer)
    checksum = Column(String(64))

    # Status
    status = Column(String(20), default="staging")  # staging, production, archived
    deployed_at = Column(DateTime)
    deployed_by = Column(Integer, ForeignKey("users.id"))

    # Notes
    training_notes = Column(Text)
    validation_results = Column(JSON)


class SyncLog(Base, TimestampMixin):
    __tablename__ = "sync_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    device_id = Column(String(100))

    sync_type = Column(String(50))  # full, incremental, push, pull
    status = Column(String(20))  # started, completed, failed

    records_synced = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)

    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
