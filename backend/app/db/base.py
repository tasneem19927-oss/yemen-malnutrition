"""
SQLAlchemy Base model.
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, Integer
from datetime import datetime

Base = declarative_base()


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class SoftDeleteMixin:
    """Mixin to add soft delete functionality."""
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Integer, default=0, nullable=False)
