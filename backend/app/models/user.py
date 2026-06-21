"""
User and Role models.
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from enum import Enum as PyEnum

from app.db.base import Base, TimestampMixin


class RoleEnum(str, PyEnum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    NURSE = "nurse"


class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    description = Column(String(255))
    permissions = Column(Text)  # JSON string of permissions

    users = relationship("User", back_populates="role")


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    phone = Column(String(50))
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    healthcare_center_id = Column(Integer, ForeignKey("healthcare_centers.id"), nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login = Column(DateTime)

    # Offline sync
    offline_access = Column(Boolean, default=False)
    last_sync = Column(DateTime)

    # Relationships
    role = relationship("Role", back_populates="users")
    healthcare_center = relationship("HealthcareCenter", back_populates="users")
    patients = relationship("Patient", back_populates="created_by_user")
    predictions = relationship("Prediction", back_populates="created_by_user")
    audit_logs = relationship("AuditLog", back_populates="user")

    def has_permission(self, permission: str) -> bool:
        if self.role.name == "admin":
            return True
        if not self.role.permissions:
            return False
        import json
        perms = json.loads(self.role.permissions)
        return permission in perms.get("permissions", [])


class HealthcareCenter(Base):
    __tablename__ = "healthcare_centers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    name_ar = Column(String(255))  # Arabic name
    code = Column(String(50), unique=True, nullable=False)
    type = Column(String(50))  # hospital, clinic, mobile_unit
    governorate = Column(String(100))
    district = Column(String(100))
    latitude = Column(String(20))
    longitude = Column(String(20))
    phone = Column(String(50))
    email = Column(String(255))
    is_active = Column(Boolean, default=True)

    users = relationship("User", back_populates="healthcare_center")
    patients = relationship("Patient", back_populates="healthcare_center")
