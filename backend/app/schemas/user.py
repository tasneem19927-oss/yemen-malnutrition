"""
Pydantic schemas for User and Authentication.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class RoleBase(BaseModel):
    name: str
    description: Optional[str] = None


class RoleResponse(RoleBase):
    id: int

    class Config:
        from_attributes = True


class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    phone: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    role_id: int
    healthcare_center_id: Optional[int] = None


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role_id: Optional[int] = None
    is_active: Optional[bool] = None
    offline_access: Optional[bool] = None


class UserResponse(UserBase):
    id: int
    role: RoleResponse
    healthcare_center_id: Optional[int]
    is_active: bool
    is_verified: bool
    last_login: Optional[datetime]
    offline_access: bool
    last_sync: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenRefresh(BaseModel):
    refresh_token: str


class HealthcareCenterBase(BaseModel):
    name: str
    name_ar: Optional[str] = None
    code: str
    type: Optional[str] = None
    governorate: Optional[str] = None
    district: Optional[str] = None
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class HealthcareCenterCreate(HealthcareCenterBase):
    pass


class HealthcareCenterResponse(HealthcareCenterBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)
