"""
Healthcare Center management endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.core.security import require_admin
from app.models.user import User, HealthcareCenter
from app.schemas.user import HealthcareCenterCreate, HealthcareCenterResponse

router = APIRouter(prefix="/users/healthcare-centers", tags=["Healthcare Centers"])


@router.get("", response_model=List[HealthcareCenterResponse])
async def list_centers(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """List all healthcare centers."""
    centers = db.query(HealthcareCenter).offset(skip).limit(limit).all()
    return [HealthcareCenterResponse.model_validate(c) for c in centers]


@router.post("", response_model=HealthcareCenterResponse)
async def create_center(
    center_data: HealthcareCenterCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Create healthcare center."""
    center = HealthcareCenter(**center_data.model_dump())
    db.add(center)
    db.commit()
    db.refresh(center)
    return HealthcareCenterResponse.model_validate(center)
