"""
Pydantic schemas for Knowledge Base.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class KnowledgeBaseCreate(BaseModel):
    title: str = Field(..., max_length=500)
    title_ar: Optional[str] = None
    authors: Optional[str] = None
    organization: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    abstract_ar: Optional[str] = None
    clinical_summary: Optional[str] = None
    clinical_summary_ar: Optional[str] = None
    keywords: Optional[List[str]] = None
    keywords_ar: Optional[List[str]] = None
    citation: Optional[str] = None
    doi: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None


class KnowledgeBaseUpdate(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    clinical_summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    status: Optional[str] = Field(None, pattern="^(pending|approved|rejected|archived)$")


class KnowledgeBaseResponse(BaseModel):
    id: int
    title: str
    title_ar: Optional[str]
    authors: Optional[str]
    organization: Optional[str]
    year: Optional[int]
    abstract: Optional[str]
    clinical_summary: Optional[str]
    keywords: Optional[List[str]]
    citation: Optional[str]
    doi: Optional[str]
    source_url: Optional[str]
    source_type: Optional[str]
    status: str
    approved_by: Optional[int]
    approved_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class KnowledgeSearchRequest(BaseModel):
    query: str
    language: str = Field(default="en", pattern="^(en|ar)$")
    top_k: int = Field(default=5, ge=1, le=20)
    source_type: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None


class KnowledgeSearchResult(BaseModel):
    id: int
    title: str
    clinical_summary: Optional[str]
    citation: Optional[str]
    relevance_score: float
    source_type: Optional[str]


class KnowledgeUpdateQueue(BaseModel):
    id: int
    title: str
    source_url: Optional[str]
    detected_at: datetime
    status: str  # pending_review, doctor_approved, admin_approved, rejected
    doctor_reviewer_id: Optional[int]
    admin_reviewer_id: Optional[int]

    class Config:
        from_attributes = True


class KnowledgeApprovalRequest(BaseModel):
    queue_id: int
    approved: bool
    notes: Optional[str] = None
