"""
Pydantic schemas for BioMobileBERT NER.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class NEREntity(BaseModel):
    text: str
    entity_type: str = Field(..., pattern="^(DISEASE|SYMPTOM|TREATMENT|MEASUREMENT|NUTRIENT|DEMOGRAPHIC)$")
    start_pos: int
    end_pos: int
    confidence: float = Field(..., ge=0, le=1)
    language: str = Field(..., pattern="^(en|ar)$")
    normalized_code: Optional[str] = None
    alt_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class NERDocumentResult(BaseModel):
    entities: List[NEREntity]
    language: str
    malnutrition_types: List[str]
    severity_level: Optional[str]
    entities_by_type: Dict[str, List[Dict[str, Any]]]
    summary: str
    total_entities: int
    processing_time_ms: float


class NERRequest(BaseModel):
    text: str
    language: str = Field(default="auto", pattern="^(auto|en|ar)$")
    entity_types: Optional[List[str]] = None
    include_confidence: bool = True


class NERBatchRequest(BaseModel):
    documents: List[NERRequest]
    batch_size: int = Field(default=8, ge=1, le=32)


class NERBatchResponse(BaseModel):
    results: List[NERDocumentResult]
    total_documents: int
    total_entities: int
    avg_processing_time_ms: float
