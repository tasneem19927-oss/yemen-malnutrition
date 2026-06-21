"""
BioMobileBERT NER endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.core.security import require_any, get_current_user
from app.models.user import User
from app.schemas.ner import NERRequest, NERDocumentResult, NERBatchRequest, NERBatchResponse
from app.services.nlp.biomobilebert_ner import BioMobileBERTNER

router = APIRouter(prefix="/ner", tags=["NER"])


@router.post("/extract", response_model=NERDocumentResult)
async def extract_entities(
    request: NERRequest,
    current_user: User = Depends(require_any),
):
    """Extract medical entities from clinical text using BioMobileBERT."""
    ner_service = BioMobileBERTNER.get_instance()

    result = ner_service.extract_medical_entities(
        text=request.text,
        language=request.language,
    )

    return NERDocumentResult.model_validate(result.to_dict())


@router.post("/extract/batch", response_model=NERBatchResponse)
async def extract_entities_batch(
    request: NERBatchRequest,
    current_user: User = Depends(require_any),
):
    """Batch entity extraction for multiple documents."""
    ner_service = BioMobileBERTNER.get_instance()

    results = []
    total_entities = 0
    total_time = 0

    for doc in request.documents:
        result = ner_service.extract_medical_entities(
            text=doc.text,
            language=doc.language,
        )
        results.append(NERDocumentResult.model_validate(result.to_dict()))
        total_entities += result.total_entities
        total_time += result.processing_time_ms

    return NERBatchResponse(
        results=results,
        total_documents=len(results),
        total_entities=total_entities,
        avg_processing_time_ms=total_time / len(results) if results else 0,
    )


@router.get("/health")
async def ner_health():
    """Check NER service health."""
    ner_service = BioMobileBERTNER.get_instance()
    return {
        "status": "healthy" if ner_service._model_loaded else "not_loaded",
        "model": ner_service.model_name,
        "cached_entities": len(ner_service._entity_cache),
        "cache_hits": ner_service._cache_hits,
        "cache_misses": ner_service._cache_misses,
    }
