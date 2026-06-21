"""
Knowledge Base and RAG endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.session import get_db
from app.core.security import require_admin, require_doctor, get_current_user
from app.models.user import User
from app.models.knowledge import KnowledgeBase
from app.schemas.knowledge import (
    KnowledgeBaseCreate, KnowledgeBaseUpdate, KnowledgeBaseResponse,
    KnowledgeSearchRequest, KnowledgeSearchResult, KnowledgeApprovalRequest
)
from app.services.rag.clinical_rag import ClinicalRAG

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])


@router.get("", response_model=List[KnowledgeBaseResponse])
async def list_knowledge(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any),
):
    """List knowledge base entries."""
    query = db.query(KnowledgeBase)

    if status:
        query = query.filter(KnowledgeBase.status == status)
    if source_type:
        query = query.filter(KnowledgeBase.source_type == source_type)

    entries = query.order_by(KnowledgeBase.created_at.desc()).offset(skip).limit(limit).all()
    return [KnowledgeBaseResponse.model_validate(e) for e in entries]


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
async def create_knowledge(
    entry: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Add new knowledge base entry (Admin only, requires approval workflow)."""
    kb_entry = KnowledgeBase(
        **entry.model_dump(),
        status="pending",
    )
    db.add(kb_entry)
    db.commit()
    db.refresh(kb_entry)
    return KnowledgeBaseResponse.model_validate(kb_entry)


@router.get("/{entry_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge(
    entry_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_any),
):
    """Get knowledge base entry by ID."""
    entry = db.query(KnowledgeBase).filter(KnowledgeBase.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")
    return KnowledgeBaseResponse.model_validate(entry)


@router.put("/{entry_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge(
    entry_id: int,
    entry_data: KnowledgeBaseUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Update knowledge base entry."""
    entry = db.query(KnowledgeBase).filter(KnowledgeBase.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")

    for field, value in entry_data.model_dump(exclude_unset=True).items():
        setattr(entry, field, value)

    db.commit()
    db.refresh(entry)
    return KnowledgeBaseResponse.model_validate(entry)


@router.post("/{entry_id}/approve")
async def approve_knowledge(
    entry_id: int,
    approval: KnowledgeApprovalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor),
):
    """Approve or reject knowledge base entry (Doctor+)."""
    entry = db.query(KnowledgeBase).filter(KnowledgeBase.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Knowledge entry not found")

    if approval.approved:
        entry.status = "approved"
        entry.approved_by = current_user.id
        entry.approved_at = datetime.utcnow()

        # Update FAISS index
        rag = ClinicalRAG()
        await rag.add_document(entry)
    else:
        entry.status = "rejected"

    db.commit()
    return {"message": f"Entry {'approved' if approval.approved else 'rejected'}"}


# RAG Search
@router.post("/search", response_model=List[KnowledgeSearchResult])
async def search_knowledge(
    request: KnowledgeSearchRequest,
    current_user: User = Depends(require_any),
):
    """Search knowledge base using RAG."""
    rag = ClinicalRAG()
    results = await rag.search(
        query=request.query,
        language=request.language,
        top_k=request.top_k,
        source_type=request.source_type,
        year_min=request.year_min,
        year_max=request.year_max,
    )
    return results


@router.post("/query")
async def query_knowledge(
    request: KnowledgeSearchRequest,
    current_user: User = Depends(require_any),
):
    """Query knowledge base with natural language and get clinical answer."""
    rag = ClinicalRAG()
    answer = await rag.query(
        query=request.query,
        language=request.language,
    )
    return {
        "query": request.query,
        "answer": answer["text"],
        "evidence": answer["sources"],
        "confidence": answer["confidence"],
    }
