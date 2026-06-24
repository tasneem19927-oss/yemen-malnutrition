"""
Vector Store service using FAISS for semantic search.
Wraps ClinicalRAG's FAISS index with a stable public API.

Satisfies the import in main.py:
    from app.services.rag.vector_store import VectorStore
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed vector store for clinical knowledge retrieval.
    Delegates to ClinicalRAG internally. Gracefully degrades if
    ML dependencies are unavailable so FastAPI can still start.
    """

    def __init__(self):
        self._rag = None
        self._initialized = False
        self._init_store()

    def _init_store(self):
        """Initialize the underlying ClinicalRAG store."""
        try:
            from app.services.rag.clinical_rag import ClinicalRAG
            self._rag = ClinicalRAG()
            self._initialized = True
            logger.info("VectorStore initialized successfully via ClinicalRAG")
        except Exception as e:
            logger.warning(
                f"VectorStore could not initialize ClinicalRAG: {e}. "
                "Running in degraded mode — knowledge search will be unavailable."
            )
            self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._rag is not None

    async def search(
        self,
        query: str,
        top_k: int = 5,
        language: str = "en",
        source_type: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base for clinical evidence."""
        if not self.is_ready:
            logger.warning("VectorStore.search called but store is not initialized.")
            return []
        try:
            results = await self._rag.search(
                query=query,
                language=language,
                top_k=top_k,
                source_type=source_type,
                year_min=year_min,
                year_max=year_max,
            )
            return results if results else []
        except Exception as e:
            logger.error(f"VectorStore search failed: {e}", exc_info=True)
            return []

    async def query(
        self,
        query: str,
        language: str = "en",
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Natural language query with clinical answer generation."""
        if not self.is_ready:
            return {"text": "Knowledge base unavailable.", "sources": [], "confidence": 0.0}
        try:
            return await self._rag.query(query=query, language=language, context=context)
        except Exception as e:
            logger.error(f"VectorStore.query failed: {e}", exc_info=True)
            return {"text": "Error during knowledge retrieval.", "sources": [], "confidence": 0.0}

    async def get_recommendations(
        self,
        prediction_result: Dict,
        patient: Any,
        measurement: Any,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Get clinical recommendations based on prediction results."""
        if not self.is_ready:
            return {
                "evidence": [],
                "recommendation": "Knowledge base unavailable. Please consult a specialist.",
                "referral_needed": False,
                "referral_urgency": None,
                "who_reference": "WHO Child Growth Standards (2006)",
            }
        try:
            return await self._rag.get_recommendations(
                prediction_result=prediction_result,
                patient=patient,
                measurement=measurement,
                language=language,
            )
        except Exception as e:
            logger.error(f"VectorStore.get_recommendations failed: {e}", exc_info=True)
            return {
                "evidence": [],
                "recommendation": "Error retrieving recommendations.",
                "referral_needed": False,
                "referral_urgency": None,
                "who_reference": "",
            }

    async def add_document(self, document: Any) -> bool:
        """Add a document to the FAISS index."""
        if not self.is_ready:
            logger.warning("VectorStore.add_document: store not initialized.")
            return False
        try:
            return await self._rag.add_document(document)
        except Exception as e:
            logger.error(f"VectorStore.add_document failed: {e}", exc_info=True)
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return vector store statistics."""
        if not self.is_ready:
            return {"status": "unavailable", "document_count": 0}
        try:
            return {
                "status": "ready",
                "initialized": self._initialized,
                "document_count": (
                    self._rag.faiss_index.ntotal
                    if self._rag.faiss_index is not None else 0
                ),
                "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
            }
        except Exception as e:
            logger.error(f"VectorStore.get_stats failed: {e}")
            return {"status": "error", "error": str(e)}

    def save(self, path: str) -> bool:
        """Persist FAISS index to disk."""
        if not self.is_ready:
            return False
        return self._rag.save_index(path)

    def load(self, path: str) -> bool:
        """Load a persisted FAISS index from disk."""
        if not self.is_ready:
            return False
        return self._rag.load_index(path)

    def cleanup(self):
        """Release resources."""
        self._rag = None
        self._initialized = False
        logger.info("VectorStore cleaned up")
