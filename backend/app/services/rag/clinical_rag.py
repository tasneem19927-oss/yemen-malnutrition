"""
Clinical RAG (Retrieval-Augmented Generation) System.
Uses FAISS with multi-qa-MiniLM-L6-cos-v1 embeddings for offline clinical evidence retrieval.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class ClinicalRAG:
    """
    Clinical RAG System for evidence-based malnutrition recommendations.

    Features:
    - FAISS vector store with IndexIVFFlat
    - multi-qa-MiniLM-L6-cos-v1 embeddings
    - Offline search capability
    - Bilingual support (Arabic/English)
    - WHO/UNICEF evidence base
    """

    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.knowledge_base = {}
        self._initialized = False
        self._init_embedding_model()
        self._init_faiss_index()

    def _init_embedding_model(self):
        """Initialize sentence transformer embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            logger.info("Embedding model loaded: multi-qa-MiniLM-L6-cos-v1")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self.embedding_model = None

    def _init_faiss_index(self):
        """Initialize FAISS vector index."""
        try:
            import faiss

            # Dimension for multi-qa-MiniLM-L6-cos-v1
            self.dimension = 384

            # Create IVF index for fast approximate search
            quantizer = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, self.dimension,
                nlist=100,  # Number of clusters
                metric=faiss.METRIC_INNER_PRODUCT,
            )

            # Train with dummy data if empty
            if not self.faiss_index.is_trained:
                dummy = np.random.random((1000, self.dimension)).astype("float32")
                self.faiss_index.train(dummy)

            logger.info("FAISS IVF index initialized")

        except ImportError:
            logger.warning("FAISS not available, using fallback search")
            self.faiss_index = None

    async def search(
        self,
        query: str,
        language: str = "en",
        top_k: int = 5,
        source_type: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant clinical evidence.

        Args:
            query: Search query
            language: "en" or "ar"
            top_k: Number of results
            source_type: Filter by source type
            year_min/year_max: Filter by publication year

        Returns:
            List of relevant knowledge base entries with scores
        """
        if not self.embedding_model:
            return await self._fallback_search(query, language, top_k)

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity

        # Search FAISS index
        if self.faiss_index and self.faiss_index.ntotal > 0:
            scores, indices = self.faiss_index.search(query_embedding, top_k * 2)  # Get extra for filtering

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue

                doc = self.knowledge_base.get(int(idx))
                if not doc:
                    continue

                # Apply filters
                if source_type and doc.get("source_type") != source_type:
                    continue
                if year_min and doc.get("year", 9999) < year_min:
                    continue
                if year_max and doc.get("year", 0) > year_max:
                    continue

                results.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "clinical_summary": doc.get("clinical_summary"),
                    "citation": doc.get("citation"),
                    "source_type": doc.get("source_type"),
                    "year": doc.get("year"),
                    "relevance_score": float(score),
                })

                if len(results) >= top_k:
                    break

            return results

        return await self._fallback_search(query, language, top_k)

    async def query(
        self,
        query: str,
        language: str = "en",
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Natural language query with clinical answer generation.

        Args:
            query: Clinical question
            language: "en" or "ar"
            context: Optional patient context

        Returns:
            Answer with evidence sources
        """
        # Retrieve relevant evidence
        evidence = await self.search(query, language, top_k=5)

        if not evidence:
            return {
                "text": self._get_default_answer(query, language),
                "sources": [],
                "confidence": 0.0,
            }

        # Generate answer from evidence (simplified - no LLM in offline mode)
        answer_text = self._generate_answer(query, evidence, language, context)

        # Calculate confidence
        avg_score = sum(e["relevance_score"] for e in evidence) / len(evidence)
        confidence = min(avg_score * 1.2, 0.95)  # Scale and cap

        return {
            "text": answer_text,
            "sources": evidence,
            "confidence": round(confidence, 3),
        }

    async def get_recommendations(
        self,
        prediction_result: Dict,
        patient: Any,
        measurement: Any,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Get clinical recommendations based on prediction results.

        Args:
            prediction_result: Output from prediction engine
            patient: Patient model
            measurement: Measurement model
            language: "en" or "ar"

        Returns:
            Recommendations with evidence
        """
        # Build clinical query
        query = prediction_result.get("clinical_query", "")

        # Retrieve evidence
        evidence = await self.search(query, language, top_k=3)

        # Determine intervention
        overall_risk = prediction_result.get("overall_risk", "normal")

        intervention = self._get_intervention(overall_risk, language)
        referral = self._determine_referral(overall_risk, prediction_result)

        return {
            "evidence": evidence,
            "recommendation": intervention,
            "referral_needed": referral["needed"],
            "referral_urgency": referral["urgency"],
            "who_reference": self._get_who_reference(overall_risk),
        }

    async def add_document(self, document: Any) -> bool:
        """Add document to FAISS index."""
        if not self.embedding_model or not self.faiss_index:
            return False

        try:
            # Generate embedding
            text = f"{document.title} {document.abstract or ''} {document.clinical_summary or ''}"
            embedding = self.embedding_model.encode([text], convert_to_numpy=True)
            embedding = embedding.astype("float32")
            faiss.normalize_L2(embedding)

            # Add to index
            doc_id = len(self.knowledge_base)
            self.faiss_index.add(embedding)
            self.knowledge_base[doc_id] = {
                "id": document.id,
                "title": document.title,
                "clinical_summary": document.clinical_summary,
                "citation": document.citation,
                "source_type": document.source_type,
                "year": document.year,
            }

            logger.info(f"Added document {document.id} to FAISS index")
            return True

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False

    async def _fallback_search(self, query: str, language: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search when FAISS is unavailable."""
        results = []
        query_terms = set(query.lower().split())

        for doc_id, doc in self.knowledge_base.items():
            score = 0
            text = f"{doc.get('title', '')} {doc.get('clinical_summary', '')}".lower()

            for term in query_terms:
                if term in text:
                    score += 1

            if score > 0:
                results.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "clinical_summary": doc.get("clinical_summary"),
                    "citation": doc.get("citation"),
                    "source_type": doc.get("source_type"),
                    "year": doc.get("year"),
                    "relevance_score": score / len(query_terms),
                })

        # Sort by score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def _generate_answer(
        self,
        query: str,
        evidence: List[Dict],
        language: str,
        context: Optional[Dict],
    ) -> str:
        """Generate answer from retrieved evidence."""
        if language == "ar":
            parts = ["بناءً على الأدلة السريرية:"]
        else:
            parts = ["Based on clinical evidence:"]

        for i, doc in enumerate(evidence[:3], 1):
            summary = doc.get("clinical_summary", doc.get("title", ""))
            if language == "ar":
                parts.append(f"{i}. {summary}")
            else:
                parts.append(f"{i}. {summary}")

        if language == "ar":
            parts.append("يرجى استشارة أخصائي تغذية للتقييم الشامل.")
        else:
            parts.append("Please consult a nutrition specialist for comprehensive evaluation.")

        return " ".join(parts)

    def _get_default_answer(self, query: str, language: str) -> str:
        """Get default answer when no evidence is found."""
        if language == "ar":
            return "لم يتم العثور على أدلة سريرية محددة. يرجى الرجوع إلى إرشادات WHO.")
        return "No specific clinical evidence found. Please refer to WHO guidelines."

    def _get_intervention(self, risk_level: str, language: str) -> str:
        """Get intervention recommendation based on risk level."""
        interventions = {
            "en": {
                "normal": "Continue routine care. Monitor growth monthly. Ensure balanced diet and immunization.",
                "mild": "Nutritional counseling. Micronutrient supplementation. Follow-up in 2-4 weeks.",
                "moderate": "Community-based management (CMAM). RUTF supplementation. Weekly monitoring. Treat infections.",
                "severe": "URGENT hospitalization. Inpatient SAM treatment (F75/F100). Medical stabilization. Refer immediately.",
            },
            "ar": {
                "normal": "استمر في الرعاية الروتينية. راقب النمو شهرياً. تأكد من التغذية المتوازنة والتحصين.",
                "mild": "الاستشارة التغذوية. مكملات المغذيات الدقيقة. المتابعة خلال 2-4 أسابيع.",
                "moderate": "الإدارة المجتمعية (CMAM). مكملات RUTF. المتابعة الأسبوعية. علاج العدوى.",
                "severe": "الاستشفاء العاجل. علاج SAM داخل المستشفى (F75/F100). التثبيت الطبي. الإحالة فوراً.",
            },
        }
        return interventions.get(language, interventions["en"]).get(risk_level, "Consult specialist.")

    def _determine_referral(self, risk_level: str, prediction: Dict) -> Dict[str, Any]:
        """Determine if referral is needed."""
        if risk_level == "severe":
            return {"needed": True, "urgency": "emergency"}
        elif risk_level == "moderate":
            return {"needed": True, "urgency": "urgent"}
        elif risk_level == "mild":
            return {"needed": False, "urgency": "routine"}
        return {"needed": False, "urgency": None}

    def _get_who_reference(self, risk_level: str) -> str:
        """Get WHO guideline reference."""
        references = {
            "normal": "WHO Child Growth Standards (2006)",
            "mild": "WHO Guideline: Updates on the management of severe acute malnutrition (2013)",
            "moderate": "WHO/UNICEF/WFP Joint Statement: Community-based management of severe acute malnutrition (2007)",
            "severe": "WHO Guidelines for the Inpatient Treatment of Severely Malnourished Children (2003)",
        }
        return references.get(risk_level, "WHO Guidelines")

    def save_index(self, path: str) -> bool:
        """Save FAISS index to disk."""
        if not self.faiss_index:
            return False

        try:
            import faiss
            faiss.write_index(self.faiss_index, path)

            # Save metadata
            metadata_path = path + ".meta.json"
            with open(metadata_path, "w") as f:
                json.dump(self.knowledge_base, f, default=str)

            logger.info(f"FAISS index saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Load FAISS index from disk."""
        try:
            import faiss
            self.faiss_index = faiss.read_index(path)

            # Load metadata
            metadata_path = path + ".meta.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.knowledge_base = json.load(f)

            logger.info(f"FAISS index loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
            def __init__(self):
    self.embedding_model = None
    self.faiss_index = None
    self.knowledge_base = {}
    self._initialized = False
    self._init_embedding_model()
    self._init_faiss_index()
