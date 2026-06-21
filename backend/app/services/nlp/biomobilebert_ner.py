"""
BioMobileBERT NER Service
Implements Chapter 3 methodology from the integration document.
Replaces dictionary-based NER with deep learning entity extraction.
Supports Arabic and English bilingual processing.
"""

import os
import re
import time
import logging
import threading
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)


@dataclass
class NERResult:
    """Single entity extraction result."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float
    language: str
    normalized_code: Optional[str] = None
    alt_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "language": self.language,
            "normalized_code": self.normalized_code,
            "alt_names": self.alt_names,
            "metadata": self.metadata,
        }


@dataclass
class NERDocumentResult:
    """Complete document NER result."""
    entities: List[NERResult]
    language: str
    malnutrition_types: List[str] = field(default_factory=list)
    severity_level: Optional[str] = None
    entities_by_type: Dict[str, List[Dict]] = field(default_factory=dict)
    summary: str = ""
    total_entities: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "language": self.language,
            "malnutrition_types": self.malnutrition_types,
            "severity_level": self.severity_level,
            "entities_by_type": self.entities_by_type,
            "summary": self.summary,
            "total_entities": self.total_entities,
            "processing_time_ms": self.processing_time_ms,
        }


class BioMobileBERTNER:
    """
    BioMobileBERT NER Singleton Service.

    Based on Chapter 3 methodology:
    - Fine-tuned on malnutrition-specific biomedical data
    - Supports Arabic and English
    - Entity types: DISEASE, SYMPTOM, TREATMENT, MEASUREMENT, NUTRIENT, DEMOGRAPHIC
    - ONNX export support for edge deployment
    """

    _instance: Optional['BioMobileBERTNER'] = None
    _lock = threading.Lock()

    # Entity label mapping (IOB2 format)
    LABEL_NAMES = [
        "O",                    # Outside
        "B-DISEASE", "I-DISEASE",
        "B-SYMPTOM", "I-SYMPTOM",
        "B-TREATMENT", "I-TREATMENT",
        "B-MEASUREMENT", "I-MEASUREMENT",
        "B-NUTRIENT", "I-NUTRIENT",
        "B-DEMOGRAPHIC", "I-DEMOGRAPHIC",
    ]

    # Arabic-specific medical terms for post-processing
    ARABIC_MEDICAL_TERMS = {
        "DISEASE": [
            "سوء التغذية", "التقزم", "الهزال", "نقص الوزن", "الأنيميا",
            "الإسهال", "الملاريا", "الحصبة", "الالتهاب الرئوي", "السُّل",
            "تقزم", "هزال", "نقص وزن", "أنيميا", "إسهال", "ملاريا",
            "malnutrition", "stunting", "wasting", "underweight", "anemia",
            "diarrhea", "malaria", "measles", "pneumonia", "tuberculosis",
        ],
        "SYMPTOM": [
            "حمى", "سعال", "إسهال", "قيء", "إرهاق", "فقدان الشهية",
            "تورم", "وذمة", "شحوب", "ضعف", "هزال", "تقزم",
            "fever", "cough", "diarrhea", "vomiting", "fatigue",
            "loss of appetite", "swelling", "oedema", "pallor", "weakness",
        ],
        "TREATMENT": [
            "RUTF", "F75", "F100", "العلاج التغذوي", "الفيتامينات",
            "الحديد", "الزنك", "الألبندازول", "التحصين", "التغذية التكميلية",
            "RUTF", "F75", "F100", "therapeutic food", "vitamins",
            "iron", "zinc", "albendazole", "immunization", "complementary feeding",
        ],
        "MEASUREMENT": [
            "وزن", "طول", "محيط الذراع", "BMI", "HAZ", "WHZ", "WAZ",
            "كيلوغرام", "سنتيمتر", "مليمتر", "درجة حرارة",
            "weight", "height", "MUAC", "BMI", "HAZ", "WHZ", "WAZ",
            "kilogram", "centimeter", "millimeter", "temperature",
        ],
        "NUTRIENT": [
            "بروتين", "طاقة", "فيتامين A", "فيتامين D", "حديد", "زنك",
            "يود", "حمض الفوليك", "الكالسيوم", "الأحماض الدهنية",
            "protein", "energy", "vitamin A", "vitamin D", "iron", "zinc",
            "iodine", "folic acid", "calcium", "fatty acids", "micronutrients",
        ],
        "DEMOGRAPHIC": [
            "طفل", "رضيع", "أم", "أسرة", "منطقة ريفية", "مخيم",
            "ذكر", "أنثى", "عمر", "شهر", "سنة",
            "child", "infant", "mother", "family", "rural", "camp",
            "male", "female", "age", "month", "year",
        ],
    }

    def __init__(
        self,
        model_name: str = "nlpie/bio-mobilebert",
        fine_tuned_model_path: Optional[str] = None,
        lazy_load: bool = True,
        cache_entities: bool = True,
        use_onnx: bool = False,
        onnx_model_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.fine_tuned_model_path = fine_tuned_model_path
        self.lazy_load = lazy_load
        self._cache_enabled = cache_entities
        self.use_onnx = use_onnx
        self.onnx_model_path = onnx_model_path

        self._model = None
        self._tokenizer = None
        self._model_loaded = False
        self._onnx_session = None

        # Entity cache
        self._entity_cache: Dict[str, List[NERResult]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        if not lazy_load:
            self._load_model()

    @classmethod
    def get_instance(cls, **kwargs) -> 'BioMobileBERTNER':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    def _load_model(self) -> None:
        """Load BioMobileBERT model and tokenizer."""
        if self._model_loaded:
            return

        try:
            path_to_load = self.fine_tuned_model_path or self.model_name
            logger.info(f"Loading BioMobileBERT from {path_to_load}")

            if self.use_onnx and self.onnx_model_path and os.path.exists(self.onnx_model_path):
                self._load_onnx_model()
            else:
                self._load_pytorch_model(path_to_load)

            self._model_loaded = True
            logger.info("BioMobileBERT NER model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load BioMobileBERT: {e}")
            self._model_loaded = False

    def _load_pytorch_model(self, model_path: str) -> None:
        """Load PyTorch model."""
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(self.LABEL_NAMES),
        )
        self._model.eval()

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._device = device
        logger.info(f"PyTorch model loaded on {device}")

    def _load_onnx_model(self) -> None:
        """Load ONNX model for edge deployment."""
        try:
            import onnxruntime as ort

            # Configure ONNX Runtime for edge optimization
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 2
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._onnx_session = ort.InferenceSession(
                self.onnx_model_path,
                sess_options,
                providers=['CPUExecutionProvider'],
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.fine_tuned_model_path or self.model_name
            )
            logger.info("ONNX model loaded for edge inference")
        except ImportError:
            logger.warning("ONNX Runtime not available, falling back to PyTorch")
            self._load_pytorch_model(self.fine_tuned_model_path or self.model_name)

    def _detect_language(self, text: str) -> str:
        """Detect if text is Arabic or English."""
        arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
        total = len(text.replace(" ", ""))
        if total == 0:
            return "en"
        ratio = arabic_chars / total
        return "ar" if ratio > 0.3 else "en"

    def extract_medical_entities(
        self,
        text: str,
        language: str = "auto",
        entity_types: Optional[List[str]] = None,
    ) -> NERDocumentResult:
        """
        Extract medical entities from clinical text.

        Args:
            text: Input clinical text (Arabic or English)
            language: "auto", "en", or "ar"
            entity_types: Filter by specific entity types

        Returns:
            NERDocumentResult with extracted entities
        """
        start_time = time.perf_counter()

        # Check cache
        if self._cache_enabled and text in self._entity_cache:
            self._cache_hits += 1
            cached = self._entity_cache[text]
            return NERDocumentResult(
                entities=cached,
                language=self._detect_language(text),
                total_entities=len(cached),
                processing_time_ms=0.0,
            )

        self._cache_misses += 1

        # Detect language
        if language == "auto":
            language = self._detect_language(text)

        # Load model if needed
        if not self._model_loaded:
            self._load_model()

        if not self._model_loaded:
            logger.error("Model not loaded, using fallback extraction")
            entities = self._fallback_extraction(text, language)
        else:
            entities = self._extract_with_model(text, language)

        # Post-process: enhance with domain knowledge
        entities = self._post_process_entities(entities, text, language)

        # Filter by entity types if requested
        if entity_types:
            entities = [e for e in entities if e.entity_type in entity_types]

        # Cache results
        if self._cache_enabled:
            self._entity_cache[text] = entities

        # Build result
        processing_time = (time.perf_counter() - start_time) * 1000

        malnutrition_types = [
            e.text for e in entities 
            if e.entity_type == "DISEASE" and "malnutrition" in e.text.lower()
        ]

        severity_level = self._detect_severity(entities, text)

        entities_by_type = {}
        for e in entities:
            if e.entity_type not in entities_by_type:
                entities_by_type[e.entity_type] = []
            entities_by_type[e.entity_type].append(e.to_dict())

        summary = self._generate_summary(entities, language)

        result = NERDocumentResult(
            entities=entities,
            language=language,
            malnutrition_types=malnutrition_types,
            severity_level=severity_level,
            entities_by_type=entities_by_type,
            summary=summary,
            total_entities=len(entities),
            processing_time_ms=processing_time,
        )

        return result

    def _extract_with_model(self, text: str, language: str) -> List[NERResult]:
        """Extract entities using BioMobileBERT model."""
        if self.use_onnx and self._onnx_session:
            return self._extract_with_onnx(text, language)
        return self._extract_with_pytorch(text, language)

    def _extract_with_pytorch(self, text: str, language: str) -> List[NERResult]:
        """Extract using PyTorch model."""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Move inputs to same device as model
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(batch_index=0)

        return self._decode_predictions(text, tokens, predictions, word_ids, language)

    def _extract_with_onnx(self, text: str, language: str) -> List[NERResult]:
        """Extract using ONNX model (edge optimized)."""
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=512,
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])),
        }

        outputs = self._onnx_session.run(None, onnx_inputs)
        predictions = np.argmax(outputs[0], axis=2)[0]
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(batch_index=0)

        return self._decode_predictions(text, tokens, predictions, word_ids, language)

    def _decode_predictions(
        self,
        text: str,
        tokens: List[str],
        predictions: np.ndarray,
        word_ids: Optional[List[int]],
        language: str,
    ) -> List[NERResult]:
        """Decode model predictions to entities."""
        entities = []
        current_entity = None

        # Build character index map
        words = text.split()
        char_idx = 0
        word_positions = []
        for word in words:
            start = text.find(word, char_idx)
            end = start + len(word)
            word_positions.append((start, end))
            char_idx = end + 1

        for i, (token, label_id) in enumerate(zip(tokens, predictions)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            label = self.LABEL_NAMES[label_id] if label_id < len(self.LABEL_NAMES) else "O"

            # Get word position
            word_idx = word_ids[i] if word_ids else None
            if word_idx is not None and word_idx < len(word_positions):
                start_char, end_char = word_positions[word_idx]
            else:
                start_char, end_char = -1, -1

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = NERResult(
                    text=token.replace("##", ""),
                    entity_type=label[2:],
                    start_pos=start_char,
                    end_pos=end_char,
                    confidence=0.95,
                    language=language,
                )
            elif label.startswith("I-"):
                if current_entity and current_entity.entity_type == label[2:]:
                    current_entity.text += " " + token.replace("##", "")
                    current_entity.end_pos = end_char
                    current_entity.confidence = min(current_entity.confidence, 0.90)
                else:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = NERResult(
                        text=token.replace("##", ""),
                        entity_type=label[2:],
                        start_pos=start_char,
                        end_pos=end_char,
                        confidence=0.85,
                        language=language,
                    )
            else:  # O tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

    def _post_process_entities(
        self,
        entities: List[NERResult],
        text: str,
        language: str,
    ) -> List[NERResult]:
        """Enhance entities with domain knowledge and normalization."""
        enhanced = []

        for entity in entities:
            # Normalize text
            normalized = entity.text.strip()

            # Check against known medical terms
            terms = self.ARABIC_MEDICAL_TERMS.get(entity.entity_type, [])
            for term in terms:
                if term.lower() in normalized.lower() or normalized.lower() in term.lower():
                    entity.confidence = min(entity.confidence + 0.05, 0.99)
                    break

            # Set normalized code
            entity.normalized_code = self._normalize_entity(entity.entity_type, normalized, language)

            enhanced.append(entity)

        return enhanced

    def _normalize_entity(self, entity_type: str, text: str, language: str) -> str:
        """Normalize entity to standard medical code."""
        normalization_map = {
            "DISEASE": {
                "ar": {
                    "سوء التغذية": "E46", "التقزم": "E34.3", "الهزال": "E41",
                    "نقص الوزن": "E44.1", "الأنيميا": "D64.9",
                },
                "en": {
                    "malnutrition": "E46", "stunting": "E34.3", "wasting": "E41",
                    "underweight": "E44.1", "anemia": "D64.9",
                },
            },
            "SYMPTOM": {
                "ar": {"حمى": "R50.9", "سعال": "R05", "إسهال": "K52.9"},
                "en": {"fever": "R50.9", "cough": "R05", "diarrhea": "K52.9"},
            },
        }

        type_map = normalization_map.get(entity_type, {})
        lang_map = type_map.get(language, {})

        for key, code in lang_map.items():
            if key in text.lower():
                return code

        return f"{entity_type}_UNK"

    def _detect_severity(self, entities: List[NERResult], text: str) -> Optional[str]:
        """Detect severity level from entities and text."""
        severity_keywords = {
            "severe": ["severe", "حاد", "شديد", "critical", "حرج"],
            "moderate": ["moderate", "متوسط", "moderate"],
            "mild": ["mild", "خفيف", "mild"],
        }

        text_lower = text.lower()
        for severity, keywords in severity_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return severity

        return None

    def _generate_summary(self, entities: List[NERResult], language: str) -> str:
        """Generate human-readable summary of extracted entities."""
        if not entities:
            return "No medical entities identified."

        entity_types = sorted(set(e.entity_type for e in entities))
        summary_parts = []

        for etype in entity_types:
            items = list(set(e.text for e in entities if e.entity_type == etype))
            if language == "ar":
                type_names = {
                    "DISEASE": "الأمراض", "SYMPTOM": "الأعراض",
                    "TREATMENT": "العلاجات", "MEASUREMENT": "القياسات",
                    "NUTRIENT": "المغذيات", "DEMOGRAPHIC": "البيانات الديموغرافية",
                }
                summary_parts.append(f"{type_names.get(etype, etype)}: {', '.join(items)}")
            else:
                summary_parts.append(f"{etype}: {', '.join(items)}")

        if language == "ar":
            return f"التقرير الطبي يحتوي على: {'؛ '.join(summary_parts)}."
        return f"Medical report contains: {'; '.join(summary_parts)}."

    def _fallback_extraction(self, text: str, language: str) -> List[NERResult]:
        """Fallback rule-based extraction when model is unavailable."""
        entities = []

        for entity_type, terms in self.ARABIC_MEDICAL_TERMS.items():
            for term in terms:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append(NERResult(
                        text=match.group(),
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.70,
                        language=language,
                    ))

        return entities

    def clear_cache(self) -> None:
        """Clear entity cache."""
        self._entity_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Entity cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._entity_cache),
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
        }
