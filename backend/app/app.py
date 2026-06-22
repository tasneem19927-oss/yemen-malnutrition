#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Child Malnutrition Prediction Platform
Flask Application - Production Ready
Features: XGBoost Prediction, RAG System, BioMobileBERT NER, Bilingual RTL/LTR
Offline-First Architecture | WHO/UNICEF Standards Compliant
"""

import os
import sys
import json
import warnings
import logging
import time
from functools import wraps
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, session
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Deep Learning imports with graceful fallback for offline operation
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
    # Check for GPU availability
    if torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info("CUDA is available. Using GPU for NER.")
    else:
        DEVICE = "cpu"
        logger.info("CUDA not available. Using CPU for NER.")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(f"Transformers not available: {e}. NER features disabled.")

warnings.filterwarnings('ignore')

# ==================== Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== Flask App ====================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Enable CORS for local development
if CORS:
    CORS(app, resources={r"/api/*": {"origins": "*"}})  # 16MB

# ==================== Paths ====================
BASE_DIR = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / 'models'
KB_DIR = BASE_DIR / 'knowledge_base'

MODEL_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)

# ==================== Global State ====================
models = {}
metadata = None
feature_importance_data = {}
ner_pipeline = None
kb_data = []
rag_system = None

# ==================== Yemen Regions ====================
YEMEN_REGIONS = [
    "Sana'a", 'Aden', 'Taiz', 'Al-Hodeidah', 'Hadramaut',
    'Ibb', 'Hajjah', "Sa'ada", 'Marib', 'Al-Mahwit'
]

# ==================== Translations ====================
translations = {
    'ar': {
        'title': 'منصة التنبؤ بسوء التغذية لدى الأطفال',
        'subtitle': 'نظام ذكي للكشف المبكر عن سوء التغذية (0-59 شهراً) - وفقاً لمعايير WHO/UNICEF',
        'patient_info': 'معلومات الطفل والأسرة',
        'anthropometry': 'القياسات الجسمية',
        'clinical': 'الحالة السريرية',
        'socioeconomic': 'العوامل الاجتماعية والاقتصادية',
        'age_months': 'العمر (بالأشهر)',
        'sex': 'الجنس',
        'male': 'ذكر',
        'female': 'أنثى',
        'weight': 'الوزن (كغ)',
        'height': 'الطول (سم)',
        'muac': 'محيط منتصف العضد (MUAC) بالسم',
        'oedema': 'وذمة ثنائية الجانب',
        'yes': 'نعم',
        'no': 'لا',
        'breastfeeding': 'حالة الرضاعة الطبيعية',
        'exclusive': 'حصرية',
        'partial': 'جزئية',
        'none': 'لا توجد',
        'vitamin_a': 'ت supplementation بفيتامين أ',
        'diarrhea': 'إسهال في الأسبوعين الماضيين',
        'fever': 'حمى في الأسبوعين الماضيين',
        'cough': 'سعال في الأسبوعين الماضيين',
        'family_size': 'عدد أفراد الأسرة',
        'maternal_education': 'المستوى التعليمي للأم',
        'none_edu': 'بدون تعليم',
        'primary': 'ابتدائي',
        'secondary': 'ثانوي',
        'higher': 'جامعي',
        'wealth_index': 'مستوى الثراء',
        'poor': 'فقير',
        'middle': 'متوسط',
        'rich': 'غني',
        'residence': 'مكان السكن',
        'urban': 'حضري',
        'rural': 'ريفي',
        'region': 'المنطقة الجغرافية',
        'predict': 'تشغيل التنبؤ',
        'results': 'نتائج التنبؤ والتقييم',
        'stunting': 'التقزم (Stunting)',
        'wasting': 'الهزال (Wasting)',
        'underweight': 'نقص الوزن (Underweight)',
        'severity': 'درجة الشدة',
        'normal': 'طبيعي',
        'mild': 'خفيف',
        'moderate': 'متوسط',
        'severe': 'شديد',
        'probability': 'احتمالية الإصابة',
        'risk_score': 'درجة الخطورة',
        'recommendations': 'التوصيات الطبية والتدخلات',
        'references': 'المراجع المعرفية (WHO/UNICEF)',
        'feature_importance': 'أهمية الميزات في التنبؤ',
        'clinical_notes': 'الملاحظات السريرية والتشخيصية',
        'analyze_notes': 'تحليل الملاحظات باستخدام BioMobileBERT NER',
        'print_report': 'طباعة التقرير الطبي',
        'save_report': 'حفظ التقرير (JSON)',
        'notes_placeholder': 'أدخل الملاحظات السريرية هنا (مثال: الطفل يعاني من إسهال مزمن، نقص شهية، وذمة في القدمين...)',
        'required': 'مطلوب',
        'normal_range': 'النطاق الطبيعي',
        'stunting_desc': 'قصر الطول بالنسبة للعمر',
        'wasting_desc': 'نقص الوزن بالنسبة للطول',
        'underweight_desc': 'نقص الوزن بالنسبة للعمر',
        'alert': 'تنبيه',
        'success': 'نجاح',
        'error': 'خطأ',
        'warning': 'تحذير',
        'info': 'معلومة',
        'language': 'اللغة',
        'arabic': 'العربية',
        'english': 'English',
        'offline': 'وضع عدم الاتصال',
        'offline_msg': 'النظام يعمل بدون إنترنت. جميع النماذج محملة محلياً.',
        'generated': 'تاريخ إنشاء التقرير',
        'footer': 'منصة التنبؤ بسوء التغذية - نظام معتمد من WHO/UNICEF Growth Standards'
    },
    'en': {
        'title': 'Child Malnutrition Prediction Platform',
        'subtitle': 'Intelligent Early Detection System (0-59 months) - Based on WHO/UNICEF Standards',
        'patient_info': 'Child & Family Information',
        'anthropometry': 'Anthropometric Measurements',
        'clinical': 'Clinical Status',
        'socioeconomic': 'Socioeconomic Factors',
        'age_months': 'Age (Months)',
        'sex': 'Sex',
        'male': 'Male',
        'female': 'Female',
        'weight': 'Weight (kg)',
        'height': 'Height (cm)',
        'muac': 'MUAC (cm)',
        'oedema': 'Bilateral Pitting Oedema',
        'yes': 'Yes',
        'no': 'No',
        'breastfeeding': 'Breastfeeding Status',
        'exclusive': 'Exclusive',
        'partial': 'Partial',
        'none': 'None',
        'vitamin_a': 'Vitamin A Supplementation',
        'diarrhea': 'Diarrhea in Last 2 Weeks',
        'fever': 'Fever in Last 2 Weeks',
        'cough': 'Cough in Last 2 Weeks',
        'family_size': 'Family Size',
        'maternal_education': 'Maternal Education',
        'none_edu': 'No Education',
        'primary': 'Primary',
        'secondary': 'Secondary',
        'higher': 'Higher',
        'wealth_index': 'Wealth Index',
        'poor': 'Poor',
        'middle': 'Middle',
        'rich': 'Rich',
        'residence': 'Residence',
        'urban': 'Urban',
        'rural': 'Rural',
        'region': 'Geographic Region',
        'predict': 'Run Prediction',
        'results': 'Prediction Results & Assessment',
        'stunting': 'Stunting (HAZ)',
        'wasting': 'Wasting (WHZ/MUAC)',
        'underweight': 'Underweight (WAZ)',
        'severity': 'Severity Level',
        'normal': 'Normal',
        'mild': 'Mild',
        'moderate': 'Moderate',
        'severe': 'Severe',
        'probability': 'Probability',
        'risk_score': 'Risk Score',
        'recommendations': 'Medical Recommendations & Interventions',
        'references': 'Knowledge References (WHO/UNICEF)',
        'feature_importance': 'Feature Importance in Prediction',
        'clinical_notes': 'Clinical Notes & Diagnosis',
        'analyze_notes': 'Analyze Notes with BioMobileBERT NER',
        'print_report': 'Print Medical Report',
        'save_report': 'Save Report (JSON)',
        'notes_placeholder': 'Enter clinical notes here (e.g., child suffers from chronic diarrhea, loss of appetite, edema in feet...)',
        'required': 'Required',
        'normal_range': 'Normal Range',
        'stunting_desc': 'Low height-for-age',
        'wasting_desc': 'Low weight-for-height',
        'underweight_desc': 'Low weight-for-age',
        'alert': 'Alert',
        'success': 'Success',
        'error': 'Error',
        'warning': 'Warning',
        'info': 'Information',
        'language': 'Language',
        'arabic': 'العربية',
        'english': 'English',
        'offline': 'Offline Mode',
        'offline_msg': 'System running without internet. All models loaded locally.',
        'generated': 'Report Generated',
        'footer': 'Child Malnutrition Prediction Platform - WHO/UNICEF Growth Standards Compliant'
    }
}

# ==================== Model Loading ====================
def load_ml_models():
    """Load XGBoost models and metadata from disk."""
    global models, metadata, feature_importance_data
    model_files = {
        'stunting': 'xgb_stunting.joblib',
        'wasting': 'xgb_wasting.joblib',
        'underweight': 'xgb_underweight.joblib'
    }

    try:
        meta_path = MODEL_DIR / 'feature_metadata.joblib'
        if meta_path.exists():
            metadata = joblib.load(meta_path)
            logger.info(f"Feature metadata loaded: {len(metadata.get('features', []))} features")
        else:
            logger.warning("feature_metadata.joblib not found. Run train_models.py first.")

        # Load feature importance if available
        imp_path = MODEL_DIR / 'feature_importance_all.joblib'
        if imp_path.exists():
            feature_importance_data = joblib.load(imp_path)
            logger.info("Feature importance data loaded.")

        for condition, filename in model_files.items():
            path = MODEL_DIR / filename
            if path.exists():
                models[condition] = joblib.load(path)
                logger.info(f"ML model loaded: {condition}")
            else:
                logger.warning(f"Model file not found: {path}")

    except Exception as e:
        logger.error(f"Error loading ML models: {e}")
        models = {}

# ==================== NER Loading ====================
def load_ner_pipeline():
    """Load BioMobileBERT NER pipeline. Downloads on first run, then works offline."""
    global ner_pipeline, TRANSFORMERS_AVAILABLE
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library not installed or import failed. NER disabled.")
        return

    # Check if NER pipeline is already loaded
    if ner_pipeline is not None:
        logger.info("NER pipeline already loaded.")
        return

    model_name = "samrawal/biomedical-mobilebert-ner"
    cache_dir = str(BASE_DIR / ".cache" / "transformers")
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}: Loading BioMobileBERT NER model: {model_name} from cache_dir: {cache_dir}")

            # Try loading tokenizer and model with local_files_only=True first for offline support
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                    cache_dir=cache_dir
                )
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    local_files_only=True,
                    cache_dir=cache_dir
                )
                logger.info(f"BioMobileBERT NER model loaded successfully from local cache on attempt {attempt}.")
            except OSError as e:
                logger.warning(f"Local NER model not found or corrupted: {e}. Attempting download.")
                # If local load fails, try downloading
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=False,
                    cache_dir=cache_dir
                )
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    local_files_only=False,
                    cache_dir=cache_dir
                )
                logger.info(f"BioMobileBERT NER model downloaded and loaded successfully on attempt {attempt}.")

            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=DEVICE
            )
            logger.info("BioMobileBERT NER pipeline initialized successfully.")
            return

        except Exception as e:
            logger.error(f"Failed to load NER pipeline on attempt {attempt}: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All retry attempts failed. NER features will be disabled.")
                ner_pipeline = None
                TRANSFORMERS_AVAILABLE = False # Disable if all attempts fail

    if ner_pipeline is None:
        logger.warning("NER pipeline is not available after all attempts.")

# ==================== Knowledge Base ====================
def load_knowledge_base():
    """Load WHO/UNICEF reference documents."""
    global kb_data
    kb_path = KB_DIR / 'who_unicef_references.json'
    if kb_path.exists():
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            logger.info(f"Knowledge base loaded: {len(kb_data)} references")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            kb_data = []
    else:
        logger.warning(f"Knowledge base not found at {kb_path}")
        kb_data = []

# ==================== RAG System ====================
class RAGSystem:
    """TF-IDF based Retrieval-Augmented Generation system."""

    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2)
        )
        self.doc_vectors = None
        if documents:
            texts = []
            for d in documents:
                text = f"{d.get('category', '')} {d.get('content', '')} {d.get('source', '')}"
                texts.append(text)
            self.doc_vectors = self.vectorizer.fit_transform(texts)
            logger.info(f"RAG index built: {len(documents)} documents, {self.doc_vectors.shape[1]} features")

    def retrieve(self, query, top_k=5):
        if not self.documents or self.doc_vectors is None:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for i in top_indices:
            if similarities[i] > 0.02:
                doc = {**self.documents[i], 'relevance': float(similarities[i])}
                results.append(doc)
        return results

def init_rag():
    global rag_system
    rag_system = RAGSystem(kb_data)

# ==================== Severity Classification ====================
def classify_severity(probability, condition):
    """Classify malnutrition severity based on probability thresholds."""
    thresholds = {
        'stunting': [(0.30, 'normal'), (0.50, 'mild'), (0.70, 'moderate'), (1.0, 'severe')],
        'wasting': [(0.20, 'normal'), (0.40, 'mild'), (0.60, 'moderate'), (1.0, 'severe')],
        'underweight': [(0.30, 'normal'), (0.50, 'mild'), (0.70, 'moderate'), (1.0, 'severe')]
    }
    for thresh, sev in thresholds.get(condition, [(0.5, 'normal'), (1.0, 'mild')]):
        if probability <= thresh:
            return sev
    return 'normal'

# ==================== Flask Routes ====================
@app.route('/')
def index():
    """Render the main application page."""
    lang = session.get('language', 'ar')
    is_rtl = lang == 'ar'
    return render_template('index.html', translations=translations[lang], current_lang=lang, is_rtl=is_rtl, regions=YEMEN_REGIONS)

@app.route('/set_language', methods=['POST'])
def set_language():
    """Endpoint to change the application language."""
    try:
        data = request.get_json()
        lang = data.get('language', 'ar')
        if lang in translations:
            session['language'] = lang
            logger.info(f"Language switched to: {lang}")
        return jsonify({'status': 'success', 'language': lang})
    except Exception as e:
        logger.error(f"Language switch error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for stunting, wasting, and underweight."""
    try:
        if not models or metadata is None:
            return jsonify({
                'error': 'ML models not loaded. Please run: python train_models.py'
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        features = metadata.get('features', [])
        if not features:
            return jsonify({'error': 'Feature metadata corrupted'}), 500

        # Build input vector in correct order
        input_data = []
        
        # Backend Categorical Mappings (Fallback/Validation)
        categorical_mappings = {
            'sex': {'male': 1, 'female': 0},
            'oedema': {'yes': 1, 'no': 0},
            'breastfeeding': {'exclusive': 2, 'partial': 1, 'none': 0},
            'vitamin_a': {'yes': 1, 'no': 0},
            'diarrhea_recent': {'yes': 1, 'no': 0},
            'fever_recent': {'yes': 1, 'no': 0},
            'cough_recent': {'yes': 1, 'no': 0},
            'maternal_education': {'none_edu': 0, 'primary': 1, 'secondary': 2, 'higher': 3},
            'wealth_index': {'poor': 0, 'middle': 1, 'rich': 2},
            'residence_type': {'rural': 0, 'urban': 1}
        }

        for feat in features:
            val = data.get(feat)
            if val is None:
                # Check if there's a mismatch in naming (e.g., diarrhea vs diarrhea_recent)
                if feat == 'diarrhea_recent' and 'diarrhea' in data:
                    val = data['diarrhea']
                elif feat == 'fever_recent' and 'fever' in data:
                    val = data['fever']
                elif feat == 'cough_recent' and 'cough' in data:
                    val = data['cough']
                elif feat == 'residence_type' and 'residence' in data:
                    val = data['residence']
                else:
                    return jsonify({'error': f'Missing required feature: {feat}'}), 400
            
            # Apply mapping if value is a string and exists in mapping
            if isinstance(val, str) and feat in categorical_mappings:
                val_lower = val.lower()
                if val_lower in categorical_mappings[feat]:
                    val = categorical_mappings[feat][val_lower]
            
            # Handle region mapping if it's a string
            if feat == 'region' and isinstance(val, str):
                try:
                    val = YEMEN_REGIONS.index(val)
                except ValueError:
                    return jsonify({'error': f'Invalid region: {val}'}), 400

            try:
                input_data.append(float(val))
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid numeric value for {feat}: {val}'}), 400

        X = np.array([input_data])

        # Run predictions for all three conditions
        results = {}
        conditions = ['stunting', 'wasting', 'underweight']

        for condition in conditions:
            model = models.get(condition)
            if model is None:
                logger.warning(f"Model missing for {condition}")
                continue

            prob = float(model.predict_proba(X)[0][1])
            pred = int(model.predict(X)[0])
            severity = classify_severity(prob, condition)

            results[condition] = {
                'prediction': pred,
                'probability': round(prob, 4),
                'severity': severity,
                'risk_score': round(prob * 100, 2)
            }

        # RAG-based recommendations
        query_parts = [
            f"age {data.get('age_months')} months",
            f"weight {data.get('weight')} kg",
            f"height {data.get('height')} cm",
            f"muac {data.get('muac')} cm"
        ]

        region_idx = int(data.get('region', 0))
        if 0 <= region_idx < len(YEMEN_REGIONS):
            query_parts.append(f"region {YEMEN_REGIONS[region_idx]}")

        if int(data.get('oedema', 0)):
            query_parts.append("oedema kwashiorkor")
        if int(data.get('diarrhea_recent', 0)):
            query_parts.append("diarrhea dehydration")
        if int(data.get('fever_recent', 0)):
            query_parts.append("fever infection")

        query = " ".join(query_parts)
        rag_results = rag_system.retrieve(query, top_k=5) if rag_system else []

        recommendations = []
        seen_cats = set()
        for rec in rag_results:
            cat = rec.get('category', 'General')
            if cat not in seen_cats:
                recommendations.append({
                    'category': cat,
                    'text': rec.get('content', ''),
                    'source': rec.get('source', 'WHO/UNICEF'),
                    'relevance': round(rec.get('relevance', 0), 3)
                })
                seen_cats.add(cat)

        # Add severity-based urgent recommendations
        for condition in conditions:
            if condition not in results:
                continue
            sev = results[condition]['severity']
            if sev in ['moderate', 'severe']:
                rec_texts = {
                    'stunting': f"Stunting detected at {sev} level. Nutritional rehabilitation, micronutrient supplementation, and dietary diversification required. Refer to nutrition counseling.",
                    'wasting': f"Wasting detected at {sev} level. URGENT: Refer to CMAM program immediately. Check for medical complications, dehydration, and infection.",
                    'underweight': f"Underweight detected at {sev} level. Enhanced dietary counseling, growth monitoring, and possible supplementary feeding needed."
                }
                recommendations.append({
                    'category': 'Urgent Intervention',
                    'text': rec_texts[condition],
                    'source': 'WHO/UNICEF Guidelines',
                    'relevance': 0.99
                })

        # Feature importance (use trained data if available)
        feature_importance = []
        if feature_importance_data:
            # Use stunting model importance as representative
            imp_df = feature_importance_data.get('xgb_stunting')
            if imp_df is not None and hasattr(imp_df, 'iterrows'):
                for _, row in imp_df.iterrows():
                    feature_importance.append({
                        'feature': str(row['feature']),
                        'importance': round(float(row['importance']), 4)
                    })

        if not feature_importance:
            # Fallback: equal distribution
            for feat in features:
                feature_importance.append({
                    'feature': feat,
                    'importance': round(1.0 / len(features), 4)
                })

        feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        # WHO/UNICEF references
        references = []
        for item in kb_data[:6]:
            references.append({
                'source': item.get('source', 'WHO'),
                'category': item.get('category', 'General'),
                'content': item.get('content', '')[:350],
                'citation': item.get('citation', '')
            })

        return jsonify({
            'stunting': results.get('stunting', {}),
            'wasting': results.get('wasting', {}),
            'underweight': results.get('underweight', {}),
            'recommendations': recommendations,
            'references': references,
            'feature_importance': feature_importance[:10],
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0.0-mics6-yemen'
        })

    except Exception as e:
        logger.exception("Prediction endpoint error")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_notes', methods=['POST'])
def analyze_notes():
    """Clinical notes NER analysis using BioMobileBERT."""
    try:
        data = request.get_json()
        notes = data.get('notes', '').strip()

        if not notes:
            return jsonify({
                'entities': [],
                'clinical_summary': '',
                'note_length': 0,
                'entity_count': 0,
                'ner_available': (ner_pipeline is not None and TRANSFORMERS_AVAILABLE)
            })

        entities = []
        if ner_pipeline and TRANSFORMERS_AVAILABLE:
            logger.info(f"Analyzing notes with NER pipeline. Device: {DEVICE}")
            try:
                ner_results = ner_pipeline(notes)
                for ent in ner_results:
                    entities.append({
                        'text': ent.get('word', ent.get('entity_group', '')),
                        'label': ent.get('entity_group', 'UNKNOWN'),
                        'score': round(float(ent.get('score', 0)), 3),
                        'start': ent.get('start', 0),
                        'end': ent.get('end', 0)
                    })
            except Exception as ner_err:
                logger.warning(f"NER processing error: {ner_err}")

        # Clinical summary generation (rule-based + entity-aware)
        summary_parts = []
        note_lower = notes.lower()

        # Check for critical markers
        critical_terms = {
            'oedema': ['oedema', 'edema', 'وذمة', 'تورم'],
            'diarrhea': ['diarrhea', 'diarrhoea', 'إسهال', 'اسهال'],
            'fever': ['fever', 'حمى', 'سخونة'],
            'cough': ['cough', 'سعال', 'كحة'],
            'appetite': ['appetite', 'anorexia', 'شهية', 'فقدان الشهية'],
            'vomiting': ['vomiting', 'vomit', 'قيء', 'استفراغ']
        }

        detected = {}
        for marker, terms in critical_terms.items():
            for term in terms:
                if term in note_lower:
                    detected[marker] = True
                    break

        if 'oedema' in detected:
            summary_parts.append("Bilateral pitting oedema detected - possible kwashiorkor (severe acute malnutrition). Immediate referral required.")
        if 'diarrhea' in detected:
            summary_parts.append("Recent diarrhea reported - high risk of dehydration and nutrient loss. Assess hydration status.")
        if 'fever' in detected:
            summary_parts.append("Fever present - check for underlying infection (malaria, respiratory, or gastrointestinal).")
        if 'cough' in detected:
            summary_parts.append("Respiratory symptoms (cough) noted - assess for pneumonia or other respiratory infection.")
        if 'appetite' in detected:
            summary_parts.append("Appetite changes reported - monitor food intake and consider micronutrient assessment.")
        if 'vomiting' in detected:
            summary_parts.append("Vomiting reported - severe dehydration risk. Urgent medical evaluation needed.")

        # Entity-based summary
        if entities:
            entity_labels = [e['label'] for e in entities]
            if any(l in ['SYMPTOM', 'DISEASE', 'PROBLEM'] for l in entity_labels):
                summary_parts.append("Clinical symptoms/diseases identified in notes via NER analysis.")
            if any(l in ['MEDICATION', 'TREATMENT'] for l in entity_labels):
                summary_parts.append("Medications/treatments mentioned - verify appropriateness for malnourished child.")

        if not summary_parts:
            summary_parts.append("No critical clinical markers identified in notes. Continue routine growth monitoring and nutritional counseling.")

        return jsonify({
            'entities': entities,
            'clinical_summary': ' '.join(summary_parts),
            'note_length': len(notes),
            'entity_count': len(entities),
            'ner_available': (ner_pipeline is not None and TRANSFORMERS_AVAILABLE)
        })

    except Exception as e:
        logger.exception("Notes analysis error")
        return jsonify({'error': str(e)}), 500

@app.route('/healthz', methods=['GET'])
def healthz():
    """Health check endpoint for the application and NER model status."""
    status = {
        "app_status": "ok",
        "ml_models_loaded": bool(models),
        "ner_pipeline_loaded": (ner_pipeline is not None and TRANSFORMERS_AVAILABLE),
        "rag_system_loaded": (rag_system is not None),
        "timestamp": datetime.now().isoformat()
    }
    if not status["ml_models_loaded"]:
        status["ml_models_error"] = "ML models not loaded. Check logs."
    if not status["ner_pipeline_loaded"]:
        status["ner_error"] = "NER pipeline not loaded. Check logs for Transformer issues."
    if not status["rag_system_loaded"]:
        status["rag_error"] = "RAG system not loaded. Check knowledge base."

    http_status = 200 if all(status.values( )) else 503
    return jsonify(status), http_status


# ==================== Startup Initialization ====================
logger.info("=" * 60 )
logger.info("Child Malnutrition Prediction Platform Starting...")
logger.info("=" * 60)

def initialize_app():
    """Initialize all components with proper error handling."""
    try:
        load_ml_models()
    except Exception as e:
        logger.error(f"ML models failed to load: {e}")

    try:
        load_ner_pipeline()
    except Exception as e:
        logger.warning(f"NER pipeline not available: {e}")

    try:
        load_knowledge_base()
    except Exception as e:
        logger.warning(f"Knowledge base not loaded: {e}")

    try:
        init_rag()
    except Exception as e:
        logger.warning(f"RAG system not initialized: {e}")

    logger.info("Application initialization complete.")

# Initialize on startup
initialize_app()

# ==================== Main Entry ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    logger.info(f"Starting server on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
