"""
XGBoost Prediction Engine for Stunting, Wasting, and Underweight.
Implements 3 independent models with feature engineering.
"""

import numpy as np
import pickle
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import xgboost as xgb

from app.core.config import settings
from app.schemas.prediction import PredictionInput, PredictionFeatures, SeverityResult

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features from raw patient data."""

    # Feature mappings
    SEX_MAP = {"male": 0, "female": 1}
    RESIDENCE_MAP = {"urban": 0, "rural": 1, "camp": 2}
    EDUCATION_MAP = {
        "none": 0, "primary": 1, "secondary": 2, "higher": 3, "unknown": -1
    }
    WEALTH_MAP = {
        "poorest": 0, "poorer": 1, "middle": 2, "richer": 3, "richest": 4, "unknown": -1
    }

    def __init__(self):
        self.feature_names = [
            "age_months", "sex", "weight", "height", "oedema",
            "breastfeeding", "vitamin_a", "diarrhea_recent", "fever_recent",
            "cough_recent", "maternal_education", "wealth_index", "residence_type",
            "haz", "whz", "waz", "bmi", "weight_height_ratio",
            "age_weight_interaction", "age_height_interaction",
            "health_risk_score", "nutrition_risk_score", "muac_zscore",
            "head_circumference_zscore", "growth_velocity",
            "wasting_stunting_interaction", "age_group", "season",
            "food_security_index"
        ]

    def engineer_features(self, input_data: PredictionInput) -> PredictionFeatures:
        """Create engineered features from raw input."""

        # Encode categorical variables
        sex_encoded = self.SEX_MAP.get(input_data.sex.lower(), 0)
        residence_encoded = self.RESIDENCE_MAP.get(input_data.residence_type.lower() if input_data.residence_type else "urban", 0)
        education_encoded = self.EDUCATION_MAP.get(input_data.maternal_education.lower() if input_data.maternal_education else "unknown", -1)
        wealth_encoded = self.WEALTH_MAP.get(input_data.wealth_index.lower() if input_data.wealth_index else "unknown", -1)

        # Calculate derived features
        bmi = input_data.weight_kg / ((input_data.height_cm / 100) ** 2)
        weight_height_ratio = input_data.weight_kg / input_data.height_cm
        age_weight_interaction = input_data.age_months * input_data.weight_kg
        age_height_interaction = input_data.age_months * input_data.height_cm

        # Health risk score (composite of recent symptoms)
        health_risk_score = sum([
            input_data.diarrhea_recent,
            input_data.fever_recent,
            input_data.cough_recent,
            input_data.oedema,
        ])

        # Nutrition risk score
        nutrition_risk_score = self._calculate_nutrition_risk(
            input_data.breastfeeding,
            input_data.vitamin_a,
            input_data.oedema,
            wealth_encoded,
        )

        # Age group (0-5, 6-11, 12-23, 24-35, 36-47, 48-59 months)
        age_group = min(input_data.age_months // 12, 4)

        # Season (simplified: based on month)
        from datetime import datetime
        season = datetime.utcnow().month % 4

        # Food security index (inverse of wealth + health factors)
        food_security_index = (wealth_encoded + 1) / 5.0 - (health_risk_score / 10.0)

        # MUAC z-score (if available, otherwise estimate)
        muac_zscore = self._estimate_muac_zscore(input_data.age_months, sex_encoded, input_data.weight_kg, input_data.height_cm)

        # Growth velocity (would need previous measurement, placeholder)
        growth_velocity = 0.0

        # Wasting-stunting interaction
        wasting_stunting_interaction = (input_data.whz or 0) * (input_data.haz or 0)

        return PredictionFeatures(
            age_months=input_data.age_months,
            sex=sex_encoded,
            weight=input_data.weight_kg,
            height=input_data.height_cm,
            oedema=int(input_data.oedema),
            breastfeeding=int(input_data.breastfeeding),
            vitamin_a=int(input_data.vitamin_a),
            diarrhea_recent=int(input_data.diarrhea_recent),
            fever_recent=int(input_data.fever_recent),
            cough_recent=int(input_data.cough_recent),
            maternal_education=education_encoded,
            wealth_index=wealth_encoded,
            residence_type=residence_encoded,
            haz=input_data.haz or 0.0,
            whz=input_data.whz or 0.0,
            waz=input_data.waz or 0.0,
            bmi=bmi,
            weight_height_ratio=weight_height_ratio,
            age_weight_interaction=age_weight_interaction,
            age_height_interaction=age_height_interaction,
            health_risk_score=health_risk_score,
            nutrition_risk_score=nutrition_risk_score,
            muac_zscore=muac_zscore,
            head_circumference_zscore=0.0,  # Placeholder
            growth_velocity=growth_velocity,
            wasting_stunting_interaction=wasting_stunting_interaction,
            age_group=age_group,
            season=season,
            food_security_index=food_security_index,
        )

    def _calculate_nutrition_risk(self, breastfeeding: bool, vitamin_a: bool, oedema: bool, wealth: int) -> float:
        """Calculate composite nutrition risk score."""
        risk = 0.0
        if not breastfeeding:
            risk += 0.3
        if not vitamin_a:
            risk += 0.2
        if oedema:
            risk += 0.4
        if wealth <= 1:  # Poorest or poorer
            risk += 0.3
        return min(risk, 1.0)

    def _estimate_muac_zscore(self, age_months: int, sex: int, weight: float, height: float) -> float:
        """Estimate MUAC z-score from other anthropometrics."""
        # Simplified estimation - in production use actual MUAC tables
        estimated_muac = 0.2 * weight + 0.1 * height + 0.5 * age_months + 80
        # Approximate z-score
        return (estimated_muac - 135) / 15  # Simplified

    def to_model_input(self, features: PredictionFeatures) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array([
            features.age_months,
            features.sex,
            features.weight,
            features.height,
            features.oedema,
            features.breastfeeding,
            features.vitamin_a,
            features.diarrhea_recent,
            features.fever_recent,
            features.cough_recent,
            features.maternal_education,
            features.wealth_index,
            features.residence_type,
            features.haz,
            features.whz,
            features.waz,
            features.bmi,
            features.weight_height_ratio,
            features.age_weight_interaction,
            features.age_height_interaction,
            features.health_risk_score,
            features.nutrition_risk_score,
            features.muac_zscore or 0,
            features.head_circumference_zscore or 0,
            features.growth_velocity or 0,
            features.wasting_stunting_interaction or 0,
            features.age_group,
            features.season,
            features.food_security_index or 0,
        ]).reshape(1, -1)


class SeverityClassifier:
    """Classify malnutrition severity from predictions."""

    THRESHOLDS = {
        "stunting": {"normal": -1.0, "mild": -2.0, "moderate": -3.0},
        "wasting": {"normal": -1.0, "mild": -2.0, "moderate": -3.0},
        "underweight": {"normal": -1.0, "mild": -2.0, "moderate": -3.0},
    }

    def classify(self, zscore: float, indicator: str) -> Tuple[str, float, float]:
        """
        Classify severity and calculate risk percentage.

        Returns:
            (severity, risk_percent, confidence)
        """
        thresholds = self.THRESHOLDS.get(indicator, self.THRESHOLDS["stunting"])

        if zscore >= thresholds["normal"]:
            severity = "normal"
            risk_percent = max(0, (1 - zscore / thresholds["normal"]) * 50)
        elif zscore >= thresholds["mild"]:
            severity = "mild"
            risk_percent = 50 + (thresholds["normal"] - zscore) / (thresholds["normal"] - thresholds["mild"]) * 20
        elif zscore >= thresholds["moderate"]:
            severity = "moderate"
            risk_percent = 70 + (thresholds["mild"] - zscore) / (thresholds["mild"] - thresholds["moderate"]) * 15
        else:
            severity = "severe"
            risk_percent = min(100, 85 + abs(zscore + 3) * 5)

        # Confidence based on distance from thresholds
        confidence = self._calculate_confidence(zscore, thresholds)

        return severity, round(risk_percent, 1), round(confidence, 3)

    def _calculate_confidence(self, zscore: float, thresholds: Dict) -> float:
        """Calculate prediction confidence."""
        distances = [
            abs(zscore - thresholds["normal"]),
            abs(zscore - thresholds["mild"]),
            abs(zscore - thresholds["moderate"]),
        ]
        min_dist = min(distances)
        # Higher confidence when further from decision boundaries
        confidence = 0.7 + min(min_dist / 3.0, 0.25)
        return min(confidence, 0.99)


class PredictionEngine:
    """Main prediction engine coordinating all models."""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.severity_classifier = SeverityClassifier()
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load XGBoost models."""
        model_dir = Path(settings.MODEL_DIR)

        model_files = {
            "stunting": settings.XGBOOST_STUNTING_MODEL,
            "wasting": settings.XGBOOST_WASTING_MODEL,
            "underweight": settings.XGBOOST_UNDERWEIGHT_MODEL,
        }

        for model_name, filename in model_files.items():
            model_path = model_dir / filename
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded {model_name} model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name} model: {e}")
                    self.models[model_name] = None
            else:
                logger.warning(f"Model file not found: {model_path}")
                self.models[model_name] = None

    async def predict(self, input_data: PredictionInput) -> Dict:
        """
        Run complete prediction pipeline.

        Returns prediction result with severity classifications.
        """
        # Engineer features
        features = self.feature_engineer.engineer_features(input_data)
        model_input = self.feature_engineer.to_model_input(features)

        # Run predictions
        stunting_result = self._predict_indicator("stunting", model_input, features)
        wasting_result = self._predict_indicator("wasting", model_input, features)
        underweight_result = self._predict_indicator("underweight", model_input, features)

        # Determine overall risk
        overall_risk = self._determine_overall_risk(
            stunting_result["severity"],
            wasting_result["severity"],
            underweight_result["severity"],
        )

        # Generate clinical query
        clinical_query = self._generate_clinical_query(
            input_data, stunting_result, wasting_result, underweight_result
        )

        return {
            "stunting": stunting_result,
            "wasting": wasting_result,
            "underweight": underweight_result,
            "overall_risk": overall_risk,
            "overall_recommendation": self._generate_recommendation(overall_risk),
            "clinical_query": clinical_query,
            "input_features": features.model_dump(),
            "model_version": "1.0.0",
        }

    def _predict_indicator(self, indicator: str, model_input: np.ndarray, features: PredictionFeatures) -> Dict:
        """Predict a single malnutrition indicator."""
        model = self.models.get(indicator)

        if model is None:
            # Fallback: use z-score based prediction
            zscore = getattr(features, "haz" if indicator == "stunting" else "whz" if indicator == "wasting" else "waz", 0)
            severity, risk_percent, confidence = self.severity_classifier.classify(zscore, indicator)
            return {
                "probability": 0.5,
                "risk_percent": risk_percent,
                "severity": severity,
                "confidence": confidence,
            }

        # XGBoost prediction
        prediction = model.predict(model_input)[0]
        probability = model.predict_proba(model_input)[0][1] if hasattr(model, "predict_proba") else prediction

        # Convert to z-score equivalent for severity classification
        zscore = -2.0 - (probability * 2.0)  # Map probability to z-score range
        severity, risk_percent, confidence = self.severity_classifier.classify(zscore, indicator)

        return {
            "probability": round(float(probability), 3),
            "risk_percent": risk_percent,
            "severity": severity,
            "confidence": confidence,
        }

    def _determine_overall_risk(self, stunting: str, wasting: str, underweight: str) -> str:
        """Determine overall risk level from individual indicators."""
        severities = [stunting, wasting, underweight]

        if "severe" in severities:
            return "severe"
        elif severities.count("moderate") >= 2 or "moderate" in severities:
            return "moderate"
        elif "mild" in severities:
            return "mild"
        return "normal"

    def _generate_recommendation(self, overall_risk: str) -> str:
        """Generate overall recommendation based on risk level."""
        recommendations = {
            "normal": "Continue routine growth monitoring. Ensure adequate nutrition and immunization.",
            "mild": "Schedule follow-up in 1 month. Provide nutritional counseling. Monitor growth velocity.",
            "moderate": "Immediate nutritional intervention required. Refer to nutrition program. Weekly monitoring.",
            "severe": "URGENT: Immediate referral to hospital for SAM treatment. Stabilize before transfer.",
        }
        return recommendations.get(overall_risk, "Consult specialist.")

    def _generate_clinical_query(
        self, input_data: PredictionInput,
        stunting: Dict, wasting: Dict, underweight: Dict
    ) -> str:
        """Generate medical query for RAG system."""
        conditions = []
        if stunting["severity"] in ["moderate", "severe"]:
            conditions.append(f"{stunting['severity']} stunting")
        if wasting["severity"] in ["moderate", "severe"]:
            conditions.append(f"{wasting['severity']} wasting")
        if underweight["severity"] in ["moderate", "severe"]:
            conditions.append(f"{underweight['severity']} underweight")

        if not conditions:
            conditions.append("growth monitoring")

        query = f"Management protocol for {', '.join(conditions)} in {input_data.age_months}-month-old child"
        if input_data.oedema:
            query += " with oedema"

        return query

    def cleanup(self):
        """Cleanup resources."""
        self.models.clear()
        logger.info("Prediction engine cleaned up")
