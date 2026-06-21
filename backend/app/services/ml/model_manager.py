"""
Model Manager for loading and versioning models.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage ML model lifecycle."""

    def __init__(self):
        self.models = {}
        self._load_all()

    def _load_all(self):
        """Load all available models."""
        from app.services.ml.prediction_engine import PredictionEngine
        self.prediction_engine = PredictionEngine()
        logger.info("Model manager initialized")

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "prediction_engine": {
                "loaded": bool(self.prediction_engine.models),
                "models": list(self.prediction_engine.models.keys()),
            }
        }

    def cleanup(self):
        """Cleanup model resources."""
        if hasattr(self, 'prediction_engine'):
            self.prediction_engine.cleanup()
        logger.info("Model manager cleaned up")
