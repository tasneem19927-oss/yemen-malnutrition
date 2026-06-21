"""
Application configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Yemen Malnutrition Prediction System"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Database
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/yemen_malnutrition"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None

    # JWT
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Model Paths
    MODEL_DIR: str = "./data/models"
    XGBOOST_STUNTING_MODEL: str = "xgboost_stunting_v1.pkl"
    XGBOOST_WASTING_MODEL: str = "xgboost_wasting_v1.pkl"
    XGBOOST_UNDERWEIGHT_MODEL: str = "xgboost_underweight_v1.pkl"

    # BioMobileBERT
    BIOMOBILEBERT_MODEL: str = "nlpie/bio-mobilebert"
    BIOMOBILEBERT_FINE_TUNED_PATH: Optional[str] = "./data/models/biomobilebert_ner"
    ONNX_MODEL_PATH: str = "./data/models/biomobilebert_ner.onnx"

    # RAG / FAISS
    FAISS_INDEX_PATH: str = "./knowledge_base/embeddings/faiss_index"
    EMBEDDING_MODEL: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    KNOWLEDGE_BASE_PATH: str = "./knowledge_base"

    # Offline Mode
    OFFLINE_MODE: bool = False
    LOCAL_MODEL_CACHE: str = "./data/models/cache"

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    METRICS_PORT: int = 9090

    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "./uploads"

    # Clinical
    WHO_GROWTH_STANDARDS: str = "./knowledge_base/who_standards"
    SEVERITY_THRESHOLDS: dict = {
        "normal": (-1.0, float('inf')),
        "mild": (-2.0, -1.0),
        "moderate": (-3.0, -2.0),
        "severe": (float('-inf'), -3.0),
    }

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
