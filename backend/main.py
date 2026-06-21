"""
Yemen Child Malnutrition Prediction & Clinical Decision Support System
FastAPI Backend Application
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.session import engine, SessionLocal
from app.db.base import Base
from app.api.v1.api import api_router
from app.middleware.rate_limiter import RateLimitMiddleware
from app.middleware.audit import AuditMiddleware

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting up Yemen Malnutrition Prediction System...")
    Base.metadata.create_all(bind=engine)

    # Initialize services
    from app.services.ml.model_manager import ModelManager
    from app.services.rag.vector_store import VectorStore
    from app.services.nlp.biomobilebert_ner import BioMobileBERTNER

    app.state.model_manager = ModelManager()
    app.state.vector_store = VectorStore()
    app.state.ner_service = BioMobileBERTNER.get_instance()

    logger.info("All services initialized successfully")
    yield

    # Shutdown
    logger.info("Shutting down...")
    if hasattr(app.state, 'model_manager'):
        app.state.model_manager.cleanup()


app = FastAPI(
    title="Yemen Child Malnutrition Prediction API",
    description="""
    Enterprise-grade API for predicting child malnutrition in Yemen.

    ## Features
    - **XGBoost Prediction Engine**: Stunting, Wasting, Underweight predictions
    - **BioMobileBERT NER**: Medical entity extraction (Arabic/English)
    - **Clinical RAG**: Evidence-based recommendations
    - **Offline Support**: Edge deployment with ONNX
    - **RBAC**: Role-based access control

    ## Roles
    - `admin`: Full system access
    - `doctor`: Patient care and predictions
    - `nurse`: Data entry and basic recommendations
    """,
    version="1.0.0",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    openapi_url="/api/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuditMiddleware)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "timestamp": time.time(),
        }
    )


# Health check
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "timestamp": time.time(),
    }


# API routes
app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
    )
