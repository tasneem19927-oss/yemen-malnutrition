"""
API router aggregator.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, patients, predictions, ner, knowledge, analytics

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(patients.router)
api_router.include_router(predictions.router)
api_router.include_router(ner.router)
api_router.include_router(knowledge.router)
api_router.include_router(analytics.router)
