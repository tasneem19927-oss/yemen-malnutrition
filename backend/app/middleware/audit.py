"""
Audit logging middleware.
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.db.session import get_db_context
from app.models.knowledge import AuditLog

logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """Log all API requests for audit trail."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Log audit (skip health checks and docs)
        path = request.url.path
        if not any(skip in path for skip in ["/health", "/docs", "/openapi", "/redoc"]):
            try:
                with get_db_context() as db:
                    audit = AuditLog(
                        action=request.method,
                        entity_type=self._get_entity_type(path),
                        ip_address=request.client.host if request.client else None,
                        user_agent=request.headers.get("user-agent"),
                    )
                    db.add(audit)
                    db.commit()
            except Exception as e:
                logger.error(f"Audit logging failed: {e}")

        return response

    def _get_entity_type(self, path: str) -> str:
        """Determine entity type from path."""
        if "/patients" in path:
            return "patient"
        elif "/predictions" in path:
            return "prediction"
        elif "/users" in path:
            return "user"
        elif "/knowledge" in path:
            return "knowledge"
        elif "/auth" in path:
            return "auth"
        return "other"
