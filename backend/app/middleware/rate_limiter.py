"""
Rate limiting middleware.
"""

import time
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # IP -> [(timestamp, count)]

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # Clean old entries
        if client_ip in self.requests:
            self.requests[client_ip] = [
                t for t in self.requests[client_ip]
                if current_time - t < 60
            ]

        # Check rate limit
        request_count = len(self.requests.get(client_ip, []))
        if request_count >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Record request
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        self.requests[client_ip].append(current_time)

        return await call_next(request)
