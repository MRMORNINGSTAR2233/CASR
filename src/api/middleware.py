"""
API Middleware

Request/response middleware for logging, auditing, and rate limiting.
"""

import time
from collections import defaultdict
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

import structlog


logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Log request and response details."""
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            "request_completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for security auditing."""
    
    # Paths that trigger audit logging
    AUDIT_PATHS = {
        "/api/v1/queries",
        "/api/v1/documents",
        "/api/v1/users",
    }
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Audit sensitive requests."""
        path = request.url.path
        should_audit = any(path.startswith(p) for p in self.AUDIT_PATHS)
        
        if should_audit:
            # Extract user info from headers (if available)
            user_id = request.headers.get("X-User-ID", "anonymous")
            
            logger.info(
                "audit_event",
                event_type="api_access",
                user_id=user_id,
                method=request.method,
                path=path,
                query_params=dict(request.query_params),
            )
        
        response = await call_next(request)
        
        if should_audit and response.status_code >= 400:
            logger.warning(
                "audit_event_failed",
                event_type="api_error",
                user_id=request.headers.get("X-User-ID", "anonymous"),
                method=request.method,
                path=path,
                status_code=response.status_code,
            )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    
    For production, use Redis-based rate limiting.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        dispatch: Callable = None,
    ):
        super().__init__(app, dispatch)
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Use Authorization header if present, else IP
        auth = request.headers.get("Authorization", "")
        if auth:
            return f"auth:{auth[:50]}"
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if t > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return True
        
        # Record request
        self.requests[client_id].append(now)
        return False
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Apply rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        
        if self._is_rate_limited(client_id):
            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                path=request.url.path,
            )
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                },
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - len(self.requests[client_id]))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
