"""
FastAPI Application Factory

Creates and configures the CASR API application.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from config import get_settings

from .middleware import (
    AuditMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from .routes import documents, health, queries, users


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan handler.
    
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    app.state.settings = settings
    
    yield
    
    # Shutdown (cleanup)
    # Close any open connections, etc.


def create_app(
    title: str = "CASR API",
    description: str = "Context-Aware Secure Retrieval API",
    version: str = "0.1.0",
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        title: API title
        description: API description
        version: API version
        
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(queries.router, prefix="/api/v1/queries", tags=["Queries"])
    
    return app


# Create default app instance
app = create_app()
