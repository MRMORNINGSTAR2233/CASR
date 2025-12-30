"""
Health Check Routes

Endpoints for service health monitoring.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from config import get_settings
from src.api.dependencies import get_vector_store
from src.storage.base import VectorStore

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    environment: str


class DetailedHealthResponse(HealthResponse):
    """Detailed health check response."""
    vector_store: dict
    dependencies: dict


@router.get("", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """Basic health check."""
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        environment=settings.environment,
    )


@router.get("/live")
async def liveness() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness(
    vector_store: VectorStore = Depends(get_vector_store),
) -> dict:
    """Kubernetes readiness probe."""
    # Check vector store connectivity
    try:
        count = vector_store.count()
        return {
            "status": "ready",
            "vector_store": "connected",
            "document_count": count,
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "vector_store": "disconnected",
            "error": str(e),
        }


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health(
    vector_store: VectorStore = Depends(get_vector_store),
) -> DetailedHealthResponse:
    """Detailed health check with dependency status."""
    settings = get_settings()
    
    # Check vector store
    vector_store_status = {"status": "unknown", "count": 0}
    try:
        count = vector_store.count()
        vector_store_status = {
            "status": "healthy",
            "type": settings.vector_store,
            "count": count,
        }
    except Exception as e:
        vector_store_status = {
            "status": "unhealthy",
            "type": settings.vector_store,
            "error": str(e),
        }
    
    # Check dependencies
    dependencies = {
        "openai": _check_api_key(settings.openai_api_key),
        "anthropic": _check_api_key(settings.anthropic_api_key),
        "cohere": _check_api_key(settings.cohere_api_key),
    }
    
    return DetailedHealthResponse(
        status="healthy" if vector_store_status["status"] == "healthy" else "degraded",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        environment=settings.environment,
        vector_store=vector_store_status,
        dependencies=dependencies,
    )


def _check_api_key(key: Optional[str]) -> dict:
    """Check if API key is configured."""
    if key and len(key) > 10:
        return {"status": "configured"}
    return {"status": "not_configured"}
