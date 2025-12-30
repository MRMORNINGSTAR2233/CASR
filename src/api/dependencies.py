"""
API Dependencies

Dependency injection for FastAPI routes.
"""

from functools import lru_cache
from typing import Annotated, Optional
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import get_settings
from src.indexing import Embedder, IndexManager
from src.models.users import User, UserRole
from src.retrieval import Retriever, SecureRetriever
from src.security.audit import AuditLogger
from src.security.policies import PolicyEngine
from src.storage.base import VectorStore
from src.storage.chroma_store import ChromaVectorStore

# Security scheme
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_settings_cached():
    """Get cached settings."""
    return get_settings()


@lru_cache()
def get_audit_logger() -> AuditLogger:
    """Get audit logger instance."""
    return AuditLogger()


@lru_cache()
def get_policy_engine() -> PolicyEngine:
    """Get policy engine instance."""
    return PolicyEngine()


@lru_cache()
def get_embedder() -> Embedder:
    """Get embedder instance."""
    settings = get_settings()
    return Embedder(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
    )


def get_vector_store() -> VectorStore:
    """Get vector store instance."""
    settings = get_settings()
    
    match settings.vector_store:
        case "chroma":
            return ChromaVectorStore(
                collection_name=settings.chroma_collection,
                persist_directory=settings.chroma_persist_dir,
            )
        case "pinecone":
            from src.storage.pinecone_store import PineconeVectorStore
            return PineconeVectorStore(
                index_name=settings.pinecone_index,
            )
        case "weaviate":
            from src.storage.weaviate_store import WeaviateVectorStore
            return WeaviateVectorStore(
                class_name=settings.weaviate_class,
                url=settings.weaviate_url,
            )
        case _:
            return ChromaVectorStore()


def get_index_manager(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> IndexManager:
    """Get index manager instance."""
    return IndexManager(
        vector_store=vector_store,
        embedder=embedder,
    )


def get_retriever(
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    policy_engine: Annotated[PolicyEngine, Depends(get_policy_engine)],
    audit_logger: Annotated[AuditLogger, Depends(get_audit_logger)],
) -> SecureRetriever:
    """Get secure retriever instance."""
    return SecureRetriever(
        vector_store=vector_store,
        embedder=embedder,
        policy_engine=policy_engine,
        audit_logger=audit_logger,
    )


def decode_token(token: str) -> dict:
    """
    Decode and validate JWT token.
    
    In production, use proper JWT validation with secret key.
    """
    import jwt
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    x_user_id: Optional[str] = Header(None),
    x_user_role: Optional[str] = Header(None),
) -> User:
    """
    Get the current authenticated user.
    
    Supports multiple authentication methods:
    1. Bearer token (JWT)
    2. Header-based (for development/testing)
    """
    settings = get_settings()
    
    # Try bearer token first
    if credentials and credentials.credentials:
        payload = decode_token(credentials.credentials)
        
        try:
            role = UserRole(payload.get("role", "guest"))
        except ValueError:
            role = UserRole.GUEST
        
        return User(
            id=UUID(payload["sub"]),
            username=payload.get("username", "unknown"),
            email=payload.get("email", ""),
            role=role,
            department=payload.get("department"),
            allowed_domains=payload.get("allowed_domains", []),
            is_active=True,
        )
    
    # Fallback to header-based auth (development only)
    if settings.debug and x_user_id:
        try:
            role = UserRole(x_user_role) if x_user_role else UserRole.GUEST
        except ValueError:
            role = UserRole.GUEST
        
        return User(
            id=UUID(x_user_id),
            username="dev-user",
            email="dev@example.com",
            role=role,
            is_active=True,
        )
    
    # No authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> Optional[User]:
    """Get user if authenticated, None otherwise."""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_role(min_role: UserRole):
    """
    Dependency factory for role-based access control.
    
    Args:
        min_role: Minimum required role
        
    Returns:
        Dependency that validates user role
    """
    role_hierarchy = {
        UserRole.GUEST: 0,
        UserRole.USER: 1,
        UserRole.ANALYST: 2,
        UserRole.MANAGER: 3,
        UserRole.EXECUTIVE: 4,
        UserRole.ADMIN: 5,
        UserRole.SYSTEM: 6,
    }
    
    async def role_checker(
        user: Annotated[User, Depends(get_current_user)]
    ) -> User:
        if role_hierarchy.get(user.role, 0) < role_hierarchy.get(min_role, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {min_role.value}",
            )
        return user
    
    return role_checker
