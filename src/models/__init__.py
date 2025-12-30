"""
CASR Models Package

Pydantic models for the CASR system.
"""

from .documents import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    SecurityClassification,
)
from .policies import (
    AccessPolicy,
    PolicyAction,
    PolicyCondition,
    PolicyEffect,
)
from .queries import (
    AnalyzedQuery,
    QueryContext,
    QueryIntent,
    RetrievalResult,
    SearchQuery,
)
from .users import (
    User,
    UserRole,
    UserSession,
)

__all__ = [
    # Documents
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "SecurityClassification",
    # Queries
    "SearchQuery",
    "QueryContext",
    "QueryIntent",
    "AnalyzedQuery",
    "RetrievalResult",
    # Users
    "User",
    "UserRole",
    "UserSession",
    # Policies
    "AccessPolicy",
    "PolicyAction",
    "PolicyCondition",
    "PolicyEffect",
]
