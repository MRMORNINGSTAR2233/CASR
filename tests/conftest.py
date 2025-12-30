"""
Test Configuration and Fixtures

Shared fixtures for CASR tests.
"""

import os
from datetime import datetime
from uuid import uuid4

import pytest

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "true"
os.environ["JWT_SECRET"] = "test-secret-key-for-testing-only"


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    from src.models.users import User, UserRole
    
    return User(
        id=uuid4(),
        username="test_user",
        email="test@example.com",
        role=UserRole.ANALYST,
        department="Research",
        allowed_domains=["research", "public"],
        is_active=True,
    )


@pytest.fixture
def admin_user():
    """Create an admin user for testing."""
    from src.models.users import User, UserRole
    
    return User(
        id=uuid4(),
        username="admin_user",
        email="admin@example.com",
        role=UserRole.ADMIN,
        department="IT",
        allowed_domains=[],
        is_active=True,
    )


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    from src.models.documents import Document, DocumentMetadata, SecurityClassification
    
    return Document(
        id=uuid4(),
        content="""
        # Test Document
        
        This is a test document for the CASR system.
        It contains multiple paragraphs of text.
        
        ## Section 1
        
        The first section discusses the architecture of the system.
        It includes various components like indexing, retrieval, and security.
        
        ## Section 2
        
        The second section covers the implementation details.
        It explains how the different modules interact with each other.
        
        ## Conclusion
        
        This document serves as a test fixture for unit tests.
        """,
        metadata=DocumentMetadata(
            title="Test Document",
            source="test",
            domain="research",
            classification=SecurityClassification.INTERNAL,
            tags=["test", "fixture"],
            owner_id="test-owner",
        ),
    )


@pytest.fixture
def sample_chunk():
    """Create a sample document chunk for testing."""
    from src.models.documents import DocumentChunk, DocumentMetadata, SecurityClassification
    
    return DocumentChunk(
        id="test-chunk-1",
        document_id=uuid4(),
        content="This is a test chunk containing sample content for testing.",
        chunk_index=0,
        start_char=0,
        end_char=59,
        metadata=DocumentMetadata(
            title="Test Document",
            source="test",
            domain="research",
            classification=SecurityClassification.INTERNAL,
        ),
        token_count=12,
    )


@pytest.fixture
def sample_policy():
    """Create a sample access policy for testing."""
    from src.models.policies import (
        AccessPolicy,
        ConditionOperator,
        PolicyAction,
        PolicyCondition,
        PolicyEffect,
    )
    
    return AccessPolicy(
        id=uuid4(),
        name="Test Policy",
        description="A test policy for unit testing",
        effect=PolicyEffect.ALLOW,
        subjects={"role": ["analyst", "admin"]},
        resources={"domain": ["research"]},
        actions=[PolicyAction.READ],
        conditions=[
            PolicyCondition(
                attribute="department",
                operator=ConditionOperator.EQUALS,
                value="Research",
            ),
        ],
    )


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    import numpy as np
    
    def _create_embedding(dim: int = 1536):
        return np.random.randn(dim).tolist()
    
    return _create_embedding
