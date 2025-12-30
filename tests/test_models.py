"""
Tests for Data Models

Tests for document, user, query, and policy models.
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4

from src.models.documents import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    SecurityClassification,
)
from src.models.users import User, UserRole, UserSession
from src.models.queries import (
    AnalyzedQuery,
    DataTypeRequirement,
    QueryContext,
    QueryIntent,
    RetrievalResult,
    SearchQuery,
)
from src.models.policies import (
    AccessPolicy,
    ConditionOperator,
    PolicyAction,
    PolicyCondition,
    PolicyEffect,
    PolicySet,
)


class TestSecurityClassification:
    """Tests for SecurityClassification enum."""
    
    def test_classification_hierarchy(self):
        """Test that classifications have correct hierarchy."""
        classifications = list(SecurityClassification)
        
        assert SecurityClassification.PUBLIC == classifications[0]
        assert SecurityClassification.TOP_SECRET == classifications[-1]
    
    def test_classification_comparison(self):
        """Test classification comparison logic."""
        public = SecurityClassification.PUBLIC
        secret = SecurityClassification.SECRET
        
        # Using enum index for comparison
        assert list(SecurityClassification).index(public) < \
               list(SecurityClassification).index(secret)


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""
    
    def test_create_metadata(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            title="Test Doc",
            source="test",
            domain="research",
        )
        
        assert metadata.title == "Test Doc"
        assert metadata.source == "test"
        assert metadata.domain == "research"
        assert metadata.classification == SecurityClassification.INTERNAL
        assert metadata.tags == []
    
    def test_metadata_with_all_fields(self):
        """Test metadata with all fields specified."""
        metadata = DocumentMetadata(
            title="Full Test",
            source="api",
            domain="finance",
            classification=SecurityClassification.CONFIDENTIAL,
            tags=["important", "quarterly"],
            author="John Doe",
            version="2.0",
            custom={"priority": "high"},
        )
        
        assert metadata.classification == SecurityClassification.CONFIDENTIAL
        assert "important" in metadata.tags
        assert metadata.author == "John Doe"
        assert metadata.custom["priority"] == "high"
    
    def test_metadata_security_context(self):
        """Test to_security_context method."""
        metadata = DocumentMetadata(
            title="Test",
            source="test",
            domain="research",
            classification=SecurityClassification.SECRET,
            owner_id="user-123",
            allowed_roles=["analyst", "admin"],
        )
        
        context = metadata.to_security_context()
        
        assert context["classification"] == "secret"
        assert context["domain"] == "research"
        assert "analyst" in context["allowed_roles"]


class TestDocument:
    """Tests for Document model."""
    
    def test_create_document(self, sample_document):
        """Test creating a document."""
        assert sample_document.id is not None
        assert sample_document.content is not None
        assert sample_document.metadata.title == "Test Document"
    
    def test_document_chunk_count(self, sample_document):
        """Test document chunk_count property."""
        # Before any chunks are set
        assert sample_document.chunk_count == 0
    
    def test_document_content_hash(self, sample_document):
        """Test that content hash is generated."""
        assert sample_document.content_hash is not None
        assert len(sample_document.content_hash) == 64  # SHA-256 hex


class TestDocumentChunk:
    """Tests for DocumentChunk model."""
    
    def test_create_chunk(self, sample_chunk):
        """Test creating a document chunk."""
        assert sample_chunk.id == "test-chunk-1"
        assert sample_chunk.chunk_index == 0
        assert sample_chunk.content is not None
    
    def test_chunk_content_for_embedding(self, sample_chunk):
        """Test content_for_embedding without context."""
        content = sample_chunk.content_for_embedding
        assert content == sample_chunk.content
    
    def test_chunk_content_for_embedding_with_context(self, sample_chunk):
        """Test content_for_embedding with context."""
        sample_chunk.context = "This chunk is about testing."
        content = sample_chunk.content_for_embedding
        
        assert "This chunk is about testing" in content
        assert sample_chunk.content in content


class TestUser:
    """Tests for User model."""
    
    def test_create_user(self, sample_user):
        """Test creating a user."""
        assert sample_user.username == "test_user"
        assert sample_user.role == UserRole.ANALYST
        assert sample_user.is_active
    
    def test_user_effective_clearance(self, sample_user):
        """Test effective clearance based on role."""
        clearance = sample_user.effective_clearance
        
        # Analyst should have CONFIDENTIAL clearance
        assert clearance == SecurityClassification.CONFIDENTIAL
    
    def test_admin_effective_clearance(self, admin_user):
        """Test admin has top secret clearance."""
        clearance = admin_user.effective_clearance
        
        assert clearance == SecurityClassification.TOP_SECRET
    
    def test_user_can_access_classification(self, sample_user):
        """Test can_access_classification method."""
        assert sample_user.can_access_classification(SecurityClassification.PUBLIC)
        assert sample_user.can_access_classification(SecurityClassification.INTERNAL)
        assert sample_user.can_access_classification(SecurityClassification.CONFIDENTIAL)
        assert not sample_user.can_access_classification(SecurityClassification.SECRET)
        assert not sample_user.can_access_classification(SecurityClassification.TOP_SECRET)


class TestSearchQuery:
    """Tests for SearchQuery model."""
    
    def test_create_query(self, sample_user):
        """Test creating a search query."""
        context = QueryContext(
            user_id=sample_user.id,
            user_role=sample_user.role.value,
            user_clearance=sample_user.effective_clearance,
        )
        
        query = SearchQuery(
            text="What is the system architecture?",
            context=context,
        )
        
        assert query.text == "What is the system architecture?"
        assert query.id is not None
        assert query.max_results == 10
    
    def test_query_with_filters(self, sample_user):
        """Test query with domain and classification filters."""
        context = QueryContext(
            user_id=sample_user.id,
            user_role=sample_user.role.value,
            user_clearance=sample_user.effective_clearance,
        )
        
        query = SearchQuery(
            text="Financial reports",
            context=context,
            domain_filter="finance",
            classification_filter=SecurityClassification.INTERNAL,
        )
        
        assert query.domain_filter == "finance"
        assert query.classification_filter == SecurityClassification.INTERNAL


class TestAccessPolicy:
    """Tests for AccessPolicy model."""
    
    def test_create_policy(self, sample_policy):
        """Test creating an access policy."""
        assert sample_policy.name == "Test Policy"
        assert sample_policy.effect == PolicyEffect.ALLOW
        assert PolicyAction.READ in sample_policy.actions
    
    def test_policy_condition(self, sample_policy):
        """Test policy condition."""
        condition = sample_policy.conditions[0]
        
        assert condition.attribute == "department"
        assert condition.operator == ConditionOperator.EQUALS
        assert condition.value == "Research"
    
    def test_policy_evaluation(self, sample_policy):
        """Test policy condition evaluation."""
        condition = sample_policy.conditions[0]
        
        assert condition.evaluate("Research") is True
        assert condition.evaluate("IT") is False
    
    def test_policy_set(self):
        """Test PolicySet container."""
        policy1 = AccessPolicy(
            name="Policy 1",
            effect=PolicyEffect.ALLOW,
            subjects={},
            resources={},
            actions=[PolicyAction.READ],
        )
        
        policy2 = AccessPolicy(
            name="Policy 2",
            effect=PolicyEffect.DENY,
            subjects={},
            resources={},
            actions=[PolicyAction.WRITE],
        )
        
        policy_set = PolicySet(
            name="Test Set",
            policies=[policy1, policy2],
        )
        
        assert len(policy_set.policies) == 2
        assert policy_set.policy_count == 2


class TestQueryIntent:
    """Tests for QueryIntent enum."""
    
    def test_all_intents_defined(self):
        """Test all expected intents are defined."""
        expected = [
            "factual",
            "analytical",
            "procedural",
            "definitional",
            "exploratory",
            "troubleshooting",
            "summarization",
            "unknown",
        ]
        
        for intent in expected:
            assert QueryIntent(intent) is not None
