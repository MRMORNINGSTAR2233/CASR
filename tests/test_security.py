"""
Tests for Security Layer

Tests for RBAC, ABAC, PolicyEngine, and AuditLogger.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from src.models.documents import DocumentMetadata, SecurityClassification
from src.models.policies import (
    AccessPolicy,
    ConditionOperator,
    PolicyAction,
    PolicyCondition,
    PolicyEffect,
)
from src.models.users import User, UserRole
from src.security.rbac import RBACEngine
from src.security.abac import ABACEngine
from src.security.policies import PolicyEngine, PolicyStore


class TestRBACEngine:
    """Tests for RBAC engine."""
    
    @pytest.fixture
    def rbac_engine(self):
        """Create RBAC engine instance."""
        return RBACEngine()
    
    def test_get_role_permissions(self, rbac_engine):
        """Test getting permissions for a role."""
        analyst_perms = rbac_engine.get_role_permissions(UserRole.ANALYST)
        
        assert "read" in analyst_perms
        assert "search" in analyst_perms
        assert "analyze" in analyst_perms
    
    def test_role_hierarchy(self, rbac_engine):
        """Test role hierarchy - higher roles inherit lower permissions."""
        admin_perms = rbac_engine.get_role_permissions(UserRole.ADMIN)
        user_perms = rbac_engine.get_role_permissions(UserRole.USER)
        
        # Admin should have all user permissions plus more
        for perm in user_perms:
            assert perm in admin_perms
    
    def test_check_permission(self, rbac_engine, sample_user):
        """Test permission checking."""
        # Analyst can read
        assert rbac_engine.check_permission(sample_user, "read") is True
        # Analyst cannot delete
        assert rbac_engine.check_permission(sample_user, "delete") is False
    
    def test_admin_permissions(self, rbac_engine, admin_user):
        """Test admin has all permissions."""
        assert rbac_engine.check_permission(admin_user, "read") is True
        assert rbac_engine.check_permission(admin_user, "delete") is True
        assert rbac_engine.check_permission(admin_user, "admin") is True
    
    def test_check_classification_access(self, rbac_engine, sample_user):
        """Test classification access based on role."""
        # Analyst can access up to CONFIDENTIAL
        assert rbac_engine.check_classification_access(
            sample_user, SecurityClassification.PUBLIC
        ) is True
        assert rbac_engine.check_classification_access(
            sample_user, SecurityClassification.CONFIDENTIAL
        ) is True
        assert rbac_engine.check_classification_access(
            sample_user, SecurityClassification.SECRET
        ) is False
    
    def test_filter_documents_by_access(self, rbac_engine, sample_user, sample_chunk):
        """Test filtering documents by access."""
        # Create chunks with different classifications
        from src.models.documents import DocumentChunk, DocumentMetadata
        
        public_chunk = DocumentChunk(
            id="public-1",
            document_id=uuid4(),
            content="Public content",
            chunk_index=0,
            start_char=0,
            end_char=14,
            metadata=DocumentMetadata(
                title="Public Doc",
                source="test",
                domain="research",
                classification=SecurityClassification.PUBLIC,
            ),
        )
        
        secret_chunk = DocumentChunk(
            id="secret-1",
            document_id=uuid4(),
            content="Secret content",
            chunk_index=0,
            start_char=0,
            end_char=14,
            metadata=DocumentMetadata(
                title="Secret Doc",
                source="test",
                domain="research",
                classification=SecurityClassification.SECRET,
            ),
        )
        
        chunks = [public_chunk, secret_chunk]
        filtered = rbac_engine.filter_documents(sample_user, chunks)
        
        assert len(filtered) == 1
        assert filtered[0].id == "public-1"


class TestABACEngine:
    """Tests for ABAC engine."""
    
    @pytest.fixture
    def abac_engine(self):
        """Create ABAC engine instance."""
        return ABACEngine()
    
    @pytest.fixture
    def department_policy(self):
        """Create a department-based policy."""
        return AccessPolicy(
            name="Department Access",
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
    
    def test_build_context(self, abac_engine, sample_user, sample_chunk):
        """Test building ABAC context."""
        context = abac_engine.build_context(
            user=sample_user,
            resource=sample_chunk.metadata,
            action=PolicyAction.READ,
        )
        
        assert context["subject"]["role"] == "analyst"
        assert context["resource"]["domain"] == "research"
        assert context["action"]["type"] == "read"
    
    def test_evaluate_conditions(self, abac_engine, sample_user, sample_chunk, department_policy):
        """Test condition evaluation."""
        abac_engine.add_policy(department_policy)
        
        context = abac_engine.build_context(
            user=sample_user,
            resource=sample_chunk.metadata,
            action=PolicyAction.READ,
        )
        
        # User is in Research department
        result = abac_engine._evaluate_conditions(department_policy.conditions, context)
        assert result is True
    
    def test_evaluate_access_allowed(self, abac_engine, sample_user, sample_chunk, department_policy):
        """Test access evaluation - allowed case."""
        abac_engine.add_policy(department_policy)
        
        allowed, reason = abac_engine.evaluate_access(
            user=sample_user,
            resource=sample_chunk.metadata,
            action=PolicyAction.READ,
        )
        
        assert allowed is True
    
    def test_deny_policy_overrides(self, abac_engine, sample_user, sample_chunk):
        """Test that DENY policies override ALLOW."""
        allow_policy = AccessPolicy(
            name="Allow All",
            effect=PolicyEffect.ALLOW,
            subjects={},
            resources={},
            actions=[PolicyAction.READ],
        )
        
        deny_policy = AccessPolicy(
            name="Deny Research",
            effect=PolicyEffect.DENY,
            subjects={},
            resources={"domain": ["research"]},
            actions=[PolicyAction.READ],
        )
        
        abac_engine.add_policy(allow_policy)
        abac_engine.add_policy(deny_policy)
        
        allowed, reason = abac_engine.evaluate_access(
            user=sample_user,
            resource=sample_chunk.metadata,
            action=PolicyAction.READ,
        )
        
        # Deny should override
        assert allowed is False
        assert "Deny Research" in reason


class TestPolicyEngine:
    """Tests for unified PolicyEngine."""
    
    @pytest.fixture
    def policy_engine(self):
        """Create policy engine instance."""
        return PolicyEngine()
    
    def test_evaluate_rbac_only(self, policy_engine, sample_user, sample_chunk):
        """Test evaluation with RBAC only (no ABAC policies)."""
        allowed, reason = policy_engine.evaluate_access(
            user=sample_user,
            resource=sample_chunk.metadata,
            action=PolicyAction.READ,
        )
        
        # Should pass RBAC check
        assert allowed is True
    
    def test_evaluate_with_abac_policy(self, policy_engine, sample_user, sample_chunk):
        """Test evaluation with both RBAC and ABAC."""
        # Add an ABAC policy
        policy = AccessPolicy(
            name="Research Access",
            effect=PolicyEffect.ALLOW,
            subjects={"role": ["analyst"]},
            resources={"domain": ["research"]},
            actions=[PolicyAction.READ],
        )
        
        policy_engine.abac.add_policy(policy)
        
        allowed, reason = policy_engine.evaluate_access(
            user=sample_user,
            resource=sample_chunk.metadata,
            action=PolicyAction.READ,
        )
        
        assert allowed is True
    
    def test_get_query_filters(self, policy_engine, sample_user):
        """Test generating query filters for a user."""
        filters = policy_engine.get_query_filters(
            user=sample_user,
            action=PolicyAction.READ,
        )
        
        # Should include classification filter
        assert "classification" in filters


class TestPolicyStore:
    """Tests for PolicyStore."""
    
    @pytest.fixture
    def policy_store(self, tmp_path):
        """Create policy store with temp directory."""
        return PolicyStore(storage_path=str(tmp_path / "policies"))
    
    def test_add_and_get_policy(self, policy_store, sample_policy):
        """Test adding and retrieving a policy."""
        policy_store.add_policy(sample_policy)
        
        retrieved = policy_store.get_policy(sample_policy.id)
        assert retrieved is not None
        assert retrieved.name == sample_policy.name
    
    def test_list_policies(self, policy_store, sample_policy):
        """Test listing policies."""
        policy_store.add_policy(sample_policy)
        
        policies = policy_store.list_policies()
        assert len(policies) >= 1
        assert any(p.name == sample_policy.name for p in policies)
    
    def test_delete_policy(self, policy_store, sample_policy):
        """Test deleting a policy."""
        policy_store.add_policy(sample_policy)
        
        success = policy_store.delete_policy(sample_policy.id)
        assert success is True
        
        retrieved = policy_store.get_policy(sample_policy.id)
        assert retrieved is None
    
    def test_find_policies_for_resource(self, policy_store):
        """Test finding policies for a resource."""
        research_policy = AccessPolicy(
            name="Research Policy",
            effect=PolicyEffect.ALLOW,
            subjects={},
            resources={"domain": ["research"]},
            actions=[PolicyAction.READ],
        )
        
        finance_policy = AccessPolicy(
            name="Finance Policy",
            effect=PolicyEffect.ALLOW,
            subjects={},
            resources={"domain": ["finance"]},
            actions=[PolicyAction.READ],
        )
        
        policy_store.add_policy(research_policy)
        policy_store.add_policy(finance_policy)
        
        matching = policy_store.find_policies_for_resource("research")
        
        assert len(matching) >= 1
        assert any(p.name == "Research Policy" for p in matching)
