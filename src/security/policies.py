"""
Policy Engine

Unified policy evaluation combining RBAC and ABAC.
"""

import json
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from src.models.documents import DocumentMetadata
from src.models.policies import AccessPolicy, PolicyAction, PolicyEffect
from src.models.users import User

from .abac import ABACEngine
from .rbac import RBACEngine


class PolicyStore:
    """
    Persistent storage for access policies.
    
    Provides CRUD operations for policies with optional file-based persistence.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the policy store.
        
        Args:
            storage_path: Optional path for file-based persistence
        """
        self._policies: dict[str, AccessPolicy] = {}
        self._storage_path = storage_path
        
        if storage_path and storage_path.exists():
            self._load_from_file()
    
    def _load_from_file(self) -> None:
        """Load policies from storage file."""
        if not self._storage_path or not self._storage_path.exists():
            return
        
        try:
            with open(self._storage_path) as f:
                data = json.load(f)
                for policy_data in data.get("policies", []):
                    policy = AccessPolicy.model_validate(policy_data)
                    self._policies[str(policy.id)] = policy
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load policies: {e}")
    
    def _save_to_file(self) -> None:
        """Save policies to storage file."""
        if not self._storage_path:
            return
        
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "policies": [p.model_dump(mode="json") for p in self._policies.values()]
        }
        
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def add(self, policy: AccessPolicy) -> AccessPolicy:
        """Add a policy to the store."""
        self._policies[str(policy.id)] = policy
        self._save_to_file()
        return policy
    
    def get(self, policy_id: str | UUID) -> Optional[AccessPolicy]:
        """Get a policy by ID."""
        return self._policies.get(str(policy_id))
    
    def update(self, policy: AccessPolicy) -> Optional[AccessPolicy]:
        """Update an existing policy."""
        policy_id = str(policy.id)
        if policy_id not in self._policies:
            return None
        self._policies[policy_id] = policy
        self._save_to_file()
        return policy
    
    def delete(self, policy_id: str | UUID) -> bool:
        """Delete a policy by ID."""
        policy_id_str = str(policy_id)
        if policy_id_str in self._policies:
            del self._policies[policy_id_str]
            self._save_to_file()
            return True
        return False
    
    def list_all(self) -> list[AccessPolicy]:
        """List all policies."""
        return list(self._policies.values())
    
    def find_by_name(self, name: str) -> list[AccessPolicy]:
        """Find policies by name (partial match)."""
        return [p for p in self._policies.values() if name.lower() in p.name.lower()]
    
    def find_by_tag(self, tag: str) -> list[AccessPolicy]:
        """Find policies by tag."""
        return [p for p in self._policies.values() if tag in p.tags]
    
    def get_active_policies(self) -> list[AccessPolicy]:
        """Get all active policies."""
        return [p for p in self._policies.values() if p.is_active]


class PolicyEngine:
    """
    Unified Policy Engine combining RBAC and ABAC.
    
    Provides a single interface for access control decisions,
    combining role-based and attribute-based policies.
    """
    
    def __init__(
        self,
        policy_store: Optional[PolicyStore] = None,
        rbac_enabled: bool = True,
        abac_enabled: bool = True
    ):
        """
        Initialize the policy engine.
        
        Args:
            policy_store: Optional policy store for persistence
            rbac_enabled: Whether to enable RBAC evaluation
            abac_enabled: Whether to enable ABAC evaluation
        """
        self.rbac = RBACEngine() if rbac_enabled else None
        self.abac = ABACEngine() if abac_enabled else None
        self.policy_store = policy_store or PolicyStore()
        
        # Load policies into ABAC engine
        if self.abac:
            for policy in self.policy_store.get_active_policies():
                self.abac.add_policy(policy)
    
    def add_policy(self, policy: AccessPolicy) -> AccessPolicy:
        """Add a policy and register it with ABAC engine."""
        stored_policy = self.policy_store.add(policy)
        if self.abac:
            self.abac.add_policy(stored_policy)
        return stored_policy
    
    def remove_policy(self, policy_id: str | UUID) -> bool:
        """Remove a policy."""
        success = self.policy_store.delete(policy_id)
        if success and self.abac:
            self.abac.remove_policy(str(policy_id))
        return success
    
    def evaluate_access(
        self,
        user: User,
        resource: Optional[DocumentMetadata] = None,
        action: PolicyAction = PolicyAction.READ,
        environment: Optional[dict[str, Any]] = None
    ) -> tuple[bool, str]:
        """
        Evaluate whether access should be granted.
        
        Combines RBAC and ABAC evaluations:
        1. RBAC: Checks role-based permissions and classification access
        2. ABAC: Checks attribute-based policies
        
        Both must allow for access to be granted.
        
        Args:
            user: The requesting user
            resource: The resource being accessed
            action: The action being performed
            environment: Environmental attributes
            
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # RBAC check first (faster, coarse-grained)
        if self.rbac and resource:
            if not self.rbac.can_access_document(user, resource):
                return False, "RBAC: Insufficient role/clearance for resource"
        
        # ABAC check (fine-grained)
        if self.abac:
            effect, policy = self.abac.evaluate(user, resource, action, environment)
            if effect == PolicyEffect.DENY:
                reason = f"ABAC: Denied by policy '{policy.name}'" if policy else "ABAC: Default deny"
                return False, reason
        
        return True, "Access granted"
    
    def can_access(
        self,
        user: User,
        resource: DocumentMetadata,
        action: PolicyAction = PolicyAction.READ
    ) -> bool:
        """
        Simple boolean check for access.
        
        Args:
            user: The requesting user
            resource: The resource being accessed
            action: The action being performed
            
        Returns:
            True if access is allowed
        """
        allowed, _ = self.evaluate_access(user, resource, action)
        return allowed
    
    def filter_accessible(
        self,
        user: User,
        resources: list[DocumentMetadata],
        action: PolicyAction = PolicyAction.READ
    ) -> list[DocumentMetadata]:
        """
        Filter resources to only those accessible by user.
        
        Args:
            user: The requesting user
            resources: List of resources to filter
            action: The action being performed
            
        Returns:
            List of accessible resources
        """
        accessible = []
        for resource in resources:
            if self.can_access(user, resource, action):
                accessible.append(resource)
        return accessible
    
    def get_query_filters(
        self,
        user: User,
        action: PolicyAction = PolicyAction.READ
    ) -> dict[str, Any]:
        """
        Generate filters for vector store queries.
        
        Combines RBAC and ABAC filters for efficient query-time filtering.
        
        Args:
            user: The requesting user
            action: The action being performed
            
        Returns:
            Dictionary of filters for vector store queries
        """
        filters: dict[str, Any] = {}
        
        # RBAC filters
        if self.rbac:
            rbac_filters = self.rbac.get_access_filter(user)
            filters.update(rbac_filters)
        
        # ABAC filters
        if self.abac:
            abac_filters = self.abac.get_resource_filters(user, action)
            # Merge with RBAC filters (ABAC may add additional restrictions)
            for key, value in abac_filters.items():
                if key in filters:
                    # Merge filter conditions
                    if isinstance(filters[key], dict) and isinstance(value, dict):
                        filters[key].update(value)
                else:
                    filters[key] = value
        
        return filters
    
    def get_user_permissions(self, user: User) -> dict[str, Any]:
        """
        Get a summary of user's permissions.
        
        Args:
            user: The user to check
            
        Returns:
            Dictionary with permission summary
        """
        result = {
            "user_id": str(user.id),
            "role": user.role.value,
            "clearance": user.effective_clearance.value,
            "permissions": [],
            "accessible_classifications": [],
            "domain_restrictions": user.allowed_domains or "unrestricted",
        }
        
        if self.rbac:
            result["permissions"] = list(self.rbac.get_effective_permissions(user))
            result["accessible_classifications"] = [
                c.value for c in self.rbac.get_accessible_classifications(user)
            ]
        
        return result
