"""
Attribute-Based Access Control (ABAC) Engine

Implements attribute-based access control for fine-grained authorization.
"""

from datetime import datetime
from typing import Any, Optional

from src.models.documents import DocumentMetadata
from src.models.policies import (
    AccessPolicy,
    ConditionOperator,
    PolicyAction,
    PolicyCondition,
    PolicyEffect,
)
from src.models.users import User


class ABACEngine:
    """
    Attribute-Based Access Control Engine.
    
    Evaluates access based on attributes of:
    - Subject (user): role, department, clearance, custom attributes
    - Resource (document): classification, domain, tags, custom metadata
    - Action: read, write, query, etc.
    - Environment: time, location, device, etc.
    """
    
    def __init__(self):
        """Initialize the ABAC engine."""
        self._policies: list[AccessPolicy] = []
    
    def add_policy(self, policy: AccessPolicy) -> None:
        """Add a policy to the engine."""
        self._policies.append(policy)
        # Keep sorted by priority
        self._policies.sort(key=lambda p: p.priority, reverse=True)
    
    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy by ID."""
        initial_count = len(self._policies)
        self._policies = [p for p in self._policies if str(p.id) != policy_id]
        return len(self._policies) < initial_count
    
    def get_policies(self) -> list[AccessPolicy]:
        """Get all registered policies."""
        return self._policies.copy()
    
    def build_context(
        self,
        user: User,
        resource: Optional[DocumentMetadata] = None,
        action: PolicyAction = PolicyAction.READ,
        environment: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Build evaluation context from user, resource, action, and environment.
        
        Args:
            user: The requesting user
            resource: The resource being accessed (optional)
            action: The action being performed
            environment: Environmental attributes (time, location, etc.)
            
        Returns:
            Dictionary containing all attributes for policy evaluation
        """
        context = {
            "subject": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "all_roles": [r.value for r in user.all_roles],
                "department": user.department,
                "team": user.team,
                "location": user.location,
                "clearance": user.effective_clearance.value,
                "clearance_level": user.effective_clearance.level,
                "allowed_domains": user.allowed_domains,
                "is_active": user.is_active,
                "is_verified": user.is_verified,
                **user.attributes,
            },
            "action": {
                "type": action.value,
            },
            "environment": {
                "timestamp": datetime.utcnow().isoformat(),
                "day_of_week": datetime.utcnow().strftime("%A"),
                "hour": datetime.utcnow().hour,
                **(environment or {}),
            },
        }
        
        if resource:
            context["resource"] = {
                "title": resource.title,
                "source": resource.source,
                "classification": resource.classification.value,
                "classification_level": resource.classification.level,
                "domain": resource.domain,
                "department": resource.department,
                "tags": resource.tags,
                "data_type": resource.data_type,
                "allowed_roles": resource.allowed_roles,
                "denied_roles": resource.denied_roles,
                "allowed_users": resource.allowed_users,
                **resource.custom,
            }
        
        return context
    
    def evaluate(
        self,
        user: User,
        resource: Optional[DocumentMetadata] = None,
        action: PolicyAction = PolicyAction.READ,
        environment: Optional[dict[str, Any]] = None
    ) -> tuple[PolicyEffect, Optional[AccessPolicy]]:
        """
        Evaluate all policies for a given request.
        
        Uses deny-override: if any policy denies, access is denied.
        
        Args:
            user: The requesting user
            resource: The resource being accessed
            action: The action being performed
            environment: Environmental attributes
            
        Returns:
            Tuple of (effect, matching_policy) where matching_policy is the
            decisive policy (None if using default)
        """
        context = self.build_context(user, resource, action, environment)
        
        allow_policies: list[AccessPolicy] = []
        
        for policy in self._policies:
            if not policy.is_active:
                continue
            
            # Check if policy applies to this action
            if not policy.matches_action(action):
                continue
            
            # Check if policy applies to this subject
            if not policy.matches_subject(user.role.value, str(user.id)):
                continue
            
            # Check if policy applies to this resource
            if resource:
                if not policy.matches_resource(
                    resource.domain,
                    resource.classification.value
                ):
                    continue
            
            # Evaluate conditions
            if not policy.evaluate_conditions(context):
                continue
            
            # Policy matches - check effect
            if policy.effect == PolicyEffect.DENY:
                # Deny-override: immediate denial
                return PolicyEffect.DENY, policy
            else:
                allow_policies.append(policy)
        
        # If we have any matching allow policies, allow
        if allow_policies:
            return PolicyEffect.ALLOW, allow_policies[0]
        
        # Default deny
        return PolicyEffect.DENY, None
    
    def filter_resources(
        self,
        user: User,
        resources: list[DocumentMetadata],
        action: PolicyAction = PolicyAction.READ,
        environment: Optional[dict[str, Any]] = None
    ) -> list[DocumentMetadata]:
        """
        Filter resources based on ABAC policies.
        
        Args:
            user: The requesting user
            resources: List of resources to filter
            action: The action being performed
            environment: Environmental attributes
            
        Returns:
            List of resources the user can access
        """
        accessible = []
        for resource in resources:
            effect, _ = self.evaluate(user, resource, action, environment)
            if effect == PolicyEffect.ALLOW:
                accessible.append(resource)
        return accessible
    
    def get_resource_filters(
        self,
        user: User,
        action: PolicyAction = PolicyAction.READ
    ) -> dict[str, Any]:
        """
        Generate filters for vector store queries based on ABAC policies.
        
        This is an optimization to push filtering to the vector store level
        rather than post-filtering results.
        
        Args:
            user: The requesting user
            action: The action being performed
            
        Returns:
            Dictionary of filters to apply
        """
        filters: dict[str, Any] = {}
        
        # Start with user's clearance level
        filters["classification_level"] = {"$lte": user.effective_clearance.level}
        
        # Add domain restrictions
        if user.allowed_domains:
            filters["domain"] = {"$in": user.allowed_domains}
        
        # Analyze policies for additional static filters
        for policy in self._policies:
            if not policy.is_active:
                continue
            
            if policy.effect == PolicyEffect.DENY:
                # Add exclusion filters for deny policies
                if policy.resource_domains:
                    # If user matches this deny policy's subject criteria
                    if policy.matches_subject(user.role.value, str(user.id)):
                        # Exclude these domains
                        if "domain" in filters and "$nin" not in filters["domain"]:
                            filters["domain"]["$nin"] = policy.resource_domains
                        elif "domain" not in filters:
                            filters["domain"] = {"$nin": policy.resource_domains}
        
        return filters


# Predefined policy templates
def create_department_isolation_policy(department: str) -> AccessPolicy:
    """
    Create a policy that restricts access to department-specific documents.
    
    Users can only access confidential documents from their own department.
    """
    return AccessPolicy(
        name=f"Department Isolation - {department}",
        description=f"Restrict confidential {department} documents to {department} users",
        effect=PolicyEffect.ALLOW,
        actions=[PolicyAction.READ, PolicyAction.QUERY],
        resource_domains=[department],
        resource_classifications=["confidential"],
        conditions=[
            PolicyCondition(
                attribute="subject.department",
                operator=ConditionOperator.EQUALS,
                value=department,
            )
        ],
    )


def create_time_based_policy(
    start_hour: int,
    end_hour: int,
    allowed_roles: list[str]
) -> AccessPolicy:
    """
    Create a policy that restricts access based on time of day.
    
    Only specified roles can access during specified hours.
    """
    return AccessPolicy(
        name=f"Time-Based Access ({start_hour}:00-{end_hour}:00)",
        description=f"Restrict access to business hours for non-privileged users",
        effect=PolicyEffect.DENY,
        actions=[PolicyAction.READ, PolicyAction.QUERY],
        conditions=[
            PolicyCondition(
                attribute="environment.hour",
                operator=ConditionOperator.LESS_THAN,
                value=start_hour,
            ),
            PolicyCondition(
                attribute="subject.role",
                operator=ConditionOperator.NOT_IN,
                value=allowed_roles,
            ),
        ],
        condition_logic="AND",
    )


def create_verified_user_policy() -> AccessPolicy:
    """
    Create a policy that requires email verification for access.
    """
    return AccessPolicy(
        name="Require Email Verification",
        description="Deny access to unverified users for non-public resources",
        effect=PolicyEffect.DENY,
        actions=[PolicyAction.READ, PolicyAction.QUERY],
        resource_classifications=["internal", "confidential", "secret", "top_secret"],
        conditions=[
            PolicyCondition(
                attribute="subject.is_verified",
                operator=ConditionOperator.EQUALS,
                value=False,
            ),
        ],
    )
