"""
Policy Models

Defines access control policies for RBAC and ABAC in the CASR system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PolicyEffect(str, Enum):
    """Effect of a policy rule."""
    ALLOW = "allow"
    DENY = "deny"


class PolicyAction(str, Enum):
    """Actions that can be controlled by policies."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    QUERY = "query"
    INDEX = "index"
    ADMIN = "admin"
    ALL = "*"


class ConditionOperator(str, Enum):
    """Operators for policy conditions."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    REGEX = "regex"


class PolicyCondition(BaseModel):
    """
    A condition for policy evaluation.
    
    Evaluates attributes against specified criteria.
    """
    
    attribute: str = Field(
        ...,
        description="Attribute path to evaluate (e.g., 'user.department', 'resource.classification')"
    )
    operator: ConditionOperator = Field(
        ...,
        description="Comparison operator"
    )
    value: Any = Field(
        ...,
        description="Value to compare against"
    )
    
    def evaluate(self, context: dict[str, Any]) -> bool:
        """
        Evaluate this condition against a context.
        
        Args:
            context: Dictionary containing attribute values
            
        Returns:
            bool: Whether the condition is satisfied
        """
        # Get attribute value using dot notation
        attr_value = self._get_nested_value(context, self.attribute)
        
        # Handle non-existent attributes
        if attr_value is None:
            if self.operator == ConditionOperator.NOT_EXISTS:
                return True
            if self.operator == ConditionOperator.EXISTS:
                return False
            return False
        
        # Evaluate based on operator
        try:
            match self.operator:
                case ConditionOperator.EQUALS:
                    return attr_value == self.value
                case ConditionOperator.NOT_EQUALS:
                    return attr_value != self.value
                case ConditionOperator.IN:
                    return attr_value in self.value
                case ConditionOperator.NOT_IN:
                    return attr_value not in self.value
                case ConditionOperator.CONTAINS:
                    return self.value in attr_value
                case ConditionOperator.STARTS_WITH:
                    return str(attr_value).startswith(str(self.value))
                case ConditionOperator.ENDS_WITH:
                    return str(attr_value).endswith(str(self.value))
                case ConditionOperator.GREATER_THAN:
                    return attr_value > self.value
                case ConditionOperator.LESS_THAN:
                    return attr_value < self.value
                case ConditionOperator.GREATER_THAN_OR_EQUAL:
                    return attr_value >= self.value
                case ConditionOperator.LESS_THAN_OR_EQUAL:
                    return attr_value <= self.value
                case ConditionOperator.EXISTS:
                    return True
                case ConditionOperator.NOT_EXISTS:
                    return False
                case ConditionOperator.REGEX:
                    import re
                    return bool(re.match(str(self.value), str(attr_value)))
                case _:
                    return False
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def _get_nested_value(obj: dict, path: str) -> Any:
        """Get a nested value using dot notation."""
        keys = path.split(".")
        value = obj
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
            if value is None:
                return None
        return value


class AccessPolicy(BaseModel):
    """
    An access control policy.
    
    Defines rules for accessing resources based on user and resource attributes.
    Supports both RBAC and ABAC patterns.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Policy ID")
    name: str = Field(..., description="Policy name")
    description: Optional[str] = Field(default=None, description="Policy description")
    
    # Policy status
    is_active: bool = Field(default=True, description="Whether policy is active")
    priority: int = Field(
        default=0,
        description="Policy priority (higher = evaluated first)"
    )
    
    # Effect
    effect: PolicyEffect = Field(..., description="Allow or deny effect")
    
    # Actions this policy applies to
    actions: list[PolicyAction] = Field(
        default_factory=lambda: [PolicyAction.ALL],
        description="Actions this policy applies to"
    )
    
    # Resource targeting
    resource_type: Optional[str] = Field(
        default=None,
        description="Type of resource (document, chunk, etc.)"
    )
    resource_domains: list[str] = Field(
        default_factory=list,
        description="Domains this policy applies to (empty = all)"
    )
    resource_classifications: list[str] = Field(
        default_factory=list,
        description="Classifications this policy applies to (empty = all)"
    )
    
    # Subject targeting (RBAC)
    subject_roles: list[str] = Field(
        default_factory=list,
        description="Roles this policy applies to (empty = all)"
    )
    subject_users: list[str] = Field(
        default_factory=list,
        description="Specific user IDs (empty = all)"
    )
    
    # Conditions (ABAC)
    conditions: list[PolicyCondition] = Field(
        default_factory=list,
        description="Conditions that must all be satisfied"
    )
    condition_logic: str = Field(
        default="AND",
        description="Logic for combining conditions: AND or OR"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None)
    
    # Tags for organization
    tags: list[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    
    def matches_subject(self, user_role: str, user_id: str) -> bool:
        """Check if policy applies to given subject."""
        # Check user ID first
        if self.subject_users and user_id not in self.subject_users:
            if self.subject_roles:  # If roles specified, check those too
                pass
            else:
                return False
        
        # Check roles
        if self.subject_roles and user_role not in self.subject_roles:
            return False
        
        return True
    
    def matches_resource(
        self,
        domain: Optional[str] = None,
        classification: Optional[str] = None
    ) -> bool:
        """Check if policy applies to given resource."""
        if self.resource_domains and domain not in self.resource_domains:
            return False
        
        if self.resource_classifications and classification not in self.resource_classifications:
            return False
        
        return True
    
    def matches_action(self, action: PolicyAction) -> bool:
        """Check if policy applies to given action."""
        if PolicyAction.ALL in self.actions:
            return True
        return action in self.actions
    
    def evaluate_conditions(self, context: dict[str, Any]) -> bool:
        """Evaluate all conditions against context."""
        if not self.conditions:
            return True
        
        if self.condition_logic.upper() == "OR":
            return any(cond.evaluate(context) for cond in self.conditions)
        else:  # AND
            return all(cond.evaluate(context) for cond in self.conditions)
    
    def evaluate(
        self,
        user_role: str,
        user_id: str,
        action: PolicyAction,
        resource_domain: Optional[str] = None,
        resource_classification: Optional[str] = None,
        context: Optional[dict[str, Any]] = None
    ) -> Optional[PolicyEffect]:
        """
        Evaluate the policy for a given request.
        
        Returns:
            PolicyEffect if policy matches, None if policy doesn't apply
        """
        if not self.is_active:
            return None
        
        if not self.matches_subject(user_role, user_id):
            return None
        
        if not self.matches_action(action):
            return None
        
        if not self.matches_resource(resource_domain, resource_classification):
            return None
        
        if context and not self.evaluate_conditions(context):
            return None
        
        return self.effect


class PolicySet(BaseModel):
    """
    A collection of policies with evaluation logic.
    
    Implements deny-override: if any policy denies, the result is deny.
    """
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Policy set name")
    description: Optional[str] = Field(default=None)
    
    policies: list[AccessPolicy] = Field(
        default_factory=list,
        description="Policies in this set"
    )
    
    # Default effect when no policies match
    default_effect: PolicyEffect = Field(
        default=PolicyEffect.DENY,
        description="Default effect when no policies match"
    )
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    
    def evaluate(
        self,
        user_role: str,
        user_id: str,
        action: PolicyAction,
        resource_domain: Optional[str] = None,
        resource_classification: Optional[str] = None,
        context: Optional[dict[str, Any]] = None
    ) -> PolicyEffect:
        """
        Evaluate all policies and determine final effect.
        
        Uses deny-override: any DENY results in final DENY.
        """
        # Sort by priority (higher first)
        sorted_policies = sorted(self.policies, key=lambda p: p.priority, reverse=True)
        
        effects = []
        for policy in sorted_policies:
            effect = policy.evaluate(
                user_role=user_role,
                user_id=user_id,
                action=action,
                resource_domain=resource_domain,
                resource_classification=resource_classification,
                context=context,
            )
            if effect is not None:
                effects.append(effect)
        
        if not effects:
            return self.default_effect
        
        # Deny-override: any DENY means final is DENY
        if PolicyEffect.DENY in effects:
            return PolicyEffect.DENY
        
        # If we have at least one ALLOW and no DENY
        if PolicyEffect.ALLOW in effects:
            return PolicyEffect.ALLOW
        
        return self.default_effect
