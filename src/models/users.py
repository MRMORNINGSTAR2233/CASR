"""
User Models

Defines user structure, roles, and session management for the CASR system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field

from .documents import SecurityClassification


class UserRole(str, Enum):
    """
    Predefined user roles with associated permissions.
    
    Roles form a hierarchy where higher roles inherit lower role permissions.
    """
    GUEST = "guest"                    # Read-only, public documents only
    USER = "user"                      # Standard user, internal documents
    ANALYST = "analyst"                # Can access confidential data
    MANAGER = "manager"                # Department-level access
    EXECUTIVE = "executive"            # Cross-department access
    ADMIN = "admin"                    # Full system access
    SYSTEM = "system"                  # Internal system operations
    
    @property
    def clearance(self) -> SecurityClassification:
        """Map role to security clearance level."""
        clearance_map = {
            "guest": SecurityClassification.PUBLIC,
            "user": SecurityClassification.INTERNAL,
            "analyst": SecurityClassification.CONFIDENTIAL,
            "manager": SecurityClassification.CONFIDENTIAL,
            "executive": SecurityClassification.SECRET,
            "admin": SecurityClassification.TOP_SECRET,
            "system": SecurityClassification.TOP_SECRET,
        }
        return clearance_map[self.value]
    
    @property
    def level(self) -> int:
        """Numeric level for role comparison."""
        levels = {
            "guest": 0,
            "user": 1,
            "analyst": 2,
            "manager": 3,
            "executive": 4,
            "admin": 5,
            "system": 6,
        }
        return levels[self.value]
    
    def has_permission(self, required_role: "UserRole") -> bool:
        """Check if this role has at least the permissions of required_role."""
        return self.level >= required_role.level


class User(BaseModel):
    """
    A user in the CASR system.
    
    Contains identity, role, and attribute information for access control.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique user ID")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    
    # Role and permissions
    role: UserRole = Field(default=UserRole.USER, description="Primary user role")
    additional_roles: list[UserRole] = Field(
        default_factory=list,
        description="Additional roles assigned to user"
    )
    
    # Organizational attributes (for ABAC)
    department: Optional[str] = Field(default=None, description="User's department")
    team: Optional[str] = Field(default=None, description="User's team")
    location: Optional[str] = Field(default=None, description="User's location")
    cost_center: Optional[str] = Field(default=None, description="User's cost center")
    
    # Domain access
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="Domains user can access (empty = all permitted by role)"
    )
    
    # Security clearance override
    clearance_override: Optional[SecurityClassification] = Field(
        default=None,
        description="Override the role-based clearance"
    )
    
    # Account status
    is_active: bool = Field(default=True, description="Whether account is active")
    is_verified: bool = Field(default=False, description="Whether email is verified")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = Field(default=None)
    
    # Custom attributes for ABAC
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom attributes for policy evaluation"
    )
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    
    @property
    def effective_clearance(self) -> SecurityClassification:
        """Get the effective security clearance considering overrides."""
        if self.clearance_override:
            return self.clearance_override
        return self.role.clearance
    
    @property
    def all_roles(self) -> list[UserRole]:
        """Get all roles including primary and additional."""
        return [self.role] + self.additional_roles
    
    @property
    def highest_role(self) -> UserRole:
        """Get the highest privilege role the user has."""
        return max(self.all_roles, key=lambda r: r.level)
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has a specific role."""
        return role in self.all_roles
    
    def can_access_domain(self, domain: str) -> bool:
        """Check if user can access a specific domain."""
        if not self.allowed_domains:
            return True  # No restrictions, use role-based access
        return domain in self.allowed_domains
    
    def to_context_dict(self) -> dict[str, Any]:
        """Convert to dictionary for query context."""
        return {
            "user_id": str(self.id),
            "username": self.username,
            "role": self.role.value,
            "all_roles": [r.value for r in self.all_roles],
            "department": self.department,
            "team": self.team,
            "clearance": self.effective_clearance.value,
            "allowed_domains": self.allowed_domains,
            "attributes": self.attributes,
        }


class UserSession(BaseModel):
    """
    An active user session with context for retrieval.
    
    Tracks session-specific information and query history for contextual retrieval.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Session ID")
    user_id: UUID = Field(..., description="Associated user ID")
    user: User = Field(..., description="User object")
    
    # Session timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Session expiration time")
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    # Session context
    current_domain: Optional[str] = Field(
        default=None,
        description="Current working domain context"
    )
    current_project: Optional[str] = Field(
        default=None,
        description="Current project context"
    )
    
    # Query history for context
    recent_queries: list[str] = Field(
        default_factory=list,
        description="Recent queries in this session (last 10)"
    )
    recent_topics: list[str] = Field(
        default_factory=list,
        description="Detected topics from recent queries"
    )
    
    # Session metadata
    ip_address: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    
    def add_query(self, query: str, topic: Optional[str] = None) -> None:
        """Add a query to session history."""
        self.recent_queries.append(query)
        if len(self.recent_queries) > 10:
            self.recent_queries = self.recent_queries[-10:]
        
        if topic:
            self.recent_topics.append(topic)
            if len(self.recent_topics) > 10:
                self.recent_topics = self.recent_topics[-10:]
        
        self.last_activity = datetime.utcnow()
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.utcnow() > self.expires_at
    
    def get_context_summary(self) -> str:
        """Generate a summary of session context for query analysis."""
        parts = []
        
        if self.current_domain:
            parts.append(f"Domain: {self.current_domain}")
        if self.current_project:
            parts.append(f"Project: {self.current_project}")
        if self.recent_topics:
            parts.append(f"Recent topics: {', '.join(self.recent_topics[-3:])}")
        
        return "; ".join(parts) if parts else "No specific context"
