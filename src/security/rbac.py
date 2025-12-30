"""
Role-Based Access Control (RBAC) Engine

Implements role-based access control for the CASR system.
"""

from typing import Optional

from src.models.documents import DocumentMetadata, SecurityClassification
from src.models.users import User, UserRole


class RBACEngine:
    """
    Role-Based Access Control Engine.
    
    Evaluates access based on user roles and resource classifications.
    Implements a hierarchical role model where higher roles inherit
    permissions from lower roles.
    """
    
    # Role hierarchy - each role includes all permissions of roles below it
    ROLE_HIERARCHY: dict[UserRole, list[UserRole]] = {
        UserRole.SYSTEM: [UserRole.ADMIN, UserRole.EXECUTIVE, UserRole.MANAGER, 
                          UserRole.ANALYST, UserRole.USER, UserRole.GUEST],
        UserRole.ADMIN: [UserRole.EXECUTIVE, UserRole.MANAGER, UserRole.ANALYST, 
                         UserRole.USER, UserRole.GUEST],
        UserRole.EXECUTIVE: [UserRole.MANAGER, UserRole.ANALYST, UserRole.USER, UserRole.GUEST],
        UserRole.MANAGER: [UserRole.ANALYST, UserRole.USER, UserRole.GUEST],
        UserRole.ANALYST: [UserRole.USER, UserRole.GUEST],
        UserRole.USER: [UserRole.GUEST],
        UserRole.GUEST: [],
    }
    
    # Role permissions for different actions
    ROLE_PERMISSIONS: dict[UserRole, set[str]] = {
        UserRole.GUEST: {"read:public"},
        UserRole.USER: {"read:public", "read:internal", "query"},
        UserRole.ANALYST: {"read:public", "read:internal", "read:confidential", "query", "export"},
        UserRole.MANAGER: {"read:public", "read:internal", "read:confidential", 
                           "query", "export", "manage:department"},
        UserRole.EXECUTIVE: {"read:public", "read:internal", "read:confidential", 
                             "read:secret", "query", "export", "manage:department"},
        UserRole.ADMIN: {"read:public", "read:internal", "read:confidential", 
                         "read:secret", "read:top_secret", "query", "export", 
                         "manage:department", "manage:users", "manage:policies", "admin"},
        UserRole.SYSTEM: {"*"},  # All permissions
    }
    
    def __init__(self):
        """Initialize the RBAC engine."""
        pass
    
    def get_effective_roles(self, user: User) -> set[UserRole]:
        """
        Get all effective roles for a user, including inherited roles.
        
        Args:
            user: The user to get roles for
            
        Returns:
            Set of all effective roles
        """
        effective_roles = set(user.all_roles)
        
        for role in user.all_roles:
            inherited = self.ROLE_HIERARCHY.get(role, [])
            effective_roles.update(inherited)
        
        return effective_roles
    
    def get_effective_permissions(self, user: User) -> set[str]:
        """
        Get all effective permissions for a user.
        
        Args:
            user: The user to get permissions for
            
        Returns:
            Set of all permission strings
        """
        effective_roles = self.get_effective_roles(user)
        permissions: set[str] = set()
        
        for role in effective_roles:
            role_perms = self.ROLE_PERMISSIONS.get(role, set())
            permissions.update(role_perms)
        
        return permissions
    
    def has_permission(self, user: User, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: The user to check
            permission: The permission string to check
            
        Returns:
            True if user has the permission
        """
        permissions = self.get_effective_permissions(user)
        
        # Wildcard permission grants all
        if "*" in permissions:
            return True
        
        return permission in permissions
    
    def can_access_classification(
        self,
        user: User,
        classification: SecurityClassification
    ) -> bool:
        """
        Check if user can access documents with given classification.
        
        Args:
            user: The user to check
            classification: The security classification to access
            
        Returns:
            True if user can access the classification level
        """
        user_clearance = user.effective_clearance
        return classification.can_access(user_clearance)
    
    def can_access_document(
        self,
        user: User,
        document_metadata: DocumentMetadata
    ) -> bool:
        """
        Check if user can access a specific document.
        
        Evaluates:
        1. Security classification
        2. Role-based document permissions
        3. Explicit user allowlists/denylists
        4. Domain restrictions
        
        Args:
            user: The user requesting access
            document_metadata: Metadata of the document
            
        Returns:
            True if access is allowed
        """
        # Check security classification first
        if not self.can_access_classification(user, document_metadata.classification):
            return False
        
        # Check if user is explicitly denied
        user_roles = [r.value for r in user.all_roles]
        if any(role in document_metadata.denied_roles for role in user_roles):
            return False
        
        # Check if user ID is explicitly denied
        if str(user.id) in document_metadata.allowed_users and document_metadata.denied_roles:
            pass  # Explicitly allowed users override role denials
        
        # Check if user ID is explicitly allowed
        if document_metadata.allowed_users:
            if str(user.id) in document_metadata.allowed_users:
                return True
        
        # Check if user role is explicitly allowed
        if document_metadata.allowed_roles:
            if not any(role in document_metadata.allowed_roles for role in user_roles):
                return False
        
        # Check domain restrictions
        if user.allowed_domains and document_metadata.domain:
            if document_metadata.domain not in user.allowed_domains:
                return False
        
        return True
    
    def filter_accessible_documents(
        self,
        user: User,
        documents: list[DocumentMetadata]
    ) -> list[DocumentMetadata]:
        """
        Filter a list of documents to only those the user can access.
        
        Args:
            user: The user requesting access
            documents: List of document metadata to filter
            
        Returns:
            List of accessible document metadata
        """
        return [doc for doc in documents if self.can_access_document(user, doc)]
    
    def get_accessible_classifications(self, user: User) -> list[SecurityClassification]:
        """
        Get all security classifications the user can access.
        
        Args:
            user: The user to check
            
        Returns:
            List of accessible classification levels
        """
        user_clearance = user.effective_clearance
        return [
            cls for cls in SecurityClassification
            if cls.level <= user_clearance.level
        ]
    
    def get_access_filter(self, user: User) -> dict:
        """
        Generate a filter dict for vector store queries based on user access.
        
        Args:
            user: The user making the query
            
        Returns:
            Dictionary of filters to apply to vector store queries
        """
        accessible_classifications = self.get_accessible_classifications(user)
        
        filters = {
            "classification": {
                "$in": [c.value for c in accessible_classifications]
            }
        }
        
        # Add domain restrictions if applicable
        if user.allowed_domains:
            filters["domain"] = {"$in": user.allowed_domains}
        
        # Add department filter if user is restricted
        if user.department and user.role.level < UserRole.EXECUTIVE.level:
            # Non-executives may be limited to their department for certain classifications
            pass  # This is handled by ABAC for more complex scenarios
        
        return filters
    
    def check_role_requirement(
        self,
        user: User,
        required_role: UserRole
    ) -> bool:
        """
        Check if user meets a minimum role requirement.
        
        Args:
            user: The user to check
            required_role: The minimum required role
            
        Returns:
            True if user has at least the required role level
        """
        return user.highest_role.has_permission(required_role)
