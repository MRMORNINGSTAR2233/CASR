"""
Audit Logging

Comprehensive audit logging for security and compliance.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field


class AuditAction(str, Enum):
    """Types of auditable actions."""
    
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    TOKEN_REFRESH = "token_refresh"
    
    # Authorization
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHECK = "permission_check"
    
    # Query operations
    QUERY_SUBMITTED = "query_submitted"
    QUERY_ANALYZED = "query_analyzed"
    RETRIEVAL_PERFORMED = "retrieval_performed"
    RESULTS_RETURNED = "results_returned"
    
    # Document operations
    DOCUMENT_INDEXED = "document_indexed"
    DOCUMENT_ACCESSED = "document_accessed"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_UPDATED = "document_updated"
    
    # Admin operations
    POLICY_CREATED = "policy_created"
    POLICY_UPDATED = "policy_updated"
    POLICY_DELETED = "policy_deleted"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    ROLE_CHANGED = "role_changed"
    
    # Security events
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class AuditSeverity(str, Enum):
    """Severity levels for audit entries."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEntry(BaseModel):
    """
    An audit log entry.
    
    Contains comprehensive information about an auditable event.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique entry ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Action details
    action: AuditAction = Field(..., description="Type of action")
    severity: AuditSeverity = Field(
        default=AuditSeverity.INFO,
        description="Severity level"
    )
    success: bool = Field(default=True, description="Whether action succeeded")
    
    # Subject (who)
    user_id: Optional[str] = Field(default=None, description="Acting user ID")
    username: Optional[str] = Field(default=None, description="Acting username")
    user_role: Optional[str] = Field(default=None, description="User's role")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    
    # Resource (what)
    resource_type: Optional[str] = Field(
        default=None,
        description="Type of resource (document, policy, user, etc.)"
    )
    resource_id: Optional[str] = Field(default=None, description="Resource ID")
    resource_name: Optional[str] = Field(default=None, description="Resource name")
    
    # Context (how/where)
    ip_address: Optional[str] = Field(default=None)
    user_agent: Optional[str] = Field(default=None)
    endpoint: Optional[str] = Field(default=None, description="API endpoint")
    method: Optional[str] = Field(default=None, description="HTTP method")
    
    # Details
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event details"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if action failed"
    )
    
    # Policy information
    policy_id: Optional[str] = Field(
        default=None,
        description="Policy that granted/denied access"
    )
    policy_name: Optional[str] = Field(default=None)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }
    
    def to_log_dict(self) -> dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "audit_id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "severity": self.severity.value,
            "success": self.success,
            "user_id": self.user_id,
            "username": self.username,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "details": self.details,
            "error": self.error_message,
        }


class AuditLog:
    """
    In-memory audit log with optional file persistence.
    
    Provides query capabilities for audit entries.
    """
    
    def __init__(
        self,
        max_entries: int = 10000,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the audit log.
        
        Args:
            max_entries: Maximum entries to keep in memory
            storage_path: Optional path for file persistence
        """
        self._entries: list[AuditEntry] = []
        self._max_entries = max_entries
        self._storage_path = storage_path
    
    def add(self, entry: AuditEntry) -> AuditEntry:
        """Add an entry to the log."""
        self._entries.append(entry)
        
        # Trim if exceeding max
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]
        
        # Persist if storage configured
        if self._storage_path:
            self._append_to_file(entry)
        
        return entry
    
    def _append_to_file(self, entry: AuditEntry) -> None:
        """Append entry to log file."""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self._storage_path, "a") as f:
            f.write(json.dumps(entry.model_dump(mode="json"), default=str) + "\n")
    
    def query(
        self,
        action: Optional[AuditAction] = None,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success: Optional[bool] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100
    ) -> list[AuditEntry]:
        """
        Query audit entries with filters.
        
        Args:
            action: Filter by action type
            user_id: Filter by user ID
            resource_id: Filter by resource ID
            start_time: Filter by start time
            end_time: Filter by end time
            success: Filter by success status
            severity: Filter by severity
            limit: Maximum entries to return
            
        Returns:
            List of matching audit entries
        """
        results = []
        
        for entry in reversed(self._entries):  # Most recent first
            if len(results) >= limit:
                break
            
            if action and entry.action != action:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if resource_id and entry.resource_id != resource_id:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if success is not None and entry.success != success:
                continue
            if severity and entry.severity != severity:
                continue
            
            results.append(entry)
        
        return results
    
    def get_user_activity(
        self,
        user_id: str,
        limit: int = 50
    ) -> list[AuditEntry]:
        """Get recent activity for a user."""
        return self.query(user_id=user_id, limit=limit)
    
    def get_security_events(self, limit: int = 100) -> list[AuditEntry]:
        """Get recent security-related events."""
        security_actions = {
            AuditAction.ACCESS_DENIED,
            AuditAction.LOGIN_FAILED,
            AuditAction.SECURITY_VIOLATION,
            AuditAction.SUSPICIOUS_ACTIVITY,
            AuditAction.RATE_LIMIT_EXCEEDED,
        }
        
        results = []
        for entry in reversed(self._entries):
            if len(results) >= limit:
                break
            if entry.action in security_actions:
                results.append(entry)
        
        return results
    
    def get_failed_access_attempts(
        self,
        user_id: Optional[str] = None,
        limit: int = 50
    ) -> list[AuditEntry]:
        """Get failed access attempts."""
        return self.query(
            action=AuditAction.ACCESS_DENIED,
            user_id=user_id,
            limit=limit
        )


class AuditLogger:
    """
    High-level audit logging interface.
    
    Provides convenient methods for logging common audit events.
    """
    
    def __init__(
        self,
        audit_log: Optional[AuditLog] = None,
        enable_console: bool = True
    ):
        """
        Initialize the audit logger.
        
        Args:
            audit_log: Optional AuditLog instance for storage
            enable_console: Whether to also log to console
        """
        self.audit_log = audit_log or AuditLog()
        self.enable_console = enable_console
        
        if enable_console:
            self._logger = structlog.get_logger("casr.audit")
    
    def _log_entry(self, entry: AuditEntry) -> AuditEntry:
        """Log an entry to all configured destinations."""
        self.audit_log.add(entry)
        
        if self.enable_console:
            log_method = getattr(self._logger, entry.severity.value, self._logger.info)
            log_method(
                entry.action.value,
                **entry.to_log_dict()
            )
        
        return entry
    
    def log_access_check(
        self,
        user_id: str,
        username: str,
        user_role: str,
        resource_type: str,
        resource_id: str,
        allowed: bool,
        policy_name: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> AuditEntry:
        """Log an access check event."""
        entry = AuditEntry(
            action=AuditAction.ACCESS_GRANTED if allowed else AuditAction.ACCESS_DENIED,
            severity=AuditSeverity.INFO if allowed else AuditSeverity.WARNING,
            success=allowed,
            user_id=user_id,
            username=username,
            user_role=user_role,
            resource_type=resource_type,
            resource_id=resource_id,
            policy_name=policy_name,
            details={"reason": reason, **kwargs} if reason else kwargs,
        )
        return self._log_entry(entry)
    
    def log_query(
        self,
        user_id: str,
        username: str,
        query_text: str,
        query_id: str,
        results_count: int = 0,
        filtered_count: int = 0,
        **kwargs
    ) -> AuditEntry:
        """Log a query event."""
        entry = AuditEntry(
            action=AuditAction.QUERY_SUBMITTED,
            severity=AuditSeverity.INFO,
            success=True,
            user_id=user_id,
            username=username,
            resource_type="query",
            resource_id=query_id,
            details={
                "query_text": query_text[:200],  # Truncate long queries
                "results_count": results_count,
                "filtered_by_security": filtered_count,
                **kwargs,
            },
        )
        return self._log_entry(entry)
    
    def log_retrieval(
        self,
        user_id: str,
        query_id: str,
        documents_retrieved: int,
        documents_filtered: int,
        retrieval_time_ms: float,
        **kwargs
    ) -> AuditEntry:
        """Log a retrieval event."""
        entry = AuditEntry(
            action=AuditAction.RETRIEVAL_PERFORMED,
            severity=AuditSeverity.INFO,
            success=True,
            user_id=user_id,
            resource_type="query",
            resource_id=query_id,
            details={
                "documents_retrieved": documents_retrieved,
                "documents_filtered_by_security": documents_filtered,
                "retrieval_time_ms": retrieval_time_ms,
                **kwargs,
            },
        )
        return self._log_entry(entry)
    
    def log_document_access(
        self,
        user_id: str,
        username: str,
        document_id: str,
        document_title: str,
        classification: str,
        **kwargs
    ) -> AuditEntry:
        """Log a document access event."""
        entry = AuditEntry(
            action=AuditAction.DOCUMENT_ACCESSED,
            severity=AuditSeverity.INFO,
            success=True,
            user_id=user_id,
            username=username,
            resource_type="document",
            resource_id=document_id,
            resource_name=document_title,
            details={
                "classification": classification,
                **kwargs,
            },
        )
        return self._log_entry(entry)
    
    def log_login(
        self,
        user_id: str,
        username: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: Optional[str] = None,
        **kwargs
    ) -> AuditEntry:
        """Log a login event."""
        entry = AuditEntry(
            action=AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            success=success,
            user_id=user_id if success else None,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=failure_reason,
            details=kwargs,
        )
        return self._log_entry(entry)
    
    def log_security_violation(
        self,
        user_id: Optional[str],
        username: Optional[str],
        violation_type: str,
        details: dict[str, Any],
        ip_address: Optional[str] = None,
        **kwargs
    ) -> AuditEntry:
        """Log a security violation."""
        entry = AuditEntry(
            action=AuditAction.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            success=False,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            details={
                "violation_type": violation_type,
                **details,
                **kwargs,
            },
        )
        return self._log_entry(entry)
    
    def log_policy_change(
        self,
        user_id: str,
        username: str,
        action: str,  # created, updated, deleted
        policy_id: str,
        policy_name: str,
        **kwargs
    ) -> AuditEntry:
        """Log a policy change event."""
        action_map = {
            "created": AuditAction.POLICY_CREATED,
            "updated": AuditAction.POLICY_UPDATED,
            "deleted": AuditAction.POLICY_DELETED,
        }
        
        entry = AuditEntry(
            action=action_map.get(action, AuditAction.POLICY_UPDATED),
            severity=AuditSeverity.WARNING,  # Policy changes are security-relevant
            success=True,
            user_id=user_id,
            username=username,
            resource_type="policy",
            resource_id=policy_id,
            resource_name=policy_name,
            details=kwargs,
        )
        return self._log_entry(entry)
