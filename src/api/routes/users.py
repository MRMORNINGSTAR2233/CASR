"""
User Routes

User management and authentication endpoints.
"""

from datetime import datetime, timedelta
from typing import Annotated, Optional
from uuid import UUID

import jwt
from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from config import get_settings
from src.api.dependencies import (
    get_audit_logger,
    get_current_user,
    get_policy_engine,
    require_role,
)
from src.models.users import User, UserRole
from src.security.audit import AuditLogger
from src.security.policies import PolicyEngine

router = APIRouter()


# Request/Response Models
class UserLoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class UserLoginResponse(BaseModel):
    """Login response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class UserCreateRequest(BaseModel):
    """Create user request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.USER
    department: Optional[str] = None
    allowed_domains: list[str] = Field(default_factory=list)


class UserUpdateRequest(BaseModel):
    """Update user request."""
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    department: Optional[str] = None
    allowed_domains: Optional[list[str]] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response model."""
    id: UUID
    username: str
    email: str
    role: str
    department: Optional[str]
    allowed_domains: list[str]
    is_active: bool
    created_at: Optional[datetime]


# Mock user database (replace with real database)
_users_db: dict[str, dict] = {
    "admin": {
        "id": "00000000-0000-0000-0000-000000000001",
        "username": "admin",
        "email": "admin@example.com",
        "password_hash": "admin123",  # In production, use proper hashing
        "role": "admin",
        "department": "IT",
        "allowed_domains": [],
        "is_active": True,
    },
    "analyst": {
        "id": "00000000-0000-0000-0000-000000000002",
        "username": "analyst",
        "email": "analyst@example.com",
        "password_hash": "analyst123",
        "role": "analyst",
        "department": "Research",
        "allowed_domains": ["research", "public"],
        "is_active": True,
    },
}


def _create_token(user: dict, expires_delta: timedelta) -> str:
    """Create JWT token for user."""
    settings = get_settings()
    
    expire = datetime.utcnow() + expires_delta
    payload = {
        "sub": user["id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "department": user.get("department"),
        "allowed_domains": user.get("allowed_domains", []),
        "exp": expire,
    }
    
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


@router.post("/login", response_model=UserLoginResponse)
async def login(
    request: UserLoginRequest,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> UserLoginResponse:
    """Authenticate user and return JWT token."""
    settings = get_settings()
    
    # Look up user
    user_data = _users_db.get(request.username)
    
    if not user_data or user_data["password_hash"] != request.password:
        audit_logger.log_authentication(
            user_id=request.username,
            success=False,
            method="password",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    if not user_data["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    # Create token
    expires = timedelta(hours=settings.jwt_expiration_hours)
    token = _create_token(user_data, expires)
    
    audit_logger.log_authentication(
        user_id=user_data["id"],
        success=True,
        method="password",
    )
    
    return UserLoginResponse(
        access_token=token,
        expires_in=int(expires.total_seconds()),
        user={
            "id": user_data["id"],
            "username": user_data["username"],
            "role": user_data["role"],
        },
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Get current authenticated user information."""
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role.value,
        department=user.department,
        allowed_domains=user.allowed_domains,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN))],
) -> UserResponse:
    """Get user by ID (admin only)."""
    # Find user
    for user_data in _users_db.values():
        if user_data["id"] == str(user_id):
            return UserResponse(
                id=UUID(user_data["id"]),
                username=user_data["username"],
                email=user_data["email"],
                role=user_data["role"],
                department=user_data.get("department"),
                allowed_domains=user_data.get("allowed_domains", []),
                is_active=user_data["is_active"],
                created_at=None,
            )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="User not found",
    )


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: UserCreateRequest,
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> UserResponse:
    """Create a new user (admin only)."""
    if request.username in _users_db:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )
    
    import uuid
    user_id = str(uuid.uuid4())
    
    _users_db[request.username] = {
        "id": user_id,
        "username": request.username,
        "email": request.email,
        "password_hash": request.password,  # Hash in production!
        "role": request.role.value,
        "department": request.department,
        "allowed_domains": request.allowed_domains,
        "is_active": True,
    }
    
    audit_logger._log_event(
        event_type="user_created",
        user_id=str(current_user.id),
        resource_id=user_id,
    )
    
    return UserResponse(
        id=UUID(user_id),
        username=request.username,
        email=request.email,
        role=request.role.value,
        department=request.department,
        allowed_domains=request.allowed_domains,
        is_active=True,
        created_at=datetime.utcnow(),
    )


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    request: UserUpdateRequest,
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> UserResponse:
    """Update a user (admin only)."""
    # Find user
    user_data = None
    for username, data in _users_db.items():
        if data["id"] == str(user_id):
            user_data = data
            break
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Update fields
    if request.email is not None:
        user_data["email"] = request.email
    if request.role is not None:
        user_data["role"] = request.role.value
    if request.department is not None:
        user_data["department"] = request.department
    if request.allowed_domains is not None:
        user_data["allowed_domains"] = request.allowed_domains
    if request.is_active is not None:
        user_data["is_active"] = request.is_active
    
    audit_logger._log_event(
        event_type="user_updated",
        user_id=str(current_user.id),
        resource_id=str(user_id),
    )
    
    return UserResponse(
        id=user_id,
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        department=user_data.get("department"),
        allowed_domains=user_data.get("allowed_domains", []),
        is_active=user_data["is_active"],
        created_at=None,
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    audit_logger: AuditLogger = Depends(get_audit_logger),
):
    """Delete a user (admin only)."""
    # Find and remove user
    username_to_delete = None
    for username, data in _users_db.items():
        if data["id"] == str(user_id):
            username_to_delete = username
            break
    
    if not username_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    del _users_db[username_to_delete]
    
    audit_logger._log_event(
        event_type="user_deleted",
        user_id=str(current_user.id),
        resource_id=str(user_id),
    )


@router.get("", response_model=list[UserResponse])
async def list_users(
    current_user: Annotated[User, Depends(require_role(UserRole.ADMIN))],
    skip: int = 0,
    limit: int = 100,
) -> list[UserResponse]:
    """List all users (admin only)."""
    users = []
    for user_data in list(_users_db.values())[skip:skip + limit]:
        users.append(
            UserResponse(
                id=UUID(user_data["id"]),
                username=user_data["username"],
                email=user_data["email"],
                role=user_data["role"],
                department=user_data.get("department"),
                allowed_domains=user_data.get("allowed_domains", []),
                is_active=user_data["is_active"],
                created_at=None,
            )
        )
    return users
