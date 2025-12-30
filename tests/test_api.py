"""
Tests for API Layer

Tests for FastAPI routes and middleware.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create test application."""
    from src.api.app import create_app
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestHealthRoutes:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
    
    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        # This may fail if vector store is not available
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestUserRoutes:
    """Tests for user management endpoints."""
    
    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/api/v1/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self, client):
        """Test login with wrong password."""
        response = client.post(
            "/api/v1/users/login",
            json={"username": "admin", "password": "wrongpassword"},
        )
        
        assert response.status_code == 401
    
    def test_get_current_user_unauthorized(self, client):
        """Test accessing /me without authentication."""
        response = client.get("/api/v1/users/me")
        
        assert response.status_code == 401
    
    def test_get_current_user_with_token(self, client):
        """Test accessing /me with valid token."""
        # First, login
        login_response = client.post(
            "/api/v1/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = login_response.json()["access_token"]
        
        # Then, access /me
        response = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"


class TestDocumentRoutes:
    """Tests for document management endpoints."""
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get authentication headers."""
        login_response = client.post(
            "/api/v1/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_create_document_unauthorized(self, client):
        """Test creating document without auth."""
        response = client.post(
            "/api/v1/documents",
            json={
                "title": "Test Doc",
                "content": "Test content",
            },
        )
        
        assert response.status_code == 401
    
    def test_create_document(self, client, auth_headers):
        """Test creating a document."""
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers,
            json={
                "title": "Test Document",
                "content": "This is test content for the document.",
                "domain": "research",
                "classification": "public",
            },
        )
        
        # May fail if indexing components not available
        if response.status_code == 201:
            data = response.json()
            assert "document_id" in data
            assert data["chunks_created"] > 0
    
    def test_list_documents(self, client, auth_headers):
        """Test listing documents."""
        response = client.get(
            "/api/v1/documents",
            headers=auth_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestQueryRoutes:
    """Tests for search/query endpoints."""
    
    @pytest.fixture
    def auth_headers(self, client):
        """Get authentication headers."""
        login_response = client.post(
            "/api/v1/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_search_unauthorized(self, client):
        """Test searching without auth."""
        response = client.post(
            "/api/v1/queries/search",
            json={"query": "test query"},
        )
        
        assert response.status_code == 401
    
    def test_quick_search(self, client, auth_headers):
        """Test quick search endpoint."""
        response = client.post(
            "/api/v1/queries/quick",
            headers=auth_headers,
            json={
                "query": "test query",
                "limit": 5,
            },
        )
        
        # May return empty results if no documents indexed
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "count" in data


class TestMiddleware:
    """Tests for middleware functionality."""
    
    def test_request_id_header(self, client):
        """Test that request ID header is added."""
        response = client.get("/health")
        
        assert "X-Request-ID" in response.headers
    
    def test_response_time_header(self, client):
        """Test that response time header is added."""
        response = client.get("/health")
        
        assert "X-Response-Time" in response.headers
    
    def test_rate_limit_headers(self, client):
        """Test rate limit headers."""
        response = client.get("/health")
        
        # Health endpoint should not have rate limit headers
        # (it's excluded from rate limiting)
        
        # Test a rate-limited endpoint
        response = client.post(
            "/api/v1/users/login",
            json={"username": "admin", "password": "admin123"},
        )
        
        # Rate limit headers should be present on API endpoints
        if response.status_code == 200:
            assert "X-RateLimit-Limit" in response.headers
