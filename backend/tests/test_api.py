"""
Backend test suite.
Coverage target: >90%
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from app.db.base import Base
from app.db.session import get_db
from app.core.security import get_password_hash
from app.models.user import User, Role

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create test client with overridden dependencies."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session):
    """Create test user."""
    role = Role(name="doctor", description="Doctor role")
    db_session.add(role)
    db_session.commit()

    user = User(
        email="doctor@test.com",
        username="testdoctor",
        hashed_password=get_password_hash("password123"),
        full_name="Test Doctor",
        role_id=role.id,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    return user


class TestAuth:
    """Authentication tests."""

    def test_login_success(self, client, test_user):
        """Test successful login."""
        response = client.post("/api/v1/auth/login", json={
            "username": "testdoctor",
            "password": "password123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["username"] == "testdoctor"

    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post("/api/v1/auth/login", json={
            "username": "wrong",
            "password": "wrong",
        })
        assert response.status_code == 401

    def test_me_endpoint(self, client, test_user):
        """Test get current user."""
        # Login first
        login_response = client.post("/api/v1/auth/login", json={
            "username": "testdoctor",
            "password": "password123",
        })
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert response.json()["username"] == "testdoctor"


class TestPatients:
    """Patient management tests."""

    def test_create_patient(self, client, test_user):
        """Test patient creation."""
        # Login
        login_response = client.post("/api/v1/auth/login", json={
            "username": "testdoctor",
            "password": "password123",
        })
        token = login_response.json()["access_token"]

        response = client.post(
            "/api/v1/patients",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "first_name": "Ahmad",
                "last_name": "Mohammed",
                "date_of_birth": "2022-01-01",
                "sex": "male",
                "caregiver_name": "Father",
                "caregiver_phone": "+967777777777",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["first_name"] == "Ahmad"
        assert "registration_number" in data

    def test_list_patients(self, client, test_user):
        """Test patient listing."""
        login_response = client.post("/api/v1/auth/login", json={
            "username": "testdoctor",
            "password": "password123",
        })
        token = login_response.json()["access_token"]

        response = client.get(
            "/api/v1/patients",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestPredictions:
    """Prediction tests."""

    def test_prediction_requires_auth(self, client):
        """Test prediction endpoint requires authentication."""
        response = client.post("/api/v1/predictions/predict", json={
            "patient_id": 1,
            "measurement_id": 1,
        })
        assert response.status_code == 403


class TestHealth:
    """Health check tests."""

    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestRBAC:
    """Role-based access control tests."""

    def test_admin_only_endpoints(self, client, test_user):
        """Test admin-only endpoints are protected."""
        login_response = client.post("/api/v1/auth/login", json={
            "username": "testdoctor",
            "password": "password123",
        })
        token = login_response.json()["access_token"]

        # Doctor trying to access admin endpoint
        response = client.get(
            "/api/v1/users",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app", "--cov-report=term-missing"])
