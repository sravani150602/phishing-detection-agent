"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models import ClassificationResult


@pytest.fixture
def mock_classify():
    """Mock the agent's classify method to avoid real OpenAI calls in tests."""
    phishing_result = ClassificationResult(
        classification="phishing",
        confidence=0.95,
        reasoning="Suspicious sender domain and urgency language detected.",
        indicators=["lookalike_domain", "urgency_language"],
        flagged=True,
    )
    with patch("app.main.agent.classify", new_callable=AsyncMock) as m:
        m.return_value = (phishing_result, 320.5)
        yield m


@pytest.fixture
def mock_db():
    """Mock the database dependency."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)
    return mock_session


def test_health_check():
    from app.main import app
    with patch("app.main.init_db", new_callable=AsyncMock):
        with TestClient(app) as client:
            response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_classify_returns_result(mock_classify, mock_db):
    from app.main import app, get_db

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db

    with patch("app.main.init_db", new_callable=AsyncMock):
        with TestClient(app) as client:
            response = client.post("/classify", json={
                "subject": "Urgent: Verify your account",
                "body": "Click here to restore your access immediately...",
                "sender": "security@paypa1.com",
            })

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["classification"] == "phishing"
    assert data["confidence"] == 0.95
    assert data["flagged"] is True
    assert "decision_id" in data


def test_classify_validates_empty_subject(mock_db):
    from app.main import app, get_db

    async def override_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_db

    with patch("app.main.init_db", new_callable=AsyncMock):
        with TestClient(app) as client:
            response = client.post("/classify", json={
                "subject": "",
                "body": "Some body",
                "sender": "test@example.com",
            })

    app.dependency_overrides.clear()
    assert response.status_code == 422  # Pydantic validation error
