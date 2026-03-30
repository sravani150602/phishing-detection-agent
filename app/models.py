"""Pydantic request/response models for the Phishing Detection Agent API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class EmailInput(BaseModel):
    """A single email to classify."""

    subject: str = Field(..., min_length=1, max_length=500, description="Email subject line")
    body: str = Field(..., min_length=1, max_length=10_000, description="Plain-text or HTML email body")
    sender: str = Field(..., description="Sender email address or display name")

    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "Urgent: Verify your PayPal account",
                "body": "Dear valued customer, your account has been compromised. Click here immediately...",
                "sender": "security@paypa1.com",
            }
        }
    }


class BatchEmailInput(BaseModel):
    """Batch of emails to classify (max 50)."""

    emails: list[EmailInput] = Field(..., min_length=1, max_length=50)


class ClassificationResult(BaseModel):
    """Result from the agent for a single email."""

    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    classification: Literal["phishing", "legitimate"] = Field(..., description="Agent classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0–1)")
    reasoning: str = Field(..., description="LLM chain-of-thought reasoning trace")
    indicators: list[str] = Field(default_factory=list, description="Detected phishing indicator tags")
    flagged: bool = Field(..., description="True if confidence exceeds the threshold and action is recommended")
    classified_at: datetime = Field(default_factory=datetime.utcnow)


class BatchClassificationResult(BaseModel):
    """Results for a batch classification request."""

    results: list[ClassificationResult]
    total: int
    phishing_count: int
    legitimate_count: int
    avg_confidence: float


class FeedbackInput(BaseModel):
    """Human feedback on an agent decision (fuels the evaluation loop)."""

    correct: bool = Field(..., description="Was the agent's classification correct?")
    human_label: Literal["phishing", "legitimate"] = Field(..., description="Human-verified label")
    notes: Optional[str] = Field(None, max_length=1000, description="Optional reviewer notes")


class MetricsResponse(BaseModel):
    """Aggregate performance metrics."""

    total_decisions: int
    phishing_detected: int
    legitimate_classified: int
    feedback_submitted: int
    accuracy_from_feedback: Optional[float]
    avg_confidence: float
    cycle_improvements: list[dict]
