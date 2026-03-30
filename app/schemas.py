"""SQLAlchemy ORM table definitions for persisting agent decisions and feedback."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class AgentDecision(Base):
    """Persists every classification decision made by the LLM agent."""

    __tablename__ = "agent_decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_subject = Column(String(500), nullable=False)
    email_sender = Column(String(255), nullable=False)
    email_body_hash = Column(String(64), nullable=False)  # SHA-256 hash — never store raw bodies
    classification = Column(String(20), nullable=False)   # "phishing" | "legitimate"
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False)
    indicators = Column(ARRAY(String), nullable=False, default=list)
    flagged = Column(Boolean, nullable=False, default=False)
    prompt_version = Column(String(20), nullable=False, default="v1")
    model_used = Column(String(50), nullable=False, default="gpt-4o-mini")
    latency_ms = Column(Float, nullable=True)
    classified_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DecisionFeedback(Base):
    """Human reviewer feedback for evaluation cycle optimization."""

    __tablename__ = "decision_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    decision_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    correct = Column(Boolean, nullable=False)
    human_label = Column(String(20), nullable=False)
    notes = Column(Text, nullable=True)
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class EvaluationCycle(Base):
    """Tracks each prompt optimization cycle and the metrics before/after."""

    __tablename__ = "evaluation_cycles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cycle_number = Column(Float, nullable=False)
    prompt_version_before = Column(String(20), nullable=False)
    prompt_version_after = Column(String(20), nullable=False)
    accuracy_before = Column(Float, nullable=True)
    accuracy_after = Column(Float, nullable=True)
    precision_before = Column(Float, nullable=True)
    precision_after = Column(Float, nullable=True)
    recall_before = Column(Float, nullable=True)
    recall_after = Column(Float, nullable=True)
    sample_count = Column(Float, nullable=False)
    ran_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
