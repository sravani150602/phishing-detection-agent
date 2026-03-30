"""
FastAPI application — Phishing Detection Agent REST API

Endpoints:
  POST /classify             — Classify a single email
  POST /classify/batch       — Classify a batch of emails (up to 50)
  GET  /decisions            — Retrieve logged agent decisions
  POST /decisions/{id}/feedback — Submit human feedback
  GET  /metrics              — Aggregate performance metrics
  GET  /health               — Health check
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Literal, Optional

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import agent, hash_body
from app.config import settings
from app.database import get_db, init_db
from app.models import (
    BatchClassificationResult,
    BatchEmailInput,
    ClassificationResult,
    EmailInput,
    FeedbackInput,
    MetricsResponse,
)
from app.schemas import AgentDecision, DecisionFeedback, EvaluationCycle

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="LLM-Powered Phishing Detection Agent",
    description=(
        "An AI security agent that uses OpenAI GPT with engineered prompt pipelines "
        "to detect phishing emails in real time. Every decision is logged to PostgreSQL "
        "and feeds an evaluation feedback loop for continuous improvement."
    ),
    version="1.0.0",
    contact={
        "name": "Sravani Elavarthi",
        "url": "https://github.com/sravani150602",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    await init_db()
    logger.info("database_initialized")


# ── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {"status": "ok", "model": settings.OPENAI_MODEL, "prompt_version": settings.PROMPT_VERSION}


# ── Classification ──────────────────────────────────────────────────────────

@app.post("/classify", response_model=ClassificationResult, tags=["Agent"])
async def classify_email(
    email: EmailInput,
    db: AsyncSession = Depends(get_db),
) -> ClassificationResult:
    """
    Classify a single email as phishing or legitimate.

    The agent runs a chain-of-thought prompt pipeline and returns a structured
    decision with confidence score, reasoning trace, and detected indicators.
    All decisions are persisted to PostgreSQL for audit and evaluation.
    """
    result, latency_ms = await agent.classify(
        subject=email.subject,
        body=email.body,
        sender=email.sender,
    )

    # Persist the decision
    decision = AgentDecision(
        id=uuid.UUID(result.decision_id),
        email_subject=email.subject,
        email_sender=email.sender,
        email_body_hash=hash_body(email.body),
        classification=result.classification,
        confidence=result.confidence,
        reasoning=result.reasoning,
        indicators=result.indicators,
        flagged=result.flagged,
        prompt_version=settings.PROMPT_VERSION,
        model_used=settings.OPENAI_MODEL,
        latency_ms=latency_ms,
    )
    db.add(decision)

    return result


@app.post("/classify/batch", response_model=BatchClassificationResult, tags=["Agent"])
async def classify_batch(
    payload: BatchEmailInput,
    db: AsyncSession = Depends(get_db),
) -> BatchClassificationResult:
    """
    Classify a batch of up to 50 emails concurrently.

    Runs all classifications in parallel using asyncio and persists every decision.
    """
    tasks = [
        agent.classify(subject=e.subject, body=e.body, sender=e.sender)
        for e in payload.emails
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[ClassificationResult] = []
    for i, (email, outcome) in enumerate(zip(payload.emails, raw_results)):
        if isinstance(outcome, Exception):
            logger.error("batch_item_failed", index=i, error=str(outcome))
            continue

        result, latency_ms = outcome
        results.append(result)

        decision = AgentDecision(
            id=uuid.UUID(result.decision_id),
            email_subject=email.subject,
            email_sender=email.sender,
            email_body_hash=hash_body(email.body),
            classification=result.classification,
            confidence=result.confidence,
            reasoning=result.reasoning,
            indicators=result.indicators,
            flagged=result.flagged,
            prompt_version=settings.PROMPT_VERSION,
            model_used=settings.OPENAI_MODEL,
            latency_ms=latency_ms,
        )
        db.add(decision)

    phishing_count = sum(1 for r in results if r.classification == "phishing")
    avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0

    return BatchClassificationResult(
        results=results,
        total=len(results),
        phishing_count=phishing_count,
        legitimate_count=len(results) - phishing_count,
        avg_confidence=round(avg_confidence, 4),
    )


# ── Decisions ───────────────────────────────────────────────────────────────

@app.get("/decisions", tags=["Decisions"])
async def list_decisions(
    classification: Optional[Literal["phishing", "legitimate"]] = Query(None),
    flagged: Optional[bool] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    List logged agent decisions with optional filters.

    Useful for audit trails, review queues, and pulling data for evaluation cycles.
    """
    stmt = select(AgentDecision).order_by(AgentDecision.classified_at.desc())

    if classification:
        stmt = stmt.where(AgentDecision.classification == classification)
    if flagged is not None:
        stmt = stmt.where(AgentDecision.flagged == flagged)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = stmt.limit(limit).offset(offset)
    rows = (await db.execute(stmt)).scalars().all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "decisions": [
            {
                "id": str(r.id),
                "email_subject": r.email_subject,
                "email_sender": r.email_sender,
                "classification": r.classification,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "indicators": r.indicators,
                "flagged": r.flagged,
                "prompt_version": r.prompt_version,
                "model_used": r.model_used,
                "latency_ms": r.latency_ms,
                "classified_at": r.classified_at.isoformat(),
            }
            for r in rows
        ],
    }


@app.post("/decisions/{decision_id}/feedback", tags=["Decisions"])
async def submit_feedback(
    decision_id: str,
    feedback: FeedbackInput,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Submit human reviewer feedback on a decision.

    Feedback is stored and used by the evaluation harness (scripts/evaluate.py)
    to identify misclassifications and optimize prompt strategies across cycles.
    """
    try:
        parsed_id = uuid.UUID(decision_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid decision_id format")

    decision = await db.get(AgentDecision, parsed_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    fb = DecisionFeedback(
        decision_id=parsed_id,
        correct=feedback.correct,
        human_label=feedback.human_label,
        notes=feedback.notes,
    )
    db.add(fb)

    logger.info(
        "feedback_submitted",
        decision_id=decision_id,
        correct=feedback.correct,
        human_label=feedback.human_label,
    )
    return {"status": "feedback_recorded", "decision_id": decision_id}


# ── Metrics ─────────────────────────────────────────────────────────────────

@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics(db: AsyncSession = Depends(get_db)) -> MetricsResponse:
    """
    Aggregate classification metrics and evaluation cycle history.
    """
    total = (await db.execute(select(func.count()).select_from(AgentDecision))).scalar_one()
    phishing_count = (
        await db.execute(
            select(func.count()).select_from(AgentDecision).where(AgentDecision.classification == "phishing")
        )
    ).scalar_one()
    feedback_count = (await db.execute(select(func.count()).select_from(DecisionFeedback))).scalar_one()
    avg_conf = (await db.execute(select(func.avg(AgentDecision.confidence)).select_from(AgentDecision))).scalar_one()

    # Accuracy from feedback
    accuracy: Optional[float] = None
    if feedback_count > 0:
        correct_count = (
            await db.execute(
                select(func.count()).select_from(DecisionFeedback).where(DecisionFeedback.correct == True)  # noqa: E712
            )
        ).scalar_one()
        accuracy = round(correct_count / feedback_count, 4)

    # Evaluation cycles
    cycles_rows = (
        await db.execute(select(EvaluationCycle).order_by(EvaluationCycle.cycle_number))
    ).scalars().all()

    cycle_improvements = [
        {
            "cycle": r.cycle_number,
            "prompt_version_before": r.prompt_version_before,
            "prompt_version_after": r.prompt_version_after,
            "accuracy_before": r.accuracy_before,
            "accuracy_after": r.accuracy_after,
            "precision_improvement": (
                round((r.precision_after or 0) - (r.precision_before or 0), 4)
                if r.precision_before and r.precision_after
                else None
            ),
            "ran_at": r.ran_at.isoformat(),
        }
        for r in cycles_rows
    ]

    return MetricsResponse(
        total_decisions=total,
        phishing_detected=phishing_count,
        legitimate_classified=total - phishing_count,
        feedback_submitted=feedback_count,
        accuracy_from_feedback=accuracy,
        avg_confidence=round(avg_conf or 0.0, 4),
        cycle_improvements=cycle_improvements,
    )
