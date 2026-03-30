"""
LLM Agent — Phishing Detection Pipeline

Architecture:
  Input (email) → Prompt Construction → OpenAI GPT → Response Parsing → Structured Output

The agent uses a chain-of-thought prompt that instructs the model to:
  1. Identify phishing indicators step by step
  2. Assign a confidence score
  3. Return a structured JSON decision
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import ClassificationResult

logger = structlog.get_logger(__name__)

# ── Prompt templates (versioned) ────────────────────────────────────────────

SYSTEM_PROMPT_V1 = """You are a cybersecurity AI agent specializing in phishing email detection.

Your task is to analyze an email and classify it as either "phishing" or "legitimate".

Follow this reasoning process step by step:
1. Examine the sender address for lookalike domains, typosquatting, or free email providers impersonating organizations
2. Analyze subject line for urgency, fear, or reward-baiting language
3. Scan the body for suspicious links, credential harvesting patterns, or social engineering tactics
4. Check for grammatical inconsistencies or generic salutations that indicate mass-sent phishing
5. Evaluate the overall threat level

After your analysis, return a JSON object in this EXACT format (no markdown, raw JSON only):
{
  "classification": "phishing" | "legitimate",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<concise explanation of your decision>",
  "indicators": [<list of detected indicator tags, e.g. "lookalike_domain", "urgency_language", "suspicious_link", "credential_harvesting", "generic_salutation", "grammatical_errors">]
}"""

SYSTEM_PROMPT_V2 = """You are an expert cybersecurity AI agent specializing in phishing email detection with access to threat intelligence patterns.

## Task
Classify the provided email as "phishing" or "legitimate" using structured threat analysis.

## Analysis Framework (apply in order)
1. **Sender Analysis** — lookalike domains, typosquatting (paypa1.com vs paypal.com), free email providers masquerading as corporations
2. **Subject Analysis** — urgency cues ("urgent", "immediately", "account suspended"), reward-bait, fear triggers
3. **Body Analysis** — suspicious URLs, credential-request patterns, social engineering phrases, spoofed branding
4. **Linguistic Analysis** — grammar errors, generic salutations ("Dear Customer"), inconsistent tone
5. **Context Coherence** — does the sender, subject, and body form a coherent, legitimate business communication?

## Output Format (raw JSON only — no markdown)
{
  "classification": "phishing" | "legitimate",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<step-by-step explanation referencing specific evidence>",
  "indicators": [<zero or more of: "lookalike_domain", "typosquatting", "urgency_language", "fear_trigger", "reward_bait", "suspicious_link", "credential_harvesting", "spoofed_branding", "generic_salutation", "grammatical_errors", "incoherent_context">]
}"""

PROMPT_VERSIONS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
}


def _get_system_prompt(version: str = "v1") -> str:
    return PROMPT_VERSIONS.get(version, SYSTEM_PROMPT_V1)


def _build_user_message(subject: str, body: str, sender: str) -> str:
    return f"""Please analyze the following email:

SENDER: {sender}
SUBJECT: {subject}
BODY:
{body}"""


def hash_body(body: str) -> str:
    """SHA-256 hash of the email body — stored instead of raw content for privacy."""
    return hashlib.sha256(body.encode()).hexdigest()


class PhishingDetectionAgent:
    """
    LLM-powered phishing detection agent.

    Uses OpenAI's GPT API with a structured chain-of-thought prompt to classify
    emails and extract reasoning traces for audit and evaluation purposes.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def classify(
        self,
        subject: str,
        body: str,
        sender: str,
        prompt_version: Optional[str] = None,
    ) -> tuple[ClassificationResult, float]:
        """
        Classify a single email.

        Returns:
            Tuple of (ClassificationResult, latency_ms)
        """
        version = prompt_version or settings.PROMPT_VERSION
        log = logger.bind(sender=sender, subject=subject[:50], prompt_version=version)

        start = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _get_system_prompt(version)},
                    {"role": "user", "content": _build_user_message(subject, body, sender)},
                ],
                max_tokens=settings.MAX_TOKENS,
                temperature=0.1,  # Low temperature for consistent, deterministic outputs
                response_format={"type": "text"},
            )

            latency_ms = (time.perf_counter() - start) * 1000
            raw_content = response.choices[0].message.content or "{}"

            parsed = self._parse_response(raw_content)

            result = ClassificationResult(
                classification=parsed["classification"],
                confidence=float(parsed["confidence"]),
                reasoning=parsed["reasoning"],
                indicators=parsed.get("indicators", []),
                flagged=float(parsed["confidence"]) >= settings.CONFIDENCE_THRESHOLD
                and parsed["classification"] == "phishing",
            )

            log.info(
                "email_classified",
                classification=result.classification,
                confidence=result.confidence,
                latency_ms=round(latency_ms, 2),
            )

            return result, latency_ms

        except Exception as exc:
            log.error("classification_failed", error=str(exc))
            raise

    def _parse_response(self, raw: str) -> dict:
        """Parse and validate the LLM's JSON response."""
        raw = raw.strip()

        # Strip markdown code fences if the model wraps its output
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("json_parse_error", raw=raw[:200], error=str(exc))
            # Fallback: treat as uncertain legitimate to avoid false positives
            return {
                "classification": "legitimate",
                "confidence": 0.5,
                "reasoning": f"Failed to parse LLM response. Raw output: {raw[:200]}",
                "indicators": [],
            }

        # Validate and sanitize fields
        classification = parsed.get("classification", "legitimate")
        if classification not in ("phishing", "legitimate"):
            classification = "legitimate"

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "classification": classification,
            "confidence": confidence,
            "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
            "indicators": [str(i) for i in parsed.get("indicators", [])],
        }


# Module-level singleton
agent = PhishingDetectionAgent()
