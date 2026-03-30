"""Unit tests for the phishing detection agent pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent import PhishingDetectionAgent, hash_body


class TestHashBody:
    def test_returns_hex_string(self):
        result = hash_body("hello world")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_input_same_hash(self):
        assert hash_body("test email") == hash_body("test email")

    def test_different_inputs_different_hashes(self):
        assert hash_body("phishing email") != hash_body("legitimate email")


class TestResponseParsing:
    def setup_method(self):
        self.agent = PhishingDetectionAgent.__new__(PhishingDetectionAgent)

    def test_parses_valid_json(self):
        raw = json.dumps({
            "classification": "phishing",
            "confidence": 0.95,
            "reasoning": "Suspicious sender domain",
            "indicators": ["lookalike_domain", "urgency_language"],
        })
        result = self.agent._parse_response(raw)
        assert result["classification"] == "phishing"
        assert result["confidence"] == 0.95
        assert "lookalike_domain" in result["indicators"]

    def test_strips_markdown_fences(self):
        raw = "```json\n{\"classification\": \"legitimate\", \"confidence\": 0.1, \"reasoning\": \"ok\", \"indicators\": []}\n```"
        result = self.agent._parse_response(raw)
        assert result["classification"] == "legitimate"

    def test_fallback_on_invalid_json(self):
        result = self.agent._parse_response("not valid json at all")
        assert result["classification"] == "legitimate"
        assert result["confidence"] == 0.5

    def test_clamps_confidence_above_one(self):
        raw = json.dumps({"classification": "phishing", "confidence": 1.5, "reasoning": "r", "indicators": []})
        result = self.agent._parse_response(raw)
        assert result["confidence"] == 1.0

    def test_clamps_confidence_below_zero(self):
        raw = json.dumps({"classification": "phishing", "confidence": -0.3, "reasoning": "r", "indicators": []})
        result = self.agent._parse_response(raw)
        assert result["confidence"] == 0.0

    def test_handles_unknown_classification(self):
        raw = json.dumps({"classification": "unknown", "confidence": 0.5, "reasoning": "r", "indicators": []})
        result = self.agent._parse_response(raw)
        assert result["classification"] == "legitimate"

    def test_handles_missing_indicators(self):
        raw = json.dumps({"classification": "phishing", "confidence": 0.9, "reasoning": "r"})
        result = self.agent._parse_response(raw)
        assert result["indicators"] == []


@pytest.mark.asyncio
class TestAgentClassify:
    @patch("app.agent.AsyncOpenAI")
    async def test_classify_phishing(self, MockOpenAI):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "classification": "phishing",
            "confidence": 0.97,
            "reasoning": "Lookalike domain detected",
            "indicators": ["lookalike_domain", "urgency_language"],
        })

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        agent = PhishingDetectionAgent()
        result, latency = await agent.classify(
            subject="Urgent: Verify your PayPal account",
            body="Click here immediately to restore access...",
            sender="security@paypa1.com",
        )

        assert result.classification == "phishing"
        assert result.confidence == 0.97
        assert result.flagged is True
        assert latency > 0

    @patch("app.agent.AsyncOpenAI")
    async def test_classify_legitimate(self, MockOpenAI):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "classification": "legitimate",
            "confidence": 0.05,
            "reasoning": "Normal business communication",
            "indicators": [],
        })

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        agent = PhishingDetectionAgent()
        result, _ = await agent.classify(
            subject="Q1 Planning Meeting Notes",
            body="Hi team, attached are the notes from today's meeting...",
            sender="sarah@company.com",
        )

        assert result.classification == "legitimate"
        assert result.flagged is False
