"""Shared ApplyGuardrail client (Bedrock's data-plane bedrock-runtime
API, not the control-plane bedrock:CreateGuardrail side that provisions
the guardrail resource). Bounded timeout + retry, returns a clean
result -- mode handling (off|shadow|enforce) and event emission are the
CALLER's job (R1/R6 wiring in each repo), since what "enforce" actually
means differs by call site: blocking a request vs. rejecting an answer.

Standalone ApplyGuardrail calls, not attached to InvokeModel -- this
works identically for the Anthropic-API rewrite path in aivalrag-chat
and any Bedrock-backed generation path in AIValRAG-Backend, and avoids
the input-tag requirement that InvokeModel-attached prompt-attack
filtering has.

API shape verified directly against the installed botocore service
model (2026-09), not guessed: qualifiers enum is exactly
{grounding_source, query, guard_content}; action is exactly
{NONE, GUARDRAIL_INTERVENED}; contextualGroundingPolicy filter types are
{GROUNDING, RELEVANCE} with {threshold, score, action, detected}.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_TIMEOUT_MS = 1500
DEFAULT_MAX_RETRIES = 1

Qualifier = Literal["grounding_source", "query", "guard_content"]
Source = Literal["INPUT", "OUTPUT"]


@dataclass
class GuardrailResult:
    action: str  # "NONE" | "GUARDRAIL_INTERVENED" | "ERROR"
    outputs_text: list[str] = field(default_factory=list)
    assessments: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def intervened(self) -> bool:
        return self.action == "GUARDRAIL_INTERVENED"

    @property
    def is_error(self) -> bool:
        return self.action == "ERROR"

    @property
    def blocked_text(self) -> Optional[str]:
        return self.outputs_text[0] if self.outputs_text else None

    def grounding_scores(self) -> dict[str, float]:
        """{"GROUNDING": score, "RELEVANCE": score} from the first
        contextualGroundingPolicy assessment found, if any -- lets a
        caller log/tune thresholds without re-parsing raw assessments."""
        for assessment in self.assessments:
            cgp = assessment.get("contextualGroundingPolicy")
            if cgp:
                return {f.get("type"): f.get("score") for f in cgp.get("filters", []) if f.get("type")}
        return {}


class GuardrailClient:
    def __init__(
        self,
        *,
        guardrail_id: str,
        guardrail_version: str,
        region: str = "us-east-1",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        boto_client=None,
    ):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.max_retries = max_retries
        # botocore's own retry disabled (max_attempts=1) -- this class
        # owns the bounded-retry loop itself so tests can assert on
        # attempt count directly against a Stubber.
        self._client = boto_client or boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                connect_timeout=timeout_ms / 1000,
                read_timeout=timeout_ms / 1000,
                retries={"max_attempts": 1},
            ),
        )

    def apply(self, *, source: Source, text: str, qualifiers: list[Qualifier] | None = None) -> GuardrailResult:
        """Single content block -- covers most calls (R1's input screen,
        R4's notes-field screening). For multiple blocks in one call
        (R6's contextual grounding needs grounding_source + query +
        guard_content as three distinct qualified blocks together), use
        apply_content."""
        return self.apply_content(source=source, parts=[(text, qualifiers)])

    def apply_content(
        self, *, source: Source, parts: list[tuple[str, list[Qualifier] | None]]
    ) -> GuardrailResult:
        content = []
        for text, qualifiers in parts:
            text_block: dict = {"text": text}
            if qualifiers:
                text_block["qualifiers"] = qualifiers
            content.append({"text": text_block})

        last_error: str | None = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self._client.apply_guardrail(
                    guardrailIdentifier=self.guardrail_id,
                    guardrailVersion=self.guardrail_version,
                    source=source,
                    content=content,
                )
                return GuardrailResult(
                    action=resp.get("action", "NONE"),
                    outputs_text=[o.get("text", "") for o in resp.get("outputs", [])],
                    assessments=resp.get("assessments", []),
                )
            except (ClientError, BotoCoreError) as e:
                last_error = str(e)
        return GuardrailResult(action="ERROR", error=last_error)
