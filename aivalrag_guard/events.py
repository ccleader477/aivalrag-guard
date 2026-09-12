"""R9 -- security events. Structured, no raw user/document text ever --
IDs, rule IDs, scores, lengths, and hashes only, so events are safe to
ship to CloudWatch/logs without becoming a second copy of sensitive data."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

EVENT_TYPES = frozenset({
    "input_blocked", "output_blocked", "grounding_low", "grounding_truncated",
    "rewrite_invalid", "rewrite_drift", "anchor_rejected", "notes_field_rejected",
    "chunk_quarantined", "chunk_suspect_retrieved", "canary_leak", "citation_mismatch",
    "guardrail_error", "extraction_rejected", "ontology_violation", "evidence_span_invalid",
    "er_merge_blocked", "confusable_entity", "lineage_cascade", "community_finding_rejected",
    "graph_query_rejected", "traversal_capped", "claim_unsupported", "xbrl_mismatch",
    "graph_anomaly",
})

MODES = frozenset({"off", "shadow", "enforce"})

_logger = logging.getLogger("aivalrag_guard.events")


def _hash(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_event(
    event_type: str,
    *,
    mode: str,
    request_id: str | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
    rule_ids: list[str] | None = None,
    scores: dict[str, float] | None = None,
    content: str | None = None,
    content_length: int | None = None,
) -> dict[str, Any]:
    """Pure function so tests can assert on the event shape without
    touching the logger. `content`, if given, is hashed and measured --
    never stored raw. `content_length` lets a caller report a length
    without handing over the content itself (e.g. after it's already
    gone out of scope)."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event_type: {event_type!r}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")

    length = content_length if content_length is not None else (len(content) if content is not None else None)

    return {
        "event_type": event_type,
        "mode": mode,
        "request_id": request_id,
        "tenant_id": tenant_id,
        "session_id_hash": _hash(session_id),
        "rule_ids": list(rule_ids) if rule_ids else [],
        "scores": dict(scores) if scores else {},
        "content_length": length,
        "content_sha256": _hash(content),
        "ts": time.time(),
    }


def log_event(event_type: str, *, mode: str, **kwargs: Any) -> dict[str, Any]:
    """Build the event and emit it as one JSON line. Returns the event
    dict (useful for tests and for callers that also want to act on it,
    e.g. incrementing a metric)."""
    event = build_event(event_type, mode=mode, **kwargs)
    _logger.info(json.dumps(event, sort_keys=True))
    return event
