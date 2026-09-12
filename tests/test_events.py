import pytest

from aivalrag_guard.events import build_event, log_event


def test_valid_event_shape():
    event = build_event(
        "input_blocked", mode="enforce", request_id="req-1", tenant_id="t1",
        session_id="sess-abc", rule_ids=["instr_ignore_previous"], scores={"i2": 3.0},
        content="the raw blocked message",
    )
    assert event["event_type"] == "input_blocked"
    assert event["mode"] == "enforce"
    assert event["request_id"] == "req-1"
    assert event["rule_ids"] == ["instr_ignore_previous"]
    assert event["scores"] == {"i2": 3.0}
    assert event["content_length"] == len("the raw blocked message")


def test_no_raw_content_in_event():
    event = build_event("output_blocked", mode="enforce", content="sensitive user text here")
    serialized = str(event)
    assert "sensitive user text here" not in serialized
    assert event["content_sha256"] is not None
    assert len(event["content_sha256"]) == 64  # sha256 hex digest length


def test_session_id_is_hashed_not_stored_raw():
    event = build_event("canary_leak", mode="enforce", session_id="real-session-id-12345")
    assert "real-session-id-12345" not in str(event)
    assert event["session_id_hash"] is not None


def test_content_length_can_be_reported_without_content():
    event = build_event("grounding_truncated", mode="shadow", content_length=12345)
    assert event["content_length"] == 12345
    assert event["content_sha256"] is None


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError):
        build_event("made_up_event", mode="enforce")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_event("input_blocked", mode="turbo")


def test_defaults_are_none_or_empty():
    event = build_event("guardrail_error", mode="shadow")
    assert event["request_id"] is None
    assert event["tenant_id"] is None
    assert event["session_id_hash"] is None
    assert event["rule_ids"] == []
    assert event["scores"] == {}
    assert event["content_length"] is None
    assert event["content_sha256"] is None


def test_log_event_returns_same_shape_as_build_event(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="aivalrag_guard.events")
    event = log_event("rewrite_drift", mode="enforce", request_id="req-9")
    assert event["event_type"] == "rewrite_drift"
    assert any("rewrite_drift" in r.message for r in caplog.records)


def test_log_event_emits_valid_json(caplog):
    import json
    import logging
    caplog.set_level(logging.INFO, logger="aivalrag_guard.events")
    log_event("chunk_quarantined", mode="enforce", rule_ids=["r1"])
    line = caplog.records[-1].message
    parsed = json.loads(line)
    assert parsed["event_type"] == "chunk_quarantined"
