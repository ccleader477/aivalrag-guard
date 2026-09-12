import pytest

from aivalrag_guard.graph_schema import (
    CommunityFinding,
    CommunityReport,
    EvidenceSpan,
    ExtractedEntity,
    ExtractedRelationship,
    GraphSchemaError,
    confusable_skeleton,
    is_confusable_with,
    validate_community_report,
    validate_entity,
    validate_evidence_span,
    validate_relationship,
)


def test_valid_entity_passes():
    validate_entity(ExtractedEntity(type="Company", name="T-Mobile US"))


def test_entity_unknown_type_rejected():
    with pytest.raises(GraphSchemaError, match="ontology_violation"):
        validate_entity(ExtractedEntity(type="INSTRUCTS_ASSISTANT", name="x"))


def test_entity_empty_name_rejected():
    with pytest.raises(GraphSchemaError):
        validate_entity(ExtractedEntity(type="Company", name=""))


def test_entity_name_with_newline_rejected():
    with pytest.raises(GraphSchemaError):
        validate_entity(ExtractedEntity(type="Company", name="Foo\nBar"))


def test_entity_name_too_long_rejected():
    with pytest.raises(GraphSchemaError):
        validate_entity(ExtractedEntity(type="Company", name="A" * 300))


def test_valid_evidence_span_passes():
    chunk = "Company B is our exclusive supplier and a related party."
    span = EvidenceSpan(chunk_id="c1", start=0, end=len("Company B is our exclusive supplier"))
    validate_evidence_span(span, chunk)


def test_evidence_span_out_of_bounds_rejected():
    chunk = "short text"
    span = EvidenceSpan(chunk_id="c1", start=0, end=9999)
    with pytest.raises(GraphSchemaError, match="evidence_span_invalid"):
        validate_evidence_span(span, chunk)


def test_evidence_span_negative_start_rejected():
    chunk = "some text here"
    with pytest.raises(GraphSchemaError):
        validate_evidence_span(EvidenceSpan(chunk_id="c1", start=-1, end=5), chunk)


def test_evidence_span_too_long_rejected():
    chunk = "x" * 1000
    with pytest.raises(GraphSchemaError):
        validate_evidence_span(EvidenceSpan(chunk_id="c1", start=0, end=700), chunk)


def test_valid_relationship_passes():
    chunk = "Company B is our exclusive supplier and a related party."
    rel = ExtractedRelationship(
        type="SUPPLIER_OF", source_entity="e1", target_entity="e2",
        evidence=EvidenceSpan(chunk_id="c1", start=0, end=36),
        description="Exclusive supplier relationship",
    )
    validate_relationship(rel, chunk_text=chunk)


def test_relationship_unknown_type_rejected():
    chunk = "text"
    rel = ExtractedRelationship(
        type="INSTRUCTS_ASSISTANT", source_entity="e1", target_entity="e2",
        evidence=EvidenceSpan(chunk_id="c1", start=0, end=4),
    )
    with pytest.raises(GraphSchemaError, match="ontology_violation"):
        validate_relationship(rel, chunk_text=chunk)


def test_relationship_free_text_type_rejected():
    # PI-32-style: a made-up/free-text relationship type must be rejected
    # by the closed ontology, not accepted because it "sounds plausible."
    chunk = "text"
    rel = ExtractedRelationship(
        type="SEEMS_RELATED_TO", source_entity="e1", target_entity="e2",
        evidence=EvidenceSpan(chunk_id="c1", start=0, end=4),
    )
    with pytest.raises(GraphSchemaError):
        validate_relationship(rel, chunk_text=chunk)


def test_relationship_description_too_long_rejected():
    chunk = "x" * 400
    rel = ExtractedRelationship(
        type="CUSTOMER_OF", source_entity="e1", target_entity="e2",
        evidence=EvidenceSpan(chunk_id="c1", start=0, end=10),
        description="d" * 400,
    )
    with pytest.raises(GraphSchemaError):
        validate_relationship(rel, chunk_text=chunk)


# -- Community reports (G5) --

def test_community_report_keeps_only_cited_findings():
    report = CommunityReport(
        id="cr1", title="Telecom sector overview",
        findings=(
            CommunityFinding(text="TMUS and VZ compete in wireless.", edge_ids=("e1",)),
            CommunityFinding(text="Uncited speculative claim with no support.", edge_ids=(), node_ids=()),
        ),
    )
    kept = validate_community_report(report)
    assert len(kept) == 1
    assert kept[0].edge_ids == ("e1",)


def test_community_report_title_too_long_rejected():
    report = CommunityReport(id="cr1", title="T" * 200, findings=())
    with pytest.raises(GraphSchemaError):
        validate_community_report(report)


def test_community_finding_text_too_long_dropped_not_errored():
    report = CommunityReport(
        id="cr1", title="ok title",
        findings=(CommunityFinding(text="x" * 500, edge_ids=("e1",)),),
    )
    kept = validate_community_report(report)
    assert kept == []


# -- Confusable entity detection (G3) --

def test_confusable_lowercase_l_detected():
    assert is_confusable_with("Apple lnc.", "Apple Inc.") is True


def test_identical_names_not_confusable():
    assert is_confusable_with("Apple Inc.", "Apple Inc.") is False


def test_genuinely_different_names_not_confusable():
    assert is_confusable_with("Microsoft Corp.", "Apple Inc.") is False


def test_cyrillic_homoglyph_detected():
    spoofed = "Netflix".replace("e", "е")  # Cyrillic е (U+0435), not Latin e
    assert confusable_skeleton(spoofed) == confusable_skeleton("Netflix")
    assert is_confusable_with(spoofed, "Netflix") is True
