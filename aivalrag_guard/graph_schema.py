"""Graph ontology, extraction schema, and community-report schema (plan
Sec 4G, D9/D10). Forward-looking scaffolding: Phase 0 (corrected
2026-09-12) confirmed AIValRAG-Backend has no live knowledge graph today
-- migrations/002_knowledge_graph.sql (kg_nodes/kg_edges) was never
applied to production. This module exists so that IF a real graph is
built later, entity/relationship typing and evidence-span validation are
closed and validated from day one, rather than retrofitted.

Per that same correction, this module deliberately does NOT include a
"switch the graph query path to a read-only principal" step or its
permission-error test (Phase 1's original item 4) -- there is no live
query path to switch. Revisit when a real graph exists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# G6 traversal limits -- defined now, applied when a real graph query path
# exists (Phase 4/6). Unused until then; not a live control.
# ---------------------------------------------------------------------------

GRAPH_MAX_HOPS = 2
GRAPH_MAX_NODES = 200
GRAPH_HUB_DEGREE_CAP = 500
GRAPH_QUERY_TIMEOUT_MS = 2000
COMMUNITY_MIN_SUPPORT = 2

# ---------------------------------------------------------------------------
# G2 -- closed ontology for a future LLM-extraction step. Broader than the
# existing (unapplied) migrations/002_knowledge_graph.sql enum, matching
# the plan's fuller design (subsidiaries, officers, litigation, etc.)
# rather than the narrower single-company bridge ontology that schema
# used. If a real graph is ever built directly on top of 002's schema
# instead, reconcile these two enums explicitly -- don't silently pick one.
# ---------------------------------------------------------------------------

ENTITY_TYPES = frozenset({
    "Company", "Person", "Product", "Segment", "Geography", "Auditor",
    "Regulator", "Litigation", "Security",
})

RELATIONSHIP_TYPES = frozenset({
    "SUBSIDIARY_OF", "CUSTOMER_OF", "SUPPLIER_OF", "ACQUIRED", "DIVESTED",
    "OFFICER_OF", "DIRECTOR_OF", "AUDITOR_OF", "RELATED_PARTY_OF",
    "LITIGATION_WITH", "COMPETITOR_OF", "OPERATES_IN", "REPORTS_SEGMENT",
})

TRUST_TIERS = frozenset({"T0", "T1", "T2", "T3", "T4"})

_MAX_ENTITY_NAME_LEN = 200
_MAX_EVIDENCE_SPAN_LEN = 600
_MAX_DESCRIPTION_LEN = 300


class GraphSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceSpan:
    chunk_id: str
    start: int
    end: int


@dataclass(frozen=True)
class ExtractedEntity:
    type: str
    name: str


@dataclass(frozen=True)
class ExtractedRelationship:
    type: str
    source_entity: str
    target_entity: str
    evidence: EvidenceSpan
    description: str | None = None


def validate_entity(entity: ExtractedEntity) -> None:
    if entity.type not in ENTITY_TYPES:
        raise GraphSchemaError(f"ontology_violation: unknown entity type {entity.type!r}")
    if not entity.name or "\n" in entity.name:
        raise GraphSchemaError("evidence_span_invalid: entity name is empty or contains a newline")
    if len(entity.name) > _MAX_ENTITY_NAME_LEN:
        raise GraphSchemaError(f"entity name exceeds {_MAX_ENTITY_NAME_LEN} characters")


def validate_evidence_span(evidence: EvidenceSpan, chunk_text: str) -> None:
    """The span must actually exist in the named chunk and be short --
    this is what lets a reviewer trust that an extracted claim is
    grounded in real text, not fabricated by the extractor."""
    if evidence.start < 0 or evidence.end <= evidence.start or evidence.end > len(chunk_text):
        raise GraphSchemaError(f"evidence_span_invalid: offsets [{evidence.start}:{evidence.end}] out of bounds")
    span_len = evidence.end - evidence.start
    if span_len > _MAX_EVIDENCE_SPAN_LEN:
        raise GraphSchemaError(f"evidence_span_invalid: span length {span_len} exceeds {_MAX_EVIDENCE_SPAN_LEN}")


def validate_relationship(rel: ExtractedRelationship, *, chunk_text: str) -> None:
    if rel.type not in RELATIONSHIP_TYPES:
        raise GraphSchemaError(f"ontology_violation: unknown relationship type {rel.type!r}")
    validate_evidence_span(rel.evidence, chunk_text)
    if rel.description is not None and len(rel.description) > _MAX_DESCRIPTION_LEN:
        raise GraphSchemaError(f"description exceeds {_MAX_DESCRIPTION_LEN} characters")


# ---------------------------------------------------------------------------
# G5 -- community report schema
# ---------------------------------------------------------------------------

_MAX_TITLE_LEN = 120
_MAX_FINDING_TEXT_LEN = 400


@dataclass(frozen=True)
class CommunityFinding:
    text: str
    edge_ids: tuple[str, ...] = ()
    node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommunityReport:
    id: str
    title: str
    findings: tuple[CommunityFinding, ...]


def validate_community_report(report: CommunityReport) -> list[CommunityFinding]:
    """Returns only the findings that cite at least one edge or node ID
    from this community's input set -- uncited findings are dropped, not
    errored, matching the plan's community_finding_rejected behavior
    (drop-and-log, not fail-the-whole-report)."""
    if len(report.title) > _MAX_TITLE_LEN:
        raise GraphSchemaError(f"community report title exceeds {_MAX_TITLE_LEN} characters")
    kept = []
    for finding in report.findings:
        if len(finding.text) > _MAX_FINDING_TEXT_LEN:
            continue
        if not finding.edge_ids and not finding.node_ids:
            continue
        kept.append(finding)
    return kept


# ---------------------------------------------------------------------------
# Confusable-name detection (G3) -- UTS #39-style skeleton via NFKC + a
# small explicit homoglyph table. Not a full UTS #39 implementation;
# documented as best-effort, same posture as normalize.py's PDF/CSS gaps.
# ---------------------------------------------------------------------------

_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",  # Cyrillic look-alikes
    "ı": "i", "ⅼ": "l",
    "l": "i",  # ASCII lowercase L vs capital I -- the classic "Apple lnc." trick (PI-25)
}
_HOMOGLYPH_RE = re.compile("|".join(re.escape(k) for k in _HOMOGLYPHS))


def confusable_skeleton(name: str) -> str:
    folded = _HOMOGLYPH_RE.sub(lambda m: _HOMOGLYPHS[m.group(0)], name)
    return folded.casefold().strip()


def is_confusable_with(candidate: str, registered_name: str) -> bool:
    """True if candidate's skeleton matches the registered name's
    skeleton but the raw strings differ -- e.g. "Apple lnc." (lowercase
    L) against "Apple Inc.". A match here should block a merge and flag
    the node `suspect`, never merge automatically (G3)."""
    return candidate != registered_name and confusable_skeleton(candidate) == confusable_skeleton(registered_name)
