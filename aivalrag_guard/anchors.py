"""R3 -- typed entity anchors. No free-text fields: every anchor is
{type, value, source, turn_id}. Anchors narrow retrieval; they never set
or widen tenant scope, and they are never extracted from assistant prose
(source is always user_turn or retrieval_metadata -- see D3).

Phase 1 scope is format/enum validation only. Existence checks (does
this CIK/accession/entity actually exist, and is it visible to this
tenant) need a DB/corpus lookup and are added in Phase 2 when this is
wired into aivalrag-chat -- see `validate_anchor`'s docstring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MAX_ANCHORS_PER_SESSION = 20

AnchorType = Literal["cik", "accession", "ticker", "fiscal_period", "metric", "form_type", "entity_id"]
AnchorSource = Literal["user_turn", "retrieval_metadata"]

# Sourced from AIValRAG-Backend's actual seed data (Phase 0 finding, item
# 10/11): migrations/002_knowledge_graph.sql's concept nodes plus
# supabase/migrations/20260810_kg_formulas.sql's result_concept values.
# Extend this list from that source of truth, not by guessing new metrics.
METRIC_CATALOG = frozenset({
    "net_income", "operating_income", "adjusted_ebitda", "core_adjusted_ebitda",
    "interest_expense", "income_tax_expense", "depreciation_amortization",
    "stock_based_compensation", "merger_related_costs", "restructuring_charges",
    "legal_settlements", "impairment_charges", "asset_disposals",
    "other_income_expense", "spectrum_amortization", "operating_lease_expense",
    "tax_adjustment", "cfo", "cfi", "cff", "ending_cash", "gross_profit", "ebit",
    "ebt", "assets", "ending_retained_earnings", "ending_ppe", "ending_debt",
})

FORM_TYPE_CATALOG = frozenset({
    "10-K", "10-Q", "8-K", "DEF 14A", "S-1", "S-3", "S-4", "20-F", "6-K",
    "424B", "424B3", "424B4", "13F", "SC 13D", "SC 13G", "3", "4", "5",
})

_PATTERNS: dict[str, re.Pattern] = {
    "cik": re.compile(r"^\d{10}$"),
    "accession": re.compile(r"^\d{10}-\d{2}-\d{6}$"),
    "ticker": re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$"),
    "fiscal_period": re.compile(r"^(FY|Q[1-4])\d{4}$"),
    # Graph node ID format -- UUID, matching kg_nodes.id's actual column
    # type (migrations/002_knowledge_graph.sql:49).
    "entity_id": re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    # AIValRAG-Backend's own resolved filing identifier -- a UUID (same
    # shape as entity_id, matching filings.id's column type in
    # AIValRAG-Backend/migrations/001_create_tables.sql:13), but kept as
    # a distinct type since it names a different thing: a resolved
    # filing document, not a knowledge-graph node. This system has no
    # structured EDGAR ingestion (no real accession numbers anywhere,
    # per PROMPT_INJECTION_PLAN.md Phase 0 item 11), so document_id is
    # the actual filing-scoping anchor in practice, not "accession".
    "document_id": re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
}

_ENUM_TYPES: dict[str, frozenset] = {
    "metric": METRIC_CATALOG,
    "form_type": FORM_TYPE_CATALOG,
}

VALID_TYPES = frozenset(_PATTERNS) | frozenset(_ENUM_TYPES)


@dataclass(frozen=True)
class Anchor:
    type: str
    value: str
    source: AnchorSource
    turn_id: str


class AnchorValidationError(ValueError):
    pass


def validate_anchor_format(anchor_type: str, value: str) -> None:
    """Format/enum check only -- raises AnchorValidationError on failure.
    Does NOT check existence (a well-formed CIK that doesn't correspond
    to any real company still passes here); Phase 2 adds an existence
    check against the tenant's entity graph/corpus on top of this."""
    if anchor_type not in VALID_TYPES:
        raise AnchorValidationError(f"unknown anchor type: {anchor_type!r}")
    if anchor_type in _PATTERNS:
        if not _PATTERNS[anchor_type].match(value):
            raise AnchorValidationError(f"{anchor_type} value {value!r} does not match required format")
    else:
        if value not in _ENUM_TYPES[anchor_type]:
            raise AnchorValidationError(f"{anchor_type} value {value!r} is not in the catalog")


def make_anchor(anchor_type: str, value: str, *, source: AnchorSource, turn_id: str) -> Anchor:
    """Construct a validated Anchor. `source` must be user_turn or
    retrieval_metadata -- there is no "assistant_turn" option by design
    (D3): anchors are never extracted from model-generated prose."""
    if source not in ("user_turn", "retrieval_metadata"):
        raise AnchorValidationError(f"invalid anchor source: {source!r}")
    validate_anchor_format(anchor_type, value)
    return Anchor(type=anchor_type, value=value, source=source, turn_id=turn_id)


def evict_lru(anchors: list[Anchor], *, max_anchors: int = MAX_ANCHORS_PER_SESSION) -> list[Anchor]:
    """Anchors list is assumed oldest-first; drop from the front until at
    or under the cap. Pure function -- caller owns persistence."""
    if len(anchors) <= max_anchors:
        return list(anchors)
    return list(anchors[len(anchors) - max_anchors:])
