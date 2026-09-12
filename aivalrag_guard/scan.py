"""I2 -- deterministic, weighted heuristic scan. Runs on ALREADY
NORMALIZED text (normalize.py) -- scanning before normalization lets an
attacker evade every pattern here with a zero-width character or a
fullwidth-ASCII substitution. Not a blocker alone; a signal that
combines with the classifier scan (I3, Phase 3) into a risk tier.

Rules are table-driven with stable IDs so a corpus/test file can name one
directly, and so tuning a weight in Phase 7 never means renaming
anything a test depends on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SUSPECT_THRESHOLD = 2.0


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    weight: float
    description: str


def _rule(id_: str, pattern: str, weight: float, description: str, flags: int = re.IGNORECASE) -> Rule:
    return Rule(id=id_, pattern=re.compile(pattern, flags), weight=weight, description=description)


RULES: tuple[Rule, ...] = (
    # -- Instructions addressed to a model --------------------------------
    _rule("instr_ignore_previous", r"ignore\s+(all|any|the)?\s*(previous|prior|above)\s+(instructions|context)",
          3.0, "attempt to discard prior instructions"),
    _rule("instr_you_are_now", r"\byou\s+are\s+now\b", 2.0, "role-reassignment attempt"),
    _rule("instr_system_prompt", r"\bsystem\s+prompt\b", 2.0, "attempts to reference/leak the system prompt"),
    _rule("instr_as_an_ai", r"\bas\s+an\s+(AI|assistant|language\s+model)\b", 1.5, "AI self-reference framing"),
    _rule("instr_hide_from_user", r"do\s+not\s+(tell|inform|mention\s+to)\s+the\s+user", 3.0,
          "instructs the model to conceal something from the user"),
    _rule("instr_respond_only", r"\brespond\s+only\b", 1.5, "attempts to constrain output format/content"),
    _rule("instr_from_now_on", r"\bfrom\s+now\s+on\b", 1.5, "attempts a persistent behavior change"),

    # -- Role and control tokens -------------------------------------------
    _rule("role_angle_tokens", r"<\|[^|]{1,40}\|>", 3.0, "chat-template-style role/control token"),
    _rule("role_line_start", r"(?m)^\s*(system|assistant|human)\s*:", 2.0, "role-label line, chat-transcript style"),
    _rule("role_inst_bracket", r"\[INST\]", 2.5, "Llama-style instruction bracket token"),

    # -- Memory-targeting language ------------------------------------------
    _rule("mem_summarize_trigger", r"(when|while)\s+summariz\w*", 2.0, "targets the summarization step specifically"),
    _rule("mem_remember_that", r"\bremember\s+that\b", 1.5, "attempts to plant persistent memory"),
    _rule("mem_store_this", r"\bstore\s+(this|the\s+following)\b", 1.5, "attempts to plant persistent memory"),
    _rule("mem_future_responses", r"\bin\s+(future|later)\s+(answers|responses)\b", 2.0,
          "targets behavior in later turns"),

    # -- Exfiltration shapes -------------------------------------------------
    _rule("exfil_markdown_image", r"!\[[^\]]{0,200}\]\([^)]{1,2000}\)", 2.5,
          "markdown image syntax -- rendered images can exfiltrate via query strings"),
    _rule("exfil_long_query_url", r"https?://\S+?\?\S{50,}", 2.0, "URL with an unusually long query string"),
    _rule("exfil_base64_run", r"[A-Za-z0-9+/]{200,}={0,2}", 1.5, "long base64-looking run in narrative text"),

    # -- Tag-like sequences matching the prompt builder's own tag names ----
    # Keep in sync with prompt_builder.py's _TAG_NAMES -- a forged closing
    # tag is exactly how an attacker would try to escape a data block.
    _rule("forged_boundary_tag", r"</?(?:documents?|graph_context|session_notes|question)_[0-9a-f]{4,}\s*>",
          3.0, "text mimics the prompt builder's own delimiter tags"),
)

_RULES_BY_ID = {r.id: r for r in RULES}

# Built from the plan's own benign example (PI-22) plus the shape it
# represents (second-person imperatives that are routine in MD&A prose).
# Grow this from a real benign-sample false-positive run in Phase 7, not
# from intuition -- see plan Sec 4/I2.
BENIGN_ALLOWLIST: tuple[re.Pattern, ...] = (
    re.compile(r"you\s+should\s+read\s+the\s+following\s+discussion", re.IGNORECASE),
)


@dataclass
class ScanResult:
    score: float
    rule_ids: list[str] = field(default_factory=list)
    suppressed_rule_ids: list[str] = field(default_factory=list)

    @property
    def is_suspect(self) -> bool:
        return self.score >= SUSPECT_THRESHOLD


def _is_allowlisted(text: str, start: int, end: int, window: int = 80) -> bool:
    lo, hi = max(0, start - window), min(len(text), end + window)
    context = text[lo:hi]
    return any(p.search(context) for p in BENIGN_ALLOWLIST)


def scan(text: str) -> ScanResult:
    """Scan already-normalized text. Each rule contributes its weight at
    most once, even if its pattern matches multiple times, so score
    reflects distinct techniques detected, not repetition count."""
    result = ScanResult(score=0.0)
    for rule in RULES:
        match = rule.pattern.search(text)
        if not match:
            continue
        if _is_allowlisted(text, match.start(), match.end()):
            result.suppressed_rule_ids.append(rule.id)
            continue
        result.score += rule.weight
        result.rule_ids.append(rule.id)
    return result


def rule_by_id(rule_id: str) -> Rule:
    return _RULES_BY_ID[rule_id]
