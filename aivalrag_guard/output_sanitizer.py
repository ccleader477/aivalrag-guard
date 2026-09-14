"""R7 -- deterministic output sanitizer. Runs on the model's OUTPUT
(the answer text), not input -- different rules than normalize.py, which
runs on untrusted INPUT. Low false-positive risk by design, so this runs
in `enforce` from day one per the plan (unlike the probabilistic
guardrail checks in R1/R6).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

SANITIZER_VERSION = "1.0.0"

DEFAULT_LINK_ALLOWLIST = frozenset({"sec.gov", "www.sec.gov"})
DEFAULT_IMAGE_ALLOWLIST: frozenset[str] = frozenset()  # empty by default, per spec

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_BARE_URL_RE = re.compile(r"(?<![(\[])\bhttps?://[^\s)\]\"'<>]+")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?>")


@dataclass
class SanitizeResult:
    text: str
    events: list[str] = field(default_factory=list)
    removed_image_count: int = 0
    delinkified_hosts: list[str] = field(default_factory=list)
    html_stripped: bool = False
    truncated: bool = False


def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_allowed(hostname: str, allowlist: frozenset[str]) -> bool:
    return hostname in {h.lower() for h in allowlist}


def sanitize_output(
    text: str,
    *,
    link_allowlist: frozenset[str] = DEFAULT_LINK_ALLOWLIST,
    image_allowlist: frozenset[str] = DEFAULT_IMAGE_ALLOWLIST,
    max_length: int | None = None,
) -> SanitizeResult:
    """Order matters: images before generic links (an image is also a
    link-shaped pattern), raw HTML strip last (so a `<img>` written as
    literal HTML doesn't survive by dodging the markdown-image pattern)."""
    result = SanitizeResult(text=text)

    def _replace_image(m: re.Match) -> str:
        host = _hostname(m.group(2))
        if _is_allowed(host, image_allowlist):
            return m.group(0)
        result.removed_image_count += 1
        result.events.append("image_removed")
        return ""

    text = _MD_IMAGE_RE.sub(_replace_image, text)

    def _replace_link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        host = _hostname(url)
        if _is_allowed(host, link_allowlist):
            return m.group(0)
        result.delinkified_hosts.append(host)
        result.events.append("link_delinkified")
        return f"{label} (link removed: {host})" if host else f"{label} (link removed)"

    text = _MD_LINK_RE.sub(_replace_link, text)

    def _replace_bare_url(m: re.Match) -> str:
        url = m.group(0)
        host = _hostname(url)
        if _is_allowed(host, link_allowlist):
            return url
        result.delinkified_hosts.append(host)
        result.events.append("link_delinkified")
        return f"(link removed: {host})" if host else "(link removed)"

    text = _BARE_URL_RE.sub(_replace_bare_url, text)

    if _HTML_TAG_RE.search(text):
        result.html_stripped = True
        result.events.append("html_stripped")
        text = _HTML_TAG_RE.sub("", text)

    if max_length is not None and len(text) > max_length:
        text = text[:max_length]
        result.truncated = True
        result.events.append("length_truncated")

    result.text = text
    return result


@dataclass
class CitationCheckResult:
    text: str
    removed_citation_ids: list[str] = field(default_factory=list)


def validate_citations(text: str, valid_ids: frozenset[str], *, citation_pattern: re.Pattern) -> CitationCheckResult:
    """Generic by design: this repo pair doesn't (yet) have one fixed
    inline-citation syntax in generated answer text, so the caller
    supplies the pattern that extracts a citation token as capture group
    1 (e.g. r'\\[cite:([\\w-]+)\\]'). Any match whose captured ID isn't in
    `valid_ids` (chunk_id/edge_id/accession actually present in this
    request's context) is removed and reported -- never silently kept."""
    result = CitationCheckResult(text=text)

    def _check(m: re.Match) -> str:
        cid = m.group(1)
        if cid in valid_ids:
            return m.group(0)
        result.removed_citation_ids.append(cid)
        return ""

    result.text = citation_pattern.sub(_check, text)
    return result
