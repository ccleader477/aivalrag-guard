"""I1 -- parse and normalize untrusted document text before it reaches
any prompt or heuristic scan. Two independent concerns:

1. HTML/iXBRL: strip elements that never render for a human reader
   (scripts, hidden nodes, ix:hidden) using a real parser -- regex over
   HTML is how these rules get bypassed.
2. Unicode: strip characters whose only function is to hide or reorder
   text from a human reader while a model still "sees" them raw.

Both return a report of what was removed (counts, a hash of the removed
text) rather than removing silently -- if hidden text turns out to
contain an I2 rule match, that's a strong signal the document is hostile,
not just poorly formatted, and I4 uses this report to raise risk.

Best-effort, not exhaustive -- documented gaps, not silent ones:
- White-on-white detection only sees inline `style` attributes, never
  resolves `<style>` blocks, external stylesheets, or CSS classes.
- font-size/text-indent/left thresholds only understand `px` units.
- PDF hidden-text detection (render mode, sub-1px fonts) depends on
  whichever PDF library the caller uses to extract spans -- this module
  exposes `is_pdf_span_hidden` for the caller to apply per-span, it does
  not parse PDF files itself.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Comment

SANITIZER_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Unicode
# ---------------------------------------------------------------------------

_ZERO_WIDTH = "".join(["​", "‌", "‍", "⁠", "﻿"])
_BIDI_CONTROLS = "".join([chr(c) for c in range(0x202A, 0x202F)] + [chr(c) for c in range(0x2066, 0x206A)])
_TAGS_BLOCK_RE = re.compile("[\U000E0000-\U000E007F]")
_ZERO_WIDTH_RE = re.compile(f"[{re.escape(_ZERO_WIDTH)}]")
_BIDI_RE = re.compile(f"[{re.escape(_BIDI_CONTROLS)}]")
_WHITESPACE_RUN_RE = re.compile(r"[ \t ]{2,}")
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Superscript/subscript digits -> plain digit, captured into an explicit
# [fnN] marker before any compatibility folding runs (so NFKC-style
# folding never silently turns "$1,234¹" into "$1,2341").
_SUPERSCRIPT_DIGITS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
}
_SUBSCRIPT_DIGITS = {chr(0x2080 + i): str(i) for i in range(10)}
_SUPER_SUB_RE = re.compile(f"[{re.escape(''.join(_SUPERSCRIPT_DIGITS) + ''.join(_SUBSCRIPT_DIGITS))}]+")

# Explicit, narrow compatibility folding -- NOT full NFKC (which folds
# superscripts into digits and would defeat the footnote-marker step
# above). Fullwidth ASCII (U+FF01-U+FF5E) is a common keyword-filter
# evasion technique.
_FULLWIDTH_RE = re.compile("[！-～]")


def _fold_super_sub(match: re.Match) -> str:
    digits = "".join(
        _SUPERSCRIPT_DIGITS.get(c) or _SUBSCRIPT_DIGITS.get(c) or "" for c in match.group(0)
    )
    return f"[fn{digits}]"


def _fold_fullwidth(match: re.Match) -> str:
    return chr(ord(match.group(0)) - 0xFEE0)


def normalize_unicode(text: str) -> str:
    """NFC (not NFKC -- see module docstring), zero-width/bidi/Tags-block
    stripping, footnote-marker extraction, narrow fullwidth folding,
    whitespace collapse."""
    text = _SUPER_SUB_RE.sub(_fold_super_sub, text)
    text = _FULLWIDTH_RE.sub(_fold_fullwidth, text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _BIDI_RE.sub("", text)
    text = _TAGS_BLOCK_RE.sub("", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf" or c in "\n\t")
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text


# ---------------------------------------------------------------------------
# HTML / iXBRL
# ---------------------------------------------------------------------------

_STRIP_TAGS = {"script", "style", "noscript", "template", "iframe", "object", "embed"}
_WHITE_COLOR_RE = re.compile(r"^\s*(white|#fff(?:fff)?|rgba?\(\s*255\s*,\s*255\s*,\s*255\b)", re.I)


@dataclass
class SanitizationReport:
    removed_counts: dict[str, int] = field(default_factory=dict)
    removed_chars: dict[str, int] = field(default_factory=dict)
    removed_text_sha256: str | None = None
    sanitizer_version: str = SANITIZER_VERSION

    def _bump(self, rule: str, text: str) -> None:
        self.removed_counts[rule] = self.removed_counts.get(rule, 0) + 1
        self.removed_chars[rule] = self.removed_chars.get(rule, 0) + len(text)

    def total_removed_chars(self) -> int:
        return sum(self.removed_chars.values())


def _parse_inline_style(style: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for decl in style.split(";"):
        if ":" not in decl:
            continue
        k, _, v = decl.partition(":")
        props[k.strip().lower()] = v.strip().lower()
    return props


def _px_value(value: str) -> float | None:
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*px$", value.strip())
    return float(m.group(1)) if m else None


def _style_is_hidden(props: dict[str, str]) -> str | None:
    if props.get("display") == "none":
        return "css_display_none"
    if props.get("visibility") == "hidden":
        return "css_visibility_hidden"
    if "opacity" in props:
        try:
            if float(props["opacity"]) <= 0:
                return "css_opacity_zero"
        except ValueError:
            pass
    fs = props.get("font-size")
    if fs is not None:
        px = _px_value(fs)
        if px is not None and px <= 1:
            return "css_font_size_zero"
    for prop in ("text-indent", "left"):
        px = _px_value(props.get(prop, ""))
        if px is not None and px <= -999:
            return "css_offscreen_position"
    return None


def _has_non_white_background(tag) -> bool:
    node = tag
    while node is not None and getattr(node, "name", None):
        style = node.attrs.get("style") if hasattr(node, "attrs") else None
        if style:
            bg = _parse_inline_style(style).get("background-color") or _parse_inline_style(style).get("background")
            if bg and not _WHITE_COLOR_RE.match(bg):
                return True
        node = node.parent
    return False


def strip_hidden_html(html: str) -> tuple[str, SanitizationReport]:
    """Parse with a real parser (BeautifulSoup/html.parser, never regex),
    remove elements that never render for a human, return (visible_text,
    report). ix:hidden content is excluded from the narrative text
    entirely -- XBRL facts belong to the structured XBRL path, not chunk
    text."""
    soup = BeautifulSoup(html, "html.parser")
    report = SanitizationReport()

    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        report._bump("html_comment", str(comment))
        comment.extract()

    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            report._bump(f"tag_{tag_name}", tag.get_text())
            tag.decompose()

    head = soup.find("head")
    if head is not None:
        title = head.find("title")
        for child in list(head.children):
            if child is title:
                continue
            text = child.get_text() if hasattr(child, "get_text") else str(child)
            if text.strip():
                report._bump("head_non_title", text)
            child.extract()

    # ix:hidden (iXBRL) -- namespace-prefixed tag name; BeautifulSoup with
    # html.parser sees "ix:hidden" as a literal tag name.
    for tag in soup.find_all(re.compile(r"^ix:hidden$", re.I)):
        report._bump("ixbrl_hidden", tag.get_text())
        tag.decompose()

    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        report._bump("aria_hidden", tag.get_text())
        tag.decompose()

    for tag in soup.find_all(attrs={"hidden": True}):
        report._bump("hidden_attr", tag.get_text())
        tag.decompose()

    for tag in list(soup.find_all(style=True)):
        props = _parse_inline_style(tag.attrs.get("style", ""))
        rule = _style_is_hidden(props)
        if rule:
            report._bump(rule, tag.get_text())
            tag.decompose()

    # Best-effort white-on-white: element's own inline color is white AND
    # no ancestor sets a non-white background.
    for tag in list(soup.find_all(style=True)):
        if not tag.parent:
            continue
        props = _parse_inline_style(tag.attrs.get("style", ""))
        color = props.get("color", "")
        if color and _WHITE_COLOR_RE.match(color) and not _has_non_white_background(tag):
            report._bump("white_on_white", tag.get_text())
            tag.decompose()

    visible_text = soup.get_text(separator=" ")

    removed_text = "".join(f"{k}:{v}" for k, v in sorted(report.removed_chars.items()))
    if report.removed_counts:
        report.removed_text_sha256 = hashlib.sha256(
            "".join(str(c) for c in report.removed_counts.values()).encode("utf-8")
        ).hexdigest()

    return visible_text, report


def normalize_document(html_or_text: str, *, is_html: bool) -> tuple[str, SanitizationReport]:
    """Full I1 pipeline: HTML stripping (if applicable) then Unicode
    normalization. Always returns a report, even for plain text input
    (report will simply show no HTML removals)."""
    if is_html:
        visible_text, report = strip_hidden_html(html_or_text)
    else:
        visible_text, report = html_or_text, SanitizationReport()
    return normalize_unicode(visible_text), report


# ---------------------------------------------------------------------------
# PDF (caller supplies span metadata; this module makes the hidden/visible
# call so the rule lives in one place regardless of which PDF library --
# pdfplumber, PyMuPDF, etc -- extracted the spans)
# ---------------------------------------------------------------------------

def is_pdf_span_hidden(*, render_mode: int | None = None, font_size: float | None = None,
                        min_visible_font_size: float = 1.0) -> bool:
    """render_mode 3 is invisible text per the PDF spec (Tr 3). Font size
    at or below the threshold is treated as hidden -- best-effort, since
    not every PDF extractor exposes render mode or exact font size."""
    if render_mode == 3:
        return True
    if font_size is not None and font_size <= min_visible_font_size:
        return True
    return False
