import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corpus_loader import load_case

from aivalrag_guard.output_sanitizer import (
    DEFAULT_LINK_ALLOWLIST,
    sanitize_output,
    validate_citations,
)


def test_pi_11_markdown_image_removed_by_default():
    case = load_case("PI-11")
    result = sanitize_output(case["payload"])
    assert "![" not in result.text
    assert "attacker.example" not in result.text
    assert result.removed_image_count == 1


def test_allowlisted_image_host_kept():
    text = "See this: ![chart](https://sec.gov/chart.png)"
    allow = frozenset({"sec.gov"})
    result = sanitize_output(text, image_allowlist=allow)
    assert "![chart](https://sec.gov/chart.png)" in result.text
    assert result.removed_image_count == 0


def test_non_allowlisted_link_delinkified_hostname_visible():
    text = "Read more [here](https://attacker.example/phish)."
    result = sanitize_output(text)
    assert "attacker.example" in result.text  # visible, but...
    assert "](https://attacker.example" not in result.text  # ...not a live link
    assert "link_delinkified" in result.events


def test_sec_gov_link_allowed_by_default():
    text = "See the filing [here](https://sec.gov/filing.htm)."
    result = sanitize_output(text)
    assert "[here](https://sec.gov/filing.htm)" in result.text
    assert result.delinkified_hosts == []


def test_app_own_domain_can_be_added_to_allowlist():
    text = "See [our page](https://app.example.com/x)."
    allow = DEFAULT_LINK_ALLOWLIST | {"app.example.com"}
    result = sanitize_output(text, link_allowlist=allow)
    assert "[our page](https://app.example.com/x)" in result.text


def test_bare_url_delinkified():
    text = "Visit https://attacker.example/steal for details."
    result = sanitize_output(text)
    assert "https://attacker.example" not in result.text
    assert "attacker.example" in result.text


def test_bare_sec_url_left_intact():
    text = "Source: https://sec.gov/Archives/filing.htm"
    result = sanitize_output(text)
    assert "https://sec.gov/Archives/filing.htm" in result.text


def test_raw_html_stripped():
    text = 'Click <a href="https://evil.example">here</a> or <img src=x onerror=alert(1)>'
    result = sanitize_output(text)
    assert "<a" not in result.text
    assert "<img" not in result.text
    assert result.html_stripped is True


def test_heart_emoticon_not_mistaken_for_html_tag():
    text = "I <3 this filing"
    result = sanitize_output(text)
    assert "<3" in result.text
    assert result.html_stripped is False


def test_length_cap_truncates_and_flags():
    text = "x" * 1000
    result = sanitize_output(text, max_length=100)
    assert len(result.text) == 100
    assert result.truncated is True


def test_no_truncation_when_under_cap():
    text = "short answer"
    result = sanitize_output(text, max_length=1000)
    assert result.text == text
    assert result.truncated is False


def test_order_image_then_link_then_html():
    # An <img> written as literal HTML must not survive by dodging the
    # markdown-image pattern -- html strip runs after image/link passes.
    text = '<img src="https://attacker.example/x.png">'
    result = sanitize_output(text)
    assert "<img" not in result.text
    assert result.html_stripped is True


# -- citation validation --

_CITE_PATTERN = re.compile(r"\[cite:([\w-]+)\]")


def test_valid_citation_kept():
    text = "Revenue grew 4% [cite:chunk-1]."
    result = validate_citations(text, frozenset({"chunk-1"}), citation_pattern=_CITE_PATTERN)
    assert "[cite:chunk-1]" in result.text
    assert result.removed_citation_ids == []


def test_unknown_citation_removed_and_reported():
    text = "Revenue grew 4% [cite:fabricated-id]."
    result = validate_citations(text, frozenset({"chunk-1"}), citation_pattern=_CITE_PATTERN)
    assert "[cite:fabricated-id]" not in result.text
    assert result.removed_citation_ids == ["fabricated-id"]


def test_mixed_valid_and_invalid_citations():
    text = "A [cite:chunk-1] and B [cite:chunk-999]."
    result = validate_citations(text, frozenset({"chunk-1"}), citation_pattern=_CITE_PATTERN)
    assert "[cite:chunk-1]" in result.text
    assert "[cite:chunk-999]" not in result.text
    assert result.removed_citation_ids == ["chunk-999"]
