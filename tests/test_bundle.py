"""DisclosureBundle parsing: facts and preview extraction, prompt formatting."""

from __future__ import annotations

import httpx
import pytest
import respx

from em_baseline.bundle import (
    extract_facts,
    extract_option_stats,
    extract_preview,
    fetch_bundle,
    format_facts,
)
from tests.conftest import INFORMATION_URL


def test_extract_facts_happy_path(sample_bundle: dict, adea_facts: list[str]) -> None:
    assert extract_facts(sample_bundle) == adea_facts


def test_extract_facts_skips_unknown_kinds(sample_bundle: dict, adea_facts: list[str]) -> None:
    sample_bundle["items"].insert(
        0, {"id": "note", "kind": "text", "source": "test", "content": "hello"}
    )
    assert extract_facts(sample_bundle) == adea_facts


def test_extract_facts_missing_item_returns_none(sample_bundle: dict) -> None:
    sample_bundle["items"] = []
    assert extract_facts(sample_bundle) is None


def test_extract_facts_empty_list_returns_none(sample_bundle: dict) -> None:
    sample_bundle["items"][0]["content"] = []
    assert extract_facts(sample_bundle) is None


def test_extract_facts_by_reference_item_returns_none(sample_bundle: dict) -> None:
    item = sample_bundle["items"][0]
    item["content"] = None
    item["url"] = "https://disclosures.test/big-facts.json"
    assert extract_facts(sample_bundle) is None


def test_extract_facts_no_items_key_returns_none() -> None:
    assert extract_facts({"schema_version": "1.0"}) is None


def test_extract_preview_returns_markdown_verbatim(nvda_bundle: dict, nvda_preview: str) -> None:
    preview = extract_preview(nvda_bundle)
    assert preview == nvda_preview
    assert preview.startswith("# NVIDIA (NVDA)")
    assert extract_facts(nvda_bundle) is not None  # both items coexist


def test_extract_preview_absent_returns_none(sample_bundle: dict) -> None:
    assert extract_preview(sample_bundle) is None  # ADEA: facts only


def test_extract_preview_blank_content_returns_none(nvda_bundle: dict) -> None:
    nvda_bundle["items"][1]["content"] = "  \n"
    assert extract_preview(nvda_bundle) is None


def test_extract_preview_by_reference_item_returns_none(nvda_bundle: dict) -> None:
    item = nvda_bundle["items"][1]
    item["content"] = None
    item["url"] = "https://disclosures.test/big-preview.md"
    assert extract_preview(nvda_bundle) is None


def test_extract_preview_requires_matching_id_and_kind(nvda_bundle: dict) -> None:
    nvda_bundle["items"][1]["kind"] = "facts"  # right id, wrong kind
    assert extract_preview(nvda_bundle) is None
    nvda_bundle["items"][1]["kind"] = "text"
    nvda_bundle["items"][1]["id"] = "analyst-note"  # right kind, wrong id
    assert extract_preview(nvda_bundle) is None


def test_items_are_selected_by_id_not_position(nvda_bundle: dict, nvda_preview: str) -> None:
    nvda_bundle["items"].reverse()
    assert extract_preview(nvda_bundle) == nvda_preview
    assert len(extract_facts(nvda_bundle) or []) == 10


def test_extract_preview_no_items_key_returns_none() -> None:
    assert extract_preview({"schema_version": "1.0"}) is None


def test_format_facts_renders_bullet_lines() -> None:
    assert format_facts(["a", "b"]) == "- a\n- b"


@respx.mock
def test_fetch_bundle_returns_parsed_json(sample_bundle: dict) -> None:
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    assert fetch_bundle(INFORMATION_URL) == sample_bundle


@respx.mock
def test_fetch_bundle_raises_on_http_error() -> None:
    respx.get(INFORMATION_URL).respond(status_code=403)
    with pytest.raises(httpx.HTTPStatusError):
        fetch_bundle(INFORMATION_URL)


OPTION_STATS_ITEM = {
    "id": "option-implied-stats",
    "kind": "stats",
    "source": "option_market",
    "media_type": "application/json",
    "content": {
        "as_of": "2026-10-05T19:36:12Z",
        "methodology": "v1",
        "implied_earnings_volatility": {"value": 0.0847, "status": "ok"},
        "implied_absolute_earnings_move": {"value": 0.0676, "status": "ok"},
        "skew_25_delta": {"value": None, "status": "illiquid_wings"},
    },
}


def test_extract_option_stats_returns_values_and_none_for_unavailable(nvda_bundle: dict) -> None:
    bundle = {**nvda_bundle, "items": [*nvda_bundle["items"], OPTION_STATS_ITEM]}
    assert extract_option_stats(bundle) == {
        "implied_earnings_volatility": 0.0847,
        "implied_absolute_earnings_move": 0.0676,
        "skew_25_delta": None,
    }


def test_extract_option_stats_absent_or_unusable_returns_none(nvda_bundle: dict) -> None:
    assert extract_option_stats(nvda_bundle) is None
    assert extract_option_stats({}) is None
    for content in ({}, "8.5%", ["a"], None):
        item = {**OPTION_STATS_ITEM, "content": content}
        assert extract_option_stats({"items": [item]}) is None
    assert extract_option_stats({"items": [{**OPTION_STATS_ITEM, "kind": "text"}]}) is None


def test_extract_option_stats_never_trusts_a_value_without_an_ok_status() -> None:
    content = {
        "implied_earnings_volatility": {"value": 0.3, "status": "provisional"},
        "implied_absolute_earnings_move": {"value": "0.06", "status": "ok"},
        "skew_25_delta": {"value": True, "status": "ok"},
    }
    stats = extract_option_stats({"items": [{**OPTION_STATS_ITEM, "content": content}]})
    assert stats == dict.fromkeys(
        ("implied_earnings_volatility", "implied_absolute_earnings_move", "skew_25_delta")
    )


def test_the_option_stats_item_changes_nothing_the_baselines_read(
    nvda_bundle: dict, nvda_preview: str
) -> None:
    """The baselines build their prompt from the facts and the preview. A
    bundle that also carries the statistics must yield exactly the same two,
    wherever the new item sits."""
    facts = extract_facts(nvda_bundle)
    for position in range(len(nvda_bundle["items"]) + 1):
        items = list(nvda_bundle["items"])
        items.insert(position, OPTION_STATS_ITEM)
        with_stats = {**nvda_bundle, "items": items}
        assert extract_facts(with_stats) == facts
        assert extract_preview(with_stats) == nvda_preview
