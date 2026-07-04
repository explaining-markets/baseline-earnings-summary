"""DisclosureBundle parsing: facts extraction and prompt formatting."""

from __future__ import annotations

import httpx
import pytest
import respx

from em_baseline.bundle import extract_facts, fetch_bundle, format_facts
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
