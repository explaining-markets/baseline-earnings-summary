"""Shared fixtures: a signing helper mirroring the competition's server-side
signer, realistic event payloads, and two disclosure bundles built from real
disseminated data — ADEA Q1 2026 (facts only, no preview) and NVDA Q2 FY2027
(facts plus the earnings preview, from the competition's public archive)."""

from __future__ import annotations

import base64
import hmac
import json
import time
from hashlib import sha256
from pathlib import Path

import pytest

from em_baseline.config import BaselineModel, Config

DATA_DIR = Path(__file__).resolve().parent / "data"

# A deterministic whsec_ secret for tests (32 zero-ish bytes, base64url).
TEST_SECRET = "whsec_" + base64.urlsafe_b64encode(b"x" * 32).decode().rstrip("=")

INFORMATION_URL = "https://disclosures.test/bundle/ea_ADEA_Q1_2026?sig=abc"
NVDA_INFORMATION_URL = "https://disclosures.test/bundle/ea_NVDA_Q2_2027?sig=def"
API_BASE_URL = "https://api.competition.test/v1"


def sign_headers(
    raw_body: bytes,
    *,
    secret: str = TEST_SECRET,
    webhook_id: str = "evt_test_delivery_1",
    timestamp: int | None = None,
) -> dict[str, str]:
    """Produce Webhook-Id/-Timestamp/-Signature headers the way the
    competition's deliver Lambda does (Standard-Webhooks HMAC-SHA256)."""
    ts = int(time.time()) if timestamp is None else timestamp
    body = secret.removeprefix("whsec_")
    key = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
    signed_payload = f"{webhook_id}.{ts}.".encode() + raw_body
    signature = base64.b64encode(hmac.new(key, signed_payload, sha256).digest()).decode()
    return {
        "content-type": "application/json",
        "webhook-id": webhook_id,
        "webhook-timestamp": str(ts),
        "webhook-signature": f"v1,{signature}",
    }


def canonical_body(payload: dict) -> bytes:
    """The deliverer sends canonical JSON: sorted keys, no spaces."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def adea_facts() -> list[str]:
    return json.loads((DATA_DIR / "adea_facts.json").read_text())["facts"]


@pytest.fixture
def sample_event() -> dict:
    return {
        "id": "evt_test_delivery_1",
        "event_id": "ea_ADEA_Q1_2026",
        "event_type": "EARNINGS_RELEASE",
        "timing_category": "SCHEDULED",
        "event_datetime": "2026-05-04T21:00:00Z",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "ADEA"}],
        "information_url": INFORMATION_URL,
        "prediction_deadline": "2026-05-04T21:10:00Z",
    }


@pytest.fixture
def sample_bundle(adea_facts: list[str]) -> dict:
    return {
        "schema_version": "1.0",
        "event_id": "ea_ADEA_Q1_2026",
        "generated_at": "2026-05-04T21:05:00Z",
        "items": [
            {
                "id": "earnings-call-facts",
                "kind": "facts",
                "source": "earnings_call",
                "media_type": "application/json",
                "content": adea_facts,
            }
        ],
    }


@pytest.fixture
def nvda_bundle() -> dict:
    """A real bundle carrying both items: ten facts and the earnings preview."""
    return json.loads((DATA_DIR / "nvda_bundle.json").read_text())


@pytest.fixture
def nvda_facts(nvda_bundle: dict) -> list[str]:
    return next(i["content"] for i in nvda_bundle["items"] if i["kind"] == "facts")


@pytest.fixture
def nvda_preview(nvda_bundle: dict) -> str:
    return next(i["content"] for i in nvda_bundle["items"] if i["id"] == "earnings-preview")


@pytest.fixture
def nvda_event() -> dict:
    return {
        "id": "evt_test_delivery_2",
        "event_id": "ea_NVDA_Q2_2027",
        "event_type": "EARNINGS_RELEASE",
        "timing_category": "SCHEDULED",
        "event_datetime": "2026-08-26T21:00:00Z",
        "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "NVDA"}],
        "information_url": NVDA_INFORMATION_URL,
        "prediction_deadline": "2026-08-26T21:10:00Z",
    }


@pytest.fixture
def test_config() -> Config:
    return Config(
        baseline_model=BaselineModel.GPT5NANO,
        lm_model="openai/gpt-5-nano-2025-08-07",
        api_key="comp_sk_test_key",
        webhook_secret=TEST_SECRET,
        api_base_url=API_BASE_URL,
    )
