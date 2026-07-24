"""The webhook receiver end-to-end: real HMAC-signed requests through the
FastAPI app, with an in-memory dedupe store and a recording processor."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from em_baseline.config import Config
from em_baseline.webhook_app import create_app
from tests.conftest import API_BASE_URL, canonical_body, sign_headers

PREDICTIONS_URL = f"{API_BASE_URL}/predictions"


@pytest.fixture
def harness(test_config: Config):
    seen: dict[str, bool] = {}
    processed: list[dict] = []
    app = create_app(
        service_name="em-baseline-test",
        seen_webhooks=seen,
        process_event=processed.append,
        config_loader=lambda: test_config,
    )
    client = TestClient(app, raise_server_exceptions=False)
    return client, seen, processed


def test_health_check(harness) -> None:
    client, _, _ = harness
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "service": "em-baseline-test"}


def test_valid_event_is_dispatched_and_acked(harness, sample_event: dict) -> None:
    client, seen, processed = harness
    body = canonical_body(sample_event)

    resp = client.post("/", content=body, headers=sign_headers(body))

    assert resp.status_code == 200
    assert processed == [sample_event]
    assert seen == {"evt_test_delivery_1": True}


def test_alias_path_works(harness, sample_event: dict) -> None:
    client, _, processed = harness
    body = canonical_body(sample_event)

    resp = client.post("/competition/webhook", content=body, headers=sign_headers(body))

    assert resp.status_code == 200
    assert len(processed) == 1


def test_redelivery_is_deduped(harness, sample_event: dict) -> None:
    client, _, processed = harness
    body = canonical_body(sample_event)

    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 200
    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 200
    assert len(processed) == 1


@respx.mock
def test_test_event_submits_neutral_prediction_without_processing(
    harness, sample_event: dict
) -> None:
    """The portal test verifies the full receive → submit loop: a TEST event
    gets a neutral 0.5 prediction via the normal submit path (never the LLM
    worker), then a 200 ACK."""
    client, seen, processed = harness
    sample_event["event_type"] = "TEST"
    sample_event["event_id"] = "test_abc123"
    sample_event["information_url"] = "https://example.invalid/test"
    sample_event["focal_assets"] = [{"identifier_type": "TICKER", "identifier_value": "TEST"}]
    submit_route = respx.post(PREDICTIONS_URL).respond(
        status_code=201, json={"accepted": True, "submission_index": 1, "status": "accepted_first"}
    )
    body = canonical_body(sample_event)

    resp = client.post("/", content=body, headers=sign_headers(body))

    assert resp.status_code == 200
    assert processed == []
    assert "evt_test_delivery_1" in seen
    submitted = json.loads(submit_route.calls.last.request.content)
    assert submitted == {
        "event_id": "test_abc123",
        "predictions": [{"identifier_value": "TEST", "predicted_percentile": 0.5}],
    }


@respx.mock
def test_test_event_submit_failure_still_acks_200(harness, sample_event: dict) -> None:
    """A broken submit path must not fail the ACK — the delivery succeeded,
    and the portal reports the missing prediction instead."""
    client, seen, processed = harness
    sample_event["event_type"] = "TEST"
    sample_event["information_url"] = "https://example.invalid/test"
    respx.post(PREDICTIONS_URL).mock(side_effect=httpx.ConnectError("api down"))
    body = canonical_body(sample_event)

    resp = client.post("/", content=body, headers=sign_headers(body))

    assert resp.status_code == 200
    assert processed == []
    assert "evt_test_delivery_1" in seen


def test_bad_signature_is_rejected_401(harness, sample_event: dict) -> None:
    client, seen, processed = harness
    body = canonical_body(sample_event)
    headers = sign_headers(body)
    headers["webhook-signature"] = "v1,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

    resp = client.post("/", content=body, headers=headers)

    assert resp.status_code == 401
    assert processed == []
    assert seen == {}


def test_tampered_body_is_rejected_401(harness, sample_event: dict) -> None:
    client, _, processed = harness
    body = canonical_body(sample_event)
    headers = sign_headers(body)

    resp = client.post("/", content=body + b" ", headers=headers)

    assert resp.status_code == 401
    assert processed == []


def test_missing_headers_rejected_401(harness, sample_event: dict) -> None:
    client, _, _ = harness
    resp = client.post("/", content=canonical_body(sample_event))
    assert resp.status_code == 401


def test_spawn_failure_returns_500_for_redelivery(test_config: Config, sample_event: dict) -> None:
    def exploding_processor(event: dict) -> None:
        raise RuntimeError("spawn failed")

    app = create_app(
        service_name="em-baseline-test",
        seen_webhooks={},
        process_event=exploding_processor,
        config_loader=lambda: test_config,
    )
    client = TestClient(app, raise_server_exceptions=False)
    body = canonical_body(sample_event)

    resp = client.post("/", content=body, headers=sign_headers(body))

    # 500 → the competition's deliverer puts the message back on the queue.
    assert resp.status_code == 500


def test_failed_spawn_is_not_marked_seen_so_redelivery_retries(
    test_config: Config, sample_event: dict
) -> None:
    seen: dict[str, bool] = {}
    attempts: list[dict] = []

    def flaky_processor(event: dict) -> None:
        attempts.append(event)
        if len(attempts) == 1:
            raise RuntimeError("first spawn failed")

    app = create_app(
        service_name="em-baseline-test",
        seen_webhooks=seen,
        process_event=flaky_processor,
        config_loader=lambda: test_config,
    )
    client = TestClient(app, raise_server_exceptions=False)
    body = canonical_body(sample_event)

    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 500
    assert seen == {}
    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 200
    assert len(attempts) == 2
    assert seen == {"evt_test_delivery_1": True}
