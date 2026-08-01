"""The webhook receiver end-to-end: real HMAC-signed requests through the
FastAPI app, with an in-memory claim store and a recording spawner."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from em_baseline.config import Config
from em_baseline.webhook_app import create_app
from tests.conftest import canonical_body, sign_headers


class FakeDedupe:
    """In-memory stand-in for the Modal Dict claim/release choreography."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def claim(self, webhook_id: str | None) -> bool:
        if not webhook_id:
            return True
        if webhook_id in self.store:
            return False
        self.store[webhook_id] = "in_flight"
        return True

    async def release(self, webhook_id: str | None) -> None:
        if webhook_id:
            self.store.pop(webhook_id, None)


def build_harness(test_config: Config, *, spawner=None):
    dedupe = FakeDedupe()
    spawned: list[dict] = []

    async def record_spawn(event: dict, webhook_id: str | None) -> None:
        spawned.append(event)

    app = create_app(
        service_name="em-baseline-test",
        claim_webhook=dedupe.claim,
        spawn_worker=spawner or record_spawn,
        release_webhook=dedupe.release,
        config_loader=lambda: test_config,
    )
    client = TestClient(app, raise_server_exceptions=False)
    return client, dedupe, spawned


@pytest.fixture
def harness(test_config: Config):
    return build_harness(test_config)


def test_health_check(harness) -> None:
    client, _, _ = harness
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "service": "em-baseline-test"}


def test_valid_event_is_dispatched_and_acked(harness, sample_event: dict) -> None:
    client, dedupe, spawned = harness
    body = canonical_body(sample_event)

    resp = client.post("/", content=body, headers=sign_headers(body))

    assert resp.status_code == 200
    assert spawned == [sample_event]
    assert dedupe.store == {"evt_test_delivery_1": "in_flight"}


def test_alias_path_works(harness, sample_event: dict) -> None:
    client, _, spawned = harness
    body = canonical_body(sample_event)

    resp = client.post("/competition/webhook", content=body, headers=sign_headers(body))

    assert resp.status_code == 200
    assert len(spawned) == 1


def test_redelivery_is_deduped(harness, sample_event: dict) -> None:
    client, _, spawned = harness
    body = canonical_body(sample_event)

    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 200
    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 200
    assert len(spawned) == 1


def test_test_event_takes_the_normal_spawn_path(harness, sample_event: dict) -> None:
    """TEST events are handled post-ACK in the worker (which submits a neutral
    prediction) — never inline on the request path, where the submit's HTTP
    round trip would eat into the 20-second ACK budget."""
    client, dedupe, spawned = harness
    sample_event["event_type"] = "TEST"
    sample_event["event_id"] = "test_abc123"
    body = canonical_body(sample_event)

    resp = client.post("/", content=body, headers=sign_headers(body))

    assert resp.status_code == 200
    assert spawned == [sample_event]
    assert "evt_test_delivery_1" in dedupe.store


def test_bad_signature_is_rejected_401(harness, sample_event: dict) -> None:
    client, dedupe, spawned = harness
    body = canonical_body(sample_event)
    headers = sign_headers(body)
    headers["webhook-signature"] = "v1,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

    resp = client.post("/", content=body, headers=headers)

    assert resp.status_code == 401
    assert spawned == []
    assert dedupe.store == {}


def test_tampered_body_is_rejected_401(harness, sample_event: dict) -> None:
    client, _, spawned = harness
    body = canonical_body(sample_event)
    headers = sign_headers(body)

    resp = client.post("/", content=body + b" ", headers=headers)

    assert resp.status_code == 401
    assert spawned == []


def test_missing_headers_rejected_401(harness, sample_event: dict) -> None:
    client, _, _ = harness
    resp = client.post("/", content=canonical_body(sample_event))
    assert resp.status_code == 401


def test_spawn_failure_returns_500_for_redelivery(test_config: Config, sample_event: dict) -> None:
    async def exploding_spawner(event: dict, webhook_id: str | None) -> None:
        raise RuntimeError("spawn failed")

    client, _, _ = build_harness(test_config, spawner=exploding_spawner)
    body = canonical_body(sample_event)

    resp = client.post("/", content=body, headers=sign_headers(body))

    # 500 → the competition's deliverer puts the message back on the queue.
    assert resp.status_code == 500


def test_failed_spawn_releases_claim_so_redelivery_retries(
    test_config: Config, sample_event: dict
) -> None:
    """A claim is taken before the spawn; if the spawn fails it must be
    dropped, or the redelivery would be skipped as a duplicate of a job that
    never started."""
    attempts: list[dict] = []

    async def flaky_spawner(event: dict, webhook_id: str | None) -> None:
        attempts.append(event)
        if len(attempts) == 1:
            raise RuntimeError("first spawn failed")

    client, dedupe, _ = build_harness(test_config, spawner=flaky_spawner)
    body = canonical_body(sample_event)

    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 500
    assert dedupe.store == {}
    assert client.post("/", content=body, headers=sign_headers(body)).status_code == 200
    assert len(attempts) == 2
    assert dedupe.store == {"evt_test_delivery_1": "in_flight"}
