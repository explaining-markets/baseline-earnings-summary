"""FastAPI app factory for the competition webhook receiver.

The handler is deliberately fast: verify the signature, dedupe on the
``Webhook-Id`` header, hand the event to an async processor (a spawned Modal
function in deployment), and ACK 200 — all in well under the competition
deliverer's 10-second POST timeout. The per-event prediction deadline starts
at the ACK, so the slow LLM work happens afterwards with the full window.

A factory (rather than a module-level app) so tests can inject an in-memory
dedupe store and a recording processor.

Note: no ``from __future__ import annotations`` here — the route handlers are
defined inside ``create_app()``, and FastAPI must see the real ``Request`` /
``Response`` classes (not stringized annotations it can't resolve from a
nested scope) to inject them correctly.
"""

import logging
from collections.abc import Callable
from typing import Any, Protocol

from fastapi import FastAPI, Request, Response

from em_baseline.client import submit_predictions
from em_baseline.config import Config
from em_baseline.event_utils import is_test, neutral_predictions
from em_baseline.webhook_verification import WebhookVerificationError, verify_webhook

logger = logging.getLogger(__name__)


class SeenStore(Protocol):
    """The slice of the mapping interface the dedupe store needs — satisfied
    structurally by both ``modal.Dict`` and a plain ``dict``."""

    def __contains__(self, key: str) -> bool: ...

    def __setitem__(self, key: str, value: bool) -> None: ...


def create_app(
    *,
    service_name: str,
    seen_webhooks: SeenStore,
    process_event: Callable[[dict[str, Any]], object],
    config_loader: Callable[[], Config] = Config.from_env,
) -> FastAPI:
    """Build the webhook receiver.

    Args:
        service_name: reported by the health check (e.g. ``em-baseline-gemini``).
        seen_webhooks: idempotency store keyed on ``Webhook-Id`` — a
            ``modal.Dict`` in deployment, a plain dict in tests.
        process_event: called with the verified event payload; must return
            quickly (in deployment it spawns the worker function). If it
            raises, the request 500s and the competition redelivers.
        config_loader: overridable for tests.
    """
    api = FastAPI(title=service_name)

    @api.get("/")
    def health() -> dict:
        return {"ok": True, "service": service_name}

    @api.post("/")
    @api.post("/competition/webhook")  # alias, so an explicit-path URL also works
    async def competition_webhook(request: Request) -> Response:
        config = config_loader()

        raw_body = await request.body()  # raw bytes — never request.json()
        try:
            event = verify_webhook(
                raw_body=raw_body,
                headers=request.headers,
                secret=config.webhook_secret,
            )
        except WebhookVerificationError as exc:
            logger.warning("webhook verification failed: %s", exc)
            return Response(content=str(exc), status_code=401)

        # Idempotency: the Webhook-Id header (== event["id"]) is stable across
        # retries. Skip anything we've already handled.
        webhook_id = event.get("id")
        if webhook_id and webhook_id in seen_webhooks:
            return Response(status_code=200)

        # The portal's "Test Webhook" button sends a synthetic TEST event.
        # Submit a neutral prediction for it (accepted by the API, never
        # scored) so the portal test verifies the full receive → submit loop,
        # then ACK. A submit failure must not fail the ACK — the delivery
        # itself succeeded, and the portal will report the missing prediction
        # so a broken API key or submit path is visible.
        if is_test(event):
            try:
                submit_predictions(
                    event_id=event["event_id"],
                    predictions=neutral_predictions(event),
                    config=config,
                )
                logger.info("TEST event %s: neutral prediction submitted", event.get("event_id"))
            except Exception:
                logger.warning(
                    "TEST event %s: prediction failed to submit",
                    event.get("event_id"),
                    exc_info=True,
                )
            if webhook_id:
                seen_webhooks[webhook_id] = True
            return Response(status_code=200)

        # Hand off the slow work; if the spawn itself fails we 500 so the
        # competition redelivers.
        process_event(event)
        logger.info("event %s accepted and dispatched", event.get("event_id"))

        if webhook_id:
            seen_webhooks[webhook_id] = True
        return Response(status_code=200)

    return api
