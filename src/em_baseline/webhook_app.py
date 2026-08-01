"""FastAPI app factory for the competition webhook receiver.

The handler is deliberately fast: verify the signature, atomically claim the
``Webhook-Id``, spawn the worker, and ACK 200 — all in well under the
competition deliverer's 20-second POST timeout. The per-event prediction
deadline (5 minutes) starts at the ACK, so all slow work — the LLM, and even
the portal's synthetic TEST events — happens in the spawned worker afterwards.

The claim, spawn, and release hooks are injected as *async* callables: in
deployment they are Modal RPCs, and Modal's blocking interfaces run their own
event loop under the hood — calling them from inside an ``async def`` route
stalls the loop, the exact latency ACK-first is meant to avoid. The ``.aio``
variants are the async-native ones; the request path must use those, and only
those.

A factory (rather than a module-level app) so tests can inject an in-memory
claim and a recording spawner.

Note: no ``from __future__ import annotations`` here — the route handlers are
defined inside ``create_app()``, and FastAPI must see the real ``Request`` /
``Response`` classes (not stringized annotations it can't resolve from a
nested scope) to inject them correctly.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request, Response

from em_baseline.config import Config
from em_baseline.webhook_verification import WebhookVerificationError, verify_webhook

logger = logging.getLogger(__name__)


def create_app(
    *,
    service_name: str,
    claim_webhook: Callable[[str | None], Awaitable[bool]],
    spawn_worker: Callable[[dict[str, Any], str | None], Awaitable[object]],
    release_webhook: Callable[[str | None], Awaitable[None]],
    config_loader: Callable[[], Config] = Config.from_env,
) -> FastAPI:
    """Build the webhook receiver.

    Args:
        service_name: reported by the health check (e.g. ``em-baseline-gemini``).
        claim_webhook: atomically reserve a ``Webhook-Id``; False means it is
            already in flight or done and the delivery is skipped. Must treat
            a missing id as claimable (return True).
        spawn_worker: hand the verified event (plus its claim id, so the worker
            can release it when done) to the async processor. If it raises, the
            request 500s and the competition redelivers.
        release_webhook: drop a claim taken by ``claim_webhook`` — used only
            when the spawn itself fails, so the redelivery isn't skipped as a
            duplicate of a job that never started.
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
        # retries. The claim is atomic, so two containers handling the same
        # redelivery at the same moment can't both spawn.
        webhook_id = event.get("id")
        if not await claim_webhook(webhook_id):
            return Response(status_code=200)

        # Everything slow — including the TEST events' neutral submit —
        # happens in the worker, after this 200 goes out.
        try:
            await spawn_worker(event, webhook_id)
        except Exception:
            await release_webhook(webhook_id)
            raise  # → 500; the competition redelivers
        logger.info("event %s accepted and dispatched", event.get("event_id"))
        return Response(status_code=200)

    return api
