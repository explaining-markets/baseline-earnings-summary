"""Modal deployment for the Explaining Markets summary baselines.

One codebase, one deployment per model — ``BASELINE_MODEL`` (read from your
shell at deploy time and baked into the image) selects the model and the
credential pair, and parameterizes every Modal resource name so the
deployments never collide:

    BASELINE_MODEL=gpt5nano uv run modal deploy modal_app.py
    BASELINE_MODEL=gemini   uv run modal deploy modal_app.py

Each deploy prints a persistent public URL like
``https://<workspace>--em-baseline-gpt5nano.modal.run`` — that URL is the
webhook URL to paste into the portal for the matching submission, as-is.

Architecture (see README for the why):

    web (ASGI)          verify signature → claim Webhook-Id → spawn worker → ACK 200
    process_event       fetch DisclosureBundle → DSPy prediction → submit

The webhook handler ACKs fast because the competition's delivery POST times
out after 20 seconds, while a reasoning-model call can take far longer. The
per-event prediction deadline (5 minutes) starts at the ACK, so the spawned
worker has the full window. Credentials come from the local ``.env`` at
deploy time.
"""

import modal

from em_baseline.config import baseline_model_from_env

BASELINE_MODEL = baseline_model_from_env()

APP_NAME = f"em-baseline-{BASELINE_MODEL.value}"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]>=0.110", "httpx>=0.27", "dspy==3.2.1", "pydantic>=2.6")
    .env({"BASELINE_MODEL": BASELINE_MODEL.value})
    .add_local_python_source("em_baseline")
)

# Distributed key-value store for idempotency, keyed on the Webhook-Id header.
# Persists across redeploys, so a retried webhook is never processed twice.
# Three states:
#
#   "in_flight"   a worker is running right now — skip duplicates
#   "done"        the prediction was submitted — skip forever
#   absent        never seen, or the last attempt failed — (re)run it
#
# Marking an event done up front would be the bug: a failed prediction would
# look handled and the redelivery would be skipped.
seen_webhooks = modal.Dict.from_name(f"{APP_NAME}-webhook-dedupe", create_if_missing=True)

secrets = [modal.Secret.from_dotenv(__file__)]


async def _claim(webhook_id: str | None) -> bool:
    """Atomically reserve this Webhook-Id; False = already in flight or done.

    Modal's blocking interfaces run their own event loop under the hood, so
    calling them from inside an ``async def`` route stalls the loop — the
    request path must use the ``.aio`` variants, and only those.
    """
    if not webhook_id:
        return True
    return await seen_webhooks.put.aio(webhook_id, "in_flight", skip_if_exists=True)


async def _drop_claim(webhook_id: str | None) -> None:
    """Drop a claim whose spawn failed, so the redelivery isn't deduped away."""
    if webhook_id:
        await seen_webhooks.pop.aio(webhook_id, None)


def _release(webhook_id: str | None, *, submitted: bool) -> None:
    """Mark the claim done on success, or drop it so a redelivery can retry.

    Runs inside the worker container (sync context), so the blocking interface
    is fine here.
    """
    if not webhook_id:
        return
    if submitted:
        seen_webhooks[webhook_id] = "done"
    else:
        seen_webhooks.pop(webhook_id, None)


@app.function(image=image, secrets=secrets, timeout=280)
def process_event(event: dict, webhook_id: str | None = None) -> dict:
    """Worker: fetch facts, run the LLM, submit the prediction.

    Spawned by the webhook handler after the ACK; a failure here is visible in
    the Modal dashboard but never blocks or fails the webhook delivery. The
    dedupe claim is released on the way out — kept as "done" only when the
    submit went through, dropped otherwise so a redelivery can retry.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from em_baseline.worker import handle_event

    submitted = False
    try:
        summary = handle_event(event)
        submitted = True
        return summary
    finally:
        _release(webhook_id, submitted=submitted)


async def _spawn_worker(event: dict, webhook_id: str | None) -> object:
    return await process_event.spawn.aio(event, webhook_id)


@app.function(image=image, secrets=secrets)
@modal.asgi_app(label=APP_NAME)
def web():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from em_baseline.webhook_app import create_app

    return create_app(
        service_name=APP_NAME,
        claim_webhook=_claim,
        spawn_worker=_spawn_worker,
        release_webhook=_drop_claim,
    )
