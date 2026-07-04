"""Modal deployment for the Explaining Markets summary baselines.

One codebase, two deployments — ``BASELINE_MODEL`` (read from your shell at
deploy time and baked into the image) selects the model and the credential
pair, and parameterizes every Modal resource name so the deployments never
collide:

    BASELINE_MODEL=gpt5nano uv run modal deploy modal_app.py
    BASELINE_MODEL=gemini   uv run modal deploy modal_app.py

Each deploy prints a persistent public URL like
``https://<workspace>--em-baseline-gpt5nano.modal.run`` — that URL is the
webhook URL to paste into the portal for the matching submission, as-is.

Architecture (see README for the why):

    web (ASGI)          verify signature → dedupe → spawn worker → ACK 200
    process_event       fetch DisclosureBundle → DSPy prediction → submit

The webhook handler ACKs fast because the competition's delivery POST times
out after 10 seconds, while a gpt-5-nano reasoning call can take longer. The
per-event prediction deadline starts at the ACK, so the spawned worker has the
full window. Credentials come from the local ``.env`` at deploy time.
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

# Distributed key-value store for idempotency. Persists across redeploys, so a
# retried webhook is never processed twice. Keyed on the Webhook-Id header.
seen_webhooks = modal.Dict.from_name(f"{APP_NAME}-webhook-dedupe", create_if_missing=True)

secrets = [modal.Secret.from_dotenv(__file__)]


@app.function(image=image, secrets=secrets, timeout=280)
def process_event(event: dict) -> dict:
    """Async worker: fetch facts, run the LLM, submit the prediction.

    Spawned by the webhook handler after the ACK; a failure here is visible in
    the Modal dashboard but never blocks or fails the webhook delivery.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from em_baseline.worker import handle_event

    return handle_event(event)


@app.function(image=image, secrets=secrets)
@modal.asgi_app(label=APP_NAME)
def web():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from em_baseline.webhook_app import create_app

    return create_app(
        service_name=APP_NAME,
        seen_webhooks=seen_webhooks,
        process_event=lambda event: process_event.spawn(event),
    )
