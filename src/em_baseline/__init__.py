"""Summary baseline for the Explaining Markets competition.

One codebase, two Modal deployments (GPT-6 Luna and Gemini Flash-Lite),
selected via the ``BASELINE_MODEL`` environment variable. Each predicts from
the earnings-call facts and, when the event carries one, the earnings
preview. See the README.

Modules:
  config               — env-driven config; picks the model spec + credential pair
  webhook_app          — FastAPI app factory: verify, dedupe, spawn, ACK fast
  worker               — async event processing: fetch facts + preview → predict → submit
  bundle               — fetch/parse the DisclosureBundle behind information_url
  predictor            — DSPy ChainOfThought port of the research pipeline
  client               — submits predictions to the competition API
  event_utils          — small helpers for event payloads
  webhook_verification — vendored HMAC-SHA256 verifier (stdlib only)
"""

from em_baseline.webhook_verification import (
    WebhookVerificationError,
    verify_webhook,
)

__all__ = ["WebhookVerificationError", "verify_webhook"]
