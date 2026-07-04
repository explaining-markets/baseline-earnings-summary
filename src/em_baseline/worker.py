"""Process one verified webhook event: fetch facts → predict → submit.

This runs in the spawned Modal worker function, *after* the webhook has been
ACKed (the handler ACKs fast because the competition's delivery POST has a
10-second timeout, while a reasoning-model call can take longer). The per-event
prediction deadline starts at the ACK, so the worker has the full window.

Failure policy:
  - No usable facts in the bundle → submit the neutral 0.5 baseline.
  - LLM answered but unparseable → submit the neutral 0.5 baseline.
  - Transient failures (bundle fetch, LLM provider, submission API) are
    retried; if retries are exhausted the exception propagates and nothing is
    submitted — the failed call is visible in the Modal dashboard/logs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

from em_baseline.bundle import extract_facts, fetch_bundle, format_facts
from em_baseline.client import TransientSubmissionError, submit_predictions
from em_baseline.config import Config
from em_baseline.event_utils import first_focal_asset, log_deadline
from em_baseline.predictor import neutral_outcome, predict_from_facts

logger = logging.getLogger(__name__)

T = TypeVar("T")

FETCH_ATTEMPTS = 3
SUBMIT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


def handle_event(event: dict[str, Any], config: Config | None = None) -> dict[str, Any]:
    """Predict and submit for one event. Returns a loggable summary dict."""
    cfg = config or Config.from_env()
    event_id = event["event_id"]
    ticker = first_focal_asset(event)
    log_deadline(event)

    bundle = _retry_transient(
        lambda: fetch_bundle(event["information_url"]),
        attempts=FETCH_ATTEMPTS,
        transient=(httpx.HTTPError,),
        what=f"bundle fetch for {event_id}",
    )
    facts = extract_facts(bundle)
    if facts is None:
        logger.warning("[%s] no usable facts in bundle — submitting neutral 0.5", event_id)
        outcome = neutral_outcome("no_facts")
    else:
        outcome = predict_from_facts(format_facts(facts), lm_model=cfg.lm_model)

    response = _retry_transient(
        lambda: submit_predictions(
            event_id=event_id,
            predictions=[{"identifier_value": ticker, "predicted_percentile": outcome.percentile}],
            config=cfg,
        ),
        attempts=SUBMIT_ATTEMPTS,
        transient=(httpx.TransportError, TransientSubmissionError),
        what=f"prediction submission for {event_id}",
    )

    summary = {
        "event_id": event_id,
        "ticker": ticker,
        "model": cfg.lm_model,
        "n_facts": len(facts) if facts else 0,
        "predicted_percentile": outcome.percentile,
        "predict_class": outcome.predict_class,
        "rationale": outcome.rationale,
        "fallback_reason": outcome.fallback_reason,
        "submission_status": response.get("status"),
        "submission_index": response.get("submission_index"),
    }
    logger.info("[%s] submitted: %s", event_id, summary)
    return summary


def _retry_transient(
    fn: Callable[[], T],
    *,
    attempts: int,
    transient: tuple[type[Exception], ...],
    what: str,
    backoff_seconds: float | None = None,
) -> T:
    """Run ``fn``, retrying transient errors with linear backoff."""
    if backoff_seconds is None:
        backoff_seconds = RETRY_BACKOFF_SECONDS
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except transient as exc:
            if attempt == attempts:
                logger.error("%s failed after %d attempts: %s", what, attempts, exc)
                raise
            logger.warning(
                "%s failed (attempt %d/%d), retrying in %.0fs: %s",
                what,
                attempt,
                attempts,
                backoff_seconds * attempt,
                exc,
            )
            time.sleep(backoff_seconds * attempt)
    raise AssertionError("unreachable")  # pragma: no cover
