"""Small helpers for working with a verified webhook event payload.

A verified event looks like::

    {
      "id": "<matches Webhook-Id; use as your idempotency key>",
      "event_id": "ea_ADEA_Q1_2026",
      "event_type": "EARNINGS_RELEASE",   # or "TEST"
      "timing_category": "SCHEDULED",
      "event_datetime": "2026-01-15T21:00:00Z",
      "focal_assets": [{"identifier_type": "TICKER", "identifier_value": "ADEA"}],
      "information_url": "https://...signed...",
      "prediction_deadline": "2026-01-15T21:05:00Z"
    }
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def is_test(event: dict) -> bool:
    """True for the portal's synthetic 'Test Webhook' deliveries.

    Submit a neutral prediction for these (see ``neutral_predictions``), then
    ACK — that's how the portal test verifies the full receive → submit loop.
    Test predictions are accepted by the API but never scored.
    """
    return event.get("event_type") == "TEST"


def neutral_predictions(event: dict) -> list[dict]:
    """A neutral 0.5 prediction per focal asset.

    Used for TEST events: exercises the credentials and submit path without
    calling the model.
    """
    return [
        {"identifier_value": asset["identifier_value"], "predicted_percentile": 0.5}
        for asset in event.get("focal_assets", [])
    ]


def first_focal_asset(event: dict) -> str:
    """The ticker to predict on.

    The competition currently disseminates a single focal asset per event; the
    list shape exists for future expansion, so this baseline deliberately
    predicts on the first entry only.
    """
    return str(event["focal_assets"][0]["identifier_value"])


def log_deadline(event: dict) -> None:
    """Log the prediction deadline and seconds remaining, best-effort."""
    deadline = event.get("prediction_deadline")
    if not deadline:
        return
    try:
        dt = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
        remaining = (dt - datetime.now(UTC)).total_seconds()
        logger.info("[%s] deadline %s (~%.0fs left)", event.get("event_id"), deadline, remaining)
    except ValueError:
        logger.info("[%s] deadline %s", event.get("event_id"), deadline)
