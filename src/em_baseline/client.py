"""Tiny client for submitting predictions to the competition API.

One call: ``POST {EM_API_BASE_URL}/predictions`` authenticated with the
``X-API-Key`` header. The API always accepts well-formed predictions with 201;
late, duplicate, and pre-broadcast submissions are tagged for scoring rather
than rejected (scoring uses the first accepted on-time submission per event).
"""

from __future__ import annotations

import httpx

from em_baseline.config import Config


class PredictionSubmissionError(Exception):
    """Raised when the API rejects a prediction submission (4xx — no retry)."""


class TransientSubmissionError(PredictionSubmissionError):
    """Raised on a 5xx from the API — safe to retry."""


def submit_predictions(
    *,
    event_id: str,
    predictions: list[dict],
    config: Config | None = None,
    timeout: float = 10.0,
) -> dict:
    """POST predictions for one event. Returns the parsed response body.

    ``predictions`` is a list of
    ``{"identifier_value": str, "predicted_percentile": float}``.
    """
    cfg = config or Config.from_env()
    url = f"{cfg.api_base_url}/predictions"
    body = {"event_id": event_id, "predictions": predictions}

    resp = httpx.post(
        url,
        json=body,
        headers={"X-API-Key": cfg.api_key, "Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code >= 500:
        raise TransientSubmissionError(
            f"prediction submission failed: {resp.status_code} {resp.text}"
        )
    if resp.status_code >= 300:
        raise PredictionSubmissionError(
            f"prediction submission failed: {resp.status_code} {resp.text}"
        )
    return resp.json()
