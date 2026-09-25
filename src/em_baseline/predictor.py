"""LLM prediction of the post-earnings-call return percentile.

This is a direct port of the research pipeline's DSPy program (signature,
prompt text, and percentile normalization are verbatim): a single
``dspy.ChainOfThought`` call over two inputs — the earnings preview (an
agent-written research note assembled from public sources before the
release) and the bullet-point fact summary of the call. The preview is
optional: when an event has none, the field carries ``NO_PREVIEW_NOTE`` and
the prompt tells the model to rely on the facts alone. The competition only
scores ``predict_percentile``; ``predict_class`` and ``rationale`` are still
requested because the class↔percentile consistency constraints are part of
the prompt's calibration — they are logged, not submitted.

Failure policy (per the baseline's design):
  - The LLM answered but the output can't be parsed → neutral 0.5 fallback.
  - Transient provider errors (timeouts, rate limits, 5xx) → retried; if
    retries are exhausted the exception propagates and the caller submits
    nothing for the event.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import dspy
import litellm
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

NEUTRAL_PERCENTILE = 0.5

# Provider errors worth retrying. Anything else (auth, bad request, ...) is
# non-recoverable and propagates immediately.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.ServiceUnavailableError,
)

LM_TIMEOUT_SECONDS = 120
DEFAULT_ATTEMPTS = 2
DEFAULT_BACKOFF_SECONDS = 3.0


class PredictEarningsReturn(dspy.Signature):
    """Predict the unexpected stock return following an earnings call.

    You are given two inputs: a pre-earnings preview report, written shortly
    before the results were released, covering consensus expectations, the key
    metrics to watch, scenarios, and positioning; and key facts from the
    company's earnings call transcript. Use the preview to understand what the
    market expected going into the release, then judge the call's facts
    against those expectations. If no preview report is available for an
    event, rely on the call's facts alone. Predict the stock's unexpected
    return as a class and percentile.

    Base rates — calibrate your predictions to these proportions:
      - ~25% of stocks go UP (price increases 5%+ after the call)
      - ~50% of stocks are NEUTRAL (price moves less than 5%)
      - ~25% of stocks go DOWN (price decreases 5%+ after the call)

    Consistency constraints between class and percentile:
      - "down"    → percentile in [0.00, 0.25]
      - "neutral" → percentile in [0.25, 0.75]
      - "up"      → percentile in [0.75, 1.00]

    Your rationale must reference substantive evidence directly
    (e.g., "Revenue grew 18% year-over-year…"). Never reference fact
    numbers (e.g., never say "fact 3 shows…" or "according to fact 7"),
    and never reference section numbers of the preview report.
    """

    pre_earnings_preview_report: str = dspy.InputField(
        desc="Pre-earnings preview report written before the earnings release (markdown), "
        "or a note that none is available"
    )
    key_facts_discussed_in_earnings_call: str = dspy.InputField(
        desc="Bullet-point summary of key facts from the earnings call"
    )

    predict_class: Literal["up", "neutral", "down"] = dspy.OutputField(
        desc='Exactly one of: "up" (5%+ increase), "neutral" (<5% move), "down" (5%+ decrease)'
    )
    predict_percentile: float = dspy.OutputField(
        desc="Percentile rank of unexpected return: 0.0 (worst) to 1.0 (best)",
    )
    rationale: str = dspy.OutputField(
        desc="2-3 sentence explanation justifying the prediction using substantive evidence"
    )


# What the preview field receives when the bundle carries no preview item.
# The signature's instructions tell the model to fall back to the facts alone.
NO_PREVIEW_NOTE = "No pre-earnings preview report is available for this event."


@dataclass(frozen=True)
class PredictionOutcome:
    percentile: float
    predict_class: str | None = None
    rationale: str | None = None
    reasoning: str | None = None
    fallback_reason: str | None = None  # None ⇒ a real model prediction


def neutral_outcome(reason: str) -> PredictionOutcome:
    return PredictionOutcome(percentile=NEUTRAL_PERCENTILE, fallback_reason=reason)


def normalize_percentile(val: float) -> float:
    """Normalize a percentile to the [0, 1] range (models occasionally answer
    on a 0-100 scale)."""
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))


@lru_cache(maxsize=4)
def _lm(model: str) -> dspy.LM:
    # cache=False: every production event is unique, and response caching would
    # only mask real provider behavior in the live tests.
    return dspy.LM(model, timeout=LM_TIMEOUT_SECONDS, cache=False)


def _run_program(facts_str: str, preview: str, lm_model: str) -> dspy.Prediction:
    """One ChainOfThought call. Isolated so tests can stub the LLM boundary."""
    predictor = dspy.ChainOfThought(PredictEarningsReturn)
    with dspy.context(lm=_lm(lm_model)):
        return predictor(
            pre_earnings_preview_report=preview,
            key_facts_discussed_in_earnings_call=facts_str,
        )


def predict_from_facts(
    facts_str: str,
    preview: str | None,
    *,
    lm_model: str,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float | None = None,
) -> PredictionOutcome:
    """Predict the return percentile for one event's facts and (optional) preview.

    ``preview`` is the earnings preview markdown, or ``None`` when the event
    has none — the prompt then receives ``NO_PREVIEW_NOTE`` in its place.
    Returns a neutral 0.5 outcome when the model's answer is unparseable.
    Raises (after ``attempts`` tries) on persistent transient provider errors.
    """
    if backoff_seconds is None:
        backoff_seconds = DEFAULT_BACKOFF_SECONDS
    preview_text = preview or NO_PREVIEW_NOTE
    for attempt in range(1, attempts + 1):
        try:
            result = _run_program(facts_str, preview_text, lm_model)
            percentile = normalize_percentile(float(result.predict_percentile))
        except AdapterParseError as exc:
            logger.warning("LLM output unparseable; falling back to neutral: %s", exc)
            return neutral_outcome(f"unparseable_output:{type(exc).__name__}")
        except (TypeError, ValueError) as exc:
            logger.warning("LLM percentile not numeric; falling back to neutral: %s", exc)
            return neutral_outcome(f"invalid_percentile:{type(exc).__name__}")
        except TRANSIENT_ERRORS as exc:
            if attempt == attempts:
                logger.error("transient LLM error, retries exhausted: %s", exc)
                raise
            logger.warning(
                "transient LLM error (attempt %d/%d), retrying in %.0fs: %s",
                attempt,
                attempts,
                backoff_seconds,
                exc,
            )
            time.sleep(backoff_seconds)
            continue
        return PredictionOutcome(
            percentile=percentile,
            predict_class=result.predict_class,
            rationale=result.rationale,
            reasoning=getattr(result, "reasoning", None),
        )
    raise AssertionError("unreachable")  # pragma: no cover
