"""Predictor logic around the LLM boundary (``_run_program`` is stubbed)."""

from __future__ import annotations

import dspy
import litellm
import pytest
from dspy.utils.exceptions import AdapterParseError

from em_baseline import predictor
from em_baseline.predictor import (
    NEUTRAL_PERCENTILE,
    PredictEarningsReturn,
    normalize_percentile,
    predict_from_facts,
)

LM_MODEL = "openai/gpt-5-nano-2025-08-07"
FACTS = "- Revenue grew 18% year-over-year.\n- Guidance was raised."


def _prediction(percentile: float = 0.83) -> dspy.Prediction:
    return dspy.Prediction(
        predict_class="up",
        predict_percentile=percentile,
        rationale="Revenue grew 18% year-over-year and guidance was raised.",
        reasoning="Strong growth signals.",
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.83, 0.83),
        (83.0, 0.83),  # model answered on a 0-100 scale
        (1.0, 1.0),
        (-0.2, 0.0),
        (150.0, 1.0),
    ],
)
def test_normalize_percentile(raw: float, expected: float) -> None:
    assert normalize_percentile(raw) == pytest.approx(expected)


def test_happy_path_returns_model_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predictor, "_run_program", lambda facts, model: _prediction())
    outcome = predict_from_facts(FACTS, lm_model=LM_MODEL)
    assert outcome.percentile == pytest.approx(0.83)
    assert outcome.predict_class == "up"
    assert outcome.rationale
    assert outcome.reasoning
    assert outcome.fallback_reason is None


def test_out_of_scale_percentile_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predictor, "_run_program", lambda facts, model: _prediction(83.0))
    assert predict_from_facts(FACTS, lm_model=LM_MODEL).percentile == pytest.approx(0.83)


def test_unparseable_output_falls_back_to_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_parse_error(facts: str, model: str) -> dspy.Prediction:
        raise AdapterParseError("ChatAdapter", PredictEarningsReturn, "not json")

    monkeypatch.setattr(predictor, "_run_program", raise_parse_error)
    outcome = predict_from_facts(FACTS, lm_model=LM_MODEL)
    assert outcome.percentile == NEUTRAL_PERCENTILE
    assert outcome.fallback_reason == "unparseable_output:AdapterParseError"


def test_non_numeric_percentile_falls_back_to_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        predictor, "_run_program", lambda facts, model: _prediction(percentile="high")
    )
    outcome = predict_from_facts(FACTS, lm_model=LM_MODEL)
    assert outcome.percentile == NEUTRAL_PERCENTILE
    assert outcome.fallback_reason == "invalid_percentile:ValueError"


def test_transient_error_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def flaky(facts: str, model: str) -> dspy.Prediction:
        calls.append(1)
        if len(calls) == 1:
            raise litellm.exceptions.RateLimitError("slow down", "openai", LM_MODEL)
        return _prediction()

    monkeypatch.setattr(predictor, "_run_program", flaky)
    outcome = predict_from_facts(FACTS, lm_model=LM_MODEL, backoff_seconds=0.0)
    assert len(calls) == 2
    assert outcome.fallback_reason is None


def test_transient_errors_exhaust_retries_and_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def always_down(facts: str, model: str) -> dspy.Prediction:
        calls.append(1)
        raise litellm.exceptions.InternalServerError("boom", "openai", LM_MODEL)

    monkeypatch.setattr(predictor, "_run_program", always_down)
    with pytest.raises(litellm.exceptions.InternalServerError):
        predict_from_facts(FACTS, lm_model=LM_MODEL, attempts=3, backoff_seconds=0.0)
    assert len(calls) == 3


def test_non_transient_provider_error_propagates_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def bad_auth(facts: str, model: str) -> dspy.Prediction:
        calls.append(1)
        raise litellm.exceptions.AuthenticationError("bad key", "openai", LM_MODEL)

    monkeypatch.setattr(predictor, "_run_program", bad_auth)
    with pytest.raises(litellm.exceptions.AuthenticationError):
        predict_from_facts(FACTS, lm_model=LM_MODEL, backoff_seconds=0.0)
    assert len(calls) == 1
