"""Predictor logic around the LLM boundary (``_run_program`` is stubbed)."""

from __future__ import annotations

import dspy
import litellm
import pytest
from dspy.utils.exceptions import AdapterParseError

from em_baseline import predictor
from em_baseline.config import MODELS, BaselineModel, ModelSpec
from em_baseline.predictor import (
    NEUTRAL_PERCENTILE,
    NO_PREVIEW_NOTE,
    PredictEarningsReturn,
    normalize_percentile,
    predict_from_facts,
)

SPEC = MODELS[BaselineModel.GEMINI]  # two attempts, so retry behaviour is testable
LM_MODEL = SPEC.lm_model
FACTS = "- Revenue grew 18% year-over-year.\n- Guidance was raised."
PREVIEW = "# Acme (ACME) — Q2 FY2026 Earnings Preview\n\nConsensus EPS $1.20."


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


def test_signature_takes_the_preview_first_and_covers_its_absence() -> None:
    """The prompt is the research pipeline's treatment signature: the preview
    is the first input, and the instructions handle a missing preview."""
    assert list(PredictEarningsReturn.input_fields) == [
        "pre_earnings_preview_report",
        "key_facts_discussed_in_earnings_call",
    ]
    instructions = " ".join(PredictEarningsReturn.instructions.split())  # unwrap lines
    assert "If no preview report is available for an event" in instructions
    assert "never reference section numbers of the preview report" in instructions


def test_preview_is_passed_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def record(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        seen.update(facts=facts, preview=preview)
        return _prediction()

    monkeypatch.setattr(predictor, "_run_program", record)
    predict_from_facts(FACTS, PREVIEW, spec=SPEC)
    assert seen == {"facts": FACTS, "preview": PREVIEW}


@pytest.mark.parametrize("missing", [None, ""])
def test_missing_preview_becomes_the_note(
    monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    seen: dict[str, str] = {}

    def record(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        seen["preview"] = preview
        return _prediction()

    monkeypatch.setattr(predictor, "_run_program", record)
    predict_from_facts(FACTS, missing, spec=SPEC)
    assert seen["preview"] == NO_PREVIEW_NOTE


def test_happy_path_returns_model_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predictor, "_run_program", lambda facts, preview, spec: _prediction())
    outcome = predict_from_facts(FACTS, None, spec=SPEC)
    assert outcome.percentile == pytest.approx(0.83)
    assert outcome.predict_class == "up"
    assert outcome.rationale
    assert outcome.reasoning
    assert outcome.fallback_reason is None


def test_out_of_scale_percentile_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predictor, "_run_program", lambda facts, preview, spec: _prediction(83.0))
    assert predict_from_facts(FACTS, None, spec=SPEC).percentile == pytest.approx(0.83)


def test_unparseable_output_falls_back_to_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_parse_error(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        raise AdapterParseError("ChatAdapter", PredictEarningsReturn, "not json")

    monkeypatch.setattr(predictor, "_run_program", raise_parse_error)
    outcome = predict_from_facts(FACTS, None, spec=SPEC)
    assert outcome.percentile == NEUTRAL_PERCENTILE
    assert outcome.fallback_reason == "unparseable_output:AdapterParseError"


def test_non_numeric_percentile_falls_back_to_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        predictor, "_run_program", lambda facts, preview, spec: _prediction(percentile="high")
    )
    outcome = predict_from_facts(FACTS, None, spec=SPEC)
    assert outcome.percentile == NEUTRAL_PERCENTILE
    assert outcome.fallback_reason == "invalid_percentile:ValueError"


def test_transient_error_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def flaky(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        calls.append(1)
        if len(calls) == 1:
            raise litellm.exceptions.RateLimitError("slow down", "openai", LM_MODEL)
        return _prediction()

    monkeypatch.setattr(predictor, "_run_program", flaky)
    outcome = predict_from_facts(FACTS, None, spec=SPEC, backoff_seconds=0.0)
    assert len(calls) == 2
    assert outcome.fallback_reason is None


def test_transient_errors_exhaust_retries_and_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def always_down(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        calls.append(1)
        raise litellm.exceptions.InternalServerError("boom", "openai", LM_MODEL)

    monkeypatch.setattr(predictor, "_run_program", always_down)
    with pytest.raises(litellm.exceptions.InternalServerError):
        predict_from_facts(FACTS, None, spec=SPEC, attempts=3, backoff_seconds=0.0)
    assert len(calls) == 3


def test_non_transient_provider_error_propagates_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def bad_auth(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        calls.append(1)
        raise litellm.exceptions.AuthenticationError("bad key", "openai", LM_MODEL)

    monkeypatch.setattr(predictor, "_run_program", bad_auth)
    with pytest.raises(litellm.exceptions.AuthenticationError):
        predict_from_facts(FACTS, None, spec=SPEC, backoff_seconds=0.0)
    assert len(calls) == 1


def test_luna_lm_uses_the_responses_api_at_max_effort() -> None:
    """The LM is built exactly as in the preview-lift analysis: Responses API,
    reasoning effort max, the spec's timeout, no caching, no hidden retries,
    and provider defaults for everything else (no temperature / max tokens)."""
    lm = predictor._lm(MODELS[BaselineModel.LUNA])
    assert lm.model == "openai/gpt-6-luna"
    assert lm.model_type == "responses"
    assert lm.kwargs["reasoning"] == {"effort": "max"}
    assert lm.kwargs["timeout"] == 240
    assert lm.cache is False
    assert lm.num_retries == 0
    assert lm.kwargs.get("temperature") is None
    assert lm.kwargs.get("max_tokens") is None


def test_gemini_lm_uses_chat_completions_with_provider_defaults() -> None:
    lm = predictor._lm(MODELS[BaselineModel.GEMINI])
    assert lm.model == "gemini/gemini-flash-lite-latest"
    assert lm.model_type == "chat"
    assert "reasoning" not in lm.kwargs
    assert lm.kwargs["timeout"] == 120


def test_attempts_default_to_the_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """Luna is configured for a single attempt: a transient error propagates
    at once, so the worker never starts a retry that would miss the deadline."""
    calls: list[int] = []

    def always_down(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        calls.append(1)
        raise litellm.exceptions.InternalServerError("boom", "openai", spec.lm_model)

    monkeypatch.setattr(predictor, "_run_program", always_down)
    with pytest.raises(litellm.exceptions.InternalServerError):
        predict_from_facts(FACTS, None, spec=MODELS[BaselineModel.LUNA], backoff_seconds=0.0)
    assert len(calls) == 1
