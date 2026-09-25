"""Full worker path with the competition API and disclosure URL mocked via
respx; only the LLM boundary (``_run_program``) is stubbed."""

from __future__ import annotations

import json

import dspy
import httpx
import litellm
import pytest
import respx

from em_baseline import predictor
from em_baseline.config import Config, ModelSpec
from em_baseline.predictor import NO_PREVIEW_NOTE
from em_baseline.worker import handle_event
from tests.conftest import API_BASE_URL, INFORMATION_URL, NVDA_INFORMATION_URL

PREDICTIONS_URL = f"{API_BASE_URL}/predictions"

ACCEPTED = {
    "accepted": True,
    "submission_index": 1,
    "status": "accepted_first",
    "prediction_deadline": "2026-05-04T21:10:00Z",
}


def _stub_llm(monkeypatch: pytest.MonkeyPatch, percentile: float = 0.83) -> dict[str, str]:
    """Stub the LLM boundary; returns a dict that records the program's inputs."""
    seen: dict[str, str] = {}

    def fake_program(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        seen.update(facts=facts, preview=preview, model=spec.lm_model)
        return dspy.Prediction(
            predict_class="up",
            predict_percentile=percentile,
            rationale="Revenue grew strongly.",
            reasoning="Growth.",
        )

    monkeypatch.setattr(predictor, "_run_program", fake_program)
    return seen


@respx.mock
def test_happy_path_submits_first_focal_asset(
    monkeypatch: pytest.MonkeyPatch,
    sample_event: dict,
    sample_bundle: dict,
    test_config: Config,
) -> None:
    seen = _stub_llm(monkeypatch)
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    summary = handle_event(sample_event, config=test_config)

    request = submit_route.calls.last.request
    assert request.headers["x-api-key"] == "comp_sk_test_key"
    body = json.loads(request.content)
    assert body == {
        "event_id": "ea_ADEA_Q1_2026",
        "predictions": [{"identifier_value": "ADEA", "predicted_percentile": 0.83}],
    }
    assert summary["predicted_percentile"] == pytest.approx(0.83)
    assert summary["predict_class"] == "up"
    assert summary["fallback_reason"] is None
    assert summary["n_facts"] == 10
    assert summary["submission_status"] == "accepted_first"
    # ADEA's bundle has no preview: the prompt is told so, and the summary says so.
    assert seen["preview"] == NO_PREVIEW_NOTE
    assert summary["has_preview"] is False
    assert summary["preview_chars"] == 0


@respx.mock
def test_preview_is_passed_through_verbatim(
    monkeypatch: pytest.MonkeyPatch,
    nvda_event: dict,
    nvda_bundle: dict,
    nvda_preview: str,
    test_config: Config,
) -> None:
    seen = _stub_llm(monkeypatch)
    respx.get(NVDA_INFORMATION_URL).respond(json=nvda_bundle)
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    summary = handle_event(nvda_event, config=test_config)

    assert seen["preview"] == nvda_preview
    assert seen["facts"].startswith("- ")
    body = json.loads(submit_route.calls.last.request.content)
    assert body["event_id"] == "ea_NVDA_Q2_2027"
    assert [p["identifier_value"] for p in body["predictions"]] == ["NVDA"]
    assert summary["has_preview"] is True
    assert summary["preview_chars"] == len(nvda_preview)


@respx.mock
def test_multiple_focal_assets_only_first_is_submitted(
    monkeypatch: pytest.MonkeyPatch,
    sample_event: dict,
    sample_bundle: dict,
    test_config: Config,
) -> None:
    _stub_llm(monkeypatch)
    sample_event["focal_assets"].append({"identifier_type": "TICKER", "identifier_value": "OTHER"})
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    handle_event(sample_event, config=test_config)

    body = json.loads(submit_route.calls.last.request.content)
    assert [p["identifier_value"] for p in body["predictions"]] == ["ADEA"]


@respx.mock
def test_no_facts_submits_neutral(
    sample_event: dict, sample_bundle: dict, test_config: Config
) -> None:
    sample_bundle["items"] = []
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    summary = handle_event(sample_event, config=test_config)

    body = json.loads(submit_route.calls.last.request.content)
    assert body["predictions"][0]["predicted_percentile"] == 0.5
    assert summary["fallback_reason"] == "no_facts"


@respx.mock
def test_unparseable_llm_output_submits_neutral(
    monkeypatch: pytest.MonkeyPatch,
    sample_event: dict,
    sample_bundle: dict,
    test_config: Config,
) -> None:
    from dspy.utils.exceptions import AdapterParseError

    from em_baseline.predictor import PredictEarningsReturn

    def raise_parse_error(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        raise AdapterParseError("ChatAdapter", PredictEarningsReturn, "garbage")

    monkeypatch.setattr(predictor, "_run_program", raise_parse_error)
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    summary = handle_event(sample_event, config=test_config)

    body = json.loads(submit_route.calls.last.request.content)
    assert body["predictions"][0]["predicted_percentile"] == 0.5
    assert summary["fallback_reason"] == "unparseable_output:AdapterParseError"


@respx.mock
def test_persistent_llm_outage_submits_nothing(
    monkeypatch: pytest.MonkeyPatch,
    sample_event: dict,
    sample_bundle: dict,
    test_config: Config,
) -> None:
    def always_down(facts: str, preview: str, spec: ModelSpec) -> dspy.Prediction:
        raise litellm.exceptions.InternalServerError("boom", "openai", "gpt-5-nano")

    monkeypatch.setattr(predictor, "_run_program", always_down)
    monkeypatch.setattr(predictor, "DEFAULT_BACKOFF_SECONDS", 0.0)
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    with pytest.raises(litellm.exceptions.InternalServerError):
        handle_event(sample_event, config=test_config)
    assert not submit_route.called


@respx.mock
def test_test_event_submits_neutral_without_fetch_or_llm(
    sample_event: dict, test_config: Config
) -> None:
    """The portal test verifies the full receive → submit loop: a TEST event
    gets a neutral 0.5 prediction via the normal submit path, and neither the
    disclosure URL nor the LLM is touched (no respx route for either — an
    accidental call would error)."""
    sample_event["event_type"] = "TEST"
    sample_event["event_id"] = "test_abc123"
    sample_event["information_url"] = "https://example.invalid/test"
    sample_event["focal_assets"] = [{"identifier_type": "TICKER", "identifier_value": "TEST"}]
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    summary = handle_event(sample_event, config=test_config)

    body = json.loads(submit_route.calls.last.request.content)
    assert body == {
        "event_id": "test_abc123",
        "predictions": [{"identifier_value": "TEST", "predicted_percentile": 0.5}],
    }
    assert summary == {
        "event_id": "test_abc123",
        "test": True,
        "submission_status": "accepted_first",
    }


@respx.mock
def test_test_event_submit_failure_propagates_after_retries(
    monkeypatch: pytest.MonkeyPatch, sample_event: dict, test_config: Config
) -> None:
    """Post-ACK, a broken submit path may fail loudly — the exception reaches
    the Modal dashboard and the dedupe claim is dropped by the caller."""
    import em_baseline.worker as worker_mod

    monkeypatch.setattr(worker_mod, "RETRY_BACKOFF_SECONDS", 0.0)
    sample_event["event_type"] = "TEST"
    sample_event["event_id"] = "test_abc123"
    submit_route = respx.post(PREDICTIONS_URL).mock(side_effect=httpx.ConnectError("api down"))

    with pytest.raises(httpx.ConnectError):
        handle_event(sample_event, config=test_config)
    assert submit_route.call_count == 3


@respx.mock
def test_bundle_fetch_retries_transient_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    sample_event: dict,
    sample_bundle: dict,
    test_config: Config,
) -> None:
    import em_baseline.worker as worker_mod

    monkeypatch.setattr(worker_mod, "RETRY_BACKOFF_SECONDS", 0.0)
    _stub_llm(monkeypatch)
    respx.get(INFORMATION_URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=sample_bundle)]
    )
    respx.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

    summary = handle_event(sample_event, config=test_config)
    assert summary["predicted_percentile"] == pytest.approx(0.83)


@respx.mock
def test_submission_5xx_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    sample_event: dict,
    sample_bundle: dict,
    test_config: Config,
) -> None:
    import em_baseline.worker as worker_mod

    monkeypatch.setattr(worker_mod, "RETRY_BACKOFF_SECONDS", 0.0)
    _stub_llm(monkeypatch)
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    submit_route = respx.post(PREDICTIONS_URL).mock(
        side_effect=[httpx.Response(500), httpx.Response(201, json=ACCEPTED)]
    )

    handle_event(sample_event, config=test_config)
    assert submit_route.call_count == 2


@respx.mock
def test_submission_4xx_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    sample_event: dict,
    sample_bundle: dict,
    test_config: Config,
) -> None:
    from em_baseline.client import PredictionSubmissionError

    _stub_llm(monkeypatch)
    respx.get(INFORMATION_URL).respond(json=sample_bundle)
    submit_route = respx.post(PREDICTIONS_URL).respond(status_code=400, text="bad asset")

    with pytest.raises(PredictionSubmissionError):
        handle_event(sample_event, config=test_config)
    assert submit_route.call_count == 1
