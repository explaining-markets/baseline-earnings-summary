"""Live integration tests against the real model provider APIs.

Deselected by default (see ``addopts`` in pyproject.toml); run with::

    uv run pytest -m live

Provider keys are read from the environment, falling back to ``.env``. The
competition API and the disclosure URL are still mocked — these tests spend
provider credits, never competition submissions. The Luna cases run GPT-6 at
maximum reasoning effort over the Responses API with the full NVDA preview:
expect ~30 s per call (up to a few minutes) and a few cents of spend.
"""

from __future__ import annotations

import json
import os

import pytest
import respx
from dotenv import load_dotenv

from em_baseline.bundle import format_facts
from em_baseline.config import MODELS, BaselineModel, Config
from em_baseline.predictor import predict_from_facts
from em_baseline.worker import handle_event
from tests.conftest import API_BASE_URL, NVDA_INFORMATION_URL, TEST_SECRET

pytestmark = pytest.mark.live

load_dotenv()

PREDICTIONS_URL = f"{API_BASE_URL}/predictions"

ACCEPTED = {
    "accepted": True,
    "submission_index": 1,
    "status": "accepted_first",
    "prediction_deadline": "2026-08-26T21:10:00Z",
}

ALL_MODELS = list(BaselineModel)


def _require_key(model: BaselineModel) -> None:
    key_var = MODELS[model].provider_key_var
    if not os.environ.get(key_var):
        pytest.skip(f"{key_var} not set")


@pytest.mark.parametrize("model", ALL_MODELS, ids=[m.value for m in ALL_MODELS])
def test_predictor_against_real_provider(
    model: BaselineModel, nvda_facts: list[str], nvda_preview: str
) -> None:
    """The ported DSPy program, given facts and the preview, produces a valid,
    calibrated prediction."""
    _require_key(model)

    outcome = predict_from_facts(format_facts(nvda_facts), nvda_preview, spec=MODELS[model])

    assert outcome.fallback_reason is None, f"unexpected fallback: {outcome.fallback_reason}"
    assert 0.0 <= outcome.percentile <= 1.0
    assert outcome.predict_class in {"up", "neutral", "down"}
    assert outcome.rationale and len(outcome.rationale) > 20


@pytest.mark.parametrize("model", ALL_MODELS, ids=[m.value for m in ALL_MODELS])
def test_full_worker_path_against_real_provider(
    model: BaselineModel, nvda_event: dict, nvda_bundle: dict
) -> None:
    """The complete worker path — bundle fetch (facts + preview), real LLM
    call, submission — with only the competition endpoints mocked; the model
    providers' hosts pass through to the real network."""
    _require_key(model)
    config = Config(
        baseline_model=model,
        spec=MODELS[model],
        api_key="comp_sk_live_test",
        webhook_secret=TEST_SECRET,
        api_base_url=API_BASE_URL,
    )

    with respx.mock(assert_all_called=False) as router:
        # Real network for the model providers; mocks for the competition side.
        router.route(host="api.openai.com").pass_through()
        router.route(host="generativelanguage.googleapis.com").pass_through()
        router.get(NVDA_INFORMATION_URL).respond(json=nvda_bundle)
        submit_route = router.post(PREDICTIONS_URL).respond(status_code=201, json=ACCEPTED)

        summary = handle_event(nvda_event, config=config)

    assert submit_route.called
    body = json.loads(submit_route.calls.last.request.content)
    assert body["event_id"] == "ea_NVDA_Q2_2027"
    (prediction,) = body["predictions"]
    assert prediction["identifier_value"] == "NVDA"
    assert 0.0 <= prediction["predicted_percentile"] <= 1.0
    assert summary["fallback_reason"] is None
    assert summary["has_preview"] is True
    assert summary["predict_class"] in {"up", "neutral", "down"}
