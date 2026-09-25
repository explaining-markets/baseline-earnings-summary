"""Configuration read from the environment.

One codebase powers two Modal deployments; ``BASELINE_MODEL`` selects which.
At deploy time it comes from your shell (``BASELINE_MODEL=gemini uv run modal
deploy modal_app.py``) and is baked into the container image, so at runtime the
same variable is simply read back from the environment.

Both deployments share a single ``.env`` (loaded by Modal at deploy time via
``Secret.from_dotenv``); the competition credentials are suffixed per baseline
and the active pair is picked here.

Required:
  BASELINE_MODEL               "luna" or "gemini"
  EM_API_KEY_{SUFFIX}          submission API key; sent as the X-API-Key header
  EM_WEBHOOK_SECRET_{SUFFIX}   signing secret (whsec_...) for incoming webhooks
  OPENAI_API_KEY               when BASELINE_MODEL=luna
  GEMINI_API_KEY               when BASELINE_MODEL=gemini

Optional:
  EM_API_BASE_URL              API base URL (default: prod)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

DEFAULT_API_BASE_URL = "https://api.explainingmarkets.ai/v1"


class BaselineModel(StrEnum):
    LUNA = "luna"
    GEMINI = "gemini"


@dataclass(frozen=True)
class ModelSpec:
    """How one baseline calls its model."""

    lm_model: str  # LiteLLM model string (DSPy routes through LiteLLM)
    provider_key_var: str  # read by LiteLLM from the env; checked for presence here
    timeout_seconds: int  # per-call LiteLLM timeout
    attempts: int  # predictor attempts on transient provider errors
    # Extra kwargs forwarded verbatim to ``dspy.LM`` (and on to the provider
    # request). Empty ⇒ provider defaults.
    lm_kwargs: dict[str, Any] = field(default_factory=dict)


MODELS: dict[BaselineModel, ModelSpec] = {
    # GPT-6 Luna over OpenAI's Responses API at maximum reasoning effort — the
    # configuration measured in the earnings-preview lift analysis. The alias
    # floats (no dated snapshot exists). No temperature / max_tokens: reasoning
    # models take provider defaults. One attempt: a max-effort call with the
    # preview runs ~30 s at the median and up to ~2.5 minutes in the tail, so a
    # retry would land after the competition's 5-minute prediction deadline.
    BaselineModel.LUNA: ModelSpec(
        "openai/gpt-6-luna",
        "OPENAI_API_KEY",
        timeout_seconds=240,
        attempts=1,
        lm_kwargs={"model_type": "responses", "reasoning": {"effort": "max"}},
    ),
    # Gemini Flash-Lite over chat completions with provider defaults. Floats on
    # the -latest alias deliberately. Calls take a few seconds.
    BaselineModel.GEMINI: ModelSpec(
        "gemini/gemini-flash-lite-latest",
        "GEMINI_API_KEY",
        timeout_seconds=120,
        attempts=2,
    ),
}


@dataclass(frozen=True)
class Config:
    baseline_model: BaselineModel
    spec: ModelSpec
    api_key: str
    webhook_secret: str
    api_base_url: str

    @classmethod
    def from_env(cls) -> Config:
        """Load and validate config. Raises if a required var is missing."""
        model = baseline_model_from_env()
        spec = MODELS[model]
        suffix = model.value.upper()
        _require(spec.provider_key_var)
        return cls(
            baseline_model=model,
            spec=spec,
            api_key=_require(f"EM_API_KEY_{suffix}"),
            webhook_secret=_require(f"EM_WEBHOOK_SECRET_{suffix}"),
            api_base_url=os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
        )


def baseline_model_from_env() -> BaselineModel:
    raw = os.environ.get("BASELINE_MODEL")
    if not raw:
        raise RuntimeError(
            "BASELINE_MODEL is not set. Deploy with e.g. "
            "`BASELINE_MODEL=luna uv run modal deploy modal_app.py`."
        )
    try:
        return BaselineModel(raw)
    except ValueError:
        valid = ", ".join(m.value for m in BaselineModel)
        raise RuntimeError(f"BASELINE_MODEL={raw!r} is invalid; expected one of: {valid}") from None


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file "
            f"(copy .env.example to .env), then re-deploy. See the README."
        )
    return value
