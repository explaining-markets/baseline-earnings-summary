"""Configuration read from the environment.

One codebase powers two Modal deployments; ``BASELINE_MODEL`` selects which.
At deploy time it comes from your shell (``BASELINE_MODEL=gemini uv run modal
deploy modal_app.py``) and is baked into the container image, so at runtime the
same variable is simply read back from the environment.

Both deployments share a single ``.env`` (loaded by Modal at deploy time via
``Secret.from_dotenv``); the competition credentials are suffixed per baseline
and the active pair is picked here.

Required:
  BASELINE_MODEL               "gpt5nano" or "gemini"
  EM_API_KEY_{SUFFIX}          submission API key; sent as the X-API-Key header
  EM_WEBHOOK_SECRET_{SUFFIX}   signing secret (whsec_...) for incoming webhooks
  OPENAI_API_KEY               when BASELINE_MODEL=gpt5nano
  GEMINI_API_KEY               when BASELINE_MODEL=gemini

Optional:
  EM_API_BASE_URL              API base URL (default: beta)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_API_BASE_URL = "https://api-beta.explainingmarkets.ai/v1"


class BaselineModel(StrEnum):
    GPT5NANO = "gpt5nano"
    GEMINI = "gemini"


# LiteLLM model strings (DSPy routes through LiteLLM). Gemini deliberately
# floats on the -latest alias.
LM_MODELS: dict[BaselineModel, str] = {
    BaselineModel.GPT5NANO: "openai/gpt-5-nano-2025-08-07",
    BaselineModel.GEMINI: "gemini/gemini-flash-lite-latest",
}

# Provider API key each baseline needs at prediction time (read by LiteLLM).
PROVIDER_KEY_VARS: dict[BaselineModel, str] = {
    BaselineModel.GPT5NANO: "OPENAI_API_KEY",
    BaselineModel.GEMINI: "GEMINI_API_KEY",
}


@dataclass(frozen=True)
class Config:
    baseline_model: BaselineModel
    lm_model: str
    api_key: str
    webhook_secret: str
    api_base_url: str

    @classmethod
    def from_env(cls) -> Config:
        """Load and validate config. Raises if a required var is missing."""
        model = baseline_model_from_env()
        suffix = model.value.upper()
        _require(PROVIDER_KEY_VARS[model])
        return cls(
            baseline_model=model,
            lm_model=LM_MODELS[model],
            api_key=_require(f"EM_API_KEY_{suffix}"),
            webhook_secret=_require(f"EM_WEBHOOK_SECRET_{suffix}"),
            api_base_url=os.environ.get("EM_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/"),
        )


def baseline_model_from_env() -> BaselineModel:
    raw = os.environ.get("BASELINE_MODEL")
    if not raw:
        raise RuntimeError(
            "BASELINE_MODEL is not set. Deploy with e.g. "
            "`BASELINE_MODEL=gpt5nano uv run modal deploy modal_app.py`."
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
