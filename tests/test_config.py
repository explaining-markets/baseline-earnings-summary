"""Config selection: BASELINE_MODEL picks the model string and credential pair."""

from __future__ import annotations

import pytest

from em_baseline.config import BaselineModel, Config

BASE_ENV = {
    "EM_API_KEY_GPT5NANO": "comp_sk_nano",
    "EM_WEBHOOK_SECRET_GPT5NANO": "whsec_nano",
    "EM_API_KEY_GEMINI": "comp_sk_gemini",
    "EM_WEBHOOK_SECRET_GEMINI": "whsec_gemini",
    "OPENAI_API_KEY": "sk-test",
    "GEMINI_API_KEY": "AI-test",
}


@pytest.fixture
def full_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("EM_API_BASE_URL", raising=False)
    return monkeypatch


def test_gpt5nano_selects_openai_model_and_nano_credentials(full_env: pytest.MonkeyPatch) -> None:
    full_env.setenv("BASELINE_MODEL", "gpt5nano")
    cfg = Config.from_env()
    assert cfg.baseline_model is BaselineModel.GPT5NANO
    assert cfg.lm_model == "openai/gpt-5-nano-2025-08-07"
    assert cfg.api_key == "comp_sk_nano"
    assert cfg.webhook_secret == "whsec_nano"
    assert cfg.api_base_url == "https://api.explainingmarkets.ai/v1"


def test_gemini_selects_gemini_model_and_gemini_credentials(full_env: pytest.MonkeyPatch) -> None:
    full_env.setenv("BASELINE_MODEL", "gemini")
    cfg = Config.from_env()
    assert cfg.baseline_model is BaselineModel.GEMINI
    assert cfg.lm_model == "gemini/gemini-flash-lite-latest"
    assert cfg.api_key == "comp_sk_gemini"
    assert cfg.webhook_secret == "whsec_gemini"


def test_missing_baseline_model_raises(full_env: pytest.MonkeyPatch) -> None:
    full_env.delenv("BASELINE_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="BASELINE_MODEL is not set"):
        Config.from_env()


def test_invalid_baseline_model_raises(full_env: pytest.MonkeyPatch) -> None:
    full_env.setenv("BASELINE_MODEL", "gpt4")
    with pytest.raises(RuntimeError, match="invalid"):
        Config.from_env()


def test_missing_credential_pair_raises(full_env: pytest.MonkeyPatch) -> None:
    full_env.setenv("BASELINE_MODEL", "gemini")
    full_env.delenv("EM_API_KEY_GEMINI")
    with pytest.raises(RuntimeError, match="EM_API_KEY_GEMINI"):
        Config.from_env()


def test_missing_provider_key_raises(full_env: pytest.MonkeyPatch) -> None:
    full_env.setenv("BASELINE_MODEL", "gpt5nano")
    full_env.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Config.from_env()


def test_api_base_url_override_strips_trailing_slash(full_env: pytest.MonkeyPatch) -> None:
    full_env.setenv("BASELINE_MODEL", "gemini")
    full_env.setenv("EM_API_BASE_URL", "https://api.example.test/v1/")
    assert Config.from_env().api_base_url == "https://api.example.test/v1"
