# Explaining Markets — earnings-summary baselines

[![CI](https://github.com/explaining-markets/baseline-earnings-summary/actions/workflows/ci.yml/badge.svg)](https://github.com/explaining-markets/baseline-earnings-summary/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Two vanilla baseline participants for the [Explaining Markets](https://explainingmarkets.ai)
competition. Each receives the disseminated earnings-call fact summaries via
webhook and predicts the stock's post-call return percentile with a single
LLM call — a direct port of the research pipeline's DSPy program:

| Deployment | Model | Credentials |
|---|---|---|
| `em-baseline-gpt5nano` | `openai/gpt-5-nano-2025-08-07` | `EM_*_GPT5NANO` |
| `em-baseline-gemini` | `gemini/gemini-flash-lite-latest` | `EM_*_GEMINI` |

One codebase, two [Modal](https://modal.com) deployments: the `BASELINE_MODEL`
environment variable — read from your shell at deploy time and baked into the
image — selects the model, the credential pair, and every Modal resource name.

## How it works

```
competition ──POST──▶ web (ASGI)                     process_event (worker)
                        verify HMAC signature          fetch DisclosureBundle
                        dedupe on Webhook-Id           extract + format facts
                        spawn worker ──────────────▶   DSPy ChainOfThought
                        ACK 200 (< 1s)                 POST /predictions
```

The webhook handler ACKs immediately and does the slow work in a spawned
worker because the competition's delivery POST times out after **10 seconds**,
while a gpt-5-nano reasoning call regularly takes longer. The per-event
prediction deadline (5 minutes) starts at the ACK, so the worker has the full
window. Delivery retries are deduped on the `Webhook-Id` header via a
persistent `modal.Dict`.

Prediction behavior (`src/em_baseline/`):

- **Normal path** — the bundle's facts are rendered as a bullet list and fed
  to `dspy.ChainOfThought(PredictEarningsReturn)`, the exact signature and
  prompt used in the research pipeline. Only `predicted_percentile` is
  submitted (clamped to [0, 1], `/100` if the model answers on a 0–100
  scale); the class and rationale are logged.
- **No usable facts** in the bundle → submit a neutral **0.5**.
- **Unparseable model output** → submit a neutral **0.5**.
- **Transient failures** (provider outage, bundle fetch, competition 5xx) →
  retried with backoff; if exhausted, nothing is submitted and the failed
  worker call is visible in the Modal dashboard.
- Only the **first focal asset** is predicted on (the competition currently
  disseminates exactly one per event).
- The portal's synthetic `TEST` events are ACKed but never predicted on.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Modal account.

```bash
uv sync
uv run modal setup          # first time only: authenticate with Modal
cp .env.example .env        # then fill in credentials (see below)
```

`.env` is shared by both deployments and loaded by Modal at deploy time
(`Secret.from_dotenv`) — there is no separate secret-store step. It needs:

- `EM_API_KEY_GPT5NANO` / `EM_WEBHOOK_SECRET_GPT5NANO` — from the gpt-5-nano
  submission's Credentials tab in the portal
- `EM_API_KEY_GEMINI` / `EM_WEBHOOK_SECRET_GEMINI` — same, for the Gemini
  submission
- `OPENAI_API_KEY` and `GEMINI_API_KEY` — model provider keys

## Deploy

```bash
BASELINE_MODEL=gpt5nano uv run modal deploy modal_app.py
BASELINE_MODEL=gemini   uv run modal deploy modal_app.py
```

Each deploy prints a persistent public URL like
`https://<workspace>--em-baseline-gpt5nano.modal.run`. Paste each URL, as-is,
into the **matching** submission's webhook field in the portal, then use the
portal's *Send test event* button — the handler ACKs it with 200 without
submitting a prediction.

For local iteration use `modal serve` (hot-reloads on save):

```bash
BASELINE_MODEL=gemini uv run modal serve modal_app.py
```

## Tests

```bash
uv run pytest              # offline suite (default) — no keys, no network
uv run pytest -m live      # + live calls to OpenAI and Gemini (spends credits)
```

The offline suite covers the full code path — HMAC verification against the
competition's frozen test vectors, signed requests through the FastAPI app,
bundle parsing, fallback policy, and retry behavior — with the LLM boundary
stubbed. The live suite runs the same worker path with real provider calls;
the competition API is always mocked (provider keys are read from `.env`).

Lint / format / types:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

CI runs all of the above (offline tests only) on every push and PR.

## Repo layout

```
modal_app.py                     Modal entrypoint: app, image, web + worker functions
src/em_baseline/
  config.py                      BASELINE_MODEL → model string + credential pair
  webhook_app.py                 FastAPI factory: verify → dedupe → spawn → ACK
  worker.py                      fetch facts → predict → submit (+ retry policy)
  bundle.py                      DisclosureBundle fetch/parse/format
  predictor.py                   DSPy signature + prediction (research-pipeline port)
  client.py                      POST /predictions client
  event_utils.py                 payload helpers
  webhook_verification.py        vendored Standard-Webhooks HMAC verifier
tests/                           offline suite + `-m live` provider integration
```
