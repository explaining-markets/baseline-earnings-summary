"""Fetch and parse the ``DisclosureBundle`` behind an event's ``information_url``.

The webhook payload carries a short-lived signed URL. GETting it returns a
versioned envelope::

    {
      "schema_version": "1.0",
      "event_id": "...",
      "generated_at": "...",
      "items": [
        {
          "id": "earnings-call-facts",
          "kind": "facts",              # discriminator; implies content shape
          "source": "earnings_call",
          "media_type": "application/json",
          "content": ["<fact 1>", ...]  # inline list[str] for kind=facts
        },
        {
          "id": "earnings-preview",     # optional — absent (not null) when
          "kind": "text",               # no preview was produced
          "source": "claude_code_web_research",
          "media_type": "text/markdown",
          "content": "# NVIDIA (NVDA) — Q2 FY2027 Earnings Preview\\n..."
        },
        {
          "id": "option-implied-stats", # optional — 2026Q3 onward;
          "kind": "stats",              # content is an OBJECT
          "source": "option_market",
          "media_type": "application/json",
          "content": {
            "as_of": "2026-10-05T19:36:12Z",
            "methodology": "v1",
            "implied_earnings_volatility":    {"value": 0.0847, "status": "ok"},
            "implied_absolute_earnings_move": {"value": 0.0676, "status": "ok"},
            "skew_25_delta":                  {"value": null, "status": "illiquid_wings"}
          }
        }
      ]
    }

Items are selected by ``id``/``kind``, never by position. The facts item is
always present for earnings events; the earnings preview — an agent-written
research note assembled from public sources *before* the release — is
optional, and a bundle without it is normal (TEST events never carry one).
Both are inline today, but items of unknown ``kind`` — or a future by-ref
item — are tolerated and skipped rather than treated as errors.

The option-implied statistics are three numbers derived from the listed options
market before the close that opens the event's return window. They are
absent when a stock has no listed options or they trade too thinly to measure.
:func:`extract_option_stats` reads them, but THE BASELINES DO NOT USE THEM: the
prompt is built from the facts and the preview only, so the item's arrival
changes nothing about what a baseline predicts. Feeding them to the model would
be a change to the baselines themselves, to be made deliberately and measured.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 15.0

FACTS_KIND = "facts"
PREVIEW_ID = "earnings-preview"
PREVIEW_KIND = "text"
OPTION_STATS_ID = "option-implied-stats"
OPTION_STATS_KIND = "stats"
OPTION_STATS_FIELDS = (
    "implied_earnings_volatility",
    "implied_absolute_earnings_move",
    "skew_25_delta",
)


def fetch_bundle(information_url: str, *, timeout: float = FETCH_TIMEOUT_SECONDS) -> dict:
    """GET the disclosure bundle. Raises ``httpx.HTTPError`` on network/HTTP
    failure (transient — the caller decides whether to retry)."""
    resp = httpx.get(information_url, timeout=timeout)
    resp.raise_for_status()
    bundle = resp.json()
    if not isinstance(bundle, dict):
        raise ValueError(f"disclosure bundle is not a JSON object: {type(bundle).__name__}")
    return bundle


def extract_facts(bundle: dict) -> list[str] | None:
    """Pull the earnings-call facts out of a bundle.

    Returns the list of fact strings, or ``None`` when the bundle carries no
    usable facts (no ``kind="facts"`` item, a non-inline item, or an empty
    list). ``None`` is the caller's cue to fall back to a neutral prediction —
    it is deliberately not an exception, because retrying won't grow facts.
    """
    items = bundle.get("items")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or item.get("kind") != FACTS_KIND:
            continue
        content = item.get("content")
        if isinstance(content, list) and content:
            return [str(fact) for fact in content]
        logger.warning(
            "facts item unusable (content=%r, url=%r) — treating as no facts",
            content,
            item.get("url"),
        )
        return None
    return None


def extract_preview(bundle: dict) -> str | None:
    """Pull the earnings preview (markdown) out of a bundle, verbatim.

    Returns the preview text, or ``None`` when the bundle carries no usable
    preview: no ``id="earnings-preview"`` / ``kind="text"`` item, a non-inline
    item, or blank content. ``None`` is the normal case for events without a
    preview — the caller substitutes a note saying so, and the prompt tells the
    model to rely on the facts alone.
    """
    items = bundle.get("items")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("id") != PREVIEW_ID or item.get("kind") != PREVIEW_KIND:
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content
        # Log the shape, never the body (a preview runs to ~16k characters).
        logger.warning(
            "preview item unusable (content type=%s, url=%r) — treating as no preview",
            type(content).__name__,
            item.get("url"),
        )
        return None
    return None


def extract_option_stats(bundle: dict) -> dict[str, float | None] | None:
    """Pull the option-implied statistics out of a bundle as plain numbers.

    Returns ``{statistic: value}`` for the three statistics, where a statistic
    whose ``status`` is anything but ``"ok"`` is ``None``. Returns ``None`` when
    the bundle carries no usable item: no ``id="option-implied-stats"`` /
    ``kind="stats"`` item, or content that is not an object. That is the normal
    case for a stock without liquid listed options.

    Provided for anyone building on these baselines. The baselines themselves
    never call it (see the module docstring).
    """
    items = bundle.get("items")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("id") != OPTION_STATS_ID or item.get("kind") != OPTION_STATS_KIND:
            continue
        content = item.get("content")
        if not isinstance(content, dict) or not content:
            return None
        values: dict[str, float | None] = {}
        for name in OPTION_STATS_FIELDS:
            block = content.get(name)
            value = block.get("value") if isinstance(block, dict) else None
            is_ok = isinstance(block, dict) and block.get("status") == "ok"
            if is_ok and isinstance(value, int | float) and not isinstance(value, bool):
                values[name] = float(value)
            else:
                values[name] = None
        return values
    return None


def format_facts(facts: list[str]) -> str:
    """Render facts as the bullet list the prediction prompt expects.

    Matches the research pipeline's formatting exactly ("- <fact>" lines).
    """
    return "\n".join(f"- {fact}" for fact in facts)
