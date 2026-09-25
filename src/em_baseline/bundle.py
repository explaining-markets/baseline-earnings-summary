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
        }
      ]
    }

Items are selected by ``id``/``kind``, never by position. The facts item is
always present for earnings events; the earnings preview — an agent-written
research note assembled from public sources *before* the release — is
optional, and a bundle without it is normal (TEST events never carry one).
Both are inline today, but items of unknown ``kind`` — or a future by-ref
item — are tolerated and skipped rather than treated as errors.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 15.0

FACTS_KIND = "facts"
PREVIEW_ID = "earnings-preview"
PREVIEW_KIND = "text"


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


def format_facts(facts: list[str]) -> str:
    """Render facts as the bullet list the prediction prompt expects.

    Matches the research pipeline's formatting exactly ("- <fact>" lines).
    """
    return "\n".join(f"- {fact}" for fact in facts)
