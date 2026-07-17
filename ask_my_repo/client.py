"""The single model-call seam.

Indexer, retrieval, and answer steps all call `embed()` and `complete()` here —
nothing else in the codebase talks to a model directly. Each call tries a local
LM Studio model first (OpenAI-compatible server) and falls back to a foundation
model if the local server is unreachable, errors, or returns nothing usable.

    embed()    : LM Studio /v1/embeddings   -> Voyage AI
    complete() : LM Studio /v1/chat/completions -> Anthropic Claude

LM Studio speaks the OpenAI wire format, so we talk to it over plain HTTP. The
Anthropic fallback goes through the official `anthropic` SDK; Voyage through the
official `voyageai` SDK.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import requests

from .config import CONFIG

log = logging.getLogger("ask_my_repo.client")


class ModelError(RuntimeError):
    """Raised when both the local model and the foundation fallback fail."""


# --------------------------------------------------------------------------- #
# Local backend (LM Studio, OpenAI-compatible)
# --------------------------------------------------------------------------- #
def _local_embed(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{CONFIG.lmstudio_base_url}/embeddings",
        json={"model": CONFIG.lmstudio_embed_model, "input": texts},
        timeout=CONFIG.local_timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # OpenAI format returns items out of order in theory; sort by index.
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def _local_complete(prompt: str, system: str | None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        f"{CONFIG.lmstudio_base_url}/chat/completions",
        json={
            "model": CONFIG.lmstudio_chat_model,
            "messages": messages,
            "temperature": 0.0,
        },
        timeout=max(CONFIG.local_timeout_s, 60),
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ModelError("local model returned empty completion")
    return content


# --------------------------------------------------------------------------- #
# Foundation fallbacks
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


@lru_cache(maxsize=1)
def _voyage_client():
    import voyageai

    return voyageai.Client()


def _foundation_embed(texts: list[str]) -> list[list[float]]:
    result = _voyage_client().embed(
        texts, model=CONFIG.voyage_embed_model, input_type="document"
    )
    return result.embeddings


def _foundation_complete(prompt: str, system: str | None) -> str:
    client = _anthropic_client()
    kwargs = {
        "model": CONFIG.anthropic_model,
        "max_tokens": 4096,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return "".join(b.text for b in response.content if b.type == "text")


# --------------------------------------------------------------------------- #
# Public seam
# --------------------------------------------------------------------------- #
def embed(texts: list[str], *, prefer_local: bool | None = None) -> list[list[float]]:
    """Embed a list of texts. Returns one vector per input, order preserved."""
    if not texts:
        return []
    if prefer_local is None:
        prefer_local = CONFIG.prefer_local
    if prefer_local:
        try:
            vecs = _local_embed(texts)
            log.debug("embed: served by LM Studio (%d texts)", len(texts))
            return vecs
        except Exception as exc:  # noqa: BLE001 - any local failure -> fall back
            log.warning("embed: local model failed (%s); falling back to Voyage", exc)
    try:
        vecs = _foundation_embed(texts)
        log.debug("embed: served by Voyage (%d texts)", len(texts))
        return vecs
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"embedding failed on both local and foundation: {exc}") from exc


def complete(prompt: str, *, system: str | None = None, prefer_local: bool | None = None) -> str:
    """Generate a completion for `prompt` with an optional `system` prompt."""
    if prefer_local is None:
        prefer_local = CONFIG.prefer_local
    if prefer_local:
        try:
            out = _local_complete(prompt, system)
            log.debug("complete: served by LM Studio")
            return out
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "complete: local model failed (%s); falling back to Claude", exc
            )
    try:
        out = _foundation_complete(prompt, system)
        log.debug("complete: served by Claude")
        return out
    except Exception as exc:  # noqa: BLE001
        raise ModelError(
            f"completion failed on both local and foundation: {exc}"
        ) from exc
