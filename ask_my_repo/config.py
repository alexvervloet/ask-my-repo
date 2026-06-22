"""Central configuration, driven by environment variables with sane defaults.

Everything tunable lives here so the eval runner can sweep one knob at a time
(k, chunk size, the class-split threshold, prepending signatures to embeddings)
and measure the effect instead of guessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Config:
    # --- Index storage: Postgres + pgvector ---
    # Connection string. Falls back to AMR_DATABASE_URL, then DATABASE_URL,
    # then a local default. The pgvector extension is created automatically.
    database_url: str = (
        os.getenv("AMR_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or "postgresql:///ask_my_repo"
    )

    # --- Local model (LM Studio, OpenAI-compatible server) ---
    # Mirrors the good-news-feed project: LM Studio is bound to 0.0.0.0 on the
    # PC's LAN, addressed by PC_HOST (set in the shell/.env to the PC's IP, e.g.
    # 192.168.1.106). AMR_LMSTUDIO_URL overrides the whole URL if set. Model ids
    # are the EXACT ids shown in the LM Studio server panel.
    lmstudio_base_url: str = os.getenv(
        "AMR_LMSTUDIO_URL",
        f"http://{os.getenv('PC_HOST', '127.0.0.1')}:1234/v1",
    )
    lmstudio_chat_model: str = os.getenv(
        "AMR_LMSTUDIO_CHAT_MODEL", "unsloth/qwen3.6-35b-a3b"
    )
    lmstudio_embed_model: str = os.getenv(
        "AMR_LMSTUDIO_EMBED_MODEL", "text-embedding-qwen3-embedding-0.6b"
    )
    # How long to wait on the local server before declaring it dead and falling back.
    local_timeout_s: float = float(os.getenv("AMR_LOCAL_TIMEOUT", "5"))

    # --- Foundation fallback: completions (Anthropic Claude) ---
    anthropic_model: str = os.getenv("AMR_ANTHROPIC_MODEL", "claude-opus-4-8")

    # --- Foundation fallback: embeddings (Voyage AI) ---
    # Anthropic has no embeddings endpoint; Voyage is its recommended partner.
    voyage_embed_model: str = os.getenv("AMR_VOYAGE_MODEL", "voyage-3")

    # --- Chunking knobs (measured by the eval, not guessed) ---
    # Classes whose body spans more than this many lines are split into
    # per-method chunks; smaller classes are kept whole.
    class_split_threshold: int = _env_int("AMR_CLASS_SPLIT_THRESHOLD", 40)
    # Prepend the signature (def/class line) to the text we embed. Helps the
    # embedding capture the "what is this" before the "how".
    prepend_signature: bool = _env_bool("AMR_PREPEND_SIGNATURE", True)
    # Prepend the module docstring to each chunk's embedding text. Gives a
    # function chunk the "what is this file about" context it otherwise lacks.
    # Default OFF: measured to help file-level recall and high-k recall but to
    # hurt low-k symbol precision (the shared docstring blurs chunks within a
    # file). See the eval sweep before enabling.
    embed_module_context: bool = _env_bool("AMR_EMBED_MODULE_CONTEXT", False)

    # --- Retrieval knobs ---
    default_k: int = _env_int("AMR_DEFAULT_K", 5)


CONFIG = Config()
