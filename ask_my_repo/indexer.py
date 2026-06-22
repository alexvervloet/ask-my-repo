"""Step 3 — indexer + schema (Postgres + pgvector).

Persists chunks and their embeddings to Postgres. The deterministic `chunk_id`
is the primary-key column, so re-indexing an unchanged chunk is an idempotent
upsert and retrieval joins back to full metadata by id.

Embeddings live in a `vector(N)` column with an HNSW cosine index, so retrieval
is a SQL nearest-neighbour query (`ORDER BY embedding <=> q`) rather than an
in-process matrix scan. The embedding dimension is discovered from the first
batch and the table is created to match; mixing embedders of different
dimensions requires `--reset`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from . import client
from .chunker import Chunk, iter_repo_chunks
from .config import CONFIG
from .walker import walk_python_files

_INSERT_COLS = (
    "chunk_id, path, qualname, kind, start_line, end_line, "
    "signature, code, embedding, embed_model"
)


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Open a connection, ensure the pgvector extension, register the type."""
    conn = psycopg.connect(dsn or CONFIG.database_url, row_factory=dict_row)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def table_exists(conn: psycopg.Connection) -> bool:
    row = conn.execute("SELECT to_regclass('chunks') AS t").fetchone()
    return row is not None and row["t"] is not None


def ensure_schema(conn: psycopg.Connection, dim: int) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    text PRIMARY KEY,
            path        text NOT NULL,
            qualname    text NOT NULL,
            kind        text NOT NULL,
            start_line  integer NOT NULL,
            end_line    integer NOT NULL,
            signature   text NOT NULL,
            code        text NOT NULL,
            embedding   vector({dim}) NOT NULL,
            embed_model text NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_qualname ON chunks(qualname)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, value text NOT NULL)"
    )
    conn.commit()


def _set_meta(conn: psycopg.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (%s, %s) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _get_meta(conn: psycopg.Connection, key: str) -> str | None:
    if not table_exists(conn):
        return None
    row = conn.execute("SELECT value FROM meta WHERE key = %s", (key,)).fetchone()
    return row["value"] if row else None


def _embed_label() -> str:
    """Best-effort label of which embedder produced the stored vectors."""
    return f"lmstudio:{CONFIG.lmstudio_embed_model}|voyage:{CONFIG.voyage_embed_model}"


def upsert_chunks(
    conn: psycopg.Connection,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    label = _embed_label()
    rows = [
        (
            ch.chunk_id,
            ch.path,
            ch.qualname,
            ch.kind,
            ch.start_line,
            ch.end_line,
            ch.signature,
            ch.code,
            np.asarray(emb, dtype=np.float32),
            label,
        )
        for ch, emb in zip(chunks, embeddings)
    ]
    conn.cursor().executemany(
        f"""
        INSERT INTO chunks ({_INSERT_COLS})
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(chunk_id) DO UPDATE SET
            path=excluded.path,
            qualname=excluded.qualname,
            kind=excluded.kind,
            start_line=excluded.start_line,
            end_line=excluded.end_line,
            signature=excluded.signature,
            code=excluded.code,
            embedding=excluded.embedding,
            embed_model=excluded.embed_model
        """,
        rows,
    )
    conn.commit()


def _batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def index_repo(
    root: str | Path,
    *,
    dsn: str | None = None,
    batch_size: int = 64,
    reset: bool = False,
) -> dict:
    """Walk, chunk, embed, and persist every Python file under `root`.

    The table is created on the first batch, sized to the embedder's dimension.
    Re-indexing with an embedder of a different dimension requires `reset=True`.
    Returns a summary dict (files, chunks, embed_dim).
    """
    root = Path(root)
    conn = connect(dsn)
    try:
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute("DROP TABLE IF EXISTS meta")
            conn.commit()

        files = list(walk_python_files(root))
        chunks = list(iter_repo_chunks(files, root))

        embed_dim = 0
        schema_ready = False
        for batch in _batched(chunks, batch_size):
            vecs = client.embed([c.embedding_text() for c in batch])
            if not vecs:
                continue
            if not schema_ready:
                embed_dim = len(vecs[0])
                _guard_dimension(conn, embed_dim)
                ensure_schema(conn, embed_dim)
                _set_meta(conn, "embed_dim", str(embed_dim))
                conn.commit()
                schema_ready = True
            upsert_chunks(conn, batch, vecs)

        if schema_ready:
            _set_meta(conn, "root", str(root.resolve()))
            conn.commit()

        return {"files": len(files), "chunks": len(chunks), "embed_dim": embed_dim}
    finally:
        conn.close()


def _guard_dimension(conn: psycopg.Connection, new_dim: int) -> None:
    existing = _get_meta(conn, "embed_dim")
    if existing is not None and int(existing) != new_dim:
        raise ValueError(
            f"existing index is {existing}-dim but this embedder produces "
            f"{new_dim}-dim vectors; re-run with reset=True (CLI: --reset)"
        )


def count_chunks(conn: psycopg.Connection) -> int:
    if not table_exists(conn):
        return 0
    return conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
