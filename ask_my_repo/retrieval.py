"""Step 4 — retrieval (Postgres + pgvector).

`retrieve(question, k)` embeds the question through the model seam and runs a
pgvector nearest-neighbour query (`ORDER BY embedding <=> q`) to return the
top-k chunks as metadata. Built standalone and clean because the eval depends on
it and every tuning change is measured through it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg

from . import client
from .config import CONFIG
from .indexer import connect, table_exists

# Cosine distance operator `<=>`; similarity = 1 - distance.
_QUERY = """
    SELECT chunk_id, path, qualname, kind, start_line, end_line,
           signature, code, 1 - (embedding <=> %s) AS score
    FROM chunks
    ORDER BY embedding <=> %s
    LIMIT %s
"""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    path: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    code: str
    score: float

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "qualname": self.qualname,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "score": self.score,
        }


def _query(conn: psycopg.Connection, qvec: np.ndarray, k: int) -> list[RetrievedChunk]:
    rows = conn.execute(_QUERY, (qvec, qvec, k)).fetchall()
    return [_to_chunk(r) for r in rows]


def retrieve(
    question: str,
    k: int | None = None,
    *,
    dsn: str | None = None,
    conn: psycopg.Connection | None = None,
) -> list[RetrievedChunk]:
    """Return the top-`k` chunks most similar to `question`.

    Pass either a `dsn` or an open `conn` (the eval reuses one connection across
    many queries via `retrieve_many`).
    """
    k = k or CONFIG.default_k
    owns_conn = conn is None
    conn = conn or connect(dsn)
    try:
        if not table_exists(conn):
            return []
        qvec = np.asarray(client.embed([question])[0], dtype=np.float32)
        return _query(conn, qvec, k)
    finally:
        if owns_conn:
            conn.close()


def retrieve_many(
    questions: list[str],
    k: int | None = None,
    *,
    dsn: str | None = None,
) -> list[list[RetrievedChunk]]:
    """Retrieve for many questions over one connection.

    The eval runner uses this so a recall@k sweep opens the DB once. Question
    embeddings are computed in a single `embed()` call; each NN query then hits
    the pgvector HNSW index.
    """
    k = k or CONFIG.default_k
    conn = connect(dsn)
    try:
        if not table_exists(conn) or not questions:
            return [[] for _ in questions]
        qvecs = client.embed(questions)
        return [
            _query(conn, np.asarray(qv, dtype=np.float32), k) for qv in qvecs
        ]
    finally:
        conn.close()


def _to_chunk(row: dict) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        path=row["path"],
        qualname=row["qualname"],
        kind=row["kind"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        signature=row["signature"],
        code=row["code"],
        score=float(row["score"]),
    )
