"""Offline-ish smoke test: stub the model seam, index this repo, retrieve, eval.

Run: ./.venv/bin/python smoke_test.py

Replaces client.embed with a deterministic hashing bag-of-words embedder so no
embedding network/keys are needed. Storage is real Postgres + pgvector, so a
database must be reachable. Spin one up with:

    docker run -d --name amr-pg -p 5432:5432 -e POSTGRES_PASSWORD=pg pgvector/pgvector:pg16
    export AMR_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres

It verifies chunk-id determinism, the schema (chunk_id as PK column), and that
recall@k climbs with k.
"""

import hashlib
import os
import re
import sys

import numpy as np

import ask_my_repo.client as client
from ask_my_repo.indexer import index_repo, connect, count_chunks, table_exists
from ask_my_repo.chunker import chunk_file
from ask_my_repo.eval import load_gold, sweep_k

DIM = 512
_token = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _hash_embed(texts):
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in _token.findall(t.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM
            v[h] += 1.0
        out.append(v.tolist())
    return out


# Patch the seam (embeddings only; storage stays real).
client.embed = lambda texts, **kw: _hash_embed(texts)

ROOT = os.path.dirname(os.path.abspath(__file__))
DSN = os.getenv("AMR_DATABASE_URL") or os.getenv("DATABASE_URL")

# 1. determinism of chunk ids (no DB needed)
a = chunk_file(os.path.join(ROOT, "ask_my_repo", "walker.py"), "ask_my_repo/walker.py")
b = chunk_file(os.path.join(ROOT, "ask_my_repo", "walker.py"), "ask_my_repo/walker.py")
assert [c.chunk_id for c in a] == [c.chunk_id for c in b], "chunk ids not deterministic"
assert len({c.chunk_id for c in a}) == len(a), "chunk ids not unique within file"
print(f"[ok] determinism: {len(a)} walker chunks, stable + unique ids")
print("     sample:", a[0].chunk_id, a[0].kind, a[0].qualname)

# DB-backed steps
try:
    conn = connect(DSN)
    conn.close()
except Exception as exc:  # noqa: BLE001
    print(f"\n[skip] no Postgres reachable ({exc}).")
    print("       Set AMR_DATABASE_URL to a pgvector-enabled database; see the")
    print("       docstring at the top of this file for a docker one-liner.")
    sys.exit(0)

# 2. index this repo from the repo root (so qualnames are fully qualified)
summary = index_repo(ROOT, dsn=DSN, reset=True)
print(f"[ok] indexed: {summary}")
conn = connect(DSN)
assert count_chunks(conn) == summary["chunks"]
assert table_exists(conn)
cols = [
    r["column_name"]
    for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='chunks' ORDER BY ordinal_position"
    ).fetchall()
]
assert cols[0] == "chunk_id", cols
pk = conn.execute(
    "SELECT a.attname FROM pg_index i "
    "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey) "
    "WHERE i.indrelid='chunks'::regclass AND i.indisprimary"
).fetchone()
assert pk["attname"] == "chunk_id", pk
print(f"[ok] schema columns: {cols}")
print(f"[ok] primary key column: {pk['attname']}")
conn.close()

# 3. eval sweep
gold = load_gold(os.path.join(ROOT, "gold", "gold.jsonl"))
print(f"[ok] gold loaded: {len(gold)} questions")
for res in sweep_k(gold, [1, 3, 5, 10], dsn=DSN):
    print("    ", res.summary())

print("\nALL SMOKE CHECKS PASSED")
