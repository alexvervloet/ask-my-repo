# ask-my-repo

Walk, chunk, index, and retrieve Python code to answer questions about it. Built
to be called behind a gateway: every model call tries a **local LM Studio model
first** and **falls back to a foundation model** if the local one is
unreachable, errors, or returns nothing usable.

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `walker.py` | Find `.py` files, skipping venvs/caches/VCS. |
| 2 | `chunker.py` | AST-split into functions/methods/(small) classes/module preambles, each with a **deterministic `chunk_id`** (hash of path + qualname + code). |
| 3 | `indexer.py` | Embed + persist to **Postgres + pgvector**. `chunk_id` is the primary-key **column**; embeddings go in a `vector(N)` column with an HNSW cosine index. |
| 4 | `retrieval.py` | `retrieve(question, k)` → top-k chunk metadata via a pgvector NN query (`ORDER BY embedding <=> q`). |
| 5 | `answer.py` | Grounded answer on top of retrieval, with `path:line` citations. |
| 6 | `eval.py` | recall@k / hit@k / MRR against a gold set. |

## The model-call seam

`client.py` is the only module that talks to a model. It exposes `embed()` and
`complete()`, used by the indexer, retrieval, and answer steps:

| Operation | Local (tried first) | Foundation fallback |
|-----------|---------------------|---------------------|
| `embed()` | LM Studio `/v1/embeddings` | Voyage AI (`voyageai` SDK) |
| `complete()` | LM Studio `/v1/chat/completions` | Anthropic Claude (`anthropic` SDK) |

LM Studio speaks the OpenAI wire format (plain HTTP). Anthropic has no embeddings
endpoint, so the embedding fallback is Voyage (its recommended partner). The
foundation completion model defaults to `claude-opus-4-8`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You need a Postgres with the `pgvector` extension available (the indexer runs
`CREATE EXTENSION IF NOT EXISTS vector` itself). Quickest path:

```bash
docker run -d --name amr-pg -p 5432:5432 -e POSTGRES_PASSWORD=pg \
  pgvector/pgvector:pg16
export AMR_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres
```

Configure via env vars (see `config.py`) — e.g. `AMR_DATABASE_URL` (or
`DATABASE_URL`), `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `AMR_LMSTUDIO_URL`
(default `http://localhost:1234/v1`).

## Use

```bash
# Index a repo (run from its root so qualnames are fully qualified)
python -m ask_my_repo.cli index .

# Ask a question
python -m ask_my_repo.cli ask "How does the local->foundation fallback work?"

# Measure retrieval — do this before tuning anything
python -m ask_my_repo.cli eval gold/gold.jsonl --ks 1,3,5,10
```

## Tuning, measured not guessed

The eval exists so every knob is judged against a number. Each is an env var
(`config.py`); flip it, re-run `eval`, compare:

- `AMR_DEFAULT_K` — retrieval depth.
- `AMR_CLASS_SPLIT_THRESHOLD` — line count above which a class is split into
  per-method chunks instead of kept whole.
- `AMR_PREPEND_SIGNATURE` — prepend the def/class signature to the embedded text.

## Gold set

`gold/gold.jsonl` is one JSON object per line:

```json
{"question": "How is a deterministic chunk id computed?", "relevant": ["compute_chunk_id"]}
```

A retrieved chunk satisfies a target by exact qualname, qualname ending in
`.<target>`, or the target appearing in the chunk path. The shipped gold set
self-indexes this repo, so the eval runs against a real index.

## Smoke check

`smoke_test.py` stubs the *embedding* seam with a deterministic hashing embedder
(no embedding network or keys) but uses real Postgres + pgvector for storage. It
verifies chunk-id determinism, the schema (`chunk_id` as PK column), and that
recall@k climbs with k. Point it at a database and run:

```bash
export AMR_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres
python smoke_test.py
```

If no database is reachable it runs the determinism check and skips the rest.
