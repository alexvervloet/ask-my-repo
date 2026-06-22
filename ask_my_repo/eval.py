"""Step 6 — eval runner: two retrieval-quality tracks against a gold set.

Run this right after retrieval works and before tuning anything. Every tuning
change (k, chunk size, the class-split threshold, prepending signatures to
embeddings) should move these numbers rather than be guessed at.

Gold set format: one JSON object per line (JSONL):

    {"question": "How are Python files discovered?",
     "relevant_file": "ask_my_repo/walker.py",
     "relevant_symbols": ["walk_python_files"]}

Two independent signals are measured and reported separately — never collapsed
into one score, because a coarse file hit and a precise symbol hit mean
different things:

  * Symbol track (strict): a chunk matches a symbol by exact qualname or by
    qualname ending in `.<symbol>` ("walk_python_files" matches the full
    "ask_my_repo.walker.walk_python_files"). Path is ignored here.
  * File track (coarse): a chunk matches by `relevant_file` appearing as a
    substring of its path. Qualname is ignored here, and a file match never
    counts toward symbol recall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG
from .retrieval import RetrievedChunk, retrieve_many


@dataclass(frozen=True)
class GoldItem:
    question: str
    relevant_file: str
    relevant_symbols: list[str]


def load_gold(path: str | Path) -> list[GoldItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        items.append(
            GoldItem(
                question=obj["question"],
                relevant_file=obj["relevant_file"],
                relevant_symbols=list(obj["relevant_symbols"]),
            )
        )
    return items


# Two independent matching signals, never combined: the symbol track measures
# whether retrieval found the right *definitions* (strict qualname matching),
# the file track measures whether it surfaced the right *file* at all (path
# matching). A chunk from the right file does NOT count toward symbol recall.


def _chunk_matches_symbol(symbol: str, chunk: RetrievedChunk) -> bool:
    return chunk.qualname == symbol or chunk.qualname.endswith(f".{symbol}")


def _chunk_in_file(relevant_file: str, chunk: RetrievedChunk) -> bool:
    return relevant_file in chunk.path


def _first_rank(chunks: list[RetrievedChunk], pred) -> int | None:
    for rank, c in enumerate(chunks, 1):
        if pred(c):
            return rank
    return None


@dataclass(frozen=True)
class EvalResult:
    k: int
    n_questions: int
    # Symbol track — strict qualname matching against relevant_symbols.
    symbol_recall_at_k: float  # mean over questions of (symbols found / symbols total)
    symbol_hit_at_k: float  # fraction of questions with >= 1 symbol matched
    symbol_mrr: float  # mean reciprocal rank of first qualname-matching chunk
    # File track — path matching against relevant_file (one file per question).
    file_hit_at_k: float  # fraction of questions with >= 1 chunk from the file
    file_mrr: float  # mean reciprocal rank of first chunk from the file
    per_question: list[dict]

    def summary(self) -> str:
        return (
            f"k={self.k}  n={self.n_questions}\n"
            f"  symbol  recall@{self.k}={self.symbol_recall_at_k:.3f}  "
            f"hit@{self.k}={self.symbol_hit_at_k:.3f}  mrr={self.symbol_mrr:.3f}\n"
            f"  file    hit@{self.k}={self.file_hit_at_k:.3f}  "
            f"mrr={self.file_mrr:.3f}"
        )


def evaluate(
    gold: list[GoldItem],
    k: int | None = None,
    *,
    dsn: str | None = None,
) -> EvalResult:
    k = k or CONFIG.default_k
    questions = [g.question for g in gold]
    retrieved = retrieve_many(questions, k, dsn=dsn)

    sym_recalls, sym_hits, sym_rr = [], [], []
    file_hits, file_rr = [], []
    per_question = []
    for item, chunks in zip(gold, retrieved):
        # Symbol track (qualname only).
        found = sum(
            1
            for s in item.relevant_symbols
            if any(_chunk_matches_symbol(s, c) for c in chunks)
        )
        total = len(item.relevant_symbols) or 1
        sym_recall = found / total
        sym_rank = _first_rank(
            chunks, lambda c: any(_chunk_matches_symbol(s, c) for s in item.relevant_symbols)
        )

        # File track (path only).
        file_rank = _first_rank(chunks, lambda c: _chunk_in_file(item.relevant_file, c))

        sym_recalls.append(sym_recall)
        sym_hits.append(1.0 if found > 0 else 0.0)
        sym_rr.append(1.0 / sym_rank if sym_rank else 0.0)
        file_hits.append(1.0 if file_rank else 0.0)
        file_rr.append(1.0 / file_rank if file_rank else 0.0)
        per_question.append(
            {
                "question": item.question,
                "symbol_recall": sym_recall,
                "symbol_hit": found > 0,
                "symbol_first_rank": sym_rank,
                "file_hit": file_rank is not None,
                "file_first_rank": file_rank,
                "top": [c.qualname for c in chunks],
            }
        )

    n = len(gold) or 1
    return EvalResult(
        k=k,
        n_questions=len(gold),
        symbol_recall_at_k=sum(sym_recalls) / n,
        symbol_hit_at_k=sum(sym_hits) / n,
        symbol_mrr=sum(sym_rr) / n,
        file_hit_at_k=sum(file_hits) / n,
        file_mrr=sum(file_rr) / n,
        per_question=per_question,
    )


def sweep_k(
    gold: list[GoldItem],
    ks: list[int],
    *,
    dsn: str | None = None,
) -> list[EvalResult]:
    """Evaluate at several k values so you can see recall climb with k."""
    return [evaluate(gold, k, dsn=dsn) for k in ks]
