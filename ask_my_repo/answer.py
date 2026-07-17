"""Step 5 — answer a question on top of retrieval.

Retrieves the top-k chunks, builds a grounded prompt, and asks the model (local
first, foundation fallback) to answer using only that context with citations
back to file:line.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from . import client
from .config import CONFIG
from .retrieval import RetrievedChunk, retrieve

SYSTEM_PROMPT = (
    "You are a precise code assistant. Answer the user's question using ONLY the "
    "provided code chunks. Cite the chunks you rely on as `path:start-end`. If the "
    "answer is not present in the chunks, say so plainly instead of guessing."
)


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    chunks: list[RetrievedChunk]

    @property
    def citations(self) -> list[str]:
        return [f"{c.path}:{c.start_line}-{c.end_line}" for c in self.chunks]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.path}:{c.start_line}-{c.end_line}  ({c.qualname})"
        blocks.append(f"{header}\n```python\n{c.code}\n```")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = _format_context(chunks)
    return (
        f"Question:\n{question}\n\n"
        f"Code chunks:\n{context}\n\n"
        "Answer the question grounded in the chunks above, with citations."
    )


def answer(
    question: str,
    k: int | None = None,
    *,
    dsn: str | None = None,
) -> Answer:
    k = k or CONFIG.default_k
    chunks = retrieve(question, k, dsn=dsn)
    if not chunks:
        return Answer(
            question=question,
            text="No indexed code found. Run the indexer first.",
            chunks=[],
        )
    prompt = build_prompt(question, chunks)
    text = client.complete(prompt, system=SYSTEM_PROMPT)
    return Answer(question=question, text=text, chunks=chunks)


def answer_stream(
    question: str,
    k: int | None = None,
    *,
    dsn: str | None = None,
) -> tuple[list[RetrievedChunk], Iterator[str]]:
    """Like `answer()`, but returns the chunks up front and the text as a
    stream of deltas — for callers (a web gateway) that want to render sources
    before the answer finishes."""
    k = k or CONFIG.default_k
    chunks = retrieve(question, k, dsn=dsn)
    if not chunks:
        return [], iter(["No indexed code found. Run the indexer first."])
    prompt = build_prompt(question, chunks)
    return chunks, client.complete_stream(prompt, system=SYSTEM_PROMPT)
