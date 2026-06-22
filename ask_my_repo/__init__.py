"""ask-my-repo: walk, chunk, index, retrieve, and answer questions about Python code.

Pipeline:
    walker   -> find .py files
    chunker  -> split into deterministic, addressable chunks
    indexer  -> embed + persist chunks (chunk_id stored as a column)
    retrieval-> retrieve(question, k) -> chunk metadata
    answer   -> answer(question) on top of retrieval
    eval     -> recall@k against a gold set

All model calls (embeddings + completions) go through `ask_my_repo.client`,
which tries a local LM Studio model first and falls back to a foundation model.
"""

__version__ = "0.1.0"
