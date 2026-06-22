# Engineering Learnings

A running log of the non-obvious lessons this project has taught me — things I
worked out by building and measuring, not by reading about them. Each entry
records what I observed, what I changed, what the numbers did, and the general
principle I'm taking forward. In chronological order, oldest first.

Each entry follows the same shape:

- **Context** — what I was doing and why.
- **Finding** — what I observed, with evidence.
- **Action** — what I changed in response.
- **Result** — what happened, in numbers where possible.
- **Takeaway** — the transferable principle.

---

## 2026-06-22 — A gold label that passes for the wrong reason hides a real retrieval gap

**Context.** Tuning the self-indexing eval, one gold entry — *"How are chunk
embeddings encoded and decoded for storage?"* — was labeled with the symbol
[`index_repo`](ask_my_repo/indexer.py) and scored a **hit** at k=10. But
`index_repo` is the orchestration loop; it isn't where embeddings are actually
encoded for storage. That work lives in `upsert_chunks` (the
`np.asarray(emb, dtype=np.float32)` conversion) and `connect` (pgvector's
`register_vector`). The label looked fine because the number looked fine.

**Finding.** The pass was a **false positive**. Inspecting the top-10 for that
question, `index_repo` sat at rank 9 (cosine ~0.613) — close enough to sneak
inside k=10, but only because it's loosely related, not because retrieval found
the storage code. When I repointed the label to the *correct* symbols
(`upsert_chunks`, `connect`), the question began to **miss outright**: those
chunks score ~0.584 to the query, below the rank-5 cutoff (~0.642), and don't
appear in the top 10 at all. The honest label didn't break retrieval — it
*revealed* a gap the wrong label had been papering over. The two-track metric
corroborated it: with the correct label, the question missed on **both** the
symbol *and* the file track at k=5, so this was a genuine retrieval failure, not
an artifact of how the symbol was named.

**Action.** Corrected the gold entry to `upsert_chunks` + `connect` and kept the
resulting miss rather than reverting to the symbol that happened to retrieve.
Logged it as a real finding for tuning (which became the next experiment).
Reverting to `index_repo` would have been gaming the metric: relabeling the
target to whatever the retriever already returns makes the eval agree with the
system by construction, which is worse than useless.

**Result.**

| metric | with `index_repo` (false pass) | with `upsert_chunks`+`connect` (honest) |
|---|---|---|
| symbol recall@10 | 0.962 | 0.885 |
| this question @ k=5 (symbol / file) | — | miss / miss |

The aggregate went *down* — and that was the correct direction, because the
previous number was inflated by a coincidence.

**Takeaway.**

- A "hit" only means as much as the matching rule behind it. A loose criterion (a
  near-enough symbol landing inside k, a path or substring match) can score a pass
  for the wrong reason and quietly hide a real failure. Audit *why* a case passes,
  not just that it does.
- Write the gold label for where the answer *truly* lives, then let the metric
  disagree with the system. The value of an eval is the gap between what you want
  and what you get; relabeling to fit the retriever's current output erases
  exactly the signal you built it to see.
- A metric moving *down* after a correctness fix is not a regression — it can be
  the eval finally telling the truth. Judge a number by whether it's honest, not
  by whether it's high.

---

## 2026-06-22 — A per-item diagnostic can confirm a fix and still hide the regression it causes

**Context.** The self-indexing eval had one stubborn miss: *"How are chunk
embeddings encoded and decoded for storage?"* The gold pointed at
[`upsert_chunks`](ask_my_repo/indexer.py), where the encoding actually happens
(`np.asarray(emb, dtype=np.float32)` plus pgvector's `register_vector`).
Retrieval never surfaced it in the top 10 — the embedder ranked the data classes
and `client.embed` higher. The function's embedding text is SQL and numpy with no
"encode / decode / storage" vocabulary anywhere; that concept lives only in the
*module* docstring, which is why the indexer module chunk retrieved at rank 7
while the function didn't.

**Finding.** A cheap, read-only diagnostic looked promising: embedding
`upsert_chunks` with its module docstring prepended lifted cosine-to-query from
**0.584 to 0.679** — from below the rank-5 cutoff (~0.642) to roughly rank 2. So
I added module-docstring context to every chunk's embedding text as a measured,
toggleable knob ([`AMR_EMBED_MODULE_CONTEXT`](ask_my_repo/config.py)),
re-indexed, and ran the *full* sweep. The target question was fixed (symbol rank
5, file rank 2) — but the symbol track regressed almost everywhere, and a
question that had been perfect, *"How does retrieval score a question against
stored chunks?"*, dropped from symbol rank 1 to a miss. Meanwhile the **file**
track *improved*. Prepending the *same* module docstring to every chunk in a file
makes those chunks look alike: excellent for finding the right file, corrosive to
telling functions within it apart.

**Action.** Reverted the default — kept the knob, off, with the measured tradeoff
documented next to it in [`config.py`](ask_my_repo/config.py) — and rebuilt the
baseline index. What made the tradeoff legible was the eval reporting two signals
separately and never collapsing them: a strict **symbol** track (qualname match)
and a coarse **file** track (path match). They moved in *opposite* directions,
which is the exact fingerprint of "chunks within a file got more similar."

**Result.**

| metric (correctly-labeled gold) | baseline | + module context |
|---|---|---|
| symbol recall@1 | 0.654 | 0.500 |
| symbol MRR@5 | 0.846 | 0.721 |
| symbol recall@5 | 0.885 | 0.846 |
| symbol recall@10 | 0.885 | 0.923 |
| file hit@3 | 0.923 | 1.000 |
| file MRR | 0.934 | 0.962 |

Net-negative at the small *k* a RAG system actually feeds the model; shelved as a
knob rather than shipped.

**Takeaway.**

- A diagnostic that scores one item in isolation can confirm the fix works *for
  that item* and be completely blind to what the same change does everywhere
  else. A global change demands a global measurement — the whole set, before and
  after — not a spot check on the case that motivated it.
- Keeping two quality signals separate paid off twice over. The first time it
  stopped a false positive (a gold answer that "passed" only because an unrelated
  chunk coincidentally retrieved); this time the *divergence* between the two
  numbers diagnosed the mechanism, not just the magnitude. A single blended score
  would have nudged a little and explained nothing.
- Make tuning changes reversible knobs, default them to the measured-better
  setting, and write the number that justified the default right beside it — so
  the next person doesn't re-run the experiment to discover it was already tried.
- Shared text across embedded chunks is a double-edged tool: context that lifts
  coarse (file) retrieval can erase the fine distinctions that precise (symbol)
  retrieval depends on. Add it where chunks are context-poor, not everywhere.
