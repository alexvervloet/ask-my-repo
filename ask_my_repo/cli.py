"""Command-line entry point: index | ask | eval.

Examples:
    python -m ask_my_repo.cli index .
    python -m ask_my_repo.cli ask "How does the local->foundation fallback work?"
    python -m ask_my_repo.cli eval gold/gold.jsonl --ks 1,3,5,10
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import CONFIG


def _cmd_index(args) -> int:
    from .indexer import index_repo

    summary = index_repo(args.root, dsn=args.dsn, reset=args.reset)
    print(
        f"Indexed {summary['files']} files -> {summary['chunks']} chunks "
        f"(embed_dim={summary['embed_dim']}) into {args.dsn or CONFIG.database_url}"
    )
    return 0


def _cmd_ask(args) -> int:
    from .answer import answer

    result = answer(args.question, k=args.k, dsn=args.dsn)
    print(result.text)
    print("\n--- sources ---")
    for c in result.chunks:
        print(f"  {c.path}:{c.start_line}-{c.end_line}  ({c.qualname})  score={c.score:.3f}")
    return 0


def _cmd_eval(args) -> int:
    from .eval import evaluate, load_gold, sweep_k

    gold = load_gold(args.gold)
    if args.ks:
        ks = [int(x) for x in args.ks.split(",")]
        for res in sweep_k(gold, ks, dsn=args.dsn):
            print(res.summary())
    else:
        res = evaluate(gold, k=args.k, dsn=args.dsn)
        print(res.summary())
        if args.verbose:
            for q in res.per_question:
                sym = "OK " if q["symbol_hit"] else "MISS"
                fil = "OK " if q["file_hit"] else "MISS"
                print(
                    f"  sym[{sym}] r={q['symbol_recall']:.2f} rank={q['symbol_first_rank']}  "
                    f"file[{fil}] rank={q['file_first_rank']}  {q['question']}"
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ask_my_repo")
    parser.add_argument(
        "--dsn", default=None, help="Postgres connection string (overrides AMR_DATABASE_URL)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="walk, chunk, embed, persist")
    p_index.add_argument("root", help="repository root to index")
    p_index.add_argument("--reset", action="store_true", help="clear existing chunks first")
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="answer a question over the index")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None)
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="run recall@k against a gold set")
    p_eval.add_argument("gold", help="path to gold JSONL")
    p_eval.add_argument("-k", type=int, default=None)
    p_eval.add_argument("--ks", default=None, help="comma-separated k values to sweep, e.g. 1,3,5,10")
    p_eval.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
