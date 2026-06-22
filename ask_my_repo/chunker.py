"""Step 2 — split Python source into addressable chunks.

Each chunk is a function, a method, a (small) class, or the module preamble.
Every chunk carries a *deterministic* `chunk_id`: a hash of the relative path,
the fully-qualified name, and the chunk's source text. The same code at the same
qualified name produces the same id across runs and machines, so re-indexing an
unchanged file is idempotent and ids are stable to reference from a gold set.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass

from .config import CONFIG
from .walker import read_source


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    path: str  # relative path as supplied by the caller
    qualname: str  # e.g. "module.ClassName.method"
    kind: str  # "function" | "method" | "class" | "module"
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    signature: str  # the def/class line(s), or "" for module chunks
    code: str  # the chunk's source text
    module_doc: str = ""  # the chunk's module docstring (for embedding context)

    def as_dict(self) -> dict:
        return asdict(self)

    def embedding_text(self) -> str:
        """The text we hand to the embedder.

        Composes up to three parts, each a measured knob:
          * the module docstring, for file-level "what is this about" context
            (AMR_EMBED_MODULE_CONTEXT) — skipped for module chunks, which
            already contain it;
          * the signature, so the vector leads with "what this is"
            (AMR_PREPEND_SIGNATURE);
          * the chunk's source code.
        """
        parts: list[str] = []
        if CONFIG.embed_module_context and self.module_doc and self.kind != "module":
            parts.append(self.module_doc)
        if CONFIG.prepend_signature and self.signature:
            parts.append(self.signature)
        parts.append(self.code)
        return "\n\n".join(parts)


def compute_chunk_id(path: str, qualname: str, code: str) -> str:
    """Deterministic id: sha256 over (path, qualname, code), hex-truncated."""
    h = hashlib.sha256()
    h.update(path.encode("utf-8"))
    h.update(b"\x00")
    h.update(qualname.encode("utf-8"))
    h.update(b"\x00")
    h.update(code.encode("utf-8"))
    return h.hexdigest()[:16]


def _module_name(path: str) -> str:
    name = path.replace("\\", "/")
    if name.endswith(".py"):
        name = name[:-3]
    if name.endswith("/__init__"):
        name = name[: -len("/__init__")]
    return name.strip("/").replace("/", ".")


def _segment(source_lines: list[str], node: ast.AST) -> tuple[str, int, int]:
    """Return (text, start_line, end_line) covering a node, including decorators."""
    start = node.lineno
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list)
    end = node.end_lineno or node.lineno
    text = "\n".join(source_lines[start - 1 : end])
    return text, start, end


def _signature(source_lines: list[str], node: ast.AST) -> str:
    """The header line(s): from `def`/`class` up to and including the `:`."""
    start = node.lineno
    end = node.end_lineno or node.lineno
    for i in range(start - 1, end):
        if source_lines[i].rstrip().endswith(":"):
            end = i + 1
            break
    return "\n".join(source_lines[start - 1 : end]).strip()


def _class_body_span(node: ast.ClassDef) -> int:
    end = node.end_lineno or node.lineno
    return end - node.lineno + 1


def chunk_source(path: str, source: str) -> list[Chunk]:
    """Parse `source` (whose logical path is `path`) into chunks.

    Files that fail to parse yield a single whole-file "module" chunk so nothing
    silently disappears from the index.
    """
    lines = source.splitlines()
    modname = _module_name(path)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        whole = source
        return [
            Chunk(
                chunk_id=compute_chunk_id(path, modname, whole),
                path=path,
                qualname=modname,
                kind="module",
                start_line=1,
                end_line=len(lines) or 1,
                signature="",
                code=whole,
            )
        ]

    module_doc = ast.get_docstring(tree) or ""

    chunks: list[Chunk] = []

    def emit(qualname: str, kind: str, node: ast.AST) -> None:
        code, start, end = _segment(lines, node)
        sig = _signature(lines, node) if kind != "module" else ""
        chunks.append(
            Chunk(
                chunk_id=compute_chunk_id(path, qualname, code),
                path=path,
                qualname=qualname,
                kind=kind,
                start_line=start,
                end_line=end,
                signature=sig,
                code=code,
                module_doc=module_doc,
            )
        )

    func_types = (ast.FunctionDef, ast.AsyncFunctionDef)

    # Module preamble: everything before the first top-level def/class
    # (imports, constants, module docstring). Captured so "where is X imported"
    # style questions have something to hit.
    body_defs = [n for n in tree.body if isinstance(n, (*func_types, ast.ClassDef))]
    preamble_end = body_defs[0].lineno - 1 if body_defs else len(lines)
    preamble = "\n".join(lines[:preamble_end]).strip()
    if preamble:
        emit(modname, "module", _PreambleNode(preamble_end))

    for node in tree.body:
        if isinstance(node, func_types):
            emit(f"{modname}.{node.name}", "function", node)
        elif isinstance(node, ast.ClassDef):
            class_qual = f"{modname}.{node.name}"
            if _class_body_span(node) > CONFIG.class_split_threshold:
                # Large class: emit a header chunk + one chunk per method.
                _emit_class_split(emit, lines, class_qual, node, func_types)
            else:
                emit(class_qual, "class", node)

    return chunks


class _PreambleNode:
    """Minimal node-like object so module preambles flow through `_segment`."""

    def __init__(self, end_line: int) -> None:
        self.lineno = 1
        self.end_lineno = end_line
        self.decorator_list: list = []


def _emit_class_split(emit, lines, class_qual, node, func_types) -> None:
    """Emit a class header chunk plus one chunk per method (large classes only)."""
    methods = [n for n in node.body if isinstance(n, func_types)]

    class _Header:
        def __init__(self) -> None:
            self.lineno = (
                min(d.lineno for d in node.decorator_list)
                if node.decorator_list
                else node.lineno
            )
            # Header runs up to the first method, or the whole class if no methods.
            self.end_lineno = (
                methods[0].lineno - 1 if methods else (node.end_lineno or node.lineno)
            )
            self.decorator_list: list = []

    emit(class_qual, "class", _Header())
    for m in methods:
        emit(f"{class_qual}.{m.name}", "method", m)


def chunk_file(path: str, logical_path: str | None = None) -> list[Chunk]:
    """Read and chunk a file on disk. `logical_path` overrides the recorded path
    (use a repo-relative path so chunk ids are portable)."""
    source = read_source(path)
    return chunk_source(logical_path or str(path), source)


def iter_repo_chunks(files, root) -> Iterator[Chunk]:
    """Chunk an iterable of file paths, recording paths relative to `root`."""
    from pathlib import Path

    root = Path(root)
    for f in files:
        rel = str(Path(f).resolve().relative_to(root.resolve()))
        yield from chunk_file(f, logical_path=rel)
