"""Step 1 — walk a repository and yield Python source files.

Prefers `git ls-files` so we honor `.gitignore` exactly (including nested and
global ignore rules) with zero maintenance. Includes tracked files *plus*
untracked-but-not-ignored ones, so a module is indexed as soon as it's written
rather than only after it's committed. Falls back to a manual walk that skips
the usual noise when the tree isn't a git repo (or git isn't installed).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

# Only used by the non-git fallback. The git path defers to .gitignore instead.
DEFAULT_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".history",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "build",
        "dist",
        ".eggs",
        "site-packages",
    }
)


def walk_python_files(
    root: str | os.PathLike[str],
    *,
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
    follow_symlinks: bool = False,
) -> Iterator[Path]:
    """Yield every `.py` file under `root`, in deterministic sorted order.

    When `root` is inside a git repo, defers to `git ls-files` so `.gitignore`
    is honored exactly; `ignore_dirs` and `follow_symlinks` only apply to the
    manual fallback used when `root` isn't a git repo.
    """
    root = Path(root)
    git_paths = _git_python_files(root)
    if git_paths is not None:
        yield from git_paths
    else:
        yield from _walk_python_files(
            root, ignore_dirs=ignore_dirs, follow_symlinks=follow_symlinks
        )


def _git_python_files(root: Path) -> list[Path] | None:
    """Return `.py` paths via git, or `None` if `root` isn't a git repo.

    `--cached` is tracked files, `--others` adds untracked ones, and
    `--exclude-standard` applies the normal ignore rules to both. `-z` keeps us
    safe against newlines and other oddities in filenames.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "*.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not a git repo, or git isn't on PATH — caller falls back to a walk.
        return None

    rel_paths = (p for p in result.stdout.split("\0") if p)
    # git's output is already sorted, but combining --cached/--others and
    # resolving against root warrants an explicit sort for determinism.
    return sorted(root / p for p in rel_paths)


def _walk_python_files(
    root: Path,
    *,
    ignore_dirs: frozenset[str],
    follow_symlinks: bool,
) -> Iterator[Path]:
    """Fallback walk for non-git trees: depth-first, pruning `ignore_dirs`.

    Paths are yielded in sorted order within each directory so a run over an
    unchanged tree visits files deterministically.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        # Prune ignored directories in place so os.walk doesn't descend into them.
        dirnames[:] = sorted(d for d in dirnames if d not in ignore_dirs)
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield Path(dirpath) / name


def read_source(path: str | os.PathLike[str]) -> str:
    """Read a source file as UTF-8, tolerating odd bytes rather than crashing."""
    return Path(path).read_text(encoding="utf-8", errors="replace")
