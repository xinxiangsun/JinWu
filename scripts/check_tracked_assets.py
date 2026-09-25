"""Guard: workflow-referenced test files and Sphinx toctree entries must be tracked.

Background
----------
``test/*`` is ignored via .gitignore with an explicit per-file allow-list, so a
new test module silently stays untracked unless it is force-added.  CI and the
publish wheel-gate can only see tracked files; ``docs/index.rst`` toctree
entries that point at untracked files vanish from a fresh checkout and break
the Read the Docs build.

This script fails (exit 1) when any of the following holds:

1. a ``test/*.py`` / ``packages/*/tests/*.py`` path referenced by any
   ``.github/workflows/*.yml`` command line is missing from the git index;
2. any ``.. toctree::`` entry under ``docs/`` has no corresponding
   git-tracked source file.

Run locally (not wired into CI yet — enabling in CI requires committing or
removing the currently untracked ``docs/usage/gbm_subthreshold.rst`` and the
two orphan usage pages; see the codex/beta review report, findings R4/R5/R7)::

    python scripts/check_tracked_assets.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def tracked_files() -> set[str]:
    return set(git("ls-files").split())


def workflow_test_paths() -> list[str]:
    paths: list[str] = []
    pattern = re.compile(r"(?:^|\s)((?:test|packages/[\w-]+/tests)/[\w./-]*\.py)")
    for yml in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for line in yml.read_text(encoding="utf-8").splitlines():
            for match in pattern.finditer(line):
                paths.append(f"{yml.name}: {match.group(1)}")
    return paths


def toctree_entries() -> list[str]:
    entries: list[str] = []
    for rst in (ROOT / "docs").rglob("*.rst"):
        lines = rst.read_text(encoding="utf-8").splitlines()
        in_toctree = False
        base = None
        for line in lines:
            if line.startswith(".. toctree::"):
                in_toctree = True
                base = len(line) - len(line.lstrip())
                continue
            if in_toctree:
                indent = len(line) - len(line.lstrip())
                if line.strip() == "" or (line.lstrip().startswith(":") and indent > (base or 0)):
                    continue
                if indent <= (base or 0):
                    in_toctree = False
                    continue
                entries.append(f"{rst.relative_to(ROOT)}: {line.strip()}")
    return entries


def main() -> int:
    tracked = tracked_files()
    problems: list[str] = []

    for ref in workflow_test_paths():
        _, path = ref.split(": ", 1)
        if path not in tracked:
            problems.append(f"workflow 引用的测试文件未被跟踪: {ref}")

    for ref in toctree_entries():
        rst, entry = ref.split(": ", 1)
        if entry.startswith(("http://", "https://")):
            continue
        docs_dir = Path(rst).parent
        candidates = [
            docs_dir / f"{entry}.rst",
            docs_dir / f"{entry}.md",
            docs_dir / entry,
        ]
        if any(str(c) in tracked for c in candidates):
            continue
        problems.append(f"toctree 条目缺少被跟踪的文档源: {ref}")

    if problems:
        print("FAIL: 以下门禁资产缺失或未被 git 跟踪：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"OK: 工作流测试路径 {len(workflow_test_paths())} 条、toctree 条目 {len(toctree_entries())} 条均有被跟踪的文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
