"""CLI entry point: ``python -m jinwu.swift.grb --root <ws> --grb <key>``."""

from __future__ import annotations

from .pipeline import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
