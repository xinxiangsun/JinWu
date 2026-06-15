"""Compatibility wrapper for trigger decision utilities.

New code should import from ``jinwu.core.utils``:

    >>> from jinwu.core.utils import BackgroundSimple, TriggerDecider
"""

from __future__ import annotations

from jinwu.core.utils import BackgroundSimple, TriggerDecider  # noqa: F401

__all__ = ["BackgroundSimple", "TriggerDecider"]
