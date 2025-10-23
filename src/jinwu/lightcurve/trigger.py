"""
Trigger decision utilities for simple, reproducible SNR checks.

This module provides a lightweight TriggerDecider that accepts either a binned
lightcurve (counts per bin) or photon event times and determines whether a
signal would trigger under two common strategies:
  1) Sliding 20-minute window: scan max Li&Ma SNR within any 1200 s window.
  2) Head 20-minute SNR: compute SNR of the first 1200 s; if not >= 7, then
     perform cumulative-from-T0 growth and check if SNR reaches 7 within data.

Notes
-----
• This is a minimal, deterministic implementation: no priors/posteriors yet.
  Background is provided as a constant pair (n_off_ref, t_off_ref) and a fixed
  area_ratio = A_on/A_off, so alpha(t) = area_ratio * (t / t_off_ref).
• The API keeps room for future MC: you can later swap in a sampled n_off_ref
  per replicate and/or add per-bin Poisson for ON counts.
• Counts are ON-region counts per bin (not rates). If you start from event
  times, use from_events to bin into counts.

Dependencies: numpy. SNR uses li_ma_snr from autohea.lightcurve.duration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np

from jinwu.lightcurve.duration import li_ma_snr


@dataclass
class BackgroundSimple:
    """
    Minimal background config for Li&Ma.

    Parameters
    ----------
    area_ratio : float
        A_on / A_off.
    t_off_ref : float
        Reference OFF exposure used for Li&Ma alpha (seconds).
    n_off_ref : float
        Total OFF counts corresponding to t_off_ref.
    """

    area_ratio: float
    t_off_ref: float
    n_off_ref: float

    def alpha(self, t_on: float) -> float:
        return float(self.area_ratio) * (float(t_on) / float(self.t_off_ref))


class TriggerDecider:
    """
    Decide triggerability from a counts lightcurve or event times.

    Core checks
    -----------
    - sliding_window(window=1200): scan max Li&Ma SNR over all windows of size window.
    - head_window(window=1200): Li&Ma SNR of the first window only.
    - cumulative_from_t0(target=7): if head window fails, grow cumulatively from T0
      (by default first non-zero bin) and check if SNR reaches target.

    Inputs
    ------
    time : 1D array of bin left edges or centers (monotonic increasing)
    counts : 1D array of ON-region counts per bin (non-negative)
    dt : float bin width in seconds (assumed constant)
    bg : BackgroundSimple with (n_off_ref, t_off_ref, area_ratio)

    Notes
    -----
    - This class does not alter the counts (no Poissonization). Future MC can
      be added by drawing per-bin ON counts around an expectation and reusing
      the same APIs.
    """

    def __init__(self, time: np.ndarray, counts: np.ndarray, dt: float, bg: BackgroundSimple) -> None:
        time = np.asarray(time, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if time.ndim != 1 or counts.ndim != 1:
            raise ValueError("time and counts must be 1D arrays")
        if time.size != counts.size:
            raise ValueError("time and counts must have the same length")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time = time
        self.counts = counts
        self.dt = float(dt)
        self.bg = bg
        self._cum = np.cumsum(self.counts)

    # ---------- Constructors ----------
    @classmethod
    def from_counts(
        cls,
        time: np.ndarray,
        counts: np.ndarray,
        dt: Optional[float],
        bg: BackgroundSimple,
    ) -> "TriggerDecider":
        time = np.asarray(time, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if dt is None:
            if time.size < 2:
                raise ValueError("Need dt or at least two time points to infer dt")
            dt = float(np.median(np.diff(time)))
        return cls(time=time, counts=counts, dt=dt, bg=bg)

    @classmethod
    def from_events(
        cls,
        events: np.ndarray,
        *,
        dt: float,
        bg: BackgroundSimple,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> "TriggerDecider":
        """
        Bin photon arrival times into counts per bin of width dt.
        If t_start/t_end are not given, they are inferred from min/max events
        with one bin padding on the right edge.
        """
        events = np.asarray(events, dtype=float)
        if events.ndim != 1:
            raise ValueError("events must be 1D array of times")
        if events.size == 0:
            raise ValueError("events is empty")
        if dt <= 0:
            raise ValueError("dt must be positive")

        if t_start is None:
            t_start = float(np.min(events))
        if t_end is None:
            t_end = float(np.max(events)) + float(dt)

        # Build bin edges and histogram
        nbins = int(np.ceil((t_end - t_start) / float(dt)))
        edges = t_start + np.arange(nbins + 1, dtype=float) * float(dt)
        counts, _ = np.histogram(events, bins=edges)
        # Use left edges as time
        time = edges[:-1]
        return cls(time=time, counts=counts.astype(float), dt=float(dt), bg=bg)

    # ---------- Core helpers ----------
    def _counts_in(self, left: float, right: float) -> float:
        i0 = int(np.searchsorted(self.time, left, side="left"))
        i1 = int(np.searchsorted(self.time, right, side="left"))
        if i1 <= i0:
            return 0.0
        return float(self._cum[i1 - 1] - (self._cum[i0 - 1] if i0 > 0 else 0.0))

    def _snr_window(self, left: float, right: float, n_off_ref: Optional[float] = None) -> float:
        n_on = self._counts_in(left, right)
        t_on = max(0.0, float(right - left))
        if t_on <= 0:
            return 0.0
        alpha = self.bg.alpha(t_on)
        n_off = float(self.bg.n_off_ref if n_off_ref is None else n_off_ref)
        return li_ma_snr(n_on=n_on, n_off=n_off, alpha=alpha)

    # ---------- Checks ----------
    def sliding_window(self, *, window: float = 1200.0, step: Optional[float] = None) -> Tuple[bool, dict]:
        """
        Scan all windows of size `window` (seconds) and return whether SNR>=7
        is achieved, with diagnostic info.
        """
        if window <= 0:
            raise ValueError("window must be positive")
        if step is None:
            step = self.dt
        step = float(step)
        if step <= 0:
            raise ValueError("step must be positive")

        t0 = float(self.time[0])
        tN = float(self.time[0] + self.counts.size * self.dt)
        starts = np.arange(t0, max(t0, tN - window) + 1e-12, step, dtype=float)
        max_snr = 0.0
        best = (t0, t0 + window)
        for s in starts:
            snr = self._snr_window(s, s + window)
            if snr > max_snr:
                max_snr = snr
                best = (s, s + window)
        return bool(max_snr >= 7.0), {"max_snr": max_snr, "best_window": best}

    def head_window(self, *, window: float = 1200.0) -> Tuple[bool, dict]:
        """SNR of the first `window` seconds from the series start."""
        left = float(self.time[0])
        right = left + float(window)
        snr = self._snr_window(left, right)
        return bool(snr >= 7.0), {"snr": snr, "window": (left, right)}

    def _find_t0(self, mode: Literal["first_nonzero", "first_time"] = "first_nonzero") -> float:
        if mode == "first_time":
            return float(self.time[0])
        # first_nonzero: first bin with counts > 0
        idx = int(np.argmax(self.counts > 0)) if np.any(self.counts > 0) else 0
        return float(self.time[idx])

    def cumulative_from_t0(
        self,
        *,
        target: float = 7.0,
        t0_mode: Literal["first_nonzero", "first_time"] = "first_nonzero",
        max_window: Optional[float] = None,
    ) -> Tuple[bool, dict]:
        """
        Grow cumulatively from T0 and search for the first time SNR>=target.
        If max_window is given, only search within [T0, T0+max_window).
        """
        T0 = self._find_t0(mode=t0_mode)
        t_end = float(self.time[0] + self.counts.size * self.dt)
        if max_window is not None:
            t_end = min(t_end, T0 + float(max_window))

        # Determine indices
        i0 = int(np.searchsorted(self.time, T0, side="left"))
        i1 = int(np.searchsorted(self.time, t_end, side="left"))
        if i1 <= i0:
            return False, {"T0": T0, "t_reach": None, "max_snr": 0.0}

        csum = np.cumsum(self.counts[i0:i1])
        max_snr = 0.0
        t_reach: Optional[float] = None
        for k in range(1, csum.size + 1):
            t_on = k * self.dt
            alpha = self.bg.alpha(t_on)
            snr = li_ma_snr(n_on=float(csum[k - 1]), n_off=float(self.bg.n_off_ref), alpha=alpha)
            if snr > max_snr:
                max_snr = snr
            if snr >= float(target) and t_reach is None:
                t_reach = T0 + t_on
                break
        return bool(t_reach is not None), {"T0": T0, "t_reach": t_reach, "max_snr": max_snr}

    # ---------- Orchestrator ----------
    def decide(
        self,
        *,
        window: float = 1200.0,
        target: float = 7.0,
        step: Optional[float] = None,
        t0_mode: Literal["first_nonzero", "first_time"] = "first_nonzero",
    ) -> dict:
        """
        Two-step decision:
        1) Sliding `window` SNR anywhere >= target? If yes, trigger.
        2) Head `window` SNR >= target? If yes, trigger.
        3) Else grow cumulatively from T0 until reach target or series ends.
        Returns a dict with decision and diagnostics.
        """
        slid_ok, slid_stat = self.sliding_window(window=window, step=step)
        if slid_ok:
            return {"triggered": True, "method": "sliding", **slid_stat}

        head_ok, head_stat = self.head_window(window=window)
        if head_ok:
            return {"triggered": True, "method": "head", **head_stat}

        cum_ok, cum_stat = self.cumulative_from_t0(target=target, t0_mode=t0_mode, max_window=None)
        return {"triggered": bool(cum_ok), "method": "cumulative", **cum_stat}


__all__ = [
    "BackgroundSimple",
    "TriggerDecider",
]
