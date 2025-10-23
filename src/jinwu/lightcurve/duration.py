"""
Lightcurve evaluation utilities for fast SNR reachability checks.

This module defines a class that, given a binned lightcurve and background prior,
determines whether the signal can reach a target Li & Ma SNR within a fixed
time window after a data-driven T0 detection.

Key conventions and assumptions:
- The input lightcurve is ON-region counts per bin (not rate), with uniform or
  near-uniform bin width dt. If rates are provided, convert to counts by counts = rate * dt.
- Background prior is provided as the total OFF-region counts collected during
  a known exposure t_off (e.g., 100000 s). This defines a Gamma-Poisson model
  for the background rate r_off or can be treated as a fixed prior count n_off.
- Li & Ma SNR uses alpha = (A_on/A_off) * (t_on/t_off) = area_ratio * (t_on/t_off).
- T0 detection uses Bayesian Blocks partitioning and within-block Li & Ma SNR≥3
  criterion; T0 is the left edge of the first block reaching SNR≥3.

This class does NOT generate a lightcurve from spectra; it only consumes a given
lightcurve and background prior to decide if SNR≥target can be achieved within
the window (default 1200 s) after T0. It offers two evaluation modes:
- fast: deterministic estimate using expected background (no MC sampling)
- mc:   Monte Carlo with Poisson fluctuations for both ON and OFF (recommended)

Dependencies: numpy, astropy.stats (bayesian_blocks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal, Union

import numpy as np
from astropy.stats import bayesian_blocks

# Background models
from jinwu.background.backprior import (
	BackgroundPrior as _BackgroundPrior,
	BackgroundCountsPosterior as _BackgroundCountsPosterior,
)


def li_ma_snr(n_on: float, n_off: float, alpha: float) -> float:
	"""
	Compute Li & Ma significance (Eq. 17 in Li & Ma 1983) given counts and alpha.

	Parameters
	----------
	n_on : float
		Total counts in ON region.
	n_off : float
		Total counts in OFF region (reference background).
	alpha : float
		Exposure/area scaling factor from OFF to ON: alpha = (A_on/A_off)*(t_on/t_off).

	Returns
	-------
	float
		The Li & Ma significance (>=0). Returns 0.0 if inputs are degenerate.
	"""
	if n_on <= 0 and n_off <= 0:
		return 0.0
	if alpha <= 0:
		return 0.0

	# Guard against log of zero
	n_on = float(max(n_on, 0.0))
	n_off = float(max(n_off, 0.0))

	# Handle corner cases per common practice
	if n_on == 0.0:
		return 0.0
	if n_off == 0.0:
		# For n_off=0, Li&Ma reduces to sqrt(2 n_on ln(1+alpha))
		return np.sqrt(2.0 * n_on * np.log(1.0 + alpha))

	term1 = 0.0
	term2 = 0.0
	denom = n_on + n_off
	# Avoid divide-by-zero; denom>0 here because n_on>0 or n_off>0
	term1 = n_on * np.log(((1.0 + alpha) / alpha) * (n_on / denom))
	term2 = n_off * np.log((1.0 + alpha) * (n_off / denom))
	val = 2.0 * (term1 + term2)
	return float(np.sqrt(max(val, 0.0)))




class LightcurveSNREvaluator:
	"""
	Evaluate whether a given lightcurve can reach a target SNR after T0.

	This class takes a binned ON-region lightcurve (time edges or centers and
	counts per bin) and a background prior, finds T0 via Bayesian Blocks using
	per-block Li&Ma SNR≥3, then computes the maximum Li&Ma SNR within a fixed
	time window after T0. It supports a fast expected-value mode and an MC mode
	with Poisson fluctuations for ON and OFF.

	Typical usage
	------------
	>>> bg = BackgroundPrior(n_off_prior=1200, t_off=100000.0, area_ratio=1/12)
	>>> ev = LightcurveSNREvaluator.from_counts(
	...     time=np.arange(0, 2000.0, 0.5),  # bin centers or left edges
	...     counts=np.random.poisson(0.1, 4000),
	...     dt=0.5,
	...     background=bg,
	... )
	>>> ok, stats = ev.reaches_snr(target=7.0, window=1200.0, mode="fast")
	>>> ok
	False
	>>> # MC with Poisson fluctuations
	>>> ok_mc, stats_mc = ev.reaches_snr(target=7.0, window=1200.0, mode="mc", n_mc=500)
	>>> ok_mc
	False

	Load from NPZ (common keys)
	---------------------------
	The loader prefers the following keys if present:
	- counts (ON-region):
		1) 'corrected_counts_src'  (preferred; counts in source region per bin)
		2) If only net+background available:
		   'corrected_counts' (net) and 'corrected_counts_back' (OFF region)
		   then ON counts reconstructed by: counts_on = net + area_ratio * back
		3) As a fallback, 'raw_corrected_counts' will be treated as ON counts
		   (approximation if it's actually net); a warning is emitted.
	- time array:
		'time_series' (preferred) else 'raw_time_series'

	>>> ev = LightcurveSNREvaluator.from_npz(
	...     "/path/to/your_lc.npz",
	...     background=bg,
	... )
	>>> ev.reaches_snr(target=7.0, window=1200.0, mode="mc", n_mc=800)
	(False, {'prob': 0.12, 'max_snrs': array([...])})
	"""

	def __init__(
		self,
		time: np.ndarray,
		counts: np.ndarray,
		dt: float,
		background: Union[_BackgroundPrior, _BackgroundCountsPosterior],
		off_exposure_ref: Optional[float] = None,
	) -> None:
		if time.ndim != 1 or counts.ndim != 1:
			raise ValueError("time and counts must be 1D arrays")
		if time.size != counts.size:
			raise ValueError("time and counts must have the same length")
		if dt <= 0:
			raise ValueError("dt must be positive")

		self.time = np.asarray(time, dtype=float)
		self.counts = np.asarray(counts, dtype=float)
		self.dt = float(dt)
		# Background can be a simple prior or a spectral-aggregated posterior
		self._bg_prior: Optional[_BackgroundPrior]
		self._bg_post: Optional[_BackgroundCountsPosterior]
		if isinstance(background, _BackgroundCountsPosterior):
			self._bg_prior = None
			self._bg_post = background
			self.area_ratio = float(background.area_ratio)
			# Reference OFF exposure used to define alpha and sample n_off for Li&Ma
			self.off_exposure_ref = float(off_exposure_ref) if off_exposure_ref is not None else 1_000_000.0
		else:
			# Treat as simple BackgroundPrior
			self._bg_prior = background  # type: ignore[assignment]
			self._bg_post = None
			self.area_ratio = float(background.area_ratio)  # type: ignore[attr-defined]
			self.off_exposure_ref = float(getattr(background, "t_off", 1_000_000.0))

		# Precompute cumulative sums for fast window queries
		self._cum_counts = np.cumsum(self.counts)

	# ---------- Construction helpers ----------
	@classmethod
	def from_counts(
		cls,
		time: np.ndarray,
		counts: np.ndarray,
		dt: Optional[float] = None,
		background: Optional[Union[_BackgroundPrior, _BackgroundCountsPosterior]] = None,
		off_exposure_ref: Optional[float] = None,
	) -> "LightcurveSNREvaluator":
		"""
		Build evaluator from counts per bin.

		Parameters
		----------
		time : array
			1D array of time stamps corresponding to each bin. If these are
			left edges, dt must be provided; if these are centers and dt is
			not given, it will be inferred from median spacing.
		counts : array
			1D array of ON-region counts per bin (non-negative).
		dt : float, optional
			Bin width in seconds. If None, inferred as median diff of time.
		background : BackgroundPrior
			Background prior object.
		"""
		time = np.asarray(time, dtype=float)
		counts = np.asarray(counts, dtype=float)
		if dt is None:
			if time.size < 2:
				raise ValueError("Need dt or at least two time points to infer dt")
			dt = float(np.median(np.diff(time)))
		if background is None:
			raise ValueError("background must be provided")
		return cls(time=time, counts=counts, dt=dt, background=background, off_exposure_ref=off_exposure_ref)

	@classmethod
	def from_npz(
		cls,
		npz_path: str,
		background: Union[_BackgroundPrior, _BackgroundCountsPosterior],
		*,
		time_key_primary: str = "time_series",
		time_key_fallback: str = "raw_time_series",
		counts_key_preferred: str = "corrected_counts_src",
		net_key: str = "corrected_counts",
		off_key: str = "corrected_counts_back",
		raw_counts_key_fallback: str = "raw_corrected_counts",
		dt: Optional[float] = None,
		off_exposure_ref: Optional[float] = None,
		verbose: bool = True,
	) -> "LightcurveSNREvaluator":
		"""
		Build evaluator by loading a standard NPZ lightcurve file.

		Selection logic:
		- time = npz[time_key_primary] if present else npz[time_key_fallback]
		- counts (ON region per bin):
			1) if counts_key_preferred in npz: use it directly
			2) elif both net_key and off_key in npz: reconstruct ON counts as
			   counts_on = net + background.area_ratio * off
			3) elif raw_counts_key_fallback in npz: use it as counts (warning)
			4) else: raise ValueError

		Parameters
		----------
		npz_path : str
			Path to NPZ file with lightcurve arrays.
		background : BackgroundPrior
			Background prior object (provides area_ratio for reconstruction).
		dt : float, optional
			Bin width; if None, inferred from median diff of the chosen time array.
		verbose : bool
			If True, prints brief notes about which keys were used.
		"""
		data = np.load(npz_path)

		# time
		if time_key_primary in data:
			time = np.asarray(data[time_key_primary], dtype=float)
			src_time_key = time_key_primary
		elif time_key_fallback in data:
			time = np.asarray(data[time_key_fallback], dtype=float)
			src_time_key = time_key_fallback
		else:
			raise ValueError(
				f"Cannot find time array in NPZ. Tried '{time_key_primary}' and '{time_key_fallback}'."
			)

		# counts
		counts = None
		used = None
		if counts_key_preferred in data:
			counts = np.asarray(data[counts_key_preferred], dtype=float)
			used = counts_key_preferred
		elif (net_key in data) and (off_key in data):
			net = np.asarray(data[net_key], dtype=float)
			off = np.asarray(data[off_key], dtype=float)
			counts = net + float(background.area_ratio) * off
			used = f"{net_key} + area_ratio*{off_key}"
		elif raw_counts_key_fallback in data:
			counts = np.asarray(data[raw_counts_key_fallback], dtype=float)
			used = raw_counts_key_fallback
			if verbose:
				print(
					"[LightcurveSNREvaluator] Using 'raw_corrected_counts' as ON counts.\n"
					"If this is actually net counts, SNR will be conservative."
				)
		else:
			raise ValueError(
				"Cannot determine ON-region counts from NPZ. Provide one of: "
				f"'{counts_key_preferred}', or both '{net_key}' & '{off_key}', "
				f"or '{raw_counts_key_fallback}'."
			)

		if dt is None:
			if time.size < 2:
				raise ValueError("Need dt or at least two time samples to infer dt")
			dt = float(np.median(np.diff(time)))

		if verbose:
			print(
				f"[LightcurveSNREvaluator] Loaded time='{src_time_key}', counts='{used}', dt={dt:.6g}s"
			)

		return cls.from_counts(time=time, counts=counts, dt=dt, background=background, off_exposure_ref=off_exposure_ref)

	# ---------- Core logic ----------
	def _block_snr(self, left: float, right: float, n_off: float) -> float:
		"""Compute Li&Ma SNR within [left, right) using cumulative sums.

		n_on is the ON-region total counts within the interval. alpha uses the
		duration t_on = right-left.
		"""
		# Locate indices
		i0 = int(np.searchsorted(self.time, left, side="left"))
		i1 = int(np.searchsorted(self.time, right, side="left"))
		if i1 <= i0:
			return 0.0
		n_on = float(self._cum_counts[i1 - 1] - (self._cum_counts[i0 - 1] if i0 > 0 else 0.0))
		t_on = (right - left)
		alpha = self._alpha(t_on)
		return li_ma_snr(n_on=n_on, n_off=n_off, alpha=alpha)

	def _alpha(self, t_on: float) -> float:
		"""Unified alpha: (A_on/A_off) * (t_on / t_off_ref).

		When a BackgroundPrior is provided, t_off_ref equals prior.t_off.
		When a BackgroundCountsPosterior is provided, t_off_ref is either user-given
		or defaults to 1e6 s.
		"""
		return float(self.area_ratio) * (float(t_on) / float(self.off_exposure_ref))

	def _find_T0_by_blocks(
		self,
		snr_thr: float = 3.0,
		n_off: Optional[float] = None,
		rng: Optional[np.random.Generator] = None,
		off_mode: Literal["fixed", "poisson"] = "fixed",
	) -> float:
		"""
		Find T0 as the left edge of the first Bayesian Block with SNR >= snr_thr.

		Parameters
		----------
		snr_thr : float
			Threshold SNR for T0 detection (default 3.0).
		n_off : float, optional
			OFF counts to use for Li&Ma within blocks. If None, determined by
			off_mode.
		rng : np.random.Generator, optional
			RNG used when off_mode='poisson'.
		off_mode : {'fixed','poisson'}
			- 'fixed': use the prior n_off as provided.
			- 'poisson': draw n_off' ~ Poisson(mu_off * t_off) where
			  mu_off = n_off_prior / t_off, once for the block evaluation.

		Returns
		-------
		float
			T0 time in the same reference frame as input time. If no block
			reaches the threshold, returns the first time stamp.
		"""
		if n_off is None:
			if self._bg_post is not None:
				# Use posterior expected OFF counts or a posterior predictive sample
				if off_mode == "fixed":
					n_off = float(self._bg_post.expected_off(self.off_exposure_ref))
				else:
					rng = rng or np.random.default_rng()
					# Sample lambda_off once for T0 decision
					lam_off = rng.gamma(shape=float(self._bg_post.a_total), scale=1.0 / float(self._bg_post.b))
					n_off = float(rng.poisson(lam_off * float(self.off_exposure_ref)))
			else:
				# Fallback to simple prior behavior
				prior = self._bg_prior  # type: ignore[assignment]
				if off_mode == "fixed":
					n_off = float(prior.n_off_prior)  # type: ignore[union-attr]
				else:
					rng = rng or np.random.default_rng()
					mu_off = float(prior.n_off_prior) / float(prior.t_off)  # type: ignore[union-attr]
					n_off = float(rng.poisson(mu_off * prior.t_off))  # type: ignore[union-attr]

		# Use Bayesian Blocks on counts-per-bin time series; for stepwise data
		# bayesian_blocks expects event times for 'events' fitness, but for
		# binned measures we use fitness='measures'. Supply measure and dt.
		# Here we use the counts as 'measures' and dt as widths.
		widths = np.full_like(self.time, fill_value=self.dt, dtype=float)
		# astropy bayesian_blocks 'measures' requires (t, x, dx=None)
		edges = bayesian_blocks(self.time, self.counts, fitness="measures")
		# Iterate blocks to find first SNR>=snr_thr
		for i in range(len(edges) - 1):
			left, right = float(edges[i]), float(edges[i + 1])
			snr = self._block_snr(left, right, n_off=n_off)
			if snr >= snr_thr:
				return left
		# Fallback: no block reaches threshold
		return float(self.time[0])

	def reaches_snr(
		self,
		target: float = 7.0,
		window: float = 1200.0,
		mode: Literal["fast", "mc"] = "mc",
		n_mc: int = 500,
		rng: Optional[np.random.Generator] = None,
		t0_snr_thr: float = 3.0,
		off_mode: Literal["fixed", "poisson"] = "poisson",
	) -> Tuple[bool, dict]:
		"""
		Decide if the lightcurve can reach target SNR within window seconds
		after T0.

		Parameters
		----------
		target : float
			Target SNR threshold (default 7.0).
		window : float
			Time window in seconds after T0 to search for max SNR (default 1200 s).
		mode : {'fast', 'mc'}
			- 'fast': use expected OFF counts (no Poisson sampling), deterministic.
			- 'mc': Monte Carlo sampling with Poisson fluctuations for ON and OFF.
		n_mc : int
			Number of MC replicates when mode='mc'.
		rng : np.random.Generator, optional
			Random generator for MC.
		t0_snr_thr : float
			SNR threshold used for T0 detection (default 3.0).
		off_mode : {'fixed','poisson'}
			OFF counts treatment for both T0 and window SNR. See _find_T0_by_blocks.

		Returns
		-------
		(bool, dict)
			- bool: True if probability (fast: indicator) of reaching target SNR
			  is >= 0.95 (only meaningful for mc; for fast it is True/False as is).
			- dict: diagnostics including T0 estimate, probability (for mc),
			  and optionally the distribution of max SNRs.
		"""
		rng = rng or np.random.default_rng()

		if mode == "fast":
			# Use expected OFF counts from either posterior or prior; ON counts as given
			if self._bg_post is not None:
				n_off_exp = float(self._bg_post.expected_off(self.off_exposure_ref))
			else:
				n_off_exp = float(self._bg_prior.n_off_prior)  # type: ignore[union-attr]
			T0 = self._find_T0_by_blocks(snr_thr=t0_snr_thr, n_off=n_off_exp, off_mode="fixed")
			# Compute cumulative ON counts starting at T0
			t_start = T0
			t_end = T0 + float(window)
			i0 = int(np.searchsorted(self.time, t_start, side="left"))
			i1 = int(np.searchsorted(self.time, t_end, side="left"))
			if i1 <= i0:
				return False, {"T0": T0, "max_snr": 0.0}
			# cumulative within window
			counts_win = self.counts[i0:i1]
			csum = np.cumsum(counts_win)
			# Evaluate SNR at each cumulative step (monotone t_on = k*dt)
			max_snr = 0.0
			for k in range(1, csum.size + 1):
				t_on = k * self.dt
				alpha = self._alpha(t_on)
				n_on = float(csum[k - 1])
				snr = li_ma_snr(n_on=n_on, n_off=float(n_off_exp), alpha=alpha)
				if snr > max_snr:
					max_snr = snr
			ok = bool(max_snr >= target)
			return ok, {"T0": T0, "max_snr": max_snr}

		# MC mode
		hits = 0
		max_snrs = []

		for _ in range(int(n_mc)):
			# OFF counts treatment
			if self._bg_post is not None:
				if off_mode == "fixed":
					n_off = float(self._bg_post.expected_off(self.off_exposure_ref))
				else:
					lam_off = rng.gamma(shape=float(self._bg_post.a_total), scale=1.0 / float(self._bg_post.b))
					n_off = float(rng.poisson(lam_off * float(self.off_exposure_ref)))
			else:
				if off_mode == "fixed":
					n_off = float(self._bg_prior.n_off_prior)  # type: ignore[union-attr]
				else:
					mu_off = float(self._bg_prior.n_off_prior) / float(self._bg_prior.t_off)  # type: ignore[union-attr]
					n_off = float(rng.poisson(mu_off * self._bg_prior.t_off))  # type: ignore[union-attr]

			# T0 for this replicate
			T0 = self._find_T0_by_blocks(snr_thr=t0_snr_thr, n_off=n_off, rng=rng, off_mode=off_mode)

			# Re-simulate ON counts within the window by adding Poisson fluctuations
			t_start = T0
			t_end = T0 + float(window)
			i0 = int(np.searchsorted(self.time, t_start, side="left"))
			i1 = int(np.searchsorted(self.time, t_end, side="left"))
			if i1 <= i0:
				max_snrs.append(0.0)
				continue
			# Build ON counts by separating source and background components when a posterior is available
			bins = slice(i0, i1)
			if self._bg_post is not None:
				# Sample a single lambda_off for the whole replicate to keep background rate consistent across bins
				lam_off = rng.gamma(shape=float(self._bg_post.a_total), scale=1.0 / float(self._bg_post.b))
				mu_bkg_bin = float(lam_off) * float(self.area_ratio) * float(self.dt)
				# Estimate source expectation as (observed ON expectation) - (background mean), clipped at 0
				lam_on_obs = np.clip(self.counts[bins], 0.0, None)
				mu_src_bin = np.clip(lam_on_obs - mu_bkg_bin, 0.0, None)
				# Sample source and background separately
				n_src_bins = rng.poisson(mu_src_bin)
				n_bkg_bins = rng.poisson(mu_bkg_bin, size=n_src_bins.size)
				n_on_bins = n_src_bins + n_bkg_bins
			else:
				# No posterior: treat provided counts as expectation for ON and draw Poisson
				lam_on = np.clip(self.counts[bins], 0.0, None)
				n_on_bins = rng.poisson(lam_on)
			csum = np.cumsum(n_on_bins)
			max_snr = 0.0
			for k in range(1, csum.size + 1):
				t_on = k * self.dt
				alpha = self._alpha(t_on)
				# Allow OFF to fluctuate per replicate; keep constant across cumulative growth
				snr = li_ma_snr(n_on=float(csum[k - 1]), n_off=n_off, alpha=alpha)
				if snr > max_snr:
					max_snr = snr
			max_snrs.append(float(max_snr))
			hits += int(max_snr >= target)

		prob = hits / float(n_mc)
		return bool(prob >= 0.95), {"prob": prob, "max_snrs": np.asarray(max_snrs)}


__all__ = [
	"LightcurveSNREvaluator",
	"li_ma_snr",
]

