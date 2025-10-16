
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import numpy as np
from astropy.io import fits

@dataclass
class BackgroundPrior:
	"""
	Background prior based on OFF-region total counts over a known exposure.

	Attributes
	----------
	n_off_prior : float
		Total counts observed in the OFF region during t_off seconds.
	t_off : float
		Exposure time in the OFF region corresponding to n_off_prior (seconds).
	area_ratio : float
		A_on / A_off. The Li&Ma alpha uses alpha = area_ratio * (t_on / t_off).
	"""

	n_off_prior: float
	t_off: float
	area_ratio: float

	def alpha(self, t_on: float) -> float:
		return self.area_ratio * (float(t_on) / float(self.t_off))

	@classmethod
	def from_epwxt_background(
		cls,
		fits_path: str,
		*,
		chan_lo: int = 53,
		chan_hi: int = 506,
		arf_path: Optional[str] = None,
		arf_bin_lo: int = 81,
		arf_bin_hi: int = 780,
		pixel_size_mm: float = 0.015,
		pixel_fov_deg: float = 0.00229,
		src_radius_arcmin: float = 9.0,
		bkg_rin_arcmin: float = 18.0,
		bkg_rout_arcmin: float = 36.0,
		t_off: float = 1_000_000.0,
		verbose: bool = True,
	) -> "BackgroundPrior":
		"""
		Construct BackgroundPrior from a single EP/WXT background FITS file that contains
		counts per cm^2 accumulated over a reference exposure (typically 1e6 s).

		This enforces the energy band via detector channels [chan_lo, chan_hi]
		(default 53–506, about 0.5–4.0 keV), computes the OFF-region prior counts
		in a desired t_off, and derives area_ratio from instrument geometry.

		Parameters
		----------
		fits_path : str
			Path to the background FITS (counts per cm^2). In practice only one combined
			background file is available.
		chan_lo, chan_hi : int
			Inclusive channel bounds to approximate 0.5–4.0 keV.
		pixel_size_mm : float
			Pixel size on detector (mm). Default 0.015 mm.
		pixel_fov_deg : float
			Sky angular size per pixel (degree). Default 0.00229 deg.
		src_radius_arcmin : float
			Source region radius (arcmin). Default 9'.
		bkg_rin_arcmin, bkg_rout_arcmin : float
			Background annulus inner/outer radii (arcmin). Default 18'–36'.
		t_off : float
			OFF prior exposure (s) for Li&Ma. Default 1e6 s (matches FITS by default).
		verbose : bool
			Print brief diagnostics.

		Returns
		-------
		BackgroundPrior
		"""

		# If ARF is provided, derive the energy band from ARF bins [arf_bin_lo, arf_bin_hi]
		band_keV: Optional[Tuple[float, float]] = None
		if arf_path is not None:
			with fits.open(arf_path) as arf:
				if 'SPECRESP' not in arf:
					raise ValueError(f"ARF file {arf_path} lacks SPECRESP extension.")
				hdu_sp = arf['SPECRESP']
				arfd = getattr(hdu_sp, 'data', None)
				if arfd is None:
					raise ValueError(f"ARF file {arf_path} SPECRESP has no data")
				elo = np.asarray(arfd['ENERG_LO'], dtype=float)
				ehi = np.asarray(arfd['ENERG_HI'], dtype=float)
				# Assume user-provided ARF bin indices are 1-based (第81个…第780个)
				i0 = max(0, int(arf_bin_lo) - 1)
				i1 = min(elo.size - 1, int(arf_bin_hi) - 1)
				if i1 < i0:
					raise ValueError("arf_bin_hi must be >= arf_bin_lo")
				band_keV = (float(np.min(elo[i0:i1 + 1])), float(np.max(ehi[i0:i1 + 1])))
				if verbose:
					print(f"[BackgroundPrior] ARF bins {arf_bin_lo}-{arf_bin_hi} -> band ~{band_keV[0]:.3g}-{band_keV[1]:.3g} keV")

		def _sum_counts_per_cm2(path: str, lo: int, hi: int, band: Optional[Tuple[float, float]]) -> Tuple[float, float]:
			with fits.open(path) as hdul:
				# Try SPECTRUM extension (OGIP PHA)
				if 'SPECTRUM' not in hdul:
					raise ValueError(f"FITS file {path} lacks SPECTRUM extension.")
				hdu_spec = hdul['SPECTRUM']
				spec = getattr(hdu_spec, 'data', None)
				hdr = getattr(hdu_spec, 'header', None)
				if spec is None or hdr is None:
					raise ValueError(f"FITS SPECTRUM in {path} missing data/header")
				channels = np.asarray(spec['CHANNEL'], dtype=int)
				mask = (channels >= lo) & (channels <= hi)
				# If energy bounds are available and ARF-derived band is given, prefer energy overlap
				if ('EBOUNDS' in hdul) and (band is not None):
					hdu_eb = hdul['EBOUNDS']
					eb = getattr(hdu_eb, 'data', None)
					if eb is not None:
						elo = np.asarray(eb['E_MIN'], dtype=float)
						ehi = np.asarray(eb['E_MAX'], dtype=float)
						mask = (ehi > band[0]) & (elo < band[1])
				counts_cm2 = float(np.sum(spec['COUNTS'][mask]))
				texp = float(hdr.get('EXPOSURE', 1_000_000.0))

				# Optional EBOUNDS cross-check
				if 'EBOUNDS' in hdul:
					hdu_eb = hdul['EBOUNDS']
					eb = getattr(hdu_eb, 'data', None)
					if eb is not None and verbose:
						elo = np.asarray(eb['E_MIN'], dtype=float)
						ehi = np.asarray(eb['E_MAX'], dtype=float)
						if band is None:
							ch = np.asarray(eb['CHANNEL'], dtype=int)
							sel = (ch >= lo) & (ch <= hi)
							emin = float(np.min(elo[sel])) if np.any(sel) else np.nan
							emax = float(np.max(ehi[sel])) if np.any(sel) else np.nan
							print(f"[BackgroundPrior] Channel {lo}-{hi} spans ~{emin:.3g}-{emax:.3g} keV (EBOUNDS)")
						else:
							print(f"[BackgroundPrior] Using ARF-derived band ~{band[0]:.3g}-{band[1]:.3g} keV for background selection")
				return counts_cm2, texp

		c_tot_cm2, t_ref = _sum_counts_per_cm2(fits_path, chan_lo, chan_hi, band_keV)

		# Geometry: map sky regions to detector cm^2 via pixelization
		pix_size_cm = float(pixel_size_mm) * 0.1  # mm -> cm
		apix_cm2 = pix_size_cm ** 2
		omega_pix_deg2 = float(pixel_fov_deg) ** 2
		# Sky areas
		r_on_deg = float(src_radius_arcmin) / 60.0
		r_in_deg = float(bkg_rin_arcmin) / 60.0
		r_out_deg = float(bkg_rout_arcmin) / 60.0
		A_on_sky_deg2 = float(np.pi * (r_on_deg ** 2))
		A_off_sky_deg2 = float(np.pi * (r_out_deg ** 2 - r_in_deg ** 2))
		# Pixel counts and detector cm^2 areas
		Npix_on = A_on_sky_deg2 / omega_pix_deg2
		Npix_off = A_off_sky_deg2 / omega_pix_deg2
		A_on_cm2 = Npix_on * apix_cm2
		A_off_cm2 = Npix_off * apix_cm2
		area_ratio = float(A_on_cm2 / A_off_cm2)
		if verbose:
			print(f"[BackgroundPrior] A_on_cm2={A_on_cm2:.4g}, A_off_cm2={A_off_cm2:.4g}, area_ratio~{area_ratio:.6g}")

		# Convert counts-per-cm2 over t_ref to rate-per-cm2
		r_bg_cm2 = c_tot_cm2 / t_ref  # counts/(s·cm^2)
		r_off = r_bg_cm2 * A_off_cm2  # counts/s in OFF annulus
		n_off_prior = r_off * float(t_off)
		if verbose:
			print(f"[BackgroundPrior] t_ref={t_ref:g}s, t_off={t_off:g}s, n_off_prior={n_off_prior:.6g}")

		return cls(n_off_prior=float(n_off_prior), t_off=float(t_off), area_ratio=float(area_ratio))

	@classmethod
	def from_epwxt_background_default(
		cls,
		*,
		chan_lo: int = 53,
		chan_hi: int = 506,
		t_off: float = 1_000_000.0,
		verbose: bool = True,
	) -> "BackgroundPrior":
		"""
		Return the ready-made BackgroundPrior values as requested by user.
		Ignores file IO and outputs fixed values for reproducibility of quick tests.
		"""
		if verbose:
			print("[BackgroundPrior] Using fixed default prior: "
			      "n_off_prior=104016.95348743448, t_off=1000000.0, area_ratio=0.08333333333333333")
		return cls(
			n_off_prior=104016.95348743448,
			t_off=1_000_000.0,
			area_ratio=0.08333333333333333,
		)


@dataclass
class BackgroundCountsPosterior:
	"""
	背景区域计数后验（聚合到选择能段之后）。

	基于 Gamma-Poisson 共轭：以 OFF 区域“计数率” λ_off ~ Gamma(a_total, b)
	（shape=a_total, rate=b），则：
	- OFF 区域在曝光 t 的后验预测 n_off ~ Poisson-Gamma 混合，等价于 NB(r=a_total, p=b/(b+t))；
	- ON 区域在曝光 t 的后验预测 n_on,bkg ~ NB(r=a_total, p=b/(b+area_ratio*t))，
	  其中 area_ratio=A_on/A_off。

	我们通过 Gamma-Poisson 的层次采样实现采样（避免不同库对 NB 参数化差异）。

	属性
	------
	a_total : float
		选择能段内所有能道的先验 shape 之和（更新后为后验 shape）。
	b : float
		先验/后验的 rate 参数（单位：秒），通常为 t_ref + sum(t_obs_off)。
	area_ratio : float
		A_on / A_off，用于把 OFF 计数率映射到 ON 背景计数率。
	"""

	a_total: float
	b: float
	area_ratio: float

	# ---- 期望值 ----
	def expected_off(self, t: float) -> float:
		return float(self.a_total) * float(t) / float(self.b)

	def expected_on(self, t: float) -> float:
		return float(self.a_total) * float(self.area_ratio) * float(t) / float(self.b)

	# ---- 采样器（后验预测）----
	def _sample_lambda_off(self, size: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
		if rng is None:
			rng = np.random.default_rng()
		# Gamma(shape=a, rate=b) -> numpy 用 scale=1/rate；当 a 非正时退化为 0
		a = float(self.a_total)
		if a <= 0:
			return np.zeros(size, dtype=float)
		return rng.gamma(shape=a, scale=1.0 / float(self.b), size=size)

	def sample_off(self, t: float, size: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
		"""采样 OFF 区域在曝光 t 的后验预测计数。"""
		if rng is None:
			rng = np.random.default_rng()
		lam = self._sample_lambda_off(size=size, rng=rng)
		return rng.poisson(lam * float(t))

	def sample_on(self, t: float, size: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
		"""采样 ON 区域（仅背景）在曝光 t 的后验预测计数。"""
		if rng is None:
			rng = np.random.default_rng()
		lam = self._sample_lambda_off(size=size, rng=rng)
		return rng.poisson(lam * float(self.area_ratio) * float(t))

	# ---- 增量式共轭更新 ----
	def update_with_off_counts(self, n_off: float, t_off: float, *, inplace: bool = False) -> "BackgroundCountsPosterior":
		"""用额外的 OFF 观测（总计数与曝光）进行一次共轭更新。"""
		a_new = float(self.a_total) + float(n_off)
		b_new = float(self.b) + float(t_off)
		if inplace:
			self.a_total = a_new
			self.b = b_new
			return self
		return BackgroundCountsPosterior(a_total=a_new, b=b_new, area_ratio=float(self.area_ratio))

	def update_with_on_bg_counts(self, n_on_bg: float, t_on_bg: float, *, inplace: bool = False) -> "BackgroundCountsPosterior":
		"""用额外的 ON 背景-only 观测进行更新（等效 OFF 曝光为 area_ratio*t_on_bg）。"""
		a_new = float(self.a_total) + float(n_on_bg)
		b_new = float(self.b) + float(self.area_ratio) * float(t_on_bg)
		if inplace:
			self.a_total = a_new
			self.b = b_new
			return self
		return BackgroundCountsPosterior(a_total=a_new, b=b_new, area_ratio=float(self.area_ratio))


@dataclass
class BackgroundSpectralPrior:
	"""
	考虑能谱的背景先验：对所选能道逐道建立 Gamma 先验 λ_k ~ Gamma(a0_k, b0)，
	其中 b0=t_ref（FITS 中的总参考曝光），a0_k 是把“计/厘米^2”乘以 OFF 区域有效面积后、
	在 t_ref 内的 OFF 区域先验计数（可理解为先验的等效观测计数）。

	该类可使用 OFF 区域的观测谱做共轭更新，并聚合为 BackgroundCountsPosterior，
	用于对 ON/OFF 的总计数进行后验预测采样或期望计算。

	字段
	-----
	a0 : np.ndarray (n_chan,)
		每个选择能道的先验 shape。
	b0 : float
		先验 rate（秒），通常为背景文件中的 EXPOSURE（如 1e6 s）。
	area_ratio : float
		A_on / A_off。
	channels : np.ndarray (n_chan,)
		参与先验的 CHANNEL 索引集合（用于与观测谱对齐/筛选）。
	"""

	a0: np.ndarray
	b0: float
	area_ratio: float
	channels: np.ndarray

	# ---------- 构造：来自 EP/WXT 背景 FITS（计/厘米^2 over t_ref） ----------
	@classmethod
	def from_epwxt_background(
		cls,
		fits_path: str,
		*,
		chan_lo: int = 53,
		chan_hi: int = 506,
		arf_path: Optional[str] = None,
		arf_bin_lo: int = 81,
		arf_bin_hi: int = 780,
		pixel_size_mm: float = 0.015,
		pixel_fov_deg: float = 0.00229,
		src_radius_arcmin: float = 9.0,
		bkg_rin_arcmin: float = 18.0,
		bkg_rout_arcmin: float = 36.0,
		verbose: bool = True,
	) -> "BackgroundSpectralPrior":
		"""
		读取单个 EP/WXT 背景 FITS（SPECTRUM: COUNTS 为“计/厘米^2”）并按能团形成逐道先验。

		若提供 ARF，则用 SPECRESP 的第 [arf_bin_lo, arf_bin_hi]（1-based）能道推断能段，
		并用 EBOUNDS 做能段重合筛选；否则回退到 CHANNEL 区间 [chan_lo, chan_hi]。
		OFF/ON 区域面积由像元尺寸及 FoV 计算得到，area_ratio=A_on/A_off。
		"""

		# 1) 可选：由 ARF 推断能段
		band_keV: Optional[Tuple[float, float]] = None
		if arf_path is not None:
			with fits.open(arf_path) as arf:
				if 'SPECRESP' not in arf:
					raise ValueError(f"ARF file {arf_path} lacks SPECRESP extension.")
				hdu_sp = arf['SPECRESP']
				arfd = getattr(hdu_sp, 'data', None)
				if arfd is None:
					raise ValueError(f"ARF file {arf_path} SPECRESP has no data")
				elo = np.asarray(arfd['ENERG_LO'], dtype=float)
				ehi = np.asarray(arfd['ENERG_HI'], dtype=float)
				i0 = max(0, int(arf_bin_lo) - 1)
				i1 = min(elo.size - 1, int(arf_bin_hi) - 1)
				if i1 < i0:
					raise ValueError("arf_bin_hi must be >= arf_bin_lo")
				band_keV = (float(np.min(elo[i0:i1+1])), float(np.max(ehi[i0:i1+1])))
				if verbose:
					print(f"[BackSpecPrior] ARF bins {arf_bin_lo}-{arf_bin_hi} -> band ~{band_keV[0]:.3g}-{band_keV[1]:.3g} keV")

		# 2) 读取背景谱，形成能道筛选和逐道计数/厘米^2
		with fits.open(fits_path) as hdul:
			if 'SPECTRUM' not in hdul:
				raise ValueError(f"FITS file {fits_path} lacks SPECTRUM extension.")
			hdu_spec = hdul['SPECTRUM']
			spec = getattr(hdu_spec, 'data', None)
			hdr = getattr(hdu_spec, 'header', None)
			if spec is None or hdr is None:
				raise ValueError(f"FITS SPECTRUM in {fits_path} missing data/header")

			ch = np.asarray(spec['CHANNEL'], dtype=int)
			mask = (ch >= int(chan_lo)) & (ch <= int(chan_hi))
			if ('EBOUNDS' in hdul) and (band_keV is not None):
				eb = getattr(hdul['EBOUNDS'], 'data', None)
				if eb is not None:
					elo = np.asarray(eb['E_MIN'], dtype=float)
					ehi = np.asarray(eb['E_MAX'], dtype=float)
					mask = (ehi > band_keV[0]) & (elo < band_keV[1])
			channels = ch[mask]
			counts_cm2 = np.asarray(spec['COUNTS'], dtype=float)[mask]
			t_ref = float(hdr.get('EXPOSURE', 1_000_000.0))

			if ('EBOUNDS' in hdul) and verbose:
				eb = getattr(hdul['EBOUNDS'], 'data', None)
				if eb is not None:
					elo = np.asarray(eb['E_MIN'], dtype=float)
					ehi = np.asarray(eb['E_MAX'], dtype=float)
					if band_keV is None:
						sel = (eb['CHANNEL'] >= int(chan_lo)) & (eb['CHANNEL'] <= int(chan_hi))
						emin = float(np.min(elo[sel])) if np.any(sel) else np.nan
						emax = float(np.max(ehi[sel])) if np.any(sel) else np.nan
						print(f"[BackSpecPrior] Channel {chan_lo}-{chan_hi} ~{emin:.3g}-{emax:.3g} keV")
					else:
						print(f"[BackSpecPrior] Using ARF-derived band ~{band_keV[0]:.3g}-{band_keV[1]:.3g} keV")

		# 3) 几何：计算 A_on, A_off 与 area_ratio
		pix_size_cm = float(pixel_size_mm) * 0.1
		apix_cm2 = pix_size_cm ** 2
		omega_pix_deg2 = float(pixel_fov_deg) ** 2
		r_on_deg = float(src_radius_arcmin) / 60.0
		r_in_deg = float(bkg_rin_arcmin) / 60.0
		r_out_deg = float(bkg_rout_arcmin) / 60.0
		A_on_sky_deg2 = float(np.pi * (r_on_deg ** 2))
		A_off_sky_deg2 = float(np.pi * (r_out_deg ** 2 - r_in_deg ** 2))
		Npix_on = A_on_sky_deg2 / omega_pix_deg2
		Npix_off = A_off_sky_deg2 / omega_pix_deg2
		A_on_cm2 = Npix_on * apix_cm2
		A_off_cm2 = Npix_off * apix_cm2
		area_ratio = float(A_on_cm2 / A_off_cm2)
		if verbose:
			print(f"[BackSpecPrior] A_on_cm2={A_on_cm2:.4g}, A_off_cm2={A_off_cm2:.4g}, area_ratio~{area_ratio:.6g}")

		# 4) 逐道先验 a0_k：把“计/厘米^2”换算为 OFF 区域在 t_ref 内的等效计数
		#    Gamma 的形参可取为 a0_k = n_off_k_ref，rate=b0=t_ref
		n_off_k_ref = counts_cm2 * A_off_cm2
		a0 = n_off_k_ref.astype(float)
		b0 = float(t_ref)
		if verbose:
			print(f"[BackSpecPrior] t_ref={t_ref:g}s, sum(a0)={np.sum(a0):.6g}")

		return cls(a0=a0, b0=b0, area_ratio=area_ratio, channels=channels)

	# ---------- 更新：使用 OFF 区域观测谱（PHA） ----------
	def update_with_off_spectrum(
		self,
		pha_path: str,
		*,
		chan_lo: Optional[int] = None,
		chan_hi: Optional[int] = None,
		use_ebounds: bool = True,
		verbose: bool = True,
	) -> BackgroundCountsPosterior:
		"""
		用 OFF 区域的观测谱（PHA，COUNTS 为整数计数）进行共轭更新，聚合为总计数后验。

		- 若 use_ebounds=True，优先用 EBOUNDS 与 self.channels 对齐；否则按 CHANNEL 匹配。
		- 曝光时间从 PHA 的 SPECTRUM header['EXPOSURE'] 读取。
		返回 BackgroundCountsPosterior，参数 a_total=sum(a0)+sum(n_off_obs)，b=b0+t_off_obs。
		"""
		with fits.open(pha_path) as hdul:
			if 'SPECTRUM' not in hdul:
				raise ValueError(f"FITS file {pha_path} lacks SPECTRUM extension.")
			hdu = hdul['SPECTRUM']
			data = getattr(hdu, 'data', None)
			hdr = getattr(hdu, 'header', None)
			if data is None or hdr is None:
				raise ValueError(f"FITS SPECTRUM in {pha_path} missing data/header")

			ch = np.asarray(data['CHANNEL'], dtype=int)
			if use_ebounds and ('EBOUNDS' in hdul):
				# 与 prior 使用的能道集合对齐（基于 CHANNEL 即可）。
				sel = np.isin(ch, self.channels)
			else:
				lo = int(chan_lo) if chan_lo is not None else int(np.min(self.channels))
				hi = int(chan_hi) if chan_hi is not None else int(np.max(self.channels))
				sel = (ch >= lo) & (ch <= hi) & np.isin(ch, self.channels)

			n_off_obs = np.asarray(data['COUNTS'], dtype=float)[sel]
			t_off_obs = float(hdr.get('EXPOSURE', 0.0))
			if t_off_obs <= 0:
				raise ValueError("Observation PHA has non-positive EXPOSURE")

		a_total = float(np.sum(self.a0)) + float(np.sum(n_off_obs))
		b = float(self.b0 + t_off_obs)
		if verbose:
			print(f"[BackSpecPrior] OFF update: t_obs={t_off_obs:g}s, add_counts={np.sum(n_off_obs):.6g}, a_total={a_total:.6g}, b={b:.6g}")

		return BackgroundCountsPosterior(a_total=a_total, b=b, area_ratio=float(self.area_ratio))

	def update_with_on_bg_spectrum(
		self,
		pha_path: str,
		*,
		chan_lo: Optional[int] = None,
		chan_hi: Optional[int] = None,
		use_ebounds: bool = True,
		verbose: bool = True,
	) -> BackgroundCountsPosterior:
		"""
		使用 ON 区域“背景-only”时段的观测谱进行更新：
		- shape 增加 sum(n_on_obs)
		- rate 增加 area_ratio * t_on_obs
		返回聚合后的 BackgroundCountsPosterior。
		"""
		with fits.open(pha_path) as hdul:
			if 'SPECTRUM' not in hdul:
				raise ValueError(f"FITS file {pha_path} lacks SPECTRUM extension.")
			hdu = hdul['SPECTRUM']
			data = getattr(hdu, 'data', None)
			hdr = getattr(hdu, 'header', None)
			if data is None or hdr is None:
				raise ValueError(f"FITS SPECTRUM in {pha_path} missing data/header")

			ch = np.asarray(data['CHANNEL'], dtype=int)
			if use_ebounds and ('EBOUNDS' in hdul):
				sel = np.isin(ch, self.channels)
			else:
				lo = int(chan_lo) if chan_lo is not None else int(np.min(self.channels))
				hi = int(chan_hi) if chan_hi is not None else int(np.max(self.channels))
				sel = (ch >= lo) & (ch <= hi) & np.isin(ch, self.channels)

			n_on_obs = np.asarray(data['COUNTS'], dtype=float)[sel]
			t_on_obs = float(hdr.get('EXPOSURE', 0.0))
			if t_on_obs <= 0:
				raise ValueError("Observation PHA has non-positive EXPOSURE")

		a_total = float(np.sum(self.a0)) + float(np.sum(n_on_obs))
		b = float(self.b0 + self.area_ratio * t_on_obs)
		if verbose:
			print(f"[BackSpecPrior] ON-bg update: t_on={t_on_obs:g}s, add_counts={np.sum(n_on_obs):.6g}, a_total={a_total:.6g}, b={b:.6g}")

		return BackgroundCountsPosterior(a_total=a_total, b=b, area_ratio=float(self.area_ratio))

	# ---------- 快速聚合：不更新，直接返回基于先验的后验（等价于 a_total=sum(a0), b=b0） ----------
	def as_counts_posterior(self) -> BackgroundCountsPosterior:
		return BackgroundCountsPosterior(a_total=float(np.sum(self.a0)), b=float(self.b0), area_ratio=float(self.area_ratio))


