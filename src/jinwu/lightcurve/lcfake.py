"""
基于观测计数光变 + pyxspec(fakeit) 的光变曲线构建器
Light-curve builder based on observed counts and pyxspec(fakeit)

目标 | Goals
- 从 NPZ 计数光变（counts per bin）读取时间序列；
	Load time series from an NPZ file containing counts per bin.
- 通过 pyxspec + 响应（RMF/ARF）计算 K=rate/flux；
	Compute K=rate/flux via pyxspec and instrument responses (RMF/ARF).
- 将计数光变转换为能通量（F=R/K），可选按红移进行时间伸缩；
	Convert counts to energy flux (F=R/K), optionally time-stretch by redshift.
- 使用（同一或目标）配置的 K 将 F→R'，并进行泊松采样得到假光变；
	Use target K to convert F→R' and draw Poisson counts to synthesize an LC.
- 输出结构兼容 TriggerDecider（time、counts、dt）。
	Output arrays are TriggerDecider-ready (time, counts, dt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import json

import numpy as np

from jinwu.spectrum.specfake import XspecKFactory

# 统一的数值类型别名（兼容 numpy 数值类型）
# Unified numeric alias (compatible with numpy numeric scalars)
Numeric = Union[float, np.floating, np.integer]


# ----------------------------- 配置与数据类 -----------------------------

@dataclass
class XspecConfig:
	"""
	pyxspec fakeit 的必要配置。
	Required configuration for pyxspec fakeit.

	字段
	- arf: ARF 路径 | ARF file path
	- rmf: RMF 路径 | RMF file path
	- background: 可选背景 PHA（作为 fakeit 的 background）| optional background PHA
	- model: XSPEC 模型表达式（例如 'phabs*pow' 或 'tbabs*ztbabs*powerlaw' 等）| model expr
	- params: 模型参数（按照 XSPEC 参数顺序）| parameter values in XSPEC order
	- band: 能段 (emin_keV, emax_keV) | energy band in keV
	- exposure: 源区曝光秒数 | source exposure (s)
	- back_exposure: 背景区曝光秒数（None 表示等于 exposure）| background exposure
	- xspec_abund/xspec_xsect/xspec_cosmo: XSPEC 环境常量 | XSPEC globals
	"""

	arf: str
	rmf: str
	background: Optional[str]
	model: str
	params: Tuple[float, ...]
	band: Tuple[float, float]
	exposure: Numeric
	back_exposure: Optional[Numeric]
	xspec_abund: str = 'wilm'
	xspec_xsect: str = 'vern'
	xspec_cosmo: str = '67.66 0 0.6888463055445441'


@dataclass
class LCSimResult:
	"""
	输出结果数据类（TriggerDecider 兼容）。
	Result container compatible with TriggerDecider.
	- time: 1D 数组，bin 左边界 | 1D array, left edges of bins
	- counts: 1D 整型数组，泊松计数或取整期望 | integer counts per bin
	- dt: 标量，bin 宽度（秒）| scalar bin width in seconds
	- rate/error: 附带速率与误差（便于检视）| derived rate and errors
	- meta: 附加信息（K_ref、K_tgt、S_t 等）| extra metadata
	"""

	time: np.ndarray
	counts: np.ndarray
	dt: float
	rate: np.ndarray
	error: np.ndarray
	meta: Dict[str, Any]

	def to_trigger_inputs(self) -> Tuple[np.ndarray, np.ndarray, float]:
		return self.time, self.counts, float(self.dt)


@dataclass
class LCSimPairResult:
	"""
	ON/OFF 成对光变曲线（分别包含源区与背景区的计数），适配需要 Li&Ma 等统计量的触发流程。
	Paired ON/OFF light curves (source and background regions) with independent fluctuations.

	字段 | Fields
	- time: 1D 数组，bin 左边界 | 1D array of left bin edges
	- counts_on: 源区总计数（源信号 + 源区背景）| ON counts (signal + ON background)
	- counts_off: 背景区计数 | OFF counts (background region)
	- dt: bin 宽度（秒）| bin width in seconds
	- rate_on/off: 速率（便于检视）| derived rates
	- error_on/off: 误差（sqrt(N)/dt）| errors
	- meta: 额外信息（K_ref/K_tgt/alpha 等）| extras
	"""

	time: np.ndarray
	counts_on: np.ndarray
	counts_off: np.ndarray
	dt: float
	rate_on: np.ndarray
	rate_off: np.ndarray
	error_on: np.ndarray
	error_off: np.ndarray
	meta: Dict[str, Any]

	def to_trigger_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
		"""
		返回触发判定需要的 (time, counts_on, counts_off, dt)
		Return (time, counts_on, counts_off, dt) for trigger modules.
		"""
		return self.time, self.counts_on, self.counts_off, float(self.dt)


# ----------------------------- NPZ 读入工具 -----------------------------

def _infer_dt(time: np.ndarray) -> float:
	"""根据时间序列估计等间隔 dt; 若异常则抛错。
	Infer bin width dt from time array; raise if invalid.
	"""
	if time.ndim != 1 or time.size < 2:
		raise ValueError("time must be 1D with at least two samples")
	dt = float(np.median(np.diff(time)))
	if not np.isfinite(dt) or dt <= 0:
		raise ValueError("failed to infer a positive dt from time array")
	return dt


def load_counts_npz(npz_path: str, *, area_ratio: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
	"""
	从标准 NPZ 读入计数光变（ON 区域）。
	Load ON-region counts light curve from a standard NPZ.

	选择逻辑（键名）：
	- time: 'time_series' > 'raw_time_series' > 'time'
	- counts:
	  1) 'corrected_counts_src'（优先，源区 counts per bin）
	  2) 若存在净+背景：'corrected_counts' 与 'corrected_counts_back'，则
		 counts_on = net + area_ratio * back（需要提供 area_ratio）
	  3) 回退：'raw_corrected_counts'（可能是净计数，谨慎使用）
	返回：time, counts_on, dt（秒）
	Return: time, counts_on, dt (seconds)
	"""
	data = np.load(npz_path)

	# 读取时间轴 | Read time axis
	if 'time_series' in data:
		time = np.asarray(data['time_series'], dtype=float)
	elif 'raw_time_series' in data:
		time = np.asarray(data['raw_time_series'], dtype=float)
	elif 'time' in data:
		time = np.asarray(data['time'], dtype=float)
	else:
		raise ValueError("NPZ lacks time array: tried 'time_series', 'raw_time_series', 'time'")

	# 读取计数序列（ON 总计数）| Read counts (ON total)
	counts = None
	if 'corrected_counts_src' in data:
		counts = np.asarray(data['corrected_counts_src'], dtype=float)
	elif ('corrected_counts' in data) and ('corrected_counts_back' in data):
		if area_ratio is None:
			raise ValueError("area_ratio required to combine net and back counts")
		net = np.asarray(data['corrected_counts'], dtype=float)
		back = np.asarray(data['corrected_counts_back'], dtype=float)
		counts = net + float(area_ratio) * back
	elif 'raw_corrected_counts' in data:
		counts = np.asarray(data['raw_corrected_counts'], dtype=float)
	else:
		# 尝试通用 'counts'
		if 'counts' in data:
			counts = np.asarray(data['counts'], dtype=float)
		else:
			raise ValueError("NPZ lacks counts array with known keys")

	if time.size != counts.size:
		raise ValueError("time and counts length mismatch")

	# 估计 dt | Infer dt
	dt = _infer_dt(time)
	return time, counts, dt


_K_FACTORY = XspecKFactory()


# ------------------------------ 主流程 API ------------------------------

def build_fake_from_npz(
	npz_path: str,
	cfg_ref: XspecConfig,
	cfg_tgt: Optional[XspecConfig] = None,
	*,
	area_ratio: Optional[float] = None,
	z0: Optional[float] = None,
	z: Optional[float] = None,
	T0: Optional[float] = None,
	target_dt: Optional[float] = None,
	add_poisson: bool = True,
	background_rate: Optional[float] = None,
	output_total_rate: bool = False,
) -> LCSimResult:
	"""
	端到端：从 NPZ 计数光变 → flux →（可选时间伸缩）→ 目标配置下的假光变。

	- cfg_ref: 用于把原始 LC 的 rate↔flux（得到 K_ref）
	- cfg_tgt: 用于把 flux→rate'(得到 K_tgt）；若 None，则使用 cfg_ref
	- z0,z,T0: 若提供，则按 S_t=(1+z)/(1+z0) 拉伸时间轴；否则不改变时间。
	- background_rate: 常数背景率(cts/s);  若 output_total_rate=False，则输出净源率
	- 返回：`LCSimResult(time, counts, dt, rate, error, meta)`
	"""
	# 1) 读入 NPZ 计数光变 | Read counts LC
	time, counts, dt_in = load_counts_npz(npz_path, area_ratio=area_ratio)
	rate_in = np.asarray(counts, dtype=float) / float(dt_in)

	# 2) 计算 K_ref 并得到 flux(t) | Compute K_ref and flux(t)
	K_ref = _K_FACTORY.get_K(
		arf=cfg_ref.arf,
		rmf=cfg_ref.rmf,
		background=cfg_ref.background,
		model=cfg_ref.model,
		params=tuple(cfg_ref.params),
		band=cfg_ref.band,
		exposure=cfg_ref.exposure,
		back_exposure=cfg_ref.back_exposure,
		xspec_abund=cfg_ref.xspec_abund,
		xspec_xsect=cfg_ref.xspec_xsect,
		xspec_cosmo=cfg_ref.xspec_cosmo,
	)
	flux = rate_in / float(K_ref)

	# 3) 时间伸缩（如需要）| Time-stretch if requested
	if (z0 is not None) and (z is not None):
		S_t = float(1.0 + z) / float(1.0 + z0)
		if T0 is None:
			T0 = float(time[0])
		t_out_native = T0 + (time - T0) * S_t
		dt_out_native = float(dt_in) * S_t
		if (target_dt is None) or (target_dt <= 0):
			t_out = t_out_native
			dt_out = dt_out_native
			flux_out = flux.copy()
		else:
			dt_out = float(target_dt)
			t_start = float(t_out_native[0])
			t_stop = float(t_out_native[-1])
			nbin = int(max(1, np.floor((t_stop - t_start) / dt_out)))
			t_out = t_start + np.arange(nbin, dtype=float) * dt_out
			# s(t)=flux/F_ref 再映射也可以；这里直接对 flux 做线性插值（守恒可后续增强）
			# Alternatively, rebin by integrating flux; here we use linear interp for simplicity.
			flux_out = np.interp(T0 + (t_out - T0) / S_t, time, flux, left=0.0, right=0.0)
	else:
		# 不伸缩
		t_out = time.copy()
		dt_out = float(dt_in)
		flux_out = flux.copy()
		S_t = 1.0

	# 4) 计算 K_tgt 并得到 rate'(t) | Compute K_tgt and source rate
	cfg_eff = cfg_tgt if cfg_tgt is not None else cfg_ref
	K_tgt = _K_FACTORY.get_K(
		arf=cfg_eff.arf,
		rmf=cfg_eff.rmf,
		background=cfg_eff.background,
		model=cfg_eff.model,
		params=tuple(cfg_eff.params),
		band=cfg_eff.band,
		exposure=cfg_eff.exposure,
		back_exposure=cfg_eff.back_exposure,
		xspec_abund=cfg_eff.xspec_abund,
		xspec_xsect=cfg_eff.xspec_xsect,
		xspec_cosmo=cfg_eff.xspec_cosmo,
	)
	rate_src = np.clip(flux_out * float(K_tgt), a_min=0.0, a_max=None)

	# 5) 背景与泊松采样 | Background add and Poisson draw
	if background_rate is not None:
		b = float(background_rate)
	else:
		b = 0.0

	dtv = float(dt_out)
	if add_poisson:
		rng = np.random.default_rng()
		if output_total_rate:
			# 采样总计数（源+背景）
			# Sample total counts (source + background)
			lam_tot = np.clip(rate_src + b, 0.0, None) * dtv
			n_tot = rng.poisson(lam_tot)
			counts_out = n_tot.astype(int)
			rate_out = counts_out / dtv
			err = np.sqrt(np.maximum(counts_out, 1)) / dtv
		else:
			# 单独采样源计数，避免“总计数-期望背景”的不正确做法
			# Draw source-only counts to avoid subtracting expected background.
			lam_src = np.clip(rate_src, 0.0, None) * dtv
			n_src = rng.poisson(lam_src)
			counts_out = n_src.astype(int)
			rate_out = counts_out / dtv
			err = np.sqrt(np.maximum(counts_out, 1)) / dtv
	else:
		if output_total_rate:
			counts_out = np.asarray(np.round(np.clip(rate_src + b, 0.0, None) * dtv), dtype=int)
		else:
			counts_out = np.asarray(np.round(np.clip(rate_src, 0.0, None) * dtv), dtype=int)
		rate_out = counts_out / dtv
		err = np.sqrt(np.maximum(counts_out, 1)) / dtv

	# 汇总元信息，便于溯源与调试 | Collect meta info for inspection
	meta = dict(K_ref=float(K_ref), K_tgt=float(K_tgt), S_t=float(S_t), dt_in=float(dt_in), dt_out=float(dt_out))
	return LCSimResult(time=t_out.astype(float), counts=counts_out.astype(int), dt=float(dt_out), rate=rate_out.astype(float), error=err.astype(float), meta=meta)


def build_fake_on_off_from_npz(
	npz_path: str,
	cfg_ref: XspecConfig,
	cfg_tgt: Optional[XspecConfig] = None,
	*,
	alpha: float,
	# 背景区的平均本底率（cts/s，OFF 区）；若提供 ON 区常数本底率 background_rate_on，则将忽略该参数
	background_rate_off: Optional[float] = None,
	# 源区常数本底率（cts/s，ON 区）；若提供则优先，若未提供且 background_rate_off 提供，则按 alpha 缩放：ON = alpha * OFF
	background_rate_on: Optional[float] = None,
	z0: Optional[float] = None,
	z: Optional[float] = None,
	T0: Optional[float] = None,
	target_dt: Optional[float] = None,
	add_poisson: bool = True,
) -> LCSimPairResult:
	"""
	端到端（成对）：从 NPZ 计数光变 → flux →（可选时间伸缩）→ 目标配置下生成 ON/OFF 成对模拟光变。

	关键点 | Key points
	- 三个独立涨落：源信号（仅 ON）、源区背景（ON 背景）、背景区背景（OFF）。
	- 面积比 alpha = BACKSCAL_src / BACKSCAL_bkg，用于将 OFF 背景缩放到 ON 背景（ON = alpha * OFF）。
	- 背景率可指定为 OFF 或 ON 的常数；若仅提供 OFF，则自动按 alpha 缩放得到 ON。
	- 输出 counts_on = n_signal + n_bkg_on，counts_off = n_bkg_off。
	"""

	# 1) 读入 NPZ（只用时间轴与源区域输入的计数来推导源信号形状）
	time, counts, dt_in = load_counts_npz(npz_path)
	rate_in = np.asarray(counts, dtype=float) / float(dt_in)

	# 2) K_ref → flux
	K_ref = _K_FACTORY.get_K(
		arf=cfg_ref.arf,
		rmf=cfg_ref.rmf,
		background=cfg_ref.background,
		model=cfg_ref.model,
		params=tuple(cfg_ref.params),
		band=cfg_ref.band,
		exposure=cfg_ref.exposure,
		back_exposure=cfg_ref.back_exposure,
		xspec_abund=cfg_ref.xspec_abund,
		xspec_xsect=cfg_ref.xspec_xsect,
		xspec_cosmo=cfg_ref.xspec_cosmo,
	)
	flux = rate_in / float(K_ref)

	# 3) 时间伸缩（如需要）
	if (z0 is not None) and (z is not None):
		S_t = float(1.0 + z) / float(1.0 + z0)
		if T0 is None:
			T0 = float(time[0])
		t_out_native = T0 + (time - T0) * S_t
		dt_out_native = float(dt_in) * S_t
		if (target_dt is None) or (target_dt <= 0):
			t_out = t_out_native
			dt_out = dt_out_native
			flux_out = flux.copy()
		else:
			dt_out = float(target_dt)
			t_start = float(t_out_native[0])
			t_stop = float(t_out_native[-1])
			nbin = int(max(1, np.floor((t_stop - t_start) / dt_out)))
			t_out = t_start + np.arange(nbin, dtype=float) * dt_out
			flux_out = np.interp(T0 + (t_out - T0) / S_t, time, flux, left=0.0, right=0.0)
	else:
		t_out = time.copy()
		dt_out = float(dt_in)
		flux_out = flux.copy()
		S_t = 1.0

	# 4) K_tgt → 源信号速率（仅 ON）
	cfg_eff = cfg_tgt if cfg_tgt is not None else cfg_ref
	K_tgt = _K_FACTORY.get_K(
		arf=cfg_eff.arf,
		rmf=cfg_eff.rmf,
		background=cfg_eff.background,
		model=cfg_eff.model,
		params=tuple(cfg_eff.params),
		band=cfg_eff.band,
		exposure=cfg_eff.exposure,
		back_exposure=cfg_eff.back_exposure,
		xspec_abund=cfg_eff.xspec_abund,
		xspec_xsect=cfg_eff.xspec_xsect,
		xspec_cosmo=cfg_eff.xspec_cosmo,
	)
	rate_signal_on = np.clip(flux_out * float(K_tgt), a_min=0.0, a_max=None)

	# 5) 背景率（OFF 与 ON）
	if (background_rate_on is None) and (background_rate_off is None):
		bkg_rate_off = 0.0
		bkg_rate_on = 0.0
	elif background_rate_on is not None:
		bkg_rate_on = float(background_rate_on)
		# 若提供了 ON，而也提供了 OFF，则忽略 OFF；若未提供 OFF，则按 alpha 反推 OFF
		bkg_rate_off = float(bkg_rate_on) / float(alpha) if alpha > 0 else 0.0
	else:
		bkg_rate_off = float(background_rate_off)  # type: ignore[arg-type]
		bkg_rate_on = float(alpha) * bkg_rate_off

	# 6) 泊松采样三路涨落
	dtv = float(dt_out)
	if add_poisson:
		rng = np.random.default_rng()
		n_sig = rng.poisson(np.clip(rate_signal_on, 0.0, None) * dtv).astype(int)
		n_bkg_off = rng.poisson(max(bkg_rate_off, 0.0) * dtv, size=t_out.size).astype(int)
		n_bkg_on = rng.poisson(max(bkg_rate_on, 0.0) * dtv, size=t_out.size).astype(int)
	else:
		n_sig = np.asarray(np.round(np.clip(rate_signal_on, 0.0, None) * dtv), dtype=int)
		n_bkg_off = np.asarray(np.round(max(bkg_rate_off, 0.0) * dtv * np.ones_like(t_out)), dtype=int)
		n_bkg_on = np.asarray(np.round(max(bkg_rate_on, 0.0) * dtv * np.ones_like(t_out)), dtype=int)

	counts_on = (n_sig + n_bkg_on).astype(int)
	counts_off = n_bkg_off.astype(int)

	rate_on = counts_on / dtv
	rate_off = counts_off / dtv
	err_on = np.sqrt(np.maximum(counts_on, 1)) / dtv
	err_off = np.sqrt(np.maximum(counts_off, 1)) / dtv

	meta: Dict[str, Any] = dict(
		K_ref=float(K_ref), K_tgt=float(K_tgt), S_t=float(S_t), dt_in=float(dt_in), dt_out=float(dt_out), alpha=float(alpha),
		bkg_rate_on=float(bkg_rate_on), bkg_rate_off=float(bkg_rate_off)
	)

	return LCSimPairResult(
		time=t_out.astype(float),
		counts_on=counts_on,
		counts_off=counts_off,
		dt=float(dt_out),
		rate_on=rate_on.astype(float),
		rate_off=rate_off.astype(float),
		error_on=err_on.astype(float),
		error_off=err_off.astype(float),
		meta=meta,
	)


def save_on_off_lightcurve(
	res: LCSimPairResult,
	out_path: Union[str, Path],
	*,
	compressed: bool = True,
	include_rates: bool = True,
) -> Path:
	"""
	将成对 ON/OFF 光变曲线保存到单个文件（.npz 或 .npy）。
	Save paired ON/OFF light curves into a single file (.npz or .npy).

	约定 | Conventions
	- .npz：推荐，键值清晰；meta 以 JSON 字符串保存到 'meta_json'（同时尽可能扁平化写入常用标量）。
	- .npy：以 object dict 方式保存（需要 allow_pickle=True），便于一次性读回字典。
	"""
	p = Path(out_path)
	p.parent.mkdir(parents=True, exist_ok=True)

	if p.suffix.lower() == '.npz':
		payload: Dict[str, Any] = {
			'time': res.time.astype(float),
			'counts_on': res.counts_on.astype(int),
			'counts_off': res.counts_off.astype(int),
			'dt': float(res.dt),
		}
		if include_rates:
			payload.update({
				'rate_on': res.rate_on.astype(float),
				'rate_off': res.rate_off.astype(float),
				'error_on': res.error_on.astype(float),
				'error_off': res.error_off.astype(float),
			})
		# 便捷的扁平化标量（若存在）
		for k in ('K_ref', 'K_tgt', 'S_t', 'dt_in', 'dt_out', 'alpha', 'bkg_rate_on', 'bkg_rate_off'):
			if k in res.meta:
				try:
					payload[k] = float(res.meta[k])
				except Exception:
					pass
		payload['meta_json'] = json.dumps(res.meta, ensure_ascii=False)
		if compressed:
			np.savez_compressed(p, **payload)
		else:
			np.savez(p, **payload)
		return p
	else:
		# .npy 或其他扩展名：保存为 object dict（需要 pickle）
		payload = {
			'time': res.time,
			'counts_on': res.counts_on,
			'counts_off': res.counts_off,
			'dt': float(res.dt),
			'rate_on': res.rate_on,
			'rate_off': res.rate_off,
			'error_on': res.error_on,
			'error_off': res.error_off,
			'meta': res.meta,
		}
		np.save(p, np.array(payload, dtype=object), allow_pickle=True)
		return p


def load_on_off_lightcurve(
	in_path: Union[str, Path]
) -> LCSimPairResult:
	"""
	从 .npz 或 .npy 文件加载成对 ON/OFF 光变，返回 LCSimPairResult。
	Load paired ON/OFF light curves saved by save_on_off_lightcurve.
	"""
	p = Path(in_path)
	if p.suffix.lower() == '.npz':
		data = np.load(p, allow_pickle=True)
		time = np.asarray(data['time'], dtype=float)
		counts_on = np.asarray(data['counts_on'], dtype=int)
		counts_off = np.asarray(data['counts_off'], dtype=int)
		dt = float(np.asarray(data['dt']))
		rate_on = np.asarray(data['rate_on'], dtype=float) if 'rate_on' in data else counts_on / dt
		rate_off = np.asarray(data['rate_off'], dtype=float) if 'rate_off' in data else counts_off / dt
		error_on = np.asarray(data['error_on'], dtype=float) if 'error_on' in data else np.sqrt(np.maximum(counts_on, 1)) / dt
		error_off = np.asarray(data['error_off'], dtype=float) if 'error_off' in data else np.sqrt(np.maximum(counts_off, 1)) / dt
		meta: Dict[str, Any] = {}
		if 'meta_json' in data:
			try:
				meta = json.loads(str(data['meta_json']))
			except Exception:
				meta = {}
		# 回填常见标量
		for k in ('K_ref', 'K_tgt', 'S_t', 'dt_in', 'dt_out', 'alpha', 'bkg_rate_on', 'bkg_rate_off'):
			if k in data and k not in meta:
				try:
					meta[k] = float(np.asarray(data[k]))
				except Exception:
					pass
	else:
		arr = np.load(p, allow_pickle=True)
		payload = arr.item() if hasattr(arr, 'item') else arr
		time = np.asarray(payload['time'], dtype=float)
		counts_on = np.asarray(payload['counts_on'], dtype=int)
		counts_off = np.asarray(payload['counts_off'], dtype=int)
		dt = float(payload['dt'])
		rate_on = np.asarray(payload.get('rate_on', counts_on / dt), dtype=float)
		rate_off = np.asarray(payload.get('rate_off', counts_off / dt), dtype=float)
		error_on = np.asarray(payload.get('error_on', np.sqrt(np.maximum(counts_on, 1)) / dt), dtype=float)
		error_off = np.asarray(payload.get('error_off', np.sqrt(np.maximum(counts_off, 1)) / dt), dtype=float)
		meta = dict(payload.get('meta', {}))

	return LCSimPairResult(
		time=time,
		counts_on=counts_on,
		counts_off=counts_off,
		dt=dt,
		rate_on=rate_on,
		rate_off=rate_off,
		error_on=error_on,
		error_off=error_off,
		meta=meta,
	)


__all__ = [
	'XspecConfig',
	'LCSimResult',
    'LCSimPairResult',
	'load_counts_npz',
	'build_fake_from_npz',
    'build_fake_on_off_from_npz',
    'save_on_off_lightcurve',
    'load_on_off_lightcurve',
]

