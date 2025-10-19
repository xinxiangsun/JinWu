"""
基于观测计数光变 + pyxspec(fakeit) 的光变曲线构建器

目标
----
- 从 NPZ 计数光变（counts per bin）读取时间序列；
- 通过 pyxspec + 响应文件（RMF/ARF）计算带内 rate/flux 换算因子 K = R/F；
- 将计数光变转换为能通量光变（F=R/K），可选进行时间伸缩（红移）；
- 使用（同一或目标）配置的 K 将 F → R'，并进行泊松采样生成假的光变曲线；
- 输出结构兼容 TriggerDecider（time、counts、dt）。

English
-------
Load counts light curve from NPZ, compute a flux↔rate conversion constant K via
pyxspec fakeit with provided RMF/ARF and model, convert counts→flux, optionally
time-stretch by redshift, then convert flux→rate using target config's K and
draw Poisson counts. Return TriggerDecider-ready arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np

from autohea.spectrum.specfake import XspecKFactory

# 统一的数值类型别名（兼容 numpy 数值类型）
Numeric = Union[float, int, np.floating, np.integer]


# ----------------------------- 配置与数据类 -----------------------------

@dataclass
class XspecConfig:
	"""
	pyxspec fakeit 的必要配置。

	字段
	- arf: ARF 路径
	- rmf: RMF 路径
	- background: 可选背景 PHA（作为 fakeit 的 background）
	- model: XSPEC 模型表达式（例如 'phabs*pow' 或 'tbabs*ztbabs*powerlaw' 等）
	- params: 模型参数（按照 XSPEC 参数顺序）
	- band: 能段 (emin_keV, emax_keV)
	- exposure: fakeit 使用的源区曝光秒数（秒）
	- back_exposure: fakeit 使用的背景区曝光秒数（秒）；缺省 None 时等于 exposure
	- xspec_abund/xspec_xsect/xspec_cosmo: XSPEC 环境常量
	"""

	arf: str
	rmf: str
	background: Optional[str]
	model: str
	params: list[float]
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
	- time: 1D 数组，bin 左边界
	- counts: 1D 整型数组，泊松计数或取整期望
	- dt: 标量，bin 宽度（秒）
	- rate/error: 附带速率与误差（便于检视）
	- meta: 附加信息（K_ref、K_tgt、S_t 等）
	"""

	time: np.ndarray
	counts: np.ndarray
	dt: float
	rate: np.ndarray
	error: np.ndarray
	meta: Dict[str, Any]

	def to_trigger_inputs(self) -> Tuple[np.ndarray, np.ndarray, float]:
		return self.time, self.counts, float(self.dt)


# ----------------------------- NPZ 读入工具 -----------------------------

def _infer_dt(time: np.ndarray) -> float:
	if time.ndim != 1 or time.size < 2:
		raise ValueError("time must be 1D with at least two samples")
	dt = float(np.median(np.diff(time)))
	if not np.isfinite(dt) or dt <= 0:
		raise ValueError("failed to infer a positive dt from time array")
	return dt


def load_counts_npz(npz_path: str, *, area_ratio: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
	"""
	从标准 NPZ 读入计数光变（ON 区域）。

	选择逻辑（键名）：
	- time: 'time_series' > 'raw_time_series' > 'time'
	- counts:
	  1) 'corrected_counts_src'（优先，源区 counts per bin）
	  2) 若存在净+背景：'corrected_counts' 与 'corrected_counts_back'，则
		 counts_on = net + area_ratio * back（需要提供 area_ratio）
	  3) 回退：'raw_corrected_counts'（可能是净计数，谨慎使用）
	返回：time, counts_on, dt（秒）
	"""
	data = np.load(npz_path)

	# time
	if 'time_series' in data:
		time = np.asarray(data['time_series'], dtype=float)
	elif 'raw_time_series' in data:
		time = np.asarray(data['raw_time_series'], dtype=float)
	elif 'time' in data:
		time = np.asarray(data['time'], dtype=float)
	else:
		raise ValueError("NPZ lacks time array: tried 'time_series', 'raw_time_series', 'time'")

	# counts
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
	- cfg_tgt: 用于把 flux→rate'（得到 K_tgt）；若 None，则使用 cfg_ref
	- z0,z,T0: 若提供，则按 S_t=(1+z)/(1+z0) 拉伸时间轴；否则不改变时间。
	- background_rate: 常数背景率（cts/s）；若 output_total_rate=False，则输出净源率
	- 返回：`LCSimResult(time, counts, dt, rate, error, meta)`
	"""
	# 1) 读入 NPZ 计数光变
	time, counts, dt_in = load_counts_npz(npz_path, area_ratio=area_ratio)
	rate_in = np.asarray(counts, dtype=float) / float(dt_in)

	# 2) 计算 K_ref 并得到 flux(t)
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
			# s(t)=flux/F_ref 再映射也可以；这里直接对 flux 做线性插值（守恒可后续增强）
			flux_out = np.interp(T0 + (t_out - T0) / S_t, time, flux, left=0.0, right=0.0)
	else:
		# 不伸缩
		t_out = time.copy()
		dt_out = float(dt_in)
		flux_out = flux.copy()
		S_t = 1.0

	# 4) 计算 K_tgt 并得到 rate'(t)
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

	# 5) 背景与泊松采样
	if background_rate is not None:
		b = float(background_rate)
	else:
		b = 0.0

	dtv = float(dt_out)
	if add_poisson:
		rng = np.random.default_rng()
		if output_total_rate:
			# 采样总计数（源+背景）
			lam_tot = np.clip(rate_src + b, 0.0, None) * dtv
			n_tot = rng.poisson(lam_tot)
			counts_out = n_tot.astype(int)
			rate_out = counts_out / dtv
			err = np.sqrt(np.maximum(counts_out, 1)) / dtv
		else:
			# 单独采样源计数，避免“总计数-期望背景”的不正确做法
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

	meta = dict(K_ref=float(K_ref), K_tgt=float(K_tgt), S_t=float(S_t), dt_in=float(dt_in), dt_out=float(dt_out))
	return LCSimResult(time=t_out.astype(float), counts=counts_out.astype(int), dt=float(dt_out), rate=rate_out.astype(float), error=err.astype(float), meta=meta)


__all__ = [
	'XspecConfig',
	'LCSimResult',
	'load_counts_npz',
	'build_fake_from_npz',
]

