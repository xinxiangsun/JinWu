"""
最大可探测红移估计器（基于文献方法）：

- 依据观测事件（光变曲线 + 最佳谱形）在不同红移 z 进行模拟；
- 使用 XSPEC fakeit（通过 lcfake 中的 K 计算）得到带内计数率，将光变曲线转换到目标红移口径；
- 对每个红移重复 1e4 次泊松抽样以体现计数涨落；
- 用 Li & Ma 公式计算每条模拟光变的 SNR，取中位数作为该红移的代表 SNR；
- 给出达到阈值（默认 SNR>=7）的最大红移估计 z_max。

English
High-z detectability estimator following the paper method: for each redshift,
simulate 10k observed light curves (with Poisson fluctuations) from the
observed LC and best-fit spectrum via pyxspec/fakeit-derived K, compute SNR via
Li&Ma, take the median SNR per z, and find the largest redshift with SNR>=7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Sequence

import numpy as np

from jinwu.lightcurve.lcfake import XspecConfig, build_fake_from_npz
from jinwu.lightcurve.trigger import TriggerDecider, BackgroundSimple


ConfigBuilder = Callable[[float], XspecConfig]


@dataclass
class DetectabilityResult:
	z: np.ndarray
	snr_median: np.ndarray
	snr_p16: Optional[np.ndarray]
	snr_p84: Optional[np.ndarray]
	z_max: Optional[float]


class HighZDetectabilityEstimator:
	"""
	用于计算“最大可探测红移”的估计器。

	参数
	- npz_path: 观测 LC（计数）NPZ 文件路径
	- cfg_ref: 参考（本地）口径的 XSPEC 配置，用来将原始 LC 计数→通量（K_ref）
	- config_for_z: 函数 z->XspecConfig，返回目标红移处的 XSPEC 配置（决定 K_tgt）；
	  若为空且 z_param_index 不为空，则默认以 cfg_ref 为模板替换 params[z_param_index]=z。
	- z0: 原始事件红移（用于时间伸缩 S_t=(1+z)/(1+z0)）
	- area_ratio: α = (t_on A_on)/(t_off A_off)，用于 Li&Ma（若仅面积比，则视作 t_on=t_off）
	- background_rate_on: 源区（on）常数背景率（cts/s）；若 None 则从 NPZ 背景曲线估算
	- n_trials: 每个红移的模拟次数（默认 10000）
	- snr_threshold: 判定“可探测”的 SNR 阈值（默认 7.0）
	- rng: 可选随机数发生器
	- z_param_index: 若未提供 config_for_z，用该索引在 params 中写入 z

	备注
	- 本类默认使用“整段积分”SNR（对整条光变累计 n_on/n_off），更贴合文献描述。
	- 若需滑动窗 SNR，可后续扩展窗口计算逻辑。
	"""

	def __init__(
		self,
		*,
		npz_path: str,
		cfg_ref: XspecConfig,
		z0: float,
		area_ratio: float,
		config_for_z: Optional[ConfigBuilder] = None,
		background_rate_on: Optional[float] = None,
		n_trials: int = 1000,
		snr_threshold: float = 7.0,
		rng: Optional[np.random.Generator] = None,
		z_param_index: Optional[int] = None,
		window: float = 1200.0,
	) -> None:
		self.npz_path = str(npz_path)
		self.cfg_ref = cfg_ref
		self.z0 = float(z0)
		self.area_ratio = float(area_ratio)
		self.config_for_z = config_for_z
		self.background_rate_on = background_rate_on
		self.n_trials = int(n_trials)
		self.snr_threshold = float(snr_threshold)
		self.rng = rng or np.random.default_rng()
		self.z_param_index = z_param_index
		self.window = float(window)

		if (self.config_for_z is None) and (self.z_param_index is None):
			# 提供一个最小保底：使用相同配置（不随 z 变），仅进行时间伸缩
			self.config_for_z = lambda z: self.cfg_ref

	def _cfg_for_z_with_index(self, z: float) -> XspecConfig:
		if self.z_param_index is None:
			return self.cfg_ref
		params = list(self.cfg_ref.params)
		idx = int(self.z_param_index)
		if not (0 <= idx < len(params)):
			raise IndexError(f"z_param_index {idx} out of range for params length {len(params)}")
		params[idx] = float(z)
		return XspecConfig(
			arf=self.cfg_ref.arf,
			rmf=self.cfg_ref.rmf,
			background=self.cfg_ref.background,
			model=self.cfg_ref.model,
			params=[float(p) for p in params],
			band=self.cfg_ref.band,
			exposure=self.cfg_ref.exposure,
			back_exposure=self.cfg_ref.back_exposure,
			xspec_abund=self.cfg_ref.xspec_abund,
			xspec_xsect=self.cfg_ref.xspec_xsect,
			xspec_cosmo=self.cfg_ref.xspec_cosmo,
		)

	def _ensure_background_rate_on(self) -> float:
		if self.background_rate_on is not None:
			return float(self.background_rate_on)
		# 从 NPZ 估算：用 corrected_counts_back 求 off 区域总背景率，再映射到 on 区域
		data = np.load(self.npz_path)
		# 取时间用于 dt
		if 'time_series' in data:
			time = np.asarray(data['time_series'], dtype=float)
		elif 'raw_time_series' in data:
			time = np.asarray(data['raw_time_series'], dtype=float)
		elif 'time' in data:
			time = np.asarray(data['time'], dtype=float)
		else:
			raise ValueError("NPZ lacks time array for background estimation")
		if 'corrected_counts_back' not in data:
			raise ValueError("NPZ lacks 'corrected_counts_back' for background estimation; please provide background_rate_on")
		counts_back_off = np.asarray(data['corrected_counts_back'], dtype=float)
		if counts_back_off.size != time.size:
			raise ValueError("background counts and time length mismatch")
		dt = float(np.median(np.diff(time)))
		total_time = dt * float(len(time))
		rate_off = float(counts_back_off.sum()) / max(total_time, 1e-12)
		# on 区域背景率 = α * off 区域背景率
		b_on = self.area_ratio * rate_off
		self.background_rate_on = b_on
		return b_on

	def _target_config(self, z: float) -> XspecConfig:
		if self.config_for_z is not None and self.config_for_z is not self._cfg_for_z_with_index:
			return self.config_for_z(float(z))
		return self._cfg_for_z_with_index(float(z))

	def simulate_snr_at_z(self, z: float, *, target_dt: Optional[float] = None) -> Tuple[float, np.ndarray]:
		"""
		在给定红移 z 处模拟 n_trials 条“观测光变”，返回（中位数SNR, 所有SNR数组）。
		- 使用 build_fake_from_npz 直接生成 on 区域总计数（源+背景）的泊松化光变；
		- 为每条模拟构造 TriggerDecider（背景以 BackgroundSimple 提供），统一用滑动窗 window 计算最大 SNR；
		- 所有 SNR 计算均通过 trigger 模块实现。
		"""
		b_on = self._ensure_background_rate_on()
		cfg_tgt = self._target_config(z)

		# 先构建目标红移下的期望总率（源+背景），不加噪声；随后每次 trial 对 per-bin 进行泊松抽样
		res = build_fake_from_npz(
			self.npz_path,
			cfg_ref=self.cfg_ref,
			cfg_tgt=cfg_tgt,
			area_ratio=self.area_ratio,
			z0=self.z0,
			z=float(z),
			T0=None,
			target_dt=target_dt,
			add_poisson=False,
			background_rate=b_on,
			output_total_rate=True,
		)
		rate_on = np.asarray(res.rate, dtype=float)
		dt = float(res.dt)
		lam_on = np.clip(rate_on * dt, 0.0, None)

		# off 区域常数率
		rate_off = float(b_on) / max(self.area_ratio, 1e-12)
		t_off_ref = float(self.window)

		snr_vals = np.empty(self.n_trials, dtype=float)
		for i in range(self.n_trials):
			# 按期望对每个 bin 采样 on 区域总计数
			counts_on = self.rng.poisson(lam_on)
			# 为该 trial 采样一个 off 参考计数
			n_off_ref = float(self.rng.poisson(max(rate_off * t_off_ref, 0.0)))

			bg = BackgroundSimple(
				area_ratio=float(self.area_ratio),
				t_off_ref=t_off_ref,
				n_off_ref=n_off_ref,
			)
			decider = TriggerDecider.from_counts(time=res.time, counts=counts_on, dt=dt, bg=bg)
			_, stat = decider.sliding_window(window=self.window, step=None)
			snr_vals[i] = float(stat.get("max_snr", 0.0))

		snr_med = float(np.nanmedian(snr_vals))
		return snr_med, snr_vals

	def sweep(self, z_grid: Sequence[float], *, target_dt: Optional[float] = None, with_spread: bool = True) -> DetectabilityResult:
		"""
		对给定 z_grid 计算每个 z 的 SNR 中位数（以及分散度），并给出 z_max 估计。
		- with_spread=True 时，额外返回 16/84 分位数用于误差带。
		- z_max 通过线性插值近似阈值交叉点；若全程 >= 阈值，则返回最大 z; 若全程 < 阈值，返回 None。
		"""
		z_arr = np.asarray(list(z_grid), dtype=float)
		snr_med = np.zeros_like(z_arr)
		snr_p16 = np.full_like(z_arr, np.nan)
		snr_p84 = np.full_like(z_arr, np.nan)

		for i, z in enumerate(z_arr):
			med, vals = self.simulate_snr_at_z(float(z), target_dt=target_dt)
			snr_med[i] = med
			if with_spread:
				snr_p16[i] = float(np.nanpercentile(vals, 16.0))
				snr_p84[i] = float(np.nanpercentile(vals, 84.0))

		# 寻找阈值穿越点（最后一个 >= 阈值 的 z）
		thr = self.snr_threshold
		idx_ok = np.where(snr_med >= thr)[0]
		if idx_ok.size == 0:
			z_max = None
		else:
			last_ok = int(idx_ok[-1])
			if last_ok == len(z_arr) - 1:
				z_max = float(z_arr[last_ok])
			else:
				# 在线性插值估算与阈值的交点
				x0, y0 = float(z_arr[last_ok]), float(snr_med[last_ok])
				x1, y1 = float(z_arr[last_ok + 1]), float(snr_med[last_ok + 1])
				if y1 == y0:
					z_max = float(x0)
				else:
					t = (thr - y0) / (y1 - y0)
					z_max = float(x0 + t * (x1 - x0))

		return DetectabilityResult(
			z=z_arr,
			snr_median=snr_med,
			snr_p16=snr_p16 if with_spread else None,
			snr_p84=snr_p84 if with_spread else None,
			z_max=z_max,
		)

