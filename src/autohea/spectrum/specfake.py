"""
XSPEC 相关封装：用于批量计算带内 K = rate/flux（基于 fakeit），支持缓存。

设计目标
- 将 pyxspec 的一次性计算打包，减少在生成 1e4 条光变时的重复开销；
- 接口简洁：get_K(arf, rmf, background, model, params, band, ...)->float；
- 缓存键严格基于配置唯一性（路径字符串、模型与参数、能段、环境变量）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import hashlib
import numpy as np

# 统一的数值类型别名（兼容 numpy 数值类型）
Numeric = Union[float, int, np.floating, np.integer]

_HAVE_XSPEC = False
try:  # pragma: no cover
	import xspec  # type: ignore
	from xspec import AllData, AllModels, FakeitSettings  # type: ignore
	_HAVE_XSPEC = True
except Exception:  # pragma: no cover
	_HAVE_XSPEC = False


@dataclass(frozen=True)
class KConfig:
	arf: str
	rmf: str
	background: Optional[str]
	model: str
	params: Tuple[float, ...]
	band: Tuple[float, float]
	exposure: Numeric
	back_exposure: Optional[Numeric]
	xspec_abund: str
	xspec_xsect: str
	xspec_cosmo: str

	def key(self) -> str:
		h = hashlib.sha256()
		def upd(s: str) -> None:
			h.update(s.encode('utf-8'))
		upd(self.arf)
		upd(self.rmf)
		upd(self.background or '')
		upd(self.model)
		upd(','.join(f"{p:.12g}" for p in self.params))
		upd(f"{self.band[0]:.6f}-{self.band[1]:.6f}")
		upd(f"{float(self.exposure):.6f}")
		be = None if self.back_exposure is None else float(self.back_exposure)
		upd("BE:" + ("None" if be is None else f"{be:.6f}"))
		upd(self.xspec_abund)
		upd(self.xspec_xsect)
		upd(self.xspec_cosmo)
		return h.hexdigest()


class XspecSession:
	"""一个轻量的 XSPEC 会话，提供 K 计算能力。"""

	def __init__(self) -> None:
		if not _HAVE_XSPEC:
			raise RuntimeError("pyxspec is not available; ensure HEASoft/XSPEC is installed and on PYTHONPATH")

	def compute_K(self, cfg: KConfig) -> float:
		# 环境
		xspec.Xset.abund = cfg.xspec_abund  # type: ignore
		xspec.Xset.xsect = cfg.xspec_xsect  # type: ignore
		xspec.Xset.cosmo = cfg.xspec_cosmo  # type: ignore
		xspec.Xset.allowPrompting = False  # type: ignore

		# 清空数据/模型
		AllData.clear()
		AllModels.clear()

		# 模型
		model = xspec.Model(cfg.model)  # type: ignore
		params = list(cfg.params)
		p_objs = []
		for comp_name in model.componentNames:
			comp = getattr(model, comp_name)
			for pname in comp.parameterNames:
				p_objs.append(getattr(comp, pname))
		if len(params) != len(p_objs):
			raise ValueError(f"model param count mismatch: got {len(params)} but model has {len(p_objs)}")
		for pobj, val in zip(p_objs, params):
			pobj.values = float(val)

		# fakeit
		bexpo = cfg.back_exposure if (cfg.back_exposure is not None) else cfg.exposure
		if cfg.background:
			fake = FakeitSettings(
				response=str(cfg.rmf),
				arf=str(cfg.arf),
				exposure=str(float(cfg.exposure)),
				backExposure=str(float(bexpo)),
				background=str(cfg.background),
			)
		else:
			fake = FakeitSettings(
				response=str(cfg.rmf),
				arf=str(cfg.arf),
				exposure=str(float(cfg.exposure)),
				backExposure=str(float(bexpo)),
			)
		AllData.fakeit(1, fake, noWrite=True)

		# 能段
		emin, emax = cfg.band
		AllData.notice("all")
		AllData.ignore(f"**-{float(emin)} {float(emax)}-**")
		AllData.ignore("bad")

		spec = AllData(1)
		spec_any = spec  # type: ignore[assignment]
		rate = float(spec_any.rate[3])  # type: ignore[attr-defined]
		xspec.AllModels.calcFlux(f"{float(emin)} {float(emax)}")
		flux = float(spec_any.flux[0])  # type: ignore[attr-defined]

		if flux <= 0:
			raise RuntimeError("XSPEC returned non-positive flux; check model and band")
		return float(rate / flux)


class XspecKFactory:
	"""
	K 计算工厂 + 缓存。

	- get_K(...): 返回 K，如缓存未命中则调用 XSPEC 计算并存入缓存。
	- clear(): 清空缓存。
	"""

	def __init__(self) -> None:
		self._cache: Dict[str, float] = {}
		self._session: Optional[XspecSession] = None

	def _ensure_session(self) -> XspecSession:
		if self._session is None:
			self._session = XspecSession()
		return self._session

	def get_K(
		self,
		*,
		arf: str,
		rmf: str,
		background: Optional[str],
		model: str,
		params: Tuple[float, ...],
		band: Tuple[float, float],
		exposure: Numeric,
		back_exposure: Optional[Numeric],
		xspec_abund: str,
		xspec_xsect: str,
		xspec_cosmo: str,
	) -> float:
		cfg = KConfig(
			arf=arf,
			rmf=rmf,
			background=background,
			model=model,
			params=params,
			band=band,
			exposure=exposure,
			back_exposure=back_exposure,
			xspec_abund=xspec_abund,
			xspec_xsect=xspec_xsect,
			xspec_cosmo=xspec_cosmo,
		)
		key = cfg.key()
		if key in self._cache:
			return self._cache[key]
		session = self._ensure_session()
		K = session.compute_K(cfg)
		self._cache[key] = float(K)
		return float(K)

	def clear(self) -> None:
		self._cache.clear()


__all__ = [
	'XspecKFactory',
]

