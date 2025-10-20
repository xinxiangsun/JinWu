"""
XSPEC 封装：计算带内 K = rate/flux（依据 fakeit），带缓存。
XSPEC wrapper: compute in-band K = rate/flux (via fakeit) with caching.

设计目标 | Goals
- 将 pyxspec 的一次性计算打包，避免在 1e4 次模拟中重复开销；
	Bundle one-shot pyxspec computations to avoid repeated overhead in 10k sims.
- 接口简洁：get_K(arf, rmf, background, model, params, band, ...)->float。
	Simple interface: get_K(arf, rmf, background, model, params, band, ...)->float.
- 缓存键严格基于配置唯一性（路径字符串、模型与参数、能段、环境变量）。
	Cache key strictly reflects uniqueness of configuration (paths/model/params/band/env).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import hashlib
import numpy as np

from astropy.io import fits
from pathlib import Path

# 统一的数值类型别名（兼容 numpy 数值类型）
# Unified numeric alias (compatible with numpy scalar types)
Numeric = Union[float, int, np.floating, np.integer]

# 尝试导入 pyxspec；若失败，延迟到会话初始化时报错
# Try importing pyxspec; if unavailable, raise when session is created
_HAVE_XSPEC = False
try:  # pragma: no cover
	import xspec  # type: ignore
	from xspec import AllData, AllModels, FakeitSettings  # type: ignore
	_HAVE_XSPEC = True
except Exception:  # pragma: no cover
	_HAVE_XSPEC = False


@dataclass(frozen=True)
class KConfig:
	"""K 计算所需的不可变配置对象。
	Immutable config for K computation.

	字段 | Fields
	- arf/rmf/background: 路径字符串 | file paths
	- model/params: XSPEC 模型与参数 | XSPEC model and parameter values
	- band: (emin, emax) [keV] 能段 | energy band in keV
	- exposure/back_exposure: 源/背景曝光秒数 | source/background exposure (s)
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
	xspec_abund: str
	xspec_xsect: str
	xspec_cosmo: str

	def key(self) -> str:
		"""将配置编码为缓存键（SHA256）。
		Encode configuration into a cache key (SHA256).

		说明 | Notes
		- 数值保留有限精度，避免浮点噪声导致缓存抖动；
		  Use limited precision for floats to avoid cache churn from tiny diffs.
		"""
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
	"""一个轻量的 XSPEC 会话，提供 K 计算能力。
	A lightweight XSPEC session that can compute K.

	使用原则 | Usage
	- 每次计算前清空 AllData/AllModels，避免状态污染；
	  Clear AllData/AllModels to avoid state leakage between runs.
	- 关闭 XSPEC 交互提示，确保无人值守运行；
	  Disable prompting for non-interactive execution.
	"""

	def __init__(self) -> None:
		if not _HAVE_XSPEC:
			raise RuntimeError("pyxspec is not available; ensure HEASoft/XSPEC is installed and on PYTHONPATH")

	def compute_K(self, cfg: KConfig) -> float:
		# 环境设置 | Set XSPEC globals
		xspec.Xset.abund = cfg.xspec_abund  # type: ignore
		xspec.Xset.xsect = cfg.xspec_xsect  # type: ignore
		xspec.Xset.cosmo = cfg.xspec_cosmo  # type: ignore
		xspec.Xset.allowPrompting = False  # type: ignore

		# 清空数据/模型 | Reset data/models
		AllData.clear()
		AllModels.clear()

		# 构建模型并设置参数 | Build model and set parameters
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

		# 调用 fakeit 生成折叠谱 | Run fakeit to create folded spectrum
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

		# 选择能段并计算带内速率与通量 | Select band and compute rate/flux
		emin, emax = cfg.band
		AllData.notice("all")
		AllData.ignore(f"**-{float(emin)} {float(emax)}-**")
		AllData.ignore("bad")

		spec = AllData(1)
		spec_any = spec  # type: ignore[assignment]
		# spec.rate[3] 通常是模型预测总率（不含噪声）
		rate = float(spec_any.rate[3])  # type: ignore[attr-defined]
		xspec.AllModels.calcFlux(f"{float(emin)} {float(emax)}")
		# spec.flux[0] 通常是 calcFlux 的能通量结果
		flux = float(spec_any.flux[0])  # type: ignore[attr-defined]

		if flux <= 0:
			raise RuntimeError("XSPEC returned non-positive flux; check model and band")
		return float(rate / flux)


class XspecKFactory:
	"""
	K 计算工厂 + 缓存。

	- get_K(...): 返回 K，如缓存未命中则调用 XSPEC 计算并存入缓存。
	  Return K; compute via XSPEC if cache miss, then store.
	- clear(): 清空缓存。
	  Clear the cache.
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
		arf: str | Path,
		rmf: str | Path,
		background: Optional[str | Path],
		model: str,
		params: Tuple[float, ...],
		band: Tuple[float, float],
		exposure: Numeric,
		back_exposure: Optional[Numeric],
		xspec_abund: str,
		xspec_xsect: str,
		xspec_cosmo: str,
	) -> float:
		arf = str(arf)
		rmf = str(rmf)
		background_str: Optional[str] = None if background is None else str(background)
		cfg = KConfig(
			arf=arf,
			rmf=rmf,
			background=background_str,
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

def prepare_background_for_fakeit(
	src_pha: Union[str, Path],
	bkg_pha: Union[str, Path],
	out_path: Optional[Union[str, Path]] = None,
	*,
	sci_notation: bool = True,
) -> Path:
	"""基于源/背景 PHA 的 BACKSCAL 比值，生成供 fakeit 使用的背景文件。
	Create a background PHA for fakeit by setting BACKSCAL := BACKSCAL_src / BACKSCAL_bkg.

	行为 | Behavior
	- 读取源文件与背景文件的 BACKSCAL；若为列（Type II），取第一行或广播更新。
	- 在背景文件的拷贝上，仅更新 BACKSCAL，其余关键字/数据保持不变（附加 HISTORY）。
	- 输出文件名默认在原背景文件名后追加 "_for_fakeit"，保留原始扩展名（.pha/.pi/.fits/.fit 等），并与原文件位于同一目录。

	参数 | Params
	- src_pha: 源 PHA 路径（Path 或 str）。
	- bkg_pha: 背景 PHA 路径（Path 或 str）。
	- out_path: 可选输出路径（Path 或 str）；不提供时自动拼接后缀。

	返回 | Returns
	- 新背景文件的路径（Path）。
	"""

	# 数值到字符串的格式函数（控制注释/HISTORY 中的显示，不影响数值存储）
	fmt = (lambda x: f"{x:.8E}") if sci_notation else (lambda x: f"{x:.6g}")

	def _find_spectrum_hdu(hdul: fits.HDUList) -> int:
		# 优先找 EXTNAME == 'SPECTRUM' 的表；否则返回第一个 BinTableHDU 索引
		for i, hdu in enumerate(hdul):
			extname = str(hdu.header.get('EXTNAME', '')).upper()
			if isinstance(hdu, fits.BinTableHDU) and extname == 'SPECTRUM':
				return i
		for i, hdu in enumerate(hdul):
			if isinstance(hdu, fits.BinTableHDU):
				return i
		raise ValueError('No SPECTRUM (BinTableHDU) found in PHA file')

	def _read_backscal(hdul: fits.HDUList) -> Tuple[int, float, bool]:
		idx = _find_spectrum_hdu(hdul)
		hdu_for_hdr = hdul[idx]
		hdr = hdu_for_hdr.header  # type: ignore[attr-defined]
		# 1) 标量关键字
		if 'BACKSCAL' in hdr:
			try:
				val = hdr['BACKSCAL']
				return idx, val, False  # is_column=False
			except Exception:
				pass
		# 2) 列（Type II 或逐道），取第一行值
		hdu = hdul[idx]
		col_names_raw = list(hdu.columns.names) if isinstance(hdu, fits.BinTableHDU) and hdu.columns is not None and hdu.columns.names is not None else []
		col_names = [str(n).strip().upper() for n in col_names_raw]
		if isinstance(hdu, fits.BinTableHDU) and 'BACKSCAL' in col_names:
			arr = hdu.data['BACKSCAL']
			val = float(np.asarray(arr).flat[0])
			return idx, val, True   # is_column=True
		# 3) 缺省 1.0
		return idx, 1.0, False

	src_path = Path(src_pha)
	bkg_path = Path(bkg_pha)

	with fits.open(src_path) as src_hdul:
		_, src_bs, _ = _read_backscal(src_hdul)

	with fits.open(bkg_path) as bkg_hdul:
		idx, bkg_bs, is_col = _read_backscal(bkg_hdul)
		if bkg_bs == 0.0 or src_bs == 0.0:
			raise ValueError('BACKGROUND or Source region BACKSCAL is zero; cannot compute ratio')
		new_bs = float(bkg_bs / src_bs)

		# 生成输出路径（同目录，保留扩展名）；若 out_path 指向目录则写入其下
		if out_path is None:
			out_path_path = bkg_path.with_name(bkg_path.stem + '_for_fakeit' + bkg_path.suffix)
		else:
			p = Path(out_path)
			if p.exists() and p.is_dir():
				out_path_path = p / (bkg_path.stem + '_for_fakeit' + bkg_path.suffix)
			else:
				out_path_path = p

		# 先将原背景文件原样写到目标路径
		bkg_hdul.writeto(out_path_path, overwrite=True)

		# 以更新模式打开目标文件并原地修改 BACKSCAL，避免复制可变长列
		with fits.open(out_path_path, mode='update') as out_hdul:
			# 在写入后的文件上重新定位 SPECTRUM HDU
			out_idx = _find_spectrum_hdu(out_hdul)
			hdu_copy = out_hdul[out_idx]
			col_names2 = list(hdu_copy.columns.names) if isinstance(hdu_copy, fits.BinTableHDU) and hdu_copy.columns is not None and hdu_copy.columns.names is not None else []
			if isinstance(hdu_copy, fits.BinTableHDU) and is_col and 'BACKSCAL' in col_names2 and hdu_copy.data is not None:
				arr = np.asarray(hdu_copy.data['BACKSCAL'])
				# 根据列的物理格式选择精度（E->float32, D->float64），保持与原列一致
				try:
					icol0 = [i for i, n in enumerate(col_names2) if str(n).strip().upper() == 'BACKSCAL'][0]
					fmt_str = str(hdu_copy.columns[icol0].format).strip().upper() if hdu_copy.columns is not None else ''
					if fmt_str.startswith('E'):
						dtype_target = np.float32
					elif fmt_str.startswith('D'):
						dtype_target = np.float64
					else:
						dtype_target = arr.dtype
				except Exception:
					dtype_target = arr.dtype
				# 原地覆盖列数据（数值存储为二进制浮点，显示由 TDISP 控制）；保持形状不变
				hdu_copy.data['BACKSCAL'][:] = np.full(arr.shape, new_bs, dtype=dtype_target)
				# 额外：同步写 header BACKSCAL，便于工具显示
				hdu_copy.header['BACKSCAL'] = (float(new_bs), f"adjusted for fakeit: src/back BACKSCAL = {fmt(new_bs)}")  # type: ignore[attr-defined]
				# 设定列显示格式，提升 FV 可读性
				try:
					icol = [i for i, n in enumerate(col_names2) if str(n).strip().upper() == 'BACKSCAL'][0] + 1
					disp = 'E12.5' if sci_notation else 'F12.6'
					hdu_copy.header[f'TDISP{icol}'] = (disp, 'display format for BACKSCAL')  # type: ignore[attr-defined]
				except Exception:
					pass
			else:
				# 标量关键字 | scalar keyword
				hdu_copy.header['BACKSCAL'] = (new_bs, f"adjusted for fakeit: src/back BACKSCAL = {fmt(new_bs)}")  # type: ignore[attr-defined]
			# 附加 HISTORY | add history
			hdu_copy.header.add_history(
				f"BACKSCAL adjusted for fakeit: new={fmt(new_bs)} (src={fmt(src_bs)} / bkg={fmt(bkg_bs)})"
			)  # type: ignore[attr-defined]
			# flush 在上下文退出时自动完成

	return out_path_path

__all__ = [
	'XspecKFactory',
    'prepare_background_for_fakeit',
]

