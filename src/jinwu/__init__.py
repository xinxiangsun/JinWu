"""
JinWu
=======

Joint Inference for high energy transient light‑curve & spectral analysis With Unifying physical modeling

Top-level package re-exports commonly used subpackages for convenience:
- core: OGIP FITS IO, unified readers, and utilities
- lightcurve: SNR evaluation and trigger decision helpers
- background: Priors/posteriors for background modeling
- spectrum, missions, response, timing, data: domain subpackages
"""

from __future__ import annotations 

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from pathlib import Path
import tomllib


# 版本获取顺序：
# 1) setuptools_scm 写入的版本文件（build 时写入 src/jinwu/_version.py）
# 2) 已安装分发的元数据（pip 安装后）
# 3) 源码树回退：解析 pyproject.toml


def _read_version_from_pyproject() -> str:
	"""在源码环境中解析 pyproject.toml 的 version 字段；失败时返回占位符。"""
	root = Path(__file__).resolve().parents[2]
	project_file = root / 'pyproject.toml'
	if not project_file.exists():
		return '0.0.0+unknown'
	with project_file.open('rb') as fh:
		data = tomllib.load(fh)
	return data.get('project', {}).get('version', '0.0.0+unknown')


# 1) setuptools_scm 写入的版本文件（优先）
try:
	from ._version import version as __version__  # type: ignore
except Exception:
	# 2) 已安装分发的元数据
	try:
		__version__ = _pkg_version('jinwu')
	except PackageNotFoundError:
		# 3) 源码树回退：解析 pyproject.toml
		__version__ = _read_version_from_pyproject()

__author__ = "Xinxiang Sun"
__email__ = "sunxx@nao.cas.cn"
__description__ = "JinWu: Joint Inference for high energy transient light‑curve & spectral analysis With Unifying physical modeling"

# Re-export subpackages for ergonomic imports
from . import core
from . import lightcurve
from . import background
from . import spectrum
from . import response
from . import timing
from . import data
from . import fermi
from . import ep
from . import ftools
from .core import netdata, readfits
__all__ = [
	# Subpackages
	'core', 'lightcurve', 'background', 'spectrum', 'fermi', 'ep', 'response', 'timing', 'data','ftools',
	
	# Core datasets & helpers from jinwu.core
	'netdata', 'readfits',
	
	# Package meta
	'__version__',
]
