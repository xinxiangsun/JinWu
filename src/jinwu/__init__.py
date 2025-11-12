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

# 版本获取顺序：
# 1) 尝试使用由 setuptools_scm 生成的版本文件（build 时写入 src/jinwu/_version.py）
# 2) 已安装分发的元数据（pip 安装后）
# 3) 源码树回退：解析 pyproject.toml 中的 version（若使用 setuptools_scm，则一般只在未安装、未生成 _version.py 时触发）

def _read_version_from_pyproject() -> str:
	"""在源码环境中解析 pyproject.toml 的 version 字段；失败时返回占位符。"""
	try:
		import pathlib, tomllib  # Python 3.11+ 标准库
		# __file__ 位于 src/jinwu/__init__.py，向上三级到项目根目录
		root = pathlib.Path(__file__).resolve().parents[2]
		pyproject = root / 'pyproject.toml'
		if not pyproject.exists():
			return '0.0.0+unknown'
		data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
		# 支持两种情况：静态 version 或使用 dynamic version
		return data.get('project', {}).get('version', '0.0.0+unknown')
	except Exception:
		return '0.0.0+unknown'

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
from . import missions
from . import response
from . import timing
from . import data
from .core.time import Time
__all__ = [
	# Subpackages
	'core', 'lightcurve', 'background', 'spectrum', 'missions', 'response', 'timing', 'data',
	'Time',
	# Package meta
	'__version__',
]
