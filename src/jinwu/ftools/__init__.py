"""轻量级 ftools Python 复刻集合（jinwu 专属）

该包提供几个常用的 HEASOFT/ftools 等价函数的纯 Python 实现，目的是把
`external_sources` 中的工具逻辑以可调用的 Python 接口放到 jinwu 内部，便于
在不依赖 Fortran/C 二进制的情况下使用基本功能（提取、分组、重分箱）。

目前包含模块：`fextract`、`ftgrouppha`、`ftrbnpha`。
"""

from . import fextract
from . import ftgrouppha
from . import ftrbnpha
from . import ftrbnrmf

__all__ = [
    'fextract', 'ftgrouppha', 'ftrbnpha', 'ftrbnrmf'
]
