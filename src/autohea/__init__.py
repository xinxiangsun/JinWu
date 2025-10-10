'''
Date: 2025-02-25 15:02:36
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-10-10 11:56:05
FilePath: /research/autohea/src/autohea/__init__.py
'''
"""
AutoHEA: A Python package for automatic high-energy astrophysics analysis.
"""

__version__ = "0.0.2"
__author__ = "Xinxiang Sun"
__email__ = "sunxinxiang24@mails.ucas.ac.cn"
__description__ = "AutoHEA: A Python package for automatic high-energy astrophysics analysis."

# 明确导入子模块
from . import core
from . import spectrum
from . import missions
from . import response
from . import timing
# 定义公开接口
__all__ = ['core', 'spectrum', 'missions', 'response', 'timing']
