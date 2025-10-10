'''
Date: 2025-04-26 14:17:45
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-10-10 11:56:04
FilePath: /research/autohea/src/autohea/core/__init__.py
'''

from . import file
from . import heasoft
from . import rsp
from . import arf
from . import rmf
from . import plot
from . import time

# from .heasoft import HeasoftEnvManager
# heasoft_env_manager = HeasoftEnvManager()
# heasoft_env_manager.init_heasoft()
from .heasoft import HeasoftEnvManager
heasoft_env_manager = HeasoftEnvManager()
heasoft_env_manager.init_heasoft()

__all__ = ['file', 'heasoft', 'rsp', 'arf', 'rmf', 'plot', 'time']