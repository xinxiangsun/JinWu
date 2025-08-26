'''
Date: 2025-04-26 14:17:45
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-13 14:09:28
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

__all__ = ['file', 'heasoft', 'rsp', 'arf', 'rmf', 'plot', 'time']