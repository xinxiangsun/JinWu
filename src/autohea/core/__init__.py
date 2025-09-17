'''
Date: 2025-04-26 14:17:45
<<<<<<< HEAD
LastEditors: Xinxiang Sun sunxinxiang24@mails.ucas.ac.cn
LastEditTime: 2025-07-16 14:04:05
=======
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-13 14:09:28
>>>>>>> 3a5dab31b2d0a69623d0f1c4fa85cbf24986acbd
FilePath: /research/autohea/src/autohea/core/__init__.py
'''

from . import file
from . import heasoft
from . import rsp
from . import arf
from . import rmf
from . import plot
from . import time

<<<<<<< HEAD
from .heasoft import HeasoftEnvManager
heasoft_env_manager = HeasoftEnvManager()
heasoft_env_manager.init_heasoft()
=======
# from .heasoft import HeasoftEnvManager
# heasoft_env_manager = HeasoftEnvManager()
# heasoft_env_manager.init_heasoft()
>>>>>>> 3a5dab31b2d0a69623d0f1c4fa85cbf24986acbd

__all__ = ['file', 'heasoft', 'rsp', 'arf', 'rmf', 'plot', 'time']