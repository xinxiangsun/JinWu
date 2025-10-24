"""
Date: 2025-03-17 13:40:01
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-10-10 11:56:01
FilePath: /research/autohea/src/autohea/core/heasoft.py
"""
import os
import sys
import subprocess
import re

class HeasoftEnvManager:
    """
    管理 HEASoft 环境的类，支持初始化和重置环境变量。
    """
    def __init__(self, headas_init_path=None):
        """
        初始化类实例。

        :param headas_init_path: HEASoft 初始化脚本的路径（可选）。
                                 如果未提供，逻辑如下：
                                 1) 若当前进程已包含 HEADAS 环境变量，则优先使用该环境（无需再次 source），并尝试推断 headas-init.sh 路径；
                                 2) 否则，尝试从 ~/.bashrc、~/.bash_profile、~/.profile、~/.zshrc 中解析 HEADAS 并定位 headas-init.sh；
                                 3) 若仍未找到，则保持为空，待 init_heasoft 时给出明确报错。
        """
        if headas_init_path is not None:
            self.headas_init_path = headas_init_path
        else:
            # 先从当前环境读取 HEADAS
            self.headas_init_path = self._get_headas_from_env()
            if self.headas_init_path is None:
                # 再从各类 rc 文件解析
                self.headas_init_path = self._get_headas_from_rc_files()
        self.original_env = os.environ.copy()  # 保存原始环境变量

    @staticmethod
    def _expand_path_like(value: str) -> str:
        """展开类似 shell 的路径字符串（处理 ~、$HOME 等）。"""
        return os.path.expanduser(os.path.expandvars(value.strip().strip('"').strip("'")))

    @classmethod
    def _get_headas_from_env(cls):
        """
        从当前进程的环境变量读取 HEADAS，并返回 headas-init.sh 路径（若存在）。
        """
        headas_dir = os.environ.get("HEADAS")
        if not headas_dir:
            return None
        headas_dir = cls._expand_path_like(headas_dir)
        # 常见布局：$HEADAS/headas-init.sh
        candidate = os.path.join(headas_dir, "headas-init.sh")
        return candidate if os.path.isfile(candidate) else None

    @classmethod
    def _parse_headas_from_file(cls, rc_path: str):
        """在给定 rc 文件中解析 HEADAS 变量，返回 headas-init.sh 的路径。"""
        if not rc_path or not os.path.exists(rc_path):
            return None
        try:
            with open(rc_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    m = re.match(r'(?:export\s+)?HEADAS=(.*)', line)
                    if m:
                        headas_dir = cls._expand_path_like(m.group(1))
                        # 忽略自引用等异常情况
                        if not headas_dir or headas_dir.upper() == "HEADAS":
                            continue
                        candidate = os.path.join(headas_dir, "headas-init.sh")
                        if os.path.isfile(candidate):
                            return candidate
        except Exception:
            return None
        return None

    @classmethod
    def _get_headas_from_rc_files(cls):
        """依次尝试在常见 rc 文件中解析 HEADAS，返回 headas-init.sh 路径。"""
        home = os.path.expanduser('~')
        rc_candidates = [
            os.path.join(home, ".bashrc"),
            os.path.join(home, ".bash_profile"),
            os.path.join(home, ".profile"),
            os.path.join(home, ".zshrc"),
        ]
        for rc in rc_candidates:
            path = cls._parse_headas_from_file(rc)
            if path:
                return path
        return None

    def init_heasoft(self):
        """
        初始化 HEASoft 环境变量并更新当前进程的环境。
        """
        # 若当前进程已存在 HEADAS，则视为已初始化，仅补充 Python 路径后返回
        if "HEADAS" in os.environ and os.environ.get("HEADAS"):
            heasoft_pylib = os.path.join(os.environ["HEADAS"], "lib", "python")
            if os.path.isdir(heasoft_pylib) and heasoft_pylib not in sys.path:
                sys.path.insert(0, heasoft_pylib)
            return

        # 检查初始化脚本是否存在
        if not self.headas_init_path or not os.path.isfile(self.headas_init_path):
            raise FileNotFoundError(
                "未能定位 HEASoft 初始化脚本。请先在 shell 中 source heainit/headas-init.sh，"
                "或将 headas_init_path 显式传入 HeasoftEnvManager。"
            )

        try:
            # 捕获 HEASoft 环境变量
            output = subprocess.check_output(
                f"source '{self.headas_init_path}' && env -0",
                shell=True,
                executable="/bin/bash",
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"HEASoft 初始化失败: {e.stderr}") from e

        # 更新当前进程环境变量
        for line in output.strip().split('\0'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

        # 添加 HEASoft 的 Python 库路径
        heasoft_pylib = os.path.join(os.environ["HEADAS"], "lib", "python")
        if os.path.isdir(heasoft_pylib):
            if heasoft_pylib not in sys.path:
                sys.path.insert(0, heasoft_pylib)

    def reset_environment(self):
        """
        重置环境变量为初始化 HEASoft 之前的状态。
        """
        os.environ.clear()
        os.environ.update(self.original_env)

    def is_heasoft_initialized(self):
        """
        检查 HEASoft 是否已初始化。

        :return: 如果 HEADAS 环境变量已设置，则返回 True，否则返回 False。
        """
        return "HEADAS" in os.environ

    @staticmethod
    def init_heasoft_in_notebook(headas_init_path: str | None = None):
        """
        在 Jupyter 中捕获 HEASoft 环境变量。

        优先使用当前进程的 HEADAS 环境；如果不存在，则：
        - 若提供 headas_init_path，则使用该脚本；
        - 否则尝试从 ~/.bashrc、~/.zshrc 等 rc 文件解析定位 headas-init.sh。
        """
        # 若已初始化，补充 Python 路径后返回
        if "HEADAS" in os.environ and os.environ.get("HEADAS"):
            heasoft_pylib = os.path.join(os.environ["HEADAS"], "lib", "python")
            if os.path.isdir(heasoft_pylib) and heasoft_pylib not in sys.path:
                sys.path.insert(0, heasoft_pylib)
            return

        # 构造一个临时管理器以复用查找逻辑
        mgr = HeasoftEnvManager(headas_init_path=headas_init_path)
        if not mgr.headas_init_path or not os.path.isfile(mgr.headas_init_path):
            raise FileNotFoundError(
                "未能定位 HEASoft 初始化脚本。请在 shell 中先执行 `source heainit` 或 `source <headas-init.sh>`，"
                "或在调用时传入 headas_init_path。"
            )

        # 通过子进程捕获环境变量
        try:
            output = subprocess.check_output(
                f"source '{mgr.headas_init_path}' && env -0",
                shell=True,
                executable="/bin/bash",
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"初始化失败: {e.stderr}") from e

        # 更新当前进程环境
        for line in output.strip().split('\0'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

        # 添加 Python 库路径
        heasoft_pylib = os.path.join(os.environ["HEADAS"], "lib", "python")
        if os.path.isdir(heasoft_pylib) and heasoft_pylib not in sys.path:
            sys.path.insert(0, heasoft_pylib)
