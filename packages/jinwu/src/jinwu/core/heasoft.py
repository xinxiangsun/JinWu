"""
Date: 2025-03-17 13:40:01
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-10-10 11:56:01
FilePath: /research/autohea/src/autohea/core/heasoft.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path


class HeasoftEnvManager:
    """初始化当前 Python 进程中的 HEASoft 环境。

    headas_init_path 可传入 HEADAS 目录或初始化脚本。缺省时复用已激活
    的环境，否则从 HEADAS 或 shell 配置文件中的静态路径定位脚本。
    """

    RC_FILES = (
        ".bashrc", ".bash_profile", ".profile",
        ".zshrc", ".zprofile", ".cshrc", ".tcshrc",
    )

    def __init__(self, headas_init_path=None):
        self._explicit = headas_init_path is not None
        self.headas_init_path = self._find_init_script(headas_init_path)
        self.original_env = os.environ.copy()
        self.original_sys_path = sys.path.copy()
        self._initialized = False

    @staticmethod
    def _as_init_path(value):
        """将 HEADAS 目录或初始化脚本路径规范化。"""
        path = Path(value).expanduser()
        if not path.exists():
            expanded = os.path.expandvars(str(value).strip().strip("'\""))
            if not expanded:
                return None
            path = Path(expanded).expanduser()
        if path.is_dir():
            path /= "headas-init.sh"
        # Preserve the installation directory when the script is a symlink.
        return path.absolute() if path.is_file() else None

    @classmethod
    def _find_init_script(cls, explicit=None):
        """依次从显式路径、环境变量及 Shell 配置文件查找。"""
        if explicit is not None:
            path = cls._as_init_path(explicit)
            if path is None:
                raise FileNotFoundError(f"无效的 HEASoft 初始化路径: {explicit}")
            return path

        if headas := os.environ.get("HEADAS"):
            if path := cls._as_init_path(headas):
                return path

        patterns = (
            r"^\s*(?:export\s+)?HEADAS\s*=\s*(.+?)(?:\s+#.*)?$",
            r"^\s*setenv\s+HEADAS\s+(.+?)(?:\s+#.*)?$",
        )

        for name in cls.RC_FILES:
            rc = Path.home() / name
            try:
                lines = rc.read_text(errors="replace").splitlines()
            except OSError:
                continue

            for line in lines:
                if line.lstrip().startswith("#"):
                    continue

                for pattern in patterns:
                    if match := re.match(pattern, line):
                        if path := cls._as_init_path(match.group(1)):
                            return path

        return None

    @staticmethod
    def _is_active(env):
        headas = env.get("HEADAS")
        if not headas or not env.get("PFILES"):
            return False
        bindir = Path(headas, "bin").resolve()
        return any(
            Path(path).resolve() == bindir
            for path in env.get("PATH", "").split(os.pathsep) if path
        )

    def init_heasoft(self):
        """初始化 HEASoft，并同步环境到当前 Python 进程。"""
        if self._initialized and self._is_active(os.environ):
            return
        if not self._explicit and self._is_active(os.environ):
            self._add_python_path()
            self._initialized = True
            return
        if self.headas_init_path is None:
            raise FileNotFoundError("未找到 headas-init.sh，请显式指定路径")

        # Some HEASoft installs put the real script in BUILD_DIR.
        headas = self.headas_init_path.parent
        if headas.name == "BUILD_DIR":
            headas = headas.parent
        result = subprocess.run(
            [
                "bash", "-c", 'source "$1" >&2 && "$2" -I -S -c "$3"',
                "bash", str(self.headas_init_path), sys.executable,
                "import json, os; print(json.dumps(dict(os.environ)))",
            ],
            env={**os.environ, "HEADAS": str(headas)},
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"HEASoft 初始化失败: {result.stderr.strip()}")
        new_env = json.loads(result.stdout)
        # Official headas-init.sh may print an error while returning zero.
        if not self._is_active(new_env):
            raise RuntimeError(
                f"HEASoft 未设置有效的 PATH 和 PFILES: {result.stderr.strip()}"
            )
        os.environ.clear()
        os.environ.update(new_env)
        self._add_python_path()
        self._initialized = True

    @staticmethod
    def _add_python_path():
        pylib = Path(os.environ["HEADAS"]) / "lib" / "python"
        if pylib.is_dir() and str(pylib) not in sys.path:
            sys.path.insert(0, str(pylib))

    def reset_environment(self):
        """恢复初始化前的环境变量及 Python 搜索路径。"""
        os.environ.clear()
        os.environ.update(self.original_env)
        sys.path[:] = self.original_sys_path
        self._initialized = False

    def is_heasoft_initialized(self):
        """检查当前进程能否使用已配置的 HEASoft。"""
        return self._is_active(os.environ)

    @classmethod
    def init_heasoft_in_notebook(cls, headas_init_path=None):
        """在 Jupyter 中初始化 HEASoft。"""
        manager = cls(headas_init_path)
        manager.init_heasoft()
        return manager
