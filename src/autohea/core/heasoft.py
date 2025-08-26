'''
Date: 2025-03-17 13:40:01
LastEditors: Xinxiang Sun sunxinxiang24@mails.ucas.ac.cn
LastEditTime: 2025-07-24 15:37:33
FilePath: /research/autohea/src/autohea/core/heasoft.py
'''
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
                                 如果未提供，则自动查找HEASoft安装。
        """
        if headas_init_path is not None:
            self.headas_init_path = headas_init_path
        else:
            # 首先尝试从配置文件读取
            self.headas_init_path = self._get_headas_from_zshrc()
            if self.headas_init_path is None:
                # 如果配置文件中没有，则自动查找
                self.headas_init_path = self._find_headas_automatically()
            if self.headas_init_path is None:
                # 最后的fallback
                self.headas_init_path = "/home/xinxiang/miniconda3/envs/hea/heasoft/headas-init.sh"
        
        self.original_env = os.environ.copy()  # 保存原始环境变量

    @staticmethod
    def _get_headas_from_zshrc(zshrc_path=os.path.expanduser("~/.zshrc")):
        """
        从~/.zshrc或~/.bashrc中读取HEADAS变量，并拼接 headas-init.sh
        """
        # 尝试多个配置文件
        config_files = [
            zshrc_path,
            os.path.expanduser("~/.bashrc"),
            os.path.expanduser("~/.bash_profile")
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    for line in f:
                        m = re.match(r'(?:export\s+)?HEADAS=(.*)', line.strip())
                        if m:
                            headas_dir = m.group(1).strip().strip('"').strip("'")
                            return os.path.join(headas_dir, "headas-init.sh")
        
        # 如果配置文件中没有找到，尝试从当前环境变量
        if "HEADAS" in os.environ:
            return os.path.join(os.environ["HEADAS"], "headas-init.sh")
            
        return None

    @staticmethod
    def _find_headas_automatically():
        """
        自动查找HEASoft安装路径
        """
        # 常见的HEASoft安装位置
        common_paths = [
            "/home/xinxiang/miniconda3/envs/hea/heasoft/headas-init.sh",
            "/opt/heasoft/headas-init.sh",
            "/usr/local/heasoft/headas-init.sh",
            "/Users/xinxiang/heasoft-6.35.1/aarch64-apple-darwin24.5.0/headas-init.sh"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
                
        # 尝试在conda环境中查找
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_headas = os.path.join(conda_prefix, "heasoft", "headas-init.sh")
            if os.path.exists(conda_headas):
                return conda_headas
        
        return None

    def init_heasoft(self):
        """
        初始化 HEASoft 环境变量并更新当前进程的环境。
        """
        # 检查初始化脚本是否存在
        if self.headas_init_path is None or not os.path.isfile(self.headas_init_path):
            raise FileNotFoundError(f"HEASoft 初始化脚本未找到: {self.headas_init_path}")

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
    def init_heasoft_in_notebook():
        """在 Jupyter 中捕获 HEASoft 环境变量"""
        # 首先尝试conda环境中的HEASoft
        conda_headas = "/home/xinxiang/miniconda3/envs/hea/heasoft/headas-init.sh"
        if os.path.isfile(conda_headas):
            headas_init = conda_headas
        else:
            # fallback到原始macOS路径
            headas_init = "/Users/xinxiang/heasoft-6.35.1/aarch64-apple-darwin24.5.0/headas-init.sh"
        
        # 1. 检查初始化脚本是否存在
        if not os.path.isfile(headas_init):
            raise FileNotFoundError(f"HEASoft 初始化脚本未找到: {headas_init}")

        # 2. 通过子进程捕获环境变量
        try:
            output = subprocess.check_output(
                f"source '{headas_init}' && env -0",
                shell=True,
                executable="/bin/bash",
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"初始化失败: {e.stderr}") from e

        # 3. 更新当前进程环境
        for line in output.strip().split('\0'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

        # 4. 添加 Python 库路径
        heasoft_pylib = os.path.join(os.environ["HEADAS"], "lib", "python")
        if os.path.isdir(heasoft_pylib):
            sys.path.insert(0, heasoft_pylib)
