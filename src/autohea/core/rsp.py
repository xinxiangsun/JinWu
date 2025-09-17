'''
Date: 2025-04-24 22:37:42
LastEditors: Xinxiang Sun sunxinxiang24@mails.ucas.ac.cn
LastEditTime: 2025-05-01 22:55:05
FilePath: /research/autohea/autohea/core/rsp.py
'''
import numpy as np
import astropy.io.fits as fits
from pathlib import Path
from .file import FitsRecToNdarray, RmfReader, ArfReader
from gdt.core.response import Rsp


class Response(ArfReader, RmfReader, Rsp):
    """
    Class to handle the response of the GBM instrument.
    """
    def __init__(self, arf_file: str | Path = None, rmf_file: str | Path = None, rsp_extension: str = "SPECRESP MATRIX"):
        """
        Initialize the SpectrumReader with optional ARF and RMF file paths.
        使用可选的 ARF 和 RMF 文件路径初始化 SpectrumReader。

        Args:
            arf_file (str | Path): Path to the ARF file. (ARF 文件路径，可选)
            rmf_file (str | Path): Path to the RMF file. (RMF 文件路径，可选)
        """
        ArfReader.__init__(self, arf_filepath=arf_file)
        RmfReader.__init__(self, rmf_filepath=rmf_file)
        self.rsp_extension = rsp_extension

    @staticmethod
    def get_rsp(rmfdata, arfdata) -> np.ndarray:
        """
        Get the response matrix from the ARF and RMF files.

        Args:
            arfdata (ndarray/list): The ARF file data.
            rmfdata (ndarray): The RMF file data.

        Returns:
            np.ndarray: The response matrix.
        """
        # 检查 ARF 数据类型
        if isinstance(arfdata, (list, np.ndarray)):
            arfdata = np.array(arfdata)
        else:
            raise TypeError("辅助响应文件应该是list或者numpy数组, arf data should be a list or numpy array")

        # 检查 RMF 数据类型
        if not isinstance(rmfdata, (list,np.ndarray)):
            raise TypeError("响应文件应当是list或者numpy数组, rmf data should be a numpy array")

        # 检查行列是否相等
        if rmfdata.shape[0] == rmfdata.shape[1]:
            print("警告: 响应矩阵的行数和列数相等, 这很多时候不太对, 请确认是否符合预期 (Warning: RMF matrix is square, please verify if this is expected).")

        # 检查 ARF 数据长度是否与 RMF 行数匹配
        if len(arfdata) != rmfdata.shape[0]:
            raise ValueError("ARF 数据的长度与 RMF 数据的行数不匹配, ARF data length must match RMF data rows")

        '''
        #下面的代码已经放弃使用了,因为使用了numpy的广播机制进行运算,效果是相同的
        #通过广播机制，arfdata[:, np.newaxis] 将 arfdata 从一维数组扩展为二维数组，形状变为 (4, 1)，然后与 rmfdata 逐元素相乘
        rspdata = np.zeros_like(rmfdata)
        for i in range(len(arfdata)):
            for j in range(len(arfdata)):
                rspdata[i][j] = arfdata[i]*rmfdata[i][j]
'''

        # 生成响应矩阵
        rspdata = rmfdata * arfdata[:, np.newaxis]  # 使用广播机制计算响应矩阵
        return rspdata
    
    @property
    def convolution(self):
        """
        Get the convolution matrix from the ARF and RMF files.
        获取 ARF 和 RMF 文件的卷积矩阵。

        Returns:
            np.ndarray: The convolution matrix. (卷积矩阵)
        """
        if self.rmfdata is None or self.arfdata is None:
            raise ValueError("RMF和ARF文件未加载, RMF and ARF files are not loaded")

        return self.get_rsp(self.rmfmatrix, self.arf)
    

    @staticmethod
    def rmf_weighted_arf(rmfdata, arfdata) -> np.ndarray:
        """
        通过能量响应将不同能量的有效面积进行加权平均得到不同能量通道对应的有效面积
        参数:
            rmfdata (ndarray): RMF文件数据
            arfdata (ndarray): ARF文件数据
        返回:
            np.ndarray: 加权平均后的有效面积

        Get the auxiliary response matrix from the response and RMF files.

        Args:
            rmfdata (ndarray/list): The response file data.
            arfdata (ndarray): The RMF file data.

        Returns:
            np.ndarray: The auxiliary response matrix.
        """
        rspdata = Rsp.get_rsp(rmfdata, arfdata)
        return rspdata.sum(axis=0) 
    
    @staticmethod
    def fits_to_rsp(rmffits: Path, arffits: Path, 
                    rmf_extension: str = "MATRIX", 
                    arf_extension: str = "SPECRESP", 
                    rmf_field: str = "MATRIX",
                    arf_field: str = "SPECRESP") -> np.ndarray:
        """
        该方法不建议使用,建议使用这个class中的convolution方法.
        而且这个方法的extension和field参数是根据EP的情况写的,应该不通用
        将RMF和ARF文件转换为响应矩阵
        参数:
            rmffits (Path): RMF文件路径
            arffits (Path): ARF文件路径
            rmf_extension (str): RMF文件中包含响应矩阵的扩展名
            arf_extension (str): ARF文件中包含有效面积的扩展名
            arf_field (str): ARF文件中需要提取的字段名
        返回:
            np.ndarray: 响应矩阵

        Convert the RMF and ARF files to the response matrix.

        Args:
            rmffits (Path): The path of the RMF file.
            arffits (Path): The path of the ARF file.
            rmf_extension (str): The extension name for the RMF file.
            arf_extension (str): The extension name for the ARF file.
            arf_field (str): The field name to extract from the ARF file.

        Returns:
            np.ndarray: The response matrix.
        """
        if not rmffits.exists():
            raise FileNotFoundError(f"RMF文件不存在: {rmffits}")
        if not arffits.exists():
            raise FileNotFoundError(f"ARF文件不存在: {arffits}")

        # 使用 FitsRecToNdarray 类读取 RMF 数据
        rmf_converter = FitsRecToNdarray(rmffits, extension_name=rmf_extension)
        rmf_converter.open_fits()
        rmfdata = rmf_converter.to_ndarray(fields=[rmf_field])  # 提取指定字段

        # 使用 FitsRecToNdarray 类读取 ARF 数据
        arf_converter = FitsRecToNdarray(arffits, extension_name=arf_extension)
        arf_converter.open_fits()
        arfdata = arf_converter.to_ndarray(fields=[arf_field])  # 提取指定字段

        # 提取 ARF 数据中的有效面积列
        # arfdata = arfdata[arf_field]  # 提取字段为一维数组
        # rmfdata = rmfdata[rmf_field]  # 提取矩阵数据
        # 这个地方的提取实际上是不必要的,因为前面实际使用了extension_name参数,所以实际上是提取了指定的字段
        # 这两行实际上是把结构化数组变成单纯的数组, 这并不必要
        # 生成响应矩阵
        return Rsp.get_rsp(rmfdata, arfdata)
#日志,这个地方应该将其转化为list或者结构化数组的,明天写的时候记得把rmfdata和arfdata转化为list或者结构化数组
#已经解决上边的问题,但是实际上调用的FitsRecToArray类并不是很好用,需要指定fields,而且还需要指定extension_name