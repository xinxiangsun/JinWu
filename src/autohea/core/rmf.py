from .file import FitsRecToNdarray
from pathlib import Path
import numpy as np


class RmfReader:
    """
    A base class for reading RMF (Redistribution Matrix File) files.
    用于读取 RMF 文件（重分布矩阵文件）的基类。
    """

    def __init__(self, rmf_file: str | Path = None):
        """
        Initialize the RMF reader with the file path.
        使用文件路径初始化 RMF 读取器。

        Args:
            rmf_file (str | Path): Path to the RMF file. (RMF 文件路径，可选)
        """
        self._rmf_file = None
        self.rmfdata = None
        self.header = None
        if rmf_file:
            self.rmf_file = rmf_file  # 使用 setter 方法设置文件路径

    @property
    def rmf_path(self) -> Path:
        """
        Get the RMF file path.
        获取 RMF 文件路径。

        Returns:
            Path: The current RMF file path. (当前 RMF 文件路径)
        """
        return self._rmf_file

    @rmf_path.setter
    def rmf_path(self, value: str | Path):
        """
        Set the RMF file path and load the data.
        设置 RMF 文件路径并加载数据。

        Args:
            value (str | Path): Path to the RMF file. (RMF 文件路径)
        """
        self._rmf_file = Path(value)
        if not self._rmf_file.exists():
            raise FileNotFoundError(f"RMF file not found: {self._rmf_file}")
        self.open_rmf()


    def open_rmf(self):
        """
        Open the RMF file and read the data and header.
        打开 RMF 文件并读取数据和头信息。
        """
        fits_reader = FitsRecToNdarray(self._rmf_file, extension_name='MATRIX')
        fits_reader.open_fits()  # RMF 数据通常在 MATRIX 扩展中
        self.rmfdata = fits_reader.data
        self.header = fits_reader.hdu.header
        print("RMF file opened successfully. RMF 文件已成功打开。")

    def get_energy_bounds(self) -> np.ndarray:
        """
        Get the energy bounds (ENERG_LO and ENERG_HI) from the RMF file.
        从 RMF 文件中获取能量边界 (ENERG_LO 和 ENERG_HI)。

        Returns:
            np.ndarray: A structured array with 'ENERG_LO' and 'ENERG_HI' fields.
                        (包含 'ENERG_LO' 和 'ENERG_HI' 字段的结构化数组)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if 'ENERG_LO' not in self.rmfdata.columns.names or 'ENERG_HI' not in self.rmfdata.columns.names:
            raise ValueError("ENERG_LO or ENERG_HI columns not found in the RMF file. RMF 文件中未找到 ENERG_LO 或 ENERG_HI 列。")

        print("Energy bounds retrieved successfully. 能量边界已成功获取。")
        return np.array(
            list(zip(self.rmfdata['ENERG_LO'], self.rmfdata['ENERG_HI'])),
            dtype=[('ENERG_LO', '<f4'), ('ENERG_HI', '<f4')]  # 显式指定数据类型为 f4 (float32)
        )

    def get_response_matrix(self) -> np.ndarray:
        """
        Get the response matrix (MATRIX) from the RMF file.
        从 RMF 文件中获取响应矩阵 (MATRIX)。

        Returns:
            np.ndarray: Response matrix values. (响应矩阵值)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if 'MATRIX' not in self.rmfdata.columns.names:
            raise ValueError("MATRIX column not found in the RMF file. RMF 文件中未找到 MATRIX 列。")

        print("Response matrix retrieved successfully. 响应矩阵已成功获取。")
        return self.rmfdata['MATRIX']



    # def get_channel_bounds(self) -> np.ndarray:
    #     #方法已经弃用,主要原因是因为EP的rmf文件中有两个用于表示通道的部分
    #     #但是实际上使用的部分是Matrix中的1980能道分布,而不是之前常用的1024能量分布
    #     #EP的能量划分更为精细,5eV一个能道,而不是之前的10eV一个能道
    # 
    #     """
    #     Get the channel bounds (CHANNEL) from the RMF file.
    #     从 RMF 文件中获取通道边界 (CHANNEL)。

    #     Returns:
    #         np.ndarray: Channel bounds. (通道边界)
    #     """
    #     if self.rmfdata is None:
    #         raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

    #     if 'CHANNEL' not in self.rmfdata.columns.names:
    #         raise ValueError("CHANNEL column not found in the RMF file. RMF 文件中未找到 CHANNEL 列。")

    #     print("Channel bounds retrieved successfully. 通道边界已成功获取。")
    #     return self.rmfdata['CHANNEL']

    # def to_ndarray(self, fields: list = None) -> np.ndarray:
    #     """
    #     本方法实际上也是没有意义的,因为暂时实在找不到什么办法可以存储嵌套的数组
    #     所以代码弃用
    #     Convert the RMF data to a structured numpy array using FitsRecToNdarray.
    #     使用 FitsRecToNdarray 将 RMF 数据转换为结构化 numpy 数组。

    #     Args:
    #         fields (list): List of field names to extract (optional). If None, all fields are used.
    #                       (要提取的字段名称列表，可选。如果为 None，则使用所有字段。)

    #     Returns:
    #         np.ndarray: Converted numpy structured array. (转换后的 numpy 结构化数组)
    #     """
    #     if self.rmfdata is None:
    #         raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

    #     # 使用 FitsRecToNdarray 的 to_ndarray 方法
    #     fits_reader = FitsRecToNdarray(self.rmf_file)
    #     fits_reader.data = self.rmfdata  # 直接使用已加载的数据
    #     print("Converting RMF data to structured ndarray. 正在将 RMF 数据转换为结构化 ndarray。")
    #     return fits_reader.to_ndarray(fields='MATRIX')  # 传递 'MATRIX' 以使用响应矩阵字段
    def emin(self) -> np.ndarray:
        """
        Get the minimum energy (ENERG_LO) from the RMF file.
        从 RMF 文件中获取最小能量 (ENERG_LO)。

        Returns:
            np.ndarray: Minimum energy values. (最小能量值)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if 'ENERG_LO' not in self.rmfdata.columns.names:
            raise ValueError("ENERG_LO column not found in the RMF file. RMF 文件中未找到 E_MIN 列。")

        print("Minimum energy retrieved successfully. 通道能量下界已成功获取。")
        return self.rmfdata['ENERG_LO']
    

    def emax(self) -> np.ndarray:
        """
        Get the maximum energy (ENERG_HI) from the RMF file.
        从 RMF 文件中获取最大能量 (ENERG_HI)。

        Returns:
            np.ndarray: Maximum energy values. (最大能量值)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if 'ENERG_HI' not in self.rmfdata.columns.names:
            raise ValueError("ENERG_HI column not found in the RMF file. RMF 文件中未找到 E_MAX 列。")

        print("Maximum energy retrieved successfully. 通道能量上界已成功获取。")
        return self.rmfdata['ENERG_HI']