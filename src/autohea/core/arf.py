from .file import FitsRecToNdarray
from pathlib import Path
import numpy as np




class ArfReader:
    """
    A base class for reading ARF (Auxiliary Response File) files.
    用于读取 ARF 文件（辅助响应文件）的基类。
    """

    def __init__(self, arf_file: str | Path = None):
        """
        Initialize the ARF reader with the file path.
        使用文件路径初始化 ARF 读取器。

        Args:
            arf_file (str | Path): Path to the ARF file. (ARF 文件路径，可选)
        """
        self._arf_file = None
        self.arfdata = None
        self.header = None
        if arf_file:
            self.arf_file = arf_file  # 使用 setter 方法设置文件路径

    @property
    def arf_file(self) -> Path:
        """
        Get the ARF file path.
        获取 ARF 文件路径。

        Returns:
            Path: The current ARF file path. (当前 ARF 文件路径)
        """
        return self._arf_file

    @arf_file.setter
    def get_arf_file(self, value: str | Path):
        """
        Set the ARF file path and load the data.
        设置 ARF 文件路径并加载数据。

        Args:
            value (str | Path): Path to the ARF file. (ARF 文件路径)
        """
        self._arf_file = Path(value)
        if not self._arf_file.exists():
            raise FileNotFoundError(f"ARF file not found: {self._arf_file}")
        self.open_arf()

    def open_arf(self):
        """
        Open the ARF file and read the data and header.
        打开 ARF 文件并读取数据和头信息。
        """
        fits_reader = FitsRecToNdarray(self._arf_file, extension_name='SPECRESP')
        fits_reader.open_fits()  # ARF 数据通常在第一个扩展（HDU 1）
        self.arfdata = fits_reader.data
        self.header = fits_reader.hdu.header
        print("ARF file opened successfully. ARF 文件已成功打开。")

    def get_energy_bounds(self) -> np.ndarray:
        """
        Get the energy bounds (ENERG_LO and ENERG_HI) from the ARF file.
        从 ARF 文件中获取能量边界 (ENERG_LO 和 ENERG_HI)。

        Returns:
            np.ndarray: A structured array with 'E_MIN' and 'E_MAX' fields.
                        (包含 'E_MIN' 和 'E_MAX' 字段的结构化数组)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_arf()' first. 数据未加载，请先调用 'open_arf()'。")

        if 'ENERG_LO' not in self.arfdata.columns.names or 'ENERG_HI' not in self.arfdata.columns.names:
            raise ValueError("ENERG_LO or ENERG_HI columns not found in the ARF file. ARF 文件中未找到 ENERG_LO 或 ENERG_HI 列。")

        print("Energy bounds retrieved successfully. 能量边界已成功获取。")
        return np.array(
            list(zip(self.arfdata['ENERG_LO'], self.arfdata['ENERG_HI'])),
            dtype=[('E_MIN', '<f4'), ('E_MAX', '<f4')]  # 显式指定数据类型为 f4 (float32)
        )


    def emin(self) -> np.ndarray:
        """
        Get the minimum energy (ENERG_LO) from the RMF file.
        从 RMF 文件中获取最小能量 (ENERG_LO)。

        Returns:
            np.ndarray: Minimum energy values. (最小能量值)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if 'ENERG_LO' not in self.arfdata.columns.names:
            raise ValueError("ENERG_LO column not found in the RMF file. RMF 文件中未找到 E_MIN 列。")

        print("Minimum energy retrieved successfully. 通道能量下界已成功获取。")
        return self.arfdata['ENERG_LO']
    

    def emax(self) -> np.ndarray:
        """
        Get the maximum energy (ENERG_HI) from the RMF file.
        从 RMF 文件中获取最大能量 (ENERG_HI)。

        Returns:
            np.ndarray: Maximum energy values. (最大能量值)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if 'ENERG_HI' not in self.arfdata.columns.names:
            raise ValueError("ENERG_HI column not found in the RMF file. RMF 文件中未找到 E_MAX 列。")

        print("Maximum energy retrieved successfully. 通道能量上界已成功获取。")
        return self.arfdata['ENERG_HI']


    def get_effective_area(self) -> np.ndarray:
        """
        Get the effective area (SPECRESP) from the ARF file.
        从 ARF 文件中获取有效面积 (SPECRESP)。

        Returns:
            np.ndarray: Effective area values. (有效面积值)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_arf()' first. 数据未加载，请先调用 'open_arf()'。")

        if 'SPECRESP' not in self.arfdata.columns.names:
            raise ValueError("SPECRESP column not found in the ARF file. ARF 文件中未找到 SPECRESP 列。")

        print("Effective area retrieved successfully. 有效面积已成功获取。单位一般为cm2")
        return self.arfdata['SPECRESP']

    def to_ndarray(self, fields: list = None) -> np.ndarray:
        """
        Convert the ARF data to a structured numpy array using FitsRecToNdarray.
        使用 FitsRecToNdarray 将 ARF 数据转换为结构化 numpy 数组。

        Args:
            fields (list): List of field names to extract (optional). If None, all fields are used.
                          (要提取的字段名称列表，可选。如果为 None，则使用所有字段。)

        Returns:
            np.ndarray: Converted numpy structured array. (转换后的 numpy 结构化数组)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_arf()' first. 数据未加载，请先调用 'open_arf()'。")

        fits_reader = FitsRecToNdarray(self._arf_file)
        fits_reader.data = self.arfdata  # 直接使用已加载的数据
        print("Converting ARF data to structured ndarray. 正在将 ARF 数据转换为结构化 ndarray。")
        return fits_reader.to_ndarray(fields=None)  # 传递 None 以使用所有字段
    

