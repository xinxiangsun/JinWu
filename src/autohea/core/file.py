'''
Date: 2025-04-26 13:46:23
LastEditors: Xinxiang Sun sunxinxiang24@mails.ucas.ac.cn
LastEditTime: 2025-07-22 22:30:17
FilePath: /research/autohea/src/autohea/core/file.py
'''
import numpy as np
import astropy.io.fits as fits
from pathlib import Path
from gdt.core.file import FitsFileContextManager


__all__ = ['FitsRecToNdarray', 'ArfReader', 'RmfReader', 'RspReader']

class FitsRecToNdarray():
    """
    A utility class to convert FITS record (FITS_rec) to numpy ndarray.
    用于将 FITS 文件记录 (FITS_rec) 转换为 numpy ndarray 的工具类。
    """
    def __init__(self, fits_file: str | Path, extension_name: str = None):
        """
        Initialize the class with the FITS file path and optional extension name.
        使用 FITS 文件路径和可选的扩展名初始化类。

        Args:
            fits_file (str): Path to the FITS file. (FITS 文件路径)
            extension_name (str): Name of the extension to extract data from (optional). (扩展名，可选)
        """
        self.fits_file = fits_file
        self.extension_name = extension_name
        self.hdu = None
        self.data = None

    def open_fits(self, hdu_index: int = None):
        """
        Open the FITS file and read the data from the specified extension or HDU index.
        打开 FITS 文件并从指定的扩展名或 HDU 索引中读取数据。

        Args:
            hdu_index (int): Index of the HDU to extract data from (optional). If None, use extension_name or default to the first HDU with data.
                            (要提取数据的 HDU 索引，可选。如果为 None，则使用扩展名或默认第一个有数据的 HDU。)
        """
        with fits.open(self.fits_file) as hdul:
            # If hdu_index is provided, use it directly
            if hdu_index is not None:
                if hdu_index < 0 or hdu_index >= len(hdul):
                    raise IndexError(f"HDU index {hdu_index} is out of range. This file has {len(hdul)} HDUs.")
                self.hdu = hdul[hdu_index]
            elif self.extension_name:
                # If extension name is provided, find the corresponding HDU
                for hdu in hdul:
                    if hdu.name == self.extension_name:
                        self.hdu = hdu
                        break
                if not self.hdu:
                    raise ValueError(f"Extension '{self.extension_name}' not found in FITS file.")
            else:
                # Default to the first extension with data
                for hdu in hdul:
                    if hdu.data is not None:
                        self.hdu = hdu
                        break
                if not self.hdu:
                    raise ValueError("No HDU with data found in the FITS file.")

            # Read the data
            self.data = self.hdu.data
            print("FITS file opened successfully. 数据已成功读取。")

    def to_ndarray(self, fields: list = None) -> np.ndarray:
        """
        Convert the FITS_rec data to a numpy structured ndarray.
        将 FITS_rec 数据转换为 numpy 结构化 ndarray。

        Args:
            fields (list): List of field names to extract (optional). If None, all fields are used.
                          (要提取的字段名称列表，可选。如果为 None，则使用所有字段。)

        Returns:
            np.ndarray: Converted numpy structured array. (转换后的 numpy 结构化数组)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please call 'open_fits()' first. 数据未加载，请先调用 'open_fits()'。")

        # Get all field names if fields are not specified
        if fields is None:
            fields = self.data.columns.names

        # Prepare the dtype for the structured array
        dtype = []
        for field in fields:
            field_data = self.data[field]
            if isinstance(field_data[0], np.ndarray):
                # Handle array fields (e.g., MATRIX)
                max_shape = max(row.shape for row in field_data)
                dtype.append((field, field_data.dtype, max_shape))
            else:
                # Handle scalar fields
                dtype.append((field, field_data.dtype))

        # Create the structured array
        structured_array = np.zeros(len(self.data), dtype=dtype)
        for field in fields:
            field_data = self.data[field]
            if isinstance(field_data[0], np.ndarray):
                # Ensure consistent shape for nested arrays
                for i, row in enumerate(field_data):
                    structured_array[field][i] = np.resize(row, max_shape)
            else:
                structured_array[field] = field_data

        print("Data successfully converted to structured ndarray. 数据已成功转换为结构化 ndarray。")
        return structured_array
    

class ArfReader:
    """
    A base class for reading ARF (Auxiliary Response File) files.
    用于读取 ARF 文件（辅助响应文件）的基类。
    """

    def __init__(self, 
                 arf_filepath: str | Path = None, 
                 arf_extension: str = 'SPECRESP', 
                 arf_elo: str = 'ENERG_LO', 
                 arf_ehi: str = 'ENERG_HI', 
                 arf_specresp: str = 'SPECRESP'):
        """
        Initialize the ARF reader with the file path and field names.
        使用文件路径和字段名称初始化 ARF 读取器。

        Args:
            arf_file (str | Path): Path to the ARF file. (ARF 文件路径，可选)
            arf_extension (str): The extension name for the ARF file. (ARF 文件扩展名)
            arf_elo (str): The field name for the lower energy bound. (能量下界字段名)
            arf_ehi (str): The field name for the upper energy bound. (能量上界字段名)
            arf_specresp (str): The field name for the effective area. (有效面积字段名)
        """
        self._arf_filepath = None
        self.arfdata = None  # 用于存储 ARF 数据
        self.arfheader = None  # 用于存储 ARF 文件头信息
        self.arf_extension = arf_extension  # ARF 扩展名
        self.arf_elo = arf_elo  # 能量下界字段名
        self.arf_ehi = arf_ehi  # 能量上界字段名
        self.arf_specresp = arf_specresp  # 有效面积字段名
        if arf_filepath:
            self._arf_filepath = arf_filepath  # 使用 setter 方法设置文件路径

        super().__init__()


    @property
    def arf_filepath(self) -> Path:
        """
        Get the ARF file path.
        获取 ARF 文件路径。

        Returns:
            Path: The current ARF file path. (当前 ARF 文件路径)
        """
        return self._arf_filepath

    @arf_filepath.setter
    def arf_filepath(self, value: str | Path):
        """
        Set the ARF file path and load the data.
        设置 ARF 文件路径并加载数据。

        Args:
            value (str | Path): Path to the ARF file. (ARF 文件路径)
        """
        self._arf_filepath = Path(value)
        if not self._arf_filepath.exists():
            raise FileNotFoundError(f"ARF file not found: {self._arf_filepath}")
        self.open_arf()

    def open_arf(self):
        """
        Open the ARF file and read the data and header.
        打开 ARF 文件并读取数据和头信息。
        """
        fits_reader = FitsRecToNdarray(self._arf_filepath, extension_name=self.arf_extension)
        fits_reader.open_fits()  # ARF 数据通常在指定扩展中
        self.arfdata = fits_reader.data
        self.arfheader = fits_reader.hdu.header
        print(f"ARF file opened successfully. Extension: {self.arf_extension}. ARF 文件已成功打开，扩展名: {self.arf_extension}。")

    def energy(self) -> np.ndarray:
        """
        Get the energy bounds (ENERG_LO and ENERG_HI) from the ARF file.
        从 ARF 文件中获取能量边界 (ENERG_LO 和 ENERG_HI)。

        Returns:
            np.ndarray: A structured array with 'ENERG_LO' and 'ENERG_HI' fields.
                        (包含 'ENERG_LO' 和 'ENERG_HI' 字段的结构化数组)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_arf()' first. 数据未加载，请先调用 'open_arf()'。")

        if self.arf_elo not in self.arfdata.columns.names or self.arf_ehi not in self.arfdata.columns.names:
            raise ValueError(f"{self.arf_elo} or {self.arf_ehi} columns not found in the ARF file. ARF 文件中未找到 {self.arf_elo} 或 {self.arf_ehi} 列。")

        print("Energy bounds retrieved successfully. 能量边界已成功获取。")
        return np.array(
            list(zip(self.arfdata[self.arf_elo], self.arfdata[self.arf_ehi])),
            dtype=[('ENERG_LO', '<f4'), ('ENERG_HI', '<f4')]  # 显式指定数据类型为 f4 (float32)
        )

    @property
    def arf(self) -> np.ndarray:
        """
        Get the effective area (SPECRESP) from the ARF file.
        从 ARF 文件中获取有效面积 (SPECRESP)。

        Returns:
            np.ndarray: Effective area values. (有效面积值)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_arf()' first. 数据未加载，请先调用 'open_arf()'。")

        if self.arf_specresp not in self.arfdata.columns.names:
            raise ValueError(f"{self.arf_specresp} column not found in the ARF file. ARF 文件中未找到 {self.arf_specresp} 列。")

        print("Effective area retrieved successfully. 有效面积已成功获取。单位一般为 cm²。")
        return self.arfdata[self.arf_specresp]

    def elo(self) -> np.ndarray:
        """
        Get the minimum energy (ENERG_LO) from the ARF file.
        从 ARF 文件中获取最小能量 (ENERG_LO)。

        Returns:
            np.ndarray: Minimum energy values. (最小能量值)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_arf()' first. 数据未加载，请先调用 'open_arf()'。")

        if self.arf_elo not in self.arfdata.columns.names:
            raise ValueError(f"{self.arf_elo} column not found in the ARF file. ARF 文件中未找到 {self.arf_elo} 列。")

        print("Minimum energy retrieved successfully. 最小能量已成功获取。")
        return self.arfdata[self.arf_elo]

    def ehi(self) -> np.ndarray:
        """
        Get the maximum energy (ENERG_HI) from the ARF file.
        从 ARF 文件中获取最大能量 (ENERG_HI)。

        Returns:
            np.ndarray: Maximum energy values. (最大能量值)
        """
        if self.arfdata is None:
            raise ValueError("No data loaded. Please call 'open_arf()' first. 数据未加载，请先调用 'open_arf()'。")

        if self.arf_ehi not in self.arfdata.columns.names:
            raise ValueError(f"{self.arf_ehi} column not found in the ARF file. ARF 文件中未找到 {self.arf_ehi} 列。")

        print("Maximum energy retrieved successfully. 最大能量已成功获取。")
        return self.arfdata[self.arf_ehi]


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

        fits_reader = FitsRecToNdarray(self._arf_filepath)
        fits_reader.data = self.arfdata  # 直接使用已加载的数据
        print("Converting ARF data to structured ndarray. 正在将 ARF 数据转换为结构化 ndarray。")
        return fits_reader.to_ndarray(fields=None)  # 传递 None 以使用所有字段
    

class RmfReader:
    """
    A base class for reading RMF (Redistribution Matrix File) files.
    用于读取 RMF 文件（重分布矩阵文件）的基类。
    """

    def __init__(self, 
                 rmf_filepath: str | Path = None, 
                 rmf_extension: str = 'MATRIX', 
                 rmfebounds_extension: str = 'EBOUNDS', 
                 rmf_ehi: str = 'ENERG_HI',
                 rmf_elo: str = 'ENERG_LO',
                 rmf_matrix: str = 'MATRIX',
                 rmf_channel: str = 'CHANNEL',
                 rmf_chanlo: str = 'E_MIN',
                 rmf_chanhi: str = 'E_MAX'):
        """
        Initialize the RMF reader with the file path and field names.
        使用文件路径和字段名称初始化 RMF 读取器。

        Args:
            rmf_file (str | Path): Path to the RMF file. (RMF 文件路径，可选)
            rmf_extension (str): The extension name for the RMF file. (RMF 文件扩展名)
            rmfebounds_extension (str): The extension name for the EBOUNDS extension. (EBOUNDS 扩展名)
            rmf_ehi (str): The field name for the upper energy bound. (光子能量上界字段名)
            rmf_elo (str): The field name for the lower energy bound. (光子能量下界字段名)
            rmf_matrix (str): The field name for the response matrix. (响应矩阵字段名)
            rmf_channel (str): The field name for the channel information. (通道信息字段名)
            rmf_chanlo (str): The field name for the lower channel bound. (通道下界字段名)
            rmf_chanhi (str): The field name for the upper channel bound. (通道上界字段名)
        """
        self._rmf_filepath = None  # 用于存储 RMF 文件路径
        self.rmfdata = None  # 用于存储 MATRIX 扩展的数据
        self.rmfheader = None  # 用于存储 MATRIX 扩展的头信息
        self.rmfeboundsdata = None  # 用于存储 EBOUNDS 扩展的数据
        self.rmfeboundsheader = None  # 用于存储 EBOUNDS 扩展的头信息
        self.rmf_extension = rmf_extension  # RMF 扩展名
        self.rmfebounds_extension = rmfebounds_extension  # EBOUNDS 扩展名
        self.rmf_ehi = rmf_ehi  # 光子能量上界字段名
        self.rmf_elo = rmf_elo  # 光子能量下界字段名
        self.rmf_matrix = rmf_matrix  # 响应矩阵字段名
        self.rmf_channel = rmf_channel  # 通道信息字段名
        self.rmf_chanlo = rmf_chanlo  # 通道下界字段名
        self.rmf_chanhi = rmf_chanhi  # 通道上界字段名
        if rmf_filepath:
            self._rmf_filepath = rmf_filepath  # 使用 setter 方法设置文件路径

        super().__init__()

    @property
    def rmf_filepath(self) -> Path:
        """
        Get the RMF file path.
        获取 RMF 文件路径。

        Returns:
            Path: The current RMF file path. (当前 RMF 文件路径)
        """
        return self._rmf_filepath

    @rmf_filepath.setter
    def rmf_filepath(self, value: str | Path):
        """
        Set the RMF file path and load the data.
        设置 RMF 文件路径并加载数据。

        Args:
            value (str | Path): Path to the RMF file. (RMF 文件路径)
        """
        self._rmf_filepath = Path(value)
        if not self._rmf_filepath.exists():
            raise FileNotFoundError(f"RMF file not found: {self._rmf_filepath}")
        self.open_rmf()

    def open_rmf(self):
        """
        Open the RMF file and read the data and header from both MATRIX and EBOUNDS extensions.
        打开 RMF 文件并从 MATRIX 和 EBOUNDS 扩展中读取数据和头信息。
        """
        # 读取 MATRIX 扩展
        fits_reader_matrix = FitsRecToNdarray(self._rmf_filepath, extension_name=self.rmf_extension)
        fits_reader_matrix.open_fits()
        self.rmfdata = fits_reader_matrix.data
        self.rmfheader = fits_reader_matrix.hdu.header

        # 读取 EBOUNDS 扩展
        fits_reader_ebounds = FitsRecToNdarray(self._rmf_filepath, extension_name=self.rmfebounds_extension)
        fits_reader_ebounds.open_fits()
        self.rmfeboundsdata = fits_reader_ebounds.data
        self.rmfeboundsheader = fits_reader_ebounds.hdu.header

        print("RMF file opened successfully. MATRIX and EBOUNDS data loaded. RMF 文件已成功打开，MATRIX 和 EBOUNDS 数据已加载。")

    def energy(self) -> np.ndarray:
        """
        Get the energy bounds (ENERG_LO and ENERG_HI) from the RMF file.
        从 RMF 文件中获取能量边界 (ENERG_LO 和 ENERG_HI)。

        Returns:
            np.ndarray: A structured array with 'ENERG_LO' and 'ENERG_HI' fields.
                        (包含 'ENERG_LO' 和 'ENERG_HI' 字段的结构化数组)
        """
        if self.rmfdata is None:
            raise ValueError("No MATRIX data loaded. Please call 'open_rmf()' first. MATRIX 数据未加载，请先调用 'open_rmf()'。")

        if self.rmf_elo not in self.rmfdata.columns.names or self.rmf_ehi not in self.rmfdata.columns.names:
            raise ValueError(f"{self.rmf_elo} or {self.rmf_ehi} columns not found in the MATRIX extension. MATRIX 扩展中未找到 {self.rmf_elo} 或 {self.rmf_ehi} 列。")

        print("Energy bounds retrieved successfully. 能量边界已成功获取。")
        return np.array(
            list(zip(self.rmfdata[self.rmf_elo], self.rmfdata[self.rmf_ehi])),
            dtype=[('ENERG_LO', '<f4'), ('ENERG_HI', '<f4')]  # 显式指定数据类型为 f4 (float32)
        )

    @property
    def rmfmatrix(self) -> np.ndarray:
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
        return self.rmfdata[self.rmf_matrix]

    def channel(self) -> np.ndarray:
        """
        Get the channel information (CHANNEL) from the EBOUNDS extension.
        从 EBOUNDS 扩展中获取通道信息 (CHANNEL)。

        Returns:
            np.ndarray: Channel information. (通道信息)
        """
        if self.rmfeboundsdata is None:
            raise ValueError("No EBOUNDS data loaded. Please call 'open_rmf()' first. EBOUNDS 数据未加载，请先调用 'open_rmf()'。")

        if self.rmf_channel not in self.rmfeboundsdata.columns.names:
            raise ValueError(f"{self.rmf_channel} column not found in the EBOUNDS extension. EBOUNDS 扩展中未找到 {self.rmf_channel} 列。")

        print("Channel information retrieved successfully. 通道信息已成功获取。")
        return np.array(
            list(zip(self.rmfeboundsdata[self.rmf_channel],self.rmfeboundsdata[self.rmf_chanlo], self.rmfeboundsdata[self.rmf_chanhi])),
            dtype=[('channel','<f4'),('chan_lo', '<f4'), ('chan_hi', '<f4')]  # 显式指定数据类型为 f4 (float32)
        )

    def elo(self) -> np.ndarray:
        """
        Get the minimum energy (ENERG_LO) from the RMF file.
        从 RMF 文件中获取最小能量 (ENERG_LO)。

        Returns:
            np.ndarray: Minimum energy values. (最小能量值)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if self.rmf_elo not in self.rmfdata.columns.names:
            raise ValueError("ENERG_LO column not found in the RMF file. RMF 文件中未找到 E_MIN 列。")

        print("Minimum energy retrieved successfully. 能量下界已成功获取。")
        return self.rmfdata[self.rmf_elo]
    

    def ehi(self) -> np.ndarray:
        """
        Get the maximum energy (ENERG_HI) from the RMF file.
        从 RMF 文件中获取最大能量 (ENERG_HI)。

        Returns:
            np.ndarray: Maximum energy values. (最大能量值)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if self.rmf_ehi not in self.rmfdata.columns.names:
            raise ValueError("ENERG_HI column not found in the RMF file. RMF 文件中未找到 E_MAX 列。")

        print("Maximum energy retrieved successfully. 能量上界已成功获取。")
        return self.rmfdata[self.rmf_ehi]

    def chanlo(self) -> np.ndarray:
        """
        Get the channel information (CHANNEL) from the RMF file.
        从 RMF 文件中获取通道信息 (CHANNEL)。

        Returns:
            np.ndarray: Channel information. (通道信息)
        """
        if self.rmfeboundsdata is None:
            raise ValueError("No EBOUNDS data loaded. Please call 'open_rmf()' first. EBOUNDS 数据未加载，请先调用 'open_rmf()'。")
        if self.rmf_chanlo not in self.rmfeboundsdata.columns.names:
            raise ValueError("E_MIN column not found in the RMF file. RMF 文件中未找到 E_MIN 列。")
        print("Channel information retrieved successfully. 通道下界已成功获取。")
        return self.rmfeboundsdata[self.rmf_chanlo]
    
    def chanhi(self) -> np.ndarray:
        """
        Get the channel information (CHANNEL) from the RMF file.
        从 RMF 文件中获取通道信息 (CHANNEL)。
        Returns:
            np.ndarray: Channel information. (通道信息)
        """
        if self.rmfeboundsdata is None:
            raise ValueError("No EBOUNDS data loaded. Please call 'open_rmf()' first. EBOUNDS 数据未加载，请先调用 'open_rmf()'。")
        if self.rmf_chanhi not in self.rmfeboundsdata.columns.names:
            raise ValueError("E_HIGH column not found in the RMF file. RMF 文件中未找到 E_HIGH 列。")
        print("Channel information retrieved successfully. 通道上界已成功获取。")
        return self.rmfeboundsdata[self.rmf_chanhi]
    

    def matrix_shape(self) -> tuple:
        """
        Get the shape of the response matrix (MATRIX) from the RMF file.
        从 RMF 文件中获取响应矩阵 (MATRIX) 的形状。

        Returns:
            tuple: Shape of the response matrix. (响应矩阵的形状)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if self.rmf_matrix not in self.rmfdata.columns.names:
            raise ValueError("MATRIX column not found in the RMF file. RMF 文件中未找到 MATRIX 列。")

        print("Response matrix shape retrieved successfully. 响应矩阵形状已成功获取。")
        return self.rmfdata[self.rmf_matrix].shape
    
    
    def channel_index(self) -> np.ndarray:
        """
        Get the channel index from the RMF file.
        从 RMF 文件中获取通道索引。

        Returns:
            np.ndarray: Channel index. (通道索引)
        """
        if self.rmfdata is None:
            raise ValueError("No data loaded. Please call 'open_rmf()' first. 数据未加载，请先调用 'open_rmf()'。")

        if self.rmf_channel not in self.rmfdata.columns.names:
            raise ValueError("CHANNEL column not found in the RMF file. RMF 文件中未找到 CHANNEL 列。")

        print("Channel index retrieved successfully. 通道索引已成功获取。")
        print(f"first channel index is {self.rmfdata[self.rmf_channel][0]}")
        return self.rmfdata[self.rmf_channel]
    


class RspReader:
    """
    A base class for reading RSP (Response Matrix File) files.
    用于读取 RSP 文件（响应矩阵文件）的基类。
    """

    def __init__(self, 
                 rsp_filepath: str | Path = None, 
                 rsp_extension: str = 'SPECRESP MATRIX', 
                 rspebounds_extension: str = 'EBOUNDS',
                 rsp_ehi: str = 'ENERG_HI',
                 rsp_elo: str = 'ENERG_LO',
                 rsp_matrix: str = 'MATRIX',
                 rsp_channel: str = 'CHANNEL',
                 rsp_chanlo: str = 'E_MIN',
                 rsp_chanhi: str = 'E_MAX'
                 ):
        """
        Initialize the RSP reader with the file path and extension name.
        使用文件路径和扩展名初始化 RSP 读取器。

        Args:
            rsp_filepath (str | Path): Path to the RSP file. (RSP 文件路径，可选)
            rsp_extension (str): The extension name for the RSP file. (RSP 文件扩展名)
        """
        self._rsp_filepath = None  # 用于存储 RSP 文件路径
        self.rspdata = None  # 用于存储 RSP 数据
        self.rspheader = None  # 用于存储 RSP 文件头信息
        self.rsp_extension = rsp_extension  # RSP 扩展名
        self.rspeboundsdata = None  # 用于存储 EBOUNDS 扩展的数据
        self.rspeboundsheader = None  # 用于存储 EBOUNDS 扩展的头信息
        self.rsp_extension = rsp_extension  # RMF 扩展名
        self.rspebounds_extension = rspebounds_extension  # EBOUNDS 扩展名
        if rsp_filepath:
            self.rsp_filepath = rsp_filepath  # 使用 setter 方法设置文件路径

        super().__init__()

    @property
    def rsp_filepath(self) -> Path:
        """
        Get the RSP file path.
        获取 RSP 文件路径。

        Returns:
            Path: The current RSP file path. (当前 RSP 文件路径)
        """
        return self._rsp_filepath

    @rsp_filepath.setter
    def rsp_filepath(self, value: str | Path):
        """
        Set the RSP file path and load the data.
        设置 RSP 文件路径并加载数据。

        Args:
            value (str | Path): Path to the RSP file. (RSP 文件路径)
        """
        self._rsp_filepath = Path(value)
        if not self._rsp_filepath.exists():
            raise FileNotFoundError(f"RSP file not found: {self._rsp_filepath}")
        self.open_rsp()

    def open_rsp(self):
        """
        Open the RSP file and read the data and header.
        打开 RSP 文件并读取数据和头信息。
        """
        fits_reader = FitsRecToNdarray(self._rsp_filepath, extension_name=self.rsp_extension)
        fits_reader.open_fits()  # RSP 数据通常在指定扩展中
        self.rspdata = fits_reader.data
        self.rspheader = fits_reader.hdu.header
        print(f"RSP file opened successfully. Extension: {self.rsp_extension}. RSP 文件已成功打开，扩展名: {self.rsp_extension}。")

    def get_response_matrix(self) -> np.ndarray:
        """
        Get the response matrix (MATRIX) from the RSP file.
        从 RSP 文件中获取响应矩阵 (MATRIX)。

        Returns:
            np.ndarray: Response matrix values. (响应矩阵值)
        """
        if self.rspdata is None:
            raise ValueError("No data loaded. Please call 'open_rsp()' first. 数据未加载，请先调用 'open_rsp()'。")

        if 'MATRIX' not in self.rspdata.columns.names:
            raise ValueError("MATRIX column not found in the RSP file. RSP 文件中未找到 MATRIX 列。")

        print("Response matrix retrieved successfully. 响应矩阵已成功获取。")
        return self.rspdata['MATRIX']

    def energy(self) -> np.ndarray:
        """
        Get the energy bounds (ENERG_LO and ENERG_HI) from the RSP file.
        从 RSP 文件中获取能量边界 (ENERG_LO 和 ENERG_HI)。

        Returns:
            np.ndarray: A structured array with 'ENERG_LO' and 'ENERG_HI' fields.
                        (包含 'ENERG_LO' 和 'ENERG_HI' 字段的结构化数组)
        """
        if self.rspdata is None:
            raise ValueError("No data loaded. Please call 'open_rsp()' first. 数据未加载，请先调用 'open_rsp()'。")

        if 'ENERG_LO' not in self.rspdata.columns.names or 'ENERG_HI' not in self.rspdata.columns.names:
            raise ValueError("ENERG_LO or ENERG_HI columns not found in the RSP file. RSP 文件中未找到 ENERG_LO 或 ENERG_HI 列。")

        print("Energy bounds retrieved successfully. 能量边界已成功获取。")
        return np.array(
            list(zip(self.rspdata['ENERG_LO'], self.rspdata['ENERG_HI'])),
            dtype=[('ENERG_LO', '<f4'), ('ENERG_HI', '<f4')]  # 显式指定数据类型为 f4 (float32)
        )

    def channels(self) -> np.ndarray:
        """
        Get the channel information (CHANNEL) from the RSP file.
        从 RSP 文件中获取通道信息 (CHANNEL)。

        Returns:
            np.ndarray: Channel information. (通道信息)
        """
        if self.rspdata is None:
            raise ValueError("No data loaded. Please call 'open_rsp()' first. 数据未加载，请先调用 'open_rsp()'。")

        if 'CHANNEL' not in self.rspdata.columns.names:
            raise ValueError("CHANNEL column not found in the RSP file. RSP 文件中未找到 CHANNEL 列。")

        print("Channel information retrieved successfully. 通道信息已成功获取。")
        return self.rspdata['CHANNEL']




