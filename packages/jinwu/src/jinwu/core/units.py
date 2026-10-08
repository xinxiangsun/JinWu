"""
Custom units and conversion functions for photometric observations.

This module provides:
1. Magnitude: A custom quantity class for magnitudes with filter information
2. FilterInfo: A class to store filter properties (wavelength, zero-point, bandwidth)
3. InstrumentFilterLibrary: Hierarchical filter organization (telescope → instrument → filters)
4. TELESCOPES: Nested dict organizing filters by telescope and instrument
5. Conversion functions between magnitude and flux/frequency flux density

Example Usage
=============
>>> from jinwu.core.units import TELESCOPES, Magnitude
>>> 
>>> # Access filters from hierarchical library
>>> filt = TELESCOPES['NOT']['ALFOSC']['NOT/ALFOSC.Bes_R']
>>> 
>>> # Create magnitude with uncertainty
>>> mag = Magnitude(20.5, filt, system='Vega', error=0.1)
>>> 
>>> # Convert to frequency flux density (erg/cm²/s/Hz)
>>> fnu = mag.to_fnu()
>>> 
>>> # Convert to wavelength flux density (erg/cm²/s/Å)
>>> flam = mag.to_flam()
>>> 
>>> # Convert to integrated flux (erg/cm²/s)
>>> F = mag.to_flux()
"""

from dataclasses import dataclass
from typing import Union, Literal, Dict
import numpy as np

import astropy.units as u
import astropy.constants as const
from astropy.units import Quantity


@dataclass
class FilterInfo:
    """Store photometric filter properties with both AB and Vega zero-points.
    
    All filters support both AB and Vega magnitude systems. The zero-points are:
    - AB system: ZP_AB = 3631 Jy (constant for all filters)
    - Vega system: ZP_Vega varies by filter (from SVO Filter Profile Service)
    
    Attributes
    ----------
    name : str
        Filter identifier (e.g., 'NOT/ALFOSC.Bes_R')
    wavelength : Quantity
        Effective wavelength λ_eff (for f_ν calculation)
    weff : Quantity
        Effective bandwidth (rectangular equivalent width)
    zero_point_vega : Quantity
        Vega system zero-point flux in Jy (from SVO FPS)
    lambda_pivot : Quantity, optional
        Pivot wavelength λ_pivot for f_ν ↔ f_λ conversion
        Defaults to wavelength if not provided
    zero_point_ab : Quantity, optional
        AB system zero-point (defaults to 3631 Jy)
    """
    name: str
    wavelength: Quantity
    weff: Quantity
    zero_point_vega: Quantity
    lambda_pivot: Quantity = None
    zero_point_ab: Quantity = 3631 * u.Jy
    
    def __post_init__(self):
        """设置滤光片默认值 / Set filter defaults and check wavelength input.

        wavelength 必须为 Quantity；lambda_pivot 缺失时引用 wavelength。
        裸数 zero_point_ab 按 Jy 包装。原位更新字段；此处不全面核验带宽、
        Vega 零点或各 Quantity 的量纲、正值条件。
        Require wavelength to be a Quantity, use it for a missing pivot, and attach
        Jy to a bare AB zero point. Mutate fields without comprehensive dimension,
        bandwidth, Vega-zero-point or positivity checks."""
        # Ensure wavelength has units
        if not isinstance(self.wavelength, Quantity):
            raise ValueError("wavelength must be an astropy Quantity with units")
        
        # Set pivot wavelength to effective wavelength if not provided
        if self.lambda_pivot is None:
            self.lambda_pivot = self.wavelength
        
        # Ensure AB zero-point has units
        if not isinstance(self.zero_point_ab, Quantity):
            self.zero_point_ab = self.zero_point_ab * u.Jy
    
    def get_zero_point(self, system: Literal['AB', 'Vega']) -> Quantity:
        """返回所选星等系统零点 / Return the selected photometric zero point.

        system 必须精确为 'AB' 或 'Vega'，否则抛 ValueError；返回对应
        Quantity 字段，通常为 Jy，不复制或自动转换系统。
        system must be exactly 'AB' or 'Vega', otherwise ValueError. Return the
        corresponding Quantity field, conventionally Jy, without copying/conversion."""
        if system == 'AB':
            return self.zero_point_ab
        elif system == 'Vega':
            return self.zero_point_vega
        else:
            raise ValueError(f"system must be 'AB' or 'Vega', got '{system}'")
    
    def __str__(self):
        """格式化滤光片属性 / Format filter name, wavelengths and zero points.

        返回带原单位的可读字符串，不用于无损序列化。
        Return a human-readable string with stored units, not a lossless serialization."""
        return (f"FilterInfo({self.name}: λ={self.wavelength:.1f}, "
                f"Weff={self.weff:.1f}, ZP_Vega={self.zero_point_vega:.2f}, "
                f"ZP_AB={self.zero_point_ab:.2f})")
    
    def __repr__(self):
        """复用可读滤光片表示 / Return the same readable text as str(self)."""
        return self.__str__()
    
    def __mul__(self, other: Union[float, Quantity]) -> 'Magnitude':
        """由滤光片乘星等创建对象 / Construct a Magnitude by multiplying a filter.

        裸数按 Vega 星等解释；含 'mag' 的 Quantity 提取 value，并根据单位
        名是否含 AB 选择 AB/Vega。不会执行光度单位等价转换，也不传递误差；
        不含星等单位的 Quantity 抛 TypeError。返回新的 Magnitude。
        Bare numbers imply Vega. A magnitude-like Quantity supplies its value;
        a unit name containing AB selects AB, otherwise Vega. No logarithmic-unit
        conversion or uncertainty transfer is performed. Other units raise TypeError.
        Return a new Magnitude instance."""
        if isinstance(other, Quantity):
            # Handle astropy Quantity (e.g., Magnitude(22, filt) * u.ABmag)
            # Check if unit is magnitude-like
            unit_str = str(other.unit).lower()
            if other.unit == u.mag or 'mag' in unit_str:
                # Extract numeric value
                mag_value = other.value
                # Try to detect system from unit (e.g., ABmag → 'AB')
                system = 'Vega'  # default
                if 'AB' in str(other.unit).upper():
                    system = 'AB'
                return Magnitude(mag_value, self, system=system, error=None)
            else:
                raise TypeError(f"Cannot multiply FilterInfo by {other.unit}. Use magnitude units (mag, ABmag, etc.)")
        else:
            # Handle plain float/int: assume Vega system
            return Magnitude(float(other), self, system='Vega', error=None)
    
    def __rmul__(self, other: Union[float, Quantity]) -> 'Magnitude':
        """支持星等乘滤光片 / Delegate magnitude * filter to filter * magnitude.
        
        行为、系统推断与错误条件同 __mul__。
        Use the same system inference and error conditions as __mul__."""
        return self.__mul__(other)


@dataclass
class InstrumentFilterLibrary:
    """Filter library for a specific telescope/instrument combination.
    
    This provides hierarchical organization: Telescope → Instrument → Filters
    
    Attributes
    ----------
    telescope : str
        Telescope name (e.g., 'NOT', 'Swift')
    instrument : str
        Instrument name (e.g., 'ALFOSC', 'UVOT')
    filters : Dict[str, FilterInfo]
        Dictionary mapping filter names to FilterInfo objects
    """
    telescope: str
    instrument: str
    filters: Dict[str, FilterInfo]
    
    def __getitem__(self, key: str) -> FilterInfo:
        """按完整名称取滤光片 / Return a filter by its exact dictionary key.

        大小写敏感；返回原 FilterInfo 引用，未知 key 抛 KeyError 并列出选项。
        Case-sensitive lookup; return the original FilterInfo reference. Unknown keys
        raise KeyError with available names."""
        if key not in self.filters:
            available = list(self.filters.keys())
            raise KeyError(
                f"Filter '{key}' not found in {self.telescope}/{self.instrument}.\n"
                f"Available: {available}"
            )
        return self.filters[key]
    
    def __getattr__(self, name: str) -> FilterInfo:
        """按简短属性名取滤光片 / Look up a filter by its case-insensitive short name.

        使用标识符最后一个点号后的部分匹配；多个匹配取首个。未找到抛
        AttributeError；数据类字段使用正常属性读取。
        Match the suffix after the last dot, ignoring case; the first match wins.
        Unknown names raise AttributeError; dataclass fields use ordinary access."""
        # Avoid infinite recursion for dataclass attributes
        if name in ('telescope', 'instrument', 'filters'):
            return object.__getattribute__(self, name)
        
        # Try to find filter with lowercase name
        for filter_name, filter_info in self.filters.items():
            # Extract the last part of filter name (e.g., 'white' from 'Swift/UVOT.white')
            filter_short_name = filter_name.split('.')[-1].lower()
            if filter_short_name == name.lower():
                return filter_info
        
        raise AttributeError(
            f"Filter '{name}' not found in {self.telescope}/{self.instrument}.\n"
            f"Available: {[f.split('.')[-1] for f in self.filters.keys()]}"
        )
    
    def __str__(self):
        """显示望远镜、仪器与数量 / Summarize telescope, instrument and filter count."""
        return f"{self.telescope}/{self.instrument} ({len(self.filters)} filters)"
    
    def __repr__(self):
        """复用滤光片库的字符串表示 / Return the library's readable string form."""
        return self.__str__()
    
    def list_filters(self) -> list:
        """列出完整滤光片名称 / List full filter keys in dictionary insertion order.

        返回新列表，不复制滤光片对象。
        Return a new list of names without copying filter objects."""
        return list(self.filters.keys())


class DotAccessor:
    """支持点号访问的包装器，允许 filters.swift.uvot.white 这样的访问方式"""
    
    def __init__(self, data: Dict):
        """保存导航字典引用 / Store the navigation dictionary by reference.

        不复制字典；后续原字典修改会反映到访问结果。
        Do not copy data; later changes to that dictionary affect lookups."""
        self._data = data
    
    def __getattr__(self, name: str):
        """大小写不敏感的层级访问 / Traverse keys case-insensitively by attribute.

        字典值包装为新 DotAccessor；其他值返回原对象；首个匹配获选。
        下划线开头名称走常规属性读取，未知名称抛 AttributeError。
        Wrap nested dictionaries, otherwise return the original value; first match
        wins. Underscore names use normal access. Missing names raise AttributeError."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        # 尝试匹配（不区分大小写）
        name_lower = name.lower()
        for key, value in self._data.items():
            if key.lower() == name_lower:
                if isinstance(value, dict):
                    return DotAccessor(value)
                elif isinstance(value, InstrumentFilterLibrary):
                    return value
                else:
                    return value
        
        available = list(self._data.keys())
        raise AttributeError(
            f"No attribute '{name}' found. Available: {available}"
        )
    
    def __getitem__(self, key: str):
        """精确 key 的层级访问 / Traverse an exact, case-sensitive dictionary key.

        字典结果包装为 DotAccessor；非字典结果直接返回。缺失 key 抛 KeyError。
        Wrap dictionary results; return other values directly. Missing keys raise KeyError."""
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return DotAccessor(value)
            return value
        raise KeyError(f"Key '{key}' not found in {list(self._data.keys())}")
    
    def __mul__(self, other: Union[float, Quantity]) -> 'Magnitude':
        """仅在终端滤光片节点创建星等 / Multiply only a terminal filter node.
        
        当前字典须只有一个 FilterInfo 值，再转交其 __mul__；中间导航层
        不能相乘，抛 TypeError。返回 Magnitude，不修改导航字典。
        Require exactly one FilterInfo value and delegate to its __mul__. Intermediate
        navigation levels raise TypeError. Return Magnitude without mutating navigation."""
        # Check if this is a terminal FilterInfo node
        # (DotAccessor should only wrap dicts or FilterInfo, not nested DotAccessor)
        if len(self._data) == 1:
            value = list(self._data.values())[0]
            if isinstance(value, FilterInfo):
                return value.__mul__(other)
        
        raise TypeError(
            f"Cannot multiply intermediate navigation level directly. "
            f"Navigate to a specific filter first (e.g., filter.swift.uvot.white * 20.5)"
        )
    
    def __rmul__(self, other: Union[float, Quantity]) -> 'Magnitude':
        """支持反向终端乘法 / Delegate reverse multiplication to __mul__."""
        return self.__mul__(other)


class Magnitude:
    """
    A quantity class for astronomical magnitudes with filter and system information.
    
    This class stores magnitude values along with filter properties and system information,
    allowing convenient conversion to flux or frequency flux density.
    
    Parameters
    ----------
    magnitude : Quantity or float
        Magnitude value (dimensionless or in u.mag units)
    filter_info : FilterInfo
        Filter properties (wavelength, zero-point, bandwidth)
    system : Literal['AB', 'Vega'], optional
        Photometric system (default: 'Vega')
    error : Quantity or float, optional
        Magnitude uncertainty (for error propagation)
    
    Examples
    --------
    >>> from astropy import units as u
    >>> from jinwu.core.units import TELESCOPES, Magnitude
    >>> 
    >>> # Get filter from hierarchical library
    >>> filt = TELESCOPES['NOT']['ALFOSC']['NOT/ALFOSC.Bes_R']
    >>> 
    >>> # Create a magnitude with Vega system
    >>> mag = Magnitude(20.5, filt, system='Vega', error=0.1)
    >>> 
    >>> # Or use AB system
    >>> mag_ab = Magnitude(20.5, filt, system='AB', error=0.1)
    >>> 
    >>> # Convert to frequency flux density
    >>> fnu = mag.to_fnu()  # Returns Quantity in erg/cm²/s/Hz
    >>> 
    >>> # Convert to wavelength flux density
    >>> flam = mag.to_flam()  # Returns Quantity in erg/cm²/s/Å
    >>> 
    >>> # Convert to integrated flux
    >>> F = mag.to_flux()  # Returns Quantity in erg/cm²/s
    """
    
    def __init__(self, magnitude: Union[float, Quantity], 
                 filter_info: FilterInfo,
                 system: Literal['AB', 'Vega'] = 'Vega',
                 error: Union[None, float, Quantity] = None):
        """保存星等、系统与可选误差 / Initialize a magnitude and its calibration.

        magnitude/error 的裸数分别按星等和星等标准误差解释；Quantity 仅取
        value，不校验或转换单位。filter_info 提供零点，system 精确为 AB/Vega；
        非法系统由 get_zero_point 报错。误差仅保存，不校验正值或分布。
        Bare magnitude/error mean magnitudes and magnitude standard error. Quantities
        supply value without unit validation/conversion. filter_info supplies the zero
        point; system must be AB/Vega. Invalid systems raise through get_zero_point.
        Uncertainty is stored without positivity/distribution validation."""
        # Normalize magnitude to float value
        if isinstance(magnitude, Quantity):
            self.magnitude = magnitude.value
        else:
            self.magnitude = float(magnitude)
        
        self.filter_info = filter_info
        self.system = system
        
        # Get appropriate zero-point for the system
        self.zero_point = filter_info.get_zero_point(system)
        
        # Normalize error
        if error is not None:
            if isinstance(error, Quantity):
                self.error = error.value
            else:
                self.error = float(error)
        else:
            self.error = None
    
    def to_fnu(self, unit: Union[str, u.Unit] = 'erg/(cm2 s Hz)') -> Quantity:
        """星等转频率通量密度 / Convert magnitude to frequency flux density.

        计算 f_nu = zero_point * 10**(-magnitude/2.5)，再转换到 unit；默认
        'erg/(cm2 s Hz)'。返回 Quantity。已有星等误差时，将一阶传播的
        sigma_f = f_nu * ln(10)/2.5 * error 存为结果的 .error 属性。
        Compute f_nu = zero_point * 10**(-magnitude/2.5) and convert to unit,
        default 'erg/(cm2 s Hz)'. Return a Quantity. If magnitude error is supplied,
        attach first-order sigma_f = f_nu * ln(10)/2.5 * error as its .error attribute.

        使用滤光片零点校准，不积分源谱、滤光片透过率或校准不确定度；
        不兼容的目标单位会由 Astropy 报错。
        Use the filter zero-point calibration; no source-spectrum/bandpass integration
        or calibration-error propagation. Incompatible units raise through Astropy."""
        # Calculate f_ν using the system-specific zero-point
        fnu_jy = self.zero_point * 10**(-self.magnitude / 2.5)
        
        # Convert to requested unit
        fnu = fnu_jy.to(unit)
        
        # Store error for later access
        if self.error is not None:
            fnu_err = fnu * (np.log(10) / 2.5) * self.error
            fnu.error = fnu_err
        
        return fnu
    
    def to_flam(self, unit: Union[str, u.Unit] = 'erg/(cm2 s Angstrom)') -> Quantity:
        """转为波长通量密度 / Convert to wavelength flux density at the pivot.

        先调用 to_fnu，再计算 f_lambda = f_nu * c / lambda_pivot**2。
        unit 默认 'erg/(cm2 s Angstrom)'，返回 Quantity；可选星等误差按
        相同相对一阶误差保存到 .error。不考虑 pivot/零点的不确定度。
        Call to_fnu, then f_lambda = f_nu * c / lambda_pivot**2. Default unit is
        'erg/(cm2 s Angstrom)'. Return Quantity with optional first-order .error;
        pivot/zero-point uncertainty is not propagated. No bandpass integration.

        参考频率/波长谱密度单位换算 / Spectral-density conversion reference:
        https://docs.astropy.org/en/stable/units/equivalencies.html#spectral-flux-density-equivalency"""
        # Get f_ν first
        fnu = self.to_fnu(unit='erg/(cm2 s Hz)')
        
        # Convert to f_λ using pivot wavelength: f_λ = f_ν × c / λ²
        lambda_pivot = self.filter_info.lambda_pivot
        flam = (fnu * const.c / lambda_pivot**2).to(unit)
        
        # Propagate error
        if self.error is not None:
            flam_err = flam * (np.log(10) / 2.5) * self.error
            flam.error = flam_err
        
        return flam
    
    def to_flux(self, unit: Union[str, u.Unit] = 'erg/(cm2 s)') -> Quantity:
        """以有效带宽估算带内通量 / Estimate band flux using effective width.

        计算 F = to_flam() * filter_info.weff，unit 默认 'erg/(cm2 s)'，
        返回 Quantity；星等误差按一阶相对误差附在 .error。weff 须与波长
        单位相容。此实现只做密度乘有效宽度，不对实际源谱与滤光片透过率
        执行积分；因此不能作为任意源谱的精确带内总通量。
        Compute F = to_flam() * filter_info.weff, default 'erg/(cm2 s)', returning
        Quantity with optional first-order magnitude .error. Width must have wavelength
        units. This is density times effective width, not an integral of a source SED
        through transmission, so it does not give exact band flux for an arbitrary SED.
        Zero-point, bandwidth and pivot uncertainties are not propagated."""
        # Get f_λ
        flam = self.to_flam(unit='erg/(cm2 s Angstrom)')
        
        # Integrate: F = f_λ × W_eff
        flux = (flam * self.filter_info.weff).to(unit)
        
        # Propagate error
        if self.error is not None:
            flux_err = flux * (np.log(10) / 2.5) * self.error
            flux.error = flux_err
        
        return flux
    
    def to_Jy(self) -> Quantity:
        """返回 Jy 频率通量密度 / Return to_fnu(unit='Jy') with the same error convention."""
        return self.to_fnu(unit='Jy')
    
    def __str__(self):
        """显示星等、误差与系统 / Format magnitude, optional error, filter and system.

        格式针对标量；数组星等可能无法用当前数值格式显示。
        The numeric formatting is scalar-oriented; array magnitudes may not format."""
        error_str = f" ± {self.error:.2f} mag" if self.error is not None else ""
        return f"Magnitude({self.magnitude:.2f}{error_str}, {self.filter_info.name}, {self.system})"
    
    def __repr__(self):
        """复用星等的可读字符串 / Return the same readable representation as str(self)."""
        return self.__str__()


# ==================== STANDARD FILTER DEFINITIONS ====================
# Organized hierarchically as: TELESCOPES[telescope][instrument] = InstrumentFilterLibrary
# Data from SVO Filter Profile Service: https://svo2.cab.inta-csic.es/theory/fps/

TELESCOPES = {
    # ==================== NOT (Nordic Optical Telescope) ====================
    'NOT': {
        'ALFOSC': InstrumentFilterLibrary(
            telescope='NOT',
            instrument='ALFOSC',
            filters={
                'NOT/ALFOSC.Bes_U': FilterInfo(
                    name='NOT/ALFOSC.Bes_U',
                    wavelength=3670.73 * u.Angstrom,
                    lambda_pivot=3600.85 * u.Angstrom,
                    weff=580.28 * u.Angstrom,
                    zero_point_vega=1758.31 * u.Jy,
                ),
                'NOT/ALFOSC.Bes_B': FilterInfo(
                    name='NOT/ALFOSC.Bes_B',
                    wavelength=4319.73 * u.Angstrom,
                    lambda_pivot=4306.12 * u.Angstrom,
                    weff=1004.43 * u.Angstrom,
                    zero_point_vega=3923.93 * u.Jy,
                ),
                'NOT/ALFOSC.Bes_V': FilterInfo(
                    name='NOT/ALFOSC.Bes_V',
                    wavelength=5365.72 * u.Angstrom,
                    lambda_pivot=5389.63 * u.Angstrom,
                    weff=885.24 * u.Angstrom,
                    zero_point_vega=3670.94 * u.Jy,
                ),
                'NOT/ALFOSC.Bes_R': FilterInfo(
                    name='NOT/ALFOSC.Bes_R',
                    wavelength=6329.59 * u.Angstrom,
                    lambda_pivot=6396.64 * u.Angstrom,
                    weff=1279.53 * u.Angstrom,
                    zero_point_vega=3085.76 * u.Jy,
                ),
                'NOT/ALFOSC.Bes_I': FilterInfo(
                    name='NOT/ALFOSC.Bes_I',
                    wavelength=8466.07 * u.Angstrom,
                    lambda_pivot=8559.60 * u.Angstrom,
                    weff=2578.97 * u.Angstrom,
                    zero_point_vega=2338.38 * u.Jy,
                ),
            }
        ),
    },
    
    # ==================== Generic Systems ====================
    'Generic': {
        'Cousins': InstrumentFilterLibrary(
            telescope='Generic',
            instrument='Cousins',
            filters={
                'Cousins.U': FilterInfo(
                    name='Cousins.U',
                    wavelength=3600 * u.Angstrom,
                    lambda_pivot=3600 * u.Angstrom,
                    weff=580 * u.Angstrom,
                    zero_point_vega=1790 * u.Jy,
                ),
                'Cousins.B': FilterInfo(
                    name='Cousins.B',
                    wavelength=4400 * u.Angstrom,
                    lambda_pivot=4400 * u.Angstrom,
                    weff=980 * u.Angstrom,
                    zero_point_vega=4063 * u.Jy,
                ),
                'Cousins.V': FilterInfo(
                    name='Cousins.V',
                    wavelength=5500 * u.Angstrom,
                    lambda_pivot=5500 * u.Angstrom,
                    weff=890 * u.Angstrom,
                    zero_point_vega=3640 * u.Jy,
                ),
                'Cousins.R': FilterInfo(
                    name='Cousins.R',
                    wavelength=6400 * u.Angstrom,
                    lambda_pivot=6400 * u.Angstrom,
                    weff=1580 * u.Angstrom,
                    zero_point_vega=3060 * u.Jy,
                ),
                'Cousins.Rc': FilterInfo(
                    name='Cousins.Rc',
                    wavelength=6410 * u.Angstrom,
                    lambda_pivot=6410 * u.Angstrom,
                    weff=1580 * u.Angstrom,
                    zero_point_vega=2930 * u.Jy,
                ),
                'Cousins.I': FilterInfo(
                    name='Cousins.I',
                    wavelength=7980 * u.Angstrom,
                    lambda_pivot=7980 * u.Angstrom,
                    weff=1540 * u.Angstrom,
                    zero_point_vega=2249 * u.Jy,
                ),
                'Cousins.Ic': FilterInfo(
                    name='Cousins.Ic',
                    wavelength=7980 * u.Angstrom,
                    lambda_pivot=7980 * u.Angstrom,
                    weff=1540 * u.Angstrom,
                    zero_point_vega=2106 * u.Jy,
                ),
            }
        ),
        'Johnson': InstrumentFilterLibrary(
            telescope='Generic',
            instrument='Johnson',
            filters={
                'Johnson.U': FilterInfo(
                    name='Johnson.U',
                    wavelength=3600 * u.Angstrom,
                    lambda_pivot=3600 * u.Angstrom,
                    weff=620 * u.Angstrom,
                    zero_point_vega=1800 * u.Jy,
                ),
                'Johnson.B': FilterInfo(
                    name='Johnson.B',
                    wavelength=4400 * u.Angstrom,
                    lambda_pivot=4400 * u.Angstrom,
                    weff=980 * u.Angstrom,
                    zero_point_vega=4260 * u.Jy,
                ),
                'Johnson.V': FilterInfo(
                    name='Johnson.V',
                    wavelength=5500 * u.Angstrom,
                    lambda_pivot=5500 * u.Angstrom,
                    weff=890 * u.Angstrom,
                    zero_point_vega=3640 * u.Jy,
                ),
                'Johnson.R': FilterInfo(
                    name='Johnson.R',
                    wavelength=6450 * u.Angstrom,
                    lambda_pivot=6450 * u.Angstrom,
                    weff=1580 * u.Angstrom,
                    zero_point_vega=3080 * u.Jy,
                ),
                'Johnson.I': FilterInfo(
                    name='Johnson.I',
                    wavelength=8750 * u.Angstrom,
                    lambda_pivot=8750 * u.Angstrom,
                    weff=1520 * u.Angstrom,
                    zero_point_vega=2550 * u.Jy,
                ),
            }
        ),
        'SDSS': InstrumentFilterLibrary(
            telescope='Generic',
            instrument='SDSS',
            filters={
                'SDSS.u': FilterInfo(
                    name='SDSS.u',
                    wavelength=3540 * u.Angstrom,
                    lambda_pivot=3540 * u.Angstrom,
                    weff=550 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
                'SDSS.g': FilterInfo(
                    name='SDSS.g',
                    wavelength=4770 * u.Angstrom,
                    lambda_pivot=4710 * u.Angstrom,
                    weff=1280 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
                'SDSS.r': FilterInfo(
                    name='SDSS.r',
                    wavelength=6230 * u.Angstrom,
                    lambda_pivot=6173 * u.Angstrom,
                    weff=1400 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
                'SDSS.i': FilterInfo(
                    name='SDSS.i',
                    wavelength=7630 * u.Angstrom,
                    lambda_pivot=7500 * u.Angstrom,
                    weff=1540 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
                'SDSS.z': FilterInfo(
                    name='SDSS.z',
                    wavelength=9130 * u.Angstrom,
                    lambda_pivot=8985 * u.Angstrom,
                    weff=1070 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
            }
        ),
        'PanSTARRS': InstrumentFilterLibrary(
            telescope='Generic',
            instrument='PanSTARRS',
            filters={
                'PanSTARRS.g': FilterInfo(
                    name='PanSTARRS.g',
                    wavelength=4820 * u.Angstrom,
                    lambda_pivot=4820 * u.Angstrom,
                    weff=1380 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
                'PanSTARRS.r': FilterInfo(
                    name='PanSTARRS.r',
                    wavelength=6210 * u.Angstrom,
                    lambda_pivot=6210 * u.Angstrom,
                    weff=1370 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
                'PanSTARRS.i': FilterInfo(
                    name='PanSTARRS.i',
                    wavelength=7500 * u.Angstrom,
                    lambda_pivot=7500 * u.Angstrom,
                    weff=1490 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
                'PanSTARRS.z': FilterInfo(
                    name='PanSTARRS.z',
                    wavelength=8700 * u.Angstrom,
                    lambda_pivot=8700 * u.Angstrom,
                    weff=980 * u.Angstrom,
                    zero_point_vega=3631 * u.Jy,
                ),
            }
        ),
        '2MASS': InstrumentFilterLibrary(
            telescope='Generic',
            instrument='2MASS',
            filters={
                '2MASS.J': FilterInfo(
                    name='2MASS.J',
                    wavelength=12350 * u.Angstrom,
                    lambda_pivot=12350 * u.Angstrom,
                    weff=1624 * u.Angstrom,
                    zero_point_vega=1594 * u.Jy,
                ),
                '2MASS.H': FilterInfo(
                    name='2MASS.H',
                    wavelength=16620 * u.Angstrom,
                    lambda_pivot=16620 * u.Angstrom,
                    weff=2509 * u.Angstrom,
                    zero_point_vega=1024 * u.Jy,
                ),
                '2MASS.Ks': FilterInfo(
                    name='2MASS.Ks',
                    wavelength=21590 * u.Angstrom,
                    lambda_pivot=21590 * u.Angstrom,
                    weff=2618 * u.Angstrom,
                    zero_point_vega=666.7 * u.Jy,
                ),
            }
        ),
    },
    
    # ==================== Swift ====================
    'Swift': {
        'UVOT': InstrumentFilterLibrary(
            telescope='Swift',
            instrument='UVOT',
            filters={
                'Swift/UVOT.UVW2': FilterInfo(
                    name='Swift/UVOT.UVW2',
                    wavelength=2083.95 * u.Angstrom,
                    lambda_pivot=2054.61 * u.Angstrom,
                    weff=667.73 * u.Angstrom,
                    zero_point_vega=755.14 * u.Jy,
                ),
                'Swift/UVOT.UVM2': FilterInfo(
                    name='Swift/UVOT.UVM2',
                    wavelength=2245.03 * u.Angstrom,
                    lambda_pivot=2246.43 * u.Angstrom,
                    weff=533.85 * u.Angstrom,
                    zero_point_vega=787.63 * u.Jy,
                ),
                'Swift/UVOT.UVW1': FilterInfo(
                    name='Swift/UVOT.UVW1',
                    wavelength=2681.67 * u.Angstrom,
                    lambda_pivot=2580.74 * u.Angstrom,
                    weff=801.92 * u.Angstrom,
                    zero_point_vega=921.00 * u.Jy,
                ),
                'Swift/UVOT.U': FilterInfo(
                    name='Swift/UVOT.U',
                    wavelength=3520.88 * u.Angstrom,
                    lambda_pivot=3467.05 * u.Angstrom,
                    weff=662.50 * u.Angstrom,
                    zero_point_vega=1457.11 * u.Jy,
                ),
                'Swift/UVOT.B': FilterInfo(
                    name='Swift/UVOT.B',
                    wavelength=4345.28 * u.Angstrom,
                    lambda_pivot=4349.56 * u.Angstrom,
                    weff=866.22 * u.Angstrom,
                    zero_point_vega=4088.50 * u.Jy,
                ),
                'Swift/UVOT.V': FilterInfo(
                    name='Swift/UVOT.V',
                    wavelength=5411.45 * u.Angstrom,
                    lambda_pivot=5425.33 * u.Angstrom,
                    weff=655.67 * u.Angstrom,
                    zero_point_vega=3657.87 * u.Jy,
                ),
                'Swift/UVOT.white': FilterInfo(
                    name='Swift/UVOT.white',
                    wavelength=3875.62 * u.Angstrom,
                    lambda_pivot=3325.21 * u.Angstrom,
                    weff=3548.07 * u.Angstrom,
                    zero_point_vega=1678.00 * u.Jy,
                ),
            }
        ),
    },
}


# Create backward-compatible flat FILTERS dictionary
# for accessing filters like: FILTERS['NOT/ALFOSC.Bes_R']
FILTERS = {}
for telescope, instruments in TELESCOPES.items():
    for instrument, lib in instruments.items():
        FILTERS.update(lib.filters)

# Create dot-accessor wrapper for pythonic access
# Usage: filter.swift.uvot.white or filter['Swift']['UVOT']['Swift/UVOT.white']
filter = DotAccessor(TELESCOPES)


def magnitude_to_flux(magnitude: Union[float, Quantity],
                      filter_info: Union[FilterInfo, str],
                      system: Literal['AB', 'Vega'] = 'Vega',
                      error: Union[None, float, Quantity] = None,
                      flux_type: Literal['fnu', 'flam', 'F'] = 'fnu',
                      unit: Union[str, u.Unit] = None) -> Quantity:
    """星等到通量的函数入口 / Convert a magnitude using a named or supplied filter.
    
    Parameters
    ----------
    magnitude : float or Quantity
        星等值，Quantity 只取 value，不自动转换单位。
        Magnitude value; Quantity supplies value without unit conversion.
    filter_info : FilterInfo or str
        滤光片对象或 FILTERS 中大小写敏感的完整名称。
        Filter object or an exact case-sensitive full name in FILTERS.
    system : {'AB', 'Vega'}
        星等系统，默认 Vega / Photometric system; default Vega.
    error : float, Quantity or None
        可选星等标准误差；Quantity 只取 value。
        Optional magnitude standard error; Quantity supplies value only.
    flux_type : {'fnu', 'flam', 'F'}
        分别为频率密度、pivot 处波长密度、有效带宽通量估计；默认 fnu。
        Frequency density, pivot wavelength density, or effective-width band-flux
        estimate; default fnu. F does not integrate an arbitrary source spectrum.
    unit : str, Unit or None
        输出单位，None 时分别为 erg/(cm2 s Hz)、erg/(cm2 s Angstrom)、
        erg/(cm2 s)。须与选择的物理量相容。
        Output unit, defaulting to those cgs units for the selected quantity.
    
    Returns
    -------
    Quantity
        对应通量，若给 error 则 .error 含一阶传播误差。
        Converted flux; supplied magnitude error becomes a first-order .error.

    Raises
    ------
    ValueError
        未知滤光片名称、星等系统或 flux_type；不兼容单位由 Astropy 报错。
        Unknown filter, system or flux type; unit errors propagate from Astropy.

    Notes
    -----
    复用 Magnitude 的转换，未传播零点、带宽和 pivot 的校准误差。
    Delegate to Magnitude, without calibration-error propagation for zero point,
    width or pivot. Filter metadata are stored locally; this function performs no
    network query."""
    # Resolve filter
    if isinstance(filter_info, str):
        if filter_info not in FILTERS:
            raise ValueError(f"Unknown filter: {filter_info}. Available: {list(FILTERS.keys())}")
        filt = FILTERS[filter_info]
    else:
        filt = filter_info
    
    # Create Magnitude with specified system
    mag_qty = Magnitude(magnitude, filt, system=system, error=error)
    
    # Convert based on flux_type
    if flux_type == 'fnu':
        if unit is None:
            unit = 'erg/(cm2 s Hz)'
        return mag_qty.to_fnu(unit=unit)
    elif flux_type == 'flam':
        if unit is None:
            unit = 'erg/(cm2 s Angstrom)'
        return mag_qty.to_flam(unit=unit)
    elif flux_type == 'F':
        if unit is None:
            unit = 'erg/(cm2 s)'
        return mag_qty.to_flux(unit=unit)
    else:
        raise ValueError(f"Unknown flux_type: {flux_type}. Must be 'fnu', 'flam', or 'F'")


# Export public API
__all__ = [
    'FilterInfo',
    'InstrumentFilterLibrary',
    'DotAccessor',
    'Magnitude',
    'TELESCOPES',
    'FILTERS',
    'filter',  # Pythonic dot-accessor
    'magnitude_to_flux',
]

# Backward compatibility alias
MagnitudeQuantity = Magnitude
