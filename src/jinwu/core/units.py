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
        """Set defaults and validate inputs."""
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
        """Get zero-point for specified photometric system.
        
        Parameters
        ----------
        system : {'AB', 'Vega'}
            Photometric system
        
        Returns
        -------
        Quantity
            Zero-point flux in Jy
        """
        if system == 'AB':
            return self.zero_point_ab
        elif system == 'Vega':
            return self.zero_point_vega
        else:
            raise ValueError(f"system must be 'AB' or 'Vega', got '{system}'")
    
    def __str__(self):
        return (f"FilterInfo({self.name}: λ={self.wavelength:.1f}, "
                f"Weff={self.weff:.1f}, ZP_Vega={self.zero_point_vega:.2f}, "
                f"ZP_AB={self.zero_point_ab:.2f})")
    
    def __repr__(self):
        return self.__str__()
    
    def __mul__(self, other: Union[float, Quantity]) -> 'Magnitude':
        """Support left multiplication: filter * magnitude (e.g., filter * 20.5)
        
        Examples
        --------
        >>> filt = FILTERS['Swift/UVOT.white']
        >>> mag1 = filt * 20.5              # Creates Magnitude(20.5, filt, system='Vega')
        >>> mag2 = filt * (22 * u.ABmag)    # Creates Magnitude(22.0, filt, system='AB')
        """
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
        """Support right multiplication: magnitude * filter (e.g., 20.5 * filter)
        
        This is called when the left operand doesn't support multiplication with FilterInfo.
        """
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
        """Access filter by name."""
        if key not in self.filters:
            available = list(self.filters.keys())
            raise KeyError(
                f"Filter '{key}' not found in {self.telescope}/{self.instrument}.\n"
                f"Available: {available}"
            )
        return self.filters[key]
    
    def __getattr__(self, name: str) -> FilterInfo:
        """Access filter by attribute name (dot notation)."""
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
        return f"{self.telescope}/{self.instrument} ({len(self.filters)} filters)"
    
    def __repr__(self):
        return self.__str__()
    
    def list_filters(self) -> list:
        """List all available filters in this instrument."""
        return list(self.filters.keys())


class DotAccessor:
    """支持点号访问的包装器，允许 filters.swift.uvot.white 这样的访问方式"""
    
    def __init__(self, data: Dict):
        self._data = data
    
    def __getattr__(self, name: str):
        """支持点号访问"""
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
        """兼容方括号访问"""
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return DotAccessor(value)
            return value
        raise KeyError(f"Key '{key}' not found in {list(self._data.keys())}")
    
    def __mul__(self, other: Union[float, Quantity]) -> 'Magnitude':
        """Support multiplication for terminal FilterInfo
        
        Examples
        --------
        >>> mag = filter.swift.uvot.white * 20.5
        >>> mag = 22 * u.ABmag * filter.swift.uvot.white
        """
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
        """Support right multiplication: magnitude * filter"""
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
        """
        Initialize a Magnitude object.
        
        Parameters
        ----------
        magnitude : float or Quantity
            The magnitude value
        filter_info : FilterInfo
            Filter properties
        system : {'AB', 'Vega'}, optional
            Photometric system (default: 'Vega')
        error : float, Quantity, optional
            Magnitude uncertainty
        """
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
        """
        Convert magnitude to frequency flux density.
        
        The conversion uses: f_ν = ZP_ν × 10^(-m/2.5)
        where ZP_ν is the zero-point for the specified photometric system.
        
        Parameters
        ----------
        unit : str or Unit, optional
            Output unit (default: erg/(cm2 s Hz))
        
        Returns
        -------
        Quantity
            Frequency flux density with uncertainty (if available)
        
        Notes
        -----
        The returned quantity represents the "effective" f_ν at the filter's 
        pivot wavelength, suitable for multi-wavelength SED fitting.
        """
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
        """
        Convert magnitude to wavelength flux density.
        
        Uses the relation: f_λ = f_ν × c / λ_pivot²
        (with energy conservation: λ_pivot² × f_λ = c × f_ν)
        
        Parameters
        ----------
        unit : str or Unit, optional
            Output unit (default: erg/(cm2 s Angstrom))
        
        Returns
        -------
        Quantity
            Wavelength flux density at pivot wavelength with uncertainty (if available)
        
        Notes
        -----
        This represents f_λ evaluated at the pivot wavelength λ_pivot,
        not averaged over the filter bandpass.
        """
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
        """
        Convert magnitude to integrated flux (flux within filter bandwidth).
        
        Calculation: F = f_λ × W_eff
        where W_eff is the effective bandwidth (rectangular equivalent width)
        
        Parameters
        ----------
        unit : str or Unit, optional
            Output unit (default: erg/(cm2 s))
        
        Returns
        -------
        Quantity
            Integrated flux within the filter bandpass with uncertainty (if available)
        
        Notes
        -----
        This is the actual observed total flux from the source through the filter.
        It's calculated by multiplying the wavelength flux density by the 
        effective bandwidth (from the filter transmission curve).
        """
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
        """Convert to Jansky (frequency flux density)."""
        return self.to_fnu(unit='Jy')
    
    def __str__(self):
        error_str = f" ± {self.error:.2f} mag" if self.error is not None else ""
        return f"Magnitude({self.magnitude:.2f}{error_str}, {self.filter_info.name}, {self.system})"
    
    def __repr__(self):
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
    """
    Convert magnitude to flux (convenient functional interface).
    
    Parameters
    ----------
    magnitude : float or Quantity
        Magnitude value
    filter_info : FilterInfo or str
        Filter properties, or name of predefined filter from FILTERS
    system : {'AB', 'Vega'}, optional
        Photometric system (default: 'Vega')
    error : float, Quantity, optional
        Magnitude uncertainty
    flux_type : {'fnu', 'flam', 'F'}, optional
        Type of flux to return:
        - 'fnu': frequency flux density (default)
        - 'flam': wavelength flux density
        - 'F': integrated flux within bandwidth
    unit : str or Unit, optional
        Output unit. If None, uses sensible defaults:
        - 'fnu': 'erg/(cm2 s Hz)'
        - 'flam': 'erg/(cm2 s Angstrom)'
        - 'F': 'erg/(cm2 s)'
    
    Returns
    -------
    Quantity
        Converted flux value with uncertainty (if provided)
    
    Examples
    --------
    >>> from astropy import units as u
    >>> from jinwu.core.units import magnitude_to_flux
    >>> 
    >>> # Using predefined filter with Vega system
    >>> fnu = magnitude_to_flux(20.5, 'NOT/ALFOSC.Bes_R', system='Vega', error=0.1)
    >>> 
    >>> # Using AB system
    >>> fnu_ab = magnitude_to_flux(20.5, 'NOT/ALFOSC.Bes_R', system='AB', error=0.1)
    >>> 
    >>> # Using custom FilterInfo
    >>> filt = FilterInfo(...)
    >>> flam = magnitude_to_flux(20.5, filt, system='Vega', flux_type='flam')
    """
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
