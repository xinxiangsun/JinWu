"""
Swift BAT Attitude Module
Swift BAT 姿态数据模块

Author: Xinxiang Sun
Date: 2026-02-06

This module provides the Attitude class for reading Swift attitude data
from *.sat, *.mkf, or *.sao files.

该模块提供 Attitude 类，用于从 *.sat、*.mkf 或 *.sao 文件读取 Swift 姿态数据。

Reference: batanalysis.attitude by Tyler Parsotan
"""

from pathlib import Path
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d

__all__ = ['Attitude']


class Attitude:
    """
    Swift attitude data from *.sat, *.mkf, or *.sao files.
    从 *.sat、*.mkf 或 *.sao 文件读取的 Swift 姿态数据。
    
    Attributes
    ----------
    time : Quantity
        Time array (MET seconds).
    ra : Quantity
        Right Ascension pointing (degrees).
    dec : Quantity
        Declination pointing (degrees).
    roll : Quantity
        Roll angle (degrees).
    quaternion : ndarray, optional
        Quaternion array [x, y, z, w].
    is_10arcmin_settled : ndarray, optional
        10 arcmin settled flag.
    is_settled : ndarray, optional
        Settled flag.
    in_saa : ndarray, optional
        SAA flag.
    in_safehold : ndarray, optional
        Safehold flag.
    
    Examples
    --------
    >>> from jinwu.swift.bat.attitude import Attitude
    >>> att = Attitude.from_file('sw00019040001sao.fits')
    >>> att.plot()
    >>> ra, dec = att.pointing_at(met_time)
    """

    def __init__(self, time, ra, dec, roll, quaternion=None,
                 is_10arcmin_settled=None, is_settled=None,
                 in_saa=None, in_safehold=None):
        """
        Initialize Attitude instance.
        
        Parameters
        ----------
        time : Quantity
            Time array with units.
        ra : Quantity
            RA array with units.
        dec : Quantity
            Dec array with units.
        roll : Quantity
            Roll array with units.
        quaternion : ndarray, optional
            Quaternion array.
        is_10arcmin_settled : ndarray, optional
        is_settled : ndarray, optional
        in_saa : ndarray, optional
        in_safehold : ndarray, optional
        """
        self.time = time
        self.ra = ra
        self.dec = dec
        self.roll = roll
        self.quaternion = quaternion
        
        self.is_10arcmin_settled = is_10arcmin_settled
        self.is_settled = is_settled
        self.in_saa = in_saa
        self.in_safehold = in_safehold
        
        # Build interpolators for pointing lookup
        self._build_interpolators()

    def _build_interpolators(self):
        """Build interpolation functions for pointing lookup."""
        t = self.time.value
        self._ra_interp = interp1d(t, self.ra.value, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
        self._dec_interp = interp1d(t, self.dec.value, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')
        self._roll_interp = interp1d(t, self.roll.value, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')

    @classmethod
    def from_file(cls, attitude_file):
        """
        Read attitude data from a *.sat, *.mkf, or *.sao file.
        从 *.sat、*.mkf 或 *.sao 文件读取姿态数据。
        
        Parameters
        ----------
        attitude_file : str or Path
            Path to the attitude file.
        
        Returns
        -------
        Attitude
            Attitude instance.
        """
        attitude_file = Path(attitude_file).expanduser().resolve()
        
        if not attitude_file.exists():
            raise FileNotFoundError(f"Attitude file not found: {attitude_file}")
        
        filename = str(attitude_file).lower()
        
        # Read data based on file type
        all_data = {}
        with fits.open(attitude_file) as hdul:
            # Read main data extension
            data = hdul[1].data
            for col in data.columns:
                unit = col.unit if col.unit else None
                all_data[col.name] = u.Quantity(data[col.name], unit)
            
            # Read ACS flags from sat file
            if 'sat' in filename:
                try:
                    acs_data = hdul['ACS_DATA'].data
                    all_data['FLAGS'] = acs_data['FLAGS']
                except (KeyError, IndexError):
                    pass
        
        # Parse based on file type
        if 'sat' in filename:
            return cls._parse_sat(all_data)
        elif 'mkf' in filename:
            return cls._parse_mkf(all_data)
        elif 'sao' in filename:
            return cls._parse_sao(all_data)
        else:
            # Try to auto-detect based on columns
            if 'POINTING' in all_data:
                return cls._parse_sat(all_data)
            elif 'RA' in all_data and 'DEC' in all_data:
                return cls._parse_mkf(all_data)
            elif 'RA_PNT' in all_data:
                return cls._parse_sao(all_data)
            else:
                raise ValueError(f"Unrecognized attitude file format: {attitude_file}")

    @classmethod
    def _parse_sat(cls, all_data):
        """Parse *.sat file data."""
        time = all_data['TIME']
        ra = all_data['POINTING'][:, 0]
        dec = all_data['POINTING'][:, 1]
        roll = all_data['POINTING'][:, 2]
        
        quaternion = all_data.get('QPARAM', None)
        if quaternion is not None:
            quaternion = quaternion.value
        
        flags = all_data.get('FLAGS', None)
        if flags is not None:
            is_10arcmin_settled = flags[:, 0]
            is_settled = flags[:, 1]
            in_saa = flags[:, 2]
            in_safehold = flags[:, 3]
        else:
            is_10arcmin_settled = is_settled = in_saa = in_safehold = None
        
        return cls(time=time, ra=ra, dec=dec, roll=roll,
                   quaternion=quaternion,
                   is_10arcmin_settled=is_10arcmin_settled,
                   is_settled=is_settled, in_saa=in_saa,
                   in_safehold=in_safehold)

    @classmethod
    def _parse_mkf(cls, all_data):
        """Parse *.mkf file data."""
        time = all_data['TIME']
        ra = all_data['RA']
        dec = all_data['DEC']
        roll = all_data['ROLL']
        
        is_10arcmin_settled = all_data.get('TEN_ARCMIN', None)
        is_settled = all_data.get('SETTLED', None)
        in_saa = all_data.get('ACS_SAA', None)
        in_safehold = all_data.get('SAFEHOLD', None)
        
        return cls(time=time, ra=ra, dec=dec, roll=roll,
                   is_10arcmin_settled=is_10arcmin_settled,
                   is_settled=is_settled, in_saa=in_saa,
                   in_safehold=in_safehold)

    @classmethod
    def _parse_sao(cls, all_data):
        """Parse *.sao file data."""
        time = all_data['TIME']
        
        # SAO files have different column names
        if 'RA_PNT' in all_data:
            ra = all_data['RA_PNT']
            dec = all_data['DEC_PNT']
        else:
            ra = all_data.get('RA', all_data.get('POINTING', np.zeros_like(time.value) * u.deg)[:, 0])
            dec = all_data.get('DEC', all_data.get('POINTING', np.zeros_like(time.value) * u.deg)[:, 1])
        
        roll = all_data.get('PA_PNT', all_data.get('ROLL', np.zeros_like(time.value) * u.deg))
        
        quaternion = all_data.get('QUATERNION', None)
        if quaternion is not None:
            quaternion = quaternion.value
        
        in_saa = all_data.get('SAA', None)
        
        return cls(time=time, ra=ra, dec=dec, roll=roll,
                   quaternion=quaternion, in_saa=in_saa)

    # ==================== Properties ====================
    
    @property
    def time_range(self):
        """Return (tmin, tmax) of the attitude data."""
        return self.time.min(), self.time.max()

    @property
    def duration(self):
        """Duration of attitude data in seconds."""
        return (self.time.max() - self.time.min()).to(u.s)

    # ==================== Pointing Methods ====================
    
    def pointing_at(self, met_time):
        """
        Get pointing (RA, Dec) at a specific MET time.
        获取特定 MET 时间的指向 (RA, Dec)。
        
        Parameters
        ----------
        met_time : float or Quantity
            MET time in seconds.
        
        Returns
        -------
        tuple
            (ra, dec) in degrees.
        """
        if hasattr(met_time, 'value'):
            t = met_time.value
        else:
            t = met_time
        
        ra = self._ra_interp(t)
        dec = self._dec_interp(t)
        
        return ra * u.deg, dec * u.deg

    def skycoord_at(self, met_time):
        """
        Get pointing as SkyCoord at a specific MET time.
        获取特定 MET 时间的 SkyCoord 指向。
        """
        ra, dec = self.pointing_at(met_time)
        return SkyCoord(ra, dec, frame='icrs')

    def roll_at(self, met_time):
        """Get roll angle at a specific MET time."""
        if hasattr(met_time, 'value'):
            t = met_time.value
        else:
            t = met_time
        return self._roll_interp(t) * u.deg

    def in_saa_at(self, met_time):
        """Check if in SAA at a specific time."""
        if self.in_saa is None:
            return None
        
        if hasattr(met_time, 'value'):
            t = met_time.value
        else:
            t = met_time
        
        # Find nearest time index
        idx = np.argmin(np.abs(self.time.value - t))
        
        if hasattr(self.in_saa, 'value'):
            return bool(self.in_saa.value[idx])
        return bool(self.in_saa[idx])

    def is_settled_at(self, met_time):
        """Check if settled at a specific time."""
        if self.is_settled is None:
            return None
        
        if hasattr(met_time, 'value'):
            t = met_time.value
        else:
            t = met_time
        
        idx = np.argmin(np.abs(self.time.value - t))
        
        if hasattr(self.is_settled, 'value'):
            return bool(self.is_settled.value[idx])
        return bool(self.is_settled[idx])

    # ==================== Plotting ====================
    
    def plot(self, T0=None, ax=None, show=True):
        """
        Plot RA/Dec/Roll vs time.
        绘制 RA/Dec/Roll 随时间的变化。
        
        Parameters
        ----------
        T0 : float or Quantity, optional
            Reference time (MET). If None, uses minimum time.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        show : bool
            Whether to show the plot.
        
        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if T0 is None:
            t_rel = self.time.min()
        else:
            if not hasattr(T0, 'unit'):
                t_rel = T0 * u.s
            else:
                t_rel = T0
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        
        t_plot = (self.time - t_rel).to(u.s).value
        
        ax.plot(t_plot, self.ra.value, label='RA', alpha=0.8)
        ax.plot(t_plot, self.dec.value, label='Dec', alpha=0.8)
        ax.plot(t_plot, self.roll.value, label='Roll', alpha=0.8)
        
        if T0 is not None:
            ax.axvline(0, ls='--', color='gray', alpha=0.5, label='T0')
        
        ax.legend()
        ax.set_xlabel(f'Time - {t_rel.value:.1f} (s)')
        ax.set_ylabel(f'Pointing ({self.ra.unit})')
        ax.set_title('Swift Attitude')
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax

    def plot_saa(self, T0=None, ax=None, show=True):
        """Plot SAA flag vs time."""
        if self.in_saa is None:
            print("No SAA data available.")
            return None
        
        if T0 is None:
            t_rel = self.time.min()
        else:
            if not hasattr(T0, 'unit'):
                t_rel = T0 * u.s
            else:
                t_rel = T0
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        
        t_plot = (self.time - t_rel).to(u.s).value
        saa_vals = self.in_saa.value if hasattr(self.in_saa, 'value') else self.in_saa
        
        ax.fill_between(t_plot, 0, saa_vals, alpha=0.5, color='red', label='In SAA')
        ax.set_xlabel(f'Time - {t_rel.value:.1f} (s)')
        ax.set_ylabel('SAA Flag')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax

    def __repr__(self):
        tmin, tmax = self.time_range
        return (f"Attitude(time=[{tmin.value:.1f}, {tmax.value:.1f}] {self.time.unit}, "
                f"n_points={len(self.time)})")
