"""
Swift BAT Observation Module
Swift BAT 观测模块

Author: Xinxiang Sun
Date: 2026-02-06

This module provides a class for handling Swift BAT observations,
including spacecraft position history, pointing, field of view visualization,
and sky map plotting with support for ligo.skymap style projections.

该模块提供处理 Swift BAT 观测的类，包括航天器位置历史、指向、
视场可视化和支持 ligo.skymap 风格投影的天图绘制。
"""

from ...core.time import Time
from astropy.coordinates import SkyCoord, get_sun, get_body
import astropy.units as u
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob
import warnings

# Local attitude module (inspired by batanalysis)
from .attitude import Attitude

# swifttools for querying observations by time
try:
    from swifttools.swift_too import ObsQuery as SwiftObsQuery
    HAS_SWIFTTOOLS = True
except ImportError:
    HAS_SWIFTTOOLS = False

# gdt-swift imports (optional, for compatibility)
try:
    from gdt.missions.swift.bat.poshist import BatSao
    from gdt.missions.swift.bat.finders import BatAuxiliaryFtp
    from gdt.missions.swift.time import Time as SwiftTime
    from gdt.core.data_primitives import Gti
    from gdt.core.plot.sky import EquatorialPlot
    from gdt.core.plot.plot import SkyCircle, SkyPoints
    HAS_GDT_SWIFT = True
except ImportError:
    HAS_GDT_SWIFT = False
    warnings.warn("gdt-swift not available. Some features may be limited.")

# Optional: ligo.skymap for advanced projections
try:
    import ligo.skymap.plot
    HAS_LIGO_SKYMAP = True
except ImportError:
    HAS_LIGO_SKYMAP = False

# Optional: healpy for healpix operations
try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False


__all__ = ['BATObservation']


class BATObservation:
    """
    A class to handle Swift BAT observations.
    处理 Swift BAT 观测的类。
    
    Supports attitude data from:
    - *.sat files (spacecraft attitude)
    - *.mkf files (filter file)
    - *.sao files (SAO position history)
    
    支持的姿态文件格式：
    - *.sat 文件（航天器姿态）
    - *.mkf 文件（滤波文件）
    - *.sao 文件（SAO 位置历史）
    
    Examples
    --------
    >>> from jinwu.swift.bat import BATObservation
    >>> # From SAT file
    >>> obs = BATObservation(
    ...     srcname='EP260119a',
    ...     ra=158.08, dec=65.50,
    ...     utc_start='2026-01-19T00:13:29',
    ...     attitude_file='sw00019040001sat.fits',
    ...     filepath='/path/to/data'
    ... )
    >>> obs.skymap()
    
    >>> # From SAO file (legacy)
    >>> obs = BATObservation(
    ...     srcname='EP260119a',
    ...     ra=158.08, dec=65.50,
    ...     utc_start='2026-01-19T00:13:29',
    ...     sao_file='sw00019040001sao.fits'
    ... )
    """

    def __init__(self,
                 srcname: str = None,
                 ra: float = None, dec: float = None,
                 utc_start: str | Time = None,
                 attitude_file: str = None,
                 sao_file: str = None,
                 obsid: str = None,
                 filepath: str | Path = None,
                 tstart_offset: float = -400,
                 tstop_offset: float = 400,
                 auto_download: bool = False):
        """
        Initialize a BATObservation instance.
        初始化 BATObservation 实例。

        Parameters
        ----------
        srcname : str
            The source name. 源名称。
        ra : float
            Right Ascension of the source (degrees). 源的赤经（度）。
        dec : float
            Declination of the source (degrees). 源的赤纬（度）。
        utc_start : str or Time
            The observation time in UTC (ISO format string or astropy Time object).
            观测的 UTC 时间（ISO 格式字符串或 astropy Time 对象）。
        attitude_file : str, optional
            Path to attitude file (*.sat, *.mkf, or *.sao).
            姿态文件路径（*.sat、*.mkf 或 *.sao）。
        sao_file : str, optional
            Path to SAO file (legacy, same as attitude_file).
            SAO 文件路径（兼容旧版，与 attitude_file 相同）。
        obsid : str, optional
            Swift observation ID (8 digits). 用于下载文件。
        filepath : str or Path
            The file path for output. 输出路径。
        tstart_offset : float
            Offset from srctime for start time (seconds). 起始时间偏移（秒）。
        tstop_offset : float
            Offset from srctime for stop time (seconds). 结束时间偏移（秒）。
        auto_download : bool
            Whether to automatically download files. 是否自动下载文件。
        """
        self._srcname = srcname
        self._utc_TSTART = utc_start
        self._ra = ra
        self._dec = dec
        self._obsid = obsid
        self._attitude_file = attitude_file or sao_file  # Support both
        self._sao_file = sao_file  # Legacy compatibility
        self._filepath = Path(filepath) if filepath else Path('.')
        self._tstart_offset = tstart_offset
        self._tstop_offset = tstop_offset
        self._auto_download = auto_download
        
        # Internal state
        self._attitude = None  # Attitude object
        self._sao = None       # Legacy BatSao (for gdt-swift compatibility)
        self._frame = None
        self._states = None
        self._one_frame = None
        
        # Initialize if all required parameters are provided
        if all([utc_start is not None, ra is not None, dec is not None]):
            self._update_time_and_coordinates()
            self._create_directory_structure()
            # Always try to initialize observation (attitude file is required)
            # 始终尝试初始化观测（姿态文件是必须的）
            self._initialize_observation()

    def _update_time_and_coordinates(self):
        """Update time and coordinate-related properties.
        更新时间和坐标相关属性。
        """
        if not all([self._utc_TSTART is not None, self._ra is not None, self._dec is not None]):
            raise ValueError("UTC start time, RA, and DEC must be set. "
                             "必须设置 UTC 起始时间、RA 和 DEC。")
        # Support both string and Time object for utc_start
        # 支持字符串和 Time 对象作为 utc_start
        if isinstance(self._utc_TSTART, Time):
            self.srctime = Time(self._utc_TSTART, scale='utc', precision=9)
        else:
            self.srctime = Time(self._utc_TSTART, scale='utc', precision=9)
        self.coord = SkyCoord(self._ra, self._dec, frame='icrs', unit='deg')
        self.isot = self.srctime.isot

    def _create_directory_structure(self):
        """Create the directory structure (simplified for BAT).
        创建目录结构（BAT 简化版）。
        
        Unlike GBMObservation, BAT uses a simple flat directory structure.
        与 GBMObservation 不同，BAT 使用简单的扁平目录结构。
        """
        if not self._filepath:
            raise ValueError("File path must be set. 必须设置文件路径。")
        
        self.yr = self.srctime.datetime.year
        self.month = f"{self.srctime.datetime.month:02d}"
        self.day = f"{self.srctime.datetime.day:02d}"
        
        # Simple directory: just use filepath directly
        # 简单目录：直接使用 filepath
        self._filepath.mkdir(parents=True, exist_ok=True)
        self.datadir = self._filepath
        self.batdir = self._filepath

    def _initialize_observation(self):
        """Initialize observation-related properties from attitude file.
        从姿态文件初始化观测相关属性。
        """
        # Determine attitude file path
        attitude_path = self._attitude_file
        
        if not attitude_path:
            # Try to find existing attitude files in datadir
            for pattern in ['sw*sat.fits*', 'sw*.mkf*', 'sw*sao.fits*']:
                files = glob(str(self.datadir / pattern))
                if files:
                    attitude_path = files[0]
                    print(f"✓ Found existing attitude file: {attitude_path}")
                    break
        
        # If still no file, try to get obsid and download
        if not attitude_path:
            # If no obsid, try to query it using swifttools
            if not self._obsid:
                self._obsid = self._query_obsid_by_time()
            
            if self._obsid:
                print(f"Attempting download with obsid={self._obsid}...")
                try:
                    attitude_path = self._download_sao()
                except Exception as e:
                    raise FileNotFoundError(
                        f"Failed to download attitude file: {e}\n"
                        f"Provide attitude_file directly or check obsid={self._obsid}.\n"
                        f"下载姿态文件失败：{e}。请直接提供 attitude_file 或检查 obsid。"
                    )
            else:
                raise FileNotFoundError(
                    "Attitude file is required but not found.\n"
                    "Options:\n"
                    "  1. Provide attitude_file='path/to/*.sat' or '*.mkf' or '*.sao'\n"
                    "  2. Provide obsid='00012345678' for auto-download\n"
                    "  3. Ensure swifttools is installed for auto-query by time\n"
                    "姿态文件是必须的但未找到。请提供 attitude_file 或 obsid。"
                )
        
        self._attitude_file = attitude_path
        
        # Determine file type and load accordingly
        att_path_lower = str(attitude_path).lower()
        
        if 'sao' in att_path_lower:
            # Use gdt-swift BatSao for *.sao files (if available)
            self._load_from_sao(attitude_path)
        else:
            # Use local Attitude class for *.sat and *.mkf files
            self._load_from_attitude(attitude_path)
        
        # Check visibility
        self.visiblecheck = self.check_visibility()
        self.gticheck = self.check_gti()
        self.observation = self.visiblecheck and self.gticheck
        
        if not self.observation:
            reason = "GTI" if not self.gticheck else "visibility"
            warnings.warn(f"Source {self._srcname} not observable due to {reason}. "
                         f"源 {self._srcname} 不可观测，原因是 {reason}。", stacklevel=2)

    def _load_from_attitude(self, attitude_path):
        """Load pointing info from Attitude file (*.sat or *.mkf).
        从 Attitude 文件加载指向信息（*.sat 或 *.mkf）。
        """
        self._attitude = Attitude.from_file(attitude_path)
        
        # Get pointing at source time
        try:
            ra_pnt, dec_pnt = self._attitude.pointing_at(self.srctime)
            self.bat_ra_pnt = ra_pnt
            self.bat_dec_pnt = dec_pnt
        except Exception as e:
            # Use middle of the attitude data
            mid_idx = len(self._attitude.time) // 2
            self.bat_ra_pnt = self._attitude.ra[mid_idx].value
            self.bat_dec_pnt = self._attitude.dec[mid_idx].value
            warnings.warn(f"Could not interpolate at srctime: {e}. Using mid-point.", stacklevel=2)
        
        self.pointing = SkyCoord(self.bat_ra_pnt, self.bat_dec_pnt, unit='deg', frame='icrs')
        
        # Create a minimal spacecraft frame for compatibility
        self._one_frame = None  # Attitude class doesn't use BatFrame
        self._sao = None

    def _load_from_sao(self, sao_path):
        """Load pointing info from SAO file using gdt-swift.
        使用 gdt-swift 从 SAO 文件加载指向信息。
        """
        if BatSao is None:
            # Fall back to Attitude class if gdt-swift not available
            warnings.warn("gdt-swift not available, trying Attitude class for SAO file", stacklevel=2)
            self._load_from_attitude(sao_path)
            return
        
        self._sao = BatSao.open(sao_path)
        self._frame = self._sao.get_spacecraft_frame()
        self._states = self._sao.get_spacecraft_states()
        
        # Get frame at source time
        try:
            self._one_frame = self._frame.at(self.srctime)
        except Exception as e:
            # If srctime is outside SAO range, use closest time
            frame_times = self._frame.obstime
            if self.srctime < frame_times.min():
                self._one_frame = self._frame[0]
                warnings.warn(f"srctime before SAO range, using first frame", stacklevel=2)
            elif self.srctime > frame_times.max():
                self._one_frame = self._frame[-1]
                warnings.warn(f"srctime after SAO range, using last frame", stacklevel=2)
            else:
                raise e
        
        # Get pointing info
        self.bat_ra_pnt, self.bat_dec_pnt = self._sao.get_bat_pointing()
        self.pointing = SkyCoord(self.bat_ra_pnt, self.bat_dec_pnt, unit='deg', frame='icrs')

    def _query_obsid_by_time(self) -> str:
        """Query obsid from Swift timeline using swifttools.
        使用 swifttools 根据时间查询 obsid。
        
        Returns
        -------
        str or None
            The observation ID if found, None otherwise.
        """
        if not HAS_SWIFTTOOLS:
            print("⚠ swifttools not available. Cannot query obsid by time.")
            print("  Install with: pip install swifttools")
            return None
        
        try:
            from datetime import timedelta
            
            print(f"Querying Swift observation at {self.srctime.isot}...")
            
            # Query Swift As-Flown Timeline by time range (more reliable)
            # Use a small time window around srctime
            t_begin = self.srctime.datetime - timedelta(seconds=60)
            t_end = self.srctime.datetime + timedelta(seconds=60)
            query = SwiftObsQuery(begin=t_begin, end=t_end)
            
            if query.status.status == "Rejected":
                print(f"⚠ Query rejected by server.")
                print("  This may happen if data is not yet available in the archive.")
                return None
            elif query.status.status != "Accepted":
                print(f"⚠ Query status: {query.status.status}")
                return None
            
            # Check if Swift was observing (not slewing)
            if len(query) == 0:
                print("⚠ No observation found at this time.")
                print("  Possible reasons:")
                print("  - Swift was slewing between targets")
                print("  - Data not yet available in archive")
                return None
            
            # Get the observation that contains srctime
            # 找到包含目标时间的观测
            obs = None
            for entry in query:
                if entry.begin <= self.srctime.datetime <= entry.end:
                    obs = entry
                    break
            
            # If no exact match, use the closest one
            if obs is None:
                obs = query[0]
                print(f"  Note: srctime not within any observation, using closest one.")
            
            obsid = obs.obsnum
            print(f"✓ Found obsid: {obsid}")
            print(f"  Target: {obs.targname}")
            print(f"  Obs time: {obs.begin} - {obs.end}")
            if obs.ra is not None and obs.dec is not None:
                print(f"  Pointing: RA={obs.ra:.4f}, Dec={obs.dec:.4f}")
            
            return obsid
            
        except Exception as e:
            print(f"⚠ Failed to query obsid: {e}")
            print("  Check network connection or provide obsid manually.")
            return None

    def _download_sao(self) -> str:
        """Download SAO file using HTTPS (preferred) or FTP (fallback).
        使用 HTTPS（首选）或 FTP（备用）下载 SAO 文件。
        """
        if not self._obsid:
            raise ValueError("obsid required for download. 下载需要 obsid。")
        
        # Try HTTPS first (more reliable)
        try:
            return self._download_sao_https()
        except Exception as e:
            print(f"HTTPS download failed: {e}")
            print("Trying FTP fallback...")
        
        # Fallback to gdt-swift FTP
        if BatAuxiliaryFtp is None:
            raise ImportError(
                "gdt-swift required for FTP download. "
                "Install with: pip install astro-gdt-swift"
            )
        
        date_str = f"{self.yr}-{self.month}"
        print(f"Downloading SAO file via FTP for obsid={self._obsid}, date={date_str}...")
        
        try:
            finder = BatAuxiliaryFtp(self._obsid, date_str)
            finder.get_sao(str(self.datadir))
            sao_files = finder.ls_sao()
            if sao_files:
                sao_path = str(self.datadir / sao_files[0])
                print(f"✓ Downloaded: {sao_path}")
                return sao_path
        except Exception as e:
            print(f"FTP download failed: {e}")
        
        raise FileNotFoundError(f"Failed to download SAO for obsid={self._obsid}")

    def _download_sao_https(self) -> str:
        """Download SAO file using HTTPS from HEASARC.
        使用 HTTPS 从 HEASARC 下载 SAO 文件。
        """
        import urllib.request
        import urllib.error
        
        # Construct HEASARC URL
        # Pattern: https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/YYYY_MM/OBSID/auxil/
        base_url = "https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs"
        date_str = f"{self.yr}_{self.month}"
        auxil_url = f"{base_url}/{date_str}/{self._obsid}/auxil/"
        
        print(f"Downloading SAO file via HTTPS for obsid={self._obsid}...")
        print(f"URL: {auxil_url}")
        
        # Possible SAO file names
        sao_names = [
            f"sw{self._obsid}sao.fits.gz",
            f"sw{self._obsid}sao.fits"
        ]
        
        for sao_name in sao_names:
            file_url = auxil_url + sao_name
            local_path = self.datadir / sao_name
            
            try:
                print(f"  Trying: {sao_name}...")
                urllib.request.urlretrieve(file_url, str(local_path))
                print(f"✓ Downloaded: {local_path}")
                return str(local_path)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue  # Try next filename
                raise
            except Exception:
                continue
        
        raise FileNotFoundError(f"SAO file not found at {auxil_url}")

    # ==================== Properties ====================
    
    @property
    def srcname(self):
        return self._srcname

    @srcname.setter
    def srcname(self, value: str):
        self._srcname = value
        if self._filepath:
            self._create_directory_structure()

    @property
    def frame(self):
        """Spacecraft frame from SAO file."""
        return self._frame
    
    @property
    def states(self):
        """Spacecraft states (sun, saa flags) from SAO file."""
        return self._states
    
    @property
    def attitude(self):
        """Attitude object for *.sat or *.mkf files."""
        return self._attitude

    @property
    def tstart(self):
        """Start time relative to srctime."""
        return self.srctime + self._tstart_offset * u.s

    @property
    def tstop(self):
        """Stop time relative to srctime."""
        return self.srctime + self._tstop_offset * u.s

    # ==================== Visibility Checks ====================
    
    def check_visibility(self) -> bool:
        """Check if the source is visible (not Earth-occulted).
        检查源是否可见（未被地球遮挡）。
        """
        # If using gdt-swift frame
        if self._one_frame is not None:
            return self._one_frame.location_visible(self.coord)
        
        # If using Attitude class, check by offset angle only
        # (Attitude doesn't have Earth occultation info)
        if self._attitude is not None:
            offset = self.source_offset_angle()
            return offset < 70.0  # Assume visible if within ~70 degrees
        
        return True  # Assume visible if no attitude info

    def check_gti(self) -> bool:
        """Check if the source time is within Good Time Intervals.
        检查源时间是否在 GTI 范围内。
        """
        # If using Attitude class
        if self._attitude is not None:
            try:
                in_saa = self._attitude.in_saa_at(self.srctime)
                return not in_saa  # Good if NOT in SAA
            except Exception:
                # If can't determine SAA status, assume good
                return True
        
        # If using gdt-swift BatSao
        if self._states is None:
            return True  # Assume good if no state info
        
        # Create GTI from SAA flag (good when not in SAA)
        times = self._states['time'].value
        good_mask = ~self._states['saa'].value
        gti = Gti.from_boolean_mask(times, good_mask)
        
        # Convert srctime to Swift MET
        met_time = SwiftTime(self.srctime).swift
        return gti.contains(met_time)

    def source_offset_angle(self) -> float:
        """Calculate angular offset between source and BAT pointing.
        计算源与 BAT 指向之间的角度偏移。
        
        Returns
        -------
        float
            Offset angle in degrees. 偏移角度（度）。
        """
        return self.coord.separation(self.pointing).deg

    def in_coded_fov(self, threshold: float = 50.0) -> bool:
        """Check if source is within BAT coded field of view.
        检查源是否在 BAT 编码视场内。
        
        Parameters
        ----------
        threshold : float
            Half-width of coded FOV in degrees. BAT half-coded ~50°.
            编码视场半宽度（度）。BAT 半编码约 50°。
        
        Returns
        -------
        bool
            True if source is within coded FOV.
        """
        offset = self.source_offset_angle()
        return offset < threshold

    # ==================== FOV Geometry ====================
    
    @staticmethod
    def get_bat_fov_polygon(fov_type: str = 'half_coded'):
        """Get BAT field of view boundary polygon vertices.
        获取 BAT 视场边界多边形顶点。
        
        Parameters
        ----------
        fov_type : str
            'fully_coded' - Fully coded FOV (~60° x 60°)
            'half_coded' - Half coded FOV (~100° x 60°)
            'partial_10' - 10% partial coding (~120° x 70°)
        
        Returns
        -------
        list
            List of (delta_ra, delta_dec) vertices in degrees.
        """
        if fov_type == 'fully_coded':
            hw, hh = 30, 30
            corner_cut = 10
        elif fov_type == 'half_coded':
            hw, hh = 50, 30
            corner_cut = 15
        elif fov_type == 'partial_10':
            hw, hh = 60, 35
            corner_cut = 20
        else:
            hw, hh = 50, 30
            corner_cut = 15
        
        # Octagonal vertices (clockwise)
        vertices = [
            (-hw + corner_cut, hh),
            (hw - corner_cut, hh),
            (hw, hh - corner_cut),
            (hw, -hh + corner_cut),
            (hw - corner_cut, -hh),
            (-hw + corner_cut, -hh),
            (-hw, -hh + corner_cut),
            (-hw, hh - corner_cut),
        ]
        return vertices

    def fov_to_sky_coords(self, ra_point: float, dec_point: float, 
                          fov_type: str = 'half_coded', roll: float = 0,
                          n_interpolate: int = 200) -> tuple:
        """Convert FOV polygon to sky coordinates.
        将 FOV 多边形转换为天球坐标。
        
        Parameters
        ----------
        ra_point : float
            Pointing RA in degrees.
        dec_point : float
            Pointing Dec in degrees.
        fov_type : str
            FOV type ('fully_coded', 'half_coded', 'partial_10')
        roll : float
            Roll angle in degrees.
        n_interpolate : int
            Number of points for smooth boundary.
        
        Returns
        -------
        tuple
            (ra_array, dec_array) in degrees.
        """
        vertices = self.get_bat_fov_polygon(fov_type)
        
        boundary_ra = []
        boundary_dec = []
        
        for dra, ddec in vertices:
            # Apply roll rotation
            if roll != 0:
                roll_rad = np.radians(roll)
                dra_rot = dra * np.cos(roll_rad) - ddec * np.sin(roll_rad)
                ddec_rot = dra * np.sin(roll_rad) + ddec * np.cos(roll_rad)
                dra, ddec = dra_rot, ddec_rot
            
            # Spherical coordinate conversion
            new_dec = dec_point + ddec
            if abs(new_dec) < 89:
                new_ra = ra_point + dra / np.cos(np.radians(new_dec))
            else:
                new_ra = ra_point
            
            new_dec = np.clip(new_dec, -90, 90)
            boundary_ra.append(new_ra)
            boundary_dec.append(new_dec)
        
        # Close polygon
        boundary_ra.append(boundary_ra[0])
        boundary_dec.append(boundary_dec[0])
        
        # Interpolate for smooth boundary
        try:
            from scipy.interpolate import splprep, splev
            tck, _ = splprep([boundary_ra, boundary_dec], s=0, per=True)
            u_new = np.linspace(0, 1, n_interpolate)
            boundary_ra, boundary_dec = splev(u_new, tck)
        except:
            pass  # Use original points if interpolation fails
        
        return np.array(boundary_ra), np.array(boundary_dec)

    # ==================== Sky Map Plotting ====================
    
    def skymap(self, projection: str = 'aitoff', show_fov: bool = True,
               show_sun_moon: bool = True, figsize: tuple = (14, 7),
               save: bool = True, show: bool = True):
        """Generate a sky map with source, BAT pointing, and optional FOV.
        生成包含源、BAT 指向和可选视场的天图。
        
        Parameters
        ----------
        projection : str
            Map projection: 'aitoff', 'mollweide', 'equatorial', or 'ligo' 
            (requires ligo.skymap).
        show_fov : bool
            Whether to show BAT field of view regions.
        show_sun_moon : bool
            Whether to show Sun and Moon positions.
        figsize : tuple
            Figure size (width, height).
        save : bool
            Whether to save the figure.
        show : bool
            Whether to display the figure.
        """
        if projection == 'equatorial':
            self._skymap_equatorial(show_fov, show_sun_moon, save, show)
        elif projection == 'ligo' and HAS_LIGO_SKYMAP:
            self._skymap_ligo(show_fov, show_sun_moon, figsize, save, show)
        else:
            self._skymap_projection(projection, show_fov, show_sun_moon, 
                                   figsize, save, show)

    def _skymap_equatorial(self, show_fov, show_sun_moon, save, show):
        """Sky map using gdt EquatorialPlot (similar to GBMObservation).
        使用 gdt EquatorialPlot 的天图（类似 GBMObservation）。
        """
        eqplot = EquatorialPlot(interactive=True)
        
        # Add spacecraft frame if available
        if self._one_frame is not None:
            eqplot.add_frame(self._one_frame)
        
        eqplot.ax.text(0.02, 0.95, "ICRS", transform=eqplot.ax.transAxes, 
                      fontsize=15, color='red', ha='center')
        
        # Mark source position with star
        src_plot = SkyPoints(
            x=self.coord.icrs.ra.deg,
            y=self.coord.icrs.dec.deg,
            ax=eqplot.ax,
            color='red',
            marker='*',
            s=300,
            edgecolors='darkred',
            linewidths=0.5,
            zorder=10
        )
        
        # Mark BAT pointing
        pnt_plot = SkyPoints(
            x=self.bat_ra_pnt,
            y=self.bat_dec_pnt,
            ax=eqplot.ax,
            color='blue',
            marker='+',
            s=200,
            linewidths=2,
            zorder=9
        )
        
        if show_sun_moon:
            self._add_sun_moon(eqplot.ax, use_radians=False)
        
        # Create legend
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markeredgecolor='darkred', markersize=15, label=f'{self._srcname}'),
            Line2D([0], [0], marker='+', color='blue', markersize=12, 
                   markeredgewidth=2, label='BAT Pointing'),
        ]
        if show_sun_moon:
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                       markeredgecolor='orange', markersize=12, label='Sun'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                       markeredgecolor='gray', markersize=10, label='Moon'),
            ])
        eqplot.ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        eqplot.ax.set_title(f'{self._srcname} BAT Sky Map @ {self.srctime.isot}', fontsize=12)
        
        if save:
            plt.savefig(self.datadir / f"{self._srcname}_bat_skymap.png", dpi=300)
        if show:
            plt.show()

    def _skymap_projection(self, projection, show_fov, show_sun_moon, 
                          figsize, save, show):
        """Sky map with standard matplotlib projections.
        使用标准 matplotlib 投影的天图。
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)
        
        # Convert coordinates for projection (RA: [0,360] -> [-π, π])
        def to_proj(ra, dec):
            ra_rad = np.radians(ra)
            ra_rad = np.where(ra_rad > np.pi, ra_rad - 2*np.pi, ra_rad)
            dec_rad = np.radians(dec)
            return ra_rad, dec_rad
        
        # Plot BAT FOV regions
        if show_fov:
            fov_configs = [
                ('partial_10', 'lightyellow', 0.2, 'BAT >10% Coding'),
                ('half_coded', 'lightblue', 0.3, 'BAT Half Coded'),
                ('fully_coded', 'blue', 0.5, 'BAT Fully Coded'),
            ]
            for fov_type, color, alpha, label in fov_configs:
                ra, dec = self.fov_to_sky_coords(
                    self.bat_ra_pnt, self.bat_dec_pnt, fov_type)
                ra_proj, dec_proj = to_proj(ra, dec)
                ax.fill(ra_proj, dec_proj, color=color, alpha=alpha, label=label)
        
        # Mark source
        src_ra, src_dec = to_proj(self._ra, self._dec)
        ax.scatter([src_ra], [src_dec], c='red', s=300, marker='*',
                  edgecolors='darkred', linewidths=0.5, zorder=10,
                  label=f'{self._srcname}')
        
        # Mark BAT pointing
        pnt_ra, pnt_dec = to_proj(self.bat_ra_pnt, self.bat_dec_pnt)
        ax.scatter([pnt_ra], [pnt_dec], c='blue', s=150, marker='+',
                  linewidths=2, zorder=9, label='BAT Pointing')
        
        # Sun and Moon
        if show_sun_moon:
            self._add_sun_moon(ax, use_radians=True)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'{self._srcname} BAT Field of View @ {self.srctime.isot}', fontsize=12)
        
        plt.tight_layout()
        if save:
            plt.savefig(self.datadir / f"{self._srcname}_bat_skymap_{projection}.png", dpi=300)
        if show:
            plt.show()

    def _skymap_ligo(self, show_fov, show_sun_moon, figsize, save, show):
        """Sky map using ligo.skymap projections.
        使用 ligo.skymap 投影的天图。
        """
        if not HAS_LIGO_SKYMAP:
            warnings.warn("ligo.skymap not available, falling back to aitoff")
            self._skymap_projection('aitoff', show_fov, show_sun_moon, 
                                   figsize, save, show)
            return
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='astro hours mollweide')
        
        # Plot FOV using astropy coordinates
        if show_fov:
            from astropy.visualization.wcsaxes import SphericalCircle
            # Note: ligo.skymap uses different coordinate handling
            # For now, fall back to basic plotting
            fov_configs = [
                ('partial_10', 'lightyellow', 0.2),
                ('half_coded', 'lightblue', 0.3),
                ('fully_coded', 'blue', 0.5),
            ]
            for fov_type, color, alpha in fov_configs:
                ra, dec = self.fov_to_sky_coords(
                    self.bat_ra_pnt, self.bat_dec_pnt, fov_type)
                # Convert to proper format for ligo.skymap
                ax.fill(ra, dec, color=color, alpha=alpha,
                       transform=ax.get_transform('world'))
        
        # Mark source
        ax.scatter([self._ra], [self._dec], c='red', s=300, marker='*',
                  edgecolors='darkred', transform=ax.get_transform('world'),
                  zorder=10, label=self._srcname)
        
        ax.grid()
        ax.set_title(f'{self._srcname} @ {self.srctime.isot}')
        
        if save:
            plt.savefig(self.datadir / f"{self._srcname}_bat_skymap_ligo.png", dpi=300)
        if show:
            plt.show()

    def _add_sun_moon(self, ax, use_radians=True):
        """Add Sun and Moon markers to the plot.
        在图中添加太阳和月亮标记。
        """
        sun_coord = get_sun(self.srctime)
        moon_coord = get_body('moon', self.srctime)
        
        if use_radians:
            def to_proj(ra, dec):
                ra_rad = np.radians(ra)
                ra_rad = ra_rad - 2*np.pi if ra_rad > np.pi else ra_rad
                return ra_rad, np.radians(dec)
            
            sun_ra, sun_dec = to_proj(sun_coord.ra.deg, sun_coord.dec.deg)
            moon_ra, moon_dec = to_proj(moon_coord.ra.deg, moon_coord.dec.deg)
        else:
            sun_ra, sun_dec = sun_coord.ra.deg, sun_coord.dec.deg
            moon_ra, moon_dec = moon_coord.ra.deg, moon_coord.dec.deg
        
        ax.scatter([sun_ra], [sun_dec], c='yellow', s=400, marker='o',
                  edgecolors='orange', linewidths=2, zorder=8, label='Sun')
        ax.scatter([moon_ra], [moon_dec], c='lightgray', s=250, marker='o',
                  edgecolors='gray', linewidths=2, zorder=7, label='Moon')

    # ==================== Earth Map ====================
    
    def earthmap(self, tstart_offset: float = -1000, tstop_offset: float = 1000,
                 save: bool = True, show: bool = True):
        """Generate Earth trajectory map showing spacecraft orbit.
        生成显示航天器轨道的地球轨迹图。
        
        Parameters
        ----------
        tstart_offset : float
            Start time offset from srctime in seconds.
        tstop_offset : float
            Stop time offset from srctime in seconds.
        save : bool
            Whether to save the figure.
        show : bool
            Whether to display the figure.
        """
        if self._frame is None:
            msg = (
                "earthmap() requires SAO file for spacecraft orbit data. "
                "Attitude files (*.sat, *.mkf) don't contain position info.\n"
                "Options:\n"
                "  1. Provide attitude_file='path/to/*.sao'\n"
                "  2. Set obsid and auto_download=True\n"
                "  3. Use skymap() instead for pointing visualization\n"
                "earthmap() 需要 SAO 文件获取航天器轨道数据。"
            )
            warnings.warn(msg, stacklevel=2)
            return None
        
        try:
            from gdt.missions.fermi.plot import FermiEarthPlot
            from gdt.missions.fermi.gbm.saa import GbmSaa
            
            # Note: Using Fermi's Earth plot as Swift doesn't have its own
            # The SAA region is similar for both missions
            saa = GbmSaa()
            earthplot = FermiEarthPlot(saa)
            
            # Get poshist time range
            frame_times = self._frame.obstime
            tmin = frame_times.min().value
            tmax = frame_times.max().value
            met_time = SwiftTime(self.srctime).swift
            
            # Adjust offsets to stay within data range
            requested_start = met_time + tstart_offset
            requested_stop = met_time + tstop_offset
            
            if requested_start < tmin:
                tstart_offset = tmin - met_time + 10
                print(f"⚠️ Adjusted start offset: {tstart_offset:.1f}")
            if requested_stop > tmax:
                tstop_offset = tmax - met_time - 10
                print(f"⚠️ Adjusted stop offset: {tstop_offset:.1f}")
            
            duration = tstop_offset - tstart_offset
            
            earthplot.add_spacecraft_frame(
                self._frame,
                tstart=self.srctime + tstart_offset * u.s,
                tstop=duration,
                trigtime=self.srctime
            )
            earthplot.standard_title()
            
            if save:
                plt.savefig(self.datadir / f"{self._srcname}_bat_earthmap.png", dpi=300)
            if show:
                plt.show()
                
        except ImportError:
            warnings.warn("Earth map requires gdt-fermi package for plotting")
            # Fallback: simple scatter plot of lat/lon
            self._earthmap_simple(tstart_offset, tstop_offset, save, show)

    def _earthmap_simple(self, tstart_offset, tstop_offset, save, show):
        """Simple Earth map fallback without gdt-fermi.
        不使用 gdt-fermi 的简单地球图。
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get spacecraft positions
        lats = self._frame.earth_location.lat.deg
        lons = self._frame.earth_location.lon.deg
        
        ax.scatter(lons, lats, c='blue', s=1, alpha=0.5, label='Swift Orbit')
        
        # Mark current position
        if self._one_frame is not None:
            curr_lat = self._one_frame.earth_location.lat.deg
            curr_lon = self._one_frame.earth_location.lon.deg
            ax.scatter([curr_lon], [curr_lat], c='red', s=100, marker='*',
                      label=f'{self._srcname} Time')
        
        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.legend()
        ax.set_title(f'Swift Orbit @ {self.srctime.isot}')
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.datadir / f"{self._srcname}_bat_earthmap_simple.png", dpi=300)
        if show:
            plt.show()

    # ==================== Information Methods ====================
    
    def info(self):
        """Print observation information summary.
        打印观测信息摘要。
        """
        print("=" * 60)
        print(f"BAT Observation: {self._srcname}")
        print("=" * 60)
        print(f"Time (UTC):      {self.srctime.isot}")
        print(f"Source RA:       {self._ra:.4f}°")
        print(f"Source Dec:      {self._dec:.4f}°")
        print(f"BAT Pointing RA: {self.bat_ra_pnt:.4f}°")
        print(f"BAT Pointing Dec:{self.bat_dec_pnt:.4f}°")
        print(f"Source Offset:   {self.source_offset_angle():.2f}°")
        print(f"In Coded FOV:    {self.in_coded_fov()}")
        print(f"Visible:         {self.visiblecheck}")
        print(f"In GTI:          {self.gticheck}")
        print(f"Observable:      {self.observation}")
        print("=" * 60)

    def close(self):
        """Close file handles.
        关闭文件句柄。
        """
        if self._sao is not None:
            self._sao.close()
            self._sao = None
        # Attitude class doesn't need explicit close
        self._attitude = None
        return False

    def __repr__(self):
        return (f"BATObservation(srcname='{self._srcname}', "
                f"ra={self._ra}, dec={self._dec}, "
                f"time='{self.srctime.isot}')")
