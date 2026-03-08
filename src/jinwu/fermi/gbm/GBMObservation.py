'''
Date: 2025-04-23 11:11:08
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-11-07 12:41:26
FilePath: /research/jinwu/src/jinwu/missions/fermi/gbm/GBMObservation.py
'''
from ...core.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from gdt.missions.fermi.gbm.finders import ContinuousFinder
from gdt.missions.fermi.gbm.detectors import GbmDetectors
from gdt.missions.fermi.gbm.poshist import GbmPosHist
from pathlib import Path
import os
from jinwu.response.gbm import contgbmrsp
from gdt.core.data_primitives import Gti
from glob import glob
import shutil
import matplotlib.pyplot as plt
from gdt.core.plot.sky import EquatorialPlot
from gdt.core.plot.plot import SkyCircle, SkyPoints
from gdt.core import data_path
from gdt.missions.fermi.gbm.localization import GbmHealPix
import numpy as np
from gdt.missions.fermi.plot import FermiEarthPlot
from gdt.missions.fermi.gbm.saa import GbmSaa


class GBMObservation:
    """
    A class to handle GBM (Gamma-ray Burst Monitor) observations.
    GBM（伽马射线暴监测器）观测处理类。
    """

    def __init__(self, 
                 srcname: str = None, 
                 ra: float = None, dec: float = None, 
                 utc_start: str | Time = None, tstart_offset: float = -400, tstop_offset: float = 400,
                 filepath: str | Path = Path('/Users/xinxiang/research/'), 
                 ):
        """
        Initialize a GBMObservation instance.
        初始化 GBMObservation 实例。

        Args:
            srcname (str): The source name. 源名称。
            utc_start (str | Time): The observation start time in UTC (ISO format string or astropy Time object).
                观测的 UTC 起始时间（ISO 格式字符串或 astropy Time 对象）。
            ra (float): Right Ascension of the source (degrees). 源的赤经（单位：度）。
            dec (float): Declination of the source (degrees). 源的赤纬（单位：度）。
            filepath (str | Path): The file path, can be a string or a Path object. 文件路径，可以是字符串或 Path 对象。
            tstart_offset (float): The offset (in seconds) from srctime for the start time. 起始时间相对于 srctime 的偏移量（单位：秒）。
            tstop_offset (float): The offset (in seconds) from srctime for the stop time. 结束时间相对于 srctime 的偏移量（单位：秒）。
        """
        self._srcname = srcname
        self._utc_TSTART = utc_start
        self._ra = ra
        self._dec = dec
        self._filepath = Path(filepath) if filepath else None
        self._tstart_offset = tstart_offset
        self._tstop_offset = tstop_offset

        # Initialize dependent properties
        # 初始化依赖属性
        if all([srcname, utc_start, ra, dec, filepath]):
            self._update_time_and_coordinates()
            self._create_directory_structure()
            self._initialize_observation()

    def _update_time_and_coordinates(self):
        """Update time and coordinate-related properties.
        更新时间和坐标相关属性。
        """
        if not all([self._utc_TSTART is not None, self._ra, self._dec]):
            raise ValueError("UTC start time, RA, and DEC must be set before updating time and coordinates. "
                             "在更新时间和坐标之前，必须设置 UTC 起始时间、RA 和 DEC。")
        # Support both string and Time object for utc_start
        # 支持字符串和 Time 对象作为 utc_start
        if isinstance(self._utc_TSTART, Time):
            self.srctime = Time(self._utc_TSTART, scale='utc', precision=9)
        else:
            self.srctime = Time(self._utc_TSTART, scale='utc', precision=9)
        self.coord = SkyCoord(self._ra, self._dec, frame='icrs', unit='deg')
        self.isot = self.srctime.isot

    @property
    def tstart_offset(self):
        """Get the value of tstart_offset.
        获取 tstart_offset 的值。
        """
        return self._tstart_offset

    @tstart_offset.setter
    def tstart_offset(self, value: float | int):
        """Set the value of tstart_offset.
        设置 tstart_offset 的值。
        """
        if not isinstance(value, (float, int)):
            raise ValueError("tstart_offset must be a float or int. tstart_offset 必须是浮点数或整数。")
        if value >= 0:
            raise ValueError("tstart_offset must be a negative value. tstart_offset 必须是负值。")
        self._tstart_offset = value
        print(f"tstart_offset updated to {self._tstart_offset} seconds. "
              f"tstart_offset 已更新为 {self._tstart_offset} 秒。")

    @property
    def tstop_offset(self):
        """Get the value of tstop_offset.
        获取 tstop_offset 的值。
        """
        return self._tstop_offset

    @tstop_offset.setter
    def tstop_offset(self, value: float):
        """Set the value of tstop_offset.
        设置 tstop_offset 的值。
        """
        if not isinstance(value, (float, int)):
            raise ValueError("tstop_offset must be a float or int. tstop_offset 必须是浮点数或整数。")
        if value <= 0:
            raise ValueError("tstop_offset must be a positive value. tstop_offset 必须是正值。")
        self._tstop_offset = value
        print(f"tstop_offset updated to {self._tstop_offset} seconds. "
              f"tstop_offset 已更新为 {self._tstop_offset} 秒。")

    @property
    def tstart(self):
        """Calculate the start time relative to srctime.
        计算相对于 srctime 的起始时间。
        """
        return self.srctime + self._tstart_offset * u.s

    @property
    def tstop(self):
        """Calculate the stop time relative to srctime.
        计算相对于 srctime 的结束时间。
        """
        return self.srctime + self._tstop_offset * u.s

    def _create_directory_structure(self):
        """Create the directory structure.
        创建目录结构。
        """
        if not all([self._srcname, self._filepath]):
            raise ValueError("Source name and file path must be set before creating directory structure. "
                             "在创建目录结构之前，必须设置源名称和文件路径。")
        self.yr = self.srctime.datetime.year
        self.month = f"{self.srctime.datetime.month:02d}"
        self.day = f"{self.srctime.datetime.day:02d}"
        self.month_day = self.srctime.strftime('%m/%d/')
        directory_structure = self._filepath / self._srcname / 'gbm' / f"daily/{self.yr}/{self.month_day}current"
        directory_structure.mkdir(parents=True, exist_ok=True)
        self.datadir = directory_structure
        self.gbmdir = self._filepath / self._srcname / 'gbm'

    def _initialize_observation(self):
        """Initialize observation-related properties.
        初始化观测相关属性。
        """
        if not self.srctime:
            raise ValueError("Time must be initialized before initializing observation. "
                             "在初始化观测之前，必须初始化时间。")
        finder = ContinuousFinder(self.srctime)
        finder.get_poshist(self.datadir)
        # GBM 文件名使用两位年份 (YY)，例如 260119 表示 2026年1月19日
        yr_short = str(self.yr)[-2:]  # 取年份后两位
        self.poshist_filepath = glob(str(self.datadir / f'glg_poshist_all_{yr_short}{self.month}{self.day}_v*.fit'))
        if not self.poshist_filepath:
            raise FileNotFoundError("Poshist file not found. 未找到 Poshist 文件。")
        poshist = GbmPosHist.open(self.poshist_filepath[0])
        self.frame = poshist.get_spacecraft_frame()
        self.states = poshist.get_spacecraft_states()
        self.one_frame = self.frame.at(self.srctime)
        self.visiblecheck = self.check_visibility()
        self.gticheck = self.check_gti()
        self.observation = self.visiblecheck and self.gticheck
        
        # 无论是否可观测都获取探测器信息（用于画图）
        self.detectors = self.get_detectors()
        
        if not self.observation:
            reason = "GTI" if not self.gticheck else "visibility"
            import warnings
            warnings.warn(f"Source {self._srcname} is not observable at {self.srctime.isot} due to {reason}. "
                         f"源 {self._srcname} 在 {self.srctime.isot} 不可观测，原因是 {reason}。"
                         f" 但仍可用于画图。", stacklevel=2)

    @property
    def srcname(self):
        return self._srcname

    @srcname.setter
    def srcname(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Source name must be a string. 源名称必须是字符串。")
        self._srcname = value
        if self._filepath:
            self._create_directory_structure()

    @property
    def utc_TSTART(self):
        return self._utc_TSTART

    @utc_TSTART.setter
    def utc_TSTART(self, value: str):
        if not isinstance(value, str):
            raise ValueError("UTC start time must be a string in ISO format. UTC 起始时间必须是 ISO 格式的字符串。")
        self._utc_TSTART = value
        if all([self._ra, self._dec]):
            self._update_time_and_coordinates()

    @property
    def ra(self):
        return self._ra

    @ra.setter
    def ra(self, value: float):
        if not isinstance(value, (float, int)):
            raise ValueError("RA must be a float or int. RA 必须是浮点数或整数。")
        self._ra = value
        if all([self._utc_TSTART, self._dec]):
            self._update_time_and_coordinates()

    @property
    def dec(self):
        return self._dec

    @dec.setter
    def dec(self, value: float):
        if not isinstance(value, (float, int)):
            raise ValueError("DEC must be a float or int. DEC 必须是浮点数或整数。")
        self._dec = value
        if all([self._utc_TSTART, self._ra]):
            self._update_time_and_coordinates()

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, value: str | Path):
        self._filepath = Path(value) if isinstance(value, str) else value
        if self._srcname:
            self._create_directory_structure()

    def check_visibility(self):
        """Check if the source is within Fermi's field of view.
        检查源是否在 Fermi 的视野范围内。
        """
        poshist = GbmPosHist.open(self.poshist_filepath[0])
        frame = poshist.get_spacecraft_frame()
        one_frame = frame.at(self.srctime)
        return one_frame.location_visible(self.coord)

    def check_gti(self):
        """Check if the source is within the GTI range.
        检查源是否在 GTI 范围内。
        """
        gti = Gti.from_boolean_mask(self.states['time'].value, self.states['good'].value)
        # 将 Time 对象转换为 Fermi MET 时间 (float)

        met_time = Time(self.srctime).fermi
        return gti.contains(met_time)

    def get_detectors(self):
        """Get detectors with an angle less than 90 degrees to the source.
        获取与源夹角小于 90 度的探测器。
        """
        det_angle_list = [(det.name, (self.one_frame.detector_angle(det.name, self.coord)[0]).to_value('deg')) for det in GbmDetectors]
        filtered_det = [(det, ang) for (det, ang) in det_angle_list if ang < 90]
        detxx = filtered_det[:4] if len(filtered_det) > 4 else filtered_det
        return [p for (p, q) in detxx]


    def download_data(self):
        """Download the data needed for the observation.
        下载观测所需的数据。
        """
        cont_finder = ContinuousFinder(self.srctime)
        cont_finder.get_ctime(self.datadir, dets=self.detectors)
        cont_finder.get_tte(self.datadir, dets=self.detectors)
        cont_finder.get_cspec(self.datadir, dets=self.detectors)
        cont_finder.get_spechist(self.datadir, dets=self.detectors)


    def skymap(self):
        """Generate a sky map with source, Sun, and Moon positions.
        生成包含源、太阳和月亮位置的天空图。
        """
        from astropy.coordinates import get_sun, get_body
        
        eqplot = EquatorialPlot(interactive=True)
        eqplot.add_frame(self.one_frame)
        eqplot.sun.zorder = 2
        eqplot.sun.size = 300
        eqplot.ax.text(0.02, 0.95, "ICRS", transform=eqplot.ax.transAxes, fontsize=15, color='red', ha='center')
        
        # 标注源位置（五角星）
        # Mark source position with a star marker
        src_plot = SkyPoints(
            x=self.coord.gcrs.ra.deg, 
            y=self.coord.gcrs.dec.deg, 
            ax=eqplot.ax, 
            color='red', 
            marker='*',  # 五角星
            s=300,       # 大小
            edgecolors='darkred',
            linewidths=0.5,
            zorder=10
        )
        
        # 获取太阳位置
        # Get Sun position at the observation time
        sun_coord = get_sun(self.srctime)
        sun_plot = SkyPoints(
            x=sun_coord.ra.deg,
            y=sun_coord.dec.deg,
            ax=eqplot.ax,
            color='yellow',
            marker='o',
            s=400,
            edgecolors='orange',
            linewidths=2,
            zorder=9
        )
        
        # 获取月亮位置
        # Get Moon position at the observation time
        moon_coord = get_body('moon', self.srctime)
        moon_plot = SkyPoints(
            x=moon_coord.ra.deg,
            y=moon_coord.dec.deg,
            ax=eqplot.ax,
            color='lightgray',
            marker='o',
            s=250,
            edgecolors='gray',
            linewidths=2,
            zorder=8
        )
        
        # 创建图例
        # Create legend with custom handles
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markeredgecolor='darkred', markersize=15, label=f'{self._srcname}'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markeredgecolor='orange', markersize=12, label='Sun'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markeredgecolor='gray', markersize=10, label='Moon'),
        ]
        eqplot.ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 添加时间标注
        # Add time annotation
        eqplot.ax.set_title(f'{self._srcname} Sky Map @ {self.srctime.isot}', fontsize=12)
        
        plt.savefig(self.datadir / f"{self._srcname}_skymap.png", dpi=300)
        plt.show()

    def earthmap(self, earthmap_start_offset: float = -1000, earthmap_stop_offset: float = 1000):
        """
        Generate a map of the Earth's trajectory for the spacecraft.
        生成航天器的地球轨迹图。

        Args:
            earthmap_start_offset (float): The offset (in seconds) from the srctime for the start of the Earth map.
                                           相对于 srctime 的地球地图起始时间偏移量（单位：秒）。
                                           默认值为 -1000 秒。
            earthmap_stop_offset (float): The offset (in seconds) from the srctime for the end of the Earth map.
                                          相对于 srctime 的地球地图结束时间偏移量（单位：秒）。
                                          默认值为 1000 秒。

        Returns:
            None: The function saves the Earth map as a PNG file in the specified directory.
                  该函数将地球地图保存为 PNG 文件到指定目录。
        """
        from gdt.missions.fermi.time import Time as FermiTime
        
        saa = GbmSaa()  # Initialize the South Atlantic Anomaly (SAA) region.
                        # 初始化南大西洋异常区（SAA）区域。
        earthplot = FermiEarthPlot(saa)  # Create an Earth plot for the spacecraft trajectory.
                                         # 创建航天器轨迹的地球图。

        # 获取 poshist 的时间范围
        # Get the time range of the poshist file
        poshist_tmin = self.frame.obstime.min()
        poshist_tmax = self.frame.obstime.max()
        srctime_met = FermiTime(self.srctime).fermi
        
        # 计算请求的时间范围
        requested_tstart = srctime_met + earthmap_start_offset
        requested_tstop = srctime_met + earthmap_stop_offset
        
        # 自动调整时间范围以适应 poshist 数据
        # Auto-adjust time range to fit within poshist data
        if requested_tstart < poshist_tmin.value:
            adjusted_start_offset = poshist_tmin.value - srctime_met + 10  # 加10秒余量
            print(f"⚠️ 起始时间超出 poshist 范围，自动调整 earthmap_start_offset: {earthmap_start_offset} → {adjusted_start_offset:.1f}")
            earthmap_start_offset = adjusted_start_offset
            
        if requested_tstop > poshist_tmax.value:
            adjusted_stop_offset = poshist_tmax.value - srctime_met - 10  # 减10秒余量
            print(f"⚠️ 结束时间超出 poshist 范围，自动调整 earthmap_stop_offset: {earthmap_stop_offset} → {adjusted_stop_offset:.1f}")
            earthmap_stop_offset = adjusted_stop_offset

        # Calculate the duration of the Earth map.
        # 计算地球地图的持续时间。
        continue_time = earthmap_stop_offset - earthmap_start_offset

        # Add the spacecraft trajectory to the Earth plot.
        # 将航天器轨迹添加到地球图中。
        earthplot.add_spacecraft_frame(
            self.frame,
            tstart=self.srctime + earthmap_start_offset * u.s,
            tstop=continue_time,
            trigtime=self.srctime
        )

        earthplot.standard_title()  # Add a standard title to the Earth plot.
                                    # 为地球图添加标准标题。

        # Save the Earth map as a PNG file in the specified directory.
        # 将地球地图保存为 PNG 文件到指定目录。
        plt.savefig(self.datadir / f"{self._srcname}_earthmap.png", dpi=300)
        plt.show()  # Display the Earth map.
                    # 显示地球地图。


    def generate_rsp(self):
        """Generate the GBM response file.
        生成 GBM 响应文件。
        """
        rsp = contgbmrsp(self._ra, self._dec, self.tstart.fermi, self.tstop.fermi, self.detectors)
        rsp.commandpl()
        rsp.gbmrsppl()


    def det_angle(self):
        """Calculate the angle between the source and the detectors.
        计算源与探测器之间的夹角。

        Returns:
            list[tuple]: A list of tuples, where each tuple contains the detector name and the angle (in degrees).
                         一个包含元组的列表，每个元组包含探测器名称和夹角（单位：度）。
        """
        
        # 检查依赖属性是否已初始化
        if not hasattr(self, 'detectors') or not self.detectors:
            print("No detectors available. Ensure 'self.detectors' is initialized. 无可用探测器，请确保 'self.detectors' 已初始化。")
            return []

        if not hasattr(self, 'one_frame') or not hasattr(self, 'coord'):
            raise ValueError("Attributes 'one_frame' and 'coord' must be initialized before calling this function. "
                             "在调用此函数之前，必须初始化 'one_frame' 和 'coord' 属性。")

        try:
            # 计算每个探测器与源之间的夹角
            det_angle_list = [
                (det.name, (self.one_frame.detector_angle(det.name, self.coord)[0]).to_value('deg'))
                for det in self.detectors
            ]
            return det_angle_list
        except Exception as e:
            print(f"Error calculating detector angles: {e} 计算探测器夹角时出错：{e}")
            return []

    def execute_observation(self):
        """Execute the observation process.
        执行观测过程。
        """

        # 检查依赖属性是否已初始化
        if not hasattr(self, 'observation'):
            raise ValueError("Attributes 'srctime', 'utc_start', 'ra' and 'dec' must be initialized before executing observation. "
                             "在执行观测之前，必须初始化 'srctime', 'utc_start', 'ra', 'dec' 属性。")
        if not self.observation:
            raise ValueError("Observation is not possible. Please check the visibility and GTI. "
                             "观测不可行，请检查可见性和 GTI。")

        try:
            # 下载观测所需的数据
            print("Downloading data... 下载数据...")
            self.download_data()

            # 生成响应文件
            print("Generating response file... 生成响应文件...")
            self.generate_rsp()

            # 生成天空图
            print("Generating sky map... 生成天空图...")
            self.skymap()

            # 生成地球轨迹图
            print("Generating Earth map... 生成地球轨迹图...")
            self.earthmap()

            # 计算探测器与源之间的夹角
            print("Calculating detector angles... 正在计算探测器夹角...")
            angles = self.det_angle()
            for det_name, angle in angles:
                print(f"Detector: {det_name}, Angle: {angle:.4f} degrees \n 探测器: {det_name}, 夹角: {angle:.4f} 度")

            print("Observation process completed successfully. 观测过程成功完成。")
        except Exception as e:
            print(f"Error during observation process: {e} 观测过程中发生错误：{e}")








