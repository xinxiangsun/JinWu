"""
Time conversion utilities for astronomy missions.

This module provides astropy.time-compatible MET (Mission Elapsed Time) formats
for various missions including Fermi, HXMT, GECAM, EP, LEIA, and Swift.

Usage:
    from jinwu.core.time import Time
    
    # Convert MET to UTC ISO string
    t = Time(746496123.0, format='fermi_met')
    print(t.isot)  # '2024-08-15T05:22:03.000'
    
    # Convert UTC ISO to MET
    t = Time('2024-08-15T05:22:03', scale='utc')
    met = t.to_value('fermi_met')
    print(met)  # 746496123.0
    
    # Cross-mission conversion
    t_ep = Time(12345.6, format='ep_met')
    fermi_met = t_ep.to_value('fermi_met')
"""

from astropy.time import Time, TimeDelta
from astropy.time.formats import TimeFromEpoch, TimeUnix
import erfa
import numpy as np
from astropy.io import fits
from typing import Tuple, List, Dict, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import DateFormatter
import astropy.units as u

__all__ = [
    'Time', 'TimeFermi', 'TimeEP', 'TimeLEIA', 'TimeGECAM', 'TimeHXMT', 'TimeSwift', 'TimeGrid',
    'check_time_overlap', 'get_overlap_duration', 'plot_time_intervals', 'compare_time_intervals',
    'extract_time_interval'
]

# ============================================================================
# 自定义 MET 格式类（兼容 astropy.time）
# ============================================================================

class TimeFermi(TimeFromEpoch):
    """
    Fermi MET: seconds since 2001-01-01 00:00:00 UTC
    
    Reference: https://heasarc.gsfc.nasa.gov/docs/fermi/
    """
    name = 'fermi'
    unit = 1.0 / erfa.DAYSEC  # seconds to days
    epoch_val = '2001-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'utc'
    epoch_format = 'iso'


class TimeEP(TimeFromEpoch):
    """
    EP MET: seconds since 2020-01-01 00:00:00 UTC
    
    Reference: Einstein Probe mission documentation
    """
    name = 'ep'
    unit = 1.0 / erfa.DAYSEC  # seconds to days
    epoch_val = '2020-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'utc'
    epoch_format = 'iso'


class TimeLEIA(TimeFromEpoch):
    """
    LEIA MET: seconds since 2021-01-01 00:00:00 UTC
    
    Reference: LEIA (XRISM) mission documentation
    """
    name = 'leia'
    unit = 1.0 / erfa.DAYSEC  # seconds to days
    epoch_val = '2021-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'utc'
    epoch_format = 'iso'


class TimeGECAM(TimeFromEpoch):
    """
    GECAM MET: seconds since 2019-01-01 00:00:00 TT
    
    Reference: GECAM mission documentation
    Note: Uses TT (Terrestrial Time) scale, NOT UTC
    """
    name = 'gecam'
    unit = 1.0 / erfa.DAYSEC  # seconds to days
    epoch_val = '2019-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'tt'
    epoch_format = 'iso'


class TimeHXMT(TimeFromEpoch):
    """
    HXMT MET: seconds since 1998-01-01 00:00:00 TT (CXC time reference)
    
    Reference: https://heasarc.gsfc.nasa.gov/docs/hxmt/
    Note: Uses TT (Terrestrial Time) scale, with CXC epoch offset
    """
    name = 'hxmt'
    unit = 1.0 / erfa.DAYSEC  # seconds to days
    epoch_val = '2011-12-31 23:59:57.000'  
    # Corresponds to 1998-01-01 TT with offset 441763197.0
    epoch_val2 = None
    epoch_scale = 'tt'
    epoch_format = 'iso'


class TimeSwift(TimeFromEpoch):
    """
    Swift SCLK: seconds since 2001-01-01 00:00:00 TT
    
    Note: Swift uses TT (Terrestrial Time) scale, NOT UTC directly.
    Swift UTC conversion requires time-dependent UTCF (UTC correlation factors) 
    table from CALDB to convert SCLK to UTC, accounting for clock drifts.
    
    Reference: https://heasarc.gsfc.nasa.gov/docs/swift/
    """
    name = 'swift'
    unit = 1.0 / erfa.DAYSEC  # seconds to days
    epoch_val = '2001-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'tt'
    epoch_format = 'iso'

class TimeGrid(TimeUnix):
    """
    GRID MET: seconds since 1970-01-01 00:00:00 UTC (Unix time)
    
    Reference: GRID mission documentation
    """
    name = 'grid'
    unit = 1.0 / erfa.DAYSEC  # seconds to days
    epoch_val = '1970-01-01 00:00:00'
    epoch_val2 = None
    epoch_scale = 'utc'
    epoch_format = 'iso'
    
# ============================================================================
# 自定义格式通过 TimeFromEpoch 元类自动注册到 astropy.time.Time
# 注：在 astropy 5.0+ 中，TimeFromEpoch 的子类在定义时自动注册
# 无需显式调用 register_class
# ============================================================================

# ============================================================================
# 向后兼容：保留旧 API 作为便利函数（非推荐）
# 这些函数直接实现了 MET <-> UTC 的常用转换，参考用户提供的实现。
# ============================================================================


# def hxmt_met_to_utc(met):

#     dt = TimeDelta(met + 441763197.0, format='sec')
#     ref_tt = Time('1998-01-01T00:00:00', format='isot', scale='tt')
#     now_utc = (ref_tt + dt).value

#     return now_utc


# def hxmt_utc_to_met(utc, format='isot'):

#     now_tt = Time(utc, scale='tt', format=format)
#     met = now_tt.cxcsec - 441763197.0

#     return met


# def fermi_met_to_utc(met):

#     dt = TimeDelta(met, format='sec')
#     ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
#     now_utc = (ref_utc + dt).value

#     return now_utc


# def fermi_utc_to_met(utc, format='isot'):

#     ref_utc = Time('2001-01-01T00:00:00.00', scale='utc', format='isot')
#     now_utc = Time(utc, scale='utc', format=format)
#     met = (now_utc - ref_utc).sec

#     return met


# def fermi_utc_goback(utc, poshist_file):
    
#     poshist = fits.open(poshist_file)[1].data
#     nt = np.size(poshist)
#     sc_time = poshist['SCLK_UTC']
#     sc_quat = np.zeros((nt,4),float)
#     sc_pos = np.zeros((nt,3),float)
#     sc_coords = np.zeros((nt,2),float)
#     try:
#         sc_coords[:,0] = poshist['SC_LON']
#         sc_coords[:,1] = poshist['SC_LAT']
#     except Exception:
#         msg = ''
#         msg += '*** No geographical coordinates available '
#         msg += 'for this file: %s' % poshist_file
#         print(msg)

#     sc_quat[:,0] = poshist['QSJ_1']
#     sc_quat[:,1] = poshist['QSJ_2']
#     sc_quat[:,2] = poshist['QSJ_3']
#     sc_quat[:,3] = poshist['QSJ_4']
#     sc_pos[:,0] = poshist['POS_X']
#     sc_pos[:,1] = poshist['POS_Y']
#     sc_pos[:,2] = poshist['POS_Z']
    
#     G = 6.67428e-11
#     M = 5.9722e24
#     r = (np.sum(sc_pos ** 2.0, 1)) ** (1 / 2.0)
#     r_avg = np.average(r)
#     r_cubed = (r_avg) ** 3.0
#     factor = r_cubed / (G * M)
#     period = 2.0 * np.pi * np.sqrt(factor)

#     utc = Time(utc, scale='utc', format='isot')
#     dt = TimeDelta(period * 30, format='sec')
#     goback_utc = (utc - dt).value
    
#     return goback_utc


# def gecam_met_to_utc(met):

#     dt = TimeDelta(met, format='sec')
#     ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
#     now_utc = (ref_utc + dt).value

#     return now_utc


# def gecam_utc_to_met(utc, format='isot'):

#     now_utc = Time(utc, scale='tt', format=format)
#     ref_utc = Time('2019-01-01T00:00:00.00', format='isot', scale='tt')
#     met = (now_utc - ref_utc).sec
    
#     return met


# def grid_met_to_utc(met):
    
#     now_utc = Time(met, scale='utc', format='unix').to_value('isot')
    
#     return now_utc


# def grid_utc_to_met(isot, format='isot'):
    
#     now_utc = Time(isot, scale='utc', format=format)
#     met = now_utc.to_value('unix')
    
#     return met


# def ep_utc_to_met(utc, format='isot'):

#     ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
#     now_utc = Time(utc, format=format, scale='utc')
#     met = (now_utc - ref_utc).sec

#     return met


# def ep_met_to_utc(met):

#     ref_utc = Time('2020-01-01T00:00:00.000', format='isot', scale='utc')
#     dt = TimeDelta(met, format='sec')
#     now_utc = (ref_utc + dt).value

#     return now_utc


# def leia_utc_to_met(utc, format='isot'):

#     ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
#     now_utc = Time(utc, format=format, scale='utc')
#     met = (now_utc - ref_utc).sec

#     return met


# def leia_met_to_utc(met):

#     ref_utc = Time('2021-01-01T00:00:00.000', format='isot', scale='utc')
#     dt = TimeDelta(met, format='sec')
#     now_utc = (ref_utc + dt).value

#     return now_utc


# def swift_met_to_utc(met, utcf):

#     dt = TimeDelta(met + utcf, format='sec')
#     ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
#     now_utc = (ref_tt + dt).value

#     return now_utc


# def swift_utc_to_met(utc, utcf, format='isot'):

#     ref_tt = Time('2001-01-01T00:00:00.00', scale='tt', format='isot')
#     now_utc = Time(utc, scale='tt', format=format)
#     met = (now_utc - ref_tt).sec - utcf

#     return met


# ============================================================================
# 时间分析工具函数 | Time Analysis Utility Functions
# ============================================================================

def extract_time_interval(
    data_obj,
    name: str,
    time_format: str = 'swift',
    color: Optional[str] = None,
    use_gti: bool = True
) -> Union[Dict[str, Union[Time, str]], List[Dict[str, Union[Time, str]]]]:
    """
    从 OGIP 数据对象中提取时间区间信息 | Extract time interval from OGIP data object.
    
    自动识别数据类型（PHA谱文件或EVT事件文件），提取时间信息并转换为指定格式。
    
    Parameters
    ----------
    data_obj : EventData or PhaData
        通过 read_ogip 读取的数据对象（事件文件或谱文件）
    name : str
        仪器/观测名称（如 'Swift BAT', 'EP WXT'）
    time_format : str, optional
        时间格式（'swift', 'ep', 'fermi', 'hxmt', 'gecam', 'leia', 'grid'）
        默认 'swift'
    color : str, optional
        用于绘图的颜色（如 'red', '#ff0000'）
    use_gti : bool, optional
        对于EVT文件，是否使用GTI（Good Time Intervals）
        如果为False，使用TSTART/TSTOP
        对于PHA文件，此参数无效（PHA总是使用TSTART/TSTOP）
        默认 True
    
    Returns
    -------
    interval : dict or list of dict
        - 对于PHA文件或use_gti=False：返回单个字典
          {'name': str, 'start': Time, 'end': Time, 'color': str, 'type': 'observation'}
        - 对于EVT文件且use_gti=True：返回字典列表（每个GTI一个字典）
          [{'name': str, 'start': Time, 'end': Time, 'color': str, 'type': 'gti', 'gti_index': int}, ...]
    
    Examples
    --------
    >>> from jinwu.core.time import extract_time_interval
    >>> from jinwu.core.file import read_ogip
    >>> 
    >>> # 读取Swift BAT谱文件（PHA）
    >>> bat_pha = read_ogip('sw00123456000b_avg.pha')
    >>> interval = extract_time_interval(bat_pha, 'Swift BAT', time_format='swift', color='red')
    >>> print(f"Observation: {interval['start'].iso} - {interval['end'].iso}")
    >>> 
    >>> # 读取事件文件（EVT）并提取GTI
    >>> bat_evt = read_ogip('sw00123456000bevshpo_uf.evt')
    >>> gti_intervals = extract_time_interval(bat_evt, 'Swift BAT', time_format='swift', use_gti=True)
    >>> print(f"Found {len(gti_intervals)} GTI intervals")
    >>> 
    >>> # 直接用于时间对比
    >>> from jinwu.core.time import compare_time_intervals, plot_time_intervals
    >>> intervals = [
    ...     extract_time_interval(ep_evt, 'EP WXT', time_format='ep', color='blue'),
    ...     extract_time_interval(bat_pha, 'Swift BAT', time_format='swift', color='red')
    ... ]
    >>> results = compare_time_intervals(intervals)
    >>> fig = plot_time_intervals(intervals)
    
    Notes
    -----
    数据类型识别：
    - PHA文件：通过 header['TSTART'] 和 header['TSTOP'] 获取观测时间
    - EVT文件：
      * use_gti=True: 通过 gti_start 和 gti_stop 数组获取所有GTI
      * use_gti=False: 通过 header['TSTART'] 和 header['TSTOP'] 获取总时间范围
    
    时间格式对应关系：
    - 'swift': Swift卫星时间（秒，自2001-01-01 TT）
    - 'ep': Einstein Probe时间（秒，自2020-01-01 UTC）
    - 'fermi': Fermi卫星时间（秒，自2001-01-01 UTC）
    - 'hxmt': HXMT时间（秒，自CXC参考时间）
    - 'gecam': GECAM时间（秒，自2019-01-01 TT）
    - 'leia': LEIA/XRISM时间（秒，自2021-01-01 UTC）
    - 'grid': GRID时间（Unix时间戳）
    """
    # 尝试识别数据类型
    has_gti = hasattr(data_obj, 'gti_start') and hasattr(data_obj, 'gti_stop')
    has_header = hasattr(data_obj, 'header')
    
    if not has_header:
        raise ValueError(f"数据对象缺少 header 属性，无法提取时间信息")
    
    header = data_obj.header
    
    # 检查是否有 TSTART 和 TSTOP
    if 'TSTART' not in header or 'TSTOP' not in header:
        raise ValueError(f"Header 中缺少 TSTART 或 TSTOP 关键字")
    
    tstart_met = header['TSTART']
    tstop_met = header['TSTOP']
    
    # 如果是EVT文件且要求使用GTI
    if has_gti and use_gti:
        gti_start_array = data_obj.gti_start
        gti_stop_array = data_obj.gti_stop
        
        if len(gti_start_array) == 0:
            # GTI为空，回退到TSTART/TSTOP
            print(f"Warning: GTI is empty for {name}, using TSTART/TSTOP instead")
            start_time = Time(tstart_met, format=time_format)
            end_time = Time(tstop_met, format=time_format)
            
            interval = {
                'name': name,
                'start': start_time,
                'end': end_time,
                'type': 'observation'
            }
            if color is not None:
                interval['color'] = color
            
            return interval
        
        # 返回多个GTI区间
        intervals = []
        for i, (gti_start, gti_stop) in enumerate(zip(gti_start_array, gti_stop_array)):
            start_time = Time(gti_start, format=time_format)
            end_time = Time(gti_stop, format=time_format)
            
            # 为每个GTI创建一个名称
            gti_name = f"{name} GTI#{i+1}" if len(gti_start_array) > 1 else name
            
            interval = {
                'name': gti_name,
                'start': start_time,
                'end': end_time,
                'type': 'gti',
                'gti_index': i
            }
            if color is not None:
                interval['color'] = color
            
            intervals.append(interval)
        
        return intervals
    
    else:
        # PHA文件或EVT文件但不使用GTI
        start_time = Time(tstart_met, format=time_format)
        end_time = Time(tstop_met, format=time_format)
        
        data_type = 'observation' if not has_gti else 'observation (TSTART/TSTOP)'
        
        interval = {
            'name': name,
            'start': start_time,
            'end': end_time,
            'type': data_type
        }
        if color is not None:
            interval['color'] = color
        
        return interval


def check_time_overlap(
    start1: Time, 
    end1: Time, 
    start2: Time, 
    end2: Time
) -> bool:
    """
    检查两个时间段是否有重叠 | Check if two time intervals overlap.
    
    Parameters
    ----------
    start1, end1 : Time
        第一个时间段的起始和结束时间 | Start and end of first interval
    start2, end2 : Time
        第二个时间段的起始和结束时间 | Start and end of second interval
    
    Returns
    -------
    bool
        是否有重叠 | True if intervals overlap
    
    Examples
    --------
    >>> from jinwu.core.time import Time, check_time_overlap
    >>> t1_start = Time(100, format='ep')
    >>> t1_end = Time(200, format='ep')
    >>> t2_start = Time(150, format='ep')
    >>> t2_end = Time(250, format='ep')
    >>> check_time_overlap(t1_start, t1_end, t2_start, t2_end)
    True
    """
    # 两个时间段不重叠的充要条件是：一个完全在另一个之前
    # 重叠条件：NOT (end1 < start2 OR end2 < start1)
    return not (end1 < start2 or end2 < start1)


def get_overlap_duration(
    start1: Time,
    end1: Time,
    start2: Time,
    end2: Time,
    unit: str = 's'
) -> Tuple[Optional[Time], Optional[Time], float]:
    """
    获取两个时间段的重叠区间和持续时间 | Get overlap interval and duration.
    
    Parameters
    ----------
    start1, end1 : Time
        第一个时间段 | First time interval
    start2, end2 : Time
        第二个时间段 | Second time interval
    unit : str, optional
        返回持续时间的单位 (默认秒) | Duration unit (default: seconds)
    
    Returns
    -------
    overlap_start : Time or None
        重叠区间起始时间，无重叠时为 None | Overlap start time or None
    overlap_end : Time or None
        重叠区间结束时间，无重叠时为 None | Overlap end time or None
    duration : float
        重叠时长，无重叠时为 0 | Overlap duration or 0
    
    Examples
    --------
    >>> from jinwu.core.time import Time, get_overlap_duration
    >>> t1_start = Time(100, format='ep')
    >>> t1_end = Time(200, format='ep')
    >>> t2_start = Time(150, format='ep')
    >>> t2_end = Time(250, format='ep')
    >>> overlap_start, overlap_end, duration = get_overlap_duration(
    ...     t1_start, t1_end, t2_start, t2_end
    ... )
    >>> print(f"Overlap: {duration:.1f} seconds")
    Overlap: 50.0 seconds
    """
    if not check_time_overlap(start1, end1, start2, end2):
        return None, None, 0.0
    
    # 重叠区间的起始时间是两个起始时间中较晚的
    overlap_start = max(start1, start2)
    # 重叠区间的结束时间是两个结束时间中较早的
    overlap_end = min(end1, end2)
    
    # 计算持续时间
    duration = (overlap_end - overlap_start).to_value(unit)
    
    return overlap_start, overlap_end, duration


def compare_time_intervals(
    intervals: List[Dict[str, Union[Time, str]]],
    reference_time: Optional[Time] = None,
    print_summary: bool = True
) -> Dict:
    """
    比较多个时间段，找出所有重叠关系 | Compare multiple time intervals.
    
    Parameters
    ----------
    intervals : list of dict
        时间段列表，每个字典包含：
        - 'name': 仪器/观测名称 (str)
        - 'start': 起始时间 (Time 对象)
        - 'end': 结束时间 (Time 对象)
        - 'color': (可选) 颜色 (str)
    reference_time : Time, optional
        参考时间（用于计算相对时间） | Reference time for relative times
    print_summary : bool, optional
        是否打印摘要信息 | Whether to print summary
    
    Returns
    -------
    dict
        包含重叠分析结果的字典 | Dictionary with overlap analysis results
    
    Examples
    --------
    >>> from jinwu.core.time import Time, compare_time_intervals
    >>> intervals = [
    ...     {'name': 'WXT', 'start': Time(100, format='ep'), 'end': Time(200, format='ep')},
    ...     {'name': 'BAT', 'start': Time(150, format='swift'), 'end': Time(250, format='swift')}
    ... ]
    >>> results = compare_time_intervals(intervals)
    """
    results = {
        'intervals': intervals,
        'overlaps': [],
        'total_overlaps': 0
    }
    
    n = len(intervals)
    
    if print_summary:
        print("=" * 70)
        print("Time Interval Comparison Analysis")
        print("=" * 70)
        print(f"\n{n} intervals to compare\n")
    
    # 两两比较所有时间段
    for i in range(n):
        int1 = intervals[i]
        start1 = int1['start']
        end1 = int1['end']
        
        # 类型检查
        if not isinstance(start1, Time) or not isinstance(end1, Time):
            raise TypeError(f"时间段 '{int1['name']}' 的 start/end 必须是 Time 对象")
        
        if print_summary:
            duration1 = (end1 - start1).to_value('s')
            print(f"📊 {int1['name']}:")
            print(f"   Start: {start1.iso}")
            print(f"   End:   {end1.iso}")
            print(f"   Duration: {duration1:.2f} s")
            
            if reference_time is not None:
                rel_start = (start1 - reference_time).to_value('s')
                rel_end = (end1 - reference_time).to_value('s')
                print(f"   Relative to T0: {rel_start:.2f} - {rel_end:.2f} s")
            print()
        
        for j in range(i + 1, n):
            int2 = intervals[j]
            start2 = int2['start']
            end2 = int2['end']
            
            # 类型检查
            if not isinstance(start2, Time) or not isinstance(end2, Time):
                raise TypeError(f"时间段 '{int2['name']}' 的 start/end 必须是 Time 对象")
            
            overlap_start, overlap_end, duration = get_overlap_duration(
                start1, end1, start2, end2
            )
            
            if duration > 0 and overlap_start is not None and overlap_end is not None:
                overlap_info = {
                    'interval1': int1['name'],
                    'interval2': int2['name'],
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'duration': duration
                }
                results['overlaps'].append(overlap_info)
                results['total_overlaps'] += 1
                
                if print_summary:
                    print(f"🔗 Overlap: {int1['name']} ↔ {int2['name']}")
                    print(f"   Start: {overlap_start.iso}")
                    print(f"   End:   {overlap_end.iso}")
                    print(f"   Duration: {duration:.2f} s")
                    
                    if reference_time is not None:
                        rel_start = (overlap_start - reference_time).to_value('s')
                        rel_end = (overlap_end - reference_time).to_value('s')
                        print(f"   Relative to T0: {rel_start:.2f} - {rel_end:.2f} s")
                    print()
    
    if print_summary:
        print(f"✅ Found {results['total_overlaps']} overlap interval(s)")
        print("=" * 70)
    
    return results


def plot_time_intervals(
    intervals: List[Dict[str, Union[Time, str]]],
    reference_time: Optional[Time] = None,
    figsize: Tuple[float, float] = (14, 6),
    title: str = "Observation Time Intervals",
    save_path: Optional[str] = None,
    show_utc: bool = True,
    show_relative: bool = True
):
    """
    绘制时间段对比图 | Plot time intervals comparison.
    
    在时间轴上可视化多个仪器的观测时间段，显示重叠关系。
    
    Parameters
    ----------
    intervals : list of dict
        时间段列表，每个字典包含：
        - 'name': 仪器/观测名称 (str)
        - 'start': 起始时间 (Time 对象)
        - 'end': 结束时间 (Time 对象)
        - 'color': (可选) 颜色 (str)
    reference_time : Time, optional
        参考时间（用于相对时间显示） | Reference time
    figsize : tuple, optional
        图形大小 | Figure size
    title : str, optional
        图形标题 | Figure title
    save_path : str, optional
        保存路径 | Path to save figure
    show_utc : bool, optional
        是否显示 UTC 时间轴 | Show UTC time axis
    show_relative : bool, optional
        是否显示相对时间轴 | Show relative time axis
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        生成的图形对象 | Generated figure
    
    Examples
    --------
    >>> from jinwu.core.time import Time, plot_time_intervals
    >>> intervals = [
    ...     {'name': 'EP WXT', 'start': Time(100, format='ep'), 
    ...      'end': Time(200, format='ep'), 'color': 'blue'},
    ...     {'name': 'Swift BAT', 'start': Time(150, format='swift'), 
    ...      'end': Time(250, format='swift'), 'color': 'red'}
    ... ]
    >>> fig = plot_time_intervals(intervals, save_path='time_comparison.png')
    """
    # 创建图形
    n_axes = int(show_utc) + int(show_relative)
    if n_axes == 0:
        raise ValueError("At least one time axis must be shown")
    
    fig, axes = plt.subplots(n_axes, 1, figsize=figsize, sharex=False)
    if n_axes == 1:
        axes = [axes]
    
    # 默认颜色循环
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # 准备数据
    n_intervals = len(intervals)
    
    # 找出全局时间范围（使用第一个时间段的刻度）
    all_times = []
    for interval in intervals:
        start = interval['start']
        end = interval['end']
        
        # 类型检查
        if not isinstance(start, Time) or not isinstance(end, Time):
            raise TypeError(f"时间段 '{interval['name']}' 的 start/end 必须是 Time 对象")
        
        all_times.extend([start, end])
    
    # 转换到统一的时间刻度用于绘图
    plot_times = [t.datetime for t in all_times]
    
    ax_idx = 0
    
    # 绘制 UTC 时间轴
    if show_utc:
        ax = axes[ax_idx]
        
        for i, interval in enumerate(intervals):
            color = interval.get('color', default_colors[i % len(default_colors)])
            name = interval['name']
            start = interval['start']
            end = interval['end']
            
            # 类型检查（绘图前确保是 Time 对象）
            if not isinstance(start, Time) or not isinstance(end, Time):
                raise TypeError(f"时间段 '{name}' 的 start/end 必须是 Time 对象")
            
            start_dt = start.datetime
            end_dt = end.datetime
            duration = (end - start).to_value('s')
            
            # 绘制时间条
            ax.barh(
                y=i,
                width=(end_dt - start_dt).total_seconds() / 3600,  # 转换为小时
                left=start_dt,
                height=0.6,
                color=color,
                alpha=0.7,
                label=f"{name} ({duration:.1f} s)"
            )
            
            # 添加标签
            ax.text(
                start_dt,
                i,
                f"  {name}",
                va='center',
                ha='left',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_yticks(range(n_intervals))
        ax.set_yticklabels([])
        ax.set_xlabel('UTC Time', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper right', fontsize=9)
        
        # 格式化时间轴
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d\n%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
        
        ax_idx += 1
    
    # 绘制相对时间轴
    if show_relative:
        if reference_time is None:
            # 如果没有指定参考时间，使用第一个时间段的起始时间
            ref = intervals[0]['start']
            if not isinstance(ref, Time):
                raise TypeError(f"时间段 '{intervals[0]['name']}' 的 start 必须是 Time 对象")
            reference_time = ref
        
        ax = axes[ax_idx]
        
        for i, interval in enumerate(intervals):
            color = interval.get('color', default_colors[i % len(default_colors)])
            name = interval['name']
            start = interval['start']
            end = interval['end']
            
            # 类型检查
            if not isinstance(start, Time) or not isinstance(end, Time):
                raise TypeError(f"时间段 '{name}' 的 start/end 必须是 Time 对象")
            
            # 相对时间（秒）
            start_rel = (start - reference_time).to_value('s')
            end_rel = (end - reference_time).to_value('s')
            duration = end_rel - start_rel
            
            # 绘制时间条
            ax.barh(
                y=i,
                width=duration,
                left=start_rel,
                height=0.6,
                color=color,
                alpha=0.7,
                label=f"{name} ({duration:.1f} s)"
            )
            
            # 添加标签
            ax.text(
                start_rel,
                i,
                f"  {name}",
                va='center',
                ha='left',
                fontsize=10,
                fontweight='bold'
            )
            
            # 标注起止时间
            ax.text(
                start_rel,
                i - 0.35,
                f"{start_rel:.1f}",
                va='top',
                ha='center',
                fontsize=8,
                color='gray'
            )
            ax.text(
                end_rel,
                i - 0.35,
                f"{end_rel:.1f}",
                va='top',
                ha='center',
                fontsize=8,
                color='gray'
            )
        
        # 标记参考时间（T=0）
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Reference Time T0')
        
        ax.set_yticks(range(n_intervals))
        ax.set_yticklabels([])
        ax.set_xlabel(f'Relative Time (s)\nT0 = {reference_time.iso}', fontsize=12)
        if not show_utc:
            ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Time comparison plot saved: {save_path}")
    
    return fig

