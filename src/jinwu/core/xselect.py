"""轻量级 xselect-like 功能（纯 Python 实现，最小可用版）

目标：提供事件筛选、从事件生成 PHA、以及写出 OGIP-style PHA 文件的基础工具。
本模块为最小可用实现，便于后续逐步增加 region、GTI 合并、复杂表达式解析、分组/重整等功能。

注意：该实现依赖于同包内已有的 `file.py` 中的数据类与读取器（`read_evt`、`PhaData` 等），以及
`astropy` 与 `numpy`。输出的 PHA 文件使用常见的 `SPECTRUM` BinTable HDU 布局。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any, cast

import numpy as np
from astropy.io import fits

from .file import EventData, PhaData, read_evt, OgipMeta, LightcurveData
from ..ftools import region as regionmod
from ..ftools import ftselect as exprmod
from . import gti as gtimod
from ..ftools import xselect_mdb
from pathlib import Path as _Path
import warnings
# ftools: local pure-Python replacements for common HEASOFT utilities
from .. import ftools

__all__ = [
    'select_events', 'accumulate_spectrum_from_events', 'write_pha', 'XSelectSession',
]


class XSelectSession:
    """Interactive-style session holding original events and current filters.

    Usage:
      sess = XSelectSession('events.evt')
      sess.apply_time(tmin=0, tmax=100)
      sess.apply_region({'type':'circle','x':10,'y':10,'r':5})
      sess.clear_region()
      ev = sess.current  # EventData

    Implementation notes:
    - The session stores `original` (the unfiltered EventData) and `current`.
    - Filters are tracked in `state` and any change recomputes `current`
      by calling the module-level `select_events` against `original`.
    - If the session is created from an `EventData` without a file `path`,
      `original` will be that object; in that case the session cannot recover
      events removed earlier by other code unless the caller provided an
      unfiltered `EventData`.
    """

    def __init__(self, source: str | Path | EventData, *, keep_original: bool = True):
        # read original events when a path is provided; otherwise use provided EventData
        if isinstance(source, EventData):
            self.provided_ev = source
            self.path = getattr(source, 'path', None)
            if self.path is not None and keep_original:
                try:
                    self.original = read_evt(self.path)
                except Exception:
                    # fallback to provided object
                    self.original = source
            else:
                self.original = source
        else:
            self.path = Path(source)
            self.original = read_evt(self.path)

        # current view and state
        self.current = self.original
        self.state: dict = {
            'tmin': None, 'tmax': None,
            'pi_min': None, 'pi_max': None,
            'region': None, 'invert_region': False,
            'expr': None,
        }

    def recompute(self):
        """Recompute `current` from `original` using tracked state."""
        invert_region_val = self.state.get('invert_region', False)
        if invert_region_val is None:
            invert_region_val = False
        self.current = select_events(
            self.original,
            tmin=self.state.get('tmin'),
            tmax=self.state.get('tmax'),
            pi_min=self.state.get('pi_min'),
            pi_max=self.state.get('pi_max'),
            region=self.state.get('region'),
            invert_region=bool(invert_region_val),
            expr=self.state.get('expr')
        )

        # apply helpers
    def apply_time(self, *, tmin: Optional[float] = None, tmax: Optional[float] = None):
        """应用时间过滤；支持 float（相对时间秒）或 astropy/jinwu Time 对象。

        参数
        ----
        tmin/tmax : float | Time | TimeDelta, optional
            - float: 相对时间（秒），直接与事件 time 比较
            - Time 对象: 自动转换为相对时间（使用 original.timezero_obj）
            - TimeDelta: 取秒数作为相对时间
        """
        tmin_f = _coerce_time_bound(tmin, self.original)
        tmax_f = _coerce_time_bound(tmax, self.original)
        self.state['tmin'] = tmin_f
        self.state['tmax'] = tmax_f
        self.recompute()

    def apply_energy(self, *, pi_min: Optional[int] = None, pi_max: Optional[int] = None):
        self.state['pi_min'] = pi_min
        self.state['pi_max'] = pi_max
        self.recompute()

    def apply_region(self, region: dict | str | Path | None, *, invert: bool = False):
        self.state['region'] = region
        self.state['invert_region'] = bool(invert)
        self.recompute()

    def apply_expr(self, expr: Optional[str]):
        self.state['expr'] = expr
        self.recompute()

    # clear helpers
    def clear_region(self):
        self.state['region'] = None
        self.state['invert_region'] = False
        self.recompute()

    def clear_time(self):
        self.state['tmin'] = None
        self.state['tmax'] = None
        self.recompute()

    def clear_energy(self):
        self.state['pi_min'] = None
        self.state['pi_max'] = None
        self.recompute()

    def clear_all(self):
        self.state = {k: None for k in ('tmin', 'tmax', 'pi_min', 'pi_max', 'region', 'expr')}
        self.state['invert_region'] = False
        self.current = self.original

    # convenience accessors
    @property
    def meta(self):
        return getattr(self.current, 'meta', None)

    @property
    def header(self):
        return getattr(self.current, 'header', None)
    
    # Extract methods for convenience
    def extract_spectrum(self, **kwargs):
        """从当前过滤后的事件提取能谱。"""
        return extract_spectrum(self.current, **kwargs)
    
    def extract_curve(self, binsize: float, **kwargs):
        """从当前过滤后的事件提取光变曲线。"""
        return extract_curve(self.current, binsize=binsize, **kwargs)
    
    def extract_image(self, **kwargs):
        """从当前过滤后的事件提取图像。"""
        return extract_image(self.current, **kwargs)

    # Save helpers
    def save_current(self, outpath: str | Path, *, kind: str = 'evt', overwrite: bool = False, **kwargs) -> Path:
        """将当前过滤后的事件或派生产品保存到文件。

        参数
        - outpath: 输出文件路径
        - kind: 'evt' (保存事件表), 'lc' (提取并保存光变曲线),
                 'pha' (提取并保存能谱), 'img' (提取并保存图像)
        - overwrite: 是否覆盖已有文件
        - kwargs: 传递给底层提取/写出函数的其它参数

        等价于在 `self.current` 上调用 `save`，但通过 session 接口更直观。
        """
        cur = self.current
        # EventData.save 已经根据 kind 选择提取和写出逻辑
        return cur.save(outpath, kind=kind, overwrite=overwrite, **kwargs)


def write_curve(lc: LightcurveData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """Write a LightcurveData to a FITS file.

    Writes a binary table HDU containing at least `TIME` and `RATE` or
    `COUNTS` depending on `lc.is_rate`. Mirrors xselect's basic write-curve
    behavior: include primary header with TELESCOP/INSTRUME/TSTART/TSTOP/EXPOSURE.
    """
    outp = Path(outpath)
    if outp.exists() and not overwrite:
        raise FileExistsError(str(outp))

    from .file import LightcurveData
    if not isinstance(lc, LightcurveData):
        raise TypeError('lc must be LightcurveData')

    cols = []
    time = np.asarray(lc.time, dtype=float)
    cols.append(fits.Column(name='TIME', format='D', array=time))
    if lc.is_rate:
        val = np.asarray(lc.value, dtype=float)
        cols.append(fits.Column(name='RATE', format='E', array=val))
        if lc.error is not None:
            cols.append(fits.Column(name='ERROR', format='E', array=np.asarray(lc.error, dtype=float)))
    else:
        val = np.asarray(lc.value, dtype=float)
        cols.append(fits.Column(name='COUNTS', format='E', array=val))
        if lc.error is not None:
            cols.append(fits.Column(name='ERROR', format='E', array=np.asarray(lc.error, dtype=float)))

    hdu_tab = fits.BinTableHDU.from_columns(cols, name='LIGHTCURVE')
    hdr = hdu_tab.header
    hdr['EXTNAME'] = 'LIGHTCURVE'
    hdr['TELESCOP'] = lc.meta.telescop if (lc.meta is not None and getattr(lc.meta, 'telescop', None)) else hdr.get('TELESCOP', 'UNKNOWN')
    hdr['INSTRUME'] = lc.meta.instrume if (lc.meta is not None and getattr(lc.meta, 'instrume', None)) else hdr.get('INSTRUME', 'UNKNOWN')
    if lc.exposure is not None:
        try:
            hdr['EXPOSURE'] = float(lc.exposure)
        except Exception:
            pass
    if lc.meta is not None:
        if getattr(lc.meta, 'tstart', None) is not None:
            hdr['TSTART'] = float(lc.meta.tstart)
        if getattr(lc.meta, 'tstop', None) is not None:
            hdr['TSTOP'] = float(lc.meta.tstop)

    prih = fits.PrimaryHDU()
    try:
        if lc.meta is not None:
            if getattr(lc.meta, 'instrume', None):
                prih.header['INSTRUME'] = lc.meta.instrume
            if getattr(lc.meta, 'telescop', None):
                prih.header['TELESCOP'] = lc.meta.telescop
            if getattr(lc.meta, 'tstart', None) is not None:
                prih.header['TSTART'] = float(lc.meta.tstart)
            if getattr(lc.meta, 'tstop', None) is not None:
                prih.header['TSTOP'] = float(lc.meta.tstop)
    except Exception:
        pass

    hdul = fits.HDUList([prih, hdu_tab])
    hdul.writeto(outp, overwrite=overwrite)
    return outp


def write_image(img: np.ndarray, xedges: np.ndarray, yedges: np.ndarray, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """Write a 2D image (numpy array) to a FITS Primary HDU.

    - `img` is a 2D array with shape (ny, nx) matching histogram2d output.
    - `xedges` and `yedges` are the bin edges used to build the image;
      helpful header keywords will be written (XMIN/XMAX/YMIN/YMAX/NX/NY).
    """
    outp = Path(outpath)
    if outp.exists() and not overwrite:
        raise FileExistsError(str(outp))

    data = np.asarray(img)
    # Primary image HDU
    prih = fits.PrimaryHDU(data=data.astype(np.float32))
    hdr = prih.header
    hdr['BUNIT'] = 'COUNT'
    try:
        hdr['NXPIX'] = int(data.shape[1])
        hdr['NYPIX'] = int(data.shape[0])
    except Exception:
        pass
    try:
        hdr['XMIN'] = float(xedges[0])
        hdr['XMAX'] = float(xedges[-1])
        hdr['YMIN'] = float(yedges[0])
        hdr['YMAX'] = float(yedges[-1])
    except Exception:
        pass

    hdul = fits.HDUList([prih])
    hdul.writeto(outp, overwrite=overwrite)
    return outp


# --- xselect.mdb integration (lazy load + optional persistent cache) ---
_MDB_TREE = None


def _get_mdb_tree(use_cache: bool = True):
    """Lazy-load parsed xselect.mdb into module cache. Returns the parsed tree."""
    global _MDB_TREE
    if _MDB_TREE is not None:
        return _MDB_TREE
    # guess path relative to package
    try:
        base = _Path(__file__).resolve().parents[1]
        mdb_path = base / 'data' / 'xselect.mdb'
        cache_path = str(mdb_path) + '.pkl'
        if mdb_path.exists():
            _MDB_TREE = xselect_mdb.load_mdb(str(mdb_path), use_cache=use_cache, cache_path=cache_path)
            return _MDB_TREE
    except Exception:
        pass
    # fallback: try to locate via package import path
    try:
        import importlib.resources as _ir
        import io
        # attempt to read resource if installed as package
        with _ir.open_text('jinwu.data', 'xselect.mdb') as f:
            lines = f.readlines()
        _MDB_TREE = xselect_mdb._parse_lines(lines)
        return _MDB_TREE
    except Exception:
        _MDB_TREE = {}
        return _MDB_TREE


def _infer_adjustgti_timepixr_and_frame(ev: EventData, tree) -> tuple[bool, Optional[float], Optional[float]]:
    """Wrapper: call into `xselect_mdb.infer_adjustgti_timepixr_and_frame` for inference."""
    return xselect_mdb.infer_adjustgti_timepixr_and_frame(tree, getattr(ev, 'header', None), getattr(ev, 'meta', None))


def select_events(path: str | Path | EventData, *, tmin: Optional[float] = None, tmax: Optional[float] = None,
                  pi_min: Optional[int] = None, pi_max: Optional[int] = None,
                  region: Optional[dict | str | Path] = None, invert_region: bool = False,
                  expr: Optional[str] = None) -> EventData:
    """读取并按简单条件筛选事件表。

    参数
    - path: 事件 FITS 文件路径
    - tmin/tmax: 时间区间（闭区间）
    - pi_min/pi_max: PI/CHANNEL 范围（闭区间），若文件中没有 PI 列会尝试 CHANNEL

    返回
    - EventData（筛选后的副本，不会修改源文件）
    """
    ev = path if isinstance(path, EventData) else read_evt(path)

    # apply time, energy, region in xselect-like order: TIME -> ENERGY -> REGION
    ev2 = ev
    if (tmin is not None) or (tmax is not None):
        ev2 = filter_time(ev2, tmin=tmin, tmax=tmax)
    if (pi_min is not None) or (pi_max is not None):
        ev2 = filter_energy(ev2, pi_min=pi_min, pi_max=pi_max)

    # optional ftselect-like expression filtering (applies after energy filter)
    if expr is not None and str(expr).strip() != '':
        mask = exprmod.expression_to_mask(ev2, expr)
        t = np.asarray(ev2.time, dtype=float)
        new_time = t[mask]
        new_pi = None if ev2.pi is None else np.asarray(ev2.pi, dtype=int)[mask]
        new_ch = None if ev2.channel is None else np.asarray(ev2.channel, dtype=int)[mask]
        new_x = None if ev2.x is None else np.asarray(ev2.x)[mask]
        new_y = None if ev2.y is None else np.asarray(ev2.y)[mask]
        new_energy = None if ev2.energy is None else np.asarray(ev2.energy)[mask]
        ev2 = EventData(
            path=ev2.path, time=new_time,
            timezero=ev2.timezero, timezero_obj=ev2.timezero_obj, telescop=ev2.telescop,
            pi=new_pi, channel=new_ch, x=new_x, y=new_y,
            gti_start=ev2.gti_start, gti_stop=ev2.gti_stop,
            gti_start_obj=ev2.gti_start_obj, gti_stop_obj=ev2.gti_stop_obj,
            gti=ev2.gti,
            raw_columns=None, colmap=ev2.colmap, energy=new_energy, ebounds=ev2.ebounds,
            header=ev2.header, meta=ev2.meta, columns=ev2.columns, headers_dump=ev2.headers_dump
        )

    if region is not None:
        # region may be a dict (inline), a path to one or more region files, or a list of shapes
        shapes = []
        if isinstance(region, dict):
            shapes = [region]
        elif isinstance(region, (str, Path)):
            # Allow multiple files separated by whitespace (xselect permits multiple region files)
            s = str(region)
            parts = s.split()
            files = [Path(p) for p in parts]
            for f in files:
                if not f.exists():
                    raise FileNotFoundError(f"Region file not found: {f}")
                shapes.extend(regionmod.parse_ds9_region_file(f))
        else:
            # assume it's a list-like of shapes or file paths
            try:
                for item in region:
                    if isinstance(item, dict):
                        shapes.append(item)
                    elif isinstance(item, (str, Path)):
                        p = Path(item)
                        if not p.exists():
                            raise FileNotFoundError(f"Region file not found: {p}")
                        shapes.extend(regionmod.parse_ds9_region_file(p))
                    else:
                        shapes.append(item)
            except Exception:
                shapes = [region]

        # Determine preferred X/Y columns based on mission DB and file metadata.
        try:
            tree = _get_mdb_tree()
            mission = None
            instr = None
            mode = None
            if ev2.meta is not None:
                mission = getattr(ev2.meta, 'telescop', None)
                instr = getattr(ev2.meta, 'instrume', None)
            if mission is None and ev2.header is not None:
                mission = ev2.header.get('TELESCOP')
            if instr is None and ev2.header is not None:
                instr = ev2.header.get('INSTRUME')
            defaults = xselect_mdb.get_defaults(tree, str(mission).upper() if mission is not None else '', str(instr).upper() if instr is not None else None, mode)
        except Exception:
            defaults = {}

        # If defaults provide explicit column names, prefer them when parsing/ applying regions.
        # We'll attach preferred column names into shapes if they are simple dict regions lacking explicit column keys.
        col_x = defaults.get('x') or defaults.get('rawx') or defaults.get('detx')
        col_y = defaults.get('y') or defaults.get('rawy') or defaults.get('dety')
        # Normalize None
        col_x = None if col_x in (None, '') else col_x
        col_y = None if col_y in (None, '') else col_y

        # If region shapes don't include an explicit 'coord' or reference frame, set preferred columns
        for shp in shapes:
            if isinstance(shp, dict):
                if ('coord' not in shp) and (col_x is not None and col_y is not None):
                    # annotate so region apply can interpret which columns to use when needed
                    shp.setdefault('xcol', col_x)
                    shp.setdefault('ycol', col_y)

        out = regionmod.apply_region_mask_to_events(ev2, cast(list, shapes), invert=invert_region)
        return out
    return ev2


def _coerce_time_bound(val: Optional[float | Any], ev: Optional[EventData] = None) -> Optional[float]:
    """将 tmin/tmax 归一化为 float 秒（相对时间），支持 astropy/jinwu 的 Time/TimeDelta。

    - 若为 None，直接返回 None；
    - 若为 float/int，直接视为相对时间秒数；
    - 若为 Time 对象且提供了 ev.timezero_obj，则转换为相对时间：
      相对时间 = (Time - timezero_obj).sec
    - 若为 TimeDelta，则直接取 .sec
    """
    if val is None:
        return None
    
    # 检查是否为 Time 对象（非 TimeDelta）
    # 优先检查是否有 timezero_obj 可用于转换
    if ev is not None and getattr(ev, 'timezero_obj', None) is not None:
        timezero_obj = ev.timezero_obj
        # 尝试判断 val 是否为 Time 对象（有 jd 属性但非 TimeDelta）
        has_jd = hasattr(val, 'jd')
        is_timedelta = hasattr(val, 'to_value') and not has_jd
        
        if has_jd and not is_timedelta:
            # val 是 Time 对象，转换为相对时间
            try:
                # 相对时间 = (val - timezero_obj) 的秒数
                diff = val - timezero_obj
                if hasattr(diff, 'sec'):
                    return float(diff.sec)
                elif hasattr(diff, 'to_value'):
                    return float(diff.to_value('sec'))
            except Exception:
                pass
    
    # astropy.time.TimeDelta 或其他有 to_value 的对象
    try:
        if hasattr(val, 'to_value'):
            try:
                return float(val.to_value('sec'))
            except Exception:
                pass
        # 某些 Time-like 对象可能有 .sec 属性
        sec_attr = getattr(val, 'sec', None)
        if sec_attr is not None and not isinstance(sec_attr, (list, tuple, np.ndarray)):
            try:
                return float(sec_attr)
            except Exception:
                pass
    except Exception:
        pass
    
    # 回退：尝试当作标量秒数（相对时间）
    try:
        return float(val)  # type: ignore[arg-type]
    except Exception as exc:
        raise TypeError("tmin/tmax must be float seconds (relative time) or Time-like object") from exc


def filter_time(ev: EventData, *, tmin: Optional[float] = None, tmax: Optional[float] = None) -> EventData:
    """按时间范围过滤 EventData（闭区间）。

    参数
    ----
    ev : EventData
        事件数据（time 为相对时间）
    tmin/tmax : float | Time | TimeDelta, optional
        时间范围限制：
        - float: 相对时间（秒），直接与 ev.time 比较
        - Time 对象: 自动转换为相对时间（需要 ev.timezero_obj）
        - TimeDelta: 取秒数作为相对时间
    
    返回
    ----
    EventData
        过滤后的副本（不修改原数据）
    
    示例
    ----
    >>> # 使用相对时间（秒）
    >>> ev_filtered = filter_time(ev, tmin=0, tmax=1000)
    >>> # 使用 Time 对象
    >>> from astropy.time import Time
    >>> t0 = Time('2024-01-01T00:00:00', format='isot')
    >>> t1 = Time('2024-01-01T00:10:00', format='isot')
    >>> ev_filtered = filter_time(ev, tmin=t0, tmax=t1)
    """
    t = np.asarray(ev.time, dtype=float)
    if t.size == 0:
        return ev
    mask = np.ones(t.size, dtype=bool)
    
    # 转换时间边界（支持 Time 对象）
    tmin_f = _coerce_time_bound(tmin, ev)
    tmax_f = _coerce_time_bound(tmax, ev)
    
    if tmin_f is not None:
        mask &= (t >= tmin_f)
    if tmax_f is not None:
        mask &= (t <= tmax_f)
    
    new_time = t[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]
    
    return EventData(
        path=ev.path, 
        time=new_time, 
        timezero=ev.timezero,
        timezero_obj=ev.timezero_obj,
        telescop=ev.telescop,
        pi=new_pi, channel=new_ch,
        x=new_x, y=new_y,
        gti_start=ev.gti_start, gti_stop=ev.gti_stop,
        gti_start_obj=ev.gti_start_obj, gti_stop_obj=ev.gti_stop_obj,
        gti=ev.gti,
        raw_columns=None,  # 过滤后不保留 raw_columns
        colmap=ev.colmap, energy=new_energy,
        ebounds=ev.ebounds,
        header=ev.header, meta=ev.meta, columns=ev.columns, headers_dump=ev.headers_dump
    )


def filter_energy(ev: EventData, *, pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> EventData:
    """Filter EventData by PI/CHANNEL range (inclusive). Prefer PI column if present."""
    # prefer PI then CHANNEL
    if ev.pi is not None:
        arr = np.asarray(ev.pi, dtype=int)
    elif ev.channel is not None:
        arr = np.asarray(ev.channel, dtype=int)
    else:
        # nothing to filter
        return ev
    if arr.size == 0:
        return ev
    mask = np.ones(arr.size, dtype=bool)
    if pi_min is not None:
        mask &= (arr >= int(pi_min))
    if pi_max is not None:
        mask &= (arr <= int(pi_max))
    new_time = np.asarray(ev.time, dtype=float)[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi, dtype=int)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel, dtype=int)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]
    return EventData(
        path=ev.path, 
        time=new_time,
        timezero=ev.timezero,
        timezero_obj=ev.timezero_obj,
        telescop=ev.telescop,
        pi=new_pi, channel=new_ch,
        x=new_x, y=new_y,
        gti_start=ev.gti_start, gti_stop=ev.gti_stop,
        gti_start_obj=ev.gti_start_obj, gti_stop_obj=ev.gti_stop_obj,
        gti=ev.gti,
        raw_columns=None,
        colmap=ev.colmap, energy=new_energy,
        ebounds=ev.ebounds,
        header=ev.header, meta=ev.meta, columns=ev.columns, headers_dump=ev.headers_dump
    )


def merge_gti(gti_start: Optional[np.ndarray], gti_stop: Optional[np.ndarray], *, tol: float = 1e-9) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """兼容包装：委托到 `gtimod.merge_gti`。"""
    return gtimod.merge_gti(gti_start, gti_stop, tol=tol)


def trim_events_to_gti(ev: EventData, *, tol: float = 1e-9) -> EventData:
    """根据 EventData 中的 GTI 裁剪事件。如果没有 GTI，返回原始对象副本。

    返回的新 `EventData` 会把 `gti_start/gti_stop` 规范化为合并后的区间。
    """
    if ev.gti_start is None or ev.gti_stop is None:
        return ev
    ms, me = merge_gti(ev.gti_start, ev.gti_stop, tol=tol)
    if ms is None or me is None:
        return ev

    t = np.asarray(ev.time, dtype=float)
    
    # 重新计算合并后 GTI 的 Time 对象
    gti_start_obj_new = None
    gti_stop_obj_new = None
    if ev.timezero_obj is not None and ms is not None and me is not None:
        try:
            from astropy.time import TimeDelta
            gti_start_obj_new = ev.timezero_obj + TimeDelta(ms, format='sec')
            gti_stop_obj_new = ev.timezero_obj + TimeDelta(me, format='sec')
        except Exception:
            pass
    
    if t.size == 0:
        # empty events, but update GTI fields
        return EventData(
            path=ev.path, time=t, 
            timezero=ev.timezero, timezero_obj=ev.timezero_obj, telescop=ev.telescop,
            pi=ev.pi, channel=ev.channel, x=ev.x, y=ev.y,
            gti_start=ms, gti_stop=me,
            gti_start_obj=gti_start_obj_new, gti_stop_obj=gti_stop_obj_new,
            gti=[(float(s), float(e)) for s, e in zip(ms, me)],
            raw_columns=None, colmap=ev.colmap, energy=ev.energy, ebounds=ev.ebounds,
            header=ev.header, meta=ev.meta, columns=ev.columns, headers_dump=ev.headers_dump
        )

    mask = np.zeros(t.size, dtype=bool)
    for s, e in zip(ms, me):
        mask |= (t >= float(s)) & (t <= float(e))

    new_time = t[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]

    return EventData(
        path=ev.path, time=new_time,
        timezero=ev.timezero, timezero_obj=ev.timezero_obj, telescop=ev.telescop,
        pi=new_pi, channel=new_ch, x=new_x, y=new_y,
        gti_start=ms, gti_stop=me,
        gti_start_obj=gti_start_obj_new, gti_stop_obj=gti_stop_obj_new,
        gti=[(float(s), float(e)) for s, e in zip(ms, me)],
        raw_columns=None, colmap=ev.colmap, energy=new_energy, ebounds=ev.ebounds,
        header=ev.header, meta=ev.meta, columns=ev.columns, headers_dump=ev.headers_dump
    )


def _read_column_from_evt(path: str | Path, colname: str):
    """从事件 FITS 文件中读取指定列，若不存在返回 None。"""
    p = Path(path)
    try:
        with fits.open(p) as h:
            hevt = None
            for ext in h:
                d = getattr(ext, 'data', None)
                if d is None:
                    continue
                cols = getattr(d, 'columns', None)
                if cols is not None and 'TIME' in cols.names:
                    hevt = ext
                    break
            if hevt is None:
                return None
            d = cast(Any, hevt).data
            if colname in d.columns.names:
                return np.asarray(d[colname])
    except Exception:
        return None
    return None

def clear_region(ev_or_path: EventData | str | Path, *, tmin: Optional[float] = None, tmax: Optional[float] = None,
                 pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> EventData:
    """Remove any region selection by re-reading the original event file and
    re-applying only time/energy filters.

    Notes:
    - If `ev_or_path` is an `EventData` and has a `path`, the full original
      events will be re-read from that file. If `ev_or_path` is a path string
      or `Path`, that file will be used.
    - If `tmin`/`tmax` or `pi_min`/`pi_max` are not provided and an
      `EventData` was passed, the function will infer them from the current
      `EventData` (i.e. keep the current time/energy constraints).
    - If no file path can be determined (in-memory `EventData` without
      `path`), a `ValueError` is raised because we cannot recover events
      excluded by the region filter.
    """
    # determine source path and optional current EventData
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_region requires a file path (pass EventData with .path or a file path)')

    # infer filters from provided EventData when arguments absent
    if ev is not None:
        if tmin is None and getattr(ev, 'time', None) is not None and len(ev.time) > 0:
            tmin = float(np.min(ev.time))
            tmax = float(np.max(ev.time)) if tmax is None else tmax
        if pi_min is None and getattr(ev, 'pi', None) is not None:
            arr = np.asarray(ev.pi, dtype=int)
            if arr.size > 0:
                pi_min = int(arr.min())
                pi_max = int(arr.max()) if pi_max is None else pi_max

    # re-read full event file and apply time/energy filters
    full = read_evt(path)
    return select_events(full, tmin=tmin, tmax=tmax, pi_min=pi_min, pi_max=pi_max)


def clear_time(ev_or_path: EventData | str | Path, *, region: Optional[dict | str | Path] = None,
               pi_min: Optional[int] = None, pi_max: Optional[int] = None) -> EventData:
    """Remove any time selection by re-reading the original event file and
    re-applying only region/energy filters.

    If `region` is not provided, the region cannot be inferred from a
    filtered `EventData` — in that case the returned EventData will have no
    region applied (only energy filters if provided or inferred).
    """
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_time requires a file path (pass EventData with .path or a file path)')

    # infer energy bounds from EventData if not supplied
    if ev is not None and pi_min is None and getattr(ev, 'pi', None) is not None:
        arr = np.asarray(ev.pi, dtype=int)
        if arr.size > 0:
            pi_min = int(arr.min())
            pi_max = int(arr.max()) if pi_max is None else pi_max

    full = read_evt(path)
    return select_events(full, region=region, pi_min=pi_min, pi_max=pi_max)


def clear_energy(ev_or_path: EventData | str | Path, *, region: Optional[dict | str | Path] = None,
                 tmin: Optional[float] = None, tmax: Optional[float] = None) -> EventData:
    """Remove any energy selection by re-reading the original event file and
    re-applying only region/time filters.
    """
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_energy requires a file path (pass EventData with .path or a file path)')

    # infer time window from EventData if not supplied
    if ev is not None and (tmin is None and getattr(ev, 'time', None) is not None and len(ev.time) > 0):
        tmin = float(np.min(ev.time))
        tmax = float(np.max(ev.time)) if tmax is None else tmax

    full = read_evt(path)
    return select_events(full, tmin=tmin, tmax=tmax, region=region)


def clear_all(ev_or_path: EventData | str | Path) -> EventData:
    """Return the complete unfiltered EventData by re-reading the source file.

    If `ev_or_path` is an `EventData` with a `path` attribute, that file will
    be re-read; if a path is provided it will be used directly. If neither is
    available, a `ValueError` is raised because the original full event set
    cannot be reconstructed from an in-memory, filtered `EventData`.
    """
    ev = ev_or_path if isinstance(ev_or_path, EventData) else None
    if ev is not None:
        path = ev.path
    elif isinstance(ev_or_path, (str, Path)):
        path = Path(ev_or_path)
    else:
        path = None
    if path is None:
        raise ValueError('clear_all requires a file path (pass EventData with .path or a file path)')
    return read_evt(path)


def filter_region(ev_or_path: EventData | str | Path, region: dict) -> EventData:
    """按给定 region（dict）过滤事件并返回新的 EventData。

    region 支持基本形式：
      - {'type':'circle', 'x': X, 'y': Y, 'r': R}
      - {'type':'annulus', 'x': X, 'y': Y, 'r_in': R1, 'r_out': R2}
      - {'type':'box', 'x': Xc, 'y': Yc, 'width': W, 'height': H}

    如果传入的是文件路径，会从 FITS 中读取必要列（默认尝试 'X'/'Y', 'RAWX'/'RAWY', 'DETX'/'DETY'）。
    """
    # normalize to EventData
    ev = None
    if isinstance(ev_or_path, (str, Path)):
        ev = read_evt(ev_or_path)
    else:
        ev = ev_or_path

    # discover X/Y columns; consult xselect.mdb defaults for instrument-specific mappings
    xs = None
    ys = None
    try:
        tree = _get_mdb_tree()
        mission = None
        instr = None
        mode = None
        if isinstance(ev_or_path, (str, Path)):
            tmp_ev = read_evt(ev_or_path)
            hdr = tmp_ev.header
        else:
            hdr = getattr(ev_or_path, 'header', None)
        if hdr is not None:
            mission = hdr.get('TELESCOP')
            instr = hdr.get('INSTRUME')
        defaults = xselect_mdb.get_defaults(tree, str(mission).upper() if mission is not None else '', str(instr).upper() if instr is not None else None, mode)
    except Exception:
        defaults = {}

    # order of preference: explicit mdb mappings, then common column names
    x_candidates = []
    y_candidates = []
    if defaults.get('x'):
        x_candidates.append(str(defaults.get('x')))
    if defaults.get('rawx'):
        x_candidates.append(str(defaults.get('rawx')))
    if defaults.get('detx'):
        x_candidates.append(str(defaults.get('detx')))
    x_candidates.extend(['X', 'X_IMAGE', 'RAWX', 'DETX', 'DET_X', 'XDET'])

    if defaults.get('y'):
        y_candidates.append(str(defaults.get('y')))
    if defaults.get('rawy'):
        y_candidates.append(str(defaults.get('rawy')))
    if defaults.get('dety'):
        y_candidates.append(str(defaults.get('dety')))
    y_candidates.extend(['Y', 'Y_IMAGE', 'RAWY', 'DETY', 'DET_Y', 'YDET'])

    xcol_used = None
    ycol_used = None
    for xc in x_candidates:
        try:
            xs = _read_column_from_evt(ev.path, xc)
        except Exception:
            xs = None
        if xs is not None:
            xcol_used = xc
            break
    for yc in y_candidates:
        try:
            ys = _read_column_from_evt(ev.path, yc)
        except Exception:
            ys = None
        if ys is not None:
            ycol_used = yc
            break

    # If EventData has attributes (unlikely), prefer them
    if hasattr(ev, 'x') and hasattr(ev, 'y'):
        xs = np.asarray(getattr(ev, 'x'))
        ys = np.asarray(getattr(ev, 'y'))

    if xs is None or ys is None:
        raise ValueError('Event file lacks X/Y columns required for region filtering')

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    typ = region.get('type', 'circle').lower()
    mask = np.zeros(xs.size, dtype=bool)
    if typ == 'circle':
        cx = float(region.get('x', 0.0))
        cy = float(region.get('y', 0.0))
        r = float(region['r'])
        dx = xs - cx
        dy = ys - cy
        mask = (dx * dx + dy * dy) <= (r * r)
    elif typ == 'annulus':
        cx = float(region.get('x', 0.0))
        cy = float(region.get('y', 0.0))
        rin = float(region['r_in'])
        rout = float(region['r_out'])
        dx = xs - cx
        dy = ys - cy
        rr = dx * dx + dy * dy
        mask = (rr >= rin * rin) & (rr <= rout * rout)
    elif typ == 'box':
        cx = float(region.get('x', 0.0))
        cy = float(region.get('y', 0.0))
        w = float(region.get('width', region.get('w', 1.0)))
        h = float(region.get('height', region.get('h', 1.0)))
        mask = (np.abs(xs - cx) <= (w / 2.0)) & (np.abs(ys - cy) <= (h / 2.0))
    else:
        raise ValueError(f'Unsupported region type: {typ}')

    # apply mask to build new EventData
    t = np.asarray(ev.time, dtype=float)
    new_time = t[mask]
    new_pi = None if ev.pi is None else np.asarray(ev.pi)[mask]
    new_ch = None if ev.channel is None else np.asarray(ev.channel)[mask]
    new_x = None if ev.x is None else np.asarray(ev.x)[mask]
    new_y = None if ev.y is None else np.asarray(ev.y)[mask]
    new_energy = None if ev.energy is None else np.asarray(ev.energy)[mask]

    return EventData(
        path=ev.path, time=new_time,
        timezero=ev.timezero, timezero_obj=ev.timezero_obj, telescop=ev.telescop,
        pi=new_pi, channel=new_ch, x=new_x, y=new_y,
        gti_start=ev.gti_start, gti_stop=ev.gti_stop,
        gti_start_obj=ev.gti_start_obj, gti_stop_obj=ev.gti_stop_obj,
        gti=ev.gti,
        raw_columns=None, colmap=ev.colmap, energy=new_energy, ebounds=ev.ebounds,
        header=ev.header, meta=ev.meta, columns=ev.columns, headers_dump=ev.headers_dump
    )


def extract_spectrum(ev_or_path: EventData | str | Path, *, region: dict | None = None,
                     tmin: Optional[float] = None, tmax: Optional[float] = None,
                     channel_col: str = 'pi', ch_min: Optional[int] = None, ch_max: Optional[int] = None,
                     nbins: Optional[int] = None) -> PhaData:
    """从事件（或事件文件）提取 PHA。支持时间与区域筛选。
    返回 PhaData。
    """
    # Delegate to jinwu.ftools.fextract.extract which implements the extraction
    # behavior in pure Python and mirrors the common fextract options.
    return ftools.fextract.extract(ev_or_path, region=region, tmin=tmin, tmax=tmax,
                                    ch_min=ch_min, ch_max=ch_max, nbins=nbins, channel_col=channel_col)


def _exposure_per_bins(ms: np.ndarray, me: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return gtimod.exposure_per_bins(ms, me, bins)


def extract_curve(ev_or_path: EventData | str | Path, *, binsize: float, tmin: Optional[float] = None,
                  tmax: Optional[float] = None) -> 'LightcurveData':
    """从事件生成光变曲线（counts per bin + per-bin exposure），返回 LightcurveData。

    binsize: bin 宽度（秒）
    """
    if isinstance(ev_or_path, (str, Path)):
        ev = read_evt(ev_or_path)
    else:
        ev = ev_or_path

    # time filter
    if tmin is not None or tmax is not None:
        ev = select_events(ev.path if isinstance(ev_or_path, (str, Path)) else ev.path, tmin=tmin, tmax=tmax)

    t = np.asarray(ev.time, dtype=float)
    if t.size == 0:
        from .file import LightcurveData
        # 空事件集时返回空的 LightcurveData，保持字段兼容
        empty = np.array([])
        return LightcurveData(
            path=ev.path,
            time=empty,
            value=empty,
            error=None,
            dt=binsize,
            exposure=0.0,
            bin_exposure=None,
            is_rate=False,
            # 与 LightcurveData 设计兼容的附加字段
            counts=empty,
            counts_err=None,
            bin_lo=empty,
            bin_hi=empty,
            gti_start=ev.gti_start,
            gti_stop=ev.gti_stop,
            columns=("TIME", "COUNTS"),
            header=ev.header,
            meta=ev.meta,
            headers_dump=ev.headers_dump,
        )

    tmin_eff = float(t.min())
    tmax_eff = float(t.max())
    nbins = max(1, int(np.ceil((tmax_eff - tmin_eff) / float(binsize))))
    edges = tmin_eff + np.arange(nbins + 1) * float(binsize)
    counts, _ = np.histogram(t, bins=edges)
    counts = counts.astype(float)

    # exposure per bin: use GTI if present
    if ev.gti_start is not None and ev.gti_stop is not None:
        ms, me = merge_gti(ev.gti_start, ev.gti_stop)
        if ms is None or me is None:
            expo = np.full(nbins, float(binsize))
        else:
            # attempt to apply adjustgti/frame alignment based on xselect.mdb defaults
            try:
                tree = _get_mdb_tree()
                # infer mission/instrument/mode from metadata/header
                adj, tp, frame_dt = _infer_adjustgti_timepixr_and_frame(ev, tree)
                if adj:
                    if frame_dt is None:
                        warnings.warn('adjustgti requested by xselect.mdb but frame_dt could not be inferred; skipping adjustgti')
                    else:
                        ms_adj, me_adj = gtimod.adjust_gti_to_frame(ms, me, frame_dt, timepixr=float(tp or 0.0))
                        if ms_adj is not None and me_adj is not None:
                            ms, me = ms_adj, me_adj
            except Exception:
                # conservative: if mdb parsing fails, proceed without adjust
                pass
            expo = _exposure_per_bins(ms, me, edges)
    else:
        expo = np.full(nbins, float(binsize))

    from .file import LightcurveData
    # 为了与当前 LightcurveData 设计兼容，这里同时填充 counts/counts_err、bin_lo/bin_hi、GTI 等字段
    return LightcurveData(
        path=ev.path,
        time=edges[:-1],
        value=counts,
        error=np.sqrt(counts),
        dt=binsize,
        exposure=float(np.sum(expo)),
        bin_exposure=expo,
        is_rate=False,
        counts=counts,
        counts_err=np.sqrt(counts),
        rate=None,
        rate_err=None,
        bin_lo=edges[:-1],
        bin_hi=edges[1:],
        tstart=float(edges[0]) if edges.size > 0 else None,
        tseg=float(edges[-1] - edges[0]) if edges.size > 0 else None,
        gti_start=ev.gti_start,
        gti_stop=ev.gti_stop,
        columns=("TIME", "COUNTS"),
        header=ev.header,
        meta=ev.meta,
        headers_dump=ev.headers_dump,
    )


def extract_image(ev_or_path: EventData | str | Path, *, xcol: str | None = None, ycol: str | None = None,
                  bins: tuple[int, int] = (64, 64), xrange: tuple[float, float] | None = None, yrange: tuple[float, float] | None = None,
                  tmin: Optional[float] = None, tmax: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从事件生成 2D 图像（numpy ndarray, xedges, yedges）。

    - xcol/ycol 可指定 FITS 中的列名；否则尝试常见列名。
    - bins: (nx, ny)
    - xrange/yrange: 可选范围，否则基于数据范围自动。
    """
    if isinstance(ev_or_path, (str, Path)):
        ev = read_evt(ev_or_path)
    else:
        ev = ev_or_path

    # time filter
    if tmin is not None or tmax is not None:
        ev = select_events(ev.path if isinstance(ev_or_path, (str, Path)) else ev.path, tmin=tmin, tmax=tmax)

    # find columns
    if xcol is None:
        x_candidates = ['X', 'X_IMAGE', 'RAWX', 'DETX']
        xval = None
        for xc in x_candidates:
            xval = _read_column_from_evt(ev.path, xc)
            if xval is not None:
                xcol = xc
                break
    else:
        xval = _read_column_from_evt(ev.path, xcol)

    if ycol is None:
        y_candidates = ['Y', 'Y_IMAGE', 'RAWY', 'DETY']
        yval = None
        for yc in y_candidates:
            yval = _read_column_from_evt(ev.path, yc)
            if yval is not None:
                ycol = yc
                break
    else:
        yval = _read_column_from_evt(ev.path, ycol)

    if xval is None or yval is None:
        raise ValueError('Cannot find X/Y columns for image extraction')

    x = np.asarray(xval, dtype=float)
    y = np.asarray(yval, dtype=float)

    # apply optional time filter to arrays (if select_events changed ev.time)
    t = np.asarray(ev.time, dtype=float)
    if t.size != x.size:
        # align by reading time from FITS directly and filtering
        tcol = _read_column_from_evt(ev.path, 'TIME')
        if tcol is not None:
            t = np.asarray(tcol, dtype=float)
    # Build mask if select_events applied
    if tmin is not None or tmax is not None:
        mask = np.ones(x.size, dtype=bool)
        if tmin is not None:
            mask &= (t >= float(tmin))
        if tmax is not None:
            mask &= (t <= float(tmax))
        x = x[mask]
        y = y[mask]

    if xrange is None:
        xmin, xmax = float(x.min()) if x.size else 0.0, float(x.max()) if x.size else 1.0
    else:
        xmin, xmax = float(xrange[0]), float(xrange[1])
    if yrange is None:
        ymin, ymax = float(y.min()) if y.size else 0.0, float(y.max()) if y.size else 1.0
    else:
        ymin, ymax = float(yrange[0]), float(yrange[1])

    nx, ny = int(bins[0]), int(bins[1])
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)
    img, xe, ye = np.histogram2d(y, x, bins=[yedges, xedges])
    # note: histogram2d returns array shape (ny, nx) with first axis y
    return img, xedges, yedges


def _estimate_exposure_from_eventdata(ev: EventData) -> float:
    """尝试从 EventData 中估算曝光：优先使用合并后的 GTI 累计时长，其次使用 meta 中的 TSTART/TSTOP，最后用事件跨度。

    该函数会调用 `merge_gti` 来规范化 GTI 区间。
    """
    if ev.gti_start is not None and ev.gti_stop is not None:
        ms, me = merge_gti(ev.gti_start, ev.gti_stop)
        if ms is not None and me is not None and ms.size > 0:
            return float(np.sum(me - ms))
    if ev.meta is not None and isinstance(ev.meta, OgipMeta):
        if ev.meta.tstart is not None and ev.meta.tstop is not None:
            return float(ev.meta.tstop - ev.meta.tstart)
    # fallback: use span of events
    if ev.time.size:
        return float(ev.time.max() - ev.time.min())
    return 0.0


def accumulate_spectrum_from_events(ev: EventData, *, channel_col: str = 'pi',
                                    ch_min: Optional[int] = None, ch_max: Optional[int] = None,
                                    nbins: Optional[int] = None) -> PhaData:
    """从 EventData 累积 PHA（计数直方）。

    参数
    - ev: 已读取或筛选的 EventData
    - channel_col: 首选的能道列，'pi' 或 'channel'
    - ch_min/ch_max: 道号截断范围（包含）
    - nbins: 如果提供，输出为 [0..nbins-1] 的道计数（否则基于数据自动）

    返回
    - PhaData 实例（未自动写入磁盘）
    """
    if channel_col not in ('pi', 'channel'):
        raise ValueError("channel_col must be 'pi' or 'channel'")

    arr = None
    if channel_col == 'pi' and ev.pi is not None:
        arr = np.asarray(ev.pi, dtype=int)
    elif channel_col == 'channel' and ev.channel is not None:
        arr = np.asarray(ev.channel, dtype=int)
    else:
        # try the other one
        if ev.pi is not None:
            arr = np.asarray(ev.pi, dtype=int)
        elif ev.channel is not None:
            arr = np.asarray(ev.channel, dtype=int)
        else:
            raise ValueError('EventData contains no PI or CHANNEL column')

    if arr.size == 0:
        # empty spectrum
        channels = np.array([], dtype=int)
        counts = np.array([], dtype=float)
        exposure = _estimate_exposure_from_eventdata(ev)
        return PhaData(path=ev.path, channels=channels, counts=counts,
                   stat_err=None, exposure=exposure, backscal=None, areascal=None,
                   quality=None, grouping=None, ebounds=None, header=ev.header, meta=ev.meta,
                   headers_dump=ev.headers_dump, columns=())

    # apply optional truncation
    if ch_min is not None:
        arr = arr[arr >= int(ch_min)]
    if ch_max is not None:
        arr = arr[arr <= int(ch_max)]

    if nbins is None:
        # choose bins from min..max inclusive
        ch_lo = int(arr.min())
        ch_hi = int(arr.max())
        nbins = ch_hi - ch_lo + 1
        bins = np.arange(ch_lo, ch_hi + 2, dtype=int)
        channels = np.arange(ch_lo, ch_hi + 1, dtype=int)
        counts, _ = np.histogram(arr, bins=bins)
    else:
        # fixed nbins: assume channels start at 0
        bins = np.arange(0, int(nbins) + 1, dtype=int)
        channels = np.arange(0, int(nbins), dtype=int)
        counts, _ = np.histogram(arr, bins=bins)

    # poisson stat error (sqrt) as default
    stat_err = np.sqrt(counts.astype(float))
    exposure = _estimate_exposure_from_eventdata(ev)

    pha = PhaData(
        path=ev.path,
        channels=channels, counts=counts.astype(float), stat_err=stat_err,
        exposure=exposure, backscal=None, areascal=None,
        quality=None, grouping=None, ebounds=None,
        header=ev.header, meta=ev.meta, headers_dump=ev.headers_dump, columns=('CHANNEL','COUNTS')
    )
    return pha


def write_pha(pha: PhaData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """将 PhaData 写出为 OGIP 风格的 PHA FITS 文件（包含 SPECTRUM 扩展）。

    注意：此函数只写基础列 (`CHANNEL`, `COUNTS`, `STAT_ERR`)，并在主头中写入 `EXPOSURE`。
    更复杂的 OGIP 字段（BACKSCAL、AREASCAL、EBOUNDS 等）可在后续迭代中支持。
    """
    outp = Path(outpath)
    if outp.exists() and not overwrite:
        raise FileExistsError(str(outp))

    cols = []
    ch = np.asarray(pha.channels, dtype=int)
    cnt = np.asarray(pha.counts, dtype=float)
    cols.append(fits.Column(name='CHANNEL', format='J', array=ch))
    cols.append(fits.Column(name='COUNTS', format='E', array=cnt))
    if pha.stat_err is not None:
        cols.append(fits.Column(name='STAT_ERR', format='E', array=np.asarray(pha.stat_err, dtype=float)))
    if pha.quality is not None:
        cols.append(fits.Column(name='QUALITY', format='J', array=np.asarray(pha.quality, dtype=int)))
    if pha.grouping is not None:
        cols.append(fits.Column(name='GROUPING', format='J', array=np.asarray(pha.grouping, dtype=int)))

    hdu_spec = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
    # add standard OGIP header keywords
    hdr = hdu_spec.header
    hdr['EXTNAME'] = 'SPECTRUM'
    hdr['TELESCOP'] = pha.meta.telescop if (pha.meta is not None and getattr(pha.meta, 'telescop', None)) else hdr.get('TELESCOP', 'UNKNOWN')
    hdr['INSTRUME'] = pha.meta.instrume if (pha.meta is not None and getattr(pha.meta, 'instrume', None)) else hdr.get('INSTRUME', 'UNKNOWN')
    # Exposure and time range
    if pha.exposure is not None:
        hdr['EXPOSURE'] = float(pha.exposure)
    if pha.meta is not None:
        if getattr(pha.meta, 'tstart', None) is not None:
            prih_tstart = float(pha.meta.tstart)
        else:
            prih_tstart = None
        if getattr(pha.meta, 'tstop', None) is not None:
            prih_tstop = float(pha.meta.tstop)
        else:
            prih_tstop = None
    else:
        prih_tstart = prih_tstop = None

    # BACKSCAL, AREASCAL and RESPFILE go into the SPECTRUM header if provided.
    # For RESPFILE we write the provided path or a conservative placeholder 'NONE'.
    if pha.backscal is not None:
        hdr['BACKSCAL'] = float(pha.backscal)
    if pha.areascal is not None:
        hdr['AREASCAL'] = float(pha.areascal)
    # RESPFILE / ANCRFILE: reference to response/ancillary files. Use placeholder 'NONE' when missing.
    try:
        resp_val = str(pha.respfile) if getattr(pha, 'respfile', None) is not None else 'NONE'
        hdr['RESPFILE'] = resp_val
    except Exception:
        pass
    try:
        ancr_val = str(pha.ancrfile) if getattr(pha, 'ancrfile', None) is not None else 'NONE'
        hdr['ANCRFILE'] = ancr_val
    except Exception:
        pass

    # Primary HDU
    prih = fits.PrimaryHDU()
    try:
        if pha.meta is not None:
            if getattr(pha.meta, 'instrume', None):
                prih.header['INSTRUME'] = pha.meta.instrume
            if getattr(pha.meta, 'telescop', None):
                prih.header['TELESCOP'] = pha.meta.telescop
            if prih_tstart is not None:
                prih.header['TSTART'] = prih_tstart
            if prih_tstop is not None:
                prih.header['TSTOP'] = prih_tstop
            if getattr(pha.meta, 'obs_id', None) is not None:
                prih.header['OBS_ID'] = pha.meta.obs_id
            # Also mirror RESPFILE/ANCRFILE into primary header when available (helps some tools)
            try:
                if getattr(pha, 'respfile', None) is not None:
                    prih.header['RESPFILE'] = str(pha.respfile)
                else:
                    prih.header.setdefault('RESPFILE', 'NONE')
            except Exception:
                pass
            try:
                if getattr(pha, 'ancrfile', None) is not None:
                    prih.header['ANCRFILE'] = str(pha.ancrfile)
                else:
                    prih.header.setdefault('ANCRFILE', 'NONE')
            except Exception:
                pass
    except Exception:
        pass

    hdul = fits.HDUList([prih, hdu_spec])

    # Add EBOUNDS extension if ebounds present
    if getattr(pha, 'ebounds', None) is not None:
        ebounds_tuple = pha.ebounds
        if ebounds_tuple is not None and len(ebounds_tuple) == 3:
            ch_eb, emin, emax = ebounds_tuple
            cols_eb = [fits.Column(name='CHANNEL', format='J', array=np.asarray(ch_eb, dtype=int)),
                       fits.Column(name='E_MIN', format='E', array=np.asarray(emin, dtype=float)),
                       fits.Column(name='E_MAX', format='E', array=np.asarray(emax, dtype=float))]
            hdu_eb = fits.BinTableHDU.from_columns(cols_eb, name='EBOUNDS')
            hdul.append(hdu_eb)

    # Optionally add GTI extension if metadata contains TSTART/TSTOP and more detailed GTI info in header
    # We check for header keys GTI_START/GTI_STOP (deprecated) or pha.meta.tstart/tstop
    # Note: PhaData does not have gti_start/gti_stop attributes, only EventData and LightcurveData do
    # This section is kept for possible future extensions or custom PhaData subclasses
    gti_start = getattr(pha, 'gti_start', None)
    gti_stop = getattr(pha, 'gti_stop', None)
    if gti_start is not None and gti_stop is not None:
        try:
            gs = np.asarray(gti_start, dtype=float)
            ge = np.asarray(gti_stop, dtype=float)
            cols_g = [fits.Column(name='START', format='D', array=gs), fits.Column(name='STOP', format='D', array=ge)]
            hdu_g = fits.BinTableHDU.from_columns(cols_g, name='GTI')
            hdul.append(hdu_g)
        except Exception:
            pass

    hdul.writeto(outp, overwrite=overwrite)
    return outp
