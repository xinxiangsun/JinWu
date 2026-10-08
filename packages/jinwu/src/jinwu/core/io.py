from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union, cast, overload

import numpy as np
from astropy.io import fits

from .base import ChannelBand, EnergyBand, FitsHeaderDump, HduHeader, OgipMeta, RegionArea
from .data import ArfData, EventData, LightcurveData, PhaData, RmfData
from .ogip import ValidationReport


def band_from_arf_bins(arf_path: str | Path, bin_lo: int = 81, bin_hi: int = 780) -> EnergyBand:
    """按 ARF 行号取能段 / Derive an energy band from ARF row indices.

    bin_lo/bin_hi 为包含端点的 1-based 行号，默认 81/780；分别取
    ENERG_LO 的下界与 ENERG_HI 的上界。只对下界负索引与上界过大作
    单侧截断，调用者须提供有效非空范围。返回 EnergyBand 并标记 keV，
    不读取 TUNIT 或自动转换能量单位。读取/索引异常向上传递。
    Use inclusive 1-based rows (defaults 81/780), selecting lower/upper energy
    edges. Only low-index underflow/high-index overflow are clipped on one side;
    callers ensure a valid nonempty range. Return EnergyBand labeled keV without
    TUNIT conversion. FITS/indexing errors propagate."""
    with fits.open(arf_path) as h:
        hd = cast(Any, h["SPECRESP"])
        d = hd.data
        elo = np.asarray(d["ENERG_LO"], float)
        ehi = np.asarray(d["ENERG_HI"], float)
    i0 = max(0, int(bin_lo) - 1)
    i1 = min(ehi.size - 1, int(bin_hi) - 1)
    emin = float(elo[i0])
    emax = float(ehi[i1])
    return EnergyBand(emin=emin, emin_unit="keV", emax=emax, emax_unit="keV")


def channel_mask_from_ebounds(
    ebounds: tuple[np.ndarray, np.ndarray, np.ndarray],
    band: EnergyBand,
    ch_band: Optional[ChannelBand] = None,
) -> np.ndarray:
    """按能量交叠与可选通道范围选道 / Select channels overlapping an energy band.

    ebounds=(channel, e_lo, e_hi)；采用 e_hi>emin 且 e_lo<emax 的严格
    交叠规则，再按包含端点的 ch_band 限制。返回同形状 bool 数组。
    band 数值须与 EBOUNDS 已同单位；不读取 band 的单位字符串换算。
    Use e_hi>emin and e_lo<emax, then optional inclusive channel limits. Return
    boolean mask of matching shape. Band numbers must already match EBOUNDS
    units; the band's unit strings are not converted here."""
    ch, e_lo, e_hi = ebounds
    mask = (e_hi > float(band.emin)) & (e_lo < float(band.emax))
    if ch_band is not None:
        mask &= (ch >= int(ch_band.ch_lo)) & (ch <= int(ch_band.ch_hi))
    return mask


def _combine_mjdref(header: Dict[str, Any]) -> Optional[float]:
    """读取合并的 MJDREF / Read a combined MJD reference from a header.

    MJDREFI/MJDREFF 任一存在时优先相加，缺项按 0；否则尝试 MJDREF。
    无字段或单字段转换失败返回 None；分拆字段转换异常可能向上传递。
    不进行 UTC/TT 等时间尺度转换。
    Prefer split MJDREFI+MJDREFF with absent components as 0, otherwise MJDREF.
    Missing/direct-conversion failure returns None; split-field conversion errors
    may propagate. No UTC/TT or other time-scale conversion is performed."""
    if header is None:
        return None
    if ("MJDREFI" in header) or ("MJDREFF" in header):
        mjdi = float(header.get("MJDREFI", 0.0))
        mjdf = float(header.get("MJDREFF", 0.0))
        return mjdi + mjdf
    if "MJDREF" in header:
        try:
            return float(header["MJDREF"])
        except Exception:
            return None
    return None


def _first_non_empty(keys: list[str], *headers: Dict[str, Any]) -> Optional[Any]:
    """按键优先级读取首个非空值 / Find the first nonempty value by key priority.

    先遍历 keys，再按 headers 顺序；排除 None、空串和单个空格。
    键大小写敏感，零值保留；无匹配返回 None，不校验数值有效性。
    Iterate keys first, then headers. Exclude None, empty string and one space;
    keep zero. Lookup is case-sensitive, with None if absent and no value validation."""
    for key in keys:
        for hdr in headers:
            if hdr is None:
                continue
            if key in hdr and hdr[key] not in (None, "", " "):
                return hdr[key]
    return None


def _collect_headers_dump(hdul: fits.HDUList) -> FitsHeaderDump:
    """保存各 HDU 头的字典快照 / Collect primary/extension header dictionary snapshots.

    返回 FitsHeaderDump，扩展保留名称及可解析 EXTVER；无法解析版本为 None。
    不读取表内容，不承诺保留重复 FITS 卡片、顺序和注释的无损表示。
    Return FitsHeaderDump with extension names and parsed EXTVER or None. No table
    reading; dict conversion is not a lossless preservation of repeated cards,
    header order or card comments."""
    hdr0 = cast(Any, getattr(hdul[0], 'header', {})) if len(hdul) > 0 else {}
    primary = dict(hdr0) if hdr0 else {}
    exts: list[HduHeader] = []
    for hdu in hdul[1:]:
        hdu_any = cast(Any, hdu)
        name = str(getattr(hdu, 'name', '') or '')
        ver_val = cast(Any, hdu_any.header).get('EXTVER', None)
        try:
            ver = int(ver_val) if ver_val is not None else None
        except Exception:
            ver = None
        exts.append(HduHeader(name=name, ver=ver, header=dict(cast(Any, hdu_any.header))))
    return FitsHeaderDump(primary=primary, extensions=exts)


def _build_meta(hdul: fits.HDUList, prefer_header: Optional[Dict[str, Any]]) -> OgipMeta:
    """按优先头提取 OGIP 元数据 / Build OGIP metadata from prioritized headers.

    prefer_header 优先于 primary 和其他扩展；具体别名由键优先级决定。
    返回 OgipMeta，尽量解析任务、时间和观测字段，缺失值为 None；
    不把 TIMEUNIT/TIMESYS/MJDREF 转换为统一时间对象，也不证明字段一致。
    Prefer the supplied header, then primary and remaining extensions, with key
    aliases ordered by priority. Return OgipMeta with parsed identity/time fields
    or None. No unified time conversion or cross-header consistency validation;
    some malformed split-reference conversions can still raise."""
    dump = _collect_headers_dump(hdul)
    primary = dump.primary
    other_ext_headers = [x.header for x in dump.extensions]
    telescop = _first_non_empty(["TELESCOP"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    instrume = _first_non_empty(["INSTRUME"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    detnam = _first_non_empty(["DETNAM", "DETNAME"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    timesys = _first_non_empty(["TIMESYS"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    timeunit = _first_non_empty(["TIMEUNIT"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))

    mjdref = None
    for hdr in [prefer_header, primary] + other_ext_headers:
        if hdr is None:
            continue
        mjdref = _combine_mjdref(hdr)
        if mjdref is not None:
            break

    tstart = None
    tstop = None
    for hdr in [prefer_header, primary] + other_ext_headers:
        if hdr is None:
            continue
        if (tstart is None) and ("TSTART" in hdr):
            try:
                tstart = float(hdr["TSTART"])
            except Exception:
                pass
        if (tstop is None) and ("TSTOP" in hdr):
            try:
                tstop = float(hdr["TSTOP"])
            except Exception:
                pass
        if (tstart is not None) and (tstop is not None):
            break

    obj = _first_non_empty(["OBJECT"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    obs_id = _first_non_empty(["OBS_ID", "OBS_ID"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))

    binsize_val = _first_non_empty(
        ["BIN_SIZE", "BINSIZE", "TIMEDEL", "DELTAT", "TBIN", "TIMEBIN"],
        *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr),
    )
    try:
        binsize = float(binsize_val) if binsize_val is not None else None
    except Exception:
        binsize = None

    timezero_val = _first_non_empty(["TIMEZERO"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        timezero = float(timezero_val) if timezero_val is not None else None
    except Exception:
        timezero = None

    trefpos_val = _first_non_empty(["TREFPOS", "TREFDIR"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        trefpos = str(trefpos_val) if trefpos_val is not None else None
    except Exception:
        trefpos = None

    dateobs_val = _first_non_empty(["DATE-OBS", "DATE_OBS"], *(hdr for hdr in [prefer_header, primary] + other_ext_headers if hdr))
    try:
        dateobs = str(dateobs_val) if dateobs_val is not None else None
    except Exception:
        dateobs = None

    return OgipMeta(
        telescop=str(telescop) if telescop is not None else None,
        instrume=str(instrume) if instrume is not None else None,
        detnam=str(detnam) if detnam is not None else None,
        timesys=str(timesys) if timesys is not None else None,
        timeunit=str(timeunit) if timeunit is not None else None,
        mjdref=float(mjdref) if mjdref is not None else None,
        tstart=float(tstart) if tstart is not None else None,
        tstop=float(tstop) if tstop is not None else None,
        object=str(obj) if obj is not None else None,
        obs_id=str(obs_id) if obs_id is not None else None,
        binsize=binsize,
        timezero=timezero,
        trefpos=trefpos,
        dateobs=dateobs,
    )


def _mission_timezero_object(telescop: Optional[str], timezero: float, *, allow_unix_fallback: bool = False):
    """由任务秒数构造绝对时间 / Construct a mission TIMEZERO object.

    已知 telescop 委托 time_from_mission_seconds；无任务名返回 None。
    未知任务在 allow_unix_fallback=True 时尝试 Unix UTC，否则 RuntimeError。
    回退只是兼容解释，不证明秒数确实是 Unix，调用方须核实时间元数据。
    Delegate known missions to time_from_mission_seconds; absent name returns None.
    Unknown missions may use Unix UTC when allowed, otherwise raise RuntimeError.
    A fallback interpretation does not prove the input is Unix; callers verify metadata."""
    if telescop is None:
        return None
    from .time import Time, time_from_mission_seconds

    mission_time = time_from_mission_seconds(telescop, timezero)
    if mission_time is not None:
        return mission_time
    if allow_unix_fallback:
        try:
            return Time(timezero, format='unix', scale='utc')
        except Exception:
            return None
    raise RuntimeError(f"Unknown telescope '{telescop}' for time conversion; "
                       "please report the related header keywords to the authors")


# 方法：GTI 扩展识别：扩展名（EXTNAME/HDU 名）='GTI'，时间列按 START/STOP 读取
#       （TSTART/TSTOP 仅作兼容别名，标准名优先）。
# 参考：OGIP/93-003 "The Proposed Timing FITS File Format for High Energy
#       Astrophysics Data"（Angelini, Pence & Tennant, Legacy 3, 32）——
#       GTI 扩展含 START/STOP 两列。
def _extract_gti(hdul: fits.HDUList) -> Optional[list[tuple[float, float]]]:
    """读取第一个可解析 GTI 表 / Extract the first parseable GTI table.

    匹配 GTI 扩展名，START/STOP 优先于 TSTART/TSTOP；返回 float 起止
    元组列表。无合适扩展返回 None；数值转换失败也返回 None。不排序、
    合并、校验区间或施加 TIMEZERO，结果保留表内时间参考与单位。
    Match GTI name and prefer START/STOP over TSTART/TSTOP. Return float boundary
    tuples or None for absent/unparseable data. No sorting, merging, interval
    validation or TIMEZERO shift; preserve table time reference/unit."""
    if hdul is None:
        return None
    for hdu in hdul:
        hdr = getattr(hdu, 'header', {})
        name = (hdr.get('EXTNAME') or '').upper() if hdr else ''
        if name == 'GTI' or getattr(hdu, 'name', '').upper() == 'GTI':
            data = getattr(hdu, 'data', None)
            if data is None:
                continue
            cols = getattr(data, 'columns', None)
            colnames = [n.upper() for n in (cols.names if cols is not None else [])]
            start_col = None
            stop_col = None
            for n in ['START', 'TSTART']:
                if n in colnames:
                    start_col = n
                    break
            for n in ['STOP', 'TSTOP']:
                if n in colnames:
                    stop_col = n
                    break
            if start_col and stop_col:
                arr_start = data[start_col]
                arr_stop = data[stop_col]
                try:
                    return [(float(s), float(e)) for s, e in zip(arr_start, arr_stop)]
                except (TypeError, ValueError):
                    return None
    return None


def _load_regions(hdul: fits.HDUList) -> Optional[RegionArea]:
    """从 REG00101 推断代表区域 / Infer a representative region from REG00101.

    仅处理圆/环面积（半径原单位的平方），根据 COMPONENT 或形状/大小
    启发式分类；优先返回首个 src，再 bkg，再首个未知区域。返回 RegionArea
    或 None，并非全部区域并集，不读取单位换算、重叠或遮挡校正。
    Compute circle/annulus area in squared native radius units and infer roles by
    COMPONENT or shape/size heuristics. Prefer first source, then background, then
    unknown; return RegionArea or None. This is not a union of all regions and
    performs no unit conversion, overlap handling or detector-mask correction."""
    reg_hdu = None
    for ext in hdul:
        name = (getattr(ext, 'name', '') or '').upper()
        if name == 'REG00101':
            reg_hdu = ext
            break
    if reg_hdu is None or getattr(reg_hdu, 'data', None) is None:
        return None
    reg_any = cast(Any, reg_hdu)
    if getattr(reg_any, 'data', None) is None:
        return None
    data = reg_any.data
    cols = getattr(data, 'columns', None)
    colnames = [str(n).upper() for n in (cols.names if cols is not None else [])]

    def _get_col(name_variants: list[str]) -> Optional[str]:
        """查找首个存在的候选列 / Return the first candidate in uppercase column names.

        无匹配返回 None / Return None if absent."""
        for nn in name_variants:
            if nn in colnames:
                return nn
        return None

    shape_col = _get_col(['SHAPE'])
    component_col = _get_col(['COMPONENT'])
    r_col = _get_col(['R', 'RADIUS', 'R0'])
    rin_col = _get_col(['R_IN', 'RIN', 'R1'])
    rout_col = _get_col(['R_OUT', 'ROUT', 'R2'])
    nrows = len(data)
    rows_info: list[Dict[str, Any]] = []

    def _as_float(v: Any) -> Optional[float]:
        """读取标量或数组首项 / Parse the first scalar/array element as float.

        None、空值或转换异常返回 None，不验证有限性。
        Return None for missing/empty/unconvertible input; no finiteness validation."""
        if v is None:
            return None
        try:
            arr = np.asarray(v)
            if arr.size == 0:
                return None
            return float(arr.reshape(-1)[0])
        except Exception:
            return None

    for i in range(nrows):
        shape_val = ''
        if shape_col and shape_col in colnames:
            try:
                shape_val = str(data[shape_col][i]).upper().strip()
            except Exception:
                shape_val = ''
        comp_val = None
        if component_col and component_col in colnames:
            tmp = _as_float(data[component_col][i])
            comp_val = int(tmp) if tmp is not None else None
        area = None
        if (shape_val == 'CIRCLE') or (shape_col is None and r_col is not None and (rin_col is None or rout_col is None)):
            rv = data[r_col][i] if r_col else None
            r = _as_float(rv) if rv is not None else None
            if r is not None and r > 0:
                area = np.pi * (r ** 2)
        elif shape_val == 'ANNULUS' or (rin_col is not None and rout_col is not None):
            rin = _as_float(data[rin_col][i]) if rin_col else None
            rout = _as_float(data[rout_col][i]) if rout_col else None
            if rin is None and rout is None and r_col:
                rv = data[r_col][i]
                try:
                    rv_arr = np.asarray(rv).reshape(-1)
                    rin = _as_float(rv_arr[0]) if rv_arr.size >= 1 else None
                    rout = _as_float(rv_arr[1]) if rv_arr.size >= 2 else None
                except Exception:
                    rin = None
                    rout = None
            if (rin is not None) and (rout is not None) and rout > rin:
                area = np.pi * (rout ** 2 - rin ** 2)
        rows_info.append({'shape': shape_val, 'area': area, 'component': comp_val, 'role': 'unknown'})
    if not rows_info:
        return None
    annuli = [r for r in rows_info if r['shape'] == 'ANNULUS' and r['area']]
    circles = [r for r in rows_info if r['shape'] == 'CIRCLE' and r['area']]
    if component_col and any(r['component'] is not None for r in rows_info):
        for r in rows_info:
            comp = r['component']
            if comp == 1:
                r['role'] = 'source'
            elif comp is not None:
                r['role'] = 'background'
    if not component_col:
        if annuli:
            for r in annuli:
                if r['role'] == 'unknown':
                    r['role'] = 'background'
            if circles:
                for r in circles:
                    if r['role'] == 'unknown':
                        r['role'] = 'source'
        elif len(circles) >= 1:
            sorted_c = sorted(circles, key=lambda x: x['area'] or 0)
            if sorted_c:
                if sorted_c[0]['role'] == 'unknown':
                    sorted_c[0]['role'] = 'source'
                for r in sorted_c[1:]:
                    if r['role'] == 'unknown':
                        r['role'] = 'background'
    if len(rows_info) == 1 and rows_info[0]['role'] == 'unknown':
        rows_info[0]['role'] = 'source'

    def _normalize_role(role: str) -> Literal['src', 'bkg', 'unk']:
        """统一区域角色字符串 / Map source/background to src/bkg, all others to unk."""
        if role == 'source':
            return 'src'
        if role == 'background':
            return 'bkg'
        return 'unk'

    regions = [
        RegionArea(role=_normalize_role(cast(Any, r['role'])), shape=r['shape'] or None, area=r['area'], component=r['component'])
        for r in rows_info
    ]
    if not regions:
        return None
    src_region = next((r for r in regions if r.role == 'src'), None)
    if src_region is not None:
        return src_region
    bkg_region = next((r for r in regions if r.role == 'bkg'), None)
    if bkg_region is not None:
        return bkg_region
    return regions[0]


def _opt_int(value: Any) -> Optional[int]:
    """宽容解析整数头字段 / Parse an optional integer keyword tolerantly.

    先 float 再 int，小数会截断；空值或转换失败返回 None，不验证整值性。
    Convert float then int, truncating fractions. Empty/invalid values return None;
    this is not an integer-value validation."""
    try:
        if value is None or str(value).strip() == '':
            return None
        return int(float(value))
    except Exception:
        return None


# 方法：F_CHAN 的 TLMIN 采用"文件声明优先、缺失回退推断"：先读 TLMINn（F_CHAN 列的
#       TLMIN，标准列布局下为第 4 列即 TLMIN4），无声明时取 F_CHAN 列最小值作为首道号。
# 参考：HEASoft 6.37 heacore/heasp/rmf.cxx rmf::readMatrix（先读 TLMIN<F_CHAN 列索引>，
#       读不到时回退 min(F_CHAN)：最小值为 0 则取 0，否则取 1）。
def _infer_rmf_tlmin(header: Mapping[str, Any], f_chan: Optional[np.ndarray]) -> Optional[int]:
    """读取或推断 RMF 首通道 / Read or infer a response first-channel value.

    优先头字典中首个可解析的 TLMIN*（排除 TLMIN1），否则取 F_CHAN
    数组的实际最小值。当前未将 TLMIN 卡与具体列名逐一关联，调用者须
    确认文件声明适用；失败或无数据返回 None。
    Prefer the first parseable TLMIN* except TLMIN1, otherwise use the actual
    minimum F_CHAN. The current implementation does not associate each card with
    its column name; callers confirm the declaration. Return None if unavailable."""
    for k, v in dict(header).items():
        ku = str(k).upper()
        if ku.startswith('TLMIN') and ku != 'TLMIN1':
            got = _opt_int(v)
            if got is not None:
                return got
    if f_chan is not None:
        try:
            if isinstance(f_chan, np.ndarray) and f_chan.dtype == object:
                vals = np.concatenate([np.atleast_1d(np.asarray(r, dtype=int)) for r in f_chan if np.size(r) > 0])
            else:
                vals = np.atleast_1d(np.asarray(f_chan, dtype=int))
            if vals.size > 0:
                return int(vals.min())
        except Exception:
            pass
    return None


class OgipArfReader:
    def __init__(self, path: str | Path):
        """保存 ARF 读取路径 / Initialize a ARF reader with a local path.

        只检查路径存在，否则 FileNotFoundError；不展开 ~、读取表或验证类型。
        Check path existence only, raising FileNotFoundError. No tilde expansion,
        table reading or product-type validation; initialize the cached data as None."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[ArfData] = None

    def read(self) -> ArfData:
        """读取 ARF 并缓存数据对象 / Read the SPECRESP ARF and cache ArfData.

        提取 ENERG_LO/HI、SPECRESP、头与元数据；裸数沿用文件单位约定，
        不做 TUNIT 换算。每次调用重新打开文件并替换 _data，不自动 validate。
        Read energy bounds, effective area and metadata using native file-unit values;
        no TUNIT conversion. Each call reopens the file and replaces _data without
        calling validate. Missing required extensions/columns and FITS errors propagate."""
        with fits.open(self.path) as h:
            hd = cast(Any, h["SPECRESP"])
            d = hd.data
            columns = tuple(getattr(d.columns, 'names', ()) or ())
            energ_lo = np.asarray(d["ENERG_LO"], float)
            energ_hi = np.asarray(d["ENERG_HI"], float)
            specresp = np.asarray(d["SPECRESP"], float)
            header = dict(cast(Any, hd.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

        self._data = ArfData(
            path=self.path,
            energ_lo=energ_lo,
            energ_hi=energ_hi,
            specresp=specresp,
            columns=columns,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
            hduvers=(str(header.get("HDUVERS")).strip() or None) if header.get("HDUVERS") is not None else None,
        )
        return self._data

    def validate(self) -> ValidationReport:
        """读取后委托对象校验 / Read if needed, then delegate to the data object's validate.

        已有 _data 时不重新读取文件；返回 ValidationReport，具体检查由数据类
        定义。校验通过不证明响应、时间系统或科学结果普遍有效。
        Reuse cached data if present without rereading disk. Return ValidationReport
        from the data class; passing its checks is not general scientific certification."""
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipRmfReader:
    def __init__(self, path: str | Path):
        """保存 RMF 读取路径 / Initialize a RMF reader with a local path.

        只检查路径存在，否则 FileNotFoundError；不展开 ~、读取表或验证类型。
        Check path existence only, raising FileNotFoundError. No tilde expansion,
        table reading or product-type validation; initialize the cached data as None."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[RmfData] = None

    def read(self) -> RmfData:
        """读取 RMF/RSP 矩阵及可选 EBOUNDS / Read a response matrix and optional EBOUNDS.

        优先 MATRIX，再 SPECRESP MATRIX；提取能量、可变长分组与矩阵数组，
        解析通道约定，返回并缓存 RmfData。裸数不做单位转换，矩阵不归一化、
        不解压为稠密矩阵、不独立区分是否已含有效面积。
        Prefer MATRIX then SPECRESP MATRIX; read energy, variable-length groups/matrix
        and channel metadata. Return/cache RmfData without unit conversion, matrix
        normalization, dense expansion or determination of whether area is included.
        Each call rereads disk; required-extension/column errors propagate."""
        with fits.open(self.path) as h:
            # OGIP RMF files produced by XSPEC/HEASoft commonly use the
            # canonical ``SPECRESP MATRIX`` extension name, while some
            # legacy JinWu products use the shorter ``MATRIX`` alias.  Both
            # names describe the same response table; rejecting the former
            # prevents deterministic ``fakeit`` templates from being read
            # back for otherwise valid survey PHA files.
            # 方法：RMF 矩阵扩展按 "MATRIX" → "SPECRESP MATRIX" 顺序接受（两者同为
            #       OGIP RMF 矩阵扩展的不同年代命名），与 heasp 的查找优先级一致。
            # 参考：HEASoft 6.37 heacore/heasp/rmf.cxx rmf::readMatrix（先按 EXTNAME=
            #       "MATRIX" 定位，失败再试 "SPECRESP MATRIX"，最后回退
            #       HDUCLAS1=RESPONSE + HDUCLAS2=RSP_MATRIX）；
            #       格式定义 CAL/GEN/92-002 "The Calibration Requirements for
            #       Spectral Analysis (Definition of RMF and ARF file formats)"。
            matrix_name = next(
                (
                    name
                    for name in ("MATRIX", "SPECRESP MATRIX")
                    if name in h
                ),
                None,
            )
            if matrix_name is None:
                raise KeyError("RMF lacks MATRIX or SPECRESP MATRIX extension")
            hm = cast(Any, h[matrix_name])
            dm = hm.data
            matrix_columns = tuple(getattr(dm.columns, 'names', ()) or ())
            energ_lo = np.asarray(dm["ENERG_LO"], float)
            energ_hi = np.asarray(dm["ENERG_HI"], float)
            header = dict(cast(Any, hm.header))
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

            matrix = np.asarray(dm["MATRIX"], dtype=object)
            n_grp = np.asarray(dm["N_GRP"]) if "N_GRP" in matrix_columns else None
            f_chan = np.asarray(dm["F_CHAN"]) if "F_CHAN" in matrix_columns else None
            n_chan = np.asarray(dm["N_CHAN"]) if "N_CHAN" in matrix_columns else None

            channel = None
            e_min = None
            e_max = None
            ebounds_columns: tuple[str, ...] = ()
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                ebounds_columns = tuple(getattr(de.columns, 'names', ()) or ())
                channel = np.asarray(de["CHANNEL"], int)
                e_min = np.asarray(de["E_MIN"], float)
                e_max = np.asarray(de["E_MAX"], float)

        columns = matrix_columns + tuple(name for name in ebounds_columns if name not in matrix_columns)
        # 通道约定键：TLMIN（F_CHAN 索引基准）与 DETCHANS，供一致性校验与回写使用。
        tlmin = _infer_rmf_tlmin(header, f_chan)
        det_chans = _opt_int(header.get("DETCHANS"))
        if det_chans is None:
            if channel is not None:
                det_chans = int(np.asarray(channel).size)
            elif e_min is not None:
                det_chans = int(np.asarray(e_min).size)
        self._data = RmfData(
            path=self.path,
            energ_lo=energ_lo,
            energ_hi=energ_hi,
            n_grp=n_grp,
            f_chan=f_chan,
            n_chan=n_chan,
            matrix=matrix,
            channel=channel,
            e_min=e_min,
            e_max=e_max,
            columns=columns,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
            tlmin=tlmin,
            det_chans=det_chans,
            hduvers=(str(header.get("HDUVERS")).strip() or None) if header.get("HDUVERS") is not None else None,
        )
        return self._data

    def validate(self) -> ValidationReport:
        """读取后委托对象校验 / Read if needed, then delegate to the data object's validate.

        已有 _data 时不重新读取文件；返回 ValidationReport，具体检查由数据类
        定义。校验通过不证明响应、时间系统或科学结果普遍有效。
        Reuse cached data if present without rereading disk. Return ValidationReport
        from the data class; passing its checks is not general scientific certification."""
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipPhaReader:
    def __init__(self, path: str | Path):
        """保存 PHA 读取路径 / Initialize a PHA reader with a local path.

        只检查路径存在，否则 FileNotFoundError；不展开 ~、读取表或验证类型。
        Check path existence only, raising FileNotFoundError. No tilde expansion,
        table reading or product-type validation; initialize the cached data as None."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[PhaData] = None

    def read(self) -> PhaData:
        """读取 PHA 通道谱 / Read and cache a PHA channel spectrum.

        要求 SPECTRUM/CHANNEL 与 COUNTS 或 RATE；保留 STAT_ERR、分组、质量
        及比例字段和可选 EBOUNDS。仅有 RATE 时，正有限 EXPOSURE 用于换算
        counts=rate*exposure；曝光无效时 counts 暂存 RATE 数值，不能当原始
        Poisson 计数。STAT_ERR 不在此自动按 rate/counts 的表示换算。
        Require SPECTRUM/CHANNEL and COUNTS or RATE; retain errors, grouping, quality,
        scales and optional EBOUNDS. RATE-only data become counts with valid positive
        exposure; otherwise counts retains rate numbers and must not be treated as raw
        Poisson counts. STAT_ERR is not converted between representations here.

        返回缓存的 PhaData；响应路径字符串保留，不自动读取背景或响应；
        重复调用重新读取，不自动运行 validate，FITS/字段错误上抛。
        Return/cache PhaData; response path strings do not load those files. Each call
        rereads disk without validate; FITS/required-field errors propagate."""
        with fits.open(self.path) as h:
            hs = cast(Any, h["SPECTRUM"])
            ds = hs.data
            spectrum_columns = tuple(getattr(ds.columns, 'names', ()) or ())
            channels = np.asarray(ds["CHANNEL"], int)
            rate = np.asarray(ds["RATE"], float) if "RATE" in spectrum_columns else None
            counts = np.asarray(ds["COUNTS"], float) if "COUNTS" in spectrum_columns else None
            stat_err = np.asarray(ds["STAT_ERR"], float) if "STAT_ERR" in spectrum_columns else None
            header_map = cast(Any, hs.header)
            # 通道编号三件套：优先读文件声明，缺失时由通道数组回退。
            pha_tlmin = _opt_int(header_map.get("TLMIN1"))
            pha_tlmax = _opt_int(header_map.get("TLMAX1"))
            pha_det_chans = _opt_int(header_map.get("DETCHANS"))
            if channels.size > 0:
                if pha_tlmin is None:
                    pha_tlmin = int(channels[0])
                if pha_tlmax is None:
                    pha_tlmax = int(channels[-1])
                if pha_det_chans is None:
                    pha_det_chans = int(channels.size)
            exposure = float(header_map.get("EXPOSURE", header_map.get("EXPTIME", np.nan)))
            if counts is None:
                if rate is None:
                    raise ValueError("PHA SPECTRUM lacks both COUNTS and RATE columns")
                if np.isfinite(exposure) and exposure > 0:
                    counts = np.asarray(rate, float) * float(exposure)
                else:
                    counts = np.asarray(rate, float)
            backscal = float(header_map.get("BACKSCAL")) if "BACKSCAL" in header_map else None
            areascal = float(header_map.get("AREASCAL")) if "AREASCAL" in header_map else None
            quality = np.asarray(ds["QUALITY"], int) if "QUALITY" in spectrum_columns else None
            grouping = np.asarray(ds["GROUPING"], int) if "GROUPING" in spectrum_columns else None
            raw_spectrum_columns: Dict[str, np.ndarray] = {}
            for cn in spectrum_columns:
                try:
                    raw_spectrum_columns[cn] = np.asarray(ds[cn])
                except Exception:
                    continue

            ebounds = None
            ebounds_columns: tuple[str, ...] = ()
            if "EBOUNDS" in h:
                de = cast(Any, h["EBOUNDS"]).data
                ebounds_columns = tuple(getattr(de.columns, 'names', ()) or ())
                ebounds = (
                    np.asarray(de["CHANNEL"], int),
                    np.asarray(de["E_MIN"], float),
                    np.asarray(de["E_MAX"], float),
                )

            header = dict(header_map)
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)

        respfile = None
        ancrfile = None
        try:
            if 'RESPFILE' in header and header.get('RESPFILE') not in (None, '', ' '):
                respfile = str(header.get('RESPFILE'))
        except Exception:
            respfile = None
        try:
            if 'ANCRFILE' in header and header.get('ANCRFILE') not in (None, '', ' '):
                ancrfile = str(header.get('ANCRFILE'))
        except Exception:
            ancrfile = None

        columns = spectrum_columns + tuple(name for name in ebounds_columns if name not in spectrum_columns)
        self._data = PhaData(
            path=self.path,
            channels=channels,
            counts=counts,
            rate=rate,
            stat_err=stat_err,
            exposure=exposure,
            backscal=backscal,
            areascal=areascal,
            respfile=respfile,
            ancrfile=ancrfile,
            quality=quality,
            grouping=grouping,
            ebounds=ebounds,
            raw_spectrum_columns=raw_spectrum_columns,
            columns=columns,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
            tlmin=pha_tlmin,
            tlmax=pha_tlmax,
            det_chans=pha_det_chans,
        )
        return self._data

    def validate(self) -> ValidationReport:
        """读取后委托对象校验 / Read if needed, then delegate to the data object's validate.

        已有 _data 时不重新读取文件；返回 ValidationReport，具体检查由数据类
        定义。校验通过不证明响应、时间系统或科学结果普遍有效。
        Reuse cached data if present without rereading disk. Return ValidationReport
        from the data class; passing its checks is not general scientific certification."""
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()

    def select_by_band(
        self,
        band: EnergyBand,
        rmf_chan_band: Optional[ChannelBand] = ChannelBand(51, 399),
    ) -> tuple[np.ndarray, np.ndarray]:
        """按能段与通道约束返回谱 / Return selected channels and stored count values.

        有 EBOUNDS 时按严格能量交叠及可选通道范围筛选，再匹配 PHA 通道。
        无 EBOUNDS 时仅用通道约束，band 不起能量筛选作用；默认通道 51..399，
        不是通用仪器能段。返回 (channels, counts) 数组，不更新缓存谱。
        With EBOUNDS use energy overlap and optional inclusive channel bounds, then
        match PHA channels. Without EBOUNDS only channel limits apply; band is ignored.
        Default 51..399 is a legacy range, not a universal energy calibration. Return
        channels/stored counts without changing the cached spectrum."""
        if self._data is None:
            _ = self.read()
        d = self._data
        assert d is not None
        if d.ebounds is not None:
            mask = channel_mask_from_ebounds(d.ebounds, band, rmf_chan_band)
            ch_all = d.ebounds[0]
            idx_map = {int(c): i for i, c in enumerate(d.channels)}
            sel_channels = ch_all[mask]
            sel_idx = np.array([idx_map[c] for c in sel_channels if c in idx_map], dtype=int)
            return d.channels[sel_idx], d.counts[sel_idx]
        if rmf_chan_band is None:
            return d.channels, d.counts
        mask = (d.channels >= rmf_chan_band.ch_lo) & (d.channels <= rmf_chan_band.ch_hi)
        return d.channels[mask], d.counts[mask]


class OgipLightcurveReader:
    def __init__(self, path: str | Path):
        """保存 lightcurve 读取路径 / Initialize a lightcurve reader with a local path.

        只检查路径存在，否则 FileNotFoundError；不展开 ~、读取表或验证类型。
        Check path existence only, raising FileNotFoundError. No tilde expansion,
        table reading or product-type validation; initialize the cached data as None."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[LightcurveData] = None

    def read(self) -> LightcurveData:
        """读取光变并构造相对时间 / Read a lightcurve with relative-time fields.

        寻找含 TIME 的表，time=TIME-TIME[0]，timezero=TIMEZERO+TIME[0]。
        当前将 TIME 视作 bin 左边缘，bin_hi=time+dt，不应用 TIMEPIXR；dt
        优先 TIMEDEL 再时间差中位数，单位按文件的既有秒数约定，不换 TUNIT。
        Find a TIME table, set time=TIME-TIME[0] and timezero=TIMEZERO+TIME[0]. Treat
        TIME as bin left edge without TIMEPIXR adjustment; infer dt from TIMEDEL or
        median differences. No TUNIT conversion; callers establish seconds/time reference.

        RATE 优先作为主 value；COUNTS 缺失时用 RATE*dt 生成。ERROR 按主
        表示解释，counts/rate 转换使用 dt；bin_exposure 则优先裁剪后的
        FRACEXP*bin_width，否则 bin_width。此读取步骤不做原生死时间校正。
        Prefer RATE as primary value, deriving absent counts as RATE*dt. ERROR follows
        that representation; count/rate conversion uses dt. Bin exposure uses clipped
        FRACEXP*width or width. No native-event-deadtime correction is performed.

        返回并缓存 LightcurveData，保留可读取 GTI 原始数值；当前没有对 GTI
        同步减去 TIME[0]。读取成功不证明 GTI 与相对时间已一致或噪声为 Poisson。
        Repeat calls reread disk and return/cache LightcurveData. GTI numbers are kept
        without subtracting TIME[0]; read success does not establish common references
        or Poisson noise. Missing fields/bin width or unknown mission conversion may raise."""
        with fits.open(self.path) as h:
            hdu = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, 'data', None) is None:
                    continue
                names = getattr(ext_any.data, 'columns', None)
                if names is not None and ('TIME' in [n.upper() for n in names.names]):
                    hdu = ext_any
                    break
            if hdu is None:
                raise ValueError('No suitable lightcurve HDU with TIME column found')
            d = cast(Any, hdu.data)
            col_names_upper = [n.upper() for n in d.columns.names]
            header = dict(cast(Any, hdu.header))
            time_raw = np.asarray(d['TIME'], dtype=float)
            timezero_raw = 0.0
            if 'TIMEZERO' in header:
                try:
                    timezero_raw = float(header['TIMEZERO'])
                except (ValueError, TypeError):
                    timezero_raw = 0.0
            time_offset = float(time_raw[0]) if len(time_raw) > 0 else 0.0
            time = time_raw - time_offset
            time_rel = time
            timezero = timezero_raw + time_offset
            telescop = None
            primary_header = dict(cast(Any, h[0]).header) if len(h) > 0 else {}
            for hdr in [header, primary_header]:
                if 'TELESCOP' in hdr:
                    telescop = str(hdr['TELESCOP']).strip().upper()
                    break
            timezero_obj = _mission_timezero_object(telescop, timezero, allow_unix_fallback=False)
            dt = None
            if 'TIMEDEL' in header:
                try:
                    dt = float(header['TIMEDEL'])
                except (ValueError, TypeError):
                    pass
            if dt is None and time.size >= 2:
                dt = float(np.median(np.diff(time)))
            if dt is None:
                raise ValueError('binsize未被正确加载')
            bin_lo = time.copy()
            bin_hi = time + dt
            rate = None
            counts = None
            is_rate = False
            value = None
            if 'RATE' in col_names_upper:
                rate = np.asarray(d['RATE'], dtype=float)
                value = rate
                is_rate = True
                if 'COUNTS' not in col_names_upper and dt > 0:
                    counts = rate * dt
            if 'COUNTS' in col_names_upper:
                counts = np.asarray(d['COUNTS'], dtype=float)
                if value is None:
                    value = counts
                    is_rate = False
                if rate is None and dt > 0:
                    rate = counts / dt
            if value is None:
                raise ValueError('Lightcurve HDU lacks RATE/COUNTS column')
            error = None
            rate_err = None
            counts_err = None
            if 'ERROR' in col_names_upper:
                error = np.asarray(d['ERROR'], dtype=float)
                if is_rate:
                    rate_err = error
                    if counts is not None and dt > 0:
                        counts_err = error * dt
                else:
                    counts_err = error
                    if rate is not None and dt > 0:
                        rate_err = error / dt
            fracexp = np.asarray(d['FRACEXP'], dtype=float) if 'FRACEXP' in col_names_upper else None
            quality = np.asarray(d['QUALITY'], dtype=int) if 'QUALITY' in col_names_upper else None
            backscal_col = d['BACKSCAL'] if 'BACKSCAL' in col_names_upper else (d['BACK_SCAL'] if 'BACK_SCAL' in col_names_upper else None)
            areascal_col = d['AREASCAL'] if 'AREASCAL' in col_names_upper else (d['AREA_SCAL'] if 'AREA_SCAL' in col_names_upper else None)
            gti_start = None
            gti_stop = None
            try:
                gti_list = _extract_gti(h)
                if gti_list is not None:
                    gti_start = np.array([s for s, _ in gti_list], dtype=float)
                    gti_stop = np.array([e for _, e in gti_list], dtype=float)
            except Exception:
                pass
            tstart = timezero
            if len(time) > 0:
                tstop = timezero + time[-1] + dt
                tseg = float(tstop - tstart)
            else:
                tseg = None
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
            # Relative light curves remain useful without mission time metadata.
            # Their absolute-time object is intentionally unavailable.
            exposure = None
            if 'EXPOSURE' in header:
                try:
                    exposure = float(header['EXPOSURE'])
                except (ValueError, TypeError):
                    pass
            if exposure is None and 'EXPTIME' in header:
                try:
                    exposure = float(header['EXPTIME'])
                except (ValueError, TypeError):
                    pass
            bin_exposure = None
            bin_width = None
            if (bin_lo is not None) and (bin_hi is not None) and (len(bin_lo) == len(bin_hi)):
                try:
                    bin_width = np.asarray(bin_hi, dtype=float) - np.asarray(bin_lo, dtype=float)
                except Exception:
                    bin_width = None
            if bin_width is None and len(time) > 0:
                dt_arr = np.asarray(dt, dtype=float) if dt is not None else np.asarray([], dtype=float)
                if dt_arr.ndim == 0 and dt_arr.size != 0 and np.isfinite(float(dt_arr)) and float(dt_arr) > 0:
                    bin_width = np.full(len(time), float(dt_arr), dtype=float)
                elif dt_arr.ndim == 1 and dt_arr.size == len(time):
                    bin_width = dt_arr
            if fracexp is not None and len(time) > 0:
                fracexp_arr = np.asarray(fracexp, dtype=float)
                if fracexp_arr.shape == (len(time),):
                    fracexp_arr = np.where(np.isfinite(fracexp_arr), fracexp_arr, 1.0)
                    fracexp_arr = np.clip(fracexp_arr, 0.0, 1.0)
                    if bin_width is not None:
                        bin_exposure = fracexp_arr * np.asarray(bin_width, dtype=float)
            if bin_exposure is None and bin_width is not None:
                bin_exposure = np.asarray(bin_width, dtype=float)
            if bin_exposure is None and exposure is not None and len(time) > 0:
                bin_exposure = np.full(len(time), exposure / len(time), dtype=float)
            err_dist = 'poisson' if counts is not None else ('gauss' if rate is not None else None)
            try:
                region = _load_regions(h)
            except Exception:
                region = None
            timesys = str(header['TIMESYS']) if 'TIMESYS' in header else (meta.timesys if meta and meta.timesys else None)
            mjdref = meta.mjdref if meta and meta.mjdref else None
            self._data = LightcurveData(
                path=self.path,
                time=time,
                time_raw=time_raw,
                time_rel=time_rel,
                timezero=timezero,
                timezero_obj=timezero_obj,
                dt=dt,
                bin_lo=bin_lo,
                bin_hi=bin_hi,
                tstart=tstart,
                tseg=tseg,
                value=value,
                error=error,
                is_rate=is_rate,
                counts=counts,
                rate=rate,
                counts_err=counts_err,
                rate_err=rate_err,
                err_dist=err_dist,
                gti_start=gti_start,
                gti_stop=gti_stop,
                quality=quality,
                fracexp=fracexp,
                exposure=exposure,
                bin_exposure=bin_exposure,
                backscal=backscal_col,
                areascal=areascal_col,
                telescop=telescop,
                timesys=timesys,
                mjdref=mjdref,
                region=region,
                header=header,
                meta=meta,
                headers_dump=headers_dump,
                columns=tuple(d.columns.names),
            )
        return self._data

    def validate(self) -> ValidationReport:
        """读取后委托对象校验 / Read if needed, then delegate to the data object's validate.

        已有 _data 时不重新读取文件；返回 ValidationReport，具体检查由数据类
        定义。校验通过不证明响应、时间系统或科学结果普遍有效。
        Reuse cached data if present without rereading disk. Return ValidationReport
        from the data class; passing its checks is not general scientific certification."""
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class OgipEventReader:
    def __init__(self, path: str | Path):
        """保存 event 读取路径 / Initialize a event reader with a local path.

        只检查路径存在，否则 FileNotFoundError；不展开 ~、读取表或验证类型。
        Check path existence only, raising FileNotFoundError. No tilde expansion,
        table reading or product-type validation; initialize the cached data as None."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._data: Optional[EventData] = None

    def read(self) -> EventData:
        """读取事件及原始列 / Read events, raw columns and relative-time GTI.

        寻找 TIME 表，time=TIME-TIME[0]，timezero=TIMEZERO+TIME[0]；保留
        原始列，并按候选名字映射坐标、能量及通道，不由 PI 推断能量单位。
        支持的 GTI START/STOP 同减 TIME[0]；TIMEZERO 绝对对象按任务转换，
        未知任务可能兼容回退 Unix，不能据此认证时间系统。
        Find a TIME table, form relative times and shifted timezero; preserve raw
        columns and candidate coordinate/energy/channel mappings without PI calibration.
        Shift supported GTI START/STOP by TIME[0]. Mission absolute conversion may
        fall back to Unix for unknown missions, which does not certify the time system.

        返回缓存 EventData；不排序事件、不应用 GTI 事件筛选或死时间校正。
        重复调用重新读文件，可选字段失败可能成为 None，必需字段错误上抛。
        Return/cache EventData without sorting, GTI filtering or deadtime correction.
        Each call rereads; optional field failures may become None, required ones raise."""
        with fits.open(self.path) as h:
            hevt = None
            for ext in h:
                ext_any = cast(Any, ext)
                if getattr(ext_any, 'data', None) is None:
                    continue
                names = getattr(ext_any.data, 'columns', None)
                if names is not None and ('TIME' in names.names):
                    hevt = ext_any
                    break
            if hevt is None:
                raise ValueError('No EVENTS-like HDU with TIME column found')
            de = cast(Any, hevt.data)
            colnames = list(getattr(de, 'columns').names) if getattr(de, 'columns', None) is not None else []
            raw_columns: Dict[str, np.ndarray] = {}
            for cn in colnames:
                try:
                    raw_columns[cn] = np.asarray(de[cn])
                except Exception:
                    try:
                        raw_columns[cn] = np.asarray([r[cn] for r in de])
                    except Exception:
                        raw_columns[cn] = np.asarray([])
            time_raw = np.asarray(raw_columns.get('TIME') if 'TIME' in raw_columns else de['TIME'], float)
            header = dict(cast(Any, hevt.header))
            primary_header = dict(cast(Any, h[0]).header) if len(h) > 0 else {}
            timezero_raw = 0.0
            raw_timezero = header.get('TIMEZERO', primary_header.get('TIMEZERO', 0.0))
            if raw_timezero is not None:
                try:
                    timezero_raw = float(raw_timezero)
                except (ValueError, TypeError):
                    timezero_raw = 0.0
            time_offset = float(time_raw[0]) if len(time_raw) > 0 else 0.0
            time = time_raw - time_offset
            time_rel = time
            timezero = timezero_raw + time_offset
            telescop = None
            for hdr in [header, primary_header]:
                if 'TELESCOP' in hdr:
                    telescop = str(hdr['TELESCOP']).strip().upper()
                    break
            timezero_obj = _mission_timezero_object(telescop, timezero, allow_unix_fallback=True) if telescop is not None and timezero != 0.0 else None
            pi = np.asarray(raw_columns['PI'], int) if 'PI' in raw_columns else None
            channel = np.asarray(raw_columns['CHANNEL'], int) if 'CHANNEL' in raw_columns else None
            headers_dump = _collect_headers_dump(h)
            meta = _build_meta(h, header)
            ebounds = None
            try:
                if 'EBOUNDS' in h:
                    de_eb = cast(Any, h['EBOUNDS']).data
                    ebounds = (
                        np.asarray(de_eb['CHANNEL'], int),
                        np.asarray(de_eb['E_MIN'], float),
                        np.asarray(de_eb['E_MAX'], float),
                    )
            except Exception:
                ebounds = None
            gti_start = None
            gti_stop = None
            gti_start_obj = None
            gti_stop_obj = None
            gti_list = None
            try:
                for hdu in h:
                    if getattr(hdu, 'name', '').upper() == 'GTI':
                        gti_data = getattr(hdu, 'data', None)
                        if gti_data is not None and 'START' in gti_data.columns.names and 'STOP' in gti_data.columns.names:
                            gti_start_raw = np.asarray(gti_data['START'], float)
                            gti_stop_raw = np.asarray(gti_data['STOP'], float)
                            gti_start = gti_start_raw - time_offset
                            gti_stop = gti_stop_raw - time_offset
                            gti_list = [(float(s), float(e)) for s, e in zip(gti_start, gti_stop)]
                            if timezero_obj is not None:
                                try:
                                    from astropy.time import TimeDelta

                                    gti_start_obj = timezero_obj + TimeDelta(gti_start, format='sec')
                                    gti_stop_obj = timezero_obj + TimeDelta(gti_stop, format='sec')
                                except Exception:
                                    gti_start_obj = gti_stop_obj = None
                            break
            except Exception:
                gti_start = gti_stop = None
                gti_start_obj = gti_stop_obj = None
                gti_list = None
            u2orig = {cn.upper(): cn for cn in colnames}

            def _find(*cands: str) -> Optional[str]:
                """映射候选列到原名称 / Return the first case-insensitive candidate's original name.

                无匹配返回 None，不读取或转换列值。
                Return None if absent, without reading/converting values."""
                for c in cands:
                    if c is None:
                        continue
                    uc = c.upper()
                    if uc in u2orig:
                        return u2orig[uc]
                return None

            colmap: Dict[str, Optional[str]] = {}
            colmap['x'] = _find('X', 'XRAW', 'RAWX', 'DETX', 'DET_X', 'SKX', 'XDET')
            colmap['y'] = _find('Y', 'YRAW', 'RAWY', 'DETY', 'DET_Y', 'SKY', 'YDET')
            colmap['ra'] = _find('RA', 'RA_OBJ', 'RAX', 'RA_DEG')
            colmap['dec'] = _find('DEC', 'DEC_OBJ', 'DECX', 'DEC_DEG')
            colmap['energy'] = _find('ENERGY', 'E', 'ENERG', 'PHOTON_ENERGY')
            colmap['pha'] = _find('PHA', 'PI')
            key_x = colmap.get('x')
            key_y = colmap.get('y')
            key_energy = colmap.get('energy')
            xarr = np.asarray(raw_columns[key_x]) if (key_x is not None and key_x in raw_columns) else None
            yarr = np.asarray(raw_columns[key_y]) if (key_y is not None and key_y in raw_columns) else None
            energy = np.asarray(raw_columns[key_energy]) if (key_energy is not None and key_energy in raw_columns) else None
        self._data = EventData(
            path=self.path,
            time=time,
            time_raw=time_raw,
            time_rel=time_rel,
            timezero=timezero,
            timezero_obj=timezero_obj,
            telescop=telescop,
            pi=pi,
            channel=channel,
            x=xarr,
            y=yarr,
            gti_start=gti_start,
            gti_stop=gti_stop,
            gti_start_obj=gti_start_obj,
            gti_stop_obj=gti_stop_obj,
            gti=gti_list,
            raw_columns=raw_columns,
            colmap=colmap,
            energy=energy,
            ebounds=ebounds,
            header=header,
            meta=meta,
            headers_dump=headers_dump,
            columns=tuple(colnames),
        )
        return self._data

    def validate(self) -> ValidationReport:
        """读取后委托对象校验 / Read if needed, then delegate to the data object's validate.

        已有 _data 时不重新读取文件；返回 ValidationReport，具体检查由数据类
        定义。校验通过不证明响应、时间系统或科学结果普遍有效。
        Reuse cached data if present without rereading disk. Return ValidationReport
        from the data class; passing its checks is not general scientific certification."""
        if self._data is None:
            self.read()
        assert self._data is not None
        return self._data.validate()


class ArfReader(OgipArfReader):
    pass


class RmfReader(OgipRmfReader):
    pass


class RspReader(OgipRmfReader):
    pass


class LightcurveReader(OgipLightcurveReader):
    pass


OgipData = Union[ArfData, RmfData, PhaData, LightcurveData, EventData]
OgipWritableData = Union[ArfData, RmfData, PhaData, LightcurveData, EventData]


def _normalize_grouping_to_flags(grouping: np.ndarray) -> np.ndarray:
    """把旧组号转为 OGIP 标记 / Convert legacy group IDs to OGIP grouping flags.

    若非零值已全为 +/-1，原样返回整数数组。其他编码下，正组号变化
    记 +1，同组记 -1，非正值记 0；零项不会重置此前组号。返回数组，
    不合并通道或计数；调用者须确认输入编码符合该规则。
    Return an integer array unchanged when nonzero values are already +/-1.
    Otherwise positive group-ID changes start +1, repeats continue -1, nonpositive
    entries become 0 without resetting the previous ID. No channel/count aggregation;
    callers ensure the legacy coding fits these rules."""
    g = np.asarray(grouping, dtype=int)
    if g.size == 0:
        return g
    nz = g[g != 0]
    if nz.size == 0:
        return g
    if np.all(np.isin(nz, [-1, 1])):
        return g
    out = np.zeros_like(g)
    prev_gid = None
    for i, gid in enumerate(g):
        if gid <= 0:
            continue
        if prev_gid != int(gid):
            out[i] = 1
            prev_gid = int(gid)
        else:
            out[i] = -1
    return out


class PhaWriter:
    def __init__(self, data: PhaData, outpath: str | Path):
        """保存 PHA 对象与输出路径 / Initialize a PHA writer by reference.

        不复制或校验 data，不创建目录、不写文件；覆盖策略在 write 中应用。
        Store data by reference and outpath as Path. No copying, validation, directory
        creation or file write; write() handles overwrite policy."""
        self.data = data
        self.outpath = Path(outpath)

    def write(self, *, overwrite: bool = False) -> Path:
        """将结构化 PHA 写为 FITS / Write selected PHA fields to a FITS spectrum.

        overwrite=False 时现有文件抛 FileExistsError；不创建父目录。写
        CHANNEL、计数/率、可选误差/质量/分组，规范 GROUPING；可写 EBOUNDS。
        RATE 存在且曝光无效时只写 RATE，不写 COUNTS/无效曝光头。
        Default overwrite=False rejects existing files; parents are not created. Write
        selected count/rate/error/quality/grouping fields and optional EBOUNDS, converting
        group IDs to flags. RATE with invalid exposure omits COUNTS/invalid exposure cards.

        返回 outpath。输出浮点列多为 FITS E（32-bit），不是原文件逐位复制；
        未保证所有 raw_spectrum_columns、重复卡片和扩展被保留。不重新分组
        计算计数、不拟合、不验证响应兼容性，单位须由输入字段保证。
        Return outpath. Float columns commonly use FITS E (32-bit), not a byte-exact
        copy; not all raw columns/cards/extensions are retained. No count regrouping,
        fit or response validation. Callers ensure field units and scientific consistency."""
        outp = self.outpath
        if outp.exists() and not overwrite:
            raise FileExistsError(str(outp))

        pha = self.data
        cols: list[fits.Column] = []

        channels = np.asarray(pha.channels, dtype=int)
        counts = np.asarray(pha.counts, dtype=float)
        cols.append(fits.Column(name='CHANNEL', format='J', array=channels))
        rate_only = (
            pha.rate is not None
            and (pha.exposure is None or not np.isfinite(pha.exposure) or pha.exposure <= 0)
        )
        if not rate_only:
            cols.append(fits.Column(name='COUNTS', format='E', array=counts))

        if getattr(pha, 'rate', None) is not None:
            cols.append(fits.Column(name='RATE', format='E', array=np.asarray(pha.rate, dtype=float)))
        if pha.stat_err is not None:
            cols.append(fits.Column(name='STAT_ERR', format='E', array=np.asarray(pha.stat_err, dtype=float)))
        if pha.quality is not None:
            cols.append(fits.Column(name='QUALITY', format='J', array=np.asarray(pha.quality, dtype=int)))
        if pha.grouping is not None:
            # 方法：GROUPING 列采用 OGIP 分组标记：+1=新分组起始道，-1=同组延续道，
            #       0=未分组/忽略；写出前把旧版"组号编码"归一化为该标记。
            # 参考：OGIP/92-007 "The OGIP Spectral File Format" §3.1.2
            #       （Grpg=+1 为新 bin 起始，-1 为延续，0 为未定义分组）。
            g = _normalize_grouping_to_flags(np.asarray(pha.grouping, dtype=int))
            cols.append(fits.Column(name='GROUPING', format='J', array=g))

        hdu_spec = fits.BinTableHDU.from_columns(cols, name='SPECTRUM')
        hdr = hdu_spec.header
        hdr['EXTNAME'] = 'SPECTRUM'
        hdr['HDUCLASS'] = 'OGIP'
        hdr['HDUCLAS1'] = 'SPECTRUM'
        hdr['CHANTYPE'] = str(getattr(pha.header, 'get', lambda *_: None)('CHANTYPE') or 'PI') if pha.header is not None else 'PI'

        # 通道编号三件套（HEASoft 6.37 heasp::pha 写入约定：总是写全）。
        # 优先用读入时解析的结构化字段；否则以本文件实际通道数组为准。
        # 方法：总是写出 TLMIN1=首道、TLMAX1=末道、DETCHANS=道数，并写
        #       HDUCLASS=OGIP、HDUCLAS1=SPECTRUM、CHANTYPE（见上方写出）。
        # 参考：HEASoft 6.37 heacore/heasp/pha.cxx pha::write（SPwriteKey：
        #       DETCHANS、TLMIN1=FirstChannel、TLMAX1=FirstChannel+DetChans-1，
        #       HDUCLASS="OGIP"、HDUCLAS1="SPECTRUM"、CHANTYPE）。
        if channels.size > 0:
            _tlmin1 = getattr(pha, 'tlmin', None)
            _tlmax1 = getattr(pha, 'tlmax', None)
            _detch = getattr(pha, 'det_chans', None)
            hdr['TLMIN1'] = int(_tlmin1 if _tlmin1 is not None else channels[0])
            hdr['TLMAX1'] = int(_tlmax1 if _tlmax1 is not None else channels[-1])
            hdr['DETCHANS'] = int(_detch if _detch is not None else channels.size)

        if pha.meta is not None and getattr(pha.meta, 'telescop', None) is not None:
            hdr['TELESCOP'] = str(pha.meta.telescop)
        elif pha.header is not None and 'TELESCOP' in pha.header:
            hdr['TELESCOP'] = str(pha.header['TELESCOP'])
        if pha.meta is not None and getattr(pha.meta, 'instrume', None) is not None:
            hdr['INSTRUME'] = str(pha.meta.instrume)
        elif pha.header is not None and 'INSTRUME' in pha.header:
            hdr['INSTRUME'] = str(pha.header['INSTRUME'])

        if pha.exposure is not None and np.isfinite(pha.exposure) and pha.exposure > 0:
            hdr['EXPOSURE'] = float(pha.exposure)
        if pha.backscal is not None:
            hdr['BACKSCAL'] = float(pha.backscal)
        if pha.areascal is not None:
            hdr['AREASCAL'] = float(pha.areascal)
        if getattr(pha, 'respfile', None) is not None:
            hdr['RESPFILE'] = str(pha.respfile)
        if getattr(pha, 'ancrfile', None) is not None:
            hdr['ANCRFILE'] = str(pha.ancrfile)

        if pha.header is not None:
            for k, v in dict(pha.header).items():
                key = str(k).upper()
                if key in hdr:
                    continue
                if key in {'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND'} or (rate_only and key in {'EXPOSURE', 'EXPTIME'}):
                    continue
                try:
                    hdr[key] = v
                except Exception:
                    continue

        prih = fits.PrimaryHDU()
        if pha.header is not None:
            for key in ('TELESCOP', 'INSTRUME', 'OBS_ID', 'OBJECT'):
                if key in pha.header:
                    try:
                        prih.header[key] = pha.header[key]
                    except Exception:
                        pass
        if pha.meta is not None:
            if getattr(pha.meta, 'tstart', None) is not None:
                prih.header['TSTART'] = float(pha.meta.tstart)
            if getattr(pha.meta, 'tstop', None) is not None:
                prih.header['TSTOP'] = float(pha.meta.tstop)

        hdul = fits.HDUList([prih, hdu_spec])

        if getattr(pha, 'ebounds', None) is not None:
            ebounds = pha.ebounds
            if ebounds is not None and len(ebounds) == 3:
                ch_eb, emin, emax = ebounds
                cols_eb = [
                    fits.Column(name='CHANNEL', format='J', array=np.asarray(ch_eb, dtype=int)),
                    fits.Column(name='E_MIN', format='E', array=np.asarray(emin, dtype=float)),
                    fits.Column(name='E_MAX', format='E', array=np.asarray(emax, dtype=float)),
                ]
                hdul.append(fits.BinTableHDU.from_columns(cols_eb, name='EBOUNDS'))

        hdul.writeto(outp, overwrite=overwrite)
        return outp


class ArfWriter:
    def __init__(self, data: ArfData, outpath: str | Path):
        """保存 ARF 对象与输出路径 / Initialize a ARF writer by reference.

        不复制或校验 data，不创建目录、不写文件；覆盖策略在 write 中应用。
        Store data by reference and outpath as Path. No copying, validation, directory
        creation or file write; write() handles overwrite policy."""
        self.data = data
        self.outpath = Path(outpath)

    def write(self, *, overwrite: bool = False) -> Path:
        """写 ARF 的选定响应字段 / Write selected ARF fields to SPECRESP FITS.

        写 ENERG_LO/HI、SPECRESP 的 32-bit 浮点列与响应分类头，保留部分
        仪器字段；不换单位、不积分或重采样响应。父目录须已存在，默认
        禁止覆盖；返回输出 Path。不是原 ARF 所有扩展/头卡的无损复制。
        Write 32-bit energy/area columns and selected response/instrument metadata,
        without unit conversion/integration/rebinning. Parents must exist; default
        rejects overwrite. Return Path, not a lossless copy of every original card/HDU."""
        outp = self.outpath
        if outp.exists() and not overwrite:
            raise FileExistsError(str(outp))
        arf = self.data
        cols = [
            fits.Column(name='ENERG_LO', format='E', array=np.asarray(arf.energ_lo, dtype=float)),
            fits.Column(name='ENERG_HI', format='E', array=np.asarray(arf.energ_hi, dtype=float)),
            fits.Column(name='SPECRESP', format='E', array=np.asarray(arf.specresp, dtype=float)),
        ]
        hdu = fits.BinTableHDU.from_columns(cols, name='SPECRESP')
        # 方法：ARF 写出列序 ENERG_LO/ENERG_HI/SPECRESP（32 位浮点）与关键字集
        #       EXTNAME=SPECRESP、HDUCLASS=OGIP、HDUCLAS1=RESPONSE、
        #       HDUCLAS2=SPECRESP、HDUVERS 缺省 1.1.0，与 heasp 写出约定一致。
        # 参考：HEASoft 6.37 heacore/heasp/arf.cxx arf::write（ttype 依次为
        #       ENERG_LO/ENERG_HI/SPECRESP，SPwriteKey HDUCLASS="OGIP"、
        #       HDUCLAS1="RESPONSE"、HDUCLAS2="SPECRESP"、HDUVERS="1.1.0"）；
        #       格式定义 CAL/GEN/92-002（ARF = SPECRESP 扩展）。
        hdu.header['EXTNAME'] = 'SPECRESP'
        hdu.header['HDUCLASS'] = 'OGIP'
        hdu.header['HDUCLAS1'] = 'RESPONSE'
        hdu.header['HDUCLAS2'] = 'SPECRESP'
        hdu.header['HDUVERS'] = str(getattr(arf, 'hduvers', None) or '1.1.0')
        if arf.header is not None:
            for key in ('TELESCOP', 'INSTRUME', 'DETNAM', 'FILTER'):
                if key in arf.header:
                    try:
                        hdu.header[key] = arf.header[key]
                    except Exception:
                        pass
        prih = fits.PrimaryHDU()
        hdul = fits.HDUList([prih, hdu])
        hdul.writeto(outp, overwrite=overwrite)
        return outp


class RmfWriter:
    def __init__(self, data: RmfData, outpath: str | Path):
        """保存 RMF 对象与输出路径 / Initialize a RMF writer by reference.

        不复制或校验 data，不创建目录、不写文件；覆盖策略在 write 中应用。
        Store data by reference and outpath as Path. No copying, validation, directory
        creation or file write; write() handles overwrite policy."""
        self.data = data
        self.outpath = Path(outpath)

    def write(self, *, overwrite: bool = False) -> Path:
        """写 RMF 的稀疏分组结构 / Write response groups and matrix rows to FITS.

        能量用 E，F_CHAN/N_CHAN 用可变长 PJ，MATRIX 用可变长 PE；
        写通道边界与组/元素统计头，满足字段时附 EBOUNDS。不会归一化
        矩阵或验证有效面积语义；保留部分源头，不保留全部任意扩展。
        Use E energy columns, variable-length PJ groups and PE matrix rows, adding
        channel/group-count metadata and available EBOUNDS. Do not normalize the matrix
        or validate area semantics. Preserve selected headers, not arbitrary extensions.

        父目录不自动创建，overwrite=False 拒绝已有文件；返回输出 Path，
        I/O 与字段形状错误上抛。浮点写出可能降低到 32-bit 精度。
        Parents are not created, overwrite=False rejects existing targets. Return Path;
        I/O/shape errors propagate and float storage can reduce precision to 32-bit."""
        outp = self.outpath
        if outp.exists() and not overwrite:
            raise FileExistsError(str(outp))
        rmf = self.data

        def _vla_array(values: np.ndarray | list[Any] | tuple[Any, ...] | None, dtype: Any) -> np.ndarray:
            """规范化可变长 FITS 行 / Build object-array rows for variable-length FITS columns.

            None 转空 object 数组，缺失行转指定 dtype 空数组，其他行至少一维。
            不检查每行长度与 N_CHAN/N_GRP 一致性。
            None becomes empty object array; absent rows become empty typed arrays and
            others at least 1-D. No N_CHAN/N_GRP row-length consistency check."""
            if values is None:
                return np.asarray([], dtype=object)
            seq = np.asarray(values, dtype=object)
            out: list[np.ndarray] = []
            for item in seq:
                if item is None:
                    out.append(np.asarray([], dtype=dtype))
                else:
                    out.append(np.atleast_1d(np.asarray(item, dtype=dtype)))
            return np.asarray(out, dtype=object)

        cols: list[fits.Column] = [
            fits.Column(name='ENERG_LO', format='E', array=np.asarray(rmf.energ_lo, dtype=float)),
            fits.Column(name='ENERG_HI', format='E', array=np.asarray(rmf.energ_hi, dtype=float)),
        ]
        if rmf.n_grp is not None:
            cols.append(fits.Column(name='N_GRP', format='J', array=np.asarray(rmf.n_grp, dtype=int)))
        if rmf.f_chan is not None:
            cols.append(fits.Column(name='F_CHAN', format='PJ()', array=_vla_array(rmf.f_chan, int)))
        if rmf.n_chan is not None:
            cols.append(fits.Column(name='N_CHAN', format='PJ()', array=_vla_array(rmf.n_chan, int)))
        cols.append(fits.Column(name='MATRIX', format='PE()', array=_vla_array(rmf.matrix, float)))

        hdu_mat = fits.BinTableHDU.from_columns(cols, name='MATRIX')
        hdu_mat.header['EXTNAME'] = 'MATRIX'
        hdu_mat.header['HDUCLASS'] = 'OGIP'
        hdu_mat.header['HDUCLAS1'] = 'RESPONSE'
        hdu_mat.header['HDUCLAS2'] = 'RSP_MATRIX'

        # 响应矩阵统计与通道约定键（对齐 HEASoft 6.37 heasp::rmf 写入约定），
        # 供下游工具的 F_CHAN+N_CHAN vs TLMIN+DETCHANS 一致性校验使用。
        # 方法：写出 DETCHANS=通道数、NUMGRP=ΣN_GRP（每行组数之和）、
        #       NUMELT=矩阵元素总数、TLMINn=F_CHAN 列首道。
        # 参考：HEASoft 6.37 heacore/heasp/rmf.cxx rmf::write（SPwriteKey
        #       DETCHANS、NUMGRP、NUMELT、TLMIN4=FirstChannel）与
        #       heacore/heasp/rmf.h（NumberTotalGroups=Σ每行组数、
        #       NumberTotalElements=Σ矩阵元素数）。
        n_det = getattr(rmf, 'det_chans', None)
        if n_det is None:
            if rmf.e_min is not None:
                n_det = int(np.asarray(rmf.e_min).size)
            elif rmf.channel is not None:
                n_det = int(np.asarray(rmf.channel).size)
        if n_det is not None:
            hdu_mat.header['DETCHANS'] = int(n_det)
        if rmf.n_grp is not None:
            hdu_mat.header['NUMGRP'] = int(np.sum(np.asarray(rmf.n_grp, dtype=int)))
        try:
            mat = rmf.matrix
            if isinstance(mat, np.ndarray) and mat.dtype == object:
                n_elt = int(sum(int(np.size(row)) for row in mat))
            else:
                n_elt = int(np.size(mat))
            hdu_mat.header['NUMELT'] = n_elt
        except Exception:
            pass
        # F_CHAN 的 TLMIN（0 基/1 基约定）：优先用读入时解析的结构化字段，
        # 其次沿用源文件声明，否则从 F_CHAN 列最小值推断。
        tlmin4 = getattr(rmf, 'tlmin', None)
        if tlmin4 is None and rmf.header is not None:
            for k, v in dict(rmf.header).items():
                ku = str(k).upper()
                if ku.startswith('TLMIN') and ku != 'TLMIN1':
                    try:
                        tlmin4 = int(v)
                        break
                    except Exception:
                        pass
        if tlmin4 is None and rmf.f_chan is not None:
            try:
                fc = rmf.f_chan
                if isinstance(fc, np.ndarray) and fc.dtype == object:
                    vals = np.concatenate([np.atleast_1d(np.asarray(r, dtype=int)) for r in fc if np.size(r) > 0])
                else:
                    vals = np.atleast_1d(np.asarray(fc, dtype=int))
                if vals.size > 0:
                    tlmin4 = int(vals.min())
            except Exception:
                pass
        f_chan_column = (hdu_mat.columns.names.index('F_CHAN') + 1) if 'F_CHAN' in hdu_mat.columns.names else None
        if tlmin4 is not None and f_chan_column is not None:
            hdu_mat.header[f'TLMIN{f_chan_column}'] = tlmin4

        # 源 header 透传（与 PhaWriter 同策略；已写入的键不被覆盖）
        if rmf.header is not None:
            for k, v in dict(rmf.header).items():
                key = str(k).upper()
                if key in hdu_mat.header or (key.startswith('TLMIN') and key[5:].isdigit()):
                    continue
                if key in {'SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND'}:
                    continue
                try:
                    hdu_mat.header[key] = v
                except Exception:
                    continue

        prih = fits.PrimaryHDU()
        hdul = fits.HDUList([prih, hdu_mat])

        if rmf.channel is not None and rmf.e_min is not None and rmf.e_max is not None:
            cols_eb = [
                fits.Column(name='CHANNEL', format='J', array=np.asarray(rmf.channel, dtype=int)),
                fits.Column(name='E_MIN', format='E', array=np.asarray(rmf.e_min, dtype=float)),
                fits.Column(name='E_MAX', format='E', array=np.asarray(rmf.e_max, dtype=float)),
            ]
            hdul.append(fits.BinTableHDU.from_columns(cols_eb, name='EBOUNDS'))

        hdul.writeto(outp, overwrite=overwrite)
        return outp


class LightcurveWriter:
    def __init__(self, data: LightcurveData, outpath: str | Path):
        """保存 lightcurve 对象与输出路径 / Initialize a lightcurve writer by reference.

        不复制或校验 data，不创建目录、不写文件；覆盖策略在 write 中应用。
        Store data by reference and outpath as Path. No copying, validation, directory
        creation or file write; write() handles overwrite policy."""
        self.data = data
        self.outpath = Path(outpath)

    def write(self, *, overwrite: bool = False) -> Path:
        """写选定光变列及 GTI / Write selected lightcurve columns and optional GTI.

        TIME 使用 lc.time，优先 RATE/rate_err，再 COUNTS/counts_err；保存
        部分曝光/仪器头，GTI 使用对象当前数值。当前未完整写出 TIMEZERO、
        TIMEDEL、TIMESYS/MJDREF 或原始头，不是可保证绝对时间无损的 round trip。
        Write lc.time, preferring RATE/errors to COUNTS/errors, selected exposure/
        instrument metadata and current GTI values. TIMEZERO/TIMEDEL/TIMESYS/MJDREF
        and the original header are not fully serialized, so this is not a guaranteed
        lossless absolute-time round trip.

        返回 Path；父目录须存在，默认禁止覆盖；不从 value 自动填 counts/rate。
        Return Path; require existing parents and default no overwrite. Do not infer
        counts/rate fields from value. FITS errors propagate; no scientific reanalysis."""
        outp = self.outpath
        if outp.exists() and not overwrite:
            raise FileExistsError(str(outp))
        lc = self.data
        t = np.asarray(lc.time if lc.time is not None else np.asarray([], dtype=float), dtype=float)
        cols = [fits.Column(name='TIME', format='D', array=t)]
        if lc.rate is not None:
            cols.append(fits.Column(name='RATE', format='E', array=np.asarray(lc.rate, dtype=float)))
            if lc.rate_err is not None:
                cols.append(fits.Column(name='ERROR', format='E', array=np.asarray(lc.rate_err, dtype=float)))
        elif lc.counts is not None:
            cols.append(fits.Column(name='COUNTS', format='E', array=np.asarray(lc.counts, dtype=float)))
            if lc.counts_err is not None:
                cols.append(fits.Column(name='ERROR', format='E', array=np.asarray(lc.counts_err, dtype=float)))
        if lc.fracexp is not None:
            cols.append(fits.Column(name='FRACEXP', format='E', array=np.asarray(lc.fracexp, dtype=float)))
        if lc.quality is not None:
            cols.append(fits.Column(name='QUALITY', format='J', array=np.asarray(lc.quality, dtype=int)))

        hdu = fits.BinTableHDU.from_columns(cols, name='LIGHTCURVE')
        hdu.header['EXTNAME'] = 'LIGHTCURVE'
        if lc.exposure is not None:
            hdu.header['EXPOSURE'] = float(lc.exposure)
        if lc.meta is not None and getattr(lc.meta, 'telescop', None) is not None:
            hdu.header['TELESCOP'] = str(lc.meta.telescop)
        if lc.meta is not None and getattr(lc.meta, 'instrume', None) is not None:
            hdu.header['INSTRUME'] = str(lc.meta.instrume)

        prih = fits.PrimaryHDU()
        hdul = fits.HDUList([prih, hdu])

        if lc.gti_start is not None and lc.gti_stop is not None:
            cols_g = [
                fits.Column(name='START', format='D', array=np.asarray(lc.gti_start, dtype=float)),
                fits.Column(name='STOP', format='D', array=np.asarray(lc.gti_stop, dtype=float)),
            ]
            hdul.append(fits.BinTableHDU.from_columns(cols_g, name='GTI'))

        hdul.writeto(outp, overwrite=overwrite)
        return outp


class EventWriter:
    def __init__(self, data: EventData, outpath: str | Path):
        """保存 event 对象与输出路径 / Initialize a event writer by reference.

        不复制或校验 data，不创建目录、不写文件；覆盖策略在 write 中应用。
        Store data by reference and outpath as Path. No copying, validation, directory
        creation or file write; write() handles overwrite policy."""
        self.data = data
        self.outpath = Path(outpath)

    def write(self, *, overwrite: bool = False) -> Path:
        """写选定事件列与 GTI / Write selected event fields and optional GTI.

        TIME 使用 ev.time，写可用 PI/CHANNEL、坐标、能量；透传部分头，
        TIMEZERO 用当前 ev.timezero 覆盖，GTI 保留当前值。不写全部 raw_columns
        或原始任意扩展，不重算能量标定或坐标转换。
        Use ev.time, available PI/CHANNEL/coordinates/energy, selected header passthrough,
        current TIMEZERO and GTI values. Not all raw columns/arbitrary extensions are
        written; no energy calibration or coordinate transformation is performed.

        父目录须存在，默认拒绝覆盖，返回 Path；当前写出不执行 validate。
        Require existing parents, default no overwrite, return Path. No validate is run;
        FITS/shape errors propagate and selected float fields use reduced precision."""
        outp = self.outpath
        if outp.exists() and not overwrite:
            raise FileExistsError(str(outp))
        ev = self.data
        cols: list[fits.Column] = [fits.Column(name='TIME', format='D', array=np.asarray(ev.time, dtype=float))]
        if ev.pi is not None:
            cols.append(fits.Column(name='PI', format='J', array=np.asarray(ev.pi, dtype=int)))
        if ev.channel is not None:
            cols.append(fits.Column(name='CHANNEL', format='J', array=np.asarray(ev.channel, dtype=int)))
        if ev.x is not None:
            cols.append(fits.Column(name='X', format='E', array=np.asarray(ev.x, dtype=float)))
        if ev.y is not None:
            cols.append(fits.Column(name='Y', format='E', array=np.asarray(ev.y, dtype=float)))
        if ev.energy is not None:
            cols.append(fits.Column(name='ENERGY', format='E', array=np.asarray(ev.energy, dtype=float)))

        hdu_evt = fits.BinTableHDU.from_columns(cols, name='EVENTS')
        hdu_evt.header['EXTNAME'] = 'EVENTS'
        if ev.header is not None:
            for k, v in dict(ev.header).items():
                key = str(k).upper()
                if key in hdu_evt.header:
                    continue
                if key in {'XTENSION', 'BITPIX', 'NAXIS'}:
                    continue
                try:
                    hdu_evt.header[key] = v
                except Exception:
                    continue
        hdu_evt.header['TIMEZERO'] = float(ev.timezero or 0.0)
        prih = fits.PrimaryHDU()
        hdul = fits.HDUList([prih, hdu_evt])

        if ev.gti_start is not None and ev.gti_stop is not None:
            cols_g = [
                fits.Column(name='START', format='D', array=np.asarray(ev.gti_start, dtype=float)),
                fits.Column(name='STOP', format='D', array=np.asarray(ev.gti_stop, dtype=float)),
            ]
            hdul.append(fits.BinTableHDU.from_columns(cols_g, name='GTI'))

        hdul.writeto(outp, overwrite=overwrite)
        return outp


def guess_ogip_kind(path) -> Literal['arf', 'rmf', 'pha', 'lc', 'evt']:
    """启发式判断 FITS 产品类型 / Guess an OGIP product type by suffix and content.

    .arf/.rmf/.rsp/.pha/.pi 扩展名优先，不读文件确认；其余读取 HDU 名和
    TIME/RATE/COUNTS 列。未知内容最后回退 pha，返回类型字符串，不能
    把猜测成功当作格式有效性证明。
    Recognized suffixes take precedence without content confirmation; otherwise
    inspect HDU names and TIME/RATE/COUNTS. Unknown content falls back to pha.
    Return a kind string, not format validation; FITS/open errors may propagate."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith('.arf'):
        return 'arf'
    if name.endswith('.rmf') or name.endswith('.rsp'):
        return 'rmf'
    if name.endswith('.pha') or name.endswith('.pi'):
        return 'pha'
    with fits.open(p) as h:
        extnames = {getattr(x, 'name', '').upper() for x in h}
        if 'SPECRESP' in extnames and 'MATRIX' not in extnames:
            return 'arf'
        if 'MATRIX' in extnames or 'SPECRESP MATRIX' in extnames:
            return 'rmf'
        if 'SPECTRUM' in extnames:
            return 'pha'
        has_time = False
        for x in h:
            d = getattr(x, 'data', None)
            cols = getattr(d, 'columns', None)
            names = getattr(cols, 'names', ()) if cols is not None else ()
            if 'TIME' in names:
                has_time = True
                break
        if 'EVENTS' in extnames or has_time:
            for x in h:
                d = getattr(x, 'data', None)
                cols = getattr(d, 'columns', None)
                names = getattr(cols, 'names', ()) if cols is not None else ()
                if 'RATE' in names or 'COUNTS' in names:
                    return 'lc'
            return 'evt'
    return 'pha'


@overload
def readfits(path, kind: Literal['arf']) -> ArfData:
    """按指定或猜测类型读 OGIP / Read an OGIP product by explicit or inferred kind.

    path 为本地路径，kind 可为 arf/rmf/pha/lc/evt；None 委托 guess_ogip_kind。
    返回相应数据类，具体读入、单位与时间契约见各 reader；不自动 validate。
    非法 kind 抛 ValueError，FITS/读取异常上抛。重载声明仅提供静态返回类型。
    Use local path and optional arf/rmf/pha/lc/evt kind; None delegates to guessing.
    Return the corresponding data class using that reader's unit/time conventions,
    without validate. Unknown kind raises ValueError; read errors propagate.
    Overload declarations provide static return types only."""
    ...


@overload
def readfits(path, kind: Literal['rmf']) -> RmfData:
    """按指定或猜测类型读 OGIP / Read an OGIP product by explicit or inferred kind.

    path 为本地路径，kind 可为 arf/rmf/pha/lc/evt；None 委托 guess_ogip_kind。
    返回相应数据类，具体读入、单位与时间契约见各 reader；不自动 validate。
    非法 kind 抛 ValueError，FITS/读取异常上抛。重载声明仅提供静态返回类型。
    Use local path and optional arf/rmf/pha/lc/evt kind; None delegates to guessing.
    Return the corresponding data class using that reader's unit/time conventions,
    without validate. Unknown kind raises ValueError; read errors propagate.
    Overload declarations provide static return types only."""
    ...


@overload
def readfits(path, kind: Literal['pha']) -> PhaData:
    """按指定或猜测类型读 OGIP / Read an OGIP product by explicit or inferred kind.

    path 为本地路径，kind 可为 arf/rmf/pha/lc/evt；None 委托 guess_ogip_kind。
    返回相应数据类，具体读入、单位与时间契约见各 reader；不自动 validate。
    非法 kind 抛 ValueError，FITS/读取异常上抛。重载声明仅提供静态返回类型。
    Use local path and optional arf/rmf/pha/lc/evt kind; None delegates to guessing.
    Return the corresponding data class using that reader's unit/time conventions,
    without validate. Unknown kind raises ValueError; read errors propagate.
    Overload declarations provide static return types only."""
    ...


@overload
def readfits(path, kind: Literal['lc']) -> LightcurveData:
    """按指定或猜测类型读 OGIP / Read an OGIP product by explicit or inferred kind.

    path 为本地路径，kind 可为 arf/rmf/pha/lc/evt；None 委托 guess_ogip_kind。
    返回相应数据类，具体读入、单位与时间契约见各 reader；不自动 validate。
    非法 kind 抛 ValueError，FITS/读取异常上抛。重载声明仅提供静态返回类型。
    Use local path and optional arf/rmf/pha/lc/evt kind; None delegates to guessing.
    Return the corresponding data class using that reader's unit/time conventions,
    without validate. Unknown kind raises ValueError; read errors propagate.
    Overload declarations provide static return types only."""
    ...


@overload
def readfits(path, kind: Literal['evt']) -> EventData:
    """按指定或猜测类型读 OGIP / Read an OGIP product by explicit or inferred kind.

    path 为本地路径，kind 可为 arf/rmf/pha/lc/evt；None 委托 guess_ogip_kind。
    返回相应数据类，具体读入、单位与时间契约见各 reader；不自动 validate。
    非法 kind 抛 ValueError，FITS/读取异常上抛。重载声明仅提供静态返回类型。
    Use local path and optional arf/rmf/pha/lc/evt kind; None delegates to guessing.
    Return the corresponding data class using that reader's unit/time conventions,
    without validate. Unknown kind raises ValueError; read errors propagate.
    Overload declarations provide static return types only."""
    ...


@overload
def readfits(path, kind: None = ...) -> OgipData:
    """按指定或猜测类型读 OGIP / Read an OGIP product by explicit or inferred kind.

    path 为本地路径，kind 可为 arf/rmf/pha/lc/evt；None 委托 guess_ogip_kind。
    返回相应数据类，具体读入、单位与时间契约见各 reader；不自动 validate。
    非法 kind 抛 ValueError，FITS/读取异常上抛。重载声明仅提供静态返回类型。
    Use local path and optional arf/rmf/pha/lc/evt kind; None delegates to guessing.
    Return the corresponding data class using that reader's unit/time conventions,
    without validate. Unknown kind raises ValueError; read errors propagate.
    Overload declarations provide static return types only."""
    ...


def readfits(path, kind: Optional[Literal['arf', 'rmf', 'pha', 'lc', 'evt']] = None) -> OgipData:
    """按指定或猜测类型读 OGIP / Read an OGIP product by explicit or inferred kind.

    path 为本地路径，kind 可为 arf/rmf/pha/lc/evt；None 委托 guess_ogip_kind。
    返回相应数据类，具体读入、单位与时间契约见各 reader；不自动 validate。
    非法 kind 抛 ValueError，FITS/读取异常上抛。重载声明仅提供静态返回类型。
    Use local path and optional arf/rmf/pha/lc/evt kind; None delegates to guessing.
    Return the corresponding data class using that reader's unit/time conventions,
    without validate. Unknown kind raises ValueError; read errors propagate.
    Overload declarations provide static return types only."""
    k = kind or guess_ogip_kind(path)
    if k == 'arf':
        return read_arf(path)
    if k == 'rmf':
        return read_rmf(path)
    if k == 'pha':
        return read_pha(path)
    if k == 'lc':
        return read_lc(path)
    if k == 'evt':
        return read_evt(path)
    raise ValueError(f"Unknown OGIP kind: {k}")


def read_arf(path) -> ArfData:
    """读取对应 OGIP 数据 / Construct OgipArfReader and return its read() result.

    path 为本地文件，单位、时间和错误条件遵循该 reader；不额外 validate。
    Use a local path with that reader's unit/time/error contract. No extra validate
    call, network fetch or persistent reader object is returned."""
    return OgipArfReader(path).read()


def read_rmf(path) -> RmfData:
    """读取对应 OGIP 数据 / Construct OgipRmfReader and return its read() result.

    path 为本地文件，单位、时间和错误条件遵循该 reader；不额外 validate。
    Use a local path with that reader's unit/time/error contract. No extra validate
    call, network fetch or persistent reader object is returned."""
    return OgipRmfReader(path).read()


def read_pha(path) -> PhaData:
    """读取对应 OGIP 数据 / Construct OgipPhaReader and return its read() result.

    path 为本地文件，单位、时间和错误条件遵循该 reader；不额外 validate。
    Use a local path with that reader's unit/time/error contract. No extra validate
    call, network fetch or persistent reader object is returned."""
    return OgipPhaReader(path).read()


def read_lc(path) -> LightcurveData:
    """读取对应 OGIP 数据 / Construct OgipLightcurveReader and return its read() result.

    path 为本地文件，单位、时间和错误条件遵循该 reader；不额外 validate。
    Use a local path with that reader's unit/time/error contract. No extra validate
    call, network fetch or persistent reader object is returned."""
    return OgipLightcurveReader(path).read()


def read_evt(path) -> EventData:
    """读取对应 OGIP 数据 / Construct OgipEventReader and return its read() result.

    path 为本地文件，单位、时间和错误条件遵循该 reader；不额外 validate。
    Use a local path with that reader's unit/time/error contract. No extra validate
    call, network fetch or persistent reader object is returned."""
    return OgipEventReader(path).read()


def write_pha(data: PhaData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """写对应 OGIP 数据 / Construct PhaWriter and write selected object fields.

    返回 outpath 的 Path，overwrite 默认 False；不创建父目录。具体列、
    精度及头字段保留范围见 writer；不是任意原始 FITS 的无损复制。
    Return output Path, default overwrite=False, without creating parents. Refer
    to the writer for columns, precision and metadata preservation; this is not
    a lossless copy of an arbitrary original FITS. Errors propagate."""
    return PhaWriter(data, outpath).write(overwrite=overwrite)


def write_arf(data: ArfData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """写对应 OGIP 数据 / Construct ArfWriter and write selected object fields.

    返回 outpath 的 Path，overwrite 默认 False；不创建父目录。具体列、
    精度及头字段保留范围见 writer；不是任意原始 FITS 的无损复制。
    Return output Path, default overwrite=False, without creating parents. Refer
    to the writer for columns, precision and metadata preservation; this is not
    a lossless copy of an arbitrary original FITS. Errors propagate."""
    return ArfWriter(data, outpath).write(overwrite=overwrite)


def write_rmf(data: RmfData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """写对应 OGIP 数据 / Construct RmfWriter and write selected object fields.

    返回 outpath 的 Path，overwrite 默认 False；不创建父目录。具体列、
    精度及头字段保留范围见 writer；不是任意原始 FITS 的无损复制。
    Return output Path, default overwrite=False, without creating parents. Refer
    to the writer for columns, precision and metadata preservation; this is not
    a lossless copy of an arbitrary original FITS. Errors propagate."""
    return RmfWriter(data, outpath).write(overwrite=overwrite)


def write_lc(data: LightcurveData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """写对应 OGIP 数据 / Construct LightcurveWriter and write selected object fields.

    返回 outpath 的 Path，overwrite 默认 False；不创建父目录。具体列、
    精度及头字段保留范围见 writer；不是任意原始 FITS 的无损复制。
    Return output Path, default overwrite=False, without creating parents. Refer
    to the writer for columns, precision and metadata preservation; this is not
    a lossless copy of an arbitrary original FITS. Errors propagate."""
    return LightcurveWriter(data, outpath).write(overwrite=overwrite)


def write_evt(data: EventData, outpath: str | Path, *, overwrite: bool = False) -> Path:
    """写对应 OGIP 数据 / Construct EventWriter and write selected object fields.

    返回 outpath 的 Path，overwrite 默认 False；不创建父目录。具体列、
    精度及头字段保留范围见 writer；不是任意原始 FITS 的无损复制。
    Return output Path, default overwrite=False, without creating parents. Refer
    to the writer for columns, precision and metadata preservation; this is not
    a lossless copy of an arbitrary original FITS. Errors propagate."""
    return EventWriter(data, outpath).write(overwrite=overwrite)


@overload
def writefits(data: PhaData, outpath: str | Path, kind: Literal['pha'], *, overwrite: bool = ...) -> Path:
    """按类型委托 OGIP 写出 / Dispatch OGIP writing by an explicit or object kind.

    kind 优先；None 对 PhaData 用 pha，其他读取 data.kind。不运行类型转换，
    强制不匹配 kind 可能在具体 writer 报错。overwrite 默认 False。
    返回 Path，不创建父目录；选择性写出，不保证原始文件无损往返。
    Prefer explicit kind; otherwise PhaData implies pha and other objects use their
    kind field. No conversion is performed; mismatched explicit kinds can fail in
    the writer. Default overwrite=False. Return Path without parent creation; these
    writers are selective, not lossless original-file round trips."""
    ...


@overload
def writefits(data: PhaData, outpath: str | Path, kind: None = ..., *, overwrite: bool = ...) -> Path:
    """按类型委托 OGIP 写出 / Dispatch OGIP writing by an explicit or object kind.

    kind 优先；None 对 PhaData 用 pha，其他读取 data.kind。不运行类型转换，
    强制不匹配 kind 可能在具体 writer 报错。overwrite 默认 False。
    返回 Path，不创建父目录；选择性写出，不保证原始文件无损往返。
    Prefer explicit kind; otherwise PhaData implies pha and other objects use their
    kind field. No conversion is performed; mismatched explicit kinds can fail in
    the writer. Default overwrite=False. Return Path without parent creation; these
    writers are selective, not lossless original-file round trips."""
    ...


@overload
def writefits(data: OgipWritableData, outpath: str | Path, kind: Literal['arf', 'rmf', 'lc', 'evt'], *, overwrite: bool = ...) -> Path:
    """按类型委托 OGIP 写出 / Dispatch OGIP writing by an explicit or object kind.

    kind 优先；None 对 PhaData 用 pha，其他读取 data.kind。不运行类型转换，
    强制不匹配 kind 可能在具体 writer 报错。overwrite 默认 False。
    返回 Path，不创建父目录；选择性写出，不保证原始文件无损往返。
    Prefer explicit kind; otherwise PhaData implies pha and other objects use their
    kind field. No conversion is performed; mismatched explicit kinds can fail in
    the writer. Default overwrite=False. Return Path without parent creation; these
    writers are selective, not lossless original-file round trips."""
    ...


def writefits(
    data: OgipWritableData,
    outpath: str | Path,
    kind: Optional[Literal['arf', 'rmf', 'pha', 'lc', 'evt']] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """按类型委托 OGIP 写出 / Dispatch OGIP writing by an explicit or object kind.

    kind 优先；None 对 PhaData 用 pha，其他读取 data.kind。不运行类型转换，
    强制不匹配 kind 可能在具体 writer 报错。overwrite 默认 False。
    返回 Path，不创建父目录；选择性写出，不保证原始文件无损往返。
    Prefer explicit kind; otherwise PhaData implies pha and other objects use their
    kind field. No conversion is performed; mismatched explicit kinds can fail in
    the writer. Default overwrite=False. Return Path without parent creation; these
    writers are selective, not lossless original-file round trips."""
    k = kind
    if k is None:
        if isinstance(data, PhaData):
            k = 'pha'
        else:
            k = cast(Optional[Literal['arf', 'rmf', 'pha', 'lc', 'evt']], getattr(data, 'kind', None))
    if k == 'pha':
        return write_pha(cast(PhaData, data), outpath, overwrite=overwrite)
    if k == 'arf':
        return write_arf(cast(ArfData, data), outpath, overwrite=overwrite)
    if k == 'rmf':
        return write_rmf(cast(RmfData, data), outpath, overwrite=overwrite)
    if k == 'lc':
        return write_lc(cast(LightcurveData, data), outpath, overwrite=overwrite)
    if k == 'evt':
        return write_evt(cast(EventData, data), outpath, overwrite=overwrite)
    raise ValueError(f"Unknown writefits kind: {k!r}")


__all__ = [
    "OgipArfReader",
    "OgipRmfReader",
    "OgipPhaReader",
    "OgipLightcurveReader",
    "OgipEventReader",
    "ArfReader",
    "RmfReader",
    "RspReader",
    "LightcurveReader",
    "OgipData",
    "band_from_arf_bins",
    "channel_mask_from_ebounds",
    "guess_ogip_kind",
    "readfits",
    "read_arf",
    "read_rmf",
    "read_pha",
    "read_lc",
    "read_evt",
    "PhaWriter",
    "ArfWriter",
    "RmfWriter",
    "LightcurveWriter",
    "EventWriter",
    "write_arf",
    "write_rmf",
    "write_lc",
    "write_evt",
    "write_pha",
    "writefits",
]
