"""OGIP 标准基类与验证框架 (初版)

涵盖：
- 通用 FITS 基类 OgipFitsBase（统一 path/header/meta/headers_dump）
- 光变/事件 (OGIP-93-003) 基类 OgipTimeSeriesBase
- PHA 能谱 (OGIP-92-007 / 007a) 基类 OgipSpectrumBase
- 响应 (CAL/GEN/92-002: RMF/ARF) 基类 OgipResponseBase

提供统一的 validate() 机制：
- 头关键字必需/可选检查
- 表格列名必需/可选检查
- 结果分级：ERROR / WARN / INFO

后续可扩展：
- 更细粒度的值域校验 (如 EXPOSURE>0, CHANNEL 单调递增等)
- 与具体 OGIP 文档的章节引用

English summary
---------------
OGIP base class & validation scaffold covering time series, spectra, and response files.
Unified validate() returning structured report with severity levels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Sequence

import numpy as np

__all__ = [
    "ValidationMessage", "ValidationReport",
    "OgipFitsBase", "OgipTimeSeriesBase", "OgipSpectrumBase", "OgipResponseBase",
    "check_response_compatibility",
]

# ---------------- Validation Data Structures ----------------

@dataclass(slots=True)
class ValidationMessage:
    level: str  # 'ERROR' | 'WARN' | 'INFO'
    code: str   # short symbolic code, e.g. MISSING_KEY
    message: str

@dataclass(slots=True)
class ValidationReport:
    kind: str
    path: Path
    ok: bool
    messages: List[ValidationMessage] = field(default_factory=list)

    def add(self, level: str, code: str, message: str):
        self.messages.append(ValidationMessage(level=level, code=code, message=message))

    def errors(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.level == 'ERROR']

    def warnings(self) -> List[ValidationMessage]:
        return [m for m in self.messages if m.level == 'WARN']

# ---------------- Base FITS Class ----------------

@dataclass(slots=True, eq=False)
class OgipFitsBase:
    path: Path
    header: Dict[str, Any]
    meta: Any  # OgipMeta (延迟导入避免循环)
    headers_dump: Any  # FitsHeaderDump
    _validation: Optional[ValidationReport] = field(default=None, init=False, repr=False, compare=False)

    def validate(self) -> ValidationReport:
        """通用层：仅检查路径与 header 存在性。子类会扩展。"""
        rpt = ValidationReport(kind=self.__class__.__name__, path=self.path, ok=True)
        if not self.path.exists():
            rpt.add('ERROR', 'FILE_NOT_FOUND', f"File not found: {self.path}")
        if self.header is None:
            rpt.add('ERROR', 'NO_HEADER', 'Primary header missing')
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

    def _has_key_ci(self, key: str) -> bool:
        return self.get_keyword_ci(key, default=None) is not None

    def _has_any_key_ci(self, keys: Sequence[str]) -> bool:
        return any(self._has_key_ci(k) for k in keys)

    def get_keyword_ci(self, key: str, default: Optional[Any] = None) -> Any:
        """大小写不敏感地从 header 中读取关键字（若 header 为 dict）.

        返回关键字值或提供的 default。便于统一处理 FITS 关键字的大小写差异。
        """
        if self.header is None:
            return default
        # header 可能是 astropy Header（支持 __contains__ case-insensitive），
        # 但通常我们将其传为 dict；先尝试直接查找，再按大写匹配。
        try:
            if key in self.header:
                return self.header[key]
        except Exception:
            pass
        # case-insensitive match
        up = key.upper()
        for k, v in dict(self.header).items():
            try:
                if str(k).upper() == up:
                    return v
            except Exception:
                continue
        return default

    @property
    def validation(self) -> Optional[ValidationReport]:
        return self._validation

# ---------------- Specialized Base Classes ----------------

@dataclass(slots=True, eq=False)
class OgipTimeSeriesBase(OgipFitsBase):
    """OGIP-93-003 风格的时间序列 (光变或事件) 基类。

    子类需提供：columns (Sequence[str]) 用于验证列名。
    """

    # 根据 OGIP-94-003 (Events) 的建议/要求，时间序列至少应包含时间系统与时间单位
    REQUIRED_KEYS = ["TELESCOP", "INSTRUME", "TIMESYS", "TIMEUNIT"]
    CRITICAL_KEYS = ["TIMESYS", "TIMEUNIT"]
    OPTIONAL_KEYS = ["OBJECT", "OBS_ID", "MJDREF", "MJDREFI", "MJDREFF", "TIMEZERO", "TREFPOS", "DATE-OBS"]
    REQUIRED_COLUMNS_ANY = [["TIME"]]  # 至少包含 TIME

    def validate(self) -> ValidationReport:
        rpt = super().validate()
        # Header keyword checks
        for k in self.REQUIRED_KEYS:
            if not self._has_key_ci(k):
                lvl = 'ERROR' if k in self.CRITICAL_KEYS else 'WARN'
                rpt.add(lvl, 'MISSING_KEY', f"Required key '{k}' not found (OGIP-93-003).")
        # Column checks
        cols = getattr(self, 'columns', ()) or ()
        colset = {c.upper() for c in cols}
        for group in self.REQUIRED_COLUMNS_ANY:
            if not any(c.upper() in colset for c in group):
                rpt.add('ERROR', 'MISSING_COLUMN', f"Missing required column group: {group}")
        # MJDREF 合并检查（MJDREFI+MJDREFF 或 MJDREF 之一应存在）
        if not (self._has_key_ci('MJDREF') or self._has_any_key_ci(['MJDREFI', 'MJDREFF'])):
            rpt.add('WARN', 'MISSING_MJDREF', 'MJDREF / (MJDREFI+MJDREFF) not found in header; absolute times may be ambiguous.')
        # TIMEUNIT/TIMESYS presence already warned above; optionally validate common values
        timeunit = None
        try:
            timeunit = self.get_keyword_ci('TIMEUNIT')
        except Exception:
            timeunit = (self.header or {}).get('TIMEUNIT')
        if timeunit is not None and str(timeunit).upper() not in ('S', 'SEC', 'SECOND', 'SECONDS'):
            # not an error, but note uncommon units
            rpt.add('INFO', 'UNUSUAL_TIMEUNIT', f"TIMEUNIT='{timeunit}'")
        # GTI 自洽：stop>=start 且区间有序（读端已填充 gti_start/gti_stop 时）
        g0 = getattr(self, 'gti_start', None)
        g1 = getattr(self, 'gti_stop', None)
        if g0 is not None and g1 is not None:
            try:
                a = np.asarray(g0, dtype=float)
                b = np.asarray(g1, dtype=float)
                if a.shape == b.shape and a.size > 0:
                    if bool(np.any(b < a)):
                        rpt.add('ERROR', 'BAD_GTI', 'GTI contains interval(s) with STOP < START.')
                    if a.size > 1 and bool(np.any(np.diff(a) < 0)):
                        rpt.add('WARN', 'UNSORTED_GTI', 'GTI START values are not sorted ascending.')
            except Exception:
                pass
        # GTI: many EVENTS files include a GTI extension; subclasses or readers should populate a gti field
        # Provide an extension point: subclasses/readers can implement `extract_gti(hdul)` to fill gti.
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

    def extract_gti(self, hdul: Any) -> Optional[list]:
        """Extension point: parse GTI from an opened `HDUList` and return list of (start, stop) tuples.

        Default implementation searches for an extension named 'GTI' (case-insensitive) and, if
        present, returns a list of (START, STOP) pairs. Readers should call this to populate event
        objects' GTI field.

        方法：GTI 扩展按 EXTNAME='GTI' 与 START/STOP 列解析（TSTART/TSTOP 为别名）。
        参考：OGIP/93-003 "The Proposed Timing FITS File Format for High Energy
              Astrophysics Data"（GTI 扩展含 START/STOP 两列）。
        """
        if hdul is None:
            return None
        for hdu in hdul:
            hdr = getattr(hdu, 'header', {})
            name = (hdr.get('EXTNAME') or '').upper() if hdr else ''
            if name == 'GTI' or getattr(hdu, 'name', '').upper() == 'GTI':
                data = getattr(hdu, 'data', None)
                if data is None:
                    return None
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
                    except Exception:
                        return None
        return None

@dataclass(slots=True, eq=False)
class OgipSpectrumBase(OgipFitsBase):
    """OGIP-92-007 / 007a PHA 能谱基类。"""

    REQUIRED_KEYS = ["TELESCOP", "INSTRUME", "CHANTYPE"]
    OPTIONAL_KEYS = ["FILTER", "EXPOSURE", "BACKSCAL", "AREASCAL", "RESPFILE", "ANCRFILE", "CORRFILE", "CORRSCAL"]
    REQUIRED_COLUMNS = ["CHANNEL"]
    OPTIONAL_RATE_COLUMNS = ["COUNTS", "RATE"]

    def validate(self) -> ValidationReport:
        # 显式调用（不用 super()）：具体类会以 `OgipSpectrumBase.validate(self)`
        # 未绑定方式委托到本方法。
        rpt = OgipFitsBase.validate(self)
        for k in self.REQUIRED_KEYS:
            if not self._has_key_ci(k):
                rpt.add('WARN', 'MISSING_KEY', f"Required key '{k}' not found (OGIP-92-007).")
        cols = getattr(self, 'columns', ()) or ()
        colset = {c.upper() for c in cols}
        for c in self.REQUIRED_COLUMNS:
            if c.upper() not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"Missing required column '{c}' for PHA.")
        if not any(c in colset for c in self.OPTIONAL_RATE_COLUMNS):
            rpt.add('ERROR', 'MISSING_COLUMN', "Missing required PHA value column: one of ['COUNTS', 'RATE']")
        # Basic sanity: exposure>0 if present
        exp_val = self.get_keyword_ci('EXPOSURE', self.get_keyword_ci('EXPTIME', None))
        if exp_val is not None:
            try:
                if float(exp_val) <= 0:
                    rpt.add('WARN', 'BAD_EXPOSURE', f"Non-positive exposure value: {exp_val}")
            except Exception:
                rpt.add('WARN', 'BAD_EXPOSURE', f"Exposure not numeric: {exp_val}")
        # HDU 分类与版本（对齐 heasp pha::read：检查 HDUCLAS1=SPECTRUM）
        # 方法：PHA 扩展要求 HDUCLAS1='SPECTRUM'（存在但不匹配则告警），
        #       与 heasp 谱扩展的定位/写出约定一致（缺 HDUVERS 亦告警，
        #       heasp 写出时总写 HDUVERS）。
        # 参考：HEASoft 6.37 heacore/heasp/pha.cxx pha::write（SPwriteKey
        #       HDUCLASS="OGIP"、HDUCLAS1="SPECTRUM"、HDUVERS="1.2.1"）；
        #       OGIP/92-007 "The OGIP Spectral File Format"（EXTNAME=SPECTRUM）。
        clas1 = self.get_keyword_ci('HDUCLAS1', None)
        if clas1 is not None and str(clas1).upper() != 'SPECTRUM':
            rpt.add('WARN', 'BAD_HDUCLAS1', f"HDUCLAS1={clas1!r}, expected 'SPECTRUM' for a PHA.")
        if not self._has_key_ci('HDUVERS'):
            rpt.add('WARN', 'MISSING_HDUVERS', 'HDUVERS missing (OGIP-92-007 version declaration).')
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

@dataclass(slots=True, eq=False)
class OgipResponseBase(OgipFitsBase):
    """CAL/GEN/92-002 响应 (ARF/RMF) 基类。"""

    REQUIRED_KEYS_ANY = [["TELESCOP"], ["INSTRUME"], ["DETNAM", "DETNAME"]]
    REQUIRED_COLUMNS_ARF = ["ENERG_LO", "ENERG_HI", "SPECRESP"]
    REQUIRED_COLUMNS_RMF_MIN = ["ENERG_LO", "ENERG_HI", "MATRIX"]  # 简化
    # HDUCLAS2 合法取值（heasp 扩展定位同时接受 EXTNAME 或这对关键字）
    RESPONSE_HDUCLAS2 = ("SPECRESP", "RSP_MATRIX")

    def validate(self) -> ValidationReport:
        # 显式调用（不用 super()）：具体类会以 `OgipResponseBase.validate(self)`
        # 未绑定方式委托到本方法。
        rpt = OgipFitsBase.validate(self)
        # Header presence: at least one from each group
        for group in self.REQUIRED_KEYS_ANY:
            if not self._has_any_key_ci(group):
                rpt.add('WARN', 'MISSING_KEY', f"Missing one of required keys {group} (CAL/GEN/92-002).")
        # HDU 分类与版本（对齐 heasp 扩展定位规则：HDUCLAS1=RESPONSE +
        # HDUCLAS2=SPECRESP/RSP_MATRIX；缺失时靠 EXTNAME 定位，降为 WARN）。
        # 方法：响应扩展的定位/校验规则：HDUCLAS1='RESPONSE' 且
        #       HDUCLAS2∈{SPECRESP(ARF), RSP_MATRIX(RMF)}；缺 HDUCLAS1 时
        #       降级由 EXTNAME 定位（仅告警，与 heasp 的回退顺序一致）。
        # 参考：HEASoft 6.37 heacore/heasp/rmf.cxx（readMatrix/read：先 EXTNAME，
        #       回退 HDUCLAS1=RESPONSE + HDUCLAS2=RSP_MATRIX/EBOUNDS）与
        #       heacore/heasp/arf.cxx（read：HDUCLAS1=RESPONSE + HDUCLAS2=SPECRESP）；
        #       格式定义 CAL/GEN/92-002。
        clas1 = self.get_keyword_ci('HDUCLAS1', None)
        clas2 = self.get_keyword_ci('HDUCLAS2', None)
        if clas1 is None or str(clas1).upper() != 'RESPONSE':
            rpt.add('WARN', 'MISSING_HDUCLAS',
                    f"HDUCLAS1 != 'RESPONSE' (got {clas1!r}); extension must be located by EXTNAME.")
        if clas2 is not None and str(clas2).upper() not in self.RESPONSE_HDUCLAS2:
            rpt.add('WARN', 'BAD_HDUCLAS2',
                    f"HDUCLAS2={clas2!r} not in {self.RESPONSE_HDUCLAS2}.")
        if not self._has_key_ci('HDUVERS'):
            rpt.add('WARN', 'MISSING_HDUVERS', 'HDUVERS missing (OGIP response version declaration).')
        # Column checks will be done in concrete subclasses where we know type
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

# 具体数据类将继承上述基类并在自身 validate() 中补充列检查。


def check_response_compatibility(spectrum: Any, response: Any) -> ValidationReport:
    """谱 ↔ 响应兼容性检查（对齐 heasp "checking an RMF and an ARF/spectrum
    for compatibility" 语义）。

    检查项（能取到字段才查，取不到则跳过）：
    - 通道数/通道范围：谱的 CHANNEL 数组应落在响应的通道约定（TLMIN+DETCHANS）内；
    - DETCHANS 与谱通道数的一致性提示。
    返回 ValidationReport（ok=False 表示不兼容，不应直接相乘/拟合）。
    """
    rpt = ValidationReport(kind='compatibility', path=Path(str(getattr(spectrum, 'path', '<in-memory>'))), ok=True)
    try:
        tlmin = getattr(response, 'tlmin', None)
        det_chans = getattr(response, 'det_chans', None)
        ch = getattr(spectrum, 'channels', None)
        channels = np.asarray([] if ch is None else ch, dtype=int)
        if channels.size > 0 and tlmin is not None and det_chans is not None:
            lo, hi = int(channels.min()), int(channels.max())
            r_lo, r_hi = int(tlmin), int(tlmin) + int(det_chans) - 1
            if lo < r_lo or hi > r_hi:
                rpt.add('ERROR', 'INCOMPATIBLE_CHANNELS',
                        f"Spectrum channels [{lo}, {hi}] exceed response channel range "
                        f"[{r_lo}, {r_hi}] (TLMIN={tlmin}, DETCHANS={det_chans}).")
            sp_det = getattr(spectrum, 'det_chans', None)
            if sp_det is not None and int(sp_det) != int(det_chans):
                rpt.add('WARN', 'DETCHANS_MISMATCH',
                        f"Spectrum DETCHANS={sp_det} differs from response DETCHANS={det_chans}.")
        elif channels.size > 0:
            rpt.add('INFO', 'COMPAT_NOT_CHECKED',
                    'Response TLMIN/DETCHANS unavailable; channel compatibility not checked.')
    except Exception as exc:
        rpt.add('WARN', 'COMPAT_CHECK_FAILED', f"Compatibility check failed: {exc}")
    rpt.ok = len(rpt.errors()) == 0
    return rpt
