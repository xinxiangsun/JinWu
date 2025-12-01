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
from typing import Dict, Any, Optional, List, Sequence, ClassVar

__all__ = [
    "ValidationMessage", "ValidationReport",
    "OgipFitsBase", "OgipTimeSeriesBase", "OgipSpectrumBase", "OgipResponseBase"
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

@dataclass(slots=True)
class OgipFitsBase:
    path: Path
    header: Dict[str, Any]
    meta: Any  # OgipMeta (延迟导入避免循环)
    headers_dump: Any  # FitsHeaderDump
    _validation: ClassVar[Optional[ValidationReport]] = None

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

@dataclass(slots=True)
class OgipTimeSeriesBase(OgipFitsBase):
    """OGIP-93-003 风格的时间序列 (光变或事件) 基类。

    子类需提供：columns (Sequence[str]) 用于验证列名。
    """

    # 根据 OGIP-94-003 (Events) 的建议/要求，时间序列至少应包含时间系统与时间单位
    REQUIRED_KEYS = ["TELESCOP", "INSTRUME", "TIMESYS", "TIMEUNIT"]
    OPTIONAL_KEYS = ["OBJECT", "OBS_ID", "MJDREF", "MJDREFI", "MJDREFF", "TIMEZERO", "TREFPOS", "DATE-OBS"]
    REQUIRED_COLUMNS_ANY = [["TIME"]]  # 至少包含 TIME

    def validate(self) -> ValidationReport:
        rpt = super().validate()
        # Header keyword checks
        for k in self.REQUIRED_KEYS:
            if k not in self.header:
                # 对于时间序列类，缺少 TIMESYS/TIMEUNIT 通常是较严重的 WARN
                rpt.add('WARN', 'MISSING_KEY', f"Required key '{k}' not found (OGIP-93-003).")
        # Column checks
        cols = getattr(self, 'columns', ()) or ()
        colset = {c.upper() for c in cols}
        for group in self.REQUIRED_COLUMNS_ANY:
            if not any(c.upper() in colset for c in group):
                rpt.add('ERROR', 'MISSING_COLUMN', f"Missing required column group: {group}")
        # MJDREF 合并检查（MJDREFI+MJDREFF 或 MJDREF 之一应存在）
        hdr = self.header or {}
        if not (('MJDREF' in hdr) or (('MJDREFI' in hdr) or ('MJDREFF' in hdr))):
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

@dataclass(slots=True)
class OgipSpectrumBase(OgipFitsBase):
    """OGIP-92-007 / 007a PHA 能谱基类。"""

    REQUIRED_KEYS = ["TELESCOP", "INSTRUME", "CHANTYPE"]
    OPTIONAL_KEYS = ["FILTER", "EXPOSURE", "BACKSCAL", "AREASCAL", "RESPFILE", "ANCRFILE", "CORRFILE", "CORRSCAL"]
    REQUIRED_COLUMNS = ["CHANNEL", "COUNTS"]  # 简化：通常至少有 CHANNEL/COUNTS

    def validate(self) -> ValidationReport:
        rpt = super().validate()
        for k in self.REQUIRED_KEYS:
            if k not in self.header:
                rpt.add('WARN', 'MISSING_KEY', f"Required key '{k}' not found (OGIP-92-007).")
        cols = getattr(self, 'columns', ()) or ()
        colset = {c.upper() for c in cols}
        for c in self.REQUIRED_COLUMNS:
            if c.upper() not in colset:
                rpt.add('ERROR', 'MISSING_COLUMN', f"Missing required column '{c}' for PHA.")
        # Basic sanity: exposure>0 if present
        exp_val = self.header.get('EXPOSURE', self.header.get('EXPTIME', None))
        if exp_val is not None:
            try:
                if float(exp_val) <= 0:
                    rpt.add('WARN', 'BAD_EXPOSURE', f"Non-positive exposure value: {exp_val}")
            except Exception:
                rpt.add('WARN', 'BAD_EXPOSURE', f"Exposure not numeric: {exp_val}")
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

@dataclass(slots=True)
class OgipResponseBase(OgipFitsBase):
    """CAL/GEN/92-002 响应 (ARF/RMF) 基类。"""

    REQUIRED_KEYS_ANY = [["TELESCOP"], ["INSTRUME"], ["DETNAM", "DETNAME"]]
    REQUIRED_COLUMNS_ARF = ["ENERG_LO", "ENERG_HI", "SPECRESP"]
    REQUIRED_COLUMNS_RMF_MIN = ["ENERG_LO", "ENERG_HI", "MATRIX"]  # 简化

    def validate(self) -> ValidationReport:
        rpt = super().validate()
        # Header presence: at least one from each group
        hdr = self.header
        for group in self.REQUIRED_KEYS_ANY:
            if not any(g in hdr for g in group):
                rpt.add('WARN', 'MISSING_KEY', f"Missing one of required keys {group} (CAL/GEN/92-002).")
        # Column checks will be done in concrete subclasses where we know type
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

# 具体数据类将继承上述基类并在自身 validate() 中补充列检查。
