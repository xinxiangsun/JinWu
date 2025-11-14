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
    _validation: Optional[ValidationReport] = field(init=False, default=None, repr=False, compare=False)

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

    @property
    def validation(self) -> Optional[ValidationReport]:
        return self._validation

# ---------------- Specialized Base Classes ----------------

@dataclass(slots=True)
class OgipTimeSeriesBase(OgipFitsBase):
    """OGIP-93-003 风格的时间序列 (光变或事件) 基类。

    子类需提供：columns (Sequence[str]) 用于验证列名。
    """

    REQUIRED_KEYS = ["TELESCOP", "INSTRUME", "TIMESYS"]
    OPTIONAL_KEYS = ["OBJECT", "OBS_ID", "TIMEUNIT", "MJDREF", "MJDREFI", "MJDREFF"]
    REQUIRED_COLUMNS_ANY = [["TIME"]]  # 至少包含 TIME

    def validate(self) -> ValidationReport:
        rpt = super().validate()
        # Header keyword checks
        for k in self.REQUIRED_KEYS:
            if k not in self.header:
                rpt.add('WARN', 'MISSING_KEY', f"Required key '{k}' not found (OGIP-93-003).")
        # Column checks
        cols = getattr(self, 'columns', ()) or ()
        colset = {c.upper() for c in cols}
        for group in self.REQUIRED_COLUMNS_ANY:
            if not any(c.upper() in colset for c in group):
                rpt.add('ERROR', 'MISSING_COLUMN', f"Missing required column group: {group}")
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

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
