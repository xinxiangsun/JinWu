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
        """追加诊断消息 / Append a diagnostic message to this report.

        修改 messages，返回 None；不会同步更新 ok，也不校验 level 字符串。
        Mutate messages and return None. Does not recompute ok or validate level.
        """
        self.messages.append(ValidationMessage(level=level, code=code, message=message))

    def errors(self) -> List[ValidationMessage]:
        """返回 ERROR 消息列表 / Return messages whose level is exactly ERROR.

        创建新列表但不复制消息对象，不改变报告。
        Return a new list referencing the same messages without changing the report.
        """
        return [m for m in self.messages if m.level == 'ERROR']

    def warnings(self) -> List[ValidationMessage]:
        """返回 WARN 消息列表 / Return messages whose level is exactly WARN.

        仅筛选消息，不把 INFO 或未知级别视作告警。
        Filter messages; INFO and unrecognized levels are not warnings here.
        """
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
        """执行并缓存基础校验 / Run and cache basic FITS-object validation.

        只检查 path.exists() 与 header 是否为 None，不重新读取文件。
        返回 ValidationReport 并更新 _validation；ok 仅表示无 ERROR。
        子类可扩展检查，基础通过不证明完整 OGIP 或科学兼容性。
        Check only path existence and whether header is None, without rereading
        the file. Return/store a report; ok means no ERROR messages. Subclasses
        extend it; passing these checks does not establish full OGIP validity.
        """
        rpt = ValidationReport(kind=self.__class__.__name__, path=self.path, ok=True)
        if not self.path.exists():
            rpt.add('ERROR', 'FILE_NOT_FOUND', f"File not found: {self.path}")
        if self.header is None:
            rpt.add('ERROR', 'NO_HEADER', 'Primary header missing')
        rpt.ok = len(rpt.errors()) == 0
        self._validation = rpt
        return rpt

    def _has_key_ci(self, key: str) -> bool:
        """判断关键字是否有非 None 值 / Test for a non-None header value.

        大小写不敏感；值为 None 的现有卡片也视为缺失。
        Case-insensitive; a present card with value None counts as missing.
        """
        return self.get_keyword_ci(key, default=None) is not None

    def _has_any_key_ci(self, keys: Sequence[str]) -> bool:
        """判断候选关键字是否至少一个有效 / Check any candidate header key.

        按 _has_key_ci 语义短路判断；空候选集返回 False。
        Short-circuit using _has_key_ci; an empty sequence returns False.
        """
        return any(self._has_key_ci(k) for k in keys)

    def get_keyword_ci(self, key: str, default: Optional[Any] = None) -> Any:
        """大小写不敏感地从 header 中读取关键字（若 header 为 dict）.

        返回关键字值或提供的 default。便于统一处理 FITS 关键字的大小写差异。
        Read a keyword case-insensitively from a dict or Header. Try direct
        lookup, then compare uppercase names; return default if absent or if
        the header is None. Values are returned without unit/type conversion.
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
        """获取最近的校验报告 / Return the latest cached validation report.

        尚未 validate 时返回 None；访问属性不会触发新的校验。
        Return None before validate() has run; property access does not revalidate.
        """
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
        """校验时间序列头、列与 GTI / Validate time-series metadata and GTI.

        扩展基础报告，检查配置关键字、TIME 列和已提供的 GTI 顺序。
        不换算时间参考系、不补齐 GTI，也不读取新数据；将报告缓存并返回。
        Extend basic validation with configured keywords, TIME column and
        available GTI ordering. No time conversion or GTI reconstruction;
        return/cache the report. Missing noncritical metadata may only warn.
        """
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
        """校验 PHA 元数据与值列 / Validate PHA metadata and value columns.

        要求 CHANNEL 及 COUNTS/RATE 至少一个值列；部分头字段、曝光或
        HDU 分类问题只记 WARN。返回并缓存报告，不核查实际谱值或响应网格。
        Require CHANNEL and at least one COUNTS/RATE column. Some metadata,
        exposure and HDU issues are WARN only. Return/cache the report without
        checking the actual spectral values or response grid.
        """
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
        """校验响应头的基础身份字段 / Validate basic response-header identity.

        检查仪器与 HDU 分类/版本，将缺失项作为 WARN 记录；具体列与
        数值网格须由子类检查。缓存并返回报告，不修改响应数组。
        Check instrument identity and HDU class/version, recording missing
        metadata as WARN. Concrete columns/grids are checked by subclasses.
        Return/cache a report without modifying response arrays.
        """
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

    Check available spectrum CHANNEL bounds against response TLMIN/DETCHANS.
    A range mismatch is ERROR; differing declared channel counts are WARN.
    Missing response bounds produce INFO, and unexpected check failures produce
    WARN. Thus ok=True means no detected ERROR, not that every compatibility
    check was possible. No energy-grid, matrix-normalization or calibration check
    is performed; inputs are left unchanged.
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
