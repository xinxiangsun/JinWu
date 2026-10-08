'''
Date: 2025-05-30 17:43:59
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-11-07 14:35:02
LastEditTime: 2025-09-25 20:34:19
FilePath: /research/jinwu/src/jinwu/core/utils.py
'''
import math
import numpy as np
from typing import Union
import os
import gzip
import shutil
from pathlib import Path


def _require_xspec():
    """延迟检查 PyXspec 可导入 / Require an importable PyXspec module lazily.

    成功返回 None；ModuleNotFoundError 被包装为带环境提示的同类异常。
    不安装软件、不初始化 HEASoft，也不保证谱/模型已经加载。
    Return None on success; wrap ModuleNotFoundError with an environment hint.
    No installation/HEASoft setup occurs, and loaded spectra/models are not verified."""
    try:
        import xspec  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xspec is required for this functionality. Please install HEASOFT/pyxspec and ensure 'xspec' is importable."
        ) from exc


def generate_download_url(isot_time):
    """GBM poshist URL 的兼容入口 / Deprecated compatibility wrapper for a GBM URL.

    isot_time 传给 jinwu.fermi.gbm.generate_download_url，返回其结果。
    调用时发 DeprecationWarning，需安装相应 Fermi 插件；不下载文件。
    Forward isot_time and return the Fermi implementation's result. Emit
    DeprecationWarning; require the instrument plugin. No file is downloaded.

    .. deprecated:: 0.2.0
        使用 / Use ``jinwu.fermi.gbm.generate_download_url``."""
    import warnings

    warnings.warn(
        "jinwu.core.utils.generate_download_url is deprecated since 0.2.0; "
        "use jinwu.fermi.gbm.generate_download_url instead "
        "(install the 'jinwu[fermi]' extra).",
        DeprecationWarning,
        stacklevel=2,
    )
    from jinwu.fermi.gbm import generate_download_url as _gbm_generate_download_url

    return _gbm_generate_download_url(isot_time)


def extract_all_gz_recursive(root_path: Union[str, os.PathLike, Path], 
                             remove_gz: bool = True,
                             verbose: bool = True) -> int:
    """递归解压 gzip 文件 / Recursively extract .gz files under a directory.
    
    root_path 须为存在目录，否则抛 FileNotFoundError/NotADirectoryError。
    输出路径仅移除最后 .gz 后缀，以 wb 写入，已有同名输出会覆盖。
    remove_gz=True（默认）在解压后删除压缩文件；verbose 控制进度打印。
    Require an existing directory, otherwise raise FileNotFoundError or
    NotADirectoryError. Remove the final .gz suffix for output; existing output
    files are overwritten. Default remove_gz=True deletes compressed originals
    after extraction. verbose controls routine progress printing.
    
    返回成功完成处理的文件数。逐文件异常打印后跳过，不回滚已写文件，
    即使 verbose=False 也会打印错误。不会核验解压后科学产品内容。
    Return successfully processed file count. Per-file failures are printed and
    skipped without rollback, even when verbose=False; partially written outputs
    can remain. Scientific product contents are not validated."""
    
    # 统一转换为 pathlib.Path 对象
    root = Path(root_path)
    
    if not root.exists():
        raise FileNotFoundError(f"路径不存在: {root}")
    
    if not root.is_dir():
        raise NotADirectoryError(f"不是目录: {root}")
    
    count = 0
    
    # 递归查找所有 .gz 文件
    for gz_file in root.rglob('*.gz'):
        try:
            # 生成输出文件路径（移除 .gz 后缀）
            output_file = gz_file.with_suffix('')
            
            if verbose:
                print(f"解压: {gz_file} -> {output_file}")
            
            # 解压
            with gzip.open(gz_file, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 删除原 .gz 文件
            if remove_gz:
                gz_file.unlink()
                if verbose:
                    print(f"  已删除: {gz_file}")
            
            count += 1
            
        except Exception as e:
            print(f"❌ 错误处理 {gz_file}: {e}")
            continue
    
    if verbose:
        print(f"\n✅ 总共解压 {count} 个文件")
    
    return count


# 便捷别名
def gunzip(root_path, remove_gz=True, verbose=True):
    """递归解压的别名 / Delegate to extract_all_gz_recursive with identical arguments.

    默认删除压缩原件、可覆盖既有输出；返回成功处理数，行为同主函数。
    Defaults delete .gz originals and may overwrite outputs; return the same count."""
    return extract_all_gz_recursive(root_path, remove_gz, verbose)



# Legacy LF/redshift extrapolator moved out of core utilities.

def get_asym_err(param):
    """读取 XSPEC 参数端点距离 / Return absolute distances to XSPEC error endpoints.

    从 param.error[:2] 减去 param.values[0]，返回 (low_distance, high_distance)。
    单位同参数原生单位；读取失败抛 RuntimeError。当前不检查 XSPEC 的
    误差状态、端点顺序或是否完成 error 扫描，不能据此认证可信区间有效。
    Subtract the parameter value from the first two error endpoints and return
    absolute distances in native units. Access failures raise RuntimeError. Error
    status, endpoint ordering and completed profiling are not checked, so returned
    numbers alone do not certify a valid uncertainty interval."""
    try:
        array = np.array(param.error[:2]) - param.values[0]
        return abs(array[0]), abs(array[1])
    except Exception as e:
        raise RuntimeError(f"Error getting asymmetric error for parameter {param.name}: {e}")


def flux_err_from_log10(lgflux, log_err_low, log_err_high):
    """将 log10 通量误差转为线性误差 / Convert log10-flux distances to linear errors.

    lgflux=log10(F)，low/high 为正向定义的对数距离，计算
    (F-10**(lgflux-low), 10**(lgflux+high)-F)。输出使用输入对数隐含的
    线性通量单位，不携带 Quantity，也不检查误差正值或置信度。
    For lgflux=log10(F) and logarithmic low/high distances, return
    (F-10**(lgflux-low), 10**(lgflux+high)-F) in the implied linear flux unit.
    No Quantity, positivity check or confidence interpretation is supplied.

    任何输入 None 或换算异常返回 (None, None)，不代替缺失值。
    Missing inputs or conversion failures return (None, None)."""
    try:
        if lgflux is None or log_err_low is None or log_err_high is None:
            return None, None
        err_low = 10.0 ** lgflux - 10.0 ** (lgflux - float(log_err_low))
        err_high = 10.0 ** (lgflux + float(log_err_high)) - 10.0 ** lgflux
        return err_low, err_high
    except Exception:
        return None, None


def generate_xspec_result(model, spectrum) -> dict:
    """委托生成 XSPEC 结果字典 / Delegate structured XSPEC result extraction.

    model/spectrum 为已加载的 PyXspec 对象，调用 fit._generate_xspec_result
    并返回 dict，复用其参数与误差状态处理。本兼容入口不传入 flux_range、
    warnings_list 或 errors_computed，因此使用委托函数的默认设置。
    Use loaded PyXspec model/spectrum objects and return the delegated result dict,
    including parameter/error-status handling. This compatibility wrapper does not
    supply flux_range, warnings_list or errors_computed and therefore uses defaults.

    部分提取问题可能作为运行时警告或缺失字段体现；调用者须检查结果
    及诊断。函数不执行拟合，结果字典不证明误差扫描或科学验收已成功。
    Some extraction issues appear as runtime warnings or absent fields. Inspect
    results/diagnostics; this function does not fit or certify completed error scans."""
    from jinwu.core.fit import _generate_xspec_result

    return _generate_xspec_result(model, spectrum)


def _parse_nhtot_response(html, coord_str=""):
    """解析 nhtot 的 ASCII 表响应 / Parse the nhtot HTML-embedded ASCII table.

    html 为响应文本，coord_str 仅用于错误说明；不联网。定位第一段 +===+
    之间以 J/B 坐标开头的数据行，读取坐标及八个平均/加权字段。
    No network I/O. Locate a J/B-prefixed coordinate row inside the first +===+
    table and parse coordinates plus eight mean/weighted fields. coord_str labels errors.

    返回 dict，始终有 ok。表结构失败时 ok=False、error 与 None 字段；
    可解析数据行时 ok=True，但个别无法转换的数值仍为 None。因此 ok
    不等于已验证所有柱密度字段；更严格验证交给 galactic.resolve。
    Return dict with ok; malformed structure gives ok=False/error/None fields.
    A parsed row can have ok=True while individual numeric fields are None. This
    is not full physical validation; galactic.resolve applies stricter checks."""
    import re

    _NONE = {
        'ra': None, 'dec': None,
        'ebv_mean': None, 'ebv_weighted': None,
        'nhi_mean': None, 'nhi_weighted': None,
        'nh2_mean': None, 'nh2_weighted': None,
        'nhtot_mean': None, 'nhtot_weighted': None,
    }

    def _num(s):
        """尝试解析浮点字段 / Parse float fields, returning None for conversion failures.

        当前不筛除 NaN/inf，后续有效性校验由调用方负责。
        Nonfinite values are not rejected here; subsequent validation belongs to callers."""
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    lines = html.split('\n')

    # Locate data row between the first and second +===+ separator rows.
    # The data row is the first line after the opening === that contains
    # both a pipe and a celestial coordinate prefix (J or B).
    data_line = None
    after_header = False
    for line in lines:
        if '+===' in line:
            if not after_header:
                after_header = True   # opening === row, data follows
            else:
                break                 # closing === row, stop
        elif after_header and '|' in line:
            stripped = line.strip()
            # Confirm this looks like a data row: starts with J/B prefix
            # (possibly after an optional leading | from old-style tables)
            if stripped and re.match(r'^\|?\s*[JB]\s', stripped):
                data_line = line
                break

    if data_line is None:
        return {**_NONE, 'ok': False,
                'error': f'No data row found for {coord_str}'}

    # Split on | and drop empty fields (fix: leading/trailing | robustness)
    fields = [f.strip() for f in data_line.split('|') if f.strip()]

    if len(fields) < 9:
        return {**_NONE, 'ok': False,
                'error': f'Expected >=9 fields, got {len(fields)} for {coord_str}'}

    # Parse position from fields[0] (fix: strict prefix removal)
    pos = fields[0]
    ra_str, dec_str = None, None
    if pos:
        pos_clean = pos
        for prefix in ('J ', 'B '):
            if pos_clean.startswith(prefix):
                pos_clean = pos_clean[len(prefix):]
                break
        parts = pos_clean.split(',')
        if len(parts) == 2:
            ra_str, dec_str = parts[0].strip(), parts[1].strip()

    return {
        'ok': True,
        'ra': ra_str,
        'dec': dec_str,
        'ebv_mean': _num(fields[1]),
        'ebv_weighted': _num(fields[2]),
        'nhi_mean': _num(fields[3]),
        'nhi_weighted': _num(fields[4]),
        'nh2_mean': _num(fields[5]),
        'nh2_weighted': _num(fields[6]),
        'nhtot_mean': _num(fields[7]),
        'nhtot_weighted': _num(fields[8]),
    }


def nhtot(ra, dec, equinox=2000, timeout=30.0):
    """查询 Swift/UKSSDC 银河柱密度 / Query the Swift/UKSSDC Galactic NH service.

    ra/dec 可为十进制度或服务支持的 sexagesimal 字符串；equinox 默认
    2000，timeout 为网络超时秒数。POST 到 donhtot.php，再调用纯解析器。
    Accept decimal-degree coordinates or service-supported sexagesimal strings;
    equinox defaults to 2000 and timeout is seconds. POST to donhtot.php, then
    parse the ASCII table response. No local coordinate validation or cache is used.

    返回 ok、坐标、E(B-V) 平均/加权值（mag）、HI/H2/总氢平均/加权柱密度
    （atoms cm^-2）。网络失败返回 ok=False/error/None；表结构解析失败
    还打印消息；ok=True 时个别无法转换字段仍可能为 None。
    Return ok, coordinates, mean/weighted E(B-V) in mag, and HI/H2/total columns
    in atoms cm^-2. Network failures return ok=False/error/None; structural parsing
    failures also print a message. Individual numeric fields may remain None even
    when ok=True. Use resolve_galactic_absorption for validated cached results.

    服务采用 Willingale et al. (2013), MNRAS 431, 394 的口径；实际返回值
    以本次查询为准，不使用示例数值作为固定柱密度。
    The service uses the Willingale et al. (2013) convention; actual values come
    from the current query, not a fixed example column.
    参考 / Reference: https://www.swift.ac.uk/analysis/nhtot/"""
    import urllib.request
    import urllib.parse

    coord_str = f"{ra} {dec}"

    params = urllib.parse.urlencode({
        'Coords': coord_str,
        'equinox': str(equinox),
        'ascii': '1',
        'jsOn': '1',
        'obname': '',
        'MAX_FILE_SIZE': '1000000',
    }).encode('ascii')

    url = "https://www.swift.ac.uk/analysis/nhtot/donhtot.php"

    try:
        req = urllib.request.Request(url, data=params)
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            html = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return {
            'ok': False, 'error': str(e),
            'ra': None, 'dec': None,
            'ebv_mean': None, 'ebv_weighted': None,
            'nhi_mean': None, 'nhi_weighted': None,
            'nh2_mean': None, 'nh2_weighted': None,
            'nhtot_mean': None, 'nhtot_weighted': None,
        }

    result = _parse_nhtot_response(html, coord_str)
    if not result.get('ok'):
        print(f"nhtot: parse failed for {coord_str}: {result.get('error', 'unknown')}")
    return result


class HydroDynamics:
    """经典/相对论流体力学辅助类"""

    @classmethod
    def show_shock_jump_conditions(cls):
        """在 notebook 显示激波方程 / Display stored shock-jump equations in a notebook.

        使用 IPython.display 的 Math/display 产生展示副作用，无参数计算，
        返回 None。不求解流体状态，也不校验某个观测的适用条件。
        Display the stored equations through IPython Math/display and return None.
        No flow-state solution or observational applicability check is performed."""
        from IPython.display import display, Math
        display(Math(r"\text{激波跳变条件（Rankine-Hugoniot conditions）:}"))
        eqs = [
            r"\frac{\rho_2}{\rho_1} = \frac{v_1}{v_2} = \frac{(\hat{\gamma}+1)M_1^2}{(\hat{\gamma}-1)M_1^2+2}",
            r"\frac{p_2}{p_1} = \frac{2\hat{\gamma} M_1^2 - \hat{\gamma} + 1}{\hat{\gamma} + 1}",
            r"\frac{T_2}{T_1} = \frac{p_2 \rho_1}{p_1 \rho_2} = \frac{(2\hat{\gamma} M_1^2 - \hat{\gamma} + 1)[(\hat{\gamma}-1)M_1^2+2]}{(\hat{\gamma}+1)^2 M_1^2}"
        ]
        for eq in eqs:
            display(Math(eq))


class SFH:
    def __init__(self):
        """初始化尚未实现的 SFH 占位类 / Initialize the currently empty SFH placeholder.

        当前只有 pass，不提供星形成历史推断或演化计算。
        The body is pass; no star-formation-history inference or evolution calculation."""
        pass


# NOTE for future AI / maintainers:
#   RedshiftExtrapolator was moved to jinwu.lf.legacy_redshift.
#   Do NOT re-add a top-level import of anything under jinwu.lf here —
#   it creates a circular import (core.utils -> lf -> detectability -> core.utils).
#   Users who need the legacy class should import it directly:
#       from jinwu.lf.legacy_redshift import RedshiftExtrapolator

# ======================================================================
# Li & Ma SNR 和触发判断工具
# ======================================================================

from .significance import li_ma_snr, snr  # noqa: F401


from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional, Tuple as _Tuple, Literal as _Literal, Union as _Union


@_dataclass
class BackgroundSimple:
    """Minimal background configuration for Li & Ma significance.

    Parameters
    ----------
    area_ratio : float
        A_on / A_off.
    t_off_ref : float
        Reference OFF exposure (seconds).
    n_off_ref : float
        Total OFF counts corresponding to t_off_ref.
    """

    area_ratio: float
    t_off_ref: float
    n_off_ref: float

    def alpha(self, t_on: float) -> float:
        """计算 ON/OFF 比例 / Compute the dimensionless ON/OFF exposure-area ratio.

        alpha=area_ratio*t_on/t_off_ref，时间使用秒；返回 float。背景配置与
        t_off_ref>0 由调用者保证，当前不校验，零参考曝光可抛除零错误。
        Return float area_ratio*t_on/t_off_ref with time in seconds. The caller must
        ensure valid positive reference exposure; no validation here, so zero may divide."""
        return float(self.area_ratio) * (float(t_on) / float(self.t_off_ref))


class TriggerDecider:
    """Decide triggerability from a binned counts lightcurve or event times.

    Core checks
    -----------
    - sliding_window(window=1200): scan max Li&Ma SNR over all windows.
    - head_window(window=1200): Li&Ma SNR of the first window only.
    - cumulative_from_t0(target=7): grow cumulatively from T0.

    Inputs
    ------
    time : 1D array of bin left edges (monotonic increasing).
    counts : 1D array of ON-region counts per bin (non-negative).
    dt : float bin width in seconds (assumed constant).
    bg : BackgroundSimple with (n_off_ref, t_off_ref, area_ratio).
    """

    def __init__(
        self,
        time: np.ndarray,
        counts: np.ndarray,
        dt: float,
        bg: BackgroundSimple,
    ) -> None:
        """初始化固定宽度计数触发判定器 / Initialize a fixed-width counts trigger evaluator.

        time 为升序 bin 左边缘秒数，counts 为 ON 总计数，dt 为正秒宽，bg
        给定参考 OFF 计数/曝光及面积比。检查一维等长与 dt>0，存数组及累计
        和；不核验时间等距、有限性、计数非负或背景配置，调用者须保证。
        Require sorted bin left edges in seconds, total ON counts, positive dt in
        seconds, and an OFF reference background. Check 1-D equal lengths and dt>0,
        store arrays/cumulative counts; callers ensure finite, regular, nonnegative
        scientific inputs and valid background. Arrays are not guaranteed to be copied."""
        time = np.asarray(time, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if time.ndim != 1 or counts.ndim != 1:
            raise ValueError("time and counts must be 1D arrays")
        if time.size != counts.size:
            raise ValueError("time and counts must have the same length")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time = time
        self.counts = counts
        self.dt = float(dt)
        self.bg = bg
        self._cum = np.cumsum(self.counts)

    @classmethod
    def from_counts(
        cls,
        time: np.ndarray,
        counts: np.ndarray,
        dt: _Optional[float],
        bg: BackgroundSimple,
    ) -> "TriggerDecider":
        """从分箱计数创建判定器 / Construct the evaluator from binned counts.

        dt=None 时用相邻 time 差的中位数推断，至少需两个时间点。
        不检验严格等距或时间系统，进一步校验交给构造器。
        If dt is None, infer median time difference, requiring two points. No full
        regularity/time-system check; constructor applies its shape/dt validation."""
        time = np.asarray(time, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if dt is None:
            if time.size < 2:
                raise ValueError("Need dt or at least two time points to infer dt")
            dt = float(np.median(np.diff(time)))
        return cls(time=time, counts=counts, dt=dt, bg=bg)

    @classmethod
    def from_events(
        cls,
        events: np.ndarray,
        *,
        dt: float,
        bg: BackgroundSimple,
        t_start: _Optional[float] = None,
        t_end: _Optional[float] = None,
    ) -> "TriggerDecider":
        """将事件直方图化后创建判定器 / Bin event times and construct the evaluator.

        events 为非空一维秒时间，dt>0；默认 t_start=min(events)，
        t_end=max(events)+dt，目标边界扩展到 ceil((end-start)/dt) 个 bin。
        范围外事件由 np.histogram 忽略；不处理 GTI、活时间或原生事件死时间。
        Use nonempty 1-D event times in seconds and positive dt. Default range starts
        at min and ends at max+dt; ceil builds uniform bins. Histogram ignores events
        outside the selected range. No GTI/livetime/native-event-deadtime correction.
        Return a TriggerDecider; invalid shape/empty events/nonpositive dt raises ValueError."""
        events = np.asarray(events, dtype=float)
        if events.ndim != 1:
            raise ValueError("events must be 1D array of times")
        if events.size == 0:
            raise ValueError("events is empty")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if t_start is None:
            t_start = float(np.min(events))
        if t_end is None:
            t_end = float(np.max(events)) + float(dt)
        nbins = int(np.ceil((t_end - t_start) / float(dt)))
        edges = t_start + np.arange(nbins + 1, dtype=float) * float(dt)
        counts, _ = np.histogram(events, bins=edges)
        time = edges[:-1]
        return cls(time=time, counts=counts.astype(float), dt=float(dt), bg=bg)

    def _counts_in(self, left: float, right: float) -> float:
        """按 bin 左边缘计数 / Sum bins whose left edges lie in [left, right).

        searchsorted 要求 time 升序；不按边缘部分交叠拆分计数。空区间返回 0。
        Use sorted-time searchsorted and cumulative differences, without fractional
        edge-bin allocation. Empty selection returns 0. Bounds use the time reference."""
        i0 = int(np.searchsorted(self.time, left, side="left"))
        i1 = int(np.searchsorted(self.time, right, side="left"))
        if i1 <= i0:
            return 0.0
        return float(self._cum[i1 - 1] - (self._cum[i0 - 1] if i0 > 0 else 0.0))

    def _snr_window(
        self, left: float, right: float, n_off_ref: _Optional[float] = None,
    ) -> float:
        """计算窗口的局部 Li-Ma 值 / Compute a window's local signed Li-Ma significance.

        ON 计数按左边缘选取，alpha 按 right-left 的完整窗口时长计算；可指定
        参考 OFF 计数，否则用 bg.n_off_ref。非正窗口返回 0。窗口超出数据
        覆盖时不自动缩短曝光；不做多窗口搜索校准。
        Count ON bins by left edges and scale alpha using full right-left duration.
        Use supplied/default reference OFF counts. Nonpositive duration returns 0;
        windows beyond coverage do not trim exposure. No search-trials calibration."""
        n_on = self._counts_in(left, right)
        t_on = max(0.0, float(right - left))
        if t_on <= 0:
            return 0.0
        alpha = self.bg.alpha(t_on)
        n_off = float(self.bg.n_off_ref if n_off_ref is None else n_off_ref)
        return li_ma_snr(n_on=n_on, n_off=n_off, alpha=alpha)

    def sliding_window(
        self, *, window: float = 1200.0, step: _Optional[float] = None,
        target: float = 7.0,
    ) -> _Tuple[bool, dict]:
        """扫描滑动窗口的最大局部显著性 / Scan the maximum local sliding-window significance.

        window 与 step 为正秒数，step 默认 dt；target 为无量纲 SNR 门限。
        返回 (max_snr>=target, {'max_snr', 'best_window'})；max_snr 从 0 开始，
        负值不成为最优窗口。时间覆盖短于 window 时仍检查一个完整宽度窗口。
        Use positive window/step seconds, default step=dt, and dimensionless target.
        Return threshold decision and max_snr/best_window. Initialize max at 0 so
        negative deficits do not win. Short coverage still tests one full-width window.
        Invalid window/step raises ValueError; empty data is not supported. The scanned
        maximum is not calibrated for the number/dependence of searched windows."""
        if window <= 0:
            raise ValueError("window must be positive")
        if step is None:
            step = self.dt
        step = float(step)
        if step <= 0:
            raise ValueError("step must be positive")
        t0 = float(self.time[0])
        tN = float(self.time[0] + self.counts.size * self.dt)
        starts = np.arange(t0, max(t0, tN - window) + 1e-12, step, dtype=float)
        max_snr = 0.0
        best = (t0, t0 + window)
        for s in starts:
            snr = self._snr_window(s, s + window)
            if snr > max_snr:
                max_snr = snr
                best = (s, s + window)
        return bool(max_snr >= target), {"max_snr": max_snr, "best_window": best}

    def head_window(
        self, *, window: float = 1200.0, target: float = 7.0,
    ) -> _Tuple[bool, dict]:
        """检查首个固定窗口 / Evaluate the first window from the first bin left edge.

        返回 (达到 target 的布尔值, {'snr', 'window'})。window 单位秒，当前
        不在此显式校验正值；超出数据的时长仍用于背景曝光比例。
        Return decision plus snr/window. window is seconds and is not explicitly
        validated here; duration beyond data coverage still scales background exposure."""
        left = float(self.time[0])
        right = left + float(window)
        snr = self._snr_window(left, right)
        return bool(snr >= target), {"snr": snr, "window": (left, right)}

    def _find_t0(
        self, mode: _Literal["first_nonzero", "first_time"] = "first_nonzero",
    ) -> float:
        """选首个时间或正计数 bin / Choose the first time or first positive-count bin.

        mode='first_time' 取 time[0]；其他值都走首个 counts>0，若全零取首时间。
        不是由物理触发或统计变点测量的时刻，空数组不支持。
        'first_time' uses time[0]; all other modes use first counts>0, falling back
        to the first time. This is not a measured physical/change-point onset."""
        if mode == "first_time":
            return float(self.time[0])
        idx = int(np.argmax(self.counts > 0)) if np.any(self.counts > 0) else 0
        return float(self.time[idx])

    def cumulative_from_t0(
        self,
        *,
        target: float = 7.0,
        t0_mode: _Literal["first_nonzero", "first_time"] = "first_nonzero",
        max_window: _Optional[float] = 1200,
    ) -> _Tuple[bool, dict]:
        """从选定 T0 累积到首次越阈 / Accumulate from T0 until the first threshold crossing.

        选择 _find_t0 起点，以 k*dt 作为曝光；max_window 限制搜索结束秒数，
        None 则到当前数据终点。返回 (hit, {'T0', 't_reach', 'max_snr'})，
        未达到时 t_reach=None。成功即停止，max_snr 只覆盖停止前已访问的长度。
        Start at _find_t0, using k*dt exposure and optional max_window seconds.
        Return hit and T0/t_reach/max_snr; t_reach=None if no hit. Stop at the first
        crossing, so max_snr covers only visited cumulative lengths, not the whole curve.
        No GTI/deadtime or repeated-look significance calibration is performed."""
        T0 = self._find_t0(mode=t0_mode)
        t_end = float(self.time[0] + self.counts.size * self.dt)
        if max_window is not None:
            t_end = min(t_end, T0 + float(max_window))
        i0 = int(np.searchsorted(self.time, T0, side="left"))
        i1 = int(np.searchsorted(self.time, t_end, side="left"))
        if i1 <= i0:
            return False, {"T0": T0, "t_reach": None, "max_snr": 0.0}
        csum = np.cumsum(self.counts[i0:i1])
        max_snr = 0.0
        t_reach: _Optional[float] = None
        for k in range(1, csum.size + 1):
            t_on = k * self.dt
            alpha = self.bg.alpha(t_on)
            snr = li_ma_snr(n_on=float(csum[k - 1]), n_off=float(self.bg.n_off_ref), alpha=alpha)
            if snr > max_snr:
                max_snr = snr
            if snr >= float(target) and t_reach is None:
                t_reach = T0 + t_on
                break
        return bool(t_reach is not None), {"T0": T0, "t_reach": t_reach, "max_snr": max_snr}

    def decide(
        self,
        *,
        window: float = 1200.0,
        target: float = 7.0,
        step: _Optional[float] = None,
        t0_mode: _Literal["first_nonzero", "first_time"] = "first_nonzero",
    ) -> dict:
        """按优先级尝试三种触发判据 / Try sliding, head, then cumulative criteria.

        先返回成功的 sliding/head；两者未达到才做不受 window 限制的 cumulative
        （max_window=None）。返回带 triggered、method 和该方法诊断的 dict。
        只评估当前局部 Li-Ma 门限，不等同于校准过的真实仪器触发效率。
        Return the first successful sliding/head result, otherwise an unbounded
        cumulative scan (max_window=None). Return triggered/method and that method's
        diagnostics. Local threshold checks do not establish calibrated instrument efficiency."""
        slid_ok, slid_stat = self.sliding_window(window=window, step=step, target=target)
        if slid_ok:
            return {"triggered": True, "method": "sliding", **slid_stat}
        head_ok, head_stat = self.head_window(window=window, target=target)
        if head_ok:
            return {"triggered": True, "method": "head", **head_stat}
        cum_ok, cum_stat = self.cumulative_from_t0(target=target, t0_mode=t0_mode, max_window=None)
        return {"triggered": bool(cum_ok), "method": "cumulative", **cum_stat}


class LightcurveSNREvaluator:
    """Evaluate whether a binned lightcurve can reach a target SNR after T0.

    T0 is detected via Bayesian Blocks with per-block Li & Ma SNR ≥ 3.
    Supports a fast expected-value mode and an MC mode with Poisson
    fluctuations for ON and OFF counts.

    Typical usage
    -------------
    >>> bg = BackgroundPrior(n_off_prior=1200, t_off=100000.0, area_ratio=1/12)
    >>> ev = LightcurveSNREvaluator.from_counts(
    ...     time=np.arange(0, 2000.0, 0.5),
    ...     counts=np.random.poisson(0.1, 4000),
    ...     dt=0.5,
    ...     background=bg,
    ... )
    >>> ok, stats = ev.reaches_snr(target=7.0, window=1200.0, mode="fast")
    """

    def __init__(
        self,
        time: np.ndarray,
        counts: np.ndarray,
        dt: float,
        background: _Union["_BackgroundPrior", "_BackgroundCountsPosterior"],
        off_exposure_ref: _Optional[float] = None,
    ) -> None:
        """保存光变与背景预测配置 / Initialize a lightcurve/background SNR predictor.

        time/counts 须为同长一维数组，dt 为正秒宽；time 是升序等宽 bin
        左边缘，counts 应为 ON 总计数。检查形状/dt，保存累计计数；不全面
        校验有限性、非负或均匀性。背景为 BackgroundPrior 或 CountsPosterior。
        Supply equal-length 1-D arrays, sorted regular-bin left edges in seconds, total
        ON counts and positive dt. Check shape/dt and store cumulative counts without
        full finiteness/nonnegativity/regularity validation. Background is a prior or posterior.

        后验使用显式 off_exposure_ref 或 1e6 秒；先验使用其 t_off，忽略传入
        的 off_exposure_ref。此参考曝光是统计比例约定，不自动读取观测 GTI。
        A posterior uses supplied off_exposure_ref or 1e6 s; a prior uses its t_off,
        ignoring that argument. Reference exposure is a statistical convention, not an
        observation-GTI/livetime measurement."""
        from jinwu.background.backprior import (
            BackgroundPrior as _BackgroundPrior,
            BackgroundCountsPosterior as _BackgroundCountsPosterior,
        )

        if time.ndim != 1 or counts.ndim != 1:
            raise ValueError("time and counts must be 1D arrays")
        if time.size != counts.size:
            raise ValueError("time and counts must have the same length")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time = np.asarray(time, dtype=float)
        self.counts = np.asarray(counts, dtype=float)
        self.dt = float(dt)
        self._bg_prior: _Optional[_BackgroundPrior]
        self._bg_post: _Optional[_BackgroundCountsPosterior]
        if isinstance(background, _BackgroundCountsPosterior):
            self._bg_prior = None
            self._bg_post = background
            self.area_ratio = float(background.area_ratio)
            self.off_exposure_ref = float(off_exposure_ref) if off_exposure_ref is not None else 1_000_000.0
        else:
            self._bg_prior = background  # type: ignore[assignment]
            self._bg_post = None
            self.area_ratio = float(background.area_ratio)
            self.off_exposure_ref = float(getattr(background, "t_off", 1_000_000.0))
        self._cum_counts = np.cumsum(self.counts)

    @classmethod
    def from_counts(
        cls,
        time: np.ndarray,
        counts: np.ndarray,
        dt: _Optional[float] = None,
        background: _Optional[_Union["_BackgroundPrior", "_BackgroundCountsPosterior"]] = None,
        off_exposure_ref: _Optional[float] = None,
    ) -> "LightcurveSNREvaluator":
        """从 ON 计数构造预测器 / Construct a predictor from total ON bin counts.

        background 必须提供；dt=None 用 median(diff(time)) 推断，至少两个点。
        返回实例，数据有效性继续遵循构造器约定，不自动扣背景或求能段。
        Require background. Infer missing dt from median time difference (two points
        minimum). Return an instance without background subtraction or band selection."""
        time = np.asarray(time, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if dt is None:
            if time.size < 2:
                raise ValueError("Need dt or at least two time points to infer dt")
            dt = float(np.median(np.diff(time)))
        if background is None:
            raise ValueError("background must be provided")
        return cls(time=time, counts=counts, dt=dt, background=background, off_exposure_ref=off_exposure_ref)

    @classmethod
    def from_npz(
        cls,
        npz_path: str,
        background: _Union["_BackgroundPrior", "_BackgroundCountsPosterior"],
        *,
        time_key_primary: str = "time_series",
        time_key_fallback: str = "raw_time_series",
        counts_key_preferred: str = "corrected_counts_src",
        net_key: str = "corrected_counts",
        off_key: str = "corrected_counts_back",
        raw_counts_key_fallback: str = "raw_corrected_counts",
        dt: _Optional[float] = None,
        off_exposure_ref: _Optional[float] = None,
        verbose: bool = True,
    ) -> "LightcurveSNREvaluator":
        """按字段优先级加载 NPZ / Load a predictor using ordered NPZ field fallbacks.

        时间优先 time_key_primary 再 fallback。计数优先 counts_key_preferred，
        否则用 net_key + area_ratio*off_key 恢复 ON，最后取 raw_counts_key_fallback。
        调用者必须确认后备字段的实际计数/曝光标度；字段名字不证明它是原始 ON。
        Choose primary/fallback time, preferred ON counts, then net+area_ratio*OFF,
        then raw fallback. Callers establish actual count/exposure scaling; a field name
        alone does not prove raw ON semantics. No additional exposure ratio is applied
        to the net+area_ratio*OFF reconstruction.

        dt 缺失按时间差中位数推断，verbose 打印选择；缺少必要字段抛 ValueError。
        读取本地文件，不联网；返回 from_counts 构造的实例。
        Infer missing dt by median difference; verbose logs selections. Missing fields
        raise ValueError. Read local data only and return the from_counts instance."""
        data = np.load(npz_path)
        if time_key_primary in data:
            time = np.asarray(data[time_key_primary], dtype=float)
            src_time_key = time_key_primary
        elif time_key_fallback in data:
            time = np.asarray(data[time_key_fallback], dtype=float)
            src_time_key = time_key_fallback
        else:
            raise ValueError(
                f"Cannot find time array in NPZ. "
                f"Tried '{time_key_primary}' and '{time_key_fallback}'."
            )
        counts = None
        used = None
        if counts_key_preferred in data:
            counts = np.asarray(data[counts_key_preferred], dtype=float)
            used = counts_key_preferred
        elif (net_key in data) and (off_key in data):
            net = np.asarray(data[net_key], dtype=float)
            off = np.asarray(data[off_key], dtype=float)
            counts = net + float(background.area_ratio) * off
            used = f"{net_key} + area_ratio*{off_key}"
        elif raw_counts_key_fallback in data:
            counts = np.asarray(data[raw_counts_key_fallback], dtype=float)
            used = raw_counts_key_fallback
            if verbose:
                print(
                    f"[LightcurveSNREvaluator] Using '{raw_counts_key_fallback}' as ON counts.\n"
                    "If this is actually net counts, SNR will be conservative."
                )
        else:
            raise ValueError(
                "Cannot determine ON-region counts from NPZ. Provide one of: "
                f"'{counts_key_preferred}', or both '{net_key}' & '{off_key}', "
                f"or '{raw_counts_key_fallback}'."
            )
        if dt is None:
            if time.size < 2:
                raise ValueError("Need dt or at least two time samples to infer dt")
            dt = float(np.median(np.diff(time)))
        if verbose:
            print(
                f"[LightcurveSNREvaluator] Loaded time='{src_time_key}', "
                f"counts='{used}', dt={dt:.6g}s"
            )
        return cls.from_counts(time=time, counts=counts, dt=dt, background=background, off_exposure_ref=off_exposure_ref)

    def _block_snr(self, left: float, right: float, n_off: float) -> float:
        """计算块内局部 Li-Ma / Compute local Li-Ma from left-edge-selected block counts.

        取 [left,right) 的 ON bin，曝光 right-left，n_off 为参考 OFF 计数；
        空选择返回 0。边缘 bin 不分数拆分，时长不做 GTI/死时间校正。
        Use ON bins in [left,right), duration right-left and reference n_off. Empty
        selection gives 0. No fractional edges or GTI/deadtime correction."""
        i0 = int(np.searchsorted(self.time, left, side="left"))
        i1 = int(np.searchsorted(self.time, right, side="left"))
        if i1 <= i0:
            return 0.0
        n_on = float(self._cum_counts[i1 - 1] - (self._cum_counts[i0 - 1] if i0 > 0 else 0.0))
        t_on = right - left
        alpha = self._alpha(t_on)
        return li_ma_snr(n_on=n_on, n_off=n_off, alpha=alpha)

    def _alpha(self, t_on: float) -> float:
        """计算参考曝光比例 / Return area_ratio*t_on/off_exposure_ref.

        时间均为秒，输出无量纲；不自行校验参考曝光。
        Times are seconds; output is dimensionless, without reference-exposure validation."""
        return float(self.area_ratio) * (float(t_on) / float(self.off_exposure_ref))

    def _find_T0_by_blocks(
        self,
        snr_thr: float = 3.0,
        n_off: _Optional[float] = None,
        rng: _Optional[np.random.Generator] = None,
        off_mode: _Literal["fixed", "poisson"] = "fixed",
    ) -> float:
        """用分块和局部门限选 T0 / Select T0 using blocks and a local significance threshold.

        对原始 self.time/counts 调用 Astropy Bayesian Blocks fitness='measures'
        （高斯测量 fitness，不是 Poisson events fitness），未显式提供 sigma。
        顺序返回首个 Li-Ma>=snr_thr 的块左边缘；未找到则回退首个时间。
        Apply Astropy Bayesian Blocks with Gaussian 'measures' fitness to the original
        stored curve, not Poisson 'events', without explicit sigma. Return the first
        block left edge meeting local snr_thr, otherwise the first stored time.

        n_off 未指定时，fixed 取背景期望，否则按背景先验/后验抽取 OFF。
        可传 rng 控制随机性。此选择不代表已校准的物理起始时刻。
        If n_off is absent, fixed uses expectation; other modes sample OFF from the
        background prescription, using rng when supplied. This is a selection heuristic,
        not a calibrated physical onset measurement.
        参考 / Reference: https://docs.astropy.org/en/stable/api/astropy.stats.bayesian_blocks.html"""
        from astropy.stats import bayesian_blocks
        from jinwu.background.backprior import (
            BackgroundPrior as _BackgroundPrior,
            BackgroundCountsPosterior as _BackgroundCountsPosterior,
        )

        if n_off is None:
            if self._bg_post is not None:
                if off_mode == "fixed":
                    n_off = float(self._bg_post.expected_off(self.off_exposure_ref))
                else:
                    rng = rng or np.random.default_rng()
                    lam_off = rng.gamma(shape=float(self._bg_post.a_total), scale=1.0 / float(self._bg_post.b))
                    n_off = float(rng.poisson(lam_off * float(self.off_exposure_ref)))
            else:
                prior = self._bg_prior
                if off_mode == "fixed":
                    n_off = float(prior.n_off_prior)  # type: ignore[union-attr]
                else:
                    rng = rng or np.random.default_rng()
                    mu_off = float(prior.n_off_prior) / float(prior.t_off)  # type: ignore[union-attr]
                    n_off = float(rng.poisson(mu_off * prior.t_off))  # type: ignore[union-attr]

        edges = bayesian_blocks(self.time, self.counts, fitness="measures")
        for i in range(len(edges) - 1):
            left, right = float(edges[i]), float(edges[i + 1])
            snr = self._block_snr(left, right, n_off=n_off)
            if snr >= snr_thr:
                return left
        return float(self.time[0])

    def reaches_snr(
        self,
        target: float = 7.0,
        window: float = 1200.0,
        mode: _Literal["fast", "mc"] = "mc",
        n_mc: int = 500,
        rng: _Optional[np.random.Generator] = None,
        t0_snr_thr: float = 3.0,
        off_mode: _Literal["fixed", "poisson"] = "poisson",
    ) -> _Tuple[bool, dict]:
        """预测累计局部 SNR 越阈概率 / Predict cumulative local-SNR threshold crossings.

        Parameters
        ----------
        target : float
            无量纲 Li-Ma 门限，默认 7 / Dimensionless Li-Ma threshold, default 7.
        window : float
            T0 后搜索时长，秒；选择范围须在调用方检查。
            Search duration after T0 in seconds; caller checks the desired range.
        mode : {'fast', 'mc'}
            fast 用固定 OFF 期望和观测 ON 累计计数；其他值当前都走 MC 分支。
            Fast uses expected OFF and observed ON; any other value currently enters MC.
        n_mc : int
            模拟次数，调用者须保证为正；不在入口显式校验。
            Trial count; caller ensures a positive value, not explicitly validated here.
        rng : numpy.random.Generator or None
            随机生成器；None 创建 default_rng，重现结果需显式固定。
            Generator; None creates default_rng. Supply a seeded generator for reproducibility.
        t0_snr_thr : float
            在原始光变分块中选 T0 的局部门限 / Local original-block threshold for T0.
        off_mode : {'fixed', 'poisson'}
            fixed 使用参考 OFF 期望，其余值按代码的背景抽样分支生成 OFF。
            Fixed uses expected OFF; other values sample OFF under the background prescription.

        Returns
        -------
        tuple
            fast 返回 (max_snr>=target, {'T0','max_snr'})。MC 返回
            (hits/n_mc>=0.95, {'prob','max_snrs'})；0.95 是固定的命中概率门槛。
            Fast returns a threshold decision and T0/max_snr. MC returns whether the
            hit fraction is >=0.95, plus prob/max_snrs; 0.95 is a fixed decision cutoff.

        Notes
        -----
        MC 每次以原始光变重新选择 T0，再在选定范围生成 Poisson ON 计数；
        后验背景情况下，OFF 与 ON 背景率分别抽取 Gamma 值，非共享一次抽样。
        这里只传播当前模拟方案的波动，不建立仪器响应、GTI、死时间或搜索
        误警率校准。prob 是该方案的命中比例，不是显著性的 p 值或置信区间。
        Each MC trial reselects T0 from the original curve then samples ON counts in
        that range. Posterior background rates for OFF and ON are drawn separately,
        not shared. No response/GTI/deadtime/search-FAR calibration is included. prob
        is the simulated hit fraction, not a detection p-value or confidence interval."""
        from jinwu.background.backprior import (
            BackgroundPrior as _BackgroundPrior,
            BackgroundCountsPosterior as _BackgroundCountsPosterior,
        )

        rng = rng or np.random.default_rng()
        if mode == "fast":
            if self._bg_post is not None:
                n_off_exp = float(self._bg_post.expected_off(self.off_exposure_ref))
            else:
                n_off_exp = float(self._bg_prior.n_off_prior)  # type: ignore[union-attr]
            T0 = self._find_T0_by_blocks(snr_thr=t0_snr_thr, n_off=n_off_exp, off_mode="fixed")
            t_start = T0
            t_end = T0 + float(window)
            i0 = int(np.searchsorted(self.time, t_start, side="left"))
            i1 = int(np.searchsorted(self.time, t_end, side="left"))
            if i1 <= i0:
                return False, {"T0": T0, "max_snr": 0.0}
            counts_win = self.counts[i0:i1]
            csum = np.cumsum(counts_win)
            max_snr = 0.0
            for k in range(1, csum.size + 1):
                t_on = k * self.dt
                alpha = self._alpha(t_on)
                n_on = float(csum[k - 1])
                snr = li_ma_snr(n_on=n_on, n_off=float(n_off_exp), alpha=alpha)
                if snr > max_snr:
                    max_snr = snr
            ok = bool(max_snr >= target)
            return ok, {"T0": T0, "max_snr": max_snr}

        # MC mode
        hits = 0
        max_snrs = []
        for _ in range(int(n_mc)):
            if self._bg_post is not None:
                if off_mode == "fixed":
                    n_off = float(self._bg_post.expected_off(self.off_exposure_ref))
                else:
                    lam_off = rng.gamma(shape=float(self._bg_post.a_total), scale=1.0 / float(self._bg_post.b))
                    n_off = float(rng.poisson(lam_off * float(self.off_exposure_ref)))
            else:
                if off_mode == "fixed":
                    n_off = float(self._bg_prior.n_off_prior)  # type: ignore[union-attr]
                else:
                    mu_off = float(self._bg_prior.n_off_prior) / float(self._bg_prior.t_off)  # type: ignore[union-attr]
                    n_off = float(rng.poisson(mu_off * self._bg_prior.t_off))  # type: ignore[union-attr]

            T0 = self._find_T0_by_blocks(snr_thr=t0_snr_thr, n_off=n_off, rng=rng, off_mode=off_mode)
            t_start = T0
            t_end = T0 + float(window)
            i0 = int(np.searchsorted(self.time, t_start, side="left"))
            i1 = int(np.searchsorted(self.time, t_end, side="left"))
            if i1 <= i0:
                max_snrs.append(0.0)
                continue
            bins = slice(i0, i1)
            if self._bg_post is not None:
                lam_off = rng.gamma(shape=float(self._bg_post.a_total), scale=1.0 / float(self._bg_post.b))
                mu_bkg_bin = float(lam_off) * float(self.area_ratio) * float(self.dt)
                lam_on_obs = np.clip(self.counts[bins], 0.0, None)
                mu_src_bin = np.clip(lam_on_obs - mu_bkg_bin, 0.0, None)
                n_src_bins = rng.poisson(mu_src_bin)
                n_bkg_bins = rng.poisson(mu_bkg_bin, size=n_src_bins.size)
                n_on_bins = n_src_bins + n_bkg_bins
            else:
                lam_on = np.clip(self.counts[bins], 0.0, None)
                n_on_bins = rng.poisson(lam_on)
            csum = np.cumsum(n_on_bins)
            max_snr = 0.0
            for k in range(1, csum.size + 1):
                t_on = k * self.dt
                alpha = self._alpha(t_on)
                snr = li_ma_snr(n_on=float(csum[k - 1]), n_off=n_off, alpha=alpha)
                if snr > max_snr:
                    max_snr = snr
            max_snrs.append(float(max_snr))
            hits += int(max_snr >= target)

        prob = hits / float(n_mc)
        return bool(prob >= 0.95), {"prob": prob, "max_snrs": np.asarray(max_snrs)}
