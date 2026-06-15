'''
Date: 2025-05-30 17:43:59
LastEditors: Xinxiang Sun sunxx@nao.cas.cn
LastEditTime: 2025-11-07 14:35:02
LastEditTime: 2025-09-25 20:34:19
FilePath: /research/jinwu/src/jinwu/core/utils.py
'''
import numpy as np
from typing import Union
import os
import gzip
import shutil
from pathlib import Path


def _require_xspec():
    try:
        import xspec  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xspec is required for this functionality. Please install HEASOFT/pyxspec and ensure 'xspec' is importable."
        ) from exc


def generate_download_url(isot_time):
    """
    根据给定的 isot (YYYY-MM-DDTHH:MM:SS) 时间生成 GBM poshist 文件的下载 URL。

    参数:
    - isot_time (str): ISOT 格式时间字符串，例如 "2024-01-01T12:00:00"

    返回:
    - url (str): 生成的 poshist 文件下载 URL
    """
    # 解析时间

    # 提取年份、月份、日期
    year = isot_time.strftime('%y')
    yr2 = isot_time.datetime.year
    month = f"{isot_time.datetime.month:02d}"  # 两位数格式
    day = f"{isot_time.datetime.day:02d}"

    # 生成文件名
    filename = f"glg_poshist_all_{year}{month}{day}_v00.fit"

    # 生成完整的下载路径
    # https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/2025/01/01/current/
    # url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{yr2}/{isot_time.strftime('%m/%d/')}current/{filename}"
    url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{yr2}/{isot_time.strftime('%m/%d/')}current"
    return url


def extract_all_gz_recursive(root_path: Union[str, os.PathLike, Path], 
                             remove_gz: bool = True,
                             verbose: bool = True) -> int:
    """
    递归解压文件夹下所有 .gz 文件。
    
    参数：
    - root_path: 根目录路径（支持 str、pathlib.Path、os.PathLike）
    - remove_gz: 解压后是否删除原 .gz 文件（默认 True）
    - verbose: 是否打印解压日志（默认 True）
    
    返回：
    - 解压的文件数量
    
    示例：
    >>> extract_all_gz_recursive('/path/to/data')
    >>> extract_all_gz_recursive(Path.home() / 'data')
    >>> extract_all_gz_recursive('C:/data', remove_gz=False)
    """
    
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
    """gunzip 的别名，用法相同"""
    return extract_all_gz_recursive(root_path, remove_gz, verbose)



# Legacy LF/redshift extrapolator moved out of core utilities.

def get_asym_err(param):
    """获取XSPEC参数的非对称误差"""
    try:
        array = np.array(param.error[:2]) - param.values[0]
        return abs(array[0]), abs(array[1])
    except Exception as e:
        raise RuntimeError(f"Error getting asymmetric error for parameter {param.name}: {e}")


def flux_err_from_log10(lgflux, log_err_low, log_err_high):
    """从对数通量误差计算线性通量误差"""
    try:
        if lgflux is None or log_err_low is None or log_err_high is None:
            return None, None
        err_low = 10.0 ** lgflux - 10.0 ** (lgflux - float(log_err_low))
        err_high = 10.0 ** (lgflux + float(log_err_high)) - 10.0 ** lgflux
        return err_low, err_high
    except Exception:
        return None, None


def generate_xspec_result(model, spectrum) -> dict:
    """
    根据XSPEC模型和光谱自动生成结果字典
    
    参数:
        model: XSPEC模型对象
        spectrum: XSPEC光谱对象
        
    返回:
        包含模型参数、flux、rate等信息的字典
    """
    _require_xspec()
    import xspec
    
    lines = []
    result = {}
    result['model'] = model.expression
    
    result['parameters'] = {}
    lines.append(f"Model: {model.expression}")
    
    processed_params = set()
    
    for comp_name in model.componentNames:
        try:
            comp = getattr(model, comp_name)
            
            for param_name in comp.parameterNames:
                param_key = f"{comp_name}.{param_name}"
                if param_key in processed_params:
                    continue
                processed_params.add(param_key)
                
                param = getattr(comp, param_name)
                param_val = param.values[0]
                
                param_dict = {
                    'value': param_val,
                    'frozen': param.frozen
                }
                
                if not param.frozen:
                    err_lo, err_hi = get_asym_err(param)
                    param_dict['error_lo'] = err_lo
                    param_dict['error_hi'] = err_hi
                    lines.append(f"{comp_name}.{param_name}: {param_val:.4f} (-{err_lo:.4f}, +{err_hi:.4f})(1sigma error)")
                else:
                    lines.append(f"{comp_name}.{param_name}: {param_val:.4f} (fixed)")
                
                result['parameters'][param_key] = param_dict
                    
        except Exception as e:
            raise RuntimeError(f"Error processing component {comp_name}: {e}")
    
    emin = model.cflux.Emin.values[0] if hasattr(model, 'cflux') else None
    emax = model.cflux.Emax.values[0] if hasattr(model, 'cflux') else None
    xspec.AllModels.calcFlux(f"{emin} {emax}")
    flux_erg = float(spectrum.flux[0])
    flux_photons = float(spectrum.flux[3])
    
    result['flux_abs'] = {
        'erg_cm2_s': flux_erg,
        'photons_cm2_s': flux_photons
    }
    
    lines.append(f"Absorbed Flux ({emin:.1f}-{emax:.1f} keV): {flux_erg:.4e} erg/cm²/s")
    lines.append(f"Absorbed Photon Flux ({emin:.1f}-{emax:.1f} keV): {flux_photons:.4e} photons/cm²/s")
    
    try:
        rate = float(spectrum.rate[0])
        rate_err = float(spectrum.rate[1]) if len(spectrum.rate) > 1 else None
    except Exception as e:
        raise RuntimeError(f"Error extracting rate: {e}")        
    
    result['rate'] = {
        'value': rate,
        'error': rate_err
    }
    
    if rate is not None:
        if rate_err is not None:
            lines.append(f"Rate: {rate:.4f} ± {rate_err:.4f} cts/s")
        else:
            lines.append(f"Rate: {rate:.4f} cts/s")
    
    exposure = spectrum.exposure if hasattr(spectrum, 'exposure') else None
    
    if rate is not None and rate > 0 and flux_erg > 0:
        conv_factor = 10**model.cflux.lg10Flux.values[0] / rate
    else:
        conv_factor = None
    photon_counts = rate * exposure if rate is not None and exposure is not None else None
    result['conversion'] = {
        'exposure_s': exposure,
        'erg_per_count': conv_factor,
        'counts': photon_counts
    }
    
    if exposure is not None:
        lines.append(f"Exposure: {exposure:.1f} s")
    
    if conv_factor is not None:
        lines.append(f"Conversion factor: {conv_factor:.4e} erg/cm²/s per cts/s")
    if photon_counts is not None:
        lines.append(f"Total counts: {photon_counts:.2f} counts")
    
    statistic = xspec.Fit.statistic
    dof = xspec.Fit.dof
    stat_method = xspec.Fit.statMethod
    
    statdof = statistic / dof
    
    lines.append(f"Stat/dof: {stat_method}={statistic:.2f}/{dof}={statdof:.2f}")
    lines.append(f"Null hypothesis probability: {xspec.Fit.nullhyp:.4f}")
    
    result['statistics'] = {
        'method': stat_method,
        'value': statistic,
        'dof': dof,
        'reduced': statdof,
        'null_hypothesis_probability': xspec.Fit.nullhyp
    }
    
    result['text'] = "\n".join(lines)
    
    return result


def _parse_nhtot_response(html, coord_str=""):
    """Parse the ASCII table from the nhtot HTML response.

    Pure parsing function — no network I/O.  Separated for testability.

    Parameters
    ----------
    html : str
        Raw HTML response body from donhtot.php.
    coord_str : str
        Coordinate string for error messages only.

    Returns
    -------
    dict
        Always contains ``ok`` (bool).  On success (ok=True), also contains
        ra, dec, ebv_mean/weighted, nhi_mean/weighted, nh2_mean/weighted,
        nhtot_mean/weighted.  On failure (ok=False), contains ``error`` (str)
        and all value fields set to None.
    """
    import re

    _NONE = {
        'ra': None, 'dec': None,
        'ebv_mean': None, 'ebv_weighted': None,
        'nhi_mean': None, 'nhi_weighted': None,
        'nh2_mean': None, 'nh2_weighted': None,
        'nhtot_mean': None, 'nhtot_weighted': None,
    }

    def _num(s):
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


def nhtot(ra, dec, equinox=2000):
    """
    Query the Swift UKSSDC nhtot service for Galactic hydrogen column density
    using the method of Willingale et al. (2013, MNRAS, 431, 394).

    Parameters
    ----------
    ra : float or str
        Right Ascension in decimal degrees (e.g. 159.386) or sexagesimal
        (e.g. "10:37:32.6").
    dec : float or str
        Declination in decimal degrees (e.g. 56.171) or sexagesimal
        (e.g. "+56:10:15.6").
    equinox : int
        Equinox: 2000 for J2000, 1950 for B1950.

    Returns
    -------
    dict
        Keys: ok (bool), ra (str), dec (str), ebv_mean (float),
        ebv_weighted (float), nhi_mean (float), nhi_weighted (float),
        nh2_mean (float), nh2_weighted (float), nhtot_mean (float),
        nhtot_weighted (float).  All NH in atoms cm⁻², E(B-V) in mag.
        On failure, ok=False, error (str) set, and all value fields None.
        On success, ok=True.

    Examples
    --------
    >>> result = nhtot(159.386, 56.171)
    >>> print(result['nhtot_weighted'])
    5.26e+19

    >>> result = nhtot("10:30:00", "+50:00:00")
    """
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
        """
        展示流体力学的激波跳变条件（Rankine-Hugoniot conditions）
        """
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
        """
        星系形成历史（SFH）类，用于处理和分析星系的形成和演化历史。
        """
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

def li_ma_snr(n_on: float, n_off: float, alpha: float, *, signed: bool = False) -> float:
    """Compute Li & Ma significance (Eq. 17 in Li & Ma 1983).

    Parameters
    ----------
    n_on : float
        Total counts in ON region.
    n_off : float
        Total counts in OFF region (reference background).
    alpha : float
        Exposure/area scaling: alpha = (A_on/A_off) * (t_on/t_off).
    signed : bool
        If ``True``, return a signed significance whose sign matches
        ``sign(n_on - alpha * n_off)`` — positive for source excess,
        negative for background deficit.  Default ``False`` preserves
        the legacy behaviour (always >= 0).

    Returns
    -------
    float
        Li & Ma significance.  Returns 0.0 for degenerate inputs.
    """
    if n_on <= 0 and n_off <= 0:
        return 0.0
    if alpha <= 0:
        return 0.0
    n_on = float(max(n_on, 0.0))
    n_off = float(max(n_off, 0.0))
    if n_on == 0.0:
        return 0.0
    if n_off == 0.0:
        s = float(np.sqrt(2.0 * n_on * np.log(1.0 + alpha)))
    else:
        term1 = n_on * np.log(((1.0 + alpha) / alpha) * (n_on / (n_on + n_off)))
        term2 = n_off * np.log((1.0 + alpha) * (n_off / (n_on + n_off)))
        val = 2.0 * (term1 + term2)
        s = float(np.sqrt(max(val, 0.0)))
    if signed:
        import math
        return math.copysign(s, n_on - alpha * n_off)
    return s


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
        i0 = int(np.searchsorted(self.time, left, side="left"))
        i1 = int(np.searchsorted(self.time, right, side="left"))
        if i1 <= i0:
            return 0.0
        return float(self._cum[i1 - 1] - (self._cum[i0 - 1] if i0 > 0 else 0.0))

    def _snr_window(
        self, left: float, right: float, n_off_ref: _Optional[float] = None,
    ) -> float:
        n_on = self._counts_in(left, right)
        t_on = max(0.0, float(right - left))
        if t_on <= 0:
            return 0.0
        alpha = self.bg.alpha(t_on)
        n_off = float(self.bg.n_off_ref if n_off_ref is None else n_off_ref)
        return li_ma_snr(n_on=n_on, n_off=n_off, alpha=alpha, signed=True)

    def sliding_window(
        self, *, window: float = 1200.0, step: _Optional[float] = None,
    ) -> _Tuple[bool, dict]:
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
        return bool(max_snr >= 7.0), {"max_snr": max_snr, "best_window": best}

    def head_window(self, *, window: float = 1200.0) -> _Tuple[bool, dict]:
        left = float(self.time[0])
        right = left + float(window)
        snr = self._snr_window(left, right)
        return bool(snr >= 7.0), {"snr": snr, "window": (left, right)}

    def _find_t0(
        self, mode: _Literal["first_nonzero", "first_time"] = "first_nonzero",
    ) -> float:
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
            snr = li_ma_snr(n_on=float(csum[k - 1]), n_off=float(self.bg.n_off_ref), alpha=alpha, signed=True)
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
        slid_ok, slid_stat = self.sliding_window(window=window, step=step)
        if slid_ok:
            return {"triggered": True, "method": "sliding", **slid_stat}
        head_ok, head_stat = self.head_window(window=window)
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
                    "[LightcurveSNREvaluator] Using 'raw_corrected_counts' as ON counts.\n"
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
        i0 = int(np.searchsorted(self.time, left, side="left"))
        i1 = int(np.searchsorted(self.time, right, side="left"))
        if i1 <= i0:
            return 0.0
        n_on = float(self._cum_counts[i1 - 1] - (self._cum_counts[i0 - 1] if i0 > 0 else 0.0))
        t_on = right - left
        alpha = self._alpha(t_on)
        return li_ma_snr(n_on=n_on, n_off=n_off, alpha=alpha, signed=True)

    def _alpha(self, t_on: float) -> float:
        return float(self.area_ratio) * (float(t_on) / float(self.off_exposure_ref))

    def _find_T0_by_blocks(
        self,
        snr_thr: float = 3.0,
        n_off: _Optional[float] = None,
        rng: _Optional[np.random.Generator] = None,
        off_mode: _Literal["fixed", "poisson"] = "fixed",
    ) -> float:
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
                snr = li_ma_snr(n_on=n_on, n_off=float(n_off_exp), alpha=alpha, signed=True)
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
                snr = li_ma_snr(n_on=float(csum[k - 1]), n_off=n_off, alpha=alpha, signed=True)
                if snr > max_snr:
                    max_snr = snr
            max_snrs.append(float(max_snr))
            hits += int(max_snr >= target)

        prob = hits / float(n_mc)
        return bool(prob >= 0.95), {"prob": prob, "max_snrs": np.asarray(max_snrs)}
