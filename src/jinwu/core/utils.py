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


def snr_li_ma(n_src, n_bkg, alpha_area_time):
    """
    Calculate the signal-to-noise ratio (SNR) using the Li & Ma formula.

    Parameters:
    n_src (int): 源区域的计数
    n_bkg (int): 背景区域的计数
    alpha_area_time (float): 	•	\alpha：背景区域与源区域之间的归一化因子，反映暴露时间或面积比：
    \alpha_area_time = \frac{t_{\text{on}} A_{\text{on}}}{t_{\text{off}} A_{\text{off}}}

    Returns:
    float: The calculated SNR.
    """
    if n_bkg == 0:
        return np.inf  # Avoid division by zero, return infinity if no background counts
    part1 = n_src*np.log((1 + alpha_area_time) * n_src / alpha_area_time /(n_bkg+n_src))
    part2 = n_bkg*np.log((1+alpha_area_time)*n_bkg/(n_bkg+n_src))
    snr = np.sqrt(2 * (part1 + part2))
    return snr


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
# Import remains here to preserve ``jinwu.core.utils.RedshiftExtrapolator``.
from jinwu.lf.legacy_redshift import RedshiftExtrapolator

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





