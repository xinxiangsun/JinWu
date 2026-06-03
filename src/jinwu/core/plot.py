# -*- coding: utf-8 -*-
"""
jinwu.core.plot

只负责绘图（plot-only）。本模块提供对 OGIP 数据的基础可视化：
- PHA 能谱图（EP/WXT、Swift/BAT 等）
- 光变曲线（支持单能段与多能段）

与 jinwu.core.data / jinwu.core.io 的数据类和读取器配合使用：
- 接受 PhaData / LightcurveData 直接绘图；
- 也可传入路径，内部自动读取并路由；
- 一切筛选/切片/重采样逻辑请放在其他模块完成后再调用这里的绘图。

"""

from __future__ import annotations
from PIL import Image
from typing import Optional, Union, List, Tuple, Any, Callable, cast, Literal
from pathlib import Path
import warnings
import os

import xspec
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
try:
        from .time import Time as AstroTime, TimeDelta as AstroTimeDelta
except Exception:  # pragma: no cover
        from astropy.time import Time as AstroTime, TimeDelta as AstroTimeDelta
from matplotlib.axes import Axes
from matplotlib.figure import Figure

"""在运行期尝试导入实际数据类；若失败，则使用哑类以便 isinstance 不抛错。"""
try:  # runtime import; avoids typing.Any isinstance crash
    from .data import PhaData, LightcurveData  # type: ignore
except Exception:  # pragma: no cover
    class _Dummy:  # minimal placeholder
        pass
    PhaData = _Dummy  # type: ignore
    LightcurveData = _Dummy  # type: ignore

try:
    from .io import readfits, guess_ogip_kind  # type: ignore
except Exception:  # 允许在未安装上游模块时静态检查通过
    def readfits(path: Union[str, Path], kind: Optional[str] = None) -> Any:  # type: ignore
        return path
    def guess_ogip_kind(path: Union[str, Path]) -> str:  # type: ignore
        return "unknown"

PathLike = Union[str, Path]

__all__ = [
    "plot_spectrum",
    "plot_lightcurve",
    "plot_event_txx",
    "plot_ogip",
    "plotfit",
    "plot_xspec_origin",
]


# ----------------------------
# 内部小工具
# ----------------------------

def _ensure_axes(ax: Optional[Axes], figsize=(7.5, 4.5)) -> Axes:
    if ax is not None:
        return ax
    fig, ax = plt.subplots(figsize=figsize)
    return ax


# ----------------------------
# PHA 能谱绘图
# ----------------------------

def plot_spectrum(
    src: Union[Any, PathLike, fits.HDUList],
    *,
    ax: Optional[Axes] = None,
    ykind: str = "rate",  # "rate" 或 "counts"
    show_errorbar: bool = True,
    color: Optional[str] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = True,
    out: Optional[PathLike] = None,
) -> Axes:
    """绘制 OGIP PHA 能谱。

    参数
    - src: 可为 PhaData 或 PHA 文件路径（或已打开的 HDUList）。
    - ykind: "rate" 时优先 COUNTS/EXPOSURE，"counts" 时直接 COUNTS。
    - out: 若提供路径，则保存图片（不做任何非绘图处理）。
    """
    # 将输入统一为 PhaData
    pha: Optional[Any] = None
    hdul: Optional[fits.HDUList] = None
    # (no file closing logic needed here; we only accept already opened dataclasses or HDUList/path)

    # Duck typing：优先按 kind == 'pha'
    if (hasattr(src, 'kind') and getattr(src, 'kind') == 'pha'):
        pha = src  # type: ignore[assignment]
    elif isinstance(src, (str, Path)):
        # 借助 file 模块读取
        obj = readfits(src, kind="pha")  # type: ignore[arg-type]
        if hasattr(obj, "kind") and getattr(obj, "kind") == "pha":
            pha = obj  # type: ignore[assignment]
        else:
            raise ValueError(f"提供的路径看起来不是 PHA：{src}")
    elif isinstance(src, fits.HDUList):
        hdul = src
    else:
        raise TypeError("plot_spectrum 需要 PhaData、路径或 HDUList 作为输入")

    ax = _ensure_axes(ax)

    # 情况 A：我们已有 PhaData
    if pha is not None:
        hdr = pha.header if hasattr(pha, "header") else {}
        # X 轴：优先能量（来自 ebounds），否则 channel
        if getattr(pha, "ebounds", None) is not None:
            ch, e_lo, e_hi = pha.ebounds  # type: ignore[misc]
            x = 0.5 * (np.asarray(e_lo, float) + np.asarray(e_hi, float))
            xerr = 0.5 * (np.asarray(e_hi, float) - np.asarray(e_lo, float))
            xlabel = "Energy (keV)"
        else:
            x = np.asarray(pha.channels, float)
            xerr = None
            xlabel = "Channel"

        # Y 轴：rate 或 counts
        if ykind == "rate":
            exp = getattr(pha, "exposure", None)
            if exp and float(exp) > 0:
                y = np.asarray(pha.counts, float) / float(exp)
                ylab = "Rate (counts s$^{-1}$)"
                yerr = (
                    np.asarray(pha.stat_err, float) / float(exp)
                    if pha.stat_err is not None
                    else None
                )
            else:
                y = np.asarray(pha.counts, float)
                ylab = "Counts (no EXPOSURE)"
                yerr = np.asarray(pha.stat_err, float) if pha.stat_err is not None else None
        else:
            y = np.asarray(pha.counts, float)
            ylab = "Counts"
            yerr = np.asarray(pha.stat_err, float) if pha.stat_err is not None else None

        if show_errorbar and yerr is not None:
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="o", ms=3.5, lw=1, color=color, label=label)
        else:
            if xerr is not None:
                ax.errorbar(x, y, xerr=xerr, fmt="o", ms=3.5, lw=1, color=color, label=label)
            else:
                ax.plot(x, y, "o-", ms=3.5, lw=1, color=color, label=label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        if grid:
            ax.grid(alpha=0.3, ls="--")
        if title is None:
            base = Path(getattr(pha, "path", "")).name
            exp = getattr(pha, "exposure", None)
            if exp:
                title = f"{base}  EXP={float(exp):.1f}s"
            else:
                title = base or "PHA"
        ax.set_title(title)
        if label:
            ax.legend()

    # 情况 B：直接从 HDUList 绘图（极少用，仅兜底）
    else:
        assert hdul is not None
        try:
            # 找到 SPECTRUM 表
            spec = None
            for h in hdul[1:]:
                if isinstance(h, fits.BinTableHDU) and h.name.upper() in ("SPECTRUM", "PHA"):
                    spec = h
                    break
            if spec is None:
                for h in hdul[1:]:
                    if isinstance(h, fits.BinTableHDU):
                        spec = h
                        break
            if spec is None:
                raise ValueError("未在 HDUList 中找到 SPECTRUM 表用于 PHA 绘图")
            d = spec.data
            cols = list(getattr(spec.columns, "names", []) or [])
            # X 轴：尝试 EBOUNDS
            try:
                eb_hdu = hdul["EBOUNDS"]
                eb = getattr(eb_hdu, "data", None)
                if eb is None:
                    raise KeyError("EBOUNDS has no data")
                e_lo = np.asarray(eb["E_MIN"], float)
                e_hi = np.asarray(eb["E_MAX"], float)
                x = 0.5 * (e_lo + e_hi)
                xerr = 0.5 * (e_hi - e_lo)
                xlabel = "Energy (keV)"
            except Exception:
                if "CHANNEL" in cols:
                    x = np.asarray(d["CHANNEL"], float)
                else:
                    x = np.arange(len(d), dtype=float)
                xerr = None
                xlabel = "Channel"

            # Y 轴
            if "RATE" in cols and ykind == "rate":
                y = np.asarray(d["RATE"], float)
                yerr = np.asarray(d["ERROR"], float) if "ERROR" in cols else None
                ylab = "Rate (counts s$^{-1}$)"
            else:
                y = np.asarray(d["COUNTS"], float) if "COUNTS" in cols else np.asarray(d["RATE"], float)
                yerr = np.asarray(d.get("STAT_ERR"), float) if "STAT_ERR" in cols else None  # type: ignore[arg-type]
                ylab = "Counts"

            if show_errorbar and yerr is not None:
                ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="o", ms=3.5, lw=1, color=color, label=label)
            else:
                if xerr is not None:
                    ax.errorbar(x, y, xerr=xerr, fmt="o", ms=3.5, lw=1, color=color, label=label)
                else:
                    ax.plot(x, y, "o-", ms=3.5, lw=1, color=color, label=label)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylab)
            if grid:
                ax.grid(alpha=0.3, ls="--")
            if title is None:
                title = Path(str(hdul.filename())).name if hasattr(hdul, "filename") else "PHA"
            ax.set_title(title)
            if label:
                ax.legend()
        finally:
            pass

    if out is not None:
        fig = cast(Figure, ax.get_figure() if hasattr(ax, "get_figure") else plt.gcf())
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
    return ax


# ----------------------------
# 光变曲线绘图
# ----------------------------

def plot_lightcurve(
    src: Union[Any, PathLike, fits.HDUList],
    *,
    ax: Optional[Axes] = None,
    srcname: Optional[str] = None,
    T0: Optional[float] = None,
    ykind: Literal['auto', 'rate', 'counts', 'flux'] = 'auto',  # auto: 根据 is_rate 自动选择；flux 需要外部提供 flux_array 或从 spectrum 反演
    multiband: Union[bool, str] = "auto",  # auto: 若 value 为 (N,M) 则分面
    color: Optional[str] = None,
    label: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = True,
    out: Optional[PathLike] = None,
    flux_array: Optional[np.ndarray] = None,  # TODO: 待基类扩展后改为 LightcurveData.flux 字段并支持自动推导
) -> Union[Axes, List[Axes]]:
    """绘制光变曲线（单能段或多能段）。

    参数
    - src: 可为 LightcurveData 或 LC 文件路径（或 HDUList）。
    - srcname: 源名称（源昵称），若提供则在标题/标签中显示
    - ykind: "rate" 或 "counts" 或 "flux"；auto 时根据 is_rate 自动选择
    - multiband: True/"auto" 时若 value 为 (N,M) 则按行分面绘制
    - color: 线条颜色
    - label: 数据标签
    - title: 图表标题（若为 None 则自动生成）
    - grid: 是否显示网格线
    - out: 若提供路径，则保存图片
    - flux_array: 若 ykind='flux' 需要手动提供 flux 数组

    说明
    - 对于从 join() 得到的多条 LC，横轴使用 "Time since <TIMEZERO_0>" 格式
    - 自动标注仪器名称（从 TELESCOP 字段）
    - 若多条 LC 有不同的 timezero，会在图上标注
    """
    lc: Optional[Any] = None
    hdul: Optional[fits.HDUList] = None

    if (hasattr(src, 'kind') and getattr(src, 'kind') == 'lc'):
        lc = src  # type: ignore[assignment]
    elif isinstance(src, (str, Path)):
        obj = readfits(src, kind="lc")  # type: ignore[arg-type]
        if hasattr(obj, "kind") and getattr(obj, "kind") == "lc":
            lc = obj  # type: ignore[assignment]
        else:
            raise ValueError(f"提供的路径看起来不是 LC：{src}")
    elif isinstance(src, fits.HDUList):
        hdul = src
    else:
        raise TypeError("plot_lightcurve 需要 LightcurveData、路径或 HDUList 作为输入")

    def _draw(
        ax_: Axes,
        t: np.ndarray,
        y: np.ndarray,
        yerr: Optional[np.ndarray],
        lab: Optional[str],
        timezero: Optional[float],
        ylab: str,
        timezero_obj: Optional[Any] = None,
        instrume: Optional[str] = None,
        binsize: Optional[float] = None,
        lc_kind: Optional[str] = None,
    ):
        """绘制光变曲线数据点和误差棒
        
        参数
        - t: 时间数组（相对于 timezero 的秒数）
        - y: 计数或速率数组
        - yerr: 误差数组（可选）
        - lab: 数据标签
        - timezero: TIMEZERO 值（第一个点的绝对时间）
        - ylab: Y 轴标签
        - timezero_obj: Time 对象（用于显示 UTC 时间）
        - instrume: 仪器/望远镜名称（用于标注）
        - binsize: bin 宽度（秒），若提供则在标签中标注
        - lc_kind: 光变曲线类型（'src'/'bkg'/'net'），用于标注
        """
        if yerr is not None:
            ax_.errorbar(t, y, yerr=yerr, fmt="-", lw=1.0, color=color, label=lab)
        else:
            ax_.plot(t, y, "-", lw=1.0, color=color, label=lab)
        
        # 构造 X 轴标签：包含 TIMEZERO 对应的 UTC 信息
        xlabel_parts = ["Time since"]
        if timezero_obj is not None:
            try:
                # 获取 UTC 时间字符串（优先使用 astropy Time 的 timezero_obj.utc.isot）
                utc_str = None
                if hasattr(timezero_obj, 'utc') and hasattr(getattr(timezero_obj, 'utc'), 'isot'):
                    utc_str = timezero_obj.utc.isot
                elif hasattr(timezero_obj, 'isot'):
                    utc_str = timezero_obj.isot
                else:
                    utc_str = str(timezero_obj)
                xlabel_parts.append(f"{utc_str} (UTC) (s)")
            except Exception:
                if timezero is not None:
                    xlabel_parts.append(f"{timezero:.3f} (s)")
                else:
                    xlabel_parts.append("start (s)")
        else:
            if timezero is not None:
                xlabel_parts.append(f"{timezero:.3f} (s)")
            else:
                xlabel_parts.append("start (s)")
        
        # 如果有仪器名称 / bin 宽度 / 数据类型，在 X 轴标签末尾补充
        extra_parts = []
        if instrume is not None:
            extra_parts.append(str(instrume))
        if binsize is not None:
            try:
                extra_parts.append(f"Δt={float(binsize):.3f} s")
            except Exception:
                pass
        if lc_kind is not None:
            extra_parts.append(lc_kind)

        xlabel = " ".join(xlabel_parts)
        if extra_parts:
            xlabel += "  [" + " | ".join(extra_parts) + "]"
        
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel(ylab)
        if grid:
            ax_.grid(alpha=0.3, ls="--")

    # 情况 A：LightcurveData
    if lc is not None:
        # 时间轴（必须存在）
        time = np.asarray(lc.time, float)

        # 主数据列：优先 value，缺失时回退到 counts/rate
        val_raw = getattr(lc, "value", None)
        if val_raw is None:
            if getattr(lc, "counts", None) is not None:
                val_raw = lc.counts
            elif getattr(lc, "rate", None) is not None:
                val_raw = lc.rate
        if val_raw is None:
            raise ValueError("LightcurveData 缺少 value/counts/rate，无法绘图")
        try:
            val_arr = np.asarray(val_raw, float)
        except Exception as exc:
            raise ValueError("无法将光变曲线主数据转换为 numpy 数组") from exc

        # 误差列：优先 error，缺失时回退到 counts_err/rate_err
        err_raw = getattr(lc, "error", None)
        if err_raw is None:
            if getattr(lc, "counts_err", None) is not None:
                err_raw = lc.counts_err
            elif getattr(lc, "rate_err", None) is not None:
                err_raw = lc.rate_err
        err = None if err_raw is None else np.asarray(err_raw, float)

        # 提取仪器/望远镜名称和 timezero_obj
        instrume = None
        # 优先使用 meta.instrume；否则退回 TELESCOP
        if getattr(lc, 'meta', None) is not None and getattr(lc.meta, 'instrume', None) is not None:  # type: ignore[union-attr]
            instrume = str(lc.meta.instrume)  # type: ignore[union-attr]
        elif hasattr(lc, 'telescop') and lc.telescop is not None:
            instrume = str(lc.telescop)

        # 检查必需的时间参考对象
        if not hasattr(lc, 'timezero_obj') or lc.timezero_obj is None:
            raise ValueError(
                "LightcurveData 缺少 timezero_obj 字段。"
                "绘图需要时间参考对象以正确显示 UTC 时间标签。"
            )
        timezero_obj_lc = lc.timezero_obj

        # 推断 bin 宽度（秒）
        binsize_val: Optional[float] = None
        try:
            if getattr(lc, 'dt', None) is not None:
                dt = lc.dt  # type: ignore[assignment]
                if isinstance(dt, (int, float)):
                    binsize_val = float(dt)
                else:
                    arr_dt = np.asarray(dt, float)
                    if arr_dt.size:
                        binsize_val = float(np.median(arr_dt))
            elif getattr(lc, 'bin_hi', None) is not None and getattr(lc, 'bin_lo', None) is not None:
                arr_hi = np.asarray(lc.bin_hi, float)
                arr_lo = np.asarray(lc.bin_lo, float)
                if arr_hi.size and arr_lo.size and arr_hi.size == arr_lo.size:
                    binsize_val = float(np.median(arr_hi - arr_lo))
        except Exception:
            binsize_val = None

        # 推断 LC 类型：src / bkg / net
        lc_kind: Optional[str] = None
        hdr = getattr(lc, 'header', None)
        if isinstance(hdr, dict):
            for key in ('HDUCLAS2', 'LCTYPE', 'DATATYPE'):
                hdr_val = hdr.get(key)
                if isinstance(hdr_val, str):
                    v_up = hdr_val.strip().upper()
                    if v_up in ('SRC', 'SOURCE'):
                        lc_kind = 'src'
                    elif v_up in ('BKG', 'BKGD', 'BACK', 'BACKGROUND'):
                        lc_kind = 'bkg'
                    elif v_up in ('NET', 'NETLC'):
                        lc_kind = 'net'
                    if lc_kind is not None:
                        break

        # 若用户未提供 legend label，则用 lc_kind 作为默认标签
        label_eff = label
        if label_eff is None and lc_kind is not None:
            label_eff = lc_kind

        # 根据 is_rate / ykind 选择绘制 counts 还是 rate；必要时用 dt/bin_exposure 进行转换
        def _width_array_like(v: np.ndarray) -> Optional[np.ndarray]:
            width = None
            if getattr(lc, 'bin_exposure', None) is not None:
                width = np.asarray(lc.bin_exposure, float)
            elif getattr(lc, 'bin_width', None) is not None:
                width = np.asarray(lc.bin_width, float)
            elif getattr(lc, 'dt', None) is not None:
                dt_raw = lc.dt
                dt_arr = np.asarray(dt_raw, float)
                if dt_arr.ndim == 0:
                    dt_val = float(dt_arr)
                    width = dt_val if (np.isfinite(dt_val) and dt_val > 0) else None
                else:
                    width = dt_arr
            if width is None:
                return None
            if np.ndim(width) == 0:
                return np.full(v.shape[0], float(width), dtype=float)
            width_arr = np.asarray(width, float)
            if width_arr.shape[0] != v.shape[0]:
                return None
            return width_arr

        def _to_rate(v: np.ndarray, e: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray], bool]:
            w = _width_array_like(v)
            if w is None:
                return v, e, False
            w = np.where(w > 0.0, w, np.nan)
            w_b = w[:, None] if (v.ndim == 2 and w.ndim == 1) else w
            rate_val = v / w_b
            rate_err = e / w_b if e is not None else None
            return rate_val, rate_err, True

        def _to_counts(v: np.ndarray, e: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray], bool]:
            w = _width_array_like(v)
            if w is None:
                return v, e, False
            w = np.where(w > 0.0, w, np.nan)
            w_b = w[:, None] if (v.ndim == 2 and w.ndim == 1) else w
            cnt_val = v * w_b
            cnt_err = e * w_b if e is not None else None
            return cnt_val, cnt_err, True

        if ykind == 'auto':
            eff_kind: Literal['rate', 'counts', 'flux'] = 'rate' if lc.is_rate else 'counts'
        else:
            eff_kind = ykind  # type: ignore[assignment]

        ylab = "Rate (counts s$^{-1}$)" if eff_kind == 'rate' else ("Counts" if eff_kind == 'counts' else "Flux (erg cm$^{-2}$ s$^{-1}$)")

        # TODO: 后续基类扩展 LightcurveData 时，添加 flux / flux_error 字段以及可选的响应矩阵信息
        #       以便自动从 counts/rate 反演 flux；目前需要外部提供 flux_array。
        if eff_kind == 'flux':
            if flux_array is not None:
                y = np.asarray(flux_array, float)
                # TODO: 计算 flux_error 的传播（需要 counts/rate error + 响应矩阵）
                yerr = None
            else:
                # 未提供 flux_array 则降级到 rate
                warnings.warn("ykind='flux' 但未提供 flux_array；降级到 rate", UserWarning)
                eff_kind = 'rate'
                ylab = "Rate (counts s$^{-1}$)"
                if lc.is_rate:
                    y = val_arr
                    yerr = err
                else:
                    y, yerr, ok = _to_rate(val_arr, err)
                    if not ok:
                        y, yerr = val_arr, err
                        ylab = "Counts"
        elif eff_kind == 'rate':
            if lc.is_rate:
                y = val_arr
                yerr = err
            else:
                y, yerr, ok = _to_rate(val_arr, err)
                if not ok:
                    # 无法转换则退回 counts 并更新标签
                    y, yerr = val_arr, err
                    ylab = "Counts"
            # 如果转换导致标签变化，保持 ylab 与实际一致
        else:  # eff_kind == 'counts'
            if not lc.is_rate:
                y = val_arr
                yerr = err
            else:
                y, yerr, ok = _to_counts(val_arr, err)
                if not ok:
                    # 无法转换则退回 rate 并更新标签
                    y, yerr = val_arr, err
                    ylab = "Rate (counts s$^{-1}$)"

        # 使用原始 TIME，不再减去 TRIGTIME/T0；仅在标签中标注 TIMEZERO
        t = time
        tz = None
        # 优先从 lc.timezero 字段
        if hasattr(lc, 'timezero') and lc.timezero is not None:
            try:
                tz = float(lc.timezero)
            except (ValueError, TypeError):
                tz = None
        # 其次 meta.timezero
        if tz is None and hasattr(lc, 'meta') and getattr(lc.meta, 'timezero', None) is not None:
            try:
                tz = float(lc.meta.timezero)  # type: ignore[arg-type]
            except Exception:
                tz = None
        # 再次 header.TIMEZERO
        if tz is None and hasattr(lc, 'header'):
            hv = lc.header.get('TIMEZERO')
            try:
                tz = float(hv) if hv is not None else None
            except Exception:
                tz = None
        
        # 检查 timezero 是否成功获取
        if tz is None:
            raise ValueError(
                "LightcurveData 缺少 timezero 字段。"
                "绘图需要 TIMEZERO 偏移量以正确标注时间轴。"
            )

        # 多能段还是单能段
        if val_arr.ndim == 2 and val_arr.shape[1] > 1 and (multiband is True or multiband == "auto"):
            nb = val_arr.shape[1]
            fig, axes = plt.subplots(nb, 1, sharex=True, figsize=(8.0, 1.9 * nb), constrained_layout=True)
            if nb == 1:
                axes = [axes]
            for i in range(nb):
                yerr_i = yerr[:, i] if (yerr is not None and yerr.ndim == 2 and yerr.shape[1] == nb) else None
                _draw(
                    axes[i],
                    t,
                    y[:, i],
                    yerr_i,
                    (label_eff or f"Band {i+1}"),
                    tz,
                    ylab,
                    timezero_obj=timezero_obj_lc,
                    instrume=instrume,
                    binsize=binsize_val,
                    lc_kind=lc_kind,
                )
                axes[i].legend(loc="upper right", fontsize=9)
            if title is None:
                base = Path(getattr(lc, "path", "")).name
                if srcname:
                    title = f"{srcname} Lightcurve" if base else srcname
                else:
                    title = base or "Lightcurve"
            axes[0].set_title(title)
            axes_to_return: Union[List[Axes], Axes]
            axes_to_return = list(axes)
        else:
            ax = _ensure_axes(ax)
            y1d = y.reshape(-1) if y.ndim > 1 else y
            yerr1d = yerr.reshape(-1) if (yerr is not None and yerr.ndim > 1) else yerr
            _draw(
                ax,
                t,
                y1d,
                yerr1d,
                label_eff,
                tz,
                ylab,
                timezero_obj=timezero_obj_lc,
                instrume=instrume,
                binsize=binsize_val,
                lc_kind=lc_kind,
            )
            if title is None:
                base = Path(getattr(lc, "path", "")).name
                if srcname:
                    title = f"{srcname} ({base})" if base else srcname
                else:
                    title = base or "Lightcurve"
            ax.set_title(title)
            if label_eff:
                ax.legend(loc="best")
            axes_to_return = ax

    # 情况 B：HDUList（兜底）
    else:
        assert hdul is not None
        rate_hdu = None
        for h in hdul[1:]:
            if isinstance(h, fits.BinTableHDU) and h.name.upper() in ("RATE", "LIGHTCURVE", "LC"):
                rate_hdu = h
                break
        if rate_hdu is None:
            for h in hdul[1:]:
                if isinstance(h, fits.BinTableHDU):
                    rate_hdu = h
                    break
        if rate_hdu is None:
            raise ValueError("未在 HDUList 中找到 RATE/LC 表")
        d = rate_hdu.data
        cols = list(getattr(rate_hdu.columns, "names", []) or [])
        time = np.asarray(d["TIME"], float)
        # 安全获取 header
        hdr0_dict: dict = {}
        if hasattr(hdul[0], "header"):
            hdr_raw = getattr(hdul[0], "header")
            if hasattr(hdr_raw, "get"):
                hdr0_dict = hdr_raw  # type: ignore
        # Raw TIME for HDUList case
        t = time
        tz = None
        hv_meta = hdr0_dict.get('TIMEZERO')
        if hv_meta is not None:
            try:
                tz = float(hv_meta)
            except Exception:
                tz = None

        # 优先 TOT_RATE；否则 RATE 可能为 (N, M)
        if "TOT_RATE" in cols:
            val = np.asarray(d["TOT_RATE"], float)
            err = np.asarray(d["TOT_ERROR"], float) if "TOT_ERROR" in cols else None
            ax = _ensure_axes(ax)
            _draw(ax, t, val, err, label, tz, "Rate (counts s$^{-1}$)", timezero_obj=None, instrume=None, binsize=None, lc_kind=None)
            axes_to_return = ax
        else:
            rate = d["RATE"] if "RATE" in cols else d["COUNTS"]
            val = np.asarray(rate)
            err = np.asarray(d["ERROR"], float) if "ERROR" in cols else None
            ylab = "Rate (counts s$^{-1}$)" if "RATE" in cols else "Counts"
            if val.ndim == 2 and val.shape[1] > 1 and (multiband is True or multiband == "auto"):
                nb = val.shape[1]
                fig, axes = plt.subplots(nb, 1, sharex=True, figsize=(8.0, 1.9 * nb), constrained_layout=True)
                if nb == 1:
                    axes = [axes]
                for i in range(nb):
                    yerr_i = err[:, i] if (err is not None and err.ndim == 2 and err.shape[1] == nb) else None
                    _draw(axes[i], t, val[:, i], yerr_i, (label or f"Band {i+1}"), tz, ylab, timezero_obj=None, instrume=None, binsize=None, lc_kind=None)
                axes_to_return = list(axes)
            else:
                ax = _ensure_axes(ax)
                _draw(ax, t, val.reshape(-1), err.reshape(-1) if (err is not None and err.ndim > 1) else err, label, tz, ylab, timezero_obj=None, instrume=None, binsize=None, lc_kind=None)
                axes_to_return = ax

        if title is None:
            base_name = ""
            if hasattr(hdul, "filename"):
                try:
                    base_name = Path(str(hdul.filename())).name  # type: ignore[attr-defined]
                except Exception:
                    base_name = ""
            title = base_name or "Lightcurve"
        if isinstance(axes_to_return, list):
            axes_to_return[0].set_title(title)
        else:
            axes_to_return.set_title(title)

    # 保存文件（若需要）
    if out is not None:
        if isinstance(axes_to_return, list):
            ax0 = axes_to_return[0]
            fig = cast(Figure, ax0.get_figure() if hasattr(ax0, "get_figure") else plt.gcf())
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
        else:
            ax_ = axes_to_return
            fig = cast(Figure, ax_.get_figure() if hasattr(ax_, "get_figure") else plt.gcf())
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
    return axes_to_return


def plot_event_txx(
    src: Union[Any, PathLike],
    txx_result: dict,
    *,
    background: Optional[Union[Any, PathLike]] = None,
    alpha: Optional[float] = None,
    srcname: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.5, 8.2),
    out: Optional[PathLike] = None,
    timezero: Optional[Any] = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """可视化事件 Txx 结果：Source/Background + Bayesian Blocks 风格图。

    参数
    - src: EventData 或事件文件路径。
    - txx_result: `ops.txx` 返回字典。
    - background: 背景 EventData 或路径；若提供则优先使用真实背景事件绘图。
    - alpha: 背景缩放系数；若缺省，将尝试从结果或数据自动估计。
    - srcname: 图中显示的源名称（缺省时自动从元信息推断）。
    - timezero: 时间零点。支持数值秒（与事件时间同标尺）或 astropy.time.Time。
    - out: 若提供路径则保存图片。
    """

    evt: Optional[Any] = None
    if hasattr(src, "kind") and getattr(src, "kind") == "evt":
        evt = src
    elif isinstance(src, (str, Path)):
        obj = readfits(src, kind="evt")  # type: ignore[arg-type]
        if hasattr(obj, "kind") and getattr(obj, "kind") == "evt":
            evt = obj
        else:
            raise ValueError(f"提供的路径看起来不是 EVT：{src}")
    else:
        raise TypeError("plot_event_txx 需要 EventData 或事件文件路径作为输入")

    bkg_evt: Optional[Any] = None
    if background is not None:
        if hasattr(background, "kind") and getattr(background, "kind") == "evt":
            bkg_evt = background
        elif isinstance(background, (str, Path)):
            bkg_obj = readfits(background, kind="evt")  # type: ignore[arg-type]
            if hasattr(bkg_obj, "kind") and getattr(bkg_obj, "kind") == "evt":
                bkg_evt = bkg_obj
            else:
                raise ValueError(f"提供的 background 看起来不是 EVT：{background}")
        else:
            raise TypeError("background 需要 EventData 或事件文件路径")

    bb_edges = np.asarray(txx_result.get("bb_edges_time", np.asarray([], dtype=float)), dtype=float).reshape(-1)
    if bb_edges.size < 2:
        raise ValueError("txx_result 缺少有效 bb_edges_time，无法绘图")

    def _safe_float_any(val: Any, default: float = np.nan) -> float:
        try:
            return float(val)
        except Exception:
            return float(default)

    def _event_timezero() -> float:
        tz_evt = _safe_float_any(getattr(evt, "timezero", np.nan), np.nan)
        if np.isfinite(tz_evt):
            return tz_evt
        meta = getattr(evt, "meta", None)
        if meta is not None:
            tz_evt = _safe_float_any(getattr(meta, "timezero", np.nan), np.nan)
            if np.isfinite(tz_evt):
                return tz_evt
        hdr = getattr(evt, "header", None)
        if isinstance(hdr, dict):
            tz_evt = _safe_float_any(hdr.get("TIMEZERO", np.nan), np.nan)
            if np.isfinite(tz_evt):
                return tz_evt
        return np.nan

    def _time_scalar(t_in: Any) -> Any:
        if AstroTime is not None and isinstance(t_in, AstroTime):
            try:
                if bool(getattr(t_in, "isscalar", True)):
                    return t_in
            except Exception:
                return t_in
            try:
                if len(t_in) == 0:
                    raise ValueError("timezero 为 Time 数组时不能为空")
                return t_in[0]
            except Exception:
                return t_in
        return t_in

    evt_tz = _event_timezero()
    evt_tz_obj = getattr(evt, "timezero_obj", None)

    def _resolve_plot_timezero(value: Any) -> tuple[float, Optional[Any]]:
        # 默认采用事件 timezero，使横轴与 time_rel 一致。
        if value is None:
            if np.isfinite(evt_tz):
                return float(evt_tz), evt_tz_obj
            return 0.0, None

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value), None

        if AstroTime is not None and isinstance(value, AstroTime):
            t0_obj = _time_scalar(value)
            if (not np.isfinite(evt_tz)) or (evt_tz_obj is None):
                raise ValueError("传入 astropy.time.Time 作为 timezero 时，事件对象需包含 timezero 与 timezero_obj")
            dt_sec = float((t0_obj - evt_tz_obj).to_value("sec"))
            return float(evt_tz + dt_sec), t0_obj

        if AstroTimeDelta is not None and isinstance(value, AstroTimeDelta):
            if not np.isfinite(evt_tz):
                raise ValueError("传入 astropy.time.TimeDelta 作为 timezero 时，事件对象需包含可用 timezero")
            dt_sec = _safe_float_any(value.to_value("sec"), np.nan)
            if not np.isfinite(dt_sec):
                raise ValueError("无法将 astropy.time.TimeDelta 转换为秒")
            return float(evt_tz + dt_sec), evt_tz_obj

        try:
            return float(value), None
        except Exception:
            pass

        if (evt_tz_obj is not None) and np.isfinite(evt_tz):
            try:
                dt_obj = value - evt_tz_obj
                if hasattr(dt_obj, "to_value"):
                    dt_sec = _safe_float_any(dt_obj.to_value("sec"), np.nan)
                    if np.isfinite(dt_sec):
                        return float(evt_tz + dt_sec), value
                if hasattr(dt_obj, "sec"):
                    dt_sec = _safe_float_any(dt_obj.sec, np.nan)
                    if np.isfinite(dt_sec):
                        return float(evt_tz + dt_sec), value
            except Exception:
                pass

        raise TypeError("timezero 仅支持数值秒、astropy.time.Time 或可转换为秒偏移的 Time-like 对象")

    tz_plot_abs, tz_plot_obj = _resolve_plot_timezero(timezero)

    def _axis_label_from_timezero(tz_abs: float, tz_obj: Optional[Any]) -> str:
        if tz_obj is not None:
            try:
                utc_str = None
                if hasattr(tz_obj, "utc") and hasattr(getattr(tz_obj, "utc"), "isot"):
                    utc_str = tz_obj.utc.isot
                elif hasattr(tz_obj, "isot"):
                    utc_str = tz_obj.isot
                else:
                    utc_str = str(tz_obj)
                return f"Time since {utc_str} (UTC) (s)"
            except Exception:
                pass
        if np.isfinite(tz_abs):
            return f"Time - {tz_abs:.3f} (s)"
        return "Time (s)"

    x_label = _axis_label_from_timezero(tz_plot_abs, tz_plot_obj)

    def _to_plot_x(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float) - float(tz_plot_abs)

    def _choose_event_time_axis(evt_obj: Any, edges_abs: np.ndarray) -> np.ndarray:
        cands: list[np.ndarray] = []

        t_raw = np.asarray(getattr(evt_obj, "time", np.asarray([], dtype=float)), dtype=float).reshape(-1)
        if t_raw.size > 0:
            t_raw = t_raw[np.isfinite(t_raw)]
            if t_raw.size > 0:
                cands.append(t_raw)

        t_rel = np.asarray(getattr(evt_obj, "time_rel", np.asarray([], dtype=float)), dtype=float).reshape(-1)
        tz_val = getattr(evt_obj, "timezero", None)
        try:
            tz = float(tz_val) if tz_val is not None else np.nan
        except Exception:
            tz = np.nan
        if t_rel.size > 0 and np.isfinite(tz):
            t_rel = t_rel[np.isfinite(t_rel)]
            if t_rel.size > 0:
                cands.append(t_rel + tz)

        abs_attr = getattr(evt_obj, "absolute_time", None)
        if abs_attr is not None:
            try:
                t_abs = np.asarray(abs_attr, dtype=float).reshape(-1)
                t_abs = t_abs[np.isfinite(t_abs)]
                if t_abs.size > 0:
                    cands.append(t_abs)
            except Exception:
                pass

        if not cands:
            return np.asarray([], dtype=float)

        lo = float(edges_abs[0])
        hi = float(edges_abs[-1])
        best = cands[0]
        best_in = int(np.sum((best >= lo) & (best <= hi)))
        for c in cands[1:]:
            n_in = int(np.sum((c >= lo) & (c <= hi)))
            if n_in > best_in:
                best = c
                best_in = n_in
        return best

    evt_time = _choose_event_time_axis(evt, bb_edges)
    if evt_time.size == 0:
        raise ValueError("事件数据 time 为空，无法绘制 Txx 结果")

    def _fit_len(arr_in: Any, n: int, fill: float = 0.0) -> np.ndarray:
        arr = np.asarray(arr_in, dtype=float).reshape(-1)
        if arr.size == n:
            return arr
        if arr.size > n:
            return arr[:n]
        out_arr = np.full(n, fill, dtype=float)
        if arr.size > 0:
            out_arr[:arr.size] = arr
        return out_arr

    # 贝叶斯分块计数与模型背景计数（用于分块标注/SNR 与背景率估计）。
    src_hist_bb, _ = np.histogram(evt_time, bins=bb_edges)
    n_bb = max(bb_edges.size - 1, 0)
    bkg_model_bb = _fit_len(txx_result.get("bb_block_bkg_model_counts", np.zeros(n_bb)), n_bb, fill=0.0)
    bb_width = np.maximum(np.diff(bb_edges), 1e-12)
    bb_center = 0.5 * (bb_edges[:-1] + bb_edges[1:])
    bb_bkg_rate = bkg_model_bb / bb_width

    def _estimate_bkg_counts_from_bb(edges_target: np.ndarray) -> np.ndarray:
        out = np.zeros(max(edges_target.size - 1, 0), dtype=float)
        if out.size == 0:
            return out
        l_bb = bb_edges[:-1]
        r_bb = bb_edges[1:]
        for i_bin, (a, b) in enumerate(zip(edges_target[:-1], edges_target[1:])):
            overlap = np.minimum(float(b), r_bb) - np.maximum(float(a), l_bb)
            overlap = np.maximum(overlap, 0.0)
            out[i_bin] = float(np.sum(overlap * bb_bkg_rate))
        return out

    def _infer_lc_binsize() -> float:
        v = txx_result.get("evt_binsize", np.nan)
        try:
            bs = float(v)
            if np.isfinite(bs) and bs > 0.0:
                return bs
        except Exception:
            pass

        e_cum = np.asarray(txx_result.get("cumulative_edges_time", np.asarray([], dtype=float)), dtype=float).reshape(-1)
        if e_cum.size >= 3:
            d = np.diff(e_cum)
            d = d[np.isfinite(d) & (d > 0.0)]
            if d.size > 0:
                med = float(np.median(d))
                if med > 0.0:
                    rel = float(np.std(d) / max(med, 1e-12))
                    if rel < 0.05:
                        return med
        return np.nan

    # 原始光变：优先使用固定 binsize 还原全时段 light curve。
    bs_lc = _infer_lc_binsize()
    if np.isfinite(bs_lc) and bs_lc > 0.0:
        lc_edges = np.arange(float(bb_edges[0]), float(bb_edges[-1]) + bs_lc, bs_lc, dtype=float)
        if lc_edges.size < 2:
            lc_edges = np.asarray([float(bb_edges[0]), float(bb_edges[-1])], dtype=float)
        elif lc_edges[-1] < float(bb_edges[-1]):
            lc_edges = np.append(lc_edges, float(bb_edges[-1]))
    else:
        lc_edges = np.asarray(bb_edges, dtype=float)

    src_hist_lc, _ = np.histogram(evt_time, bins=lc_edges)
    n_lc = max(lc_edges.size - 1, 0)

    bkg_model_lc_raw = np.asarray(txx_result.get("background_model_counts", np.asarray([], dtype=float)), dtype=float).reshape(-1)
    if bkg_model_lc_raw.size == n_lc:
        bkg_model_lc = bkg_model_lc_raw
    else:
        bkg_model_lc = _estimate_bkg_counts_from_bb(lc_edges)

    bkg_hist_lc_raw = np.zeros(n_lc, dtype=float)
    has_real_bkg = False
    if bkg_evt is not None:
        bkg_evt_time = _choose_event_time_axis(bkg_evt, bb_edges)
        if bkg_evt_time.size > 0:
            bkg_hist_lc_raw, _ = np.histogram(bkg_evt_time, bins=lc_edges)
            has_real_bkg = True

    alpha_eff = _safe_float_any(alpha, np.nan)
    if (not np.isfinite(alpha_eff)) or (alpha_eff < 0.0):
        alpha_eff = _safe_float_any(txx_result.get("alpha", np.nan), np.nan)
    if (not np.isfinite(alpha_eff)) or (alpha_eff < 0.0):
        sum_bkg_raw = float(np.sum(np.maximum(bkg_hist_lc_raw, 0.0))) if has_real_bkg else 0.0
        sum_bkg_model = float(np.sum(np.maximum(bkg_model_lc, 0.0)))
        if sum_bkg_raw > 0.0 and sum_bkg_model > 0.0:
            alpha_eff = sum_bkg_model / sum_bkg_raw

    if has_real_bkg:
        if np.isfinite(alpha_eff) and alpha_eff >= 0.0:
            bkg_counts_lc = bkg_hist_lc_raw.astype(float) * float(alpha_eff)
            bkg_err_counts_lc = np.sqrt(np.maximum(bkg_hist_lc_raw.astype(float), 0.0)) * float(alpha_eff)
            bkg_var_for_net = (float(alpha_eff) ** 2) * np.maximum(bkg_hist_lc_raw.astype(float), 0.0)
            bkg_label_top = f"Scaled background (α={alpha_eff:.4f})"
        else:
            bkg_counts_lc = bkg_hist_lc_raw.astype(float)
            bkg_err_counts_lc = np.sqrt(np.maximum(bkg_hist_lc_raw.astype(float), 0.0))
            bkg_var_for_net = np.maximum(bkg_hist_lc_raw.astype(float), 0.0)
            bkg_label_top = "Background (unscaled)"
    else:
        bkg_counts_lc = bkg_model_lc.astype(float)
        bkg_err_counts_lc = np.sqrt(np.maximum(bkg_model_lc.astype(float), 0.0))
        bkg_var_for_net = np.maximum(bkg_model_lc.astype(float), 0.0)
        bkg_label_top = "Scaled background (model)"

    lc_width = np.maximum(np.diff(lc_edges), 1e-12)
    lc_center = 0.5 * (lc_edges[:-1] + lc_edges[1:])

    src_err_counts_lc = np.sqrt(np.maximum(src_hist_lc.astype(float), 0.0))

    src_rate_lc = src_hist_lc.astype(float) / lc_width
    src_err_rate_lc = src_err_counts_lc / lc_width
    net_rate_lc = (src_hist_lc.astype(float) - bkg_counts_lc) / lc_width
    net_err_rate_lc = np.sqrt(np.maximum(src_hist_lc.astype(float), 0.0) + bkg_var_for_net) / lc_width

    def _safe_float(key: str, default: float = np.nan) -> float:
        val = txx_result.get(key, default)
        try:
            return float(val)
        except Exception:
            return float(default)

    t100_start = _safe_float("t100_tstart", bb_edges[0])
    t100_stop = _safe_float("t100_tstop", bb_edges[-1])
    t90_start = _safe_float("t90_tstart")
    t90_stop = _safe_float("t90_tstop")
    t90_val = _safe_float("t90")

    t90_err_raw = np.asarray(txx_result.get("t90_err", np.asarray([], dtype=float)), dtype=float).reshape(-1)
    if t90_err_raw.size >= 2 and np.all(np.isfinite(t90_err_raw[:2])):
        t90_err_lo = float(abs(t90_err_raw[0]))
        t90_err_hi = float(abs(t90_err_raw[1]))
    elif t90_err_raw.size == 1 and np.isfinite(t90_err_raw[0]):
        t90_err_lo = float(abs(t90_err_raw[0]))
        t90_err_hi = float(abs(t90_err_raw[0]))
    else:
        t90_err_lo = np.nan
        t90_err_hi = np.nan

    if np.isfinite(t90_val):
        if np.isfinite(t90_err_lo) and np.isfinite(t90_err_hi):
            t90_text = f"{t90_val:.1f} (-{t90_err_lo:.1f}/+{t90_err_hi:.1f}) s"
        else:
            t90_text = f"{t90_val:.1f} s"
    else:
        t90_text = "N/A"

    snr_arr = _fit_len(txx_result.get("bb_block_snr", np.zeros(n_bb)), n_bb, fill=np.nan)
    snr_thr = _safe_float("bb_snr_threshold")

    if srcname is None:
        srcname = None
        meta = getattr(evt, "meta", None)
        if meta is not None and getattr(meta, "object", None):
            srcname = str(meta.object)
        if srcname is None:
            hdr = getattr(evt, "header", None)
            if isinstance(hdr, dict):
                obj_name = hdr.get("OBJECT")
                if obj_name:
                    srcname = str(obj_name)
        if srcname is None:
            evt_path = getattr(evt, "path", "")
            srcname = Path(str(evt_path)).stem if str(evt_path) else "Unknown source"

    x_bb_edges = _to_plot_x(bb_edges)
    x_lc_edges = _to_plot_x(lc_edges)
    x_lc_center = _to_plot_x(lc_center)
    x_t100_start = _safe_float_any(t100_start - tz_plot_abs)
    x_t100_stop = _safe_float_any(t100_stop - tz_plot_abs)
    x_t90_start = _safe_float_any(t90_start - tz_plot_abs)
    x_t90_stop = _safe_float_any(t90_stop - tz_plot_abs)

    fig, (ax_top, ax_mid, ax_blocks) = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.25, 2.1, 0.72]},
    )

    err_src_color = "#2f6ea6"
    err_bkg_color = "#6f6f6f"
    err_net_color = "#d95f5f"

    # 上面板：原始 Source vs Background（柱状 + 误差）
    ax_top.bar(x_lc_center, src_hist_lc.astype(float), width=lc_width * 0.82, alpha=0.72, label="Source counts (real events)", color="steelblue")
    ax_top.bar(x_lc_center, bkg_counts_lc, width=lc_width * 0.82, alpha=0.50, label=bkg_label_top, color="gray")
    ax_top.errorbar(
        x_lc_center,
        src_hist_lc.astype(float),
        yerr=src_err_counts_lc,
        fmt="none",
        ecolor=err_src_color,
        elinewidth=1.0,
        capsize=1.5,
        alpha=0.75,
    )
    ax_top.errorbar(
        x_lc_center,
        bkg_counts_lc,
        yerr=bkg_err_counts_lc,
        fmt="none",
        ecolor=err_bkg_color,
        elinewidth=1.0,
        capsize=1.5,
        alpha=0.72,
    )
    ax_top.set_ylabel("Counts")
    if title is not None:
        top_title = str(title)
    else:
        top_title = f"{srcname} Duration Diagnostic" if srcname is not None else "Duration Diagnostic"
    ax_top.set_title(top_title)
    ax_top.legend(loc="upper right")
    ax_top.grid(alpha=0.3)

    if np.isfinite(bs_lc) and bs_lc > 0.0:
        binsize_text = f"Bin size: {bs_lc:.3f} s"
    else:
        binsize_text = "Bin size: adaptive"

    info_lines = [
        f"Source: {srcname}",
        f"Burst window: [{x_t100_start:.3f}, {x_t100_stop:.3f}] s",
        binsize_text,
    ]
    if np.isfinite(t90_val):
        info_lines.append(f"T90: {t90_text}")
    ax_top.text(
        0.01,
        0.98,
        "\n".join(info_lines),
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    # 中间面板：原始 step 光变 + 误差 + 分块/SNR + T0/T100/T90
    if src_rate_lc.size > 0:
        ax_mid.step(x_lc_edges[:-1], src_rate_lc, where="post", color="steelblue", linewidth=1.1, label="Source rate")
        ax_mid.hlines(src_rate_lc[-1], x_lc_edges[-2], x_lc_edges[-1], colors="steelblue", linewidth=1.1)
        ax_mid.fill_between(x_lc_edges[:-1], 0.0, src_rate_lc, step="post", alpha=0.28, color="steelblue")
    ax_mid.errorbar(
        x_lc_center,
        src_rate_lc,
        yerr=src_err_rate_lc,
        fmt="none",
        ecolor=err_src_color,
        elinewidth=0.95,
        capsize=1.5,
        alpha=0.76,
    )
    ax_mid.errorbar(
        x_lc_center,
        net_rate_lc,
        yerr=net_err_rate_lc,
        fmt="none",
        ecolor=err_net_color,
        elinewidth=0.95,
        capsize=1.5,
        alpha=0.74,
    )

    for e in x_bb_edges:
        ax_mid.axvline(float(e), color="green", linestyle="-", linewidth=1.0, alpha=0.45, label="_nolegend_")

    sig_mask = np.isfinite(snr_arr) & np.isfinite(snr_thr) & (snr_arr > snr_thr)
    for i in range(n_bb):
        if sig_mask[i]:
            ax_mid.axvspan(float(x_bb_edges[i]), float(x_bb_edges[i + 1]), alpha=0.25, color="yellow", label="_nolegend_")

    if np.isfinite(x_t100_start):
        ax_mid.axvline(x_t100_start, color="blue", linestyle="--", linewidth=2.0, label=f"T100 start: {x_t100_start:.1f}s")
    if np.isfinite(x_t100_stop):
        ax_mid.axvline(x_t100_stop, color="blue", linestyle="--", linewidth=2.0, label=f"T100 end: {x_t100_stop:.1f}s")
    if np.isfinite(x_t90_start) and np.isfinite(x_t90_stop) and (x_t90_stop > x_t90_start):
        if np.isfinite(t90_val):
            lab_t90 = f"T90 interval: {t90_text}"
        else:
            lab_t90 = "T90 interval"
        ax_mid.axvspan(x_t90_start, x_t90_stop, alpha=0.20, color="orange", label=lab_t90)

    title_bottom = "Light Curve with Bayesian Blocks and Duration"
    ax_mid.set_ylabel("Rate (counts/s)")
    ax_mid.set_title(title_bottom)
    ax_mid.grid(True, alpha=0.3)

    h_b, l_b = ax_mid.get_legend_handles_labels()
    if h_b:
        seen = set()
        h_show = []
        l_show = []
        for h, l in zip(h_b, l_b):
            if l == "_nolegend_" or l in seen:
                continue
            seen.add(l)
            h_show.append(h)
            l_show.append(l)
        if h_show:
            ax_mid.legend(h_show, l_show, loc="upper right", fontsize=9)

    # 下面板：贝叶斯分块区间色带（独立子图）
    sig_label_added = False
    other_label_added = False
    x_span = max(float(x_bb_edges[-1] - x_bb_edges[0]), 1e-9)
    for i in range(n_bb):
        is_sig = bool(sig_mask[i]) if i < sig_mask.size else False
        if is_sig:
            block_color = "#f4a261"
            label = f"Significant blocks (Li-Ma > {snr_thr:.1f}σ)" if (np.isfinite(snr_thr) and not sig_label_added) else "_nolegend_"
            sig_label_added = sig_label_added or np.isfinite(snr_thr)
            edge_col = "#aa5f1d"
            hatch_pat = "///"
        else:
            block_color = "#8fb9dd" if (i % 2 == 0) else "#bfd9ee"
            label = "Non-significant Bayesian blocks" if not other_label_added else "_nolegend_"
            other_label_added = True
            edge_col = "#2c5374"
            hatch_pat = None
        ax_blocks.axvspan(
            float(x_bb_edges[i]),
            float(x_bb_edges[i + 1]),
            ymin=0.08,
            ymax=0.92,
            facecolor=block_color,
            alpha=0.94,
            edgecolor=edge_col,
            linewidth=1.2,
            hatch=hatch_pat,
            label=label,
        )

        x_left = float(x_bb_edges[i])
        x_right = float(x_bb_edges[i + 1])
        x_mid = 0.5 * (x_left + x_right)
        block_w = max(x_right - x_left, 0.0)
        snr_txt = f"{snr_arr[i]:.1f}σ" if (i < snr_arr.size and np.isfinite(snr_arr[i])) else "--"
        txt = f"B{i+1}\n{snr_txt}"
        rotate_txt = 90 if (block_w < 0.06 * x_span) else 0
        ax_blocks.text(
            x_mid,
            0.5,
            txt,
            ha="center",
            va="center",
            fontsize=7,
            color=("#7a2e00" if is_sig else "#16324a"),
            rotation=rotate_txt,
            alpha=0.95,
            fontweight=("bold" if is_sig else "normal"),
            clip_on=True,
            zorder=5,
        )

    for e in x_bb_edges:
        ax_blocks.axvline(float(e), color="#1f2937", linewidth=1.0, alpha=0.68, label="_nolegend_")

    if np.isfinite(x_t100_start):
        ax_blocks.axvline(x_t100_start, color="blue", linestyle="--", linewidth=1.6, alpha=0.9, label="_nolegend_")
    if np.isfinite(x_t100_stop):
        ax_blocks.axvline(x_t100_stop, color="blue", linestyle="--", linewidth=1.6, alpha=0.9, label="_nolegend_")
    if np.isfinite(x_t90_start) and np.isfinite(x_t90_stop) and (x_t90_stop > x_t90_start):
        ax_blocks.axvspan(x_t90_start, x_t90_stop, ymin=0.0, ymax=1.0, color="orange", alpha=0.16, label="_nolegend_")

    ax_blocks.set_ylim(0.0, 1.0)
    ax_blocks.set_yticks([])
    ax_blocks.set_ylabel("BB")
    ax_blocks.set_title("Bayesian Block Significance (Li-Ma)")
    ax_blocks.grid(axis="x", alpha=0.25, linestyle="--")
    ax_blocks.set_xlabel(x_label)

    h_blk, l_blk = ax_blocks.get_legend_handles_labels()
    if h_blk:
        seen_blk = set()
        h_show_blk = []
        l_show_blk = []
        for h, l in zip(h_blk, l_blk):
            if l == "_nolegend_" or l in seen_blk:
                continue
            seen_blk.add(l)
            h_show_blk.append(h)
            l_show_blk.append(l)
        if h_show_blk:
            ax_blocks.legend(h_show_blk, l_show_blk, loc="upper right", fontsize=8)

    if x_lc_edges.size >= 2:
        x_lo = float(x_lc_edges[0])
        x_hi = float(x_lc_edges[-1])
        ax_mid.set_xlim(x_lo, x_hi)
        ax_blocks.set_xlim(x_lo, x_hi)

    if out is not None:
        fig.savefig(str(out), dpi=150, bbox_inches="tight")

    return fig, (ax_top, ax_mid)


# ----------------------------
# 统一路由：自动判断类型并绘制
# ----------------------------

def plot_ogip(
    obj: Union[Any, PathLike, fits.HDUList],
    **kwargs,
) -> Union[Axes, List[Axes]]:
    """统一入口：
    - 若传入 PhaData -> 调用 plot_spectrum
    - 若传入 LightcurveData -> 调用 plot_lightcurve
    - 若是路径 -> 基于 guess_ogip_kind 读取并路由
    - 若是 HDUList -> 尝试基于扩展名路由（SPECTRUM->PHA，RATE/LC->LC）
    """
    if isinstance(obj, PhaData):
        return plot_spectrum(obj, **kwargs)
    if isinstance(obj, LightcurveData):
        return plot_lightcurve(obj, **kwargs)
    if isinstance(obj, (str, Path)):
        kind = guess_ogip_kind(obj)
        if kind == "pha":
            return plot_spectrum(obj, **kwargs)
        elif kind == "lc":
            return plot_lightcurve(obj, **kwargs)
        else:
            # 不确定类型，尝试读取后按内容再试一次
            try:
                with fits.open(obj) as h:
                    names = {getattr(h[i], "name", "").upper() for i in range(1, len(h))}
                if "SPECTRUM" in names or "PHA" in names:
                    return plot_spectrum(obj, **kwargs)
                return plot_lightcurve(obj, **kwargs)
            except Exception:
                raise ValueError(f"无法识别 OGIP 类型：{obj}")
    if isinstance(obj, fits.HDUList):
        names = {getattr(obj[i], "name", "").upper() for i in range(1, len(obj))}
        if ("SPECTRUM" in names) or ("PHA" in names):
            return plot_spectrum(obj, **kwargs)
        return plot_lightcurve(obj, **kwargs)
    raise TypeError("plot_ogip 仅接受 PhaData/LightcurveData/路径/HDUList")







def plot_xspec_origin(
    plottype: str,srcname: str,
    group_min: int,redshift: float,
    modelname:str,instname:str, 
    outputdir: Optional[Path],
    output_format: str = "png",
    device: str = "cps",
    density: int = 300,
    xlog: bool = True,
    ylog: bool = True,
    **kwargs
) -> Optional[Path]:
    """
    使用自定义命令进行 XSPEC 绘图
    
    参数:
        plottype: 绘图类型
        output_path: 输出路径
        output_format: 输出格式
        device: XSPEC 设备
        density: 转换分辨率
        xlog, ylog: 对数坐标
        **kwargs: 其他 PLT 命令 (如 title="xxx", xlabel="xxx" 等)
        
    返回:
        输出文件路径
    """
    
    
    now_dir = Path.cwd()
    
    if outputdir is None:
        outputdir = now_dir
    else:
        pass
    os.chdir(outputdir)
    # 构建 PS 文件路径
    if redshift is not None:
        if redshift != 0:
            redshiftstr = 'True'
        else:
            redshiftstr = 'False'
        ps_file = str(f"{srcname}_{instname}_{plottype}_{modelname}_redshift{redshiftstr}_groupmin{group_min}.ps")
    else:
        ps_file = str(f"{srcname}_{instname}_{plottype}_{modelname}_redshiftUnknown_groupmin{group_min}.ps")
    xspec.Plot.device = f"{ps_file}/{device}"
    
    # 设置绘图参数
    xspec.Plot.xAxis = "keV"
    xspec.Plot.xLog = xlog
    xspec.Plot.yLog = ylog
    
    # 构建自定义命令
    commands = []
    for key, value in kwargs.items():
        if value is not None:
            if key == "title":
                commands.append(f'title "{value}"')
            elif key == "xlabel":
                commands.append(f'xlabel "{value}"')
            elif key == "ylabel":
                commands.append(f'ylabel "{value}"')
            elif key == "label":
                commands.append(value)  # 完整 label 命令
    
    if commands:
        xspec.Plot.commands = tuple(commands)
    
    # 绘图
    try:
        xspec.Plot(plottype)
    except Exception as e:
        print(f"绘图失败: {e}")
        return None
    
    # 转换格式
    if output_format != "ps":
        try:
            output_file = _convert_ps(ps_file, output_format, density)

            return output_file
        except Exception as e:
            raise RuntimeError(f"转换失败: {e}")
    else:
        return Path(ps_file)
    

    


def _convert_ps(ps_file: str, output_format: str, density: int = 300) -> Path:
    """使用 Pillow 转换 PS 文件"""

    
    ps_path = Path(ps_file)
    output_file = ps_path.with_suffix(f".{output_format}")
    
    # 使用 Pillow 打开 PS 文件 (需要先通过 Ghostscript 转为 PDF/PNG)
    # Pillow 依赖 ghostscript 来处理 PS/eps 文件
    try:
        # 方法 1: 直接用 Pillow 打开 (需要 ghostscript)
        img = Image.open(ps_file)
        
        # 根据输出格式设置质量
        if output_format.lower() in ['jpg', 'jpeg']:
            img.save(output_file, "JPEG", quality=95)
        elif output_format.lower() == 'png':
            img.save(output_file, "PNG")
        elif output_format.lower() == 'gif':
            img.save(output_file, "GIF")
        else:
            img.save(output_file)
        
        return output_file
    except Exception as e:
        # 方法 2: 如果 Pillow 失败，尝试用 ghostscript 直接转换
        raise RuntimeError(f"Pillow 转换失败: {e}")


def _safe_xspec_plot_command(xspec_module, command: str) -> None:
    try:
        xspec_module.Plot(command)
    except Exception:
        if command != "eeufspec":
            raise
        xspec_module.Plot("eeuf")


def _xspec_plot_groups(xspec_module) -> int:
    try:
        return max(1, int(xspec_module.AllData.nSpectra))
    except Exception:
        return 1


def _xspec_plot_arrays(xspec_module, command: str, *, model: bool) -> list[dict[str, np.ndarray]]:
    _safe_xspec_plot_command(xspec_module, command)
    arrays = []
    for group in range(1, _xspec_plot_groups(xspec_module) + 1):
        item = {
            "x": np.asarray(xspec_module.Plot.x(group), dtype=float),
            "y": np.asarray(xspec_module.Plot.y(group), dtype=float),
            "xerr": np.asarray(xspec_module.Plot.xErr(group), dtype=float),
            "yerr": np.asarray(xspec_module.Plot.yErr(group), dtype=float),
        }
        if model:
            try:
                item["model"] = np.asarray(xspec_module.Plot.model(group), dtype=float)
            except Exception:
                item["model"] = np.asarray([], dtype=float)
        arrays.append(item)
    return arrays


def _same_length_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.asarray([], dtype=bool)
    size = min(array.size for array in arrays)
    if size == 0:
        return np.asarray([], dtype=bool)
    return np.ones(size, dtype=bool)


def plotfit(
    srcname: str,
    instname: str,
    group_min: int,
    modelname: str,
    redshift: float = 0.0,
    plottype: str = "ldata_eeufspec_delchi",
    outputdir: Optional[Path | str] = None,
    backend: str = "matplotlib",
    output_format: str = "png",
    density: int = 300,
    **kwargs
) -> Tuple[Optional[Path], Optional[Figure]]:
    """
    XSPEC 光谱拟合结果绘图

    参数:
        srcname: 源名称
        instname: 仪器名称
        group_min: 分组最小计数
        modelname: 模型名称
        redshift: 红移（默认0.0）
        plottype: 保留兼容参数；matplotlib 输出固定为 ldata + eeufspec + delchi 三联图
        outputdir: 输出目录（默认当前目录）
        backend: 绘图后端，'matplotlib'或'xspec'
        output_format: 输出格式，'png'、'jpg'等
        density: 输出分辨率（默认300）
        **kwargs: 其他绘图参数

    返回:
        (输出文件路径, matplotlib Figure对象)

    示例:
        >>> path, fig = plotfit(
        ...     srcname="WXT_test",
        ...     instname="WXT",
        ...     group_min=1,
        ...     modelname="tbabs*ztbabs*cflux*powerlaw",
        ...     redshift=5.47,
        ...     outputdir="output"
        ... )
    """
    import xspec

    if outputdir is None:
        outputdir = Path.cwd()
    else:
        outputdir = Path(outputdir)

    outputdir.mkdir(parents=True, exist_ok=True)

    redshiftstr = "True" if redshift and redshift != 0 else "False"
    plot_tag = "ldata_eeufspec_delchi"
    ps_file = outputdir / f"{srcname}_{instname}_{plot_tag}_{modelname}_redshift{redshiftstr}_groupmin{group_min}.ps"

    if backend == "xspec":
        xspec.Plot.device = f"{ps_file}/cps"
        xspec.Plot.xAxis = "keV"
        xspec.Plot.xLog = True
        xspec.Plot.yLog = True

        commands = []
        if kwargs.get("title"):
            commands.append(f'title "{kwargs["title"]}"')
        if kwargs.get("xlabel"):
            commands.append(f'xlabel "{kwargs["xlabel"]}"')
        if kwargs.get("ylabel"):
            commands.append(f'ylabel "{kwargs["ylabel"]}"')
        if kwargs.get("label"):
            commands.append(kwargs["label"])

        if commands:
            xspec.Plot.commands = tuple(commands)

        try:
            xspec.Plot("ldata", "eeufspec", "delchi")
        except Exception:
            xspec.Plot("ldata", "eeuf", "delchi")

        if output_format != "ps":
            try:
                img = Image.open(str(ps_file))
                output_file = ps_file.with_suffix(f".{output_format}")
                if output_format.lower() in ["jpg", "jpeg"]:
                    img.save(output_file, "JPEG", quality=95)
                else:
                    img.save(output_file, output_format.upper())
                return output_file, None
            except Exception as e:
                raise RuntimeError(f"XSPEC绘图转换失败: {e}")

        return ps_file, None

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=(3, 3, 2), hspace=0.12)
    ax_ldata = fig.add_subplot(grid[0])
    ax_eeuf = fig.add_subplot(grid[1], sharex=ax_ldata)
    ax_delchi = fig.add_subplot(grid[2], sharex=ax_ldata)

    xspec.Plot.device = "/null"
    xspec.Plot.xAxis = "keV"
    xspec.Plot.xLog = True
    xspec.Plot.yLog = True
    ldata = _xspec_plot_arrays(xspec, "ldata", model=True)
    eeuf = _xspec_plot_arrays(xspec, "eeufspec", model=True)
    delchi = _xspec_plot_arrays(xspec, "delchi", model=False)
    n_groups = max(len(ldata), len(eeuf), len(delchi))

    def plot_data_model(ax, groups, *, positive_y: bool) -> None:
        for index, item in enumerate(groups, start=1):
            x_vals = item["x"]
            y_vals = item["y"]
            x_errs = item["xerr"]
            y_errs = item["yerr"]
            model_y = item.get("model", np.asarray([], dtype=float))
            mask = _same_length_mask(x_vals, y_vals, x_errs, y_errs, model_y)
            size = mask.size
            if size == 0:
                continue
            mask &= (
                np.isfinite(x_vals[:size])
                & np.isfinite(y_vals[:size])
                & np.isfinite(x_errs[:size])
                & np.isfinite(y_errs[:size])
                & np.isfinite(model_y[:size])
                & (x_vals[:size] > 0)
            )
            if positive_y:
                mask &= (y_vals[:size] > 0) & (model_y[:size] > 0)
            if not np.any(mask):
                continue
            data_label = f"Data {index}" if n_groups > 1 else "Data"
            model_label = f"Model {index}" if n_groups > 1 else "Model"
            line = ax.errorbar(
                x_vals[:size][mask],
                y_vals[:size][mask],
                xerr=x_errs[:size][mask],
                yerr=y_errs[:size][mask],
                fmt="o",
                markersize=5,
                capsize=3,
                elinewidth=1,
                label=data_label,
            )
            ax.plot(
                x_vals[:size][mask],
                model_y[:size][mask],
                linewidth=2,
                color=line[0].get_color(),
                label=model_label,
            )

    plot_data_model(ax_ldata, ldata, positive_y=True)
    plot_data_model(ax_eeuf, eeuf, positive_y=True)
    for index, item in enumerate(delchi, start=1):
        x_vals = item["x"]
        y_vals = item["y"]
        x_errs = item["xerr"]
        y_errs = item["yerr"]
        mask = _same_length_mask(x_vals, y_vals, x_errs, y_errs)
        size = mask.size
        if size == 0:
            continue
        mask &= (
            np.isfinite(x_vals[:size])
            & np.isfinite(y_vals[:size])
            & np.isfinite(x_errs[:size])
            & np.isfinite(y_errs[:size])
            & (x_vals[:size] > 0)
        )
        if np.any(mask):
            label = f"Data {index}" if n_groups > 1 else None
            ax_delchi.errorbar(
                x_vals[:size][mask],
                y_vals[:size][mask],
                xerr=x_errs[:size][mask],
                yerr=y_errs[:size][mask],
                fmt="o",
                markersize=5,
                capsize=3,
                elinewidth=1,
                label=label,
            )

    title = kwargs.get("title")
    if title is None:
        title = f"{srcname} {instname}: {modelname}"
        if redshift:
            title += f" (z={redshift})"
        try:
            title += f"\ngroup min {group_min} {xspec.Fit.statMethod}={xspec.Fit.statistic:.2f}/{xspec.Fit.dof}"
        except Exception:
            title += f"\ngroup min {group_min}"
    ax_ldata.set_title(title)
    ax_ldata.set_xscale("log")
    ax_ldata.set_yscale("log")
    ax_ldata.set_ylabel(r"ldata [ct s$^{-1}$ keV$^{-1}$]")
    ax_ldata.grid(True, alpha=0.3)
    ax_ldata.tick_params(labelbottom=False)
    ax_ldata.legend(loc="best", fontsize=10)

    ax_eeuf.set_xscale("log")
    ax_eeuf.set_yscale("log")
    ax_eeuf.set_ylabel(r"E$^2$N(E) [keV cm$^{-2}$ s$^{-1}$]")
    ax_eeuf.grid(True, alpha=0.3)
    ax_eeuf.tick_params(labelbottom=False)
    ax_eeuf.legend(loc="best", fontsize=10)

    ax_delchi.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax_delchi.axhline(3, color="gray", linestyle=":", linewidth=1)
    ax_delchi.axhline(-3, color="gray", linestyle=":", linewidth=1)
    ax_delchi.set_xscale("log")
    ax_delchi.set_xlabel("Energy (keV)")
    ax_delchi.set_ylabel("(Data-Model)/Error")
    ax_delchi.grid(True, alpha=0.3)
    if n_groups > 1:
        ax_delchi.legend(loc="best", fontsize=10)

    output_file = None
    if output_format:
        output_file = outputdir / f"{srcname}_{instname}_{plot_tag}_{modelname}_redshift{redshiftstr}_groupmin{group_min}.{output_format}"
        fig.savefig(output_file, dpi=density, bbox_inches="tight")
    return output_file, fig
