# -*- coding: utf-8 -*-
"""
jinwu.core.plot

只负责绘图（plot-only）。本模块提供对 OGIP 数据的基础可视化：
- PHA 能谱图（EP/WXT、Swift/BAT 等）
- 光变曲线（支持单能段与多能段）

与 jinwu.core.file 的数据类配合使用：
- 接受 PhaData / LightcurveData 直接绘图；
- 也可传入路径，内部自动读取并路由；
- 一切筛选/切片/重采样逻辑请放在其他模块完成后再调用这里的绘图。

"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple, Any, Callable, cast, Literal
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.axes import Axes
from matplotlib.figure import Figure

"""在运行期尝试导入实际数据类；若失败，则使用哑类以便 isinstance 不抛错。"""
try:  # runtime import; avoids typing.Any isinstance crash
    from .file import PhaData, LightcurveData  # type: ignore
except Exception:  # pragma: no cover
    class _Dummy:  # minimal placeholder
        pass
    PhaData = _Dummy  # type: ignore
    LightcurveData = _Dummy  # type: ignore

try:
    from .file import readfits, guess_ogip_kind  # type: ignore
except Exception:  # 允许在未安装上游模块时静态检查通过
    def readfits(path: Union[str, Path], kind: Optional[str] = None) -> Any:  # type: ignore
        return path
    def guess_ogip_kind(path: Union[str, Path]) -> str:  # type: ignore
        return "unknown"

PathLike = Union[str, Path]

__all__ = [
    "plot_spectrum",
    "plot_lightcurve",
    "plot_ogip",
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
                # 获取 UTC 时间字符串
                utc_str = timezero_obj.isot if hasattr(timezero_obj, 'isot') else str(timezero_obj)
                xlabel_parts.append(f"{utc_str} (s)")
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

        timezero_obj_lc = None
        if hasattr(lc, 'timezero_obj') and lc.timezero_obj is not None:
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
            elif getattr(lc, 'dt', None) is not None and lc.dt:
                width = float(lc.dt)
            if width is None:
                return None
            if np.ndim(width) == 0:
                return np.full(v.shape[0], float(width), dtype=float)
            width_arr = np.asarray(width, float)
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
                    title = f"{srcname} ({base})" if base else srcname
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
