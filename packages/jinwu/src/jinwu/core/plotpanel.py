"""jinwu 多对象绘图框架：multi_panel（多面板共享轴）与 overlay（单面板叠加）。

对所有数据类型通用：LightcurveData (kind='lc')、PhaData (kind='pha')、
裸 dict、裸数组 tuple。未来新类型只需在 _extract_series 加分派分支。

与 plot.py 的关系：互补不替代。plot.py 的 plot_lightcurve/plot_spectrum
是单对象精细绘图（含丰富标注）；本模块是多对象组合布局。

约束：
- matplotlib.pyplot 只在函数内部惰性导入，模块顶层绝不 import pyplot；
- 绝不读取 LightcurveData 上已废弃的 .value / .error（会触发 DeprecationWarning）；
- 任何地方都不调用 plt.show()，是否显示交由调用方决定。
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

__all__ = ["LightcurveLike", "SpectrumLike", "PanelSpec", "multi_panel", "overlay"]


# --------------------------------------------------------------------------- #
# 类型协议：只声明"鸭子类型"所需字段，供类型检查器与文档使用。
# runtime_checkable 只校验属性是否存在，不校验类型。
# --------------------------------------------------------------------------- #
@runtime_checkable
class LightcurveLike(Protocol):
    """光变曲线类对象的结构协议（对齐 LightcurveDataBase 字段）。"""

    time: np.ndarray | None
    counts: np.ndarray | None
    rate: np.ndarray | None
    counts_err: np.ndarray | None
    rate_err: np.ndarray | None
    bin_exposure: np.ndarray | None
    bin_width: np.ndarray | float | None
    dt: np.ndarray | float | None
    is_rate: bool
    timezero: float
    telescop: str | None


@runtime_checkable
class SpectrumLike(Protocol):
    """能谱类对象的结构协议（对齐 PhaBase 字段）。"""

    channels: np.ndarray
    counts: np.ndarray
    rate: np.ndarray | None
    stat_err: np.ndarray | None
    exposure: float
    ebounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None


@dataclass(frozen=True, slots=True)
class PanelSpec:
    """multi_panel 中单个面板的绘图规格。"""

    label: str | None = None
    ykind: Literal["auto", "rate", "counts"] = "auto"
    height_ratio: float = 1.0
    yscale: Literal["linear", "log", "symlog"] = "linear"
    color: str | None = None
    ylim: tuple[float, float] | None = None
    plot_kwargs: dict = field(default_factory=dict)


@dataclass(slots=True)
class SeriesData:
    """内部统一的绘图序列：所有数据类型都被规约成 x/y/yerr + 轴标签。"""

    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None
    xlabel: str
    ylabel: str
    timezero: float  # 用于 xmode 的时间对齐


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _is_array_like(obj) -> bool:
    """判断 obj 是否可当作一维（或多维）数值数组处理。"""
    if isinstance(obj, (np.ndarray, list, tuple)):
        try:
            arr = np.asarray(obj)
        except Exception:
            return False
        return arr.ndim >= 1 and arr.dtype.kind in "biufc"
    return False


def _as_float_or_none(arr) -> np.ndarray | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=float)


def _is_raw_array_tuple(items) -> bool:
    """items 是否是 (x, y) 或 (x, y, yerr) 这样的裸数组元组。"""
    if not isinstance(items, (tuple, list)):
        return False
    if len(items) not in (2, 3):
        return False
    return all(_is_array_like(e) for e in items)


def _normalize_items(items) -> list:
    """把用户传入的 items 统一成"数据对象列表"。"""
    # 裸数组元组本身即单个序列
    if _is_raw_array_tuple(items):
        return [items]
    if isinstance(items, (list, tuple)):
        return list(items)
    # LightcurveDataset 之类：有 .data 与 .labels 则展开
    if hasattr(items, "data") and hasattr(items, "labels"):
        return list(items.data)
    return [items]


def _iter_data_objects(item):
    """递归展开面板内容，产出底层数据对象（用于 timezero 扫描）。"""
    if isinstance(item, (list, tuple)) and not _is_raw_array_tuple(item) and len(item) > 0:
        for sub in item:
            yield from _iter_data_objects(sub)
    else:
        yield item


# --------------------------------------------------------------------------- #
# 类型分派：把任意支持的数据对象规约成 SeriesData
# --------------------------------------------------------------------------- #
def _lc_exposure(obj, n: int) -> np.ndarray | None:
    """按 bin_exposure → bin_width → dt 的优先级取每 bin 曝光（秒）。"""
    for attr in ("bin_exposure", "bin_width", "dt"):
        val = getattr(obj, attr, None)
        if val is None:
            continue
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if arr.size == n:
            return arr
    return None


def _series_from_lightcurve(obj, ykind: str) -> SeriesData:
    x = np.asarray(obj.time, dtype=float)
    n = x.size
    counts = getattr(obj, "counts", None)
    rate = getattr(obj, "rate", None)
    counts_err = getattr(obj, "counts_err", None)
    rate_err = getattr(obj, "rate_err", None)
    is_rate = bool(getattr(obj, "is_rate", False))

    # 期望单位：ykind 显式优先，auto 时看 is_rate
    if ykind == "counts" or (ykind == "auto" and not is_rate):
        prefer = "counts"
    else:
        prefer = "rate"

    # 选择可用的原生数组（记录其真实单位 target）
    if prefer == "counts":
        if counts is not None:
            y, yerr, target = np.asarray(counts, float), _as_float_or_none(counts_err), "counts"
        elif rate is not None:
            y, yerr, target = np.asarray(rate, float), _as_float_or_none(rate_err), "rate"
        else:
            raise ValueError("光变对象既无 counts 也无 rate，无法绘图")
    else:
        if rate is not None:
            y, yerr, target = np.asarray(rate, float), _as_float_or_none(rate_err), "rate"
        elif counts is not None:
            y, yerr, target = np.asarray(counts, float), _as_float_or_none(counts_err), "counts"
        else:
            raise ValueError("光变对象既无 rate 也无 counts，无法绘图")

    # 单位不一致时按曝光换算；无曝光信息则退回原生单位
    if target != prefer:
        exposure = _lc_exposure(obj, n)
        if exposure is None:
            prefer = target
        elif prefer == "rate":  # counts → rate
            y = y / exposure
            yerr = yerr / exposure if yerr is not None else None
        else:  # rate → counts
            y = y * exposure
            yerr = yerr * exposure if yerr is not None else None

    ylabel = "Rate (counts/s)" if prefer == "rate" else "Counts"
    timezero = float(getattr(obj, "timezero", 0.0) or 0.0)
    return SeriesData(x, y, yerr, "Time (s)", ylabel, timezero)


def _series_from_pha(obj, ykind: str) -> SeriesData:
    counts = getattr(obj, "counts", None)
    rate = getattr(obj, "rate", None)
    stat_err = getattr(obj, "stat_err", None)
    exposure = getattr(obj, "exposure", None)
    ebounds = getattr(obj, "ebounds", None)
    channels = getattr(obj, "channels", None)

    # x 轴：有能段边界用能量中点，否则用通道号
    if ebounds is not None:
        try:
            elo = np.asarray(ebounds[1], dtype=float)
            ehi = np.asarray(ebounds[2], dtype=float)
            x = 0.5 * (elo + ehi)
            xlabel = "Energy (keV)"
        except Exception:
            x = np.asarray(channels, dtype=float)
            xlabel = "Channel"
    elif channels is not None:
        x = np.asarray(channels, dtype=float)
        xlabel = "Channel"
    else:
        ref = counts if counts is not None else rate
        x = np.arange(0 if ref is None else len(ref), dtype=float)
        xlabel = "Channel"

    exp = float(exposure) if exposure else 0.0
    if ykind == "rate" or (ykind == "auto" and counts is None and rate is not None):
        if rate is not None:
            y = np.asarray(rate, dtype=float)
            ylabel = "Rate (counts/s)"
        elif counts is not None and exp > 0:
            y = np.asarray(counts, dtype=float) / exp
            ylabel = "Rate (counts/s)"
        else:
            y = np.asarray(counts, dtype=float)
            ylabel = "Counts"
    else:
        if counts is not None:
            y = np.asarray(counts, dtype=float)
            ylabel = "Counts"
        elif rate is not None and exp > 0:
            y = np.asarray(rate, dtype=float) * exp
            ylabel = "Counts"
        else:
            y = np.asarray(rate, dtype=float)
            ylabel = "Rate (counts/s)"

    return SeriesData(x, y, _as_float_or_none(stat_err), xlabel, ylabel, 0.0)


def _default_xlabel(xkey: str) -> str:
    return {"time": "Time (s)", "channels": "Channel", "x": "X"}.get(xkey, "X")


def _default_ylabel(ykey: str) -> str:
    return {
        "rate": "Rate (counts/s)",
        "counts": "Counts",
        "flux": "Flux",
        "y": "Y",
        "value": "Y",
    }.get(ykey, "Y")


def _series_from_mapping(obj: Mapping, ykind: str) -> SeriesData:
    x = y = None
    xkey = ykey = ""
    for k in ("time", "x", "channels"):
        if obj.get(k) is not None:
            x, xkey = np.asarray(obj[k], dtype=float), k
            break
    if x is None:
        raise TypeError("dict 缺少 x 数据键（time/x/channels）")
    for k in ("rate", "counts", "flux", "y", "value"):
        if obj.get(k) is not None:
            y, ykey = np.asarray(obj[k], dtype=float), k
            break
    if y is None:
        raise TypeError("dict 缺少 y 数据键（rate/counts/flux/y/value）")

    yerr = None
    for k in ("rate_err", "counts_err", "yerr", "error", "flux_err"):
        if obj.get(k) is not None:
            yerr = np.asarray(obj[k], dtype=float)
            break

    xlabel = obj.get("xlabel") or _default_xlabel(xkey)
    ylabel = obj.get("ylabel") or _default_ylabel(ykey)
    timezero = float(obj.get("timezero", 0.0) or 0.0)
    return SeriesData(x, y, yerr, xlabel, ylabel, timezero)


def _series_from_tuple(obj) -> SeriesData:
    x = np.asarray(obj[0], dtype=float)
    y = np.asarray(obj[1], dtype=float)
    yerr = np.asarray(obj[2], dtype=float) if len(obj) == 3 else None
    return SeriesData(x, y, yerr, "X", "Y", 0.0)


def _extract_series(obj, ykind: str = "auto") -> SeriesData:
    """把任意支持的数据对象规约成 SeriesData（类型分派入口）。

    支持：LightcurveData(kind='lc') / PhaData(kind='pha') / Mapping /
    裸数组 tuple ``(x, y)`` 或 ``(x, y, yerr)``。新类型在此加分派分支即可。
    """
    kind = getattr(obj, "kind", None)

    if kind == "lc" or (hasattr(obj, "time") and getattr(obj, "time", None) is not None):
        return _series_from_lightcurve(obj, ykind)
    if kind == "pha" or hasattr(obj, "channels"):
        return _series_from_pha(obj, ykind)
    if isinstance(obj, Mapping):
        return _series_from_mapping(obj, ykind)
    if _is_raw_array_tuple(obj):
        return _series_from_tuple(obj)

    raise TypeError(
        f"无法从 {type(obj).__name__} 提取绘图序列。可接受类型："
        "LightcurveData(kind='lc')、PhaData(kind='pha')、dict（含 time/x/channels + "
        "rate/counts/flux/y/value）、或 (x, y[, yerr]) 数组元组。"
    )


def _resolve_time_offset(series: SeriesData, xmode: str, reference_tz: float) -> float:
    """按 xmode 计算 x 轴平移量。"""
    if xmode == "raw":
        return 0.0
    if xmode == "timezero_relative":
        return series.timezero
    if xmode == "absolute":
        if series.timezero == 0:
            raise ValueError(
                "xmode='absolute' 需要非零 timezero 才能做绝对时间对齐；"
                "该序列 timezero=0，请改用 'raw' 或 'timezero_relative'。"
            )
        return series.timezero - reference_tz
    raise ValueError(f"未知 xmode: {xmode!r}（可选 'raw'/'timezero_relative'/'absolute'）")


# --------------------------------------------------------------------------- #
# 绘制单条序列
# --------------------------------------------------------------------------- #
def _draw_series(ax, series: SeriesData, offset: float, color: str | None,
                 label: str | None, plot_kwargs: dict | None) -> None:
    x = np.asarray(series.x, dtype=float) - offset
    y = np.asarray(series.y, dtype=float)
    yerr = series.yerr
    kw = dict(plot_kwargs) if plot_kwargs else {}
    if x.size > 5000:
        # 大数据量：折线 + 半透明误差带，避免 errorbar 逐点绘制卡顿
        ax.plot(x, y, color=color, label=label, **kw)
        if yerr is not None:
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.3, linewidth=0)
    else:
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=color, markersize=2, label=label, **kw)


# --------------------------------------------------------------------------- #
# 公共 API：overlay
# --------------------------------------------------------------------------- #
def overlay(items, *, ax=None, labels=None, xmode="raw", ykind="auto", colors=None,
            yscale=None, xlabel=None, ylabel=None, title=None, show_legend=True,
            **plot_kwargs):
    """把多个数据对象叠加绘制到同一个 axes。

    返回 ``(fig, ax)``。ax 为 None 时新建 figure。
    """
    import matplotlib.pyplot as plt
    from .plotstyle import apply_style, SERIES_COLORS

    apply_style()

    seq = _normalize_items(items)
    if not seq:
        raise ValueError("overlay: items 为空")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    series_list = [_extract_series(it, ykind=ykind) for it in seq]
    reference_tz = series_list[0].timezero

    for i, series in enumerate(series_list):
        offset = _resolve_time_offset(series, xmode, reference_tz)
        color = colors[i] if (colors is not None and i < len(colors)) else None
        color = color or SERIES_COLORS[i % len(SERIES_COLORS)]
        label = labels[i] if (labels is not None and i < len(labels)) else None
        _draw_series(ax, series, offset, color, label, plot_kwargs)

    ax.set_xlabel(xlabel or series_list[0].xlabel)
    ax.set_ylabel(ylabel or series_list[0].ylabel)
    if title:
        ax.set_title(title)
    if yscale:
        ax.set_yscale(yscale)
    if show_legend and labels is not None and any(labels):
        ax.legend(loc="best", fontsize=9)
    return fig, ax


# --------------------------------------------------------------------------- #
# 公共 API：multi_panel
# --------------------------------------------------------------------------- #
def multi_panel(items, *, labels=None, panels=None, sharex=True, xmode="raw",
                figsize=None, height_ratios=None, xlabel=None, title=None,
                xscale=None, out=None, formats=("png",), show_legend=True):
    """每个数据对象独占一个面板，纵向排列并共享 x 轴。

    返回 ``(fig, list[Axes])``。不调用 plt.show()；``out`` 非空时保存图片。
    """
    import matplotlib.pyplot as plt
    from .plotstyle import apply_style, save_figure, SERIES_COLORS, format_log_axis

    apply_style()

    # LightcurveDataset 之类：展开为对象列表并沿用其 labels
    if hasattr(items, "data") and hasattr(items, "labels") and not isinstance(items, (list, tuple)):
        if labels is None:
            labels = list(getattr(items, "labels", []) or [])
        items = list(items.data)
    items = list(items)
    n = len(items)
    if n == 0:
        raise ValueError("multi_panel: items 为空")

    if panels is not None:
        panels = [p if p is not None else PanelSpec() for p in panels]
        if len(panels) != n:
            raise ValueError(f"panels 数量（{len(panels)}）须与 items 数量（{n}）一致")
    else:
        panels = [PanelSpec() for _ in range(n)]

    if height_ratios is None:
        height_ratios = [p.height_ratio for p in panels]
    if figsize is None:
        figsize = (10, 3 * n)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(n, 1, height_ratios=height_ratios, hspace=0.08)

    axes = []
    xlabels: list[str] = []
    for i, item in enumerate(items):
        ax = fig.add_subplot(grid[i], sharex=axes[0] if (sharex and i > 0) else None)
        axes.append(ax)
        spec = panels[i]
        label = labels[i] if (labels is not None and i < len(labels)) else spec.label
        color = spec.color or SERIES_COLORS[i % len(SERIES_COLORS)]

        is_overlay_panel = (
            isinstance(item, (list, tuple)) and not _is_raw_array_tuple(item) and len(item) > 0
        )
        if is_overlay_panel:
            # 面板内叠加多个对象：委托 overlay，共享本面板颜色由 overlay 循环决定
            panel_colors = [color] if spec.color else None
            overlay(item, ax=ax, labels=None, xmode=xmode, colors=panel_colors,
                    show_legend=show_legend, **spec.plot_kwargs)
            xlabels.append(_extract_series(next(_iter_data_objects(item)), ykind=spec.ykind).xlabel)
        else:
            series = _extract_series(item, ykind=spec.ykind)
            reference_tz = series.timezero
            offset = _resolve_time_offset(series, xmode, reference_tz)
            _draw_series(ax, series, offset, color, label, spec.plot_kwargs)
            ax.set_ylabel(series.ylabel)
            xlabels.append(series.xlabel)

        if spec.yscale:
            ax.set_yscale(spec.yscale)
            if spec.yscale in ("log", "symlog"):
                format_log_axis(ax, "y")
        if spec.ylim is not None:
            ax.set_ylim(*spec.ylim)

    # 共享 x 轴清理：非末面板隐藏 x 刻度标签，仅末面板显示 x 轴标签
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    axes[-1].set_xlabel(xlabel or xlabels[-1])

    if xscale:
        for ax in axes:
            ax.set_xscale(xscale)
            if xscale in ("log", "symlog"):
                format_log_axis(ax, "x")

    # 跨仪器时间对齐告警：raw + sharex 时若 timezero 不一致会导致时间错位
    if sharex and xmode == "raw":
        timezeros = set()
        for i, item in enumerate(items):
            for obj in _iter_data_objects(item):
                try:
                    timezeros.add(round(_extract_series(obj, ykind=panels[i].ykind).timezero, 9))
                except Exception:
                    continue
        if len(timezeros) > 1:
            warnings.warn(
                "各面板 timezero 不一致，共享 x 轴可能导致时间错位。"
                "建议使用 xmode='timezero_relative'。",
                UserWarning,
                stacklevel=2,
            )

    if title:
        fig.suptitle(title)
    if out:
        save_figure(fig, out, formats=formats)
    return fig, list(axes)
