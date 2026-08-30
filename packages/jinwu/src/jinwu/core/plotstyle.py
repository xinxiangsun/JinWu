"""jinwu 科学产品的统一绘图样式。

所有绘图模块（plot/products/upperlimit/fit/time 的绘图路径）都应通过本模块
取色、应用 rcParams 并保存图片，避免散落各处的硬编码颜色/字号/dpi。

配色遵循 Okabe-Ito 色觉无障碍调色板（Color Universal Design, 2008）：
在常见色觉缺陷（绿色盲/红色盲/蓝黄色盲）下所有语义色仍保持可分辨，
序列色优先依靠明度与色相的双重差异，不单独依赖红绿对比。

用法::

    from jinwu.core.plotstyle import apply_style, PALETTE, save_figure

    apply_style()
    ax.plot(x, y, color=PALETTE["data"])
    save_figure(fig, output_base, formats=("png", "svg"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl

__all__ = [
    "PALETTE",
    "SERIES_COLORS",
    "apply_style",
    "save_figure",
    "format_log_axis",
]

#: 语义调色板（Okabe-Ito 基准）：绘图代码按语义取色，不直接写十六进制。
PALETTE: dict[str, str] = {
    "data": "#0072B2",        # 蓝 — 源数据点/误差棒
    "model": "#D55E00",       # 朱红 — 折叠模型曲线（蓝/朱红是最稳的色盲安全对比对）
    "background": "#8C8C8C",  # 中灰 — 本底（靠明度区分，不依赖色相）
    "net": "#009E73",         # 蓝绿 — 净信号
    "residual": "#0072B2",    # 残差面板数据点
    "reference": "#404040",   # 参考线（零线/阈值）：中性深灰，不与任何序列混淆
    "band": "#E69F00",        # 橙 — 高亮色带 / T90 区间 / 注释
    "secondary": "#CC79A7",   # 红紫 — 次要序列
    "tertiary": "#56B4E9",    # 天蓝 — 次要序列
    "text": "#1F2937",        # 标注正文
    "muted": "#767676",       # 次要标注（坐标数字等弱化文字）
    # Bayesian block 色带：显著块用橙、非显著块用浅蓝交替——橙/蓝
    # 是色觉缺陷下区分度最高的一对，明度差同时提供冗余编码。
    "block_sig": "#E69F00",        # 显著块填充
    "block_sig_edge": "#9C6200",   # 显著块边框
    "block_sig_text": "#5C3D00",   # 显著块内文字
    "block_alt_a": "#9ECAE1",      # 非显著块交替填充（浅蓝两档）
    "block_alt_b": "#C6DBEF",
    "block_alt_edge": "#2C5985",   # 非显著块边框
    "block_alt_text": "#17375E",   # 非显著块内文字
    "block_edge": "#404040",       # 块分隔线
}

#: 多序列循环取色：Okabe-Ito 六色，按明度交替排列使相邻序列对比最大化。
SERIES_COLORS: tuple[str, ...] = (
    PALETTE["data"],        # 蓝（深）
    PALETTE["band"],        # 橙（浅）
    PALETTE["net"],         # 蓝绿（中）
    PALETTE["tertiary"],    # 天蓝（浅）
    PALETTE["model"],       # 朱红（深）
    PALETTE["secondary"],   # 红紫（中）
)

#: 中文字体回退链（标签/标题含中文时避免豆腐块）。
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans CN",
    "SimHei",
    "Microsoft YaHei",
)

_STYLE_APPLIED = False


def _available_cjk_fonts() -> list[str]:
    """在已安装字体中筛选可用的中文字体（静默失败）。"""
    try:
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        return [name for name in _CJK_FONT_CANDIDATES if name in available]
    except Exception:
        return []


def apply_style() -> None:
    """应用 jinwu 统一 rcParams（每个进程只应用一次，幂等）。"""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    # DejaVu 在前、CJK 在后：matplotlib ≥3.6 按字形逐级回退，
    # 拉丁字母/数学符号（含 U+2212 负号）用 DejaVu，中文字符落到 CJK 字体。
    sans_serif = ["DejaVu Sans", *_available_cjk_fonts()]
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": sans_serif,
            # mathtext 始终用 DejaVu：CJK 字体缺 U+2212（负号）等数学字形，
            # 混排时会导致 10^-9 这类指数出现豆腐块。
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "axes.titlesize": 11,
            "axes.titlepad": 8,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.unicode_minus": False,
        }
    )
    _STYLE_APPLIED = True


def save_figure(
    fig,
    base: str | Path,
    formats: str | Sequence[str] = ("png",),
    dpi: int = 300,
) -> dict[str, Path]:
    """把 figure 保存为一种或多种格式，返回 ``{格式: 路径}``。

    这是全库唯一的图片保存入口：统一 dpi、统一 bbox 处理，
    替代原先 savefig(dpi=150)/savefig(density=300) 等各自为政的调用。
    """
    if isinstance(formats, str):
        formats = (formats,)
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for fmt in formats:
        fmt = fmt.lstrip(".").lower()
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs[fmt] = path
    return outputs


def format_log_axis(ax, axis: str = "x") -> None:
    """让对数轴在小跨度下显示普通数字而非 ``2×10⁰`` 这类刻度标签。"""
    import matplotlib.ticker as mticker
    import numpy as np

    if axis == "x":
        locator = ax.xaxis
    else:
        locator = ax.yaxis

    def _plain(value, _pos):
        return f"{value:g}"

    def _apply():
        try:
            lo, hi = (ax.get_xlim() if axis == "x" else ax.get_ylim())
        except TypeError:  # 空轴
            return
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
            return
        decades = np.log10(hi) - np.log10(lo)
        if decades < 1.0:
            fmt = mticker.FuncFormatter(_plain)
            locator.set_major_formatter(fmt)
            locator.set_minor_formatter(mticker.NullFormatter())

    locator.set_major_locator(mticker.LogLocator(base=10))
    _apply()
    # 绘制完成后（autoscale 生效）再校准一次
    ax.figure.canvas.mpl_connect("draw_event", lambda _event: _apply())

