"""Plot supplied magnitudes in hours, without extinction correction or band conversion."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from jinwu.core.plotstyle import save_figure


def plot_optical_table(csv_path: Path, output_prefix: Path) -> dict[str, Path]:
    """Read time (hr), magnitude/error and limit rows; return exported figure paths."""
    data = pd.read_csv(csv_path)
    for font in font_manager.findSystemFonts(fontpaths=[str(Path.home()/".local/share/fonts/TimesNewRoman")]):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    colors = dict(u="#8064A2", g="#38866E", r="#BD514F", i="#C18A46", z="#4B7195",
                  VT_B="#4B7195", VT_R="#B77B9D", **{"sdss-g": "#38866E", "sdss-r": "#BD514F"})
    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    fig.subplots_adjust(left=.12, right=.97, bottom=.16, top=.95)
    ax.set(xscale="log", xlim=(1.28, 42), ylim=(24.25, 18.05), ylabel="Magnitude",
           xlabel=r"$T_{\rm mid}-T_0$ (hr)")
    ax.set_xticks([2, 3, 5, 10, 20, 30], labels=["2", "3", "5", "10", "20", "30"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
    pre = ax.inset_axes([.61, .135, .345, .32])
    pre.set(xlim=(-46.1, -43.1), ylim=(24.05, 21.8), xticks=[-46, -45, -44], yticks=[22, 23, 24])
    pre.set_title("WFST / pre-trigger", loc="left", fontsize=10, pad=7)
    pre.set_xlabel(r"$T_{\rm mid}-T_0$ (hr)", fontsize=9)
    pre.tick_params(labelsize=9)
    for spine in pre.spines.values():
        spine.set(color=".55", linewidth=.6)
    labels = [(7,-14,"left"), (-7,11,"right"), (-5,11,"right"), (-8,-17,"right"),
              (12,-9,"left"), (9,8,"left"), (-8,12,"right"), (9,8,"left"), (9,-12,"left")]
    offsets = iter(labels)
    for row in data.itertuples(index=False):
        t, mag, band = row.time_mid_minus_t0_hr, row.magnitude, row.filter
        panel, color = (pre if t < 0 else ax), colors[band]
        style = dict(ms=5.2 if t < 0 else 6.4, mfc=color, mec="#252525", mew=.55, zorder=5)
        if row.measurement == "detection":
            panel.errorbar(t, mag, yerr=row.mag_error, fmt="o", ecolor=color, lw=.95, capsize=2.4, **style)
            dx, dy, align = next(offsets)
            name = "LT" if row.telescope == "Liverpool" else row.telescope
            panel.annotate(f"{name}  {band}", (t,mag), xytext=(dx,dy), textcoords="offset points",
                           ha=align, fontsize=9.5, color=color)
        else:
            vt = band.startswith("VT_")
            panel.plot(t, mag, "D" if vt else "o", linestyle="none",
                       fillstyle=("left" if band == "VT_B" else "right") if vt else "full",
                       markerfacecoloralt="none", **style)
            if band != "VT_R":
                panel.annotate("", (t,mag+.43), (t,mag+.10),
                               arrowprops=dict(arrowstyle="->", color="#252525", lw=.8, mutation_scale=7))
            if pd.notna(row.limit_sigma):
                panel.annotate(band+r" ($3\sigma$)", (t,mag), xytext=(0,9),
                               textcoords="offset points", ha="center", fontsize=9, color=color)
    ax.annotate("SVOM/VT\n"+r"VT$_B$, VT$_R$", (1.66,23.5), xytext=(11,2),
                textcoords="offset points", fontsize=10, color=".3", linespacing=1.3)
    for panel in (ax, pre):
        panel.tick_params(which="both", direction="in", top=False, right=False)
    paths = save_figure(fig, output_prefix, formats=("png", "pdf", "svg"), dpi=600)
    paths["preview"] = save_figure(fig, output_prefix.with_name(output_prefix.name+"_preview"), formats=("png",), dpi=140)["png"]
    plt.close(fig)
    return paths


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    plot_optical_table(folder / "observations.csv", folder / "optical_lightcurve_v2")
