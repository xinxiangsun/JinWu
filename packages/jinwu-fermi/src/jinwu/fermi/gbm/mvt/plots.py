"""Independent diagnostic figures without process-wide plotting settings."""
import json
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .engine import load_validation_curve


def make_mvt_plots(summary, workspace):
    """Save counts/background, Haar, resampling/stability and validation panels."""
    fig = Figure(figsize=(10, 8), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax_lc, ax_haar, ax_stable, ax_curve = fig.subplots(2, 2).ravel()
    rows = summary["resolutions"]
    last = rows[-1]
    source = summary["source_interval_s"]
    rate = sum(summary["background"][d]["source_average_rate_cps"] for d in summary["selected_detectors"])
    with np.load(last["products"]["lightcurve"], allow_pickle=False) as lc:
        counts, edges = lc["counts"], lc["edges_s"]
        factor = max(1, len(counts) // 600)
        n = len(counts) // factor
        grouped = counts[:n * factor].reshape(n, factor).sum(axis=1)
        dt = last["bin_width_s"] * factor
        ax_lc.stairs(grouped / dt, edges[::factor][:n + 1], label="Observed counts / bin width")
    ax_lc.axhline(rate, color="tab:orange", label="Mean fitted background")
    ax_lc.set(xlabel="Time since trigger (s)", ylabel="Counts s$^{-1}$", xlim=source)
    ax_lc.legend(fontsize=8)
    with np.load(last["products"]["haar"], allow_pickle=False) as haar:
        if "tau" in haar:
            tau, power, error = haar["tau"] * 1000, haar["power"], haar["power_error"]
            good = (power > 0) & np.isfinite(power) & np.isfinite(error)
            ax_haar.errorbar(tau[good], power[good], yerr=error[good], fmt=".", markersize=3, label="Single observed realization")
            ax_haar.set(xscale="log", yscale="log")
    ax_haar.set(xlabel="Haar timescale (ms)", ylabel="Original normalized power",
                title=f"Single-realization state: {last['base']['estimator_status']}")
    for row in rows:
        q = row["resampling"]["percentiles_s"]
        if q is not None:
            p16, median, p84 = np.array(q) * 1000
            ax_stable.errorbar(row["bin_width_s"] * 1000, median,
                              yerr=[[median - p16], [p84 - median]], fmt="o", color="tab:blue")
    ax_stable.set(xscale="log", yscale="log", xlabel="Analysis bin width (ms)", ylabel="Conditional median MVT (ms)",
                  title=f"Stability: {summary['stability_passed']}")
    model = load_validation_curve()
    mvt = 10 ** model["mvt_grid_log"]
    ax_curve.fill_betweenx(mvt, 10 ** model["snr_lower_log"], 10 ** model["snr_upper_log"],
                           alpha=.25, color="tab:orange", label="Paper bootstrap 95% band")
    ax_curve.plot(10 ** model["snr_median_log"], mvt, color="tab:red")
    if "validation" in last:
        v = last["validation"]
        ax_curve.plot(v["snr_mvt"], v["mvt_ms"], "o", color="black", label=summary["interpretation"])
    ax_curve.set(xscale="log", yscale="log", xlabel="Frozen helper SNR$_{MVT}$ (total counts)", ylabel="MVT (ms)")
    ax_curve.legend(fontsize=8)
    fig.suptitle(f"GBM MVT: observer frame, {summary['energy_range_keV']} keV\nDetectors: {', '.join(summary['selected_detectors'])}", fontsize=12)
    path = Path(workspace) / "mvt_diagnostics.png"
    fig.savefig(path, dpi=160)

    dist = Figure(figsize=(7, 4), constrained_layout=True)
    FigureCanvasAgg(dist)
    ax = dist.subplots()
    samples = json.loads(Path(last["products"]["resamples"]).read_text())["samples"]
    values = [s["mvt_s"] * 1000 for s in samples if s["estimator_status"] == "measurement"]
    if values:
        ax.hist(values, bins="auto", histtype="step", color="tab:blue")
    ax.set(xlabel="MVT (ms): measured samples only", ylabel="Resample count",
           title=str(last["resampling"]["state_counts"]))
    path_dist = Path(workspace) / "mvt_resampling.png"
    dist.savefig(path_dist, dpi=160)
    return {"diagnostics_plot": str(path), "resampling_plot": str(path_dist)}
