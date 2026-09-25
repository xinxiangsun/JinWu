"""Static all-sky and GBM diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np

from .models import CoverageResult, Localization, SkyFootprint, SkyMap


def _xy(coords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
    # Negative longitude makes RA increase toward the left, as in standard
    # astronomical all-sky plots.
    ra = coords.ra.wrap_at(180 * u.deg).radian
    return -np.asarray(ra), np.asarray(coords.dec.radian)


def _draw_moc(ax, moc, *, color: str, label: str, linestyle: str = "-", alpha: float = 0.9) -> None:
    # Geometry products retain their converged order (up to 13) for
    # integration, but expanding every boundary cell at that order can turn a
    # static plot into millions of Matplotlib paths.  A display-only degraded
    # copy preserves the visible region without changing any reported MOC.
    try:
        order = int(getattr(moc, "max_order"))
        if order > 7:
            moc = moc.degrade_to_order(7)
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in neighbours_nested")
            warnings.filterwarnings("ignore", message="This method is not stable")
            boundaries = moc.get_boundaries()
    except (AttributeError, ValueError):
        return
    first = True
    for boundary in boundaries:
        x, y = _xy(boundary)
        jumps = np.flatnonzero(np.abs(np.diff(x)) > np.pi)
        starts = np.r_[0, jumps + 1]
        stops = np.r_[jumps + 1, x.size]
        for start, stop in zip(starts, stops):
            if stop - start < 2:
                continue
            ax.plot(
                x[start:stop],
                y[start:stop],
                color=color,
                lw=1.0,
                ls=linestyle,
                alpha=alpha,
                label=label if first else None,
            )
            first = False


def plot_allsky(
    skymap: SkyMap,
    *,
    output: str | Path,
    event_label: str = "GW event",
    footprints: Iterable[SkyFootprint] = (),
    localizations: Iterable[Localization] = (),
    coverages: Iterable[CoverageResult] = (),
    title_suffix: str = "",
    dpi: int = 180,
) -> dict[str, str]:
    """Write an RA-left Mollweide probability map as PNG and PDF."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    coords = skymap.skycoord
    x, y = _xy(coords)
    density = np.asarray(skymap.prob_density_sr, dtype=float)
    positive = density[np.isfinite(density) & (density > 0)]
    floor = float(np.percentile(positive, 0.5)) if positive.size else np.finfo(float).tiny
    color = np.ma.masked_where(density <= 0, np.log10(np.maximum(density, floor)))
    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="mollweide")
    points = ax.scatter(x, y, c=color, s=2.0, cmap="viridis", linewidths=0, rasterized=True)
    cbar = fig.colorbar(points, ax=ax, orientation="horizontal", pad=0.07, fraction=0.05)
    cbar.set_label(r"log$_{10}$ probability density [sr$^{-1}$]")
    for footprint in footprints:
        if footprint.kind == "credible":
            color, linestyle = "#d95f02", ("-" if "50" in footprint.name else "-.")
        else:
            color, linestyle = "#e66101", "--"
        _draw_moc(ax, footprint.moc, color=color, label=footprint.name, linestyle=linestyle)
        if footprint.center is not None and footprint.kind.lower() in {"localization", "position", "candidate"}:
            lx, ly = _xy(footprint.center)
            ax.scatter(
                lx,
                ly,
                marker="*",
                s=90,
                color="#e66101",
                edgecolor="black",
                zorder=8,
                label=f"{footprint.name} center",
            )
    for coverage in coverages:
        if coverage.geometry_footprint is not None:
            _draw_moc(
                ax,
                coverage.geometry_footprint.moc,
                color="#555555",
                label=f"{coverage.instrument} geometric",
                linestyle=":",
                alpha=0.75,
            )
        if coverage.state_footprint is not None:
            _draw_moc(
                ax,
                coverage.state_footprint.moc,
                color="#d95f02",
                label=f"{coverage.instrument} state",
                linestyle="--",
                alpha=0.8,
            )
        if coverage.footprint is not None:
            color_name = "#5e3c99" if coverage.instrument.upper() == "GBM" else "#1b9e77"
            _draw_moc(ax, coverage.footprint.moc, color=color_name, label=f"{coverage.instrument} ({coverage.status})")
    for localization in localizations:
        lx, ly = _xy(localization.center)
        ax.scatter(lx, ly, marker="*", s=100, color="white", edgecolor="black", zorder=8, label=localization.name)
        if localization.radius is not None:
            circle = _circle_points(localization.center, localization.radius)
            cx, cy = _xy(circle)
            jumps = np.flatnonzero(np.abs(np.diff(cx)) > np.pi)
            for start, stop in zip(np.r_[0, jumps + 1], np.r_[jumps + 1, cx.size]):
                if stop - start >= 2:
                    ax.plot(cx[start:stop], cy[start:stop], color="white", lw=1.0, ls=":")
    ax.grid(color="0.7", lw=0.45)
    ax.set_xticklabels(["150°", "120°", "90°", "60°", "30°", "0°", "330°", "300°", "270°", "240°", "210°"])
    ax.set_xlabel("Right ascension")
    ax.set_ylabel("Declination")
    ax.set_title(f"{event_label}{title_suffix}")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return {"allsky_png": str(png), "allsky_pdf": str(pdf)}


def _circle_points(center: SkyCoord, radius: u.Quantity, n: int = 180) -> SkyCoord:
    position = np.linspace(0.0, 360.0, n) * u.deg
    return SkyCoord(
        ra=np.repeat(center.ra, n),
        dec=np.repeat(center.dec, n),
        frame="icrs",
    ).directional_offset_by(position, np.repeat(radius, n))


def plot_gbm_diagnostic(
    skymap: SkyMap,
    coverage: CoverageResult,
    *,
    output: str | Path,
    event_label: str = "GW event",
    dpi: int = 180,
) -> dict[str, str]:
    """Write a compact GBM geometry diagnostic with detector directions."""
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="mollweide")
    coords = skymap.skycoord
    x, y = _xy(coords)
    ax.scatter(x, y, c=np.asarray(skymap.pixel_probability), s=1.2, cmap="Greys", alpha=0.55, rasterized=True)
    if coverage.geometry_footprint is not None:
        _draw_moc(ax, coverage.geometry_footprint.moc, color="#555555", label="Earth-visible geometry", linestyle=":", alpha=0.75)
    if coverage.state_footprint is not None:
        _draw_moc(ax, coverage.state_footprint.moc, color="#d95f02", label="POSHIST state", linestyle="--", alpha=0.8)
    if coverage.footprint is not None:
        _draw_moc(ax, coverage.footprint.moc, color="#5e3c99", label=f"GBM ({coverage.status})", linestyle="-")
    detector_centers = coverage.metadata.get("detector_centers", {})
    for name, value in detector_centers.items():
        try:
            center = SkyCoord(value[0] * u.deg, value[1] * u.deg)
            dx, dy = _xy(center)
            ax.scatter(dx, dy, marker="^" if name.startswith("n") else "s", s=45, label=name)
        except (TypeError, ValueError, IndexError):
            continue
    ax.grid(color="0.7", lw=0.45)
    status_label = (
        "30-orbit prediction"
        if coverage.status == "predicted_30_orbit"
        else coverage.status
    )
    ax.set_title(f"{event_label} — GBM geometry ({status_label})")
    geometry_state = coverage.metadata.get("geometry_state", {})
    if isinstance(geometry_state, dict):
        saa = geometry_state.get("saa")
        good = geometry_state.get("good")
        reference = coverage.reference_time.utc.isot if coverage.reference_time is not None else "unknown"
        ax.text(
            0.99,
            0.02,
            f"SAA={saa}  good={good}\nreference={reference}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.9)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    return {"gbm_png": str(png), "gbm_pdf": str(pdf)}
