"""Mission-independent pipeline artifacts, light curves, flux curves, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

__all__ = [
    "ExternalRunArtifacts",
    "FitProductSet",
    "FluxPoint",
    "FluxCurveResult",
    "NetLightcurve",
    "collect_xselect_artifacts",
    "build_artifact_index",
    "collect_runtime_environment",
    "build_net_lightcurve",
    "save_net_lightcurve",
    "load_net_lightcurve",
    "plot_net_lightcurve",
    "build_flux_curve",
    "build_quicklook_flux_curve",
    "save_flux_curve",
    "save_xspec_session",
    "render_observation_summary",
    "render_wechat_fit_messages",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


@dataclass(frozen=True, slots=True)
class ExternalRunArtifacts:
    """Files and execution metadata produced by one external command."""

    products: dict[str, Path]
    command_file: Path | None
    log_file: Path | None
    command: tuple[str, ...] = ()
    cwd: Path | None = None
    checksums: dict[str, str] = field(default_factory=dict)

    def outputs(self, prefix: str = "") -> dict[str, str]:
        stem = f"{prefix}_" if prefix else ""
        outputs = {f"{stem}{key}": str(value) for key, value in self.products.items()}
        if self.command_file is not None:
            outputs[f"{stem}xco"] = str(self.command_file)
        if self.log_file is not None:
            outputs[f"{stem}log"] = str(self.log_file)
        return outputs


@dataclass(frozen=True, slots=True)
class FitProductSet:
    """Replayable files for one spectral fit."""

    result_json: Path
    report_txt: Path
    xspec_log: Path
    xcm: Path
    plots: tuple[Path, ...]
    replay_cwd: Path


@dataclass(frozen=True, slots=True)
class NetLightcurve:
    """Strictly aligned source-minus-background light curve.

    ``time`` stores bin centers, following OGIP ``TIMEPIXR=0.5`` semantics.
    """

    time: np.ndarray
    bin_width: np.ndarray
    fractional_exposure: np.ndarray
    source_rate: np.ndarray
    source_error: np.ndarray
    background_rate: np.ndarray
    background_error: np.ndarray
    net_rate: np.ndarray
    net_error: np.ndarray
    alpha: float
    timezero: float


@dataclass(frozen=True, slots=True)
class FluxPoint:
    """One model-derived or quicklook flux measurement."""

    label: str
    start: float
    stop: float
    time: float
    time_error: float
    flux: float
    error_low: float | None
    error_high: float | None
    energy_min_keV: float
    energy_max_keV: float
    kind: str
    is_detection: bool
    significance: float | None = None
    model: str | None = None
    model_key: str | None = None
    model_expression: str | None = None


@dataclass(frozen=True, slots=True)
class FluxCurveResult:
    """Science and fixed-shape quicklook flux points."""

    science_points: tuple[FluxPoint, ...]
    aggregate_points: tuple[FluxPoint, ...] = ()
    quicklook_points: tuple[FluxPoint, ...] = ()
    status: str = "ok"


def collect_xselect_artifacts(result: Any) -> ExternalRunArtifacts:
    """Collect all products and replay files from an XSelectRunResult."""

    selected = result.outputs.selected()
    products = {str(key): Path(value) for key, value in selected.items()}
    command_file = Path(result.outputs.command_file)
    log_file = Path(result.outputs.log_file)
    files = list(products.values()) + [command_file, log_file]
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing XSELECT artifacts: {', '.join(map(str, missing))}")
    checksums = {path.name: _sha256(path) for path in files}
    return ExternalRunArtifacts(
        products=products,
        command_file=command_file,
        log_file=log_file,
        command=tuple(getattr(result, "commands", ())),
        cwd=command_file.parent,
        checksums=checksums,
    )


def build_artifact_index(
    stage_outputs: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], ...]:
    """Build a checksum index for every existing file exposed by pipeline stages."""

    artifacts = []
    seen: set[Path] = set()
    for stage, outputs in stage_outputs.items():
        for role, value in outputs.items():
            path = Path(value).expanduser().resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            artifacts.append(
                {
                    "stage": str(stage),
                    "role": str(role),
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return tuple(artifacts)


def collect_runtime_environment() -> dict[str, Any]:
    """Collect software and external-ecosystem versions without requiring them."""

    try:
        from importlib.metadata import version

        jinwu_version = version("jinwu")
    except Exception:
        jinwu_version = None
    environment: dict[str, Any] = {
        "jinwu_version": jinwu_version,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "HEADAS": os.environ.get("HEADAS"),
        "CALDB": os.environ.get("CALDB"),
        "CALDBCONFIG": os.environ.get("CALDBCONFIG"),
        "heasoft_version": os.environ.get("HEADAS_VERSION"),
        "xspec_version": None,
    }
    ftversion = shutil.which("ftversion")
    if ftversion is not None:
        try:
            completed = subprocess.run(
                [ftversion], capture_output=True, text=True, timeout=10, check=False
            )
            version_text = (completed.stdout or completed.stderr).strip()
            if completed.returncode == 0 and version_text:
                environment["heasoft_version"] = version_text
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import xspec

        environment["xspec_version"] = getattr(xspec.Xset, "version", None)
    except Exception:
        pass
    return environment


def _lightcurve_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with fits.open(path, memmap=False) as hdul:
        hdu = next(
            item for item in hdul
            if getattr(item, "data", None) is not None
            and "TIME" in (getattr(item.data, "names", None) or ())
        )
        names = {name.upper(): name for name in hdu.data.names}
        time = np.asarray(hdu.data[names["TIME"]], dtype=float)
        timedel = hdu.header.get("TIMEDEL")
        if timedel is None:
            timedel = np.median(np.diff(time)) if len(time) > 1 else 1.0
        width = np.full(len(time), float(timedel), dtype=float)
        timepixr = float(hdu.header.get("TIMEPIXR", 0.5))
        time = time + (0.5 - timepixr) * width
        fractional_exposure = (
            np.asarray(hdu.data[names["FRACEXP"]], dtype=float)
            if "FRACEXP" in names
            else np.ones(len(time), dtype=float)
        )
        if np.any(~np.isfinite(fractional_exposure)) or np.any(
            (fractional_exposure < 0.0) | (fractional_exposure > 1.0)
        ):
            raise ValueError(f"Light curve has invalid FRACEXP values: {path}")
        effective_exposure = width * fractional_exposure
        if "RATE" in names:
            rate = np.asarray(hdu.data[names["RATE"]], dtype=float)
            error = (
                np.asarray(hdu.data[names["ERROR"]], dtype=float)
                if "ERROR" in names
                else np.divide(
                    np.sqrt(np.maximum(rate * effective_exposure, 0.0)),
                    effective_exposure,
                    out=np.full(len(time), np.nan),
                    where=effective_exposure > 0,
                )
            )
        elif "COUNTS" in names:
            counts = np.asarray(hdu.data[names["COUNTS"]], dtype=float)
            rate = np.divide(
                counts,
                effective_exposure,
                out=np.full(len(time), np.nan),
                where=effective_exposure > 0,
            )
            count_error = (
                np.asarray(hdu.data[names["ERROR"]], dtype=float)
                if "ERROR" in names else np.sqrt(np.maximum(counts, 0.0))
            )
            error = np.divide(
                count_error,
                effective_exposure,
                out=np.full(len(time), np.nan),
                where=effective_exposure > 0,
            )
        else:
            raise ValueError(f"Light curve has no RATE or COUNTS column: {path}")
        timezero = float(hdu.header.get("TIMEZERO", 0.0))
    return time, width, fractional_exposure, rate, error, timezero


def build_net_lightcurve(source: str | Path, background: str | Path, *, alpha: float) -> NetLightcurve:
    """Build a net rate curve without interpolating incompatible bins."""

    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("alpha must be finite and positive")
    st, sw, sf, sr, se, stz = _lightcurve_arrays(Path(source))
    bt, bw, bf, br, be, btz = _lightcurve_arrays(Path(background))
    if st.shape != bt.shape or not np.allclose(st + stz, bt + btz, rtol=0, atol=1e-7):
        raise ValueError("Source and background light curves do not have aligned time bins")
    if sw.shape != bw.shape or not np.allclose(sw, bw, rtol=1e-7, atol=1e-10):
        raise ValueError("Source and background light curves have different bin widths")
    if sf.shape != bf.shape or not np.allclose(sf, bf, rtol=1e-7, atol=1e-10):
        raise ValueError("Source and background light curves have different fractional exposures")
    a = float(alpha)
    return NetLightcurve(
        time=st,
        bin_width=sw,
        fractional_exposure=sf,
        source_rate=sr,
        source_error=se,
        background_rate=br,
        background_error=be,
        net_rate=sr - a * br,
        net_error=np.sqrt(se**2 + (a * be) ** 2),
        alpha=a,
        timezero=stz,
    )


def save_net_lightcurve(curve: NetLightcurve, output_base: str | Path) -> dict[str, Path]:
    """Save the canonical net light curve as FITS, ECSV, and CSV."""

    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    table = Table(
        {
            "time": curve.time,
            "bin_width": curve.bin_width,
            "fractional_exposure": curve.fractional_exposure,
            "source_rate": curve.source_rate,
            "source_error": curve.source_error,
            "background_rate": curve.background_rate,
            "background_error": curve.background_error,
            "net_rate": curve.net_rate,
            "net_error": curve.net_error,
        }
    )
    table.meta.update(
        {
            "alpha": curve.alpha,
            "timezero": curve.timezero,
            "TIMEPIXR": 0.5,
        }
    )
    table["time"].unit = u.s
    table["bin_width"].unit = u.s
    for name in (
        "source_rate", "source_error", "background_rate", "background_error",
        "net_rate", "net_error",
    ):
        table[name].unit = u.ct / u.s
    fits_path = base.with_suffix(".fits")
    ecsv_path = base.with_suffix(".ecsv")
    csv_path = base.with_suffix(".csv")
    table.write(fits_path, format="fits", overwrite=True)
    table.write(ecsv_path, format="ascii.ecsv", overwrite=True)
    table.write(csv_path, format="ascii.csv", overwrite=True)
    return {"net_fits": fits_path, "net_ecsv": ecsv_path, "net_csv": csv_path}


def load_net_lightcurve(path: str | Path) -> NetLightcurve:
    """Load the canonical ECSV/FITS net-lightcurve representation."""

    table = Table.read(path)
    return NetLightcurve(
        time=np.asarray(table["time"], dtype=float),
        bin_width=np.asarray(table["bin_width"], dtype=float),
        fractional_exposure=np.asarray(table["fractional_exposure"], dtype=float),
        source_rate=np.asarray(table["source_rate"], dtype=float),
        source_error=np.asarray(table["source_error"], dtype=float),
        background_rate=np.asarray(table["background_rate"], dtype=float),
        background_error=np.asarray(table["background_error"], dtype=float),
        net_rate=np.asarray(table["net_rate"], dtype=float),
        net_error=np.asarray(table["net_error"], dtype=float),
        alpha=float(table.meta["alpha"]),
        timezero=float(table.meta.get("timezero", 0.0)),
    )


def plot_net_lightcurve(
    curve: NetLightcurve,
    output_base: str | Path,
    *,
    title: str,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Plot source, scaled background, and net count-rate curves."""

    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    mission_time = curve.time + curve.timezero
    top.errorbar(
        mission_time, curve.source_rate, yerr=curve.source_error, fmt=".", label="source"
    )
    top.errorbar(
        mission_time, curve.alpha * curve.background_rate,
        yerr=curve.alpha * curve.background_error, fmt=".", label="scaled background",
    )
    top.set_ylabel("Rate [count s$^{-1}$]")
    top.legend()
    top.grid(alpha=0.25)
    bottom.axhline(0.0, color="0.5", lw=1)
    bottom.errorbar(
        mission_time, curve.net_rate, yerr=curve.net_error, fmt=".", color="black"
    )
    bottom.set_xlabel("Mission time [s]")
    bottom.set_ylabel("Net rate [count s$^{-1}$]")
    bottom.grid(alpha=0.25)
    fig.suptitle(title)
    base = Path(output_base)
    paths = []
    for extension in formats:
        path = base.with_suffix(f".{extension}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return tuple(paths)


def _parameter(parameters: Mapping[str, Any], suffix: str) -> Mapping[str, Any] | None:
    target = suffix.lower()
    for key, value in parameters.items():
        if str(key).lower().endswith(target):
            return value if isinstance(value, Mapping) else None
    return None


def _flux_point(
    label: str,
    fit: Mapping[str, Any],
    start: float,
    stop: float,
    *,
    kind: str,
    energy_range_keV: tuple[float, float],
    significance: float | None = None,
) -> FluxPoint:
    parameters = fit.get("parameters") or {}
    cflux = _parameter(parameters, "cflux.lg10flux")
    if cflux is None:
        raise ValueError(f"Fit {label} does not contain cflux.lg10Flux")
    lg_flux = float(cflux["value"])
    flux = 10.0**lg_flux
    error_low = cflux.get("error_lo")
    error_high = cflux.get("error_hi")
    lo = flux - 10.0 ** (lg_flux - float(error_low)) if error_low is not None else None
    hi = 10.0 ** (lg_flux + float(error_high)) - flux if error_high is not None else None
    return FluxPoint(
        label=label,
        start=float(start),
        stop=float(stop),
        time=0.5 * (float(start) + float(stop)),
        time_error=0.5 * (float(stop) - float(start)),
        flux=flux,
        error_low=lo,
        error_high=hi,
        energy_min_keV=float(energy_range_keV[0]),
        energy_max_keV=float(energy_range_keV[1]),
        kind=kind,
        is_detection=True,
        significance=significance,
        model=str(fit.get("model")) if fit.get("model") is not None else None,
        model_key=str(fit.get("model_key")) if fit.get("model_key") is not None else None,
        model_expression=(
            str(fit.get("model")) if fit.get("model") is not None else None
        ),
    )


def build_flux_curve(
    fits: Mapping[str, Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    duration: Mapping[str, Any],
    *,
    energy_range_keV: tuple[float, float],
    include_t90_aggregate: bool = True,
) -> FluxCurveResult:
    """Construct the model-derived flux curve from time-resolved cflux fits."""

    science = []
    for segment in segments:
        label = f"bb{int(segment['index']):03d}"
        if label not in fits:
            continue
        science.append(
            _flux_point(
                label, fits[label], float(segment["start"]), float(segment["stop"]),
                kind="time_resolved", energy_range_keV=energy_range_keV,
                significance=float(segment["significance"]),
            )
        )
    aggregate = []
    if include_t90_aggregate and "t90" in fits:
        aggregate.append(
            _flux_point(
                "t90", fits["t90"], float(duration["t90_tstart"]),
                float(duration["t90_tstop"]), kind="aggregate",
                energy_range_keV=energy_range_keV,
            )
        )
    status = "ok" if science else "insufficient_time_resolution"
    return FluxCurveResult(tuple(science), tuple(aggregate), status=status)


def build_quicklook_flux_curve(
    curve: NetLightcurve,
    *,
    t90_flux: float,
    t90_start: float,
    t90_stop: float,
    energy_range_keV: tuple[float, float],
    model_key: str | None = None,
    model_expression: str | None = None,
) -> tuple[FluxPoint, ...]:
    """Apply a fixed T90 spectral conversion while retaining signed bins."""

    centers = curve.time + curve.timezero
    bin_start = centers - 0.5 * curve.bin_width
    bin_stop = centers + 0.5 * curve.bin_width
    overlap = np.maximum(
        0.0,
        np.minimum(bin_stop, float(t90_stop))
        - np.maximum(bin_start, float(t90_start)),
    )
    mask = overlap > 0
    weights = overlap[mask] * curve.fractional_exposure[mask]
    if not np.any(mask) or np.sum(weights) <= 0:
        raise ValueError("T90 does not overlap the quicklook light curve")
    t90_rate = float(np.sum(curve.net_rate[mask] * weights) / np.sum(weights))
    if not math.isfinite(t90_rate) or t90_rate <= 0:
        raise ValueError("T90 net rate must be positive for count-to-flux conversion")
    factor = float(t90_flux) / t90_rate
    points = []
    for index, (center, start, stop, rate, error) in enumerate(
        zip(centers, bin_start, bin_stop, curve.net_rate, curve.net_error)
    ):
        flux = float(rate * factor)
        sigma = float(abs(error * factor))
        points.append(
            FluxPoint(
                label=f"quicklook{index:05d}", start=float(start), stop=float(stop),
                time=float(center), time_error=float(0.5 * (stop - start)),
                flux=flux, error_low=sigma, error_high=sigma,
                energy_min_keV=float(energy_range_keV[0]),
                energy_max_keV=float(energy_range_keV[1]), kind="fixed_shape_quicklook",
                is_detection=bool(flux > 0 and flux / sigma >= 3.0) if sigma > 0 else False,
                model_key=model_key,
                model_expression=model_expression,
                model=model_expression,
            )
        )
    return tuple(points)


def save_flux_curve(
    result: FluxCurveResult,
    output_base: str | Path,
    *,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
    title: str = "Flux curve",
) -> dict[str, Path]:
    """Save machine-readable points and science/quicklook plots."""

    import matplotlib.pyplot as plt

    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    points = result.science_points + result.aggregate_points + result.quicklook_points
    table = Table(
        {
            "label": [point.label for point in points],
            "start": [point.start for point in points],
            "stop": [point.stop for point in points],
            "time": [point.time for point in points],
            "time_error": [point.time_error for point in points],
            "flux": [point.flux for point in points],
            "error_low": [
                np.nan if point.error_low is None else point.error_low for point in points
            ],
            "error_high": [
                np.nan if point.error_high is None else point.error_high for point in points
            ],
            "energy_min_keV": [point.energy_min_keV for point in points],
            "energy_max_keV": [point.energy_max_keV for point in points],
            "kind": [point.kind for point in points],
            "is_detection": [point.is_detection for point in points],
            "model_key": [point.model_key for point in points],
            "model_expression": [point.model_expression for point in points],
        }
    )
    for name in ("start", "stop", "time", "time_error"):
        table[name].unit = u.s
    for name in ("flux", "error_low", "error_high"):
        table[name].unit = u.erg / (u.cm**2 * u.s)
    table["energy_min_keV"].unit = u.keV
    table["energy_max_keV"].unit = u.keV
    ecsv = base.with_suffix(".ecsv")
    csv = base.with_suffix(".csv")
    manifest = base.with_suffix(".json")
    table.write(ecsv, format="ascii.ecsv", overwrite=True)
    table.write(csv, format="ascii.csv", overwrite=True)
    _write_json(manifest, result)
    outputs = {"ecsv": ecsv, "csv": csv, "manifest": manifest}
    if not formats:
        return outputs

    fig, (science_ax, quicklook_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    science = result.science_points or result.aggregate_points
    if science:
        x = np.asarray([point.time for point in science])
        y = np.asarray([point.flux for point in science])
        lo = np.asarray([point.error_low or np.nan for point in science])
        hi = np.asarray([point.error_high or np.nan for point in science])
        science_ax.errorbar(x, y, xerr=[point.time_error for point in science], yerr=(lo, hi), fmt="o")
        science_ax.set_yscale("log")
    science_ax.set_ylabel("Unabsorbed flux\n[erg cm$^{-2}$ s$^{-1}$]")
    science_ax.grid(alpha=0.25)
    quick = result.quicklook_points
    if quick:
        quicklook_ax.errorbar(
            [point.time for point in quick], [point.flux for point in quick],
            xerr=[point.time_error for point in quick],
            yerr=[point.error_high for point in quick], fmt=".", color="0.25",
        )
        quicklook_ax.axhline(0.0, color="0.5", lw=1)
    quicklook_ax.set_ylabel("Fixed-shape quicklook flux")
    quicklook_ax.set_xlabel("Mission time [s]")
    quicklook_ax.grid(alpha=0.25)
    fig.suptitle(title)
    for extension in formats:
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs[f"plot_{extension}"] = path
    plt.close(fig)
    return outputs


def save_xspec_session(
    xspec: Any,
    *,
    output_dir: str | Path,
    label: str,
    result: Mapping[str, Any],
    report_txt: str | Path,
    transcript: str,
    plots: Sequence[str | Path],
    input_paths: Sequence[str | Path] = (),
) -> FitProductSet:
    """Persist a replayable XSPEC command file and structured fit products."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    xcm = output / f"{label}_fit.xcm"
    save = getattr(getattr(xspec, "Xset", None), "save", None)
    if callable(save):
        if xcm.exists():
            xcm.unlink()
        save(str(xcm), info="a")
        text = xcm.read_text(encoding="utf-8", errors="replace")
        # Xset.save writes ``cd`` relative to the process cwd.  Normalize each
        # directory transition so the XCM can be restored from any cwd.
        virtual_cwd = Path.cwd().resolve()
        normalized_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("cd "):
                target_text = stripped[3:].strip().strip('"')
                target = Path(target_text).expanduser()
                if not target.is_absolute():
                    target = virtual_cwd / target
                virtual_cwd = target.resolve()
                line = f"cd {virtual_cwd}"
            normalized_lines.append(line)
        xcm.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")
    else:
        xcm.write_text("# XSPEC Xset.save unavailable in this backend\n", encoding="utf-8")
    log = output / f"{label}_fit.log"
    log.write_text(transcript.rstrip() + "\n", encoding="utf-8")
    result_json = output / f"{label}_fit.json"
    serializable = {key: value for key, value in result.items() if key != "prepared"}
    _write_json(result_json, serializable)
    return FitProductSet(
        result_json=result_json,
        report_txt=Path(report_txt),
        xspec_log=log,
        xcm=xcm,
        plots=tuple(Path(path) for path in plots),
        replay_cwd=output,
    )


def _value_error(
    parameters: Mapping[str, Any],
    suffix: str,
    *,
    include_errors: bool = True,
) -> str:
    parameter = _parameter(parameters, suffix)
    if not parameter:
        return "N/A"
    value = float(parameter["value"])
    lo = parameter.get("error_lo")
    hi = parameter.get("error_hi")
    if not include_errors or lo is None or hi is None:
        return f"{value:.4g}"
    return f"{value:.4g} (-{float(lo):.3g}/+{float(hi):.3g})"


def _profile_errors_available(fit: Mapping[str, Any]) -> bool:
    settings = fit.get("xspec_settings") or {}
    if settings.get("profile_errors_succeeded") is False:
        return False
    warnings = [str(item).lower() for item in fit.get("warnings") or ()]
    return not any("error calculation failed" in item for item in warnings)


def _compact_scientific_measurement(
    value: float,
    error_low: float | None = None,
    error_high: float | None = None,
) -> str:
    """Format a flux as ``1.2(-0.4/+0.6)e-10`` for short reports."""

    if not np.isfinite(value):
        return "N/A"
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    scale = 10.0**exponent
    central = value / scale
    if (
        error_low is not None
        and error_high is not None
        and np.isfinite(error_low)
        and np.isfinite(error_high)
        and error_low >= 0.0
        and error_high >= 0.0
    ):
        return (
            f"{central:.3g}(-{error_low / scale:.2g}/+{error_high / scale:.2g})"
            f"e{exponent:d}"
        )
    return f"{central:.3g}e{exponent:d}"


def _fit_flux_text(fit: Mapping[str, Any], *, include_errors: bool) -> str:
    flux = _parameter(fit.get("parameters") or {}, "cflux.lg10flux")
    if not flux:
        return "N/A"
    value = float(flux["value"])
    central = 10.0**value
    lo = flux.get("error_lo")
    hi = flux.get("error_hi")
    if include_errors and lo is not None and hi is not None:
        error_low = central - 10.0 ** (value - float(lo))
        error_high = 10.0 ** (value + float(hi)) - central
        return _compact_scientific_measurement(central, error_low, error_high)
    return _compact_scientific_measurement(central)


def _fit_interval_text(interval: Mapping[str, Any], t0_met: Any) -> str:
    description = str(interval.get("description") or "").strip()
    start = interval.get("start_met")
    stop = interval.get("stop_met")
    try:
        start_value = float(start)
        stop_value = float(stop)
    except (TypeError, ValueError):
        return description or "时间范围未记录"
    text = f"MET {start_value:.3f}-{stop_value:.3f} s"
    try:
        zero = float(t0_met)
    except (TypeError, ValueError):
        zero = math.nan
    if np.isfinite(zero):
        text += f"，相对 T0 为 {start_value - zero:+.3f} 至 {stop_value - zero:+.3f} s"
    return f"{description}；{text}" if description else text


def _xray_shape_parameter_text(
    fit: Mapping[str, Any],
    *,
    include_errors: bool,
) -> str:
    parameters = fit.get("parameters") or {}
    family = str(fit.get("model_family") or "").lower()
    if family == "apec" or _parameter(parameters, "apec.kt"):
        return f"apec.kT={_value_error(parameters, 'apec.kt', include_errors=include_errors)} keV"
    if family == "bbody" or _parameter(parameters, "bbody.kt"):
        text = f"bbody.kT={_value_error(parameters, 'bbody.kt', include_errors=include_errors)} keV"
        rest = (fit.get("derived_parameters") or {}).get("rest_kT_keV")
        return text + (f"；rest-frame kT={float(rest):.4g} keV" if rest is not None else "")
    if family == "bknpower" or _parameter(parameters, "bknpower.breake"):
        text = (
            f"PhoIndx1={_value_error(parameters, 'bknpower.phoindx1', include_errors=include_errors)}；"
            f"BreakE={_value_error(parameters, 'bknpower.breake', include_errors=include_errors)} keV；"
            f"PhoIndx2={_value_error(parameters, 'bknpower.phoindx2', include_errors=include_errors)}"
        )
        rest = (fit.get("derived_parameters") or {}).get("rest_break_energy_keV")
        return text + (f"；rest-frame BreakE={float(rest):.4g} keV" if rest is not None else "")
    return f"PhoIndex={_value_error(parameters, 'powerlaw.phoindex', include_errors=include_errors)}"


_XRAY_MODEL_EXPRESSIONS = {
    "powerlaw_free_nh": "tbabs*ztbabs*cflux*powerlaw",
    "powerlaw_nh0": "tbabs*ztbabs*cflux*powerlaw",
    "apec": "cflux*apec",
    "bbody_free_nh": "tbabs*ztbabs*cflux*bbody",
    "bbody_nh0": "tbabs*ztbabs*cflux*bbody",
    "bknpower_free_nh": "tbabs*ztbabs*cflux*bknpower",
    "bknpower_nh0": "tbabs*ztbabs*cflux*bknpower",
}


def _display_xray_model_expression(
    fit: Mapping[str, Any],
    candidate_key: str,
) -> str:
    """Return a reader-facing model expression without neutral nH=0 terms."""

    expression = str(
        fit.get("model")
        or _XRAY_MODEL_EXPRESSIONS.get(candidate_key)
        or candidate_key
        or "N/A"
    )
    absorption_mode = str(fit.get("absorption_mode") or "").lower()
    if absorption_mode == "zero" or candidate_key.lower().endswith("_nh0"):
        expression = "*".join(
            part for part in expression.split("*") if part.strip().lower() != "ztbabs"
        )
    canonical = {
        "tbabs": "TBabs",
        "ztbabs": "zTBabs",
        "cflux": "cflux",
        "powerlaw": "powerlaw",
        "bbody": "bbody",
        "bknpower": "bknpower",
        "apec": "apec",
    }
    return "*".join(
        canonical.get(part.strip().lower(), part.strip())
        for part in expression.split("*")
    )


def _wechat_duration_text(duration: Mapping[str, Any], key: str) -> str:
    value = duration.get(key)
    if value is None:
        return "N/A"
    error = np.asarray(duration.get(f"{key}_err", []), dtype=float).reshape(-1)
    if error.size >= 2 and np.all(np.isfinite(error[:2])):
        return f"{float(value):.4g} (-{abs(error[0]):.3g}/+{abs(error[1]):.3g}) s"
    return f"{float(value):.4g} s"


def _wechat_t0_source(value: Any) -> str:
    source = str(value or "N/A").strip()
    aliases = {
        "t100 start": "T100时段开始时间",
        "t100_start": "T100时段开始时间",
        "t100时段开始": "T100时段开始时间",
        "external trigger": "外部触发时间",
        "trigger_time_utc": "外部触发时间",
    }
    return aliases.get(source.lower(), source)


def _wechat_number(value: Any, *, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A" if value is None else str(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{decimals}f}"


def _wechat_interval_name(label: str, interval: Mapping[str, Any]) -> str:
    defaults = {
        "pipeline": "输入目录原始 pipeline 整段观测",
        "t100": "T100时段",
        "t90": "T90时段",
    }
    if label in defaults:
        return defaults[label]
    display = str(interval.get("display_name") or "").strip()
    if display:
        return display.removesuffix("谱").strip()
    if label.startswith("bb"):
        return f"Bayesian-block 时间分辨时段 {label}"
    return f"{label}时段"


def _wechat_shape_text(fit: Mapping[str, Any], *, include_errors: bool) -> str:
    parameters = fit.get("parameters") or {}
    family = str(fit.get("model_family") or "").lower()
    if family == "apec" or _parameter(parameters, "apec.kt"):
        value = _value_error(parameters, "apec.kt", include_errors=include_errors)
        return f"等离子体温度 kT={value} keV"
    if family == "bbody" or _parameter(parameters, "bbody.kt"):
        value = _value_error(parameters, "bbody.kt", include_errors=include_errors)
        return f"黑体温度 kT={value} keV"
    if family == "bknpower" or _parameter(parameters, "bknpower.breake"):
        return (
            f"低能谱指数={_value_error(parameters, 'bknpower.phoindx1', include_errors=include_errors)}，"
            f"折断能={_value_error(parameters, 'bknpower.breake', include_errors=include_errors)} keV，"
            f"高能谱指数={_value_error(parameters, 'bknpower.phoindx2', include_errors=include_errors)}"
        )
    return f"谱指数={_value_error(parameters, 'powerlaw.phoindex', include_errors=include_errors)}"


def _wechat_statistic_name(value: Any) -> str:
    key = str(value or "stat").lower()
    return {
        "wstat": "W-stat",
        "cstat": "C-stat",
        "chi": "chi-square",
        "chi2": "chi-square",
    }.get(key, str(value or "stat"))


def _wechat_common_lines(payload: Mapping[str, Any]) -> list[str]:
    duration = payload.get("duration") or {}
    counts = payload.get("t90_counts") or {}
    t0 = payload.get("t0") or {}
    target_id = str(payload.get("target_id") or "未命名暂现源")
    candidate_id = str(payload.get("candidate_id") or target_id)
    mission_time = _wechat_number(t0.get("mission_time_s"), decimals=3)
    lines = [
        (
            f"各位好，本次 WXT 数据中发现一例疑似短时标暂现源 {candidate_id}"
            f"（ObsID {payload.get('obsid', 'N/A')}，{payload.get('detector', 'N/A')}）。"
        ),
        "",
        (
            f"暂现活动开始时间约为 {t0.get('utc') or 'N/A'}（UTC；"
            f"MET {mission_time} s；"
            f"T0 来源：{_wechat_t0_source(t0.get('source'))}）。"
            f"时标分析得到 T100={_wechat_duration_text(duration, 't100')}，"
            f"T90={_wechat_duration_text(duration, 't90')}，"
            f"T50={_wechat_duration_text(duration, 't50')}。"
            f"T90 内净计数={_wechat_number(counts.get('net_counts'))}，"
            f"Li & Ma 显著性={_wechat_number(counts.get('significance'))}。"
        ),
    ]
    return lines


def _render_wechat_fit_message(
    payload: Mapping[str, Any],
    *,
    label: str,
    candidate_key: str,
    fit: Mapping[str, Any] | None,
    adopted: bool,
    failure: str | None = None,
) -> str:
    lines = _wechat_common_lines(payload)
    interval = (payload.get("fit_intervals") or {}).get(label) or {}
    interval_name = _wechat_interval_name(label, interval)
    target_id = str(payload.get("target_id") or "未命名暂现源")
    energy_band = payload.get("energy_band") or "N/A"
    if isinstance(energy_band, (tuple, list)) and len(energy_band) >= 2:
        energy_text = f"[{float(energy_band[0]):g}, {float(energy_band[1]):.1f}]"
    else:
        energy_text = str(energy_band)

    if fit is None:
        expression = _display_xray_model_expression({}, candidate_key)
        lines.extend(
            [
                (
                    f"采用{interval_name}数据拟合候选模型 {expression} 时失败："
                    f"{failure or '未记录失败原因'}。该候选未参与自动模型选择。"
                ),
                "",
                f"暂命名为 {target_id}。",
            ]
        )
        return "\n".join(lines)

    errors_ok = _profile_errors_available(fit)
    expression = _display_xray_model_expression(fit, candidate_key)
    model_status = "该时段自动采用模型" if adopted else "候选对比模型"
    lines.append(
        f"采用{interval_name}数据进行拟合；模型为 {expression}，"
        f"拟合能段为 {energy_text} keV（{model_status}）。"
    )

    nh = payload.get("galactic_absorption") or {}
    family = str(fit.get("model_family") or "").lower()
    absorption_mode = str(fit.get("absorption_mode") or "").lower()
    if family == "apec" or candidate_key == "apec":
        absorption_text = "该 APEC 候选按定义不包含银河系或本征吸收。"
    else:
        try:
            nh_cm2 = _compact_scientific_measurement(float(nh["nhtot_weighted_cm2"]))
        except (KeyError, TypeError, ValueError):
            nh_cm2 = "N/A"
        absorption_text = (
            f"银河系 nH 固定为 {nh_cm2} cm^-2"
            f"（TBabs.nH={nh.get('tbabs_nh_1e22', 'N/A')} x10^22 cm^-2）"
        )
        if absorption_mode == "zero" or candidate_key.endswith("_nh0"):
            absorption_text += "，本征吸收固定为 0。"
        else:
            intrinsic_nh = _value_error(
                fit.get("parameters") or {},
                "ztbabs.nh",
                include_errors=errors_ok,
            )
            absorption_text += (
                f"，本征 nH={intrinsic_nh} "
                "x10^22 cm^-2。"
            )

    stats = fit.get("statistics") or {}
    metrics = fit.get("metrics") or {}
    comparison_metrics = (
        ((payload.get("model_comparisons") or {}).get(label) or {}).get("metrics") or {}
    ).get(candidate_key) or {}
    metrics = comparison_metrics or metrics
    stat_name = _wechat_statistic_name(
        fit.get("effective_statistic") or stats.get("method")
    )
    statistic_value = _wechat_number(stats.get("value"))
    try:
        statistic_dof = str(int(stats["dof"]))
    except (KeyError, TypeError, ValueError):
        statistic_dof = str(stats.get("dof", "N/A"))
    fit_line = (
        f"{absorption_text}{_wechat_shape_text(fit, include_errors=errors_ok)}；"
        f"0.5-4 keV 未吸收流量={_fit_flux_text(fit, include_errors=errors_ok)} erg s^-1 cm^-2；"
        f"{stat_name}/dof={statistic_value}/{statistic_dof}"
    )
    if metrics.get("aicc") is not None:
        fit_line += f"；AICc={float(metrics['aicc']):.4g}"
    fit_line += "。"
    lines.extend([fit_line, "", f"暂命名为 {target_id}。"])
    return "\n".join(lines)


def render_wechat_fit_messages(payload: Mapping[str, Any]) -> dict[str, str]:
    """Render one standalone Chinese WeChat message per interval and model.

    The returned mapping preserves a deterministic order.  Keys use
    ``<interval>__<candidate>`` so callers can persist each message separately.
    A neutral ``zTBabs(nH=0)`` component remains in the XSPEC fit but is omitted
    from the reader-facing model expression.
    """

    fits = dict(payload.get("fits") or {})
    fit = payload.get("fit") or {}
    if fit and "t90" not in fits:
        fits["t90"] = fit
    comparisons = payload.get("model_comparisons") or {}
    available_labels = set(fits) | set(comparisons)
    labels = [
        label
        for label in ("pipeline", "t100", "t90")
        if label in available_labels
    ]
    labels.extend(sorted(label for label in available_labels if label.startswith("bb")))
    labels.extend(sorted(label for label in available_labels if label not in labels))

    messages: dict[str, str] = {}
    for label in labels:
        comparison = comparisons.get(label) or {}
        candidates = comparison.get("candidates") or {}
        failures = comparison.get("failures") or {}
        adopted_key = str(
            comparison.get("adopted_key")
            or (fits.get(label) or {}).get("model_key")
            or "selected"
        )
        ordered_keys = [adopted_key]
        ordered_keys.extend(
            key
            for key in comparison.get("ranking") or ()
            if key not in ordered_keys
        )
        ordered_keys.extend(key for key in candidates if key not in ordered_keys)
        ordered_keys.extend(key for key in failures if key not in ordered_keys)
        if not candidates and fits.get(label):
            candidates = {adopted_key: fits[label]}
        for candidate_key in ordered_keys:
            candidate = candidates.get(candidate_key)
            messages[f"{label}__{candidate_key}"] = _render_wechat_fit_message(
                payload,
                label=label,
                candidate_key=str(candidate_key),
                fit=candidate,
                adopted=str(candidate_key) == adopted_key,
                failure=failures.get(candidate_key),
            )

    if not messages:
        lines = _wechat_common_lines(payload)
        nh = payload.get("galactic_absorption") or {}
        lines.extend(
            [
                (
                    f"银河系 nH={nh.get('nhtot_weighted_cm2', 'N/A')} cm^-2；"
                    "当前运行未产生可报告的能谱拟合结果。"
                ),
                "",
                f"暂命名为 {payload.get('target_id') or '未命名暂现源'}。",
            ]
        )
        messages["event__no_fit"] = "\n".join(lines)
    return messages


def render_observation_summary(payload: Mapping[str, Any]) -> str:
    """Render the detailed Chinese observation and model-comparison report."""

    duration = payload.get("duration") or {}
    fit = payload.get("fit") or {}
    nh = payload.get("galactic_absorption") or {}
    counts = payload.get("t90_counts") or {}
    t0 = payload.get("t0") or {}
    fits = dict(payload.get("fits") or {})
    if fit and "t90" not in fits:
        fits["t90"] = fit
    for label, integrated in (payload.get("integrated_fits") or {}).items():
        fits.setdefault(str(label), integrated)
    intervals = payload.get("fit_intervals") or {}
    model_comparisons = payload.get("model_comparisons") or {}

    def duration_text(key: str) -> str:
        value = duration.get(key)
        if value is None:
            return "N/A"
        error = np.asarray(duration.get(f"{key}_err", []), dtype=float).reshape(-1)
        if error.size >= 2 and np.all(np.isfinite(error[:2])):
            return f"{float(value):.4g} (-{abs(error[0]):.3g}/+{abs(error[1]):.3g}) s"
        return f"{float(value):.4g} s"
    target_id = str(payload.get("target_id") or "未命名暂现源")
    candidate_id = str(payload.get("candidate_id") or target_id)
    detector = payload.get("detector", "N/A")
    obsid = payload.get("obsid", "N/A")
    lines = [
        (
            f"各位好，本次 WXT 数据中发现一例疑似短时标暂现源 {candidate_id}"
            f"（ObsID {obsid}，{detector}）。"
        ),
    ]
    if target_id != candidate_id:
        lines.append(f"暂命名为 {target_id}。")
    lines.extend(
        [
            (
                f"暂现活动开始时间约为 {t0.get('utc') or 'N/A'}（UTC；"
                f"MET {t0.get('mission_time_s', 'N/A')} s；T0 来源：{t0.get('source', 'N/A')}）。"
            ),
            (
                f"自动时标分析得到 T100={duration_text('t100')}，"
                f"T90={duration_text('t90')}，T50={duration_text('t50')}。"
            ),
            (
                f"T90 内 N_on={counts.get('n_on', 'N/A')}，N_off={counts.get('n_off', 'N/A')}，"
                f"alpha={counts.get('alpha', 'N/A')}，净计数={counts.get('net_counts', 'N/A')}，"
                f"signed Li & Ma 显著性={counts.get('significance', 'N/A')} sigma。"
            ),
            (
                f"采用统一数据、能段和统计量进行多候选 X 射线模型比较；"
                f"模型为 {fit.get('model') or next((item.get('model') for item in fits.values() if item), 'N/A')}，"
                f"拟合能段为 {payload.get('energy_band', 'N/A')} keV，红移 z={payload.get('redshift', 'N/A')}。"
            ),
            (
                f"银河系 nH 固定为 {nh.get('nhtot_weighted_cm2', 'N/A')} cm^-2"
                f"（TBabs.nH={nh.get('tbabs_nh_1e22', 'N/A')} x10^22 cm^-2；"
                f"由坐标 RA={payload.get('ra_deg', 'N/A')} deg、Dec={payload.get('dec_deg', 'N/A')} deg 自动查询）。"
            ),
        ]
    )

    preferred = ["pipeline", "t100", "t90"]
    preferred.extend(sorted(label for label in fits if label.startswith("bb")))
    preferred.extend(label for label in fits if label not in preferred)
    display_names = {
        "pipeline": "输入目录原始 pipeline 整段观测谱",
        "t100": "T100 时段谱",
        "t90": "T90 时段谱",
    }
    for label in preferred:
        interval_fit = fits.get(label) or {}
        if not interval_fit:
            continue
        interval = intervals.get(label) or {}
        params = interval_fit.get("parameters") or {}
        stats = interval_fit.get("statistics") or {}
        errors_ok = _profile_errors_available(interval_fit)
        display_name = str(
            interval.get("display_name")
            or display_names.get(label)
            or f"时间分辨谱 {label}"
        )
        interval_text = _fit_interval_text(interval, t0.get("mission_time_s"))
        stat_name = interval_fit.get("effective_statistic") or stats.get("method", "N/A")
        comparison = model_comparisons.get(label) or {}
        metrics = interval_fit.get("metrics") or {}
        lines.extend(
            [
                "",
                f"【{display_name}】",
                f"时间范围：{interval_text}。",
                (
                    f"自动采用：{interval_fit.get('model_key') or interval_fit.get('model', 'N/A')}；"
                    f"理由：{comparison.get('adopted_reason', '单模型拟合')}。"
                ),
                (
                    f"{_xray_shape_parameter_text(interval_fit, include_errors=errors_ok)}；"
                    f"zTBabs.nH={_value_error(params, 'ztbabs.nh', include_errors=errors_ok)} "
                    f"x10^22 cm^-2；0.5-4 keV 未吸收流量="
                    f"{_fit_flux_text(interval_fit, include_errors=errors_ok)} erg s^-1 cm^-2；"
                    f"{stat_name}={stats.get('value', 'N/A')}/{stats.get('dof', 'N/A')}；"
                    f"AIC={metrics.get('aic', 'N/A')}，AICc={metrics.get('aicc', 'N/A')}，"
                    f"BIC={metrics.get('bic', 'N/A')}。"
                ),
            ]
        )
        if not errors_ok:
            lines.append("该时段 XSPEC profile error 未收敛，以上仅报告 best-fit 值，误差需复核。")
        if str(interval_fit.get("model_family") or "").lower() == "apec":
            lines.append(
                "注意：该 APEC 候选按当前比较定义不包含银河系或本征吸收，"
                "其吸收假设与其余候选不同。"
            )
        candidates = comparison.get("candidates") or {}
        comparison_metrics = comparison.get("metrics") or {}
        if candidates:
            lines.append("候选模型比较：")
            for candidate_key in comparison.get("ranking") or candidates:
                candidate = candidates.get(candidate_key) or {}
                candidate_stats = candidate.get("statistics") or {}
                candidate_metric = comparison_metrics.get(candidate_key) or candidate.get("metrics") or {}
                lines.append(
                    f"  {candidate_key}："
                    f"{_xray_shape_parameter_text(candidate, include_errors=True)}；"
                    f"stat/dof={candidate_stats.get('value', 'N/A')}/{candidate_stats.get('dof', 'N/A')}；"
                    f"AIC={candidate_metric.get('aic', 'N/A')}，"
                    f"AICc={candidate_metric.get('aicc', 'N/A')}，"
                    f"BIC={candidate_metric.get('bic', 'N/A')}，"
                    f"Delta={candidate_metric.get('delta', 'N/A')}，"
                    f"weight={candidate_metric.get('akaike_weight', 'N/A')}。"
                )

    peak_flux = (payload.get("flux_curve") or {}).get("quicklook_peak_flux")
    if peak_flux is not None:
        try:
            peak_text = _compact_scientific_measurement(float(peak_flux))
        except (TypeError, ValueError):
            peak_text = "N/A"
        lines.extend(
            [
                "",
                (
                    f"按 T90 谱形由固定分箱净计数率换算的 quicklook 峰值流量约为 "
                    f"{peak_text} erg s^-1 cm^-2；该值仅用于快速浏览，不替代峰值时段独立谱拟合。"
                ),
            ]
        )
    if float(payload.get("redshift", 0.0) or 0.0) == 0.0:
        lines.extend(
            [
                "",
                "注意：redshift=0 时，zTBabs.nH 仅表示局域额外吸收，不能直接解释为宿主系柱密度。",
            ]
        )
    lines.extend(
        [
            "",
            "时标诊断图、full/soft/hard 计数光变、各时段能谱拟合图和 flux curve 见下方 Notebook 输出。",
        ]
    )
    return "\n".join(lines)
