"""One-source WXT/FXT analysis and a common-band, common-clock flux curve."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.table import Table

from ..core.config import InstrumentConfig, WXT
from ..core.fit import fit_xray_models
from ..core.instruments import DataFile, Manifest, SpectrumBundle
from ..core.plotstyle import apply_style, save_figure
from ..core.products import collect_runtime_environment, write_json
from ..core.spectrum_prep import prepare_spectra
from ..core.time import Time
from .fxt import FXTInput, FXTPipeline, FXTResult
from .wxt import WXTPointingInput, WXTPointingPipeline, WXTPointingResult

__all__ = ["TAInput", "TAPipeline", "TAResult", "combine_ta_flux_rows", "save_ta_flux_curve"]


@dataclass(frozen=True, slots=True, kw_only=True)
class TAInput:
    """One target and explicit instrument inputs, with an independent output tree.

    Coordinates are degrees and T0 is ISO UTC. ``redshift=None`` means unknown:
    additional absorption is fitted at z=0, without a host-column interpretation.
    Each WXT input identifies one observation/CMOS/source. FXT can contain many
    observations, with explicit shorter intervals for early autonomous followup.
    """
    target_id: str
    ra_deg: float
    dec_deg: float
    output_root: Path | str
    wxt: tuple[WXTPointingInput, ...] = ()
    fxt: FXTInput | None = None
    trigger_time_utc: str | None = None
    redshift: float | None = None


@dataclass(slots=True)
class TAResult:
    workspace: Path
    status: str
    wxt_results: list[WXTPointingResult] = field(default_factory=list)
    fxt_result: FXTResult | None = None
    flux_rows: list[dict[str, Any]] = field(default_factory=list)
    products: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def combine_ta_flux_rows(rows: Sequence[Mapping[str, Any]], *, trigger_time_utc: str | None = None) -> list[dict[str, Any]]:
    """Normalize independent measurements to seconds since one UTC reference.

    Rows explicitly describe observer-frame unabsorbed 0.5--4 keV flux in
    erg cm^-2 s^-1, and EP MET or seconds since a stated UTC T0. No spectral
    conversion or averaging occurs. Without explicit T0, prefer a WXT T0, then
    any stated T0, then earliest observation start (not an inferred trigger).
    """
    if not rows:
        return []
    absolute, keys = [], set()
    for source in rows:
        row = dict(source)
        band = np.asarray(row.get("flux_band_keV", (np.nan, np.nan)))
        if (row.get("flux_frame") != "observer" or row.get("flux_kind") != "unabsorbed"
                or band.shape != (2,) or not np.allclose(band, (0.5, 4.0))):
            raise ValueError("TA requires observer-frame unabsorbed 0.5--4 keV flux")
        if row.get("flux_unit", "erg cm^-2 s^-1") != "erg cm^-2 s^-1":
            raise ValueError("TA flux_unit must be erg cm^-2 s^-1")
        reference = row.get("time_reference")
        if reference == "ep":
            offset = 0.0
        elif reference == "seconds_since_t0" and row.get("t0_utc"):
            offset = float(Time(row["t0_utc"], scale="utc").ep)
        else:
            raise ValueError("Rows require EP MET or seconds_since_t0 with t0_utc")
        bounds = np.array([row["t_start"], row["t_stop"], row["t_center"]], dtype=float) + offset
        if not np.all(np.isfinite(bounds)) or not bounds[0] < bounds[1] or not bounds[0] <= bounds[2] <= bounds[1]:
            raise ValueError("Flux times require finite start < stop, with center inside")
        if not np.isfinite(row["flux"]) or row["flux"] <= 0:
            raise ValueError("Flux measurements/upper limits must be positive and finite")
        for name in ("error_minus", "error_plus"):
            error = row.get(name)
            if error is not None and (not np.isfinite(error) or error < 0):
                raise ValueError("Flux errors must be nonnegative or explicitly missing")
        key = (row["instrument"], row["obsid"], row.get("detector"), row["segment"])
        if key in keys:
            raise ValueError(f"Duplicate flux measurement: {key}")
        keys.add(key)
        row.update(t_start_ep=float(bounds[0]), t_stop_ep=float(bounds[1]), t_center_ep=float(bounds[2]))
        absolute.append(row)
    anchor_source = "user"
    if trigger_time_utc is None:
        anchored = sorted(absolute, key=lambda row: (row["instrument"] != "WXT", row["t_start_ep"]))
        trigger_time_utc = next((row.get("t0_utc") for row in anchored if row.get("t0_utc")), None)
        anchor_source = "instrument_t0" if trigger_time_utc else "earliest_observation_start"
        if trigger_time_utc is None:
            trigger_time_utc = Time(min(row["t_start_ep"] for row in absolute), format="ep").utc.isot
    anchor = float(Time(trigger_time_utc, scale="utc").ep)
    for row in absolute:
        row.update(t_start=row["t_start_ep"] - anchor, t_stop=row["t_stop_ep"] - anchor,
                   t_center=row["t_center_ep"] - anchor, t0_utc=trigger_time_utc,
                   time_reference="seconds_since_t0", time_unit="s", t0_source=anchor_source,
                   flux_unit="erg cm^-2 s^-1")
    return sorted(absolute, key=lambda row: (row["t_center"], row["instrument"], row["obsid"]))


def save_ta_flux_curve(rows: Sequence[Mapping[str, Any]], output_root: Path | str, *, title: str) -> dict[str, str]:
    """Save normalized rows as JSON/ECSV/CSV and plot WXT/FXT flux with UL arrows.

    Missing intervals stay missing; upper-limit confidence/provenance remains
    in the tables. This function consumes derived rows without refitting spectra.
    """
    import matplotlib.pyplot as plt
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    products = {"flux_rows": str(write_json(output / "flux_rows.json", list(rows)))}
    if not rows:
        return products
    columns = ("instrument", "obsid", "detector", "segment", "t_start", "t_stop", "t_center",
               "exposure_s", "flux", "error_minus", "error_plus", "is_upper_limit", "confidence",
               "error_kind", "error_status", "shape_status", "statistics_scope", "construction",
               "model_key", "status", "t0_utc", "t0_source", "redshift", "redshift_status", "target_redshift",
               "galactic_nh_1e22", "extra_nh_interpretation", "pileup_status", "cosmic_ray_status")
    table = Table(rows=[{key: row.get(key) for key in columns} for row in rows])
    for key in ("t_start", "t_stop", "t_center", "exposure_s"):
        table[key].unit = "s"
    for key in ("flux", "error_minus", "error_plus"):
        table[key].unit = "erg cm-2 s-1"
    table.meta.update(flux_band_keV=[0.5, 4.0], flux_frame="observer", flux_kind="unabsorbed",
                      time_reference="seconds_since_t0")
    for extension, format_name in (("ecsv", "ascii.ecsv"), ("csv", "ascii.csv")):
        path = output / f"flux_rows.{extension}"
        table.write(path, format=format_name, overwrite=True)
        products[f"flux_{extension}"] = str(path)
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for instrument, color, marker in (("WXT", "#2378a0", "o"), ("FXT", "#c05a35", "s")):
        labeled = False
        for row in rows:
            if row["instrument"] != instrument:
                continue
            label = instrument if not labeled else None
            labeled = True
            time, flux = row["t_center"], row["flux"]
            xerr = np.array([[time - row["t_start"]], [row["t_stop"] - time]])
            if row.get("is_upper_limit"):
                ax.errorbar(time, flux, xerr=xerr, yerr=0.25 * flux, uplims=True,
                            fmt=marker, color=color, ms=4, label=label)
            else:
                errors = (row.get("error_minus"), row.get("error_plus"))
                yerr = np.array(errors).reshape(2, 1) if all(value is not None for value in errors) else None
                ax.errorbar(time, flux, xerr=xerr, yerr=yerr, fmt=marker, color=color,
                            ms=4, capsize=2, label=label)
    if all(row["t_start"] > 0 for row in rows):
        ax.set_xscale("log")
    else:
        widths = [row["t_stop"] - row["t_start"] for row in rows]
        ax.set_xscale("symlog", linthresh=max(min(widths) / 2, 1.0))
    ax.set_yscale("log")
    ax.set(xlabel="Time since T0 (s)", ylabel=r"Unabsorbed 0.5–4 keV flux (erg cm$^{-2}$ s$^{-1}$)", title=title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.18, which="major")
    fig.tight_layout()
    products.update({f"plot_{key}": str(path) for key, path in save_figure(fig, output / "flux_curve", ("png", "pdf")).items()})
    plt.close(fig)
    return products


class TAPipeline:
    """Run instrument pipelines; retain review states and combine compatible flux.

    Absorption is shared between eligible WXT bins, including multiple WXT
    observations, never across instruments. Thermal preference is a candidate,
    external evidence for a physical identification.
    """
    def __init__(self, input_data: TAInput, *, wxt_config: InstrumentConfig | None = None,
                 fxt_options: Mapping[str, Any] | None = None, wxt_options: Mapping[str, Any] | None = None):
        self.input = input_data
        self.wxt_config = wxt_config or WXT()
        self.fxt_options = dict(fxt_options or {})
        self.wxt_options = dict(wxt_options or {})
        self.wxt_pipelines: dict[int, WXTPointingPipeline] = {}
        if not input_data.wxt and input_data.fxt is None:
            raise ValueError("TA needs at least one WXT or FXT input")
        if not np.isfinite(input_data.ra_deg) or not 0 <= input_data.ra_deg < 360 or not np.isfinite(input_data.dec_deg) or not -90 <= input_data.dec_deg <= 90:
            raise ValueError("TA coordinates must be finite degrees within RA/Dec ranges")
        if input_data.redshift is not None and (not np.isfinite(input_data.redshift) or input_data.redshift < 0):
            raise ValueError("redshift must be nonnegative or None (unknown)")

    def _fit_all_wxt_bins(self, children, rows, output, redshift):
        """Link one absorption across observations, retaining each bin's response."""
        import json
        selected, columns = [], []
        for child in children:
            if child.files is None:
                continue
            bins = {(item["label"] if item.get("label") else f"bb{int(item['index']):03d}"): item for item in child.spectrum_bins
                    if item.get("fit_eligible", True) and item.get("source_pha")}
            for row in rows:
                if (row["obsid"] != child.files.obsid or row.get("detector") != child.files.detector
                        or row["segment"] not in bins):
                    continue
                item = bins[row["segment"]]
                selected.append((child, item))
                columns.append(row)
        if len({str(child.workspace) for child, item in selected}) < 2:
            return
        destination, prepared, galactic = output / "wxt_shared_fit", [], []
        for index, (child, item) in enumerate(selected):
            label = f"bin{index:03d}"
            identities = dict(obsid=child.files.obsid, detector=child.files.detector, source_id=label)
            files = [DataFile(Path(item[key]), role, "TA WXT time bin", "EP", "WXT", **identities)
                     for key, role in (("source_pha", "source_pha"), ("background_pha", "background_pha"),
                                       ("arf", "arf"), ("rmf", "rmf"))]
            bundle = SpectrumBundle(*files, module=None, **identities)
            catalog = prepare_spectra(Manifest(root=Path(item["source_pha"]).parent, mission="EP",
                instrument="WXT", layout="TA WXT bins", files=files, bundles=[bundle],
                obsid=child.files.obsid, detector=child.files.detector),
                outdir=destination / "prepared", group_min=self.wxt_config.spectrum.group_min_counts, overwrite=True)
            if not catalog.spectra or not catalog.spectra[0].ready:
                raise ValueError(f"WXT bin preparation failed: {catalog.diagnostics}")
            prepared.append(catalog.spectra[0])
            cache = json.loads((child.workspace / "galactic_absorption" / "nhtot.json").read_text())
            galactic.append(float(cache["result"]["tbabs_nh_1e22"]))
        if not np.allclose(galactic, galactic[0], rtol=1e-6, atol=1e-12):
            raise ValueError("WXT observations have inconsistent fixed Galactic NH")
        cfg = self.wxt_config.fitting
        if cfg.method != "mle":
            raise NotImplementedError("WXT shared-NH comparison currently requires method='mle'")
        band = self.wxt_config.spectrum.fit_energy_range_keV or self.wxt_config.energy_range_keV
        comparison = fit_xray_models(prepared, outdir=destination / "fit", candidate_keys=cfg.candidate_keys,
            model_class=cfg.model_class, absorption_mode=cfg.absorption_mode,
            emin=band[0], emax=band[1], selection_metric=cfg.selection_metric,
            stat_method=cfg.statistic, abundance=cfg.abundance,
            cross_section=cfg.cross_section, apec_root=cfg.apec_root, galactic_nh_1e22=galactic[0],
            redshift=redshift, flux_band_keV=(0.5, 4.0), shared_absorption_only=True,
            calculate_errors=cfg.calculate_errors, error_delta_stat=cfg.error_delta_stat,
            srcname=self.input.target_id, instname="WXT_all_bins",
            plot_formats=self.wxt_config.plotting.formats if self.wxt_config.plotting.enabled else ())
        groups = comparison.adopted_fit.get("per_group", ())
        if len(groups) != len(columns):
            raise ValueError("Joint WXT fit requires one independent flux/shape group per bin")
        for group in groups:
            row = columns[int(group["spectrum_indices"][0]) - 1]
            flux = group["flux_unabs"]
            settings = comparison.adopted_fit.get("xspec_settings", {})
            row.update(flux=flux["erg_cm2_s"], error_minus=flux.get("error_lo"), error_plus=flux.get("error_hi"),
                       model_key=comparison.adopted_key, status="joint_WXT_spectral_fit",
                       error_status=flux.get("error_status"), confidence=settings.get("profile_confidence"),
                       error_kind=settings.get("profile_error_kind"),
                       shape_status=comparison.adopted_fit.get("shape_status", {}).get("status"),
                       statistics_scope="all_WXT_bins", comparison_report=str(destination / "fit" / "model_comparison.json"))

    def run(self, *, overwrite: bool = False) -> TAResult:
        output = Path(self.input.output_root).expanduser().resolve()
        inputs = (*self.input.wxt, *((self.input.fxt,) if self.input.fxt else ()))
        for instrument_input in inputs:
            root = instrument_input.resolved_root()
            if output == root or output.is_relative_to(root):
                raise ValueError("TA output must be outside every input product tree")
        report = output / "ta_result.json"
        if report.exists() and not overwrite:
            raise FileExistsError(f"Preserve existing TA results: {report}; choose a new output directory")
        output.mkdir(parents=True, exist_ok=True)
        result, rows = TAResult(output, "missing_inputs"), []
        trigger = self.input.trigger_time_utc
        z = 0.0 if self.input.redshift is None else self.input.redshift
        for index, source in enumerate(self.input.wxt):
            selected = replace(source, target_id=self.input.target_id, ra_deg=self.input.ra_deg,
                               dec_deg=self.input.dec_deg, redshift=z, trigger_time_utc=trigger or source.trigger_time_utc,
                               output_root=output / "wxt" / f"obs{index:03d}")
            try:
                config = copy.copy(self.wxt_config)
                config.execution = replace(config.execution, workspace=selected.resolved_output_root())
                pipeline = WXTPointingPipeline(selected, config=config, **self.wxt_options)
                self.wxt_pipelines[index] = pipeline
                child = pipeline.run(resume=True)
                result.wxt_results.append(child)
                rows.extend(getattr(child, "flux_rows", ()))
                if child.status != "completed":
                    result.warnings.append(f"WXT {index}: {child.status}; see {child.report or child.workspace}")
                if trigger is None and child.duration:
                    trigger = (child.duration.get("t0") or {}).get("utc")
            except Exception as exc:
                result.warnings.append(f"WXT {index}: {type(exc).__name__}: {exc}")
        shared_fit_failed = False
        try:
            self._fit_all_wxt_bins(result.wxt_results, rows, output, z)
        except Exception as exc:
            # Independent fits remain diagnostic; never call them a shared-NH result.
            shared_fit_failed = True
            result.warnings.append(f"WXT shared absorption failed: {type(exc).__name__}: {exc}")
        if self.input.fxt is not None:
            selected = replace(self.input.fxt, target_id=self.input.target_id, ra_deg=self.input.ra_deg,
                               dec_deg=self.input.dec_deg, redshift=z, trigger_time_utc=trigger or self.input.fxt.trigger_time_utc,
                               output_root=output / "fxt")
            try:
                result.fxt_result = FXTPipeline(selected, **self.fxt_options).run(overwrite=overwrite)
                rows.extend(result.fxt_result.flux_rows)
                result.warnings.extend(result.fxt_result.warnings)
                if result.fxt_result.status != "completed":
                    result.warnings.append(f"FXT: {result.fxt_result.status}")
            except Exception as exc:
                result.warnings.append(f"FXT: {type(exc).__name__}: {exc}")
        result.flux_rows = combine_ta_flux_rows(rows, trigger_time_utc=trigger)
        for row in result.flux_rows:
            stellar_thermal = str(row.get("model_key", "")).startswith(("apec", "double_apec"))
            bare_thermal = row.get("model_key") in {"apec", "double_apec"}
            row.update(redshift=0.0 if stellar_thermal else z, target_redshift=self.input.redshift,
                       redshift_status=("fixed_stellar_zero" if stellar_thermal else
                                        "unknown_fixed_zero" if self.input.redshift is None else "provided"),
                       extra_nh_interpretation=("not_applicable_bare_thermal" if bare_thermal else
                                               "local_stellar_absorption" if stellar_thermal else
                                               "observer_frame_effective" if self.input.redshift is None else "at_provided_redshift"))
        result.products.update(save_ta_flux_curve(result.flux_rows, output / "combined", title=self.input.target_id))
        statuses = [child.status for child in result.wxt_results]
        if result.fxt_result is not None:
            statuses.append(result.fxt_result.status)
        if result.flux_rows:
            result.status = ("completed" if len(statuses) == len(inputs) and all(status == "completed" for status in statuses)
                             else "needs_review" if len(statuses) == len(inputs) and all(status in {"completed", "needs_review"} for status in statuses)
                             else "partial")
        else:
            result.status = "no_flux_measurements"
        if shared_fit_failed and result.flux_rows:
            result.status = "partial"
        preferences = [{"instrument": row["instrument"], "obsid": row["obsid"], "segment": row["segment"],
                        "model_key": row.get("model_key"), "stellar_flare_candidate":
                        str(row.get("model_key", "")).startswith(("apec", "double_apec"))} for row in result.flux_rows]
        result.products["report"] = str(report)
        write_json(report, {"result": result, "input": self.input, "spectral_preferences": preferences,
                            "classification_status": "spectral_candidates_only; physical identification needs external evidence",
                            "environment": collect_runtime_environment()})
        return result
