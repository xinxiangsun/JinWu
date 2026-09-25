from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any, ClassVar, Iterator, Literal, Mapping

__all__ = [
    "XSPEC_COSMO_PLANCK18",
    "EnergyBandConfig",
    "ExtractionConfig",
    "RegionConfig",
    "BackgroundScalingConfig",
    "DurationConfig",
    "BayesianBlockSpectrumConfig",
    "SpectrumConfig",
    "FitConfig",
    "BXAConfig",
    "PlotConfig",
    "FluxCurveConfig",
    "ReportConfig",
    "GalacticAbsorptionConfig",
    "UpperLimitConfig",
    "ExecutionConfig",
    "SwiftGRBDataConfig",
    "SwiftXRTRequestConfig",
    "SwiftGRBSegmentationConfig",
    "SwiftBATSurveyConfig",
    "SwiftBATSurveySelectionConfig",
    "SwiftBATSurveyDownloadConfig",
    "SwiftBATSurveyMosaicConfig",
    "BATSurveyConfig",
    "BATSurveySelectionConfig",
    "BATSurveyDownloadConfig",
    "BATSurveyMosaicConfig",
    "GBMAnalysisConfig",
    "InstrumentConfig",
    "register_instrument",
    "instrument",
    "FXT",
    "WXT",
    "BAT",
    "BATSurvey",
    "GBM",
    "GBMContinuous",
    "GECAM",
    "UVOT",
    "SwiftGRB",
    # Process-wide fit settings API
    "get_fit_settings",
    "set_fit_settings",
    "reset_fit_settings",
    "set_fit_method",
    "get_fit_method",
    "fit_settings",
]

XSPEC_COSMO_PLANCK18 = "67.66 -0.534016305544544 0.6888463055445441"


@dataclass(frozen=True, slots=True)
class EnergyBandConfig:
    """One named analysis band and its event-channel selection."""

    energy_range_keV: tuple[float, float]
    pi_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Mission-independent event-product extraction defaults."""

    backend: str = "xselect"
    time_format: str = "scc"
    lightcurve_binsize_s: float = 1.0
    image_binsize: int = 1
    timeout_s: float | None = None
    bands: Mapping[str, EnergyBandConfig] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegionConfig:
    """Region generation and review defaults."""

    require_review: bool = True
    source_radius_arcsec: float | None = None
    background_sectors: tuple[tuple[float, float, float, float], ...] = ()
    background_strategy: str = "catalog_or_generated"
    background_orientation: str = "event_roll"
    background_polygon_samples: int = 80


@dataclass(frozen=True, slots=True)
class BackgroundScalingConfig:
    """Exposure-map background-scaling policy."""

    method: str = "exposure_map_ratio"
    mask_mode: str = "exact"
    alpha_rtol: float = 1e-6
    exposure_rtol: float = 1e-6


@dataclass(frozen=True, slots=True)
class DurationConfig:
    """T100/T90/T50 defaults for event-based duration estimation."""

    p0: float = 0.05
    block_snr_threshold: float = 3.0
    cumulative_mode: str = "adaptive"
    event_binsize_s: float = 1.0
    diagnostic_min_t100_bins: int = 10
    focus_t100: bool = False
    diagnostic_context_fraction: float = 0.25
    nmc: int = 3000
    seed: int = 42


@dataclass(frozen=True, slots=True)
class BayesianBlockSpectrumConfig:
    """Rules for merging Bayesian blocks into fit-worthy spectra."""

    enabled: bool = True
    minimum_net_counts: float = 1.0
    minimum_significance: float = 3.0


@dataclass(frozen=True, slots=True)
class SpectrumConfig:
    """PHA construction and grouping defaults."""

    group_min_counts: int = 1
    fit_energy_range_keV: tuple[float, float] | None = None


#: Inference methods selectable via ``FitConfig.method`` / ``fit_spectral``.
_FIT_METHODS: frozenset[str] = frozenset({"mle", "chain", "bxa"})
#: Likelihood engines selectable via ``FitConfig.backend``.  ``xspec`` is the
#: only supported engine today; a native ``jinwu.model`` backend is reserved.
_FIT_BACKENDS: frozenset[str] = frozenset({"xspec"})
#: Model-selection metrics.  ``logz`` is the Bayesian evidence ranking used by
#: the BXA path; ``none`` disables ranking (survey / continuous presets).
_FIT_SELECTION_METRICS: frozenset[str] = frozenset(
    {"aicc", "aic", "bic", "logz", "none"}
)


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Backend-neutral fit defaults used by instrument pipelines.

    ``method`` selects *how* the spectrum is inferred (``mle`` maximum
    likelihood, ``chain`` XSPEC MCMC, ``bxa`` UltraNest nested sampling) and is
    consumed by :func:`jinwu.core.fit.fit_spectral` and instrument pipelines.
    ``backend`` is the orthogonal *likelihood engine* identifier (``xspec``;
    a native ``jinwu.model`` engine is reserved for the future).
    """

    enabled: bool = True
    backend: str = "xspec"
    method: Literal["mle", "chain", "bxa"] = "mle"
    model_name: str = "tbabs*powerlaw"
    statistic: str = "cstat"
    abundance: str = "wilm"
    cross_section: str = "vern"
    calculate_errors: bool = True
    error_delta_stat: float = 1.0
    model_class: str = "auto"
    absorption_mode: str = "auto"
    candidate_keys: tuple[str, ...] | None = None
    selection_metric: str = "aicc"
    comparison_intervals: tuple[str, ...] = ("pipeline", "t100", "t90")

    def __post_init__(self) -> None:
        try:
            value = float(self.error_delta_stat)
        except (TypeError, ValueError) as exc:
            raise ValueError("error_delta_stat must be finite and positive") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("error_delta_stat must be finite and positive")
        object.__setattr__(self, "error_delta_stat", value)
        if self.method not in _FIT_METHODS:
            raise ValueError(
                f"FitConfig.method must be one of {sorted(_FIT_METHODS)}; "
                f"got {self.method!r}"
            )
        if self.backend not in _FIT_BACKENDS:
            raise ValueError(
                f"FitConfig.backend must be one of {sorted(_FIT_BACKENDS)}; "
                f"got {self.backend!r}"
            )
        if self.selection_metric not in _FIT_SELECTION_METRICS:
            raise ValueError(
                f"FitConfig.selection_metric must be one of "
                f"{sorted(_FIT_SELECTION_METRICS)}; got {self.selection_metric!r}"
            )


@dataclass(frozen=True, slots=True)
class BXAConfig:
    """Advanced BXA / UltraNest nested-sampling tuning.

    These are *not* part of the process-wide :class:`FitConfig` singleton so the
    global "basic settings" stay lean; they live on ``InstrumentConfig.bxa`` and
    are threaded into :func:`jinwu.core.bxa_fit.fit_prepared_bxa`.  Field names
    are contractually identical to that function's keyword arguments.
    """

    n_live_points: int | None = None
    evidence_tolerance: float = 0.5
    speed: str = "safe"
    resume: bool = False
    calculate_flux_chain: bool = True
    flux_erange: str = "2.0 10.0"


@dataclass(frozen=True, slots=True)
class PlotConfig:
    """Reusable plotting defaults for pipeline science products."""

    enabled: bool = True
    formats: tuple[str, ...] = ("png", "svg")
    dpi: int = 300
    required: bool = True


@dataclass(frozen=True, slots=True)
class FluxCurveConfig:
    """Time-resolved and fixed-shape quicklook flux-curve policy."""

    enabled: bool = True
    energy_range_keV: tuple[float, float] | None = None
    include_t90_aggregate: bool = True
    build_quicklook: bool = True


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Human- and machine-readable pipeline reporting defaults."""

    enabled: bool = True
    print_summary: bool = True
    language: str = "zh"


@dataclass(frozen=True, slots=True)
class GalacticAbsorptionConfig:
    """Coordinate-based Galactic column-density lookup policy."""

    service: str = "swift_ukssdc_nhtot"
    equinox: int = 2000
    timeout_s: float = 30.0
    use_cache: bool = True


class _DerivedAliasFloat(float):
    """Mark a computed alias so ``dataclasses.replace`` can recompute it."""


@dataclass(frozen=True, slots=True)
class UpperLimitConfig:
    """Instrument-selected policy for response-aware upper limits.

    The strategy is deliberately part of the instrument configuration rather
    than a free argument to the calculator.  This prevents, for example, a
    coded-mask product from being interpreted as a simple aperture ON/OFF
    measurement.
    """

    strategy: Literal[
        "spatial_onoff",
        "modeled_count_spectrum",
        "coded_mask_spectrum",
        "photometric",
        "unsupported",
    ] = "unsupported"
    background_likelihood: Literal[
        "poisson_onoff",
        "gaussian_model",
        "gaussian_net_rate",
        "poisson_gaussian_profile",
        "known_background",
        "photometric",
    ] = "known_background"
    response_folding: Literal["rmf_arf", "rsp", "count_to_flux"] = "count_to_flux"
    combine: Literal["single", "joint_detectors", "joint_modules"] = "single"
    result_modes: tuple[str, ...] = ()
    default_sigma: float = 3.0
    # Fixed photon index used by conditional upper-limit profiles when an
    # instrument does not fit the spectral shape.  Instrument-specific
    # configuration groups may expose a more descriptive alias, but keeping
    # this value in the shared policy makes the result contract auditable.
    spectral_index: float = 2.0
    detection_power: float = 0.90
    calibration: Literal["bootstrap", "asymptotic", "empirical_if_available"] = "bootstrap"
    null_trials: int = 200_000
    signal_trials: int = 20_000
    fractional_background_systematic: float | None = None
    enabled: bool = False
    unavailable_reason: str | None = "No upper-limit strategy is configured."
    # Explicit science-facing aliases.  ``default_sigma`` and ``calibration``
    # remain for backwards compatibility with existing callers.
    upper_confidence_sigma: float | None = None
    detection_false_alarm_probability: float | None = None
    calibration_mode: Literal[
        "conditional_model", "empirical_if_available", "empirical_fixed_position",
        "empirical_global_search",
    ] = "conditional_model"

    def __post_init__(self) -> None:
        allowed_strategies = {
            "spatial_onoff",
            "modeled_count_spectrum",
            "coded_mask_spectrum",
            "photometric",
            "unsupported",
        }
        if self.strategy not in allowed_strategies:
            raise ValueError(f"Unknown upper-limit strategy: {self.strategy}")
        if self.calibration not in {"bootstrap", "asymptotic", "empirical_if_available"}:
            raise ValueError(f"Unknown upper-limit calibration: {self.calibration}")
        if self.calibration_mode not in {
            "conditional_model",
            "empirical_if_available",
            "empirical_fixed_position",
            "empirical_global_search",
        }:
            raise ValueError(f"Unknown upper-limit calibration_mode: {self.calibration_mode}")
        if self.calibration_mode.startswith("empirical_") and self.calibration != "empirical_if_available":
            raise ValueError("empirical calibration_mode requires calibration='empirical_if_available'")
        if self.calibration == "empirical_if_available" and self.calibration_mode == "conditional_model":
            object.__setattr__(self, "calibration_mode", "empirical_if_available")
        allowed_modes = {"observed_upper_bound", "detection_sensitivity"}
        unknown_modes = set(self.result_modes) - allowed_modes
        if unknown_modes:
            raise ValueError(f"Unknown upper-limit result modes: {sorted(unknown_modes)}")
        if len(set(self.result_modes)) != len(self.result_modes):
            raise ValueError("upper-limit result_modes must not contain duplicates")

        sigma = float(self.default_sigma)
        if self.upper_confidence_sigma is not None and not isinstance(self.upper_confidence_sigma, _DerivedAliasFloat):
            sigma_alias = float(self.upper_confidence_sigma)
            if not math.isfinite(sigma_alias) or sigma_alias <= 0:
                raise ValueError("upper_confidence_sigma must be finite and positive")
            if sigma == 3.0:
                sigma = sigma_alias
            elif not math.isclose(sigma, sigma_alias, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "default_sigma and upper_confidence_sigma disagree; specify one value"
                )
        power = float(self.detection_power)
        spectral_index = float(self.spectral_index)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("upper-limit default_sigma must be finite and positive")
        if not math.isfinite(spectral_index) or spectral_index <= 0:
            raise ValueError("upper-limit spectral_index must be finite and positive")
        if not math.isfinite(power) or not 0.0 < power < 1.0:
            raise ValueError("upper-limit detection_power must be between 0 and 1")
        if int(self.null_trials) <= 0 or int(self.signal_trials) <= 0:
            raise ValueError("upper-limit Monte Carlo trial counts must be positive")
        object.__setattr__(self, "default_sigma", sigma)
        object.__setattr__(self, "spectral_index", spectral_index)
        object.__setattr__(self, "upper_confidence_sigma", _DerivedAliasFloat(sigma))
        object.__setattr__(self, "detection_power", power)
        object.__setattr__(self, "null_trials", int(self.null_trials))
        object.__setattr__(self, "signal_trials", int(self.signal_trials))
        # 1 - Phi(sigma), written without importing scipy in the core
        # configuration layer.
        false_alarm = (
            0.5 * math.erfc(sigma / math.sqrt(2.0))
            if self.detection_false_alarm_probability is None
            or isinstance(self.detection_false_alarm_probability, _DerivedAliasFloat)
            else float(self.detection_false_alarm_probability)
        )
        if not math.isfinite(false_alarm) or not 0.0 < false_alarm < 1.0:
            raise ValueError("detection_false_alarm_probability must be between 0 and 1")
        if self.detection_false_alarm_probability is None or isinstance(
            self.detection_false_alarm_probability, _DerivedAliasFloat
        ):
            object.__setattr__(self, "detection_false_alarm_probability", _DerivedAliasFloat(false_alarm))

        systematic = self.fractional_background_systematic
        if systematic is not None:
            systematic = float(systematic)
            if not math.isfinite(systematic) or systematic < 0:
                raise ValueError(
                    "fractional_background_systematic must be finite and non-negative"
                )
            object.__setattr__(self, "fractional_background_systematic", systematic)

        if self.strategy == "unsupported":
            if self.enabled:
                raise ValueError("unsupported upper-limit strategy cannot be enabled")
            if self.result_modes:
                raise ValueError("unsupported upper-limit strategy cannot define result modes")
            if not self.unavailable_reason:
                raise ValueError("unsupported upper-limit strategy requires unavailable_reason")
            return

        if not self.enabled:
            raise ValueError("a configured upper-limit strategy must be enabled")
        if "observed_upper_bound" not in self.result_modes:
            raise ValueError("enabled upper-limit strategies must provide observed_upper_bound")

        valid_combinations = {
            "spatial_onoff": ({"poisson_onoff"}, {"rmf_arf"}, {"single", "joint_modules"}),
            "modeled_count_spectrum": (
                {
                    "gaussian_model",
                    "gaussian_net_rate",
                    "poisson_gaussian_profile",
                    "known_background",
                },
                {"rsp"},
                {"single", "joint_detectors"},
            ),
            "coded_mask_spectrum": (
                {
                    "gaussian_model",
                    "gaussian_net_rate",
                    "poisson_gaussian_profile",
                    "known_background",
                },
                {"rsp"},
                {"single"},
            ),
            "photometric": ({"photometric"}, {"count_to_flux"}, {"single"}),
        }
        likelihoods, responses, combinations = valid_combinations[self.strategy]
        if self.background_likelihood not in likelihoods:
            raise ValueError(
                f"{self.strategy} is incompatible with background likelihood "
                f"{self.background_likelihood}"
            )
        if self.response_folding not in responses:
            raise ValueError(
                f"{self.strategy} is incompatible with response folding {self.response_folding}"
            )
        if self.combine not in combinations:
            raise ValueError(f"{self.strategy} is incompatible with combine={self.combine}")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Pipeline execution, persistence, and external-tool defaults."""

    workspace: Path | None = None
    resume: bool = True
    overwrite: bool = False
    command_timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class SwiftGRBDataConfig:
    """Product-selection defaults for one Swift BAT+XRT GRB.

    ``BATBand`` is the Burst Analyser 15--50 keV flux product; it is not the
    full BAT detector bandpass.  Keeping that distinction here prevents plots
    and NPZ products from silently labelling the flux as 15--150 keV.
    """

    bat_binning: str = "SNR5_sinceT0"
    bat_band: str = "BATBand"
    xrt_band: str = "XRTBand"
    include_tar: bool = False
    download_retries: int = 3
    retry_wait_s: float = 5.0
    request_timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class SwiftXRTRequestConfig:
    """Explicit, resumable UKSSDC single-target Product Generator policy."""

    enabled: bool = False
    max_polls: int = 1
    poll_interval_s: float = 60.0
    archive_format: str = "zip"
    pc_counts: int = 20
    wt_counts: int = 30
    dynamic: bool = True
    pos_err_arcsec: float = 2.0


@dataclass(frozen=True, slots=True)
class SwiftGRBSegmentationConfig:
    """Event-level Bayesian-block and fit-worthiness settings for Swift/XRT."""

    p0: float = 0.05
    wt_snr_threshold: float = 3.0
    pc_snr_threshold: float = 3.0
    pc_min_source_counts: float = 20.0
    wt_merge_scope: Literal["gti"] = "gti"
    pc_merge_scope: Literal["obsid", "gti"] = "obsid"
    response_policy: Literal["interval0_compat", "require_interval_response"] = (
        "interval0_compat"
    )


@dataclass(frozen=True, slots=True)
class SwiftBATSurveyConfig:
    """BAT survey product and quality policy for one target.

    BAT survey rates are background-subtracted, on-axis corrected rates in
    count/s/fully-illuminated-detector.  They are not Burst Analyser energy
    fluxes and must be fitted with a Gaussian-rate statistic.  ``detthresh``
    and ``detthresh2`` are BatAnalysis-compatible detector-count settings;
    they are not presented as HEASoft official defaults.
    """

    detthresh: int = 10000
    detthresh2: int = 10000
    min_pcode: float = 0.05
    energy_range_keV: tuple[float, float] = (14.0, 195.0)
    channel_edges_keV: tuple[float, ...] = (
        14.0,
        20.0,
        24.0,
        35.0,
        50.0,
        75.0,
        100.0,
        150.0,
        195.0,
    )
    source_snr_threshold: float = 3.0
    strong_snr_threshold: float = 5.0
    upper_limit_photon_index: float = 2.0
    upper_limit_delta_stat: float = 9.0
    processes: int = 1
    internal_threads: int = 1
    task_timeout_s: float = 3600.0
    catalog_name: str = "survey6b_2.cat"

    def __post_init__(self) -> None:
        if int(self.detthresh) <= 0 or int(self.detthresh2) <= 0:
            raise ValueError("BAT survey detector thresholds must be positive")
        object.__setattr__(self, "detthresh", int(self.detthresh))
        object.__setattr__(self, "detthresh2", int(self.detthresh2))
        emin, emax = (float(value) for value in self.energy_range_keV)
        if not math.isfinite(emin) or not math.isfinite(emax) or not 0 < emin < emax:
            raise ValueError("BAT survey energy_range_keV must be finite and increasing")
        object.__setattr__(self, "energy_range_keV", (emin, emax))
        edges = tuple(float(value) for value in self.channel_edges_keV)
        if len(edges) != 9 or any(not math.isfinite(value) for value in edges):
            raise ValueError("BAT survey requires the eight native channel edges")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("BAT survey channel_edges_keV must be strictly increasing")
        if edges[0] != emin or edges[-1] != emax:
            raise ValueError("energy_range_keV must match the native channel edges")
        object.__setattr__(self, "channel_edges_keV", edges)
        pcode = float(self.min_pcode)
        if not math.isfinite(pcode) or not 0 <= pcode <= 1:
            raise ValueError("BAT survey min_pcode must be between 0 and 1")
        object.__setattr__(self, "min_pcode", pcode)
        for name in ("source_snr_threshold", "strong_snr_threshold", "upper_limit_photon_index"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"BAT survey {name} must be finite and positive")
            object.__setattr__(self, name, value)
        delta = float(self.upper_limit_delta_stat)
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError("BAT survey upper_limit_delta_stat must be finite and positive")
        object.__setattr__(self, "upper_limit_delta_stat", delta)
        if int(self.processes) < 1 or int(self.internal_threads) < 1:
            raise ValueError("BAT survey processes and internal_threads must be positive")
        object.__setattr__(self, "processes", int(self.processes))
        object.__setattr__(self, "internal_threads", int(self.internal_threads))
        timeout = float(self.task_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("BAT survey task_timeout_s must be finite and positive")
        object.__setattr__(self, "task_timeout_s", timeout)


# Public semantic alias: the detector/partial-coding fields are also exposed
# as an explicit ``selection`` group on ``InstrumentConfig``.
SwiftBATSurveySelectionConfig = SwiftBATSurveyConfig
BATSurveyConfig = SwiftBATSurveyConfig
BATSurveySelectionConfig = SwiftBATSurveySelectionConfig


@dataclass(frozen=True, slots=True)
class SwiftBATSurveyDownloadConfig:
    """Optional HEASARC observation query/download policy."""

    query_enabled: bool = False
    download_enabled: bool = False
    timeout_s: float = 60.0
    retries: int = 3
    retry_wait_s: float = 5.0
    query_margin_s: float = 0.0

    def __post_init__(self) -> None:
        timeout = float(self.timeout_s)
        margin = float(self.query_margin_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("BAT survey download timeout_s must be finite and positive")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("BAT survey query_margin_s must be finite and non-negative")
        if int(self.retries) < 1:
            raise ValueError("BAT survey retries must be positive")
        retry_wait = float(self.retry_wait_s)
        if not math.isfinite(retry_wait) or retry_wait < 0:
            raise ValueError("BAT survey retry_wait_s must be finite and non-negative")
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "query_margin_s", margin)
        object.__setattr__(self, "retries", int(self.retries))
        object.__setattr__(self, "retry_wait_s", retry_wait)


BATSurveyDownloadConfig = SwiftBATSurveyDownloadConfig


@dataclass(frozen=True, slots=True)
class SwiftBATSurveyMosaicConfig:
    """Optional BAT survey mosaic policy."""

    enabled: bool = False
    min_pcode: float = 0.15
    overlap_selection: bool = True
    detection_threshold: float = 3.0

    def __post_init__(self) -> None:
        pcode = float(self.min_pcode)
        threshold = float(self.detection_threshold)
        if not math.isfinite(pcode) or not 0 <= pcode <= 1:
            raise ValueError("BAT mosaic min_pcode must be between 0 and 1")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("BAT mosaic detection_threshold must be positive")
        object.__setattr__(self, "min_pcode", pcode)
        object.__setattr__(self, "detection_threshold", threshold)


BATSurveyMosaicConfig = SwiftBATSurveyMosaicConfig


@dataclass(frozen=True, slots=True)
class GBMAnalysisConfig:
    """Fermi/GBM continuous-data analysis policy for one target.

    Mirrors the gated workflow of the reference ``batsurvey_lmjagn`` GBM
    branch: geometry-selected detectors, flanking polynomial background
    windows, an official or pure-Python response generator, and fixed-shape
    power-law profile likelihoods.  No flux or upper limit is ever published
    without a validated PHA/BAK/RSP triplet.
    """

    response_backend: Literal["auto", "official", "gbm_drm_gen"] = "auto"
    response_generator_executable: str = "SA_GBM_RSP_Gen.pl"
    # A response generator is an external scientific task.  Keep the default
    # bounded but long enough for a real TTE/CSPEC invocation on a shared
    # workstation; callers can lower it for tests.
    response_timeout_s: float = 3600.0
    rsp2_delta_time_s: float = 3.0
    max_nai_angle_deg: float = 60.0
    max_nai_detectors: int = 3
    max_bgo_angle_deg: float = 90.0
    max_bgo_detectors: int = 2
    background_guard_min_s: float = 10.0
    background_context_min_s: float = 300.0
    background_context_max_s: float = 1800.0
    photon_index_primary: float = 2.0
    photon_index_sensitivity: tuple[float, ...] = (1.5, 2.5)
    nai_band_keV: tuple[float, float] = (8.0, 900.0)
    bgo_band_keV: tuple[float, float] = (200.0, 40000.0)
    bat_comparable_band_keV: tuple[float, float] = (14.0, 195.0)
    lc_bin_s: float = 1.0
    significance_threshold_sigma: float = 3.0
    # Kept as an explicit opt-in compatibility knob.  A fixed 5% default is
    # not scientifically calibrated for continuous GBM background models.
    fractional_systematic_floor: float | None = None

    def __post_init__(self) -> None:
        if self.response_backend not in {"auto", "official", "gbm_drm_gen"}:
            raise ValueError(
                "GBM response_backend must be 'auto', 'official' or 'gbm_drm_gen'"
            )
        for name in (
            "response_timeout_s",
            "rsp2_delta_time_s",
            "max_nai_angle_deg",
            "max_bgo_angle_deg",
            "background_guard_min_s",
            "background_context_min_s",
            "background_context_max_s",
            "photon_index_primary",
            "lc_bin_s",
            "significance_threshold_sigma",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"GBM analysis {name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not 0.0 < self.max_nai_angle_deg <= 180.0 or not 0.0 < self.max_bgo_angle_deg <= 180.0:
            raise ValueError("GBM detector angle limits must be in (0, 180] degrees")
        if int(self.max_nai_detectors) < 1 or int(self.max_bgo_detectors) < 1:
            raise ValueError("GBM detector counts must be positive")
        object.__setattr__(self, "max_nai_detectors", int(self.max_nai_detectors))
        object.__setattr__(self, "max_bgo_detectors", int(self.max_bgo_detectors))
        if self.background_context_min_s > self.background_context_max_s:
            raise ValueError(
                "GBM background_context_min_s must not exceed background_context_max_s"
            )
        for name in ("nai_band_keV", "bgo_band_keV", "bat_comparable_band_keV"):
            emin, emax = (float(value) for value in getattr(self, name))
            if not math.isfinite(emin) or not math.isfinite(emax) or not 0 < emin < emax:
                raise ValueError(f"GBM analysis {name} must be finite and increasing")
            object.__setattr__(self, name, (emin, emax))
        sensitivity = tuple(float(value) for value in self.photon_index_sensitivity)
        if any(not math.isfinite(value) or value <= 0 for value in sensitivity):
            raise ValueError("GBM photon_index_sensitivity values must be finite and positive")
        object.__setattr__(self, "photon_index_sensitivity", sensitivity)
        if self.fractional_systematic_floor is not None:
            floor = float(self.fractional_systematic_floor)
            if not math.isfinite(floor) or not 0 <= floor < 1:
                raise ValueError("GBM fractional_systematic_floor must be in [0, 1)")
            object.__setattr__(self, "fractional_systematic_floor", floor)

    @property
    def photon_indices(self) -> tuple[float, ...]:
        """Primary photon index first, then the sensitivity grid."""
        grid = tuple(
            value
            for value in self.photon_index_sensitivity
            if value != self.photon_index_primary
        )
        return (self.photon_index_primary, *grid)


_INSTRUMENT_REGISTRY: dict[str, type["InstrumentConfig"]] = {}


def _registry_key(name: str) -> str:
    return name.strip().upper().replace("-", "_").replace("/", "_")


def register_instrument(cls: type["InstrumentConfig"]) -> type["InstrumentConfig"]:
    """Register an instrument config class by class name and aliases."""
    _INSTRUMENT_REGISTRY[_registry_key(cls.__name__)] = cls
    for alias in getattr(cls, "aliases", ()):
        _INSTRUMENT_REGISTRY[_registry_key(alias)] = cls
    return cls


def instrument(name: str, **kwargs: Any) -> "InstrumentConfig":
    """Build a registered instrument configuration."""
    key = _registry_key(name)
    if key not in _INSTRUMENT_REGISTRY:
        available = ", ".join(sorted(_INSTRUMENT_REGISTRY))
        raise ValueError(f"Unknown instrument: {name}. Available: {available}")
    return _INSTRUMENT_REGISTRY[key](**kwargs)


@dataclass(slots=True)
class InstrumentConfig:
    """Static instrument metadata used by data scanners and analysis defaults."""

    name: str
    mission: str
    energy_range_keV: tuple[float, float]
    detector: str | None = None
    scanner: str | None = None
    modules: tuple[str, ...] = ()
    detector_pattern: str | None = None
    group_min_counts: int | None = None
    band: str | None = None
    place: str | None = "space"
    background_type: str | None = None
    stat_method: str | None = None
    # ``response_type``＝该仪器响应文件的 OGIP 类型（0.2.0 起强制校验，只允许
    # rsp2 / drm / rsp / rmf 之一，None=未声明）：
    #   - "rsp" : 单窗口响应（TYPE-I RSP；一能量行一条重分布谱，有效面积已
    #             折算在内，无需 ARF）；
    #   - "rsp2": 多窗口响应（TYPE-II RSP；多行矩阵+时间列，时变谱标准产物，
    #             如 Fermi/GBM CSPEC/TTE 时间分段响应）；
    #   - "drm" : 任务自定义单窗口 DRM 文件（内容等价 RSP，如 gbm_drm_gen 产物）；
    #   - "rmf" : 纯重分布矩阵，必须与 ARF（有效面积）配对使用——OGIP 规定
    #             RMF 只描述光子能量→通道的重分布，有效面积存于 ARF，二者
    #             相乘才是完整响应；此时 response_requires_arf 自动置 True。
    # 参考：OGIP CAL/GEN/92-002 (George et al. 1992), "The Calibration
    #       Requirements for Spectral Analysis"（RMF/ARF 格式定义）；
    #       OGIP/92-007, "The multi-mission RMF file format"（TYPE-I/II RSP，
    #       即 .rsp/.rsp2）。两份备忘录见 HEASARC CALDB docs/memos。
    response_type: str | None = None
    # rmf 类型必须搭配 ARF 文件（IO/拟合层据此校验 ARF 是否在场）；
    # 由 __post_init__ 依 response_type 自动维护，无需手工设置。
    response_requires_arf: bool = False
    filtername: str | None = None
    pipeline: str | None = None
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    background_scaling: BackgroundScalingConfig = field(default_factory=BackgroundScalingConfig)
    duration: DurationConfig = field(default_factory=DurationConfig)
    bayesian_block_spectra: BayesianBlockSpectrumConfig = field(
        default_factory=BayesianBlockSpectrumConfig
    )
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    fitting: FitConfig = field(default_factory=FitConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)
    flux_curve: FluxCurveConfig = field(default_factory=FluxCurveConfig)
    reporting: ReportConfig = field(default_factory=ReportConfig)
    galactic_absorption: GalacticAbsorptionConfig = field(
        default_factory=GalacticAbsorptionConfig
    )
    upper_limit: UpperLimitConfig = field(default_factory=UpperLimitConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    # Swift BAT survey uses these optional groups; keeping them on the common
    # config object lets generic serializers and pipeline fingerprints handle
    # them without importing the Swift plugin.
    survey: SwiftBATSurveyConfig | None = None
    # ``selection`` is an explicit alias for the detector/partial-coding
    # selection group.  BATSurvey populates both ``survey`` and ``selection``
    # with the same immutable policy so callers can address either concern
    # without importing a Swift-specific config class.
    selection: SwiftBATSurveyConfig | None = None
    downloads: SwiftBATSurveyDownloadConfig | None = None
    mosaic: SwiftBATSurveyMosaicConfig | None = None
    # Fermi/GBM continuous-data pipelines use this optional group; keeping it
    # on the common config object lets generic serializers and pipeline
    # fingerprints handle it without importing the Fermi plugin.
    gbm_analysis: GBMAnalysisConfig | None = None
    # BXA / UltraNest nested-sampling tuning.  Kept off the process-wide
    # ``FitConfig`` singleton (which stays lean) and attached per instrument so
    # generic serializers and pipeline fingerprints handle it uniformly.
    bxa: BXAConfig = field(default_factory=BXAConfig)

    aliases: ClassVar[tuple[str, ...]] = ()

    # 允许的响应文件类型（OGIP 词表；见 response_type 字段注释与参考）
    _RESPONSE_TYPES: ClassVar[tuple[str, ...]] = ("rsp2", "drm", "rsp", "rmf")

    def __post_init__(self) -> None:
        """校验 response_type 词表并维护 rmf⇒ARF 配对约束（0.2.0）。"""
        self.energy_range_keV = self.energy_range_keV
        if (
            self.response_type is not None
            and self.response_type not in self._RESPONSE_TYPES
        ):
            allowed = ", ".join(self._RESPONSE_TYPES)
            raise ValueError(
                f"{self.name}: response_type must be one of {allowed} or None, "
                f"got {self.response_type!r}"
            )
        if self.response_type == "rmf":
            # rmf 必须有 arf 文件（OGIP CAL/GEN/92-002：RMF×ARF 才是完整响应）
            self.response_requires_arf = True
        else:
            self.response_requires_arf = False

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "energy_range_keV":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("energy_range_keV must contain two limits")
            lo, hi = (float(x) for x in value)
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo < hi):
                raise ValueError("energy_range_keV must be finite, positive and increasing")
            value = (lo, hi)
        if name == "response_type":
            if value is not None and value not in self._RESPONSE_TYPES:
                raise ValueError(f"invalid response_type: {value!r}")
            object.__setattr__(self, "response_requires_arf", value == "rmf")
        object.__setattr__(self, name, value)

    @property
    def telescope(self) -> str:
        """Backward-compatible alias for the mission name."""
        return self.mission

    @property
    def Emin_keV(self) -> float:
        return self.energy_range_keV[0]

    @property
    def Emax_keV(self) -> float:
        return self.energy_range_keV[1]

    @property
    def grouping_min_counts(self) -> int | None:
        """Backward-compatible alias for the spectrum grouping default."""
        return self.group_min_counts


@register_instrument
class FXT(InstrumentConfig):
    """Einstein Probe Follow-up X-ray Telescope."""

    aliases = ("EP_FXT",)

    def __init__(self, **kwargs: Any):
        defaults: dict[str, Any] = {
            "name": "FXT",
            "mission": "EP",
            "energy_range_keV": (0.3, 10.0),
            "scanner": "fxt",
            "modules": ("FXTA", "FXTB"),
            "group_min_counts": 3,
            "band": "X",
            "background_type": "spatial",
            "stat_method": "wstat",
            "response_type": "rmf",
            "spectrum": SpectrumConfig(group_min_counts=3, fit_energy_range_keV=(0.3, 10.0)),
            "upper_limit": UpperLimitConfig(
                strategy="spatial_onoff",
                background_likelihood="poisson_onoff",
                response_folding="rmf_arf",
                combine="joint_modules",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
                enabled=True,
                unavailable_reason=None,
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


@register_instrument
class WXT(InstrumentConfig):
    """Einstein Probe Wide-field X-ray Telescope."""

    aliases = ("EP_WXT",)

    def __init__(self, **kwargs: Any):
        defaults: dict[str, Any] = {
            "name": "WXT",
            "mission": "EP",
            "energy_range_keV": (0.5, 4.0),
            "scanner": "wxt",
            "detector_pattern": r"CMOS\d+",
            "group_min_counts": 3,
            "band": "X",
            "background_type": "spatial",
            "stat_method": "cstat",
            "response_type": "rmf",
            "pipeline": "ep.wxt.pointing",
            "upper_limit": UpperLimitConfig(
                strategy="spatial_onoff",
                background_likelihood="poisson_onoff",
                response_folding="rmf_arf",
                combine="single",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
                enabled=True,
                unavailable_reason=None,
            ),
            "extraction": ExtractionConfig(
                lightcurve_binsize_s=0.5,
                image_binsize=16,
                bands={
                    "full": EnergyBandConfig((0.5, 4.0), (50, 400)),
                    "soft": EnergyBandConfig((0.5, 1.4), (50, 140)),
                    "hard": EnergyBandConfig((1.4, 4.0), (140, 400)),
                },
            ),
            "regions": RegionConfig(
                require_review=True,
                source_radius_arcsec=540.0,
                background_sectors=(
                    (15.0, 75.0, 1080.0, 2160.0),
                    (105.0, 165.0, 1080.0, 2160.0),
                    (195.0, 255.0, 1080.0, 2160.0),
                    (285.0, 345.0, 1080.0, 2160.0),
                ),
                background_strategy="generated_cross4lobes",
                background_orientation="footprint_edge",
                background_polygon_samples=240,
            ),
            "background_scaling": BackgroundScalingConfig(),
            "duration": DurationConfig(),
            "bayesian_block_spectra": BayesianBlockSpectrumConfig(
                minimum_net_counts=10.0,
                minimum_significance=3.0,
            ),
            "spectrum": SpectrumConfig(
                group_min_counts=1,
                fit_energy_range_keV=(0.5, 4.0),
            ),
            "fitting": FitConfig(
                model_name="tbabs*ztbabs*cflux*powerlaw",
                statistic="cstat",
                abundance="wilm",
                cross_section="vern",
                model_class="auto",
                absorption_mode="auto",
                selection_metric="aicc",
                comparison_intervals=("pipeline", "t100", "t90"),
            ),
            "plotting": PlotConfig(),
            "flux_curve": FluxCurveConfig(energy_range_keV=(0.5, 4.0)),
            "reporting": ReportConfig(),
            "galactic_absorption": GalacticAbsorptionConfig(),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


@register_instrument
class BAT(InstrumentConfig):
    """Swift Burst Alert Telescope."""

    aliases = ("SWIFT_BAT",)

    def __init__(self, **kwargs: Any):
        defaults: dict[str, Any] = {
            "name": "BAT",
            "mission": "Swift",
            "energy_range_keV": (15.0, 150.0),
            "group_min_counts": 25,
            "band": "Gamma",
            "background_type": "detector_shadow",
            "stat_method": "pgstat",
            "response_type": "rsp",
            "upper_limit": UpperLimitConfig(
                strategy="coded_mask_spectrum",
                background_likelihood="gaussian_net_rate",
                response_folding="rsp",
                combine="single",
                result_modes=("observed_upper_bound",),
                enabled=True,
                unavailable_reason=None,
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


@register_instrument
class BATSurvey(InstrumentConfig):
    """Swift BAT survey single-target analysis preset.

    ``profile="lmjagn"`` is an explicit compatibility profile for the
    reference ``batsurvey_lmjagn`` project.  It is never selected implicitly:
    ordinary users get the conservative BatAnalysis-compatible thresholds.
    """

    aliases = ("SWIFT_BAT_SURVEY", "SWIFT_SURVEY", "swift.bat.survey")

    def __init__(self, *, profile: str | None = None, **kwargs: Any):
        profile_key = profile.strip().lower() if profile is not None else None
        if profile_key not in {None, "default", "lmjagn"}:
            raise ValueError("BATSurvey profile must be 'default' or 'lmjagn'")
        survey = SwiftBATSurveyConfig(
            detthresh=8000 if profile_key == "lmjagn" else 10000,
            detthresh2=8000 if profile_key == "lmjagn" else 10000,
            min_pcode=0.01 if profile_key == "lmjagn" else 0.05,
        )
        supplied_survey = kwargs.pop("survey", None)
        if supplied_survey is not None:
            if not isinstance(supplied_survey, SwiftBATSurveyConfig):
                raise TypeError("BATSurvey survey must be SwiftBATSurveyConfig or None")
            survey = supplied_survey
        supplied_selection = kwargs.pop("selection", None)
        if supplied_selection is not None:
            if not isinstance(supplied_selection, SwiftBATSurveyConfig):
                raise TypeError(
                    "BATSurvey selection must be SwiftBATSurveyConfig or None"
                )
            # When only the explicit selection group is supplied, keep the
            # historical ``survey`` alias synchronized.  Passing both groups
            # remains an intentional way to keep them distinct.
            if supplied_survey is None:
                survey = supplied_selection
        # Keep the common constructor ergonomic while making overrides
        # explicit: ``BATSurvey(detthresh=8000, processes=2)`` is equivalent
        # to passing a customized ``SwiftBATSurveyConfig``.
        survey_keys = set(SwiftBATSurveyConfig.__dataclass_fields__)
        direct_survey = {
            key: kwargs.pop(key)
            for key in tuple(survey_keys)
            if key in kwargs
        }
        if direct_survey:
            survey_values = {
                name: getattr(survey, name)
                for name in survey.__dataclass_fields__
            }
            survey_values.update(direct_survey)
            survey = SwiftBATSurveyConfig(**survey_values)
        mosaic = SwiftBATSurveyMosaicConfig(
            enabled=False,
            min_pcode=0.01 if profile_key == "lmjagn" else 0.15,
        )
        # ``UpperLimitConfig`` is the canonical runtime policy.  The older
        # survey fields remain accepted as explicit compatibility overrides,
        # but defaults are copied into the shared policy once here so fitting
        # code never has to choose between two independent values.
        supplied_upper_limit = kwargs.get("upper_limit")
        if supplied_upper_limit is None:
            upper_limit = UpperLimitConfig(
                strategy="coded_mask_spectrum",
                background_likelihood="gaussian_net_rate",
                response_folding="rsp",
                combine="single",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
                default_sigma=math.sqrt(survey.upper_limit_delta_stat),
                spectral_index=survey.upper_limit_photon_index,
                calibration="empirical_if_available",
                calibration_mode="empirical_if_available",
                enabled=True,
                unavailable_reason=None,
            )
        else:
            upper_limit = supplied_upper_limit
        defaults: dict[str, Any] = {
            "name": "BATSurvey" if profile_key != "lmjagn" else "BATSurvey_lmjagn",
            "mission": "Swift",
            # The detector's calibrated survey product band is 14--195 keV;
            # this intentionally differs from the Burst Analyser BATBand.
            "energy_range_keV": survey.energy_range_keV,
            "group_min_counts": 1,
            "band": "Gamma",
            "background_type": "coded_mask_survey_rate",
            "stat_method": "chi",
            "response_type": "rsp",
            "pipeline": "swift.bat.survey",
            "extraction": ExtractionConfig(
                backend="batanalysis",
                time_format="swift",
                bands={"survey": EnergyBandConfig(survey.energy_range_keV)},
            ),
            "spectrum": SpectrumConfig(
                group_min_counts=1,
                fit_energy_range_keV=survey.energy_range_keV,
            ),
            "fitting": FitConfig(
                model_name="cflux*powerlaw",
                statistic="chi",
                abundance="wilm",
                cross_section="vern",
                calculate_errors=True,
                error_delta_stat=1.0,
                model_class="powerlaw",
                absorption_mode="none",
                selection_metric="none",
                comparison_intervals=("survey",),
            ),
            "flux_curve": FluxCurveConfig(energy_range_keV=survey.energy_range_keV),
            "plotting": PlotConfig(),
            "reporting": ReportConfig(),
            "upper_limit": upper_limit,
            "execution": ExecutionConfig(command_timeout_s=survey.task_timeout_s),
            "survey": survey,
            "selection": supplied_selection or survey,
            "downloads": SwiftBATSurveyDownloadConfig(),
            "mosaic": mosaic,
        }
        if profile_key == "lmjagn":
            defaults["name"] = "BATSurvey_lmjagn"
        defaults.update(kwargs)
        if supplied_selection is None:
            defaults["selection"] = defaults["survey"]
        super().__init__(**defaults)


@register_instrument
@dataclass(slots=True)
class SwiftGRB(InstrumentConfig):
    """Swift BAT+XRT single-GRB analysis preset.

    This is intentionally a core-only configuration object.  The concrete
    ``jinwu.swift.grb`` implementation remains in the optional jinwu-swift
    package and is discovered through the normal pipeline entry point.
    """

    name: str = "Swift/BAT+XRT"
    mission: str = "Swift"
    energy_range_keV: tuple[float, float] = (0.3, 150.0)
    group_min_counts: int | None = 20
    band: str | None = "BAT+XRT"
    background_type: str | None = "xrt_event_area"
    stat_method: str | None = "cstat"
    # XRT 光谱产品为 RMF+ARF 配对（swxrt*.rmf/.arf），按 OGIP 词表声明为 "rmf"
    # （旧值 "rmf_arf" 不在词表内，0.2.0 规范化；ARF 配对约束由
    # response_requires_arf=True 表达）。
    response_type: str | None = "rmf"
    pipeline: str | None = "swift.grb"
    extraction: ExtractionConfig = field(
        default_factory=lambda: ExtractionConfig(
            time_format="scc",
            bands={"XRTBand": EnergyBandConfig((0.3, 10.0))},
        )
    )
    spectrum: SpectrumConfig = field(
        default_factory=lambda: SpectrumConfig(
            group_min_counts=20, fit_energy_range_keV=(0.3, 10.0)
        )
    )
    fitting: FitConfig = field(
        default_factory=lambda: FitConfig(
            model_name="tbabs*ztbabs*cflux*powerlaw",
            statistic="cstat",
            abundance="wilm",
            cross_section="vern",
            error_delta_stat=1.0,
        )
    )
    data: SwiftGRBDataConfig = field(default_factory=SwiftGRBDataConfig)
    xrt_request: SwiftXRTRequestConfig = field(default_factory=SwiftXRTRequestConfig)
    segmentation: SwiftGRBSegmentationConfig = field(default_factory=SwiftGRBSegmentationConfig)

    aliases: ClassVar[tuple[str, ...]] = ("SWIFT_GRB", "SWIFT_BAT_XRT")


@register_instrument
class GBM(InstrumentConfig):
    """Fermi Gamma-ray Burst Monitor detector config."""

    aliases = ("FERMI_GBM",)
    detectors = {
        "NAI_1": (8.0, 1000.0),
        "NAI_2": (8.0, 1000.0),
        "NAI_3": (8.0, 1000.0),
        "NAI_4": (8.0, 1000.0),
        "NAI_5": (8.0, 1000.0),
        "NAI_6": (8.0, 1000.0),
        "BGO_1": (200.0, 40000.0),
        "BGO_2": (200.0, 40000.0),
        # Native GBM archive identifiers.  Keep the historical NAI_*/BGO_*
        # aliases above for backward compatibility, but allow every physical
        # detector to be represented by its FITS/GDT name.
        **{f"N{index}": (8.0, 900.0) for index in range(10)},
        "NA": (8.0, 900.0),
        "NB": (8.0, 900.0),
        "B0": (200.0, 40000.0),
        "B1": (200.0, 40000.0),
    }

    def __init__(self, detector: str = "NAI_1", **kwargs: Any):
        detector = detector.upper()
        if detector not in self.detectors:
            choices = ", ".join(sorted(self.detectors))
            raise ValueError(f"Unknown GBM detector: {detector}. Available: {choices}")
        defaults: dict[str, Any] = {
            "name": f"GBM_{detector}",
            "mission": "Fermi",
            "detector": detector,
            "energy_range_keV": self.detectors[detector],
            "group_min_counts": 25,
            "band": "Gamma",
            "background_type": "temporal",
            "stat_method": "pgstat",
            # GBM 爆发谱（TTE/CSPEC 时间分段）的官方响应为多窗口 TYPE-II RSP
            # （.rsp2，每时间 bin 一行 DRM）；单窗口 .rsp 亦可读取。
            # 参考：HEASARC GBM data products 响应文件说明；OGIP/92-007。
            "response_type": "rsp2",
            "spectrum": SpectrumConfig(group_min_counts=25, fit_energy_range_keV=self.detectors[detector]),
            "upper_limit": UpperLimitConfig(
                strategy="modeled_count_spectrum",
                background_likelihood="poisson_gaussian_profile",
                response_folding="rsp",
                combine="joint_detectors",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
                spectral_index=2.0,
                calibration="empirical_if_available",
                calibration_mode="empirical_if_available",
                enabled=True,
                unavailable_reason=None,
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.detector = detector


@register_instrument
class GBMContinuous(InstrumentConfig):
    """Fermi/GBM continuous-data single-target analysis preset.

    Drives the ``"fermi.gbm"`` pipeline implemented in the optional
    ``jinwu-fermi`` package: poshist coverage checks, geometry-based detector
    selection, TTE/CSPEC retrieval, flanking polynomial backgrounds, dual
    response generation (official ``SA_GBM_RSP_Gen.pl`` or pure-Python
    ``gbm_drm_gen``) and fixed-shape power-law profile likelihoods.
    """

    aliases = ("FERMI_GBM_CONTINUOUS", "fermi.gbm")

    def __init__(self, **kwargs: Any):
        analysis = GBMAnalysisConfig()
        supplied_analysis = kwargs.pop("gbm_analysis", None)
        if supplied_analysis is not None:
            if not isinstance(supplied_analysis, GBMAnalysisConfig):
                raise TypeError(
                    "GBMContinuous gbm_analysis must be GBMAnalysisConfig or None"
                )
            analysis = supplied_analysis
        # Keep the common constructor ergonomic: ``GBMContinuous(lc_bin_s=0.5)``
        # is equivalent to passing a customized ``GBMAnalysisConfig``.
        analysis_keys = set(GBMAnalysisConfig.__dataclass_fields__)
        direct_analysis = {
            key: kwargs.pop(key) for key in tuple(analysis_keys) if key in kwargs
        }
        if direct_analysis:
            analysis_values = {
                name: getattr(analysis, name)
                for name in analysis.__dataclass_fields__
            }
            analysis_values.update(direct_analysis)
            analysis = GBMAnalysisConfig(**analysis_values)
        defaults: dict[str, Any] = {
            "name": "GBMContinuous",
            "mission": "Fermi",
            "energy_range_keV": analysis.nai_band_keV,
            "group_min_counts": 25,
            "band": "Gamma",
            "background_type": "temporal",
            "stat_method": "pgstat",
            # 连续数据管线按 rsp2_delta_time_s 生成时间分段响应（gbm_drm_gen
            # 的 create_rsp2 / 官方 SA_GBM_RSP_Gen.pl），文件类型为 .rsp2。
            "response_type": "rsp2",
            "pipeline": "fermi.gbm",
            "spectrum": SpectrumConfig(group_min_counts=25, fit_energy_range_keV=analysis.nai_band_keV),
            "fitting": FitConfig(
                model_name="powerlaw",
                statistic="pgstat",
                absorption_mode="none",
                selection_metric="none",
            ),
            "upper_limit": UpperLimitConfig(
                strategy="modeled_count_spectrum",
                background_likelihood="poisson_gaussian_profile",
                response_folding="rsp",
                combine="joint_detectors",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
                default_sigma=analysis.significance_threshold_sigma,
                spectral_index=analysis.photon_index_primary,
                calibration="empirical_if_available",
                calibration_mode="empirical_if_available",
                fractional_background_systematic=None,
                enabled=True,
                unavailable_reason=None,
            ),
            "gbm_analysis": analysis,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


@register_instrument
class GECAM(InstrumentConfig):
    """GECAM detector config with an explicit calibrated analysis band."""

    aliases = ("GECAM_A", "GECAM_B")

    def __init__(
        self,
        *,
        detector: str | None = None,
        energy_range_keV: tuple[float, float] | None = None,
        **kwargs: Any,
    ):
        if detector is None or energy_range_keV is None:
            raise ValueError(
                "GECAM requires explicit detector and calibrated energy_range_keV; "
                "no scientifically reliable defaults are inferred"
            )
        detector_key = str(detector).strip().upper()
        if not detector_key:
            raise ValueError("GECAM detector must not be empty")
        emin, emax = (float(value) for value in energy_range_keV)
        if not (math.isfinite(emin) and math.isfinite(emax) and 0 < emin < emax):
            raise ValueError("GECAM energy_range_keV must be finite, positive and increasing")
        defaults: dict[str, Any] = {
            "name": f"GECAM_{detector_key}",
            "mission": "GECAM",
            "detector": detector_key,
            "energy_range_keV": (emin, emax),
            "group_min_counts": 25,
            "band": "Gamma",
            "background_type": "temporal",
            "stat_method": "pgstat",
            "response_type": "rsp",
            "spectrum": SpectrumConfig(group_min_counts=25, fit_energy_range_keV=(emin, emax)),
            "upper_limit": UpperLimitConfig(
                strategy="modeled_count_spectrum",
                background_likelihood="poisson_gaussian_profile",
                response_folding="rsp",
                combine="joint_detectors",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
                spectral_index=2.0,
                calibration="empirical_if_available",
                calibration_mode="empirical_if_available",
                fractional_background_systematic=None,
                enabled=True,
                unavailable_reason=None,
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.detector = detector_key


@register_instrument
class UVOT(InstrumentConfig):
    """Swift UV/Optical Telescope filter config."""

    aliases = ("SWIFT_UVOT",)
    filters = {
        "V": (0.0023, 0.0035),
        "B": (0.0025, 0.0035),
        "U": (0.0030, 0.0042),
        "UVW1": (0.0032, 0.0045),
        "UVM2": (0.0040, 0.0050),
        "UVW2": (0.0045, 0.0060),
        "WHITE": (0.0020, 0.0060),
    }

    def __init__(self, filter: str = "V", **kwargs: Any):
        filter_key = filter.upper()
        if filter_key not in self.filters:
            choices = ", ".join(sorted(self.filters))
            raise ValueError(f"Unknown UVOT filter: {filter}. Available: {choices}")
        defaults: dict[str, Any] = {
            "name": f"UVOT_{filter_key}",
            "mission": "Swift",
            "energy_range_keV": self.filters[filter_key],
            "band": "UV/Optical/IR",
            "background_type": "spatial",
            "filtername": filter_key,
            "upper_limit": UpperLimitConfig(
                strategy="unsupported",
                enabled=False,
                result_modes=(),
                unavailable_reason=(
                    "UVOT requires a photometric upper-limit backend; the high-energy "
                    "count-spectrum engine is not applicable."
                ),
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# ---------------------------------------------------------------------------
# Process-wide fit settings (module-level singleton).
#
# A single immutable :class:`FitConfig` snapshot is the default source for bare
# ``fit_prepared`` / ``fit_xray_models`` / ``fit_spectral`` calls that do not
# pass an explicit value.  Module-level global state is an established jinwu
# convention (``_INSTRUMENT_REGISTRY`` here, ``_PIPELINES`` in pipeline.py,
# ``_default_registry`` in fit.py).  Reading the frozen snapshot is thread safe;
# mutation only happens through the explicit setters / context manager below.
# ---------------------------------------------------------------------------

_FIT_SETTINGS: FitConfig = FitConfig()


def get_fit_settings() -> FitConfig:
    """Return the current process-wide fit settings (immutable snapshot)."""
    return _FIT_SETTINGS


def set_fit_settings(
    settings: FitConfig | None = None, /, **overrides: Any
) -> FitConfig:
    """Replace or partially update the process-wide fit settings.

    Pass a whole :class:`FitConfig` positionally, or keyword overrides that are
    applied to the current settings via :func:`dataclasses.replace`.  Returns
    the settings in effect after the call.
    """
    global _FIT_SETTINGS
    if "settings" in overrides:
        raise TypeError(
            "set_fit_settings() takes 'settings' as a positional-only "
            "argument. Call set_fit_settings(cfg) to install a whole FitConfig, "
            "or pass field overrides such as set_fit_settings(method='bxa'); "
            "'settings=' is not a valid field override."
        )
    if settings is not None and overrides:
        raise TypeError(
            "set_fit_settings accepts either a FitConfig or keyword overrides, "
            "not both"
        )
    if settings is not None:
        if not isinstance(settings, FitConfig):
            raise TypeError("settings must be a FitConfig instance or None")
        _FIT_SETTINGS = settings
    elif overrides:
        _FIT_SETTINGS = replace(_FIT_SETTINGS, **overrides)
    return _FIT_SETTINGS


def reset_fit_settings() -> None:
    """Restore the process-wide fit settings to the packaged defaults."""
    global _FIT_SETTINGS
    _FIT_SETTINGS = FitConfig()


def set_fit_method(method: Literal["mle", "chain", "bxa"]) -> FitConfig:
    """Convenience switch for the inference method (``mle``/``chain``/``bxa``)."""
    return set_fit_settings(method=method)


def get_fit_method() -> str:
    """Return the inference method currently in effect process-wide."""
    return _FIT_SETTINGS.method


@contextmanager
def fit_settings(**overrides: Any) -> Iterator[FitConfig]:
    """Temporarily override the process-wide fit settings within a ``with`` block.

    The previous settings are restored on exit, even if the block raises.
    """
    global _FIT_SETTINGS
    previous = _FIT_SETTINGS
    set_fit_settings(**overrides)
    try:
        yield _FIT_SETTINGS
    finally:
        _FIT_SETTINGS = previous
