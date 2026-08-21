from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, ClassVar, Literal, Mapping

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
    "PlotConfig",
    "FluxCurveConfig",
    "ReportConfig",
    "GalacticAbsorptionConfig",
    "UpperLimitConfig",
    "ExecutionConfig",
    "InstrumentConfig",
    "register_instrument",
    "instrument",
    "FXT",
    "WXT",
    "BAT",
    "GBM",
    "GECAM",
    "UVOT",
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


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Backend-neutral fit defaults used by instrument pipelines."""

    enabled: bool = True
    backend: str = "xspec"
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
        "known_background",
        "photometric",
    ] = "known_background"
    response_folding: Literal["rmf_arf", "rsp", "count_to_flux"] = "count_to_flux"
    combine: Literal["single", "joint_detectors", "joint_modules"] = "single"
    result_modes: tuple[str, ...] = ()
    default_sigma: float = 3.0
    detection_power: float = 0.90
    calibration: Literal["bootstrap", "asymptotic"] = "bootstrap"
    null_trials: int = 200_000
    signal_trials: int = 20_000
    fractional_background_systematic: float | None = None
    enabled: bool = False
    unavailable_reason: str | None = "No upper-limit strategy is configured."

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
        if self.calibration not in {"bootstrap", "asymptotic"}:
            raise ValueError(f"Unknown upper-limit calibration: {self.calibration}")
        allowed_modes = {"observed_upper_bound", "detection_sensitivity"}
        unknown_modes = set(self.result_modes) - allowed_modes
        if unknown_modes:
            raise ValueError(f"Unknown upper-limit result modes: {sorted(unknown_modes)}")
        if len(set(self.result_modes)) != len(self.result_modes):
            raise ValueError("upper-limit result_modes must not contain duplicates")

        sigma = float(self.default_sigma)
        power = float(self.detection_power)
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("upper-limit default_sigma must be finite and positive")
        if not math.isfinite(power) or not 0.0 < power < 1.0:
            raise ValueError("upper-limit detection_power must be between 0 and 1")
        if int(self.null_trials) <= 0 or int(self.signal_trials) <= 0:
            raise ValueError("upper-limit Monte Carlo trial counts must be positive")
        object.__setattr__(self, "default_sigma", sigma)
        object.__setattr__(self, "detection_power", power)
        object.__setattr__(self, "null_trials", int(self.null_trials))
        object.__setattr__(self, "signal_trials", int(self.signal_trials))

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
                {"gaussian_model", "known_background"},
                {"rsp"},
                {"single", "joint_detectors"},
            ),
            "coded_mask_spectrum": (
                {"gaussian_model", "known_background"},
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
    scanner: str | None = None
    modules: tuple[str, ...] = ()
    detector_pattern: str | None = None
    group_min_counts: int | None = None
    band: str | None = None
    place: str | None = "space"
    background_type: str | None = None
    stat_method: str | None = None
    response_type: str | None = None
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

    aliases: ClassVar[tuple[str, ...]] = ()

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
                background_likelihood="gaussian_model",
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
    }

    def __init__(self, detector: str = "NAI_1", **kwargs: Any):
        detector = detector.upper()
        if detector not in self.detectors:
            choices = ", ".join(sorted(self.detectors))
            raise ValueError(f"Unknown GBM detector: {detector}. Available: {choices}")
        defaults: dict[str, Any] = {
            "name": f"GBM_{detector}",
            "mission": "Fermi",
            "energy_range_keV": self.detectors[detector],
            "group_min_counts": 25,
            "band": "Gamma",
            "background_type": "temporal",
            "stat_method": "pgstat",
            "response_type": "rsp",
            "upper_limit": UpperLimitConfig(
                strategy="modeled_count_spectrum",
                background_likelihood="gaussian_model",
                response_folding="rsp",
                combine="joint_detectors",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
                enabled=True,
                unavailable_reason=None,
            ),
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.detector = detector


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
            "energy_range_keV": (emin, emax),
            "group_min_counts": 25,
            "band": "Gamma",
            "background_type": "temporal",
            "stat_method": "pgstat",
            "response_type": "rsp",
            "upper_limit": UpperLimitConfig(
                strategy="modeled_count_spectrum",
                background_likelihood="gaussian_model",
                response_folding="rsp",
                combine="joint_detectors",
                result_modes=("observed_upper_bound", "detection_sensitivity"),
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
