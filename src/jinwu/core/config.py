from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, ClassVar, Mapping

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
    "ExecutionConfig",
    "InstrumentConfig",
    "register_instrument",
    "instrument",
    "FXT",
    "WXT",
    "BAT",
    "GBM",
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
    minimum_coverage_fraction: float = 0.1
    source_radius_arcsec: float | None = None
    background_sectors: tuple[tuple[float, float, float, float], ...] = ()


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
                minimum_coverage_fraction=0.9,
                source_radius_arcsec=547.605,
                background_sectors=(
                    (15.0, 75.0, 1095.211, 2738.025),
                    (105.0, 165.0, 1095.211, 2738.025),
                    (195.0, 255.0, 1095.211, 2738.025),
                    (285.0, 345.0, 1095.211, 2738.025),
                ),
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
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.detector = detector


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
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
