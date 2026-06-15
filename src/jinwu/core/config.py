from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

__all__ = [
    "XSPEC_COSMO_PLANCK18",
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
            "stat_method": "wstat",
            "response_type": "rmf",
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
