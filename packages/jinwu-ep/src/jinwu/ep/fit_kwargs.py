"""Shared EP keyword configuration for :func:`jinwu.core.fit.fit_xray_models`."""

from __future__ import annotations

from typing import Any

from ..core.config import InstrumentConfig

__all__ = ["model_fit_options"]


def model_fit_options(config: InstrumentConfig, *, plots: bool = False) -> dict[str, Any]:
    """Return the config-derived ``fit_xray_models`` keyword arguments.

    Each call site still supplies its own output directory, candidate set,
    Galactic NH, redshift, flux band, source labels, and any shared-absorption
    or reference-band options. ``plots=True`` additionally forwards the plot
    density and required-output policy for call sites that already required
    plot success; by default both keys are omitted so plot failures stay
    non-fatal.
    """
    fitting, plotting = config.fitting, config.plotting
    options = {
        "stat_method": fitting.statistic,
        "abundance": fitting.abundance,
        "cross_section": fitting.cross_section,
        "apec_root": fitting.apec_root,
        "calculate_errors": fitting.calculate_errors,
        "error_delta_stat": fitting.error_delta_stat,
        "plot_formats": plotting.formats if plotting.enabled else (),
    }
    if plots:
        options["plot_density"] = plotting.dpi
        options["plot_required"] = plotting.enabled and plotting.required
    return options
