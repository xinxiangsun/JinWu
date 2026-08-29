"""Instrument-configured, response-aware upper-limit calculations.

The legacy :class:`UpperLimit` class remains available for direct XSPEC chain
and ``Fit.error`` work.  New analyses should use :func:`estimate_upper_limit`,
which separates an observed profile-likelihood bound from a calibrated
detection sensitivity and obtains its statistical strategy from
``InstrumentConfig.upper_limit``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import tempfile
from typing import Any, Literal, Protocol, runtime_checkable

from astropy.io import fits
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import xlogy

from jinwu.core.config import InstrumentConfig, UpperLimitConfig
from jinwu.core.fit import (
    XspecChainResult,
    _require_xspec,
    _xspec_chain_models,
    run_xspec_chain,
)

__all__ = [
    "DEFAULT_CHAIN_LEVELS",
    "DEFAULT_ONE_SIDED_CHAIN_LEVELS",
    "DEFAULT_ERROR_DELTAS",
    "UpperLimit",
    "UpperLimitPoint",
    "UpperLimitResult",
    "OneSidedLevel",
    "CountPredictor",
    "CountTemplateModel",
    "XspecCountPredictor",
    "CallablePhotonModelPredictor",
    "UpperLimitObservation",
    "ObservedUpperBound",
    "ProfileAmplitudeResult",
    "DetectionSensitivity",
    "DetectionSensitivityAdapter",
    "ResponseAwareUpperLimitResult",
    "UpperLimitStrategy",
    "register_upper_limit_strategy",
    "estimate_upper_limit",
    "profile_source_amplitude",
]


DEFAULT_CHAIN_LEVELS = {
    # Historical Jinwu defaults.  These central Gaussian coverages were used
    # directly as posterior quantiles and are retained for numerical backward
    # compatibility only.
    "1sigma": 0.6826894921370859,
    "90%": 0.9,
    "2sigma": 0.9544997361036416,
    "3sigma": 0.9973002039367398,
}

DEFAULT_ONE_SIDED_CHAIN_LEVELS = {
    "1sigma": NormalDist().cdf(1.0),
    "90%": 0.9,
    "2sigma": NormalDist().cdf(2.0),
    "3sigma": NormalDist().cdf(3.0),
}

DEFAULT_ERROR_DELTAS = {
    "1sigma": 1.0,
    # Legacy XSPEC central 90% profile interval.  The new response-aware API
    # uses OneSidedLevel instead and does not reinterpret this historical key.
    "90%": 2.706,
    "2sigma": 4.0,
    "3sigma": 9.0,
}

_CHAIN_INDEX_RE = re.compile(r"__(\d+)$")


@dataclass(frozen=True, slots=True)
class OneSidedLevel:
    """One-sided Gaussian-equivalent profile-likelihood level."""

    sigma: float
    confidence: float
    delta_stat: float
    label: str

    @classmethod
    def from_sigma(cls, sigma: float) -> "OneSidedLevel":
        value = float(sigma)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("one-sided sigma must be finite and positive")
        confidence = NormalDist().cdf(value)
        return cls(
            sigma=value,
            confidence=float(confidence),
            delta_stat=float(value * value),
            label=f"one-sided {value:g} sigma",
        )

    @classmethod
    def from_confidence(cls, confidence: float) -> "OneSidedLevel":
        value = float(confidence)
        if not math.isfinite(value) or not 0.5 < value < 1.0:
            raise ValueError("one-sided confidence must be between 0.5 and 1")
        sigma = NormalDist().inv_cdf(value)
        return cls(
            sigma=float(sigma),
            confidence=value,
            delta_stat=float(sigma * sigma),
            label=f"one-sided {100.0 * value:g}%",
        )


@runtime_checkable
class CountPredictor(Protocol):
    """Adapter that folds a fixed-shape model to expected detector counts."""

    def expected_counts(
        self,
        observation: "UpperLimitObservation",
        *,
        interval: tuple[float, float],
        energy_band: tuple[float, float],
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class CountTemplateModel:
    """A fixed-shape model represented by unit-amplitude expected counts.

    ``predictor`` may call XSPEC with deterministic ``fakeit(applyStats=False)``
    or fold a callable/native photon model through a response.  The likelihood
    engine only consumes the resulting count template and therefore remains
    independent of the response implementation.
    """

    name: str
    predictor: CountPredictor | Callable[..., np.ndarray] | None = None
    amplitude_unit: str = "model normalization"
    flux_per_amplitude: float | None = None
    flux_unit: str = "erg cm^-2 s^-1"
    fluence_per_amplitude: float | None = None
    fluence_unit: str = "erg cm^-2"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def expected_counts(
        self,
        observation: "UpperLimitObservation",
        *,
        interval: tuple[float, float],
        energy_band: tuple[float, float],
    ) -> np.ndarray:
        method = getattr(self.predictor, "expected_counts", None)
        if callable(method):
            values = method(observation, interval=interval, energy_band=energy_band)
        elif callable(self.predictor):
            values = self.predictor(
                observation,
                interval=interval,
                energy_band=energy_band,
            )
        else:
            raise TypeError(
                "model predictor must be callable or implement expected_counts when an "
                "observation does not supply unit_source_counts"
            )
        return np.asarray(values, dtype=float)


@dataclass(frozen=True, slots=True)
class XspecCountPredictor:
    """Deterministically fold one fixed XSPEC model through an OGIP response.

    The supplied parameter tuple must contain every XSPEC model parameter in
    flat component order.  Its normalization must define one amplitude unit.
    This adapter clears the process-global PyXspec data/model state and is
    therefore intended for serialized worker processes, not concurrent calls.
    """

    model_expression: str
    parameters: tuple[float, ...]
    abundance: str = "wilm"
    cross_section: str = "vern"

    def expected_counts(
        self,
        observation: "UpperLimitObservation",
        *,
        interval: tuple[float, float],
        energy_band: tuple[float, float],
    ) -> np.ndarray:
        del interval
        if observation.response_path is None:
            raise ValueError("XSPEC count prediction requires observation.response_path")
        if observation.exposure_s is None or float(observation.exposure_s) <= 0:
            raise ValueError("XSPEC count prediction requires a positive exposure_s")
        try:
            import xspec  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on HEASoft
            raise RuntimeError("PyXspec is required by XspecCountPredictor") from exc

        response = str(Path(observation.response_path).expanduser())
        arf = "" if observation.arf_path is None else str(Path(observation.arf_path).expanduser())
        xspec.AllData.clear()
        xspec.AllModels.clear()
        xspec.Xset.abund = self.abundance
        xspec.Xset.xsect = self.cross_section
        xspec.Xset.allowPrompting = False
        try:
            model = xspec.Model(self.model_expression)
            parameter_objects = []
            for component_name in model.componentNames:
                component = getattr(model, component_name)
                for parameter_name in component.parameterNames:
                    parameter_objects.append(getattr(component, parameter_name))
            if len(parameter_objects) != len(self.parameters):
                raise ValueError(
                    f"XSPEC model expects {len(parameter_objects)} parameters, "
                    f"got {len(self.parameters)}"
                )
            for parameter, value in zip(parameter_objects, self.parameters, strict=True):
                parameter.values = float(value)

            with tempfile.TemporaryDirectory(prefix="jinwu-upperlimit-") as tempdir:
                output = Path(tempdir) / "expected.pha"
                settings = xspec.FakeitSettings(
                    response=response,
                    arf=arf,
                    exposure=float(observation.exposure_s),
                    fileName=str(output),
                )
                xspec.AllData.fakeit(
                    1,
                    settings,
                    applyStats=False,
                    noWrite=False,
                )
                values, e_min, e_max = _read_fakeit_expected_counts(
                    output,
                    response_path=observation.response_path,
                )
                return _select_template_channels(
                    values,
                    e_min=e_min,
                    e_max=e_max,
                    energy_band=energy_band,
                    observation=observation,
                )
        finally:
            xspec.AllData.clear()
            xspec.AllModels.clear()


@dataclass(frozen=True, slots=True)
class CallablePhotonModelPredictor:
    """Fold a callable photon-density model through Jinwu RMF/ARF readers.

    The callable receives an energy array in keV and must return photon density
    in ``photon cm^-2 s^-1 keV^-1`` (plain arrays are interpreted in that
    unit).  Eight-point Gauss-Legendre integration is used inside every RMF
    energy bin.  Set ``matrix_includes_area=True`` for a combined RSP whose
    MATRIX already contains effective area.
    """

    photon_model: Callable[[np.ndarray], Any]
    matrix_includes_area: bool = False
    quadrature_order: int = 8

    def expected_counts(
        self,
        observation: "UpperLimitObservation",
        *,
        interval: tuple[float, float],
        energy_band: tuple[float, float],
    ) -> np.ndarray:
        del interval
        if observation.response_path is None:
            raise ValueError("Native response folding requires observation.response_path")
        if observation.exposure_s is None or float(observation.exposure_s) <= 0:
            raise ValueError("Native response folding requires a positive exposure_s")
        if int(self.quadrature_order) < 2:
            raise ValueError("quadrature_order must be at least 2")

        from jinwu.core.io import read_arf, read_rmf

        response = read_rmf(observation.response_path)
        matrix = np.asarray(response.dense_matrix, dtype=float)
        energ_lo = np.asarray(response.energ_lo, dtype=float)
        energ_hi = np.asarray(response.energ_hi, dtype=float)
        if (
            matrix.ndim != 2
            or energ_lo.ndim != 1
            or energ_lo.size == 0
            or energ_hi.shape != energ_lo.shape
            or np.any(~np.isfinite(matrix))
            or np.any(matrix < 0)
            or np.any(~np.isfinite(energ_lo))
            or np.any(~np.isfinite(energ_hi))
            or np.any(energ_hi <= energ_lo)
        ):
            raise ValueError("RMF contains an invalid energy grid or response matrix")
        if matrix.shape[0] != energ_lo.size:
            raise ValueError("RMF energy grid and MATRIX row count do not match")

        if self.matrix_includes_area:
            area_energy = None
            area_values = None
            area_range = None
        else:
            if observation.arf_path is None:
                raise ValueError("RMF folding requires arf_path unless matrix_includes_area=True")
            arf = read_arf(observation.arf_path)
            arf_lo = np.asarray(arf.energ_lo, dtype=float)
            arf_hi = np.asarray(arf.energ_hi, dtype=float)
            area_energy = 0.5 * (arf_lo + arf_hi)
            area_values = np.asarray(arf.specresp, dtype=float)
            if (
                area_energy.ndim != 1
                or area_energy.size == 0
                or area_values.shape != area_energy.shape
                or np.any(~np.isfinite(area_energy))
                or np.any(~np.isfinite(area_values))
                or np.any(area_values < 0)
                or np.any(arf_hi <= arf_lo)
            ):
                raise ValueError("ARF contains an invalid energy grid or effective area")
            area_range = (
                float(np.min(arf_lo)),
                float(np.max(arf_hi)),
            )

        nodes, weights = np.polynomial.legendre.leggauss(int(self.quadrature_order))
        centers = 0.5 * (energ_lo + energ_hi)
        half_widths = 0.5 * (energ_hi - energ_lo)
        energies = centers[:, None] + half_widths[:, None] * nodes[None, :]
        density = _photon_density_values(self.photon_model(energies.reshape(-1))).reshape(
            energies.shape
        )
        if area_energy is None:
            effective_area = 1.0
        else:
            assert area_values is not None
            effective_area = np.interp(
                energies,
                area_energy,
                area_values,
                left=float(area_values[0]),
                right=float(area_values[-1]),
            )
            assert area_range is not None
            effective_area = np.where(
                (energies >= area_range[0]) & (energies <= area_range[1]),
                effective_area,
                0.0,
            )
        incident_counts = (
            half_widths
            * np.sum(weights[None, :] * density * effective_area, axis=1)
            * float(observation.exposure_s)
        )
        channel_counts = incident_counts @ matrix
        e_min = None if response.e_min is None else np.asarray(response.e_min, dtype=float)
        e_max = None if response.e_max is None else np.asarray(response.e_max, dtype=float)
        return _select_template_channels(
            np.asarray(channel_counts, dtype=float),
            e_min=e_min,
            e_max=e_max,
            energy_band=energy_band,
            observation=observation,
        )


@dataclass(frozen=True, slots=True)
class UpperLimitObservation:
    """One detector/module count spectrum and its background measurement."""

    name: str
    source_counts: np.ndarray
    unit_source_counts: np.ndarray | None = None
    background_counts: np.ndarray | None = None
    alpha: float | np.ndarray | None = None
    background_model: np.ndarray | None = None
    background_sigma: np.ndarray | None = None
    background_covariance: np.ndarray | None = None
    exposure_s: float | None = None
    response_path: str | Path | None = None
    arf_path: str | Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ObservedUpperBound:
    """Observed-data profile-likelihood bound for a non-negative amplitude."""

    amplitude_mle: float
    amplitude_upper: float
    amplitude_unit: str
    flux_upper: float | None
    flux_unit: str | None
    fluence_upper: float | None
    fluence_unit: str | None
    level: OneSidedLevel
    fit_statistic: float
    null_statistic: float
    status: str = "ready"


@dataclass(frozen=True, slots=True)
class ProfileAmplitudeResult:
    """Constrained source-amplitude profile fit for detection classification.

    The amplitude is constrained to be non-negative, so ``significance_sigma``
    is a one-sided asymptotic ``sqrt(DeltaStat)`` diagnostic.  It is not an
    ON/OFF Li--Ma significance and callers must retain background/response
    validation before calling a result a detection.
    """

    amplitude_mle: float
    amplitude_unit: str
    null_statistic: float
    fit_statistic: float
    significance_sigma: float
    interval: tuple[float, float]
    energy_band_keV: tuple[float, float]
    instrument: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectionSensitivity:
    """Minimum amplitude recovered at the configured detection probability."""

    amplitude: float | None
    amplitude_unit: str
    flux: float | None
    flux_unit: str | None
    fluence: float | None
    fluence_unit: str | None
    threshold: float | None
    false_alarm_confidence: float
    target_power: float
    achieved_power: float | None
    null_trials: int
    signal_trials: int
    status: Literal["ready", "unavailable", "failed"]
    empirical_false_alarm_probability: float | None = None
    reason: str | None = None


@runtime_checkable
class DetectionSensitivityAdapter(Protocol):
    """Instrument search-statistic adapter for detection sensitivity.

    Coded-mask instruments should implement their native detector-plane/search
    statistic here rather than substituting the generic count-spectrum
    likelihood ratio.  The observations contain a validated, response-folded
    source template but retain the caller's raw background uncertainty.  The
    adapter must apply ``policy.fractional_background_systematic`` exactly once.
    """

    def estimate_sensitivity(
        self,
        observations: Sequence[UpperLimitObservation],
        *,
        model: CountTemplateModel | str | Any,
        policy: UpperLimitConfig,
        level: OneSidedLevel,
        interval: tuple[float, float],
        energy_band: tuple[float, float],
        seed: int,
    ) -> DetectionSensitivity | tuple[DetectionSensitivity, Mapping[str, np.ndarray]]: ...


@dataclass(slots=True)
class ResponseAwareUpperLimitResult:
    """Complete response-aware upper-limit result and persisted artifacts."""

    instrument: str
    strategy: str
    interval: tuple[float, float]
    energy_band_keV: tuple[float, float]
    model_name: str
    observed_upper_bound: ObservedUpperBound | None
    detection_sensitivity: DetectionSensitivity | None
    config: dict[str, Any]
    observations: list[dict[str, Any]]
    provenance: dict[str, Any]
    warnings: list[str]
    likelihood_scan: dict[str, np.ndarray]
    sensitivity_diagnostics: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    artifacts: dict[str, str] = field(default_factory=dict)
    status: str = "upper_limits_ready"

    def write(self, output_dir: str | Path, *, plots: bool = True) -> dict[str, str]:
        """Write machine-readable results and compact diagnostic figures."""
        outdir = Path(output_dir).expanduser()
        outdir.mkdir(parents=True, exist_ok=True)

        scan_path = outdir / "likelihood_scan.csv"
        amplitudes = np.asarray(self.likelihood_scan.get("amplitude", []), dtype=float)
        delta_stat = np.asarray(self.likelihood_scan.get("delta_stat", []), dtype=float)
        with scan_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("amplitude", "delta_stat"))
            writer.writerows(zip(amplitudes, delta_stat, strict=True))

        summary_path = outdir / "upper_limit_summary.txt"
        summary_path.write_text(_render_summary(self), encoding="utf-8")

        artifacts = {
            "likelihood_scan_csv": str(scan_path),
            "summary": str(summary_path),
        }
        if self.sensitivity_diagnostics:
            trials_path = outdir / "sensitivity_trials.npz"
            np.savez_compressed(trials_path, **self.sensitivity_diagnostics)
            artifacts["sensitivity_trials"] = str(trials_path)
        if plots:
            try:
                artifacts.update(_write_diagnostic_plots(self, outdir))
            except Exception as exc:  # plotting must not invalidate inference
                self.warnings.append(f"Diagnostic plotting failed: {exc}")
        json_path = outdir / "upper_limit.json"
        artifacts["json"] = str(json_path)
        self.artifacts.update(artifacts)
        json_path.write_text(
            json.dumps(_result_to_json(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return dict(self.artifacts)


@dataclass(frozen=True, slots=True)
class UpperLimitStrategy:
    """Registered statistical behavior for one config strategy key."""

    key: str
    supported_likelihoods: tuple[str, ...]
    supports_sensitivity: bool


_UPPER_LIMIT_STRATEGIES: dict[str, UpperLimitStrategy] = {}


def register_upper_limit_strategy(strategy: UpperLimitStrategy) -> UpperLimitStrategy:
    """Register one strategy without coupling the engine to mission names."""
    if strategy.key in _UPPER_LIMIT_STRATEGIES:
        raise ValueError(f"Upper-limit strategy is already registered: {strategy.key}")
    _UPPER_LIMIT_STRATEGIES[strategy.key] = strategy
    return strategy


for _strategy in (
    UpperLimitStrategy("spatial_onoff", ("poisson_onoff",), True),
    UpperLimitStrategy(
        "modeled_count_spectrum", ("gaussian_model", "known_background"), True
    ),
    UpperLimitStrategy(
        "coded_mask_spectrum", ("gaussian_model", "known_background"), False
    ),
):
    register_upper_limit_strategy(_strategy)


@dataclass(frozen=True, slots=True)
class _PreparedObservation:
    name: str
    source_counts: np.ndarray
    unit_source_counts: np.ndarray
    background_counts: np.ndarray | None
    alpha: np.ndarray | None
    background_model: np.ndarray | None
    background_sigma: np.ndarray | None
    background_covariance: np.ndarray | None
    background_cholesky: tuple[np.ndarray, bool] | None
    background_covariance_condition: float | None
    exposure_s: float | None
    response_path: str | None
    arf_path: str | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ProfileFitState:
    amplitude_mle: float
    maximum_loglike: float
    amplitude_scale: float
    fit_statistic: float
    null_statistic: float


class _ProfileLikelihood:
    def __init__(self, observations: Sequence[_PreparedObservation], likelihood: str):
        self.observations = tuple(observations)
        self.likelihood = likelihood

    def loglike(self, amplitude: float) -> float:
        amplitude = float(amplitude)
        if not math.isfinite(amplitude) or amplitude < 0:
            return -math.inf
        total = 0.0
        for observation in self.observations:
            signal = amplitude * observation.unit_source_counts
            if self.likelihood == "poisson_onoff":
                total += _onoff_profile_loglike(observation, signal)
            elif self.likelihood == "gaussian_model":
                total += _gaussian_background_profile_loglike(observation, signal)
            elif self.likelihood == "known_background":
                assert observation.background_model is not None
                total += _poisson_loglike(
                    observation.source_counts,
                    signal + observation.background_model,
                )
            else:  # pragma: no cover - config validation protects this
                raise ValueError(f"Unsupported background likelihood: {self.likelihood}")
        return float(total)

    def fit(self) -> _ProfileFitState:
        scale = _amplitude_scale(self.observations, self.likelihood)
        high = max(4.0 * scale, 1e-8)
        result = None
        for _ in range(12):
            candidate = minimize_scalar(
                lambda value: -self.loglike(float(value)),
                bounds=(0.0, high),
                method="bounded",
                options={"xatol": max(1e-12, high * 1e-9)},
            )
            if not candidate.success:
                raise RuntimeError(f"Amplitude optimization failed: {candidate.message}")
            result = candidate
            if candidate.x < 0.95 * high:
                break
            high *= 4.0
        assert result is not None
        amplitude = max(0.0, float(result.x))
        loglike = self.loglike(amplitude)
        if not math.isfinite(loglike):
            raise RuntimeError("Amplitude optimization produced a non-finite likelihood")
        boundary_loglike = self.loglike(0.0)
        if boundary_loglike >= loglike - 1e-10 * max(1.0, abs(loglike)):
            amplitude = 0.0
            loglike = boundary_loglike
        return _ProfileFitState(
            amplitude_mle=amplitude,
            maximum_loglike=loglike,
            amplitude_scale=scale,
            fit_statistic=max(0.0, 2.0 * (self.saturated_loglike() - loglike)),
            null_statistic=max(0.0, 2.0 * (loglike - boundary_loglike)),
        )

    def fit_amplitude(self) -> tuple[float, float]:
        """Compatibility wrapper returning the fitted amplitude and log likelihood."""
        state = self.fit()
        return state.amplitude_mle, state.maximum_loglike

    def delta_stat(self, amplitude: float, fit_state: _ProfileFitState) -> float:
        value = 2.0 * (fit_state.maximum_loglike - self.loglike(amplitude))
        tolerance = 1e-8 * max(1.0, abs(fit_state.maximum_loglike))
        if value < -tolerance:
            raise RuntimeError(
                "Profile likelihood exceeds the stored maximum; amplitude optimization "
                "and likelihood scan are inconsistent"
            )
        return max(0.0, float(value))

    def upper_bound(
        self,
        level: OneSidedLevel,
        *,
        fit_state: _ProfileFitState | None = None,
    ) -> tuple[float, float, float, float]:
        state = self.fit() if fit_state is None else fit_state

        def crossing(amplitude: float) -> float:
            return self.delta_stat(amplitude, state) - level.delta_stat

        lower = state.amplitude_mle
        upper = max(
            state.amplitude_mle + state.amplitude_scale,
            2.0 * state.amplitude_mle,
            1e-8,
        )
        for _ in range(80):
            value = crossing(upper)
            if math.isfinite(value) and value >= 0:
                break
            upper *= 2.0
        else:
            raise RuntimeError("Could not bracket the profile-likelihood upper bound")

        amplitude_upper = float(brentq(crossing, lower, upper, xtol=1e-12, rtol=1e-10))
        return (
            state.amplitude_mle,
            amplitude_upper,
            state.fit_statistic,
            state.null_statistic,
        )

    def detection_statistic(self) -> float:
        return self.fit().null_statistic

    def saturated_loglike(self) -> float:
        """Return the data-only saturated log likelihood, omitting constants."""
        total = 0.0
        for observation in self.observations:
            total += _poisson_loglike(
                observation.source_counts, observation.source_counts
            )
            if self.likelihood == "poisson_onoff":
                assert observation.background_counts is not None
                total += _poisson_loglike(
                    observation.background_counts, observation.background_counts
                )
        return float(total)


def estimate_upper_limit(
    observations: UpperLimitObservation | Sequence[UpperLimitObservation],
    *,
    model: CountTemplateModel | str | Any,
    instrument_config: InstrumentConfig,
    interval: tuple[float, float],
    energy_band: tuple[float, float] | None = None,
    sensitivity_adapter: DetectionSensitivityAdapter | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
    plots: bool = True,
) -> ResponseAwareUpperLimitResult:
    """Estimate instrument-configured observed bounds and sensitivities.

    The model shape is fixed.  Each observation must either contain
    ``unit_source_counts`` or the supplied :class:`CountTemplateModel` must
    fold the model through that observation's response.  No ``method``
    argument is accepted: the strategy is selected by
    ``instrument_config.upper_limit``.
    """
    policy = instrument_config.upper_limit
    if not policy.enabled or policy.strategy == "unsupported":
        reason = policy.unavailable_reason or "Upper-limit calculation is disabled."
        raise NotImplementedError(f"{instrument_config.name}: {reason}")
    strategy = _UPPER_LIMIT_STRATEGIES.get(policy.strategy)
    if strategy is None:
        raise NotImplementedError(f"No upper-limit engine for strategy {policy.strategy!r}")
    if policy.background_likelihood not in strategy.supported_likelihoods:
        raise ValueError(
            f"Strategy {policy.strategy} does not support "
            f"{policy.background_likelihood}"
        )
    _validate_instrument_policy(instrument_config, policy)

    interval_checked = _validate_interval(interval)
    band_checked = _validate_energy_band(
        instrument_config.energy_range_keV if energy_band is None else energy_band
    )
    raw_observations = (
        [observations]
        if isinstance(observations, UpperLimitObservation)
        else list(observations)
    )
    if not raw_observations:
        raise ValueError("At least one upper-limit observation is required")
    names = [str(item.name) for item in raw_observations]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Upper-limit observation names must be non-empty and unique")
    _validate_combination_count(policy, len(raw_observations))

    model_info = _model_information(model)
    warnings_list: list[str] = []
    prepared = [
        _prepare_observation(
            observation,
            model=model,
            interval=interval_checked,
            energy_band=band_checked,
            policy=policy,
            warnings_list=warnings_list,
        )
        for observation in raw_observations
    ]
    likelihood = _ProfileLikelihood(prepared, policy.background_likelihood)
    level = OneSidedLevel.from_sigma(policy.default_sigma)

    observed = None
    scan: dict[str, np.ndarray] = {
        "amplitude": np.asarray([], dtype=float),
        "delta_stat": np.asarray([], dtype=float),
    }
    if "observed_upper_bound" in policy.result_modes:
        fit_state = likelihood.fit()
        amplitude_mle, amplitude_upper, fit_statistic, null_statistic = (
            likelihood.upper_bound(level, fit_state=fit_state)
        )
        observed = ObservedUpperBound(
            amplitude_mle=amplitude_mle,
            amplitude_upper=amplitude_upper,
            amplitude_unit=model_info["amplitude_unit"],
            flux_upper=_scaled_value(amplitude_upper, model_info["flux_per_amplitude"]),
            flux_unit=model_info["flux_unit"],
            fluence_upper=_scaled_value(
                amplitude_upper, model_info["fluence_per_amplitude"]
            ),
            fluence_unit=model_info["fluence_unit"],
            level=level,
            fit_statistic=fit_statistic,
            null_statistic=null_statistic,
        )
        scan_amplitudes = np.linspace(0.0, max(1e-12, 1.25 * amplitude_upper), 240)
        scan = {
            "amplitude": scan_amplitudes,
            "delta_stat": np.asarray(
                [likelihood.delta_stat(value, fit_state) for value in scan_amplitudes],
                dtype=float,
            ),
        }

    sensitivity = None
    sensitivity_diagnostics: dict[str, np.ndarray] = {}
    if "detection_sensitivity" in policy.result_modes:
        if sensitivity_adapter is not None:
            adapter_result = sensitivity_adapter.estimate_sensitivity(
                [
                    _adapter_observation(raw, item)
                    for raw, item in zip(raw_observations, prepared, strict=True)
                ],
                model=model,
                policy=policy,
                level=level,
                interval=interval_checked,
                energy_band=band_checked,
                seed=int(seed),
            )
            if isinstance(adapter_result, tuple):
                sensitivity, adapter_diagnostics = adapter_result
                sensitivity_diagnostics = {
                    str(key): np.asarray(value)
                    for key, value in adapter_diagnostics.items()
                }
            else:
                sensitivity = adapter_result
            if not isinstance(sensitivity, DetectionSensitivity):
                raise TypeError(
                    "sensitivity_adapter must return DetectionSensitivity or "
                    "(DetectionSensitivity, diagnostics)"
                )
        elif not strategy.supports_sensitivity:
            sensitivity = DetectionSensitivity(
                amplitude=None,
                amplitude_unit=model_info["amplitude_unit"],
                flux=None,
                flux_unit=model_info["flux_unit"],
                fluence=None,
                fluence_unit=model_info["fluence_unit"],
                threshold=None,
                false_alarm_confidence=level.confidence,
                target_power=policy.detection_power,
                achieved_power=None,
                null_trials=0,
                signal_trials=0,
                status="unavailable",
                empirical_false_alarm_probability=None,
                reason=(
                    "This strategy requires an instrument-specific search-statistic "
                    "adapter; a count-spectrum likelihood ratio is not a valid substitute."
                ),
            )
        else:
            sensitivity, sensitivity_diagnostics = _estimate_sensitivity(
                prepared,
                likelihood_name=policy.background_likelihood,
                policy=policy,
                level=level,
                model_info=model_info,
                initial_amplitude=(
                    observed.amplitude_upper if observed is not None else None
                ),
                seed=int(seed),
                warnings_list=warnings_list,
            )

    result_status = "upper_limits_ready"
    if sensitivity is not None and sensitivity.status != "ready":
        result_status = "partial"
    result = ResponseAwareUpperLimitResult(
        instrument=instrument_config.name,
        strategy=policy.strategy,
        interval=interval_checked,
        energy_band_keV=band_checked,
        model_name=model_info["name"],
        observed_upper_bound=observed,
        detection_sensitivity=sensitivity,
        config=_json_safe(asdict(policy)),
        observations=[_observation_summary(item) for item in prepared],
        provenance={
            "calculator": "jinwu.core.upperlimit.estimate_upper_limit",
            "random_seed": int(seed),
            "model": _json_safe(model_info["metadata"]),
            "response_folding": policy.response_folding,
            "background_likelihood": policy.background_likelihood,
            "combine": policy.combine,
            "numerical_policy": {
                "covariance_condition_warning": 1e8,
                "covariance_condition_reject": 1e12,
                "covariance_solver": "cholesky",
            },
            "adapter_observation_contract": (
                None
                if sensitivity_adapter is None
                else "validated_template_raw_background_v1"
            ),
            "sensitivity_adapter": (
                None
                if sensitivity_adapter is None
                else f"{type(sensitivity_adapter).__module__}.{type(sensitivity_adapter).__qualname__}"
            ),
        },
        warnings=warnings_list,
        likelihood_scan=scan,
        sensitivity_diagnostics=sensitivity_diagnostics,
        status=result_status,
    )
    if output_dir is not None:
        result.write(output_dir, plots=plots)
    return result


def profile_source_amplitude(
    observations: UpperLimitObservation | Sequence[UpperLimitObservation],
    *,
    model: CountTemplateModel | str | Any,
    instrument_config: InstrumentConfig,
    interval: tuple[float, float],
    energy_band: tuple[float, float] | None = None,
) -> ProfileAmplitudeResult:
    """Fit a non-negative response-folded source amplitude.

    This is the detection-side counterpart to :func:`estimate_upper_limit`.
    It deliberately reuses the same instrument policy, response template and
    Gaussian-model background likelihood, so a pipeline cannot accidentally
    classify an event with one statistical contract and limit it with another.
    """
    policy = instrument_config.upper_limit
    if not policy.enabled or policy.strategy == "unsupported":
        reason = policy.unavailable_reason or "Profile likelihood is disabled."
        raise NotImplementedError(f"{instrument_config.name}: {reason}")
    strategy = _UPPER_LIMIT_STRATEGIES.get(policy.strategy)
    if strategy is None:
        raise NotImplementedError(f"No upper-limit engine for strategy {policy.strategy!r}")
    if policy.background_likelihood not in strategy.supported_likelihoods:
        raise ValueError(
            f"Strategy {policy.strategy} does not support "
            f"{policy.background_likelihood}"
        )
    _validate_instrument_policy(instrument_config, policy)
    interval_checked = _validate_interval(interval)
    band_checked = _validate_energy_band(
        instrument_config.energy_range_keV if energy_band is None else energy_band
    )
    raw_observations = (
        [observations]
        if isinstance(observations, UpperLimitObservation)
        else list(observations)
    )
    if not raw_observations:
        raise ValueError("At least one profile observation is required")
    names = [str(item.name) for item in raw_observations]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Profile observation names must be non-empty and unique")
    _validate_combination_count(policy, len(raw_observations))
    warnings_list: list[str] = []
    prepared = [
        _prepare_observation(
            observation,
            model=model,
            interval=interval_checked,
            energy_band=band_checked,
            policy=policy,
            warnings_list=warnings_list,
        )
        for observation in raw_observations
    ]
    state = _ProfileLikelihood(prepared, policy.background_likelihood).fit()
    model_info = _model_information(model)
    return ProfileAmplitudeResult(
        amplitude_mle=state.amplitude_mle,
        amplitude_unit=str(model_info["amplitude_unit"]),
        null_statistic=state.null_statistic,
        fit_statistic=state.fit_statistic,
        significance_sigma=math.sqrt(max(0.0, state.null_statistic)),
        interval=interval_checked,
        energy_band_keV=band_checked,
        instrument=instrument_config.name,
        warnings=tuple(warnings_list),
    )


def _prepare_observation(
    observation: UpperLimitObservation,
    *,
    model: CountTemplateModel | str | Any,
    interval: tuple[float, float],
    energy_band: tuple[float, float],
    policy: UpperLimitConfig,
    warnings_list: list[str],
) -> _PreparedObservation:
    source = _one_dimensional_array(observation.source_counts, "source_counts", nonnegative=True)
    if observation.unit_source_counts is not None:
        template = _one_dimensional_array(
            observation.unit_source_counts,
            "unit_source_counts",
            nonnegative=True,
        )
    elif isinstance(model, CountTemplateModel):
        template = _one_dimensional_array(
            model.expected_counts(
                observation,
                interval=interval,
                energy_band=energy_band,
            ),
            "model expected counts",
            nonnegative=True,
        )
    else:
        method = getattr(model, "expected_counts", None)
        if not callable(method):
            raise ValueError(
                f"Observation {observation.name!r} has no unit_source_counts and the "
                "model does not implement expected_counts"
            )
        template = _one_dimensional_array(
            method(observation, interval=interval, energy_band=energy_band),
            "model expected counts",
            nonnegative=True,
        )
    if template.shape != source.shape:
        raise ValueError(
            f"Observation {observation.name!r} source/template shape mismatch: "
            f"{source.shape} != {template.shape}"
        )
    if not np.any(template > 0):
        raise ValueError(f"Observation {observation.name!r} has a zero source template")

    background_counts = None
    alpha = None
    background_model = None
    background_sigma = None
    background_covariance = None
    background_cholesky = None
    background_covariance_condition = None

    if policy.background_likelihood == "poisson_onoff":
        if observation.background_counts is None or observation.alpha is None:
            raise ValueError(
                f"Observation {observation.name!r} requires background_counts and alpha"
            )
        background_counts = _one_dimensional_array(
            observation.background_counts,
            "background_counts",
            nonnegative=True,
        )
        if background_counts.shape != source.shape:
            raise ValueError(f"Observation {observation.name!r} ON/OFF shape mismatch")
        alpha = np.broadcast_to(np.asarray(observation.alpha, dtype=float), source.shape).copy()
        if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0):
            raise ValueError(f"Observation {observation.name!r} alpha must be finite and positive")
        if policy.fractional_background_systematic not in (None, 0.0):
            raise NotImplementedError(
                "A fractional ON/OFF background systematic requires an explicit nuisance "
                "model and is not silently approximated."
            )
    else:
        if observation.background_model is None:
            raise ValueError(f"Observation {observation.name!r} requires background_model")
        background_model = _one_dimensional_array(
            observation.background_model,
            "background_model",
            nonnegative=(policy.background_likelihood == "known_background"),
        )
        if background_model.shape != source.shape:
            raise ValueError(f"Observation {observation.name!r} background shape mismatch")

        if policy.background_likelihood == "gaussian_model":
            if observation.background_covariance is not None:
                covariance = np.asarray(observation.background_covariance, dtype=float)
                if covariance.shape != (source.size, source.size):
                    raise ValueError(
                        f"Observation {observation.name!r} covariance shape mismatch"
                    )
                if np.any(~np.isfinite(covariance)) or not np.allclose(
                    covariance, covariance.T, rtol=1e-10, atol=1e-12
                ):
                    raise ValueError("background_covariance must be finite and symmetric")
                if np.any(np.linalg.eigvalsh(covariance) <= 0):
                    raise ValueError("background_covariance must be positive definite")
                background_covariance = covariance.copy()
            elif observation.background_sigma is not None:
                background_sigma = _one_dimensional_array(
                    observation.background_sigma,
                    "background_sigma",
                    positive=True,
                )
                if background_sigma.shape != source.shape:
                    raise ValueError(
                        f"Observation {observation.name!r} background sigma shape mismatch"
                    )
            else:
                raise ValueError(
                    f"Observation {observation.name!r} requires background_sigma or "
                    "background_covariance"
                )

            fractional = policy.fractional_background_systematic
            if fractional not in (None, 0.0):
                extra_variance = (float(fractional) * np.abs(background_model)) ** 2
                if background_covariance is not None:
                    background_covariance = background_covariance + np.diag(extra_variance)
                else:
                    assert background_sigma is not None
                    background_sigma = np.sqrt(background_sigma**2 + extra_variance)
            if background_covariance is not None:
                eigenvalues = np.linalg.eigvalsh(background_covariance)
                if np.any(eigenvalues <= 0):
                    raise ValueError(
                        "background_covariance must remain positive definite after "
                        "applying systematic uncertainty"
                    )
                background_covariance_condition = float(
                    np.max(eigenvalues) / np.min(eigenvalues)
                )
                if background_covariance_condition > 1e12:
                    raise ValueError(
                        f"Observation {observation.name!r} background covariance is "
                        f"numerically singular (condition number "
                        f"{background_covariance_condition:.3e} > 1e12)"
                    )
                if background_covariance_condition > 1e8:
                    warnings_list.append(
                        f"Observation {observation.name!r} background covariance is "
                        f"ill-conditioned (condition number "
                        f"{background_covariance_condition:.3e}); results may be "
                        "numerically sensitive."
                    )
                try:
                    background_cholesky = cho_factor(
                        background_covariance,
                        lower=True,
                        check_finite=False,
                    )
                except np.linalg.LinAlgError as exc:
                    raise ValueError(
                        "background_covariance Cholesky factorization failed"
                    ) from exc

    exposure = None if observation.exposure_s is None else float(observation.exposure_s)
    if exposure is not None and (not math.isfinite(exposure) or exposure <= 0):
        raise ValueError(f"Observation {observation.name!r} exposure_s must be positive")
    return _PreparedObservation(
        name=str(observation.name),
        source_counts=source,
        unit_source_counts=template,
        background_counts=background_counts,
        alpha=alpha,
        background_model=background_model,
        background_sigma=background_sigma,
        background_covariance=background_covariance,
        background_cholesky=background_cholesky,
        background_covariance_condition=background_covariance_condition,
        exposure_s=exposure,
        response_path=_optional_path(observation.response_path),
        arf_path=_optional_path(observation.arf_path),
        metadata=dict(observation.metadata),
    )


def _poisson_loglike(counts: np.ndarray, expectation: np.ndarray) -> float:
    if np.any(expectation < 0) or np.any(~np.isfinite(expectation)):
        return -math.inf
    if np.any((expectation == 0) & (counts > 0)):
        return -math.inf
    return float(np.sum(xlogy(counts, expectation) - expectation))


def _onoff_profile_background(
    source_counts: np.ndarray,
    background_counts: np.ndarray,
    alpha: np.ndarray,
    signal: np.ndarray,
) -> np.ndarray:
    coefficient_a = alpha * (1.0 + alpha)
    coefficient_b = (1.0 + alpha) * signal - alpha * (
        source_counts + background_counts
    )
    coefficient_c = -background_counts * signal
    discriminant = np.maximum(0.0, coefficient_b**2 - 4.0 * coefficient_a * coefficient_c)
    return np.maximum(0.0, (-coefficient_b + np.sqrt(discriminant)) / (2.0 * coefficient_a))


def _onoff_profile_loglike(observation: _PreparedObservation, signal: np.ndarray) -> float:
    assert observation.background_counts is not None and observation.alpha is not None
    background = _onoff_profile_background(
        observation.source_counts,
        observation.background_counts,
        observation.alpha,
        signal,
    )
    return _poisson_loglike(
        observation.source_counts, signal + observation.alpha * background
    ) + _poisson_loglike(observation.background_counts, background)


def _gaussian_background_profile_loglike(
    observation: _PreparedObservation,
    signal: np.ndarray,
) -> float:
    assert observation.background_model is not None
    measured = observation.background_model
    if observation.background_covariance is None:
        assert observation.background_sigma is not None
        variance = observation.background_sigma**2
        coefficient = variance - measured - signal
        expected_total = 0.5 * (-coefficient + np.sqrt(coefficient**2 + 4.0 * observation.source_counts * variance))
        background = np.maximum(0.0, expected_total - signal)
        return _poisson_loglike(observation.source_counts, signal + background) - 0.5 * float(
            np.sum(((measured - background) / observation.background_sigma) ** 2)
        )

    cholesky = observation.background_cholesky
    if cholesky is None:  # pragma: no cover - validated during preparation
        raise RuntimeError("Prepared covariance is missing its Cholesky factor")

    def objective(background: np.ndarray) -> float:
        poisson = _poisson_loglike(observation.source_counts, signal + background)
        if not math.isfinite(poisson):
            return math.inf
        difference = measured - background
        solved = cho_solve(cholesky, difference, check_finite=False)
        return -poisson + 0.5 * float(difference @ solved)

    initial = np.maximum(
        measured,
        np.where(observation.source_counts > 0, 1e-6, 0.0),
    )
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * initial.size,
    )
    if not fitted.success:
        raise RuntimeError(f"Gaussian background profiling failed: {fitted.message}")
    return -float(fitted.fun)


def _amplitude_scale(
    observations: Sequence[_PreparedObservation],
    likelihood_name: str,
) -> float:
    template_total = sum(float(np.sum(item.unit_source_counts)) for item in observations)
    if template_total <= 0:
        raise ValueError("The combined source count template is zero")
    excess = 0.0
    noise_counts = 0.0
    for observation in observations:
        source_total = float(np.sum(observation.source_counts))
        if likelihood_name == "poisson_onoff":
            assert observation.background_counts is not None and observation.alpha is not None
            estimated_background = float(
                np.sum(observation.alpha * observation.background_counts)
            )
        else:
            assert observation.background_model is not None
            estimated_background = float(np.sum(np.maximum(observation.background_model, 0.0)))
        excess += source_total - estimated_background
        noise_counts += source_total + estimated_background
    estimate = max(0.0, excess) / template_total
    noise_scale = max(1.0, math.sqrt(max(1.0, noise_counts))) / template_total
    return max(estimate, noise_scale, 1e-8)


def _estimate_sensitivity(
    observations: Sequence[_PreparedObservation],
    *,
    likelihood_name: str,
    policy: UpperLimitConfig,
    level: OneSidedLevel,
    model_info: Mapping[str, Any],
    initial_amplitude: float | None,
    seed: int,
    warnings_list: list[str],
) -> tuple[DetectionSensitivity, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    if policy.calibration == "bootstrap":
        null_statistics = _simulate_detection_statistics(
            observations,
            likelihood_name=likelihood_name,
            amplitude=0.0,
            trials=policy.null_trials,
            rng=rng,
        )
        threshold = _empirical_detection_threshold(null_statistics, level.confidence)
        false_alarm_probability = float(np.mean(null_statistics >= threshold))
    else:
        null_statistics = np.asarray([], dtype=float)
        threshold = level.delta_stat
        false_alarm_probability = None
        warnings_list.append(
            "Detection threshold used the asymptotic likelihood-ratio approximation."
        )

    tested_amplitudes: list[float] = []
    tested_powers: list[float] = []

    def measure_power(amplitude: float) -> float:
        statistics = _simulate_detection_statistics(
            observations,
            likelihood_name=likelihood_name,
            amplitude=amplitude,
            trials=policy.signal_trials,
            rng=rng,
        )
        power = float(np.mean(statistics >= threshold))
        tested_amplitudes.append(float(amplitude))
        tested_powers.append(power)
        return power

    low = 0.0
    high = max(
        float(initial_amplitude or 0.0),
        _amplitude_scale(observations, likelihood_name),
        1e-8,
    )
    high_power = measure_power(high)
    for _ in range(30):
        if high_power >= policy.detection_power:
            break
        low = high
        high *= 2.0
        high_power = measure_power(high)
    else:
        result = DetectionSensitivity(
            amplitude=None,
            amplitude_unit=model_info["amplitude_unit"],
            flux=None,
            flux_unit=model_info["flux_unit"],
            fluence=None,
            fluence_unit=model_info["fluence_unit"],
            threshold=threshold,
            false_alarm_confidence=level.confidence,
            target_power=policy.detection_power,
            achieved_power=high_power,
            null_trials=int(null_statistics.size),
            signal_trials=policy.signal_trials,
            status="failed",
            empirical_false_alarm_probability=false_alarm_probability,
            reason="Could not bracket the requested detection power.",
        )
        return result, {
            "null_statistics": null_statistics,
            "tested_amplitudes": np.asarray(tested_amplitudes),
            "detection_powers": np.asarray(tested_powers),
        }

    for _ in range(18):
        midpoint = 0.5 * (low + high)
        midpoint_power = measure_power(midpoint)
        if midpoint_power >= policy.detection_power:
            high = midpoint
            high_power = midpoint_power
        else:
            low = midpoint

    amplitude = float(high)
    result = DetectionSensitivity(
        amplitude=amplitude,
        amplitude_unit=model_info["amplitude_unit"],
        flux=_scaled_value(amplitude, model_info["flux_per_amplitude"]),
        flux_unit=model_info["flux_unit"],
        fluence=_scaled_value(amplitude, model_info["fluence_per_amplitude"]),
        fluence_unit=model_info["fluence_unit"],
        threshold=threshold,
        false_alarm_confidence=level.confidence,
        target_power=policy.detection_power,
        achieved_power=high_power,
        null_trials=int(null_statistics.size),
        signal_trials=policy.signal_trials,
        status="ready",
        empirical_false_alarm_probability=false_alarm_probability,
    )
    diagnostics = {
        "null_statistics": null_statistics,
        "tested_amplitudes": np.asarray(tested_amplitudes, dtype=float),
        "detection_powers": np.asarray(tested_powers, dtype=float),
        "threshold": np.asarray([threshold], dtype=float),
    }
    return result, diagnostics


def _empirical_detection_threshold(statistics: np.ndarray, confidence: float) -> float:
    """Choose a threshold whose empirical ``P(T >= threshold)`` does not exceed alpha."""
    values = np.asarray(statistics, dtype=float)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("Detection-threshold statistics must be finite and non-empty")
    threshold = float(np.quantile(values, confidence, method="higher"))
    allowed = 1.0 - float(confidence)
    if float(np.mean(values >= threshold)) > allowed:
        threshold = float(np.nextafter(threshold, math.inf))
    return threshold


def _simulate_detection_statistics(
    observations: Sequence[_PreparedObservation],
    *,
    likelihood_name: str,
    amplitude: float,
    trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    statistics = np.empty(int(trials), dtype=float)
    for trial in range(int(trials)):
        simulated = [
            _simulate_observation(
                observation,
                likelihood_name=likelihood_name,
                amplitude=amplitude,
                rng=rng,
            )
            for observation in observations
        ]
        statistics[trial] = _ProfileLikelihood(
            simulated, likelihood_name
        ).detection_statistic()
    return statistics


def _simulate_observation(
    observation: _PreparedObservation,
    *,
    likelihood_name: str,
    amplitude: float,
    rng: np.random.Generator,
) -> _PreparedObservation:
    signal = amplitude * observation.unit_source_counts
    if likelihood_name == "poisson_onoff":
        assert observation.background_counts is not None and observation.alpha is not None
        background_true = _onoff_profile_background(
            observation.source_counts,
            observation.background_counts,
            observation.alpha,
            np.zeros_like(signal),
        )
        simulated_off = rng.poisson(background_true).astype(float)
        simulated_on = rng.poisson(signal + observation.alpha * background_true).astype(float)
        return replace(
            observation,
            source_counts=simulated_on,
            background_counts=simulated_off,
        )

    assert observation.background_model is not None
    background_true = np.maximum(observation.background_model, 0.0)
    simulated_on = rng.poisson(signal + background_true).astype(float)
    if likelihood_name == "known_background":
        return replace(observation, source_counts=simulated_on)
    if observation.background_covariance is not None:
        simulated_background = rng.multivariate_normal(
            background_true, observation.background_covariance
        )
    else:
        assert observation.background_sigma is not None
        simulated_background = rng.normal(background_true, observation.background_sigma)
    return replace(
        observation,
        source_counts=simulated_on,
        background_model=np.asarray(simulated_background, dtype=float),
    )


def _model_information(model: CountTemplateModel | str | Any) -> dict[str, Any]:
    if isinstance(model, CountTemplateModel):
        metadata = dict(model.metadata)
        predictor = model.predictor
        if isinstance(predictor, XspecCountPredictor):
            metadata.setdefault("predictor", "xspec")
            metadata.setdefault("model_expression", predictor.model_expression)
            metadata.setdefault("parameters", list(predictor.parameters))
            metadata.setdefault("abundance", predictor.abundance)
            metadata.setdefault("cross_section", predictor.cross_section)
            metadata.setdefault("fakeit_apply_stats", False)
        elif isinstance(predictor, CallablePhotonModelPredictor):
            callable_name = getattr(predictor.photon_model, "__qualname__", None)
            callable_module = getattr(predictor.photon_model, "__module__", None)
            metadata.setdefault("predictor", "callable_photon_model")
            metadata.setdefault(
                "callable",
                ".".join(part for part in (callable_module, callable_name) if part),
            )
            metadata.setdefault("matrix_includes_area", predictor.matrix_includes_area)
            metadata.setdefault("quadrature_order", predictor.quadrature_order)
        elif predictor is not None:
            metadata.setdefault(
                "predictor",
                f"{type(predictor).__module__}.{type(predictor).__qualname__}",
            )
        return {
            "name": model.name,
            "amplitude_unit": model.amplitude_unit,
            "flux_per_amplitude": _optional_positive(model.flux_per_amplitude),
            "flux_unit": model.flux_unit if model.flux_per_amplitude is not None else None,
            "fluence_per_amplitude": _optional_positive(model.fluence_per_amplitude),
            "fluence_unit": (
                model.fluence_unit if model.fluence_per_amplitude is not None else None
            ),
            "metadata": metadata,
        }
    return {
        "name": str(model),
        "amplitude_unit": "model normalization",
        "flux_per_amplitude": None,
        "flux_unit": None,
        "fluence_per_amplitude": None,
        "fluence_unit": None,
        "metadata": {},
    }


def _observation_summary(observation: _PreparedObservation) -> dict[str, Any]:
    return {
        "name": observation.name,
        "channels": int(observation.source_counts.size),
        "source_counts": float(np.sum(observation.source_counts)),
        "unit_source_counts": float(np.sum(observation.unit_source_counts)),
        "background_counts": (
            None
            if observation.background_counts is None
            else float(np.sum(observation.background_counts))
        ),
        "background_model_counts": (
            None
            if observation.background_model is None
            else float(np.sum(observation.background_model))
        ),
        "background_covariance_condition": observation.background_covariance_condition,
        "alpha_min": None if observation.alpha is None else float(np.min(observation.alpha)),
        "alpha_max": None if observation.alpha is None else float(np.max(observation.alpha)),
        "exposure_s": observation.exposure_s,
        "response": _path_provenance(observation.response_path),
        "arf": _path_provenance(observation.arf_path),
        "metadata": _json_safe(dict(observation.metadata)),
    }


def _adapter_observation(
    raw: UpperLimitObservation,
    prepared: _PreparedObservation,
) -> UpperLimitObservation:
    raw_alpha = None
    if raw.alpha is not None:
        alpha_array = np.asarray(raw.alpha, dtype=float)
        raw_alpha = float(alpha_array) if alpha_array.ndim == 0 else alpha_array.copy()
    metadata = dict(raw.metadata)
    metadata.update(
        {
            "jinwu_adapter_contract": "validated_template_raw_background_v1",
            "fractional_background_systematic_applied": False,
        }
    )
    return UpperLimitObservation(
        name=prepared.name,
        source_counts=prepared.source_counts.copy(),
        unit_source_counts=prepared.unit_source_counts.copy(),
        background_counts=(
            None
            if raw.background_counts is None
            else np.asarray(raw.background_counts, dtype=float).copy()
        ),
        alpha=raw_alpha,
        background_model=(
            None
            if raw.background_model is None
            else np.asarray(raw.background_model, dtype=float).copy()
        ),
        background_sigma=(
            None
            if raw.background_sigma is None
            else np.asarray(raw.background_sigma, dtype=float).copy()
        ),
        background_covariance=(
            None
            if raw.background_covariance is None
            else np.asarray(raw.background_covariance, dtype=float).copy()
        ),
        exposure_s=prepared.exposure_s,
        response_path=prepared.response_path,
        arf_path=prepared.arf_path,
        metadata=metadata,
    )


def _validate_combination_count(policy: UpperLimitConfig, count: int) -> None:
    if policy.combine == "single" and count != 1:
        raise ValueError(f"combine='single' requires exactly one observation, got {count}")
    if policy.combine in {"joint_detectors", "joint_modules"} and count < 1:
        raise ValueError(f"combine={policy.combine!r} requires at least one observation")


def _validate_instrument_policy(
    instrument_config: InstrumentConfig,
    policy: UpperLimitConfig,
) -> None:
    if policy.strategy == "spatial_onoff" and instrument_config.background_type != "spatial":
        raise ValueError("spatial_onoff requires an instrument with spatial background")
    if policy.strategy == "modeled_count_spectrum" and instrument_config.background_type != "temporal":
        raise ValueError(
            "modeled_count_spectrum requires an instrument with temporal background"
        )
    if policy.strategy == "coded_mask_spectrum" and instrument_config.background_type != "detector_shadow":
        raise ValueError(
            "coded_mask_spectrum requires an instrument with detector_shadow background"
        )
    if policy.response_folding == "rsp" and instrument_config.response_type != "rsp":
        raise ValueError("rsp upper-limit folding requires InstrumentConfig.response_type='rsp'")
    if policy.response_folding == "rmf_arf" and instrument_config.response_type not in {
        "rmf",
        "rmf_arf",
    }:
        raise ValueError(
            "rmf_arf upper-limit folding requires an RMF-capable instrument config"
        )


def _validate_interval(interval: tuple[float, float]) -> tuple[float, float]:
    start, stop = (float(value) for value in interval)
    if not (math.isfinite(start) and math.isfinite(stop) and stop > start):
        raise ValueError("interval must contain finite values with stop > start")
    return start, stop


def _validate_energy_band(band: tuple[float, float]) -> tuple[float, float]:
    emin, emax = (float(value) for value in band)
    if not (math.isfinite(emin) and math.isfinite(emax) and 0 < emin < emax):
        raise ValueError("energy_band must be finite, positive and increasing")
    return emin, emax


def _one_dimensional_array(
    values: Any,
    name: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be positive")
    return array.copy()


def _read_fakeit_expected_counts(
    path: Path,
    *,
    response_path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    from jinwu.core.io import read_pha, read_rmf

    pha = read_pha(path)
    values = np.asarray(pha.counts, dtype=float)
    channels = np.asarray(pha.channels, dtype=int)
    if pha.ebounds is not None:
        bound_channels, e_min, e_max = pha.ebounds
    else:
        response = read_rmf(response_path)
        if response.channel is None or response.e_min is None or response.e_max is None:
            return values, None, None
        bound_channels = np.asarray(response.channel, dtype=int)
        e_min = np.asarray(response.e_min, dtype=float)
        e_max = np.asarray(response.e_max, dtype=float)

    if (
        np.asarray(e_min).shape != np.asarray(bound_channels).shape
        or np.asarray(e_max).shape != np.asarray(bound_channels).shape
    ):
        raise ValueError("OGIP EBOUNDS channel and energy columns have inconsistent shapes")
    index = {int(channel): idx for idx, channel in enumerate(bound_channels)}
    try:
        selected = np.asarray([index[int(channel)] for channel in channels], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"XSPEC fakeit PHA channel {int(exc.args[0])} is absent from response EBOUNDS"
        ) from exc
    return values, np.asarray(e_min, dtype=float)[selected], np.asarray(e_max, dtype=float)[selected]


def _select_template_channels(
    values: np.ndarray,
    *,
    e_min: np.ndarray | None,
    e_max: np.ndarray | None,
    energy_band: tuple[float, float],
    observation: UpperLimitObservation,
) -> np.ndarray:
    counts = np.asarray(values, dtype=float)
    explicit_mask = observation.metadata.get("channel_mask")
    if explicit_mask is not None:
        mask = np.asarray(explicit_mask, dtype=bool)
        if mask.shape != counts.shape:
            raise ValueError("observation.metadata['channel_mask'] shape does not match response")
        return counts[mask]
    if e_min is None or e_max is None:
        raise ValueError(
            "Cannot apply energy_band because neither the predicted PHA nor the "
            "response provides EBOUNDS; provide observation.metadata['channel_mask']"
        )
    if e_min.shape != counts.shape or e_max.shape != counts.shape:
        raise ValueError("Response EBOUNDS shape does not match predicted channel counts")
    from jinwu.core.base import EnergyBand
    from jinwu.core.io import channel_mask_from_ebounds

    emin, emax = energy_band
    channels = np.arange(counts.size, dtype=int)
    mask = channel_mask_from_ebounds(
        (channels, e_min, e_max),
        EnergyBand(emin=emin, emin_unit="keV", emax=emax, emax_unit="keV"),
        None,
    )
    return counts[mask]


def _photon_density_values(values: Any) -> np.ndarray:
    if hasattr(values, "unit") and hasattr(values, "to_value"):
        from astropy import units as u

        array = np.asarray(
            values.to_value(u.photon / (u.cm**2 * u.s * u.keV)),
            dtype=float,
        )
    else:
        array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Photon model must return finite, non-negative density")
    return array


def _optional_positive(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError("model flux/fluence conversion factors must be finite and positive")
    return numeric


def _scaled_value(amplitude: float, factor: float | None) -> float | None:
    return None if factor is None else float(amplitude * factor)


def _optional_path(value: str | Path | None) -> str | None:
    return None if value is None else str(Path(value).expanduser())


def _path_provenance(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(value)
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
        result["size"] = path.stat().st_size
    return result


def _result_to_json(result: ResponseAwareUpperLimitResult) -> dict[str, Any]:
    return {
        "instrument": result.instrument,
        "strategy": result.strategy,
        "interval": list(result.interval),
        "energy_band_keV": list(result.energy_band_keV),
        "model_name": result.model_name,
        "observed_upper_bound": _json_safe(
            None if result.observed_upper_bound is None else asdict(result.observed_upper_bound)
        ),
        "detection_sensitivity": _json_safe(
            None if result.detection_sensitivity is None else asdict(result.detection_sensitivity)
        ),
        "config": result.config,
        "observations": result.observations,
        "provenance": result.provenance,
        "warnings": list(result.warnings),
        "artifacts": dict(result.artifacts),
        "status": result.status,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _render_summary(result: ResponseAwareUpperLimitResult) -> str:
    lines = [
        f"Instrument: {result.instrument}",
        f"Strategy: {result.strategy}",
        f"Interval: [{result.interval[0]:.9g}, {result.interval[1]:.9g}] s",
        f"Energy band: {result.energy_band_keV[0]:g}-{result.energy_band_keV[1]:g} keV",
        f"Fixed-shape model: {result.model_name}",
    ]
    observed = result.observed_upper_bound
    if observed is not None:
        lines.extend(
            [
                "Observed-data upper bound:",
                f"  level: {observed.level.label} (p={observed.level.confidence:.9f}, "
                f"delta_stat={observed.level.delta_stat:g})",
                f"  amplitude <= {observed.amplitude_upper:.8g} {observed.amplitude_unit}",
            ]
        )
        if observed.flux_upper is not None:
            lines.append(f"  flux <= {observed.flux_upper:.8g} {observed.flux_unit}")
        if observed.fluence_upper is not None:
            lines.append(f"  fluence <= {observed.fluence_upper:.8g} {observed.fluence_unit}")
    sensitivity = result.detection_sensitivity
    if sensitivity is not None:
        lines.append("Detection sensitivity (not an observed-data confidence bound):")
        if sensitivity.status == "ready":
            lines.append(
                f"  amplitude = {sensitivity.amplitude:.8g} {sensitivity.amplitude_unit}; "
                f"power={sensitivity.achieved_power:.4f} at threshold={sensitivity.threshold:.6g}"
            )
            if sensitivity.empirical_false_alarm_probability is not None:
                lines.append(
                    "  empirical false-alarm probability = "
                    f"{sensitivity.empirical_false_alarm_probability:.8g}"
                )
            if sensitivity.flux is not None:
                lines.append(f"  flux = {sensitivity.flux:.8g} {sensitivity.flux_unit}")
        else:
            lines.append(f"  {sensitivity.status}: {sensitivity.reason}")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
    return "\n".join(lines) + "\n"


def _write_diagnostic_plots(
    result: ResponseAwareUpperLimitResult,
    outdir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    artifacts: dict[str, str] = {}
    amplitudes = np.asarray(result.likelihood_scan.get("amplitude", []), dtype=float)
    delta_stat = np.asarray(result.likelihood_scan.get("delta_stat", []), dtype=float)
    if amplitudes.size:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(amplitudes, delta_stat, color="black", lw=1.8)
        if result.observed_upper_bound is not None:
            bound = result.observed_upper_bound
            ax.axhline(bound.level.delta_stat, color="tab:red", ls="--", label=bound.level.label)
            ax.axvline(bound.amplitude_upper, color="tab:red", ls=":")
        ax.set_xlabel("Model amplitude")
        ax.set_ylabel("Profile delta statistic")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = outdir / f"upper_limit_likelihood.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None)
            artifacts[f"likelihood_{suffix}"] = str(path)
        plt.close(fig)

    tested = np.asarray(
        result.sensitivity_diagnostics.get("tested_amplitudes", []), dtype=float
    )
    powers = np.asarray(result.sensitivity_diagnostics.get("detection_powers", []), dtype=float)
    if tested.size:
        order = np.argsort(tested)
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.plot(tested[order], powers[order], "o-", color="tab:blue")
        if result.detection_sensitivity is not None:
            ax.axhline(
                result.detection_sensitivity.target_power,
                color="tab:red",
                ls="--",
                label="target detection power",
            )
        ax.set_xlabel("Injected model amplitude")
        ax.set_ylabel("Detection probability")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = outdir / f"upper_limit_sensitivity.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None)
            artifacts[f"sensitivity_{suffix}"] = str(path)
        plt.close(fig)
    return artifacts


@dataclass(slots=True)
class UpperLimitPoint:
    """Upper-limit value for one chain confidence or XSPEC delta-stat level."""

    level: str
    upper: float | None
    confidence: float | None
    delta_stat: float | None
    lower: float | None
    linear_upper: float | None
    status: str | None
    confidence_convention: (
        Literal["posterior_quantile", "profile_delta_stat"] | None
    ) = None
    central_coverage: float | None = None
    one_sided_probability: float | None = None


@dataclass(slots=True)
class UpperLimitResult:
    """Upper-limit result for one XSPEC model parameter."""

    method: Literal["chain", "error"]
    parameter: str
    parameter_index: int | None
    unit: str | None
    limits: dict[str, UpperLimitPoint]
    chain_path: str | None
    sample_count: int | None
    sample_min: float | None
    sample_max: float | None
    chain_result: XspecChainResult | None
    warnings: list[str]
    status: str


class UpperLimit:
    """Calculate XSPEC parameter upper limits from chains or ``Fit.error``.

    Chain upper limits are one-sided quantiles of the selected chain column.
    Error upper limits are the high bounds stored in the selected PyXspec
    parameter after ``xspec.Fit.error``.  XSPEC controls the error semantics:
    if chains are loaded in the active session, ``Fit.error`` uses its
    chain-based rule; otherwise it performs the usual error search.
    """

    def __init__(self, parameter: int | str):
        if not isinstance(parameter, (int, str)):
            raise TypeError("parameter must be an XSPEC parameter index or chain column name")
        if isinstance(parameter, int) and parameter < 1:
            raise ValueError("parameter index must be at least 1")
        if isinstance(parameter, str) and not parameter.strip():
            raise ValueError("parameter string must not be empty")

        self.parameter = parameter

    def from_chain(
        self,
        chain_path: str | Path,
        *,
        levels: Mapping[str, float] | None = None,
    ) -> UpperLimitResult:
        """Read an XSPEC chain FITS file and calculate one-sided upper limits."""
        chain_file = Path(chain_path).expanduser()
        using_legacy_defaults = levels is None
        levels = _validated_levels(
            DEFAULT_CHAIN_LEVELS if using_legacy_defaults else levels,
            "chain level",
        )

        with fits.open(chain_file) as hdul:
            try:
                chain_hdu = hdul["CHAIN"]
            except KeyError as exc:
                raise ValueError(f"XSPEC chain file has no CHAIN extension: {chain_file}") from exc

            available_columns = _chain_parameter_columns(chain_hdu.columns.names)
            column_name = self._resolve_chain_column(available_columns)
            samples = np.asarray(chain_hdu.data[column_name], dtype=float)
            unit = _column_unit(chain_hdu, column_name)

        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            raise ValueError(f"XSPEC chain column has no finite samples: {column_name}")

        parameter_index = _column_parameter_index(column_name)
        limits = {
            str(level): _chain_limit_point(
                str(level),
                confidence,
                samples,
                is_lg10_flux=_is_lg10_flux(column_name),
            )
            for level, confidence in levels.items()
        }
        warnings_list = []
        if using_legacy_defaults:
            warnings_list.append(
                "Legacy UpperLimit.from_chain sigma labels use central Gaussian "
                "coverage values directly as posterior quantiles; for example, "
                "'3sigma' is the 0.997300 quantile, not the one-sided +3 sigma "
                "quantile 0.998650. Pass DEFAULT_ONE_SIDED_CHAIN_LEVELS explicitly "
                "for one-sided Gaussian-equivalent limits."
            )
        return UpperLimitResult(
            method="chain",
            parameter=column_name,
            parameter_index=parameter_index,
            unit=unit,
            limits=limits,
            chain_path=str(chain_file),
            sample_count=int(samples.size),
            sample_min=float(np.min(samples)),
            sample_max=float(np.max(samples)),
            chain_result=None,
            warnings=warnings_list,
            status="upper_limits_ready",
        )

    def run_chain(
        self,
        *,
        chain_path: str | Path,
        levels: Mapping[str, float] | None = None,
        **run_xspec_chain_kwargs,
    ) -> UpperLimitResult:
        """Run the current XSPEC thawed-parameter chain and read its upper limits."""
        chain_result = run_xspec_chain(chain_path=chain_path, **run_xspec_chain_kwargs)
        if chain_result.status != "chain_ready":
            return UpperLimitResult(
                method="chain",
                parameter=str(self.parameter),
                parameter_index=_optional_parameter_index(self.parameter),
                unit=None,
                limits={},
                chain_path=chain_result.chain_path,
                sample_count=None,
                sample_min=None,
                sample_max=None,
                chain_result=chain_result,
                warnings=list(chain_result.warnings),
                status="failed",
            )

        result = self.from_chain(chain_result.chain_path, levels=levels)
        result.chain_result = chain_result
        result.warnings.extend(chain_result.warnings)
        return result

    def error(
        self,
        *,
        deltas: Mapping[str, float] | None = None,
    ) -> UpperLimitResult:
        """Run XSPEC ``Fit.error`` for this parameter and collect high bounds."""
        xspec = _require_xspec()
        parameter_index = self._error_parameter_index()
        parameter = _xspec_parameter(xspec, parameter_index)
        using_legacy_defaults = deltas is None
        deltas = _validated_levels(
            DEFAULT_ERROR_DELTAS if using_legacy_defaults else deltas,
            "delta statistic",
        )

        limits = {}
        warnings_list = []
        if using_legacy_defaults:
            warnings_list.append(
                "Legacy UpperLimit.error label '90%' uses XSPEC delta_stat=2.706 "
                "(central 90% profile interval). Use OneSidedLevel for an explicit "
                "one-sided confidence bound."
            )
        parameter_name = str(getattr(parameter, "name", self.parameter))
        parameter_unit = getattr(parameter, "unit", None) or None
        is_lg10_flux = _is_lg10_flux(parameter_name)

        for level, delta_stat in deltas.items():
            level_name = str(level)
            try:
                xspec.Fit.error(f"{delta_stat} {parameter_index}")
                lower, upper, status = parameter.error
                lower_bound = _float_or_none(lower)
                upper_bound = _float_or_none(upper)
                limits[level_name] = UpperLimitPoint(
                    level=level_name,
                    upper=upper_bound,
                    confidence=None,
                    delta_stat=float(delta_stat),
                    lower=lower_bound,
                    linear_upper=_linear_flux_upper(upper_bound, is_lg10_flux),
                    status=str(status) if status is not None else None,
                    confidence_convention="profile_delta_stat",
                    central_coverage=_profile_central_coverage(delta_stat),
                    one_sided_probability=_profile_one_sided_probability(delta_stat),
                )
            except Exception as exc:
                warnings_list.append(
                    f"XSPEC error failed for parameter {parameter_index} at {level_name}: {exc}"
                )
                limits[level_name] = UpperLimitPoint(
                    level=level_name,
                    upper=None,
                    confidence=None,
                    delta_stat=float(delta_stat),
                    lower=None,
                    linear_upper=None,
                    status=None,
                    confidence_convention="profile_delta_stat",
                    central_coverage=_profile_central_coverage(delta_stat),
                    one_sided_probability=_profile_one_sided_probability(delta_stat),
                )

        failed_levels = sum(point.upper is None for point in limits.values())
        if failed_levels == len(limits):
            status = "failed"
        elif failed_levels:
            status = "partial"
        else:
            status = "upper_limits_ready"

        return UpperLimitResult(
            method="error",
            parameter=parameter_name,
            parameter_index=parameter_index,
            unit=parameter_unit,
            limits=limits,
            chain_path=None,
            sample_count=None,
            sample_min=None,
            sample_max=None,
            chain_result=None,
            warnings=warnings_list,
            status=status,
        )

    def _resolve_chain_column(self, columns: list[str]) -> str:
        if not columns:
            raise ValueError("XSPEC chain has no parameter columns.")

        available = ", ".join(columns)
        parameter = self.parameter
        if isinstance(parameter, int):
            matches = [column for column in columns if _column_parameter_index(column) == parameter]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"XSPEC chain parameter index {parameter} is ambiguous: {available}")
            raise ValueError(f"XSPEC chain has no parameter index {parameter}. Available: {available}")

        query = parameter.strip()
        if query in columns:
            return query

        matches = [column for column in columns if _column_stem(column) == query]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"XSPEC chain parameter name {query!r} is ambiguous. Available: {available}")
        raise ValueError(f"XSPEC chain has no parameter {query!r}. Available: {available}")

    def _error_parameter_index(self) -> int:
        parameter_index = _optional_parameter_index(self.parameter)
        if parameter_index is None:
            raise ValueError(
                "XSPEC error upper limits require a parameter index or a string ending in '__<index>'."
            )
        return parameter_index


def _chain_parameter_columns(column_names) -> list[str]:
    return [str(name) for name in column_names if str(name) != "FIT_STATISTIC"]


def _chain_limit_point(
    level: str,
    confidence: float,
    samples: np.ndarray,
    *,
    is_lg10_flux: bool,
) -> UpperLimitPoint:
    upper = float(np.quantile(samples, confidence))
    return UpperLimitPoint(
        level=level,
        upper=upper,
        confidence=confidence,
        delta_stat=None,
        lower=None,
        linear_upper=_linear_flux_upper(upper, is_lg10_flux),
        status=None,
        confidence_convention="posterior_quantile",
        central_coverage=None,
        one_sided_probability=confidence,
    )


def _profile_central_coverage(delta_stat: float) -> float:
    sigma = math.sqrt(float(delta_stat))
    return float(2.0 * NormalDist().cdf(sigma) - 1.0)


def _profile_one_sided_probability(delta_stat: float) -> float:
    return float(NormalDist().cdf(math.sqrt(float(delta_stat))))


def _column_unit(chain_hdu, column_name: str) -> str | None:
    for column in chain_hdu.columns:
        if column.name == column_name:
            return column.unit or None
    return None


def _column_stem(column_name: str) -> str:
    return _CHAIN_INDEX_RE.sub("", column_name)


def _column_parameter_index(column_name: str) -> int | None:
    match = _CHAIN_INDEX_RE.search(column_name)
    return int(match.group(1)) if match else None


def _optional_parameter_index(parameter: int | str) -> int | None:
    if isinstance(parameter, int):
        return parameter

    stripped = parameter.strip()
    if stripped.isdigit():
        return int(stripped)

    return _column_parameter_index(stripped)


def _validated_levels(levels: Mapping[str, float], name: str) -> dict[str, float]:
    if not levels:
        raise ValueError(f"{name} mapping must not be empty")

    checked = {}
    for level, value in levels.items():
        numeric = float(value)
        if name == "chain level":
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1: {level}={value}")
        elif numeric <= 0.0:
            raise ValueError(f"{name} must be greater than 0: {level}={value}")
        checked[str(level)] = numeric
    return checked


def _xspec_parameter(xspec, parameter_index: int):
    models = _xspec_chain_models(xspec)
    if not models:
        raise RuntimeError("UpperLimit.error requires a loaded XSPEC model.")

    try:
        return models[0](parameter_index)
    except Exception as exc:
        raise ValueError(f"XSPEC model has no parameter index {parameter_index}.") from exc


def _is_lg10_flux(name: str) -> bool:
    return _column_stem(name).lower() == "lg10flux"


def _linear_flux_upper(upper: float | None, is_lg10_flux: bool) -> float | None:
    if upper is None or not is_lg10_flux:
        return None
    return float(10.0**upper)


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
