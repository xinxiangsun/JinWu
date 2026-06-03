"""Upper-limit helpers for XSPEC chain and error calculations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

from astropy.io import fits
import numpy as np

from jinwu.core.fit import (
    XspecChainResult,
    _require_xspec,
    _xspec_chain_models,
    run_xspec_chain,
)

__all__ = [
    "DEFAULT_CHAIN_LEVELS",
    "DEFAULT_ERROR_DELTAS",
    "UpperLimit",
    "UpperLimitPoint",
    "UpperLimitResult",
]


DEFAULT_CHAIN_LEVELS = {
    "1sigma": 0.6826894921370859,
    "90%": 0.9,
    "2sigma": 0.9544997361036416,
    "3sigma": 0.9973002039367398,
}

DEFAULT_ERROR_DELTAS = {
    "1sigma": 1.0,
    "90%": 2.706,
    "2sigma": 4.0,
    "3sigma": 9.0,
}

_CHAIN_INDEX_RE = re.compile(r"__(\d+)$")


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
        levels = _validated_levels(
            DEFAULT_CHAIN_LEVELS if levels is None else levels,
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
            warnings=[],
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
        deltas = _validated_levels(
            DEFAULT_ERROR_DELTAS if deltas is None else deltas,
            "delta statistic",
        )

        limits = {}
        warnings_list = []
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
    )


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
