"""Single-target Swift/BAT survey processing.

This module is a small, testable adapter around BatAnalysis.  It owns the
survey data contract (native channels, signed background-subtracted rates,
GTI overlap and provenance).  Importing it does not import BatAnalysis,
HEASoft, XSPEC or start a network request.
"""

from __future__ import annotations

from dataclasses import InitVar, asdict, dataclass, field, fields, is_dataclass
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
import gzip
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np

from ...core.config import (
    BATSurvey,
    InstrumentConfig,
    SwiftBATSurveyConfig,
    SwiftBATSurveyDownloadConfig,
)
from ...core.fit import fit_prepared
from ...core.pipeline import (
    InstrumentPipeline,
    PipelineInput,
    PipelineStage,
    PipelineStatus,
    StageResult,
    _file_fingerprint,
    _fingerprint,
    register_pipeline,
)
from ...core.products import safe_filename_token, write_json
from ...core.spectrum_prep import PreparedSpectrum
from ...core.time import Time

logger = logging.getLogger(__name__)

__all__ = [
    "BATSurveyInput",
    "BATSurveyResult",
    "BATSurveyPipeline",
    "BatAnalysisSurveyBackend",
    "AreaRow",
    "SurveyRatePoint",
    "SurveyPhaValidation",
    "SurveyUpperLimitResult",
    "build_source_catalog",
    "calculate_background_scale",
    "parse_area_table",
    "read_gti_intervals",
    "gti_overlap_duration",
    "read_bat_survey_rates",
    "select_overlapping_pointings",
    "safe_extract_archive",
    "signed_snr",
    "validate_observation_directory",
    "validate_survey_pha",
    "profile_survey_upper_limit",
    "estimate_bat_survey_sensitivity",
    "BATSurveySensitivityAdapter",
]

_RATE_NAMES = ("CENT_RATE", "RATE", "COUNT_RATE", "COUNTRATE")
# ``RATE_ERR`` is the Gaussian error returned by ``BatSurvey.get_count_rate``.
# ``BKG_VAR`` is retained as a fallback (and is used for an all-band SNR when
# available), but it must not silently replace the source-rate error when both
# columns are present.
_ERROR_NAMES = (
    "RATE_ERR",
    "RATE_ERROR",
    "RATEERR",
    "STAT_ERR",
    "STAT_ERROR",
    "BKG_VAR",
    "ERROR",
)
_FLUX_NAMES = ("FLUX", "FLUX_ERG_CM2_S", "FLUX_ERG_CM2_SEC")
_ECF_NAMES = ("ECF", "ENERGY_CONVERSION_FACTOR")
_TIME_NAMES = (
    "TIME", "TSTART", "START", "START_TIME", "UTC_START", "START_UTC",
)
_STOP_NAMES = (
    "TIME_STOP", "TSTOP", "STOP", "STOP_TIME", "UTC_STOP", "STOP_UTC",
)
_EXPOSURE_NAMES = ("EXPOSURE", "EXPO", "LIVETIME")
_PCODE_NAMES = (
    "PCODEFR", "PCODEAPP", "PCODE", "PCODE_FRACTION", "PARTIAL_CODING",
)
_SOURCE_NAMES = (
    "NAME", "SOURCE", "SRCNAME", "SOURCE_NAME_BAT", "SOURCE_NAME_RAW",
)
_POINTING_NAMES = (
    "IMAGE_ID", "IMAGEID", "POINTING", "POINTING_ID", "PNT_ID",
)
_SURVEY_PRODUCT_SUFFIXES = (".cat", ".pha", ".fits", ".fit", ".csv", ".dat", ".qdp")


def _has_suffix(path: Path, suffixes: Sequence[str]) -> bool:
    """Match a product suffix, including the gzip form used by Swift files."""
    name = path.name.lower()
    return any(name.endswith(suffix) or name.endswith(suffix + ".gz") for suffix in suffixes)


def _response_is_arf(path: Path) -> bool:
    """Return whether a response path is an ARF, including compressed ARFs.

    Swift archives commonly contain ``.arf.gz`` files.  ``Path.suffix`` only
    reports ``.gz`` for those products, so all response consumers must remove
    the transport-compression suffix before deciding whether the companion
    RMF belongs in ``arf_path`` or ``response_path``.
    """
    name = path.name.lower()
    for compression in (".gz", ".bz2", ".fz"):
        if name.endswith(compression):
            name = name[: -len(compression)]
            break
    return name.endswith(".arf")


def _source_name_matches(actual: str | None, requested: str | None) -> bool:
    """Compare BAT catalogue aliases using BatAnalysis' punctuation rule."""
    if not requested or not actual:
        return True
    normalize = lambda value: re.sub(r"[^0-9A-Za-z.]", "", value).lower()
    return normalize(str(actual)) == normalize(str(requested))


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _float(value: Any) -> float | None:
    value = _decode(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NONE", "NULL", "N/A", "NA", "INDEF"}:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _float_array(value: Any) -> tuple[float, ...] | None:
    """Return finite or non-finite numeric vector values without flattening
    away the native BAT survey channel contract.

    FITS ``RATE``/``CENT_RATE``/``BKG_VAR`` fields are commonly nine-element
    vectors (the eight native bands plus the total band), while CSV exports
    may serialize the same vector as ``"[... ]"``.  A scalar is represented as
    a one-element tuple so callers can use one normalization path.
    """
    value = _decode(value)
    if value is None:
        return None
    if isinstance(value, u.Quantity):
        value = value.value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() in {"NONE", "NULL", "N/A", "NA", "INDEF"}:
            return None
        text = text.strip("[]()")
        # FITS-like vector text can be comma or whitespace separated.
        try:
            values = np.fromstring(text.replace(",", " "), sep=" ", dtype=float)
        except ValueError:
            values = np.asarray((), dtype=float)
        if values.size == 0:
            scalar = _float(text)
            return (scalar,) if scalar is not None else None
    else:
        try:
            values = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            scalar = _float(value)
            return (scalar,) if scalar is not None else None
    return tuple(float(item) for item in values)


def _scalar_value(value: Any, *, index: int = -1) -> float | None:
    values = _float_array(value)
    if not values:
        return None
    try:
        candidate = float(values[index])
    except (IndexError, TypeError, ValueError):
        return None
    return candidate if np.isfinite(candidate) else None


def _totalize_native_channels(
    channel_rate: tuple[float, ...] | None,
    channel_error: tuple[float, ...] | None,
    channel_snr: tuple[float, ...] | None,
    background_variance: tuple[float, ...] | None,
) -> tuple[
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    int | None,
    bool,
]:
    """Expose eight native BAT bands plus one derived all-band entry.

    BatAnalysis source CAT files contain eight-element ``CENT_RATE`` and
    ``RATE_ERR`` vectors.  ``load_source_information`` appends a ninth total
    entry by summing those native bands.  Newer exports may already include
    that ninth value; in that case it is preserved and never summed again.
    """
    rate_length = len(channel_rate) if channel_rate else 0
    # The total-band convention can only be established from the rate vector.
    # Do not infer it from an unrelated error/SNR vector when a row carries a
    # scalar rate (a mixed CSV export is otherwise easy to mislabel).
    if rate_length not in {8, 9}:
        return (
            channel_rate,
            channel_error,
            channel_snr,
            rate_length or None,
            False,
        )
    if rate_length == 9:
        # Some intermediate CAT writers append the all-band rate but leave
        # the error/SNR vectors at their native eight-band length.  Normalize
        # those companion vectors as well; otherwise selecting ``[-1]`` below
        # would silently use band 8 as the total-band error.
        total_rate = _scalar_value(channel_rate)
        total_error: float | None = None
        if channel_error is not None and len(channel_error) == 8:
            values = np.asarray(channel_error, dtype=float)
            if np.all(np.isfinite(values)):
                total_error = float(np.sqrt(np.sum(values**2)))
            channel_error = tuple(channel_error) + (
                total_error if total_error is not None else float("nan"),
            )
        elif channel_error is not None and len(channel_error) >= 9:
            total_error = _scalar_value(channel_error)
        total_snr: float | None = None
        if total_rate is not None and background_variance is not None and len(background_variance) >= 8:
            values = np.asarray(background_variance[:8], dtype=float)
            if np.all(np.isfinite(values)):
                total_snr = signed_snr(
                    total_rate,
                    float(np.sqrt(np.sum(values**2))),
                )
        if total_rate is not None and total_snr is None and total_error is not None:
            total_snr = signed_snr(total_rate, total_error)
        if channel_snr is not None and len(channel_snr) == 8:
            channel_snr = tuple(channel_snr) + (
                total_snr if total_snr is not None else float("nan"),
            )
        elif channel_snr is None or len(channel_snr) == 1:
            native_snr = [float("nan")] * 8
            if channel_error is not None and len(channel_error) >= 8:
                native_snr = [
                    signed_snr(channel_rate[index], channel_error[index])
                    for index in range(8)
                ]
            scalar_fallback = (
                channel_snr[0]
                if channel_snr is not None and len(channel_snr) == 1
                else float("nan")
            )
            channel_snr = tuple(native_snr) + (
                total_snr if total_snr is not None else scalar_fallback,
            )
        return channel_rate, channel_error, channel_snr, 8, True

    total_rate: float | None = None
    if channel_rate is not None and len(channel_rate) == 8:
        values = np.asarray(channel_rate, dtype=float)
        if np.all(np.isfinite(values)):
            total_rate = float(np.sum(values))
        channel_rate = tuple(channel_rate) + (total_rate if total_rate is not None else float("nan"),)

    total_error: float | None = None
    if channel_error is not None and len(channel_error) == 8:
        values = np.asarray(channel_error, dtype=float)
        if np.all(np.isfinite(values)):
            total_error = float(np.sqrt(np.sum(values**2)))
        channel_error = tuple(channel_error) + (total_error if total_error is not None else float("nan"),)

    total_snr: float | None = None
    if total_rate is not None:
        if background_variance is not None and len(background_variance) >= 8:
            values = np.asarray(background_variance[:8], dtype=float)
            if np.all(np.isfinite(values)):
                total_snr = signed_snr(total_rate, float(np.sqrt(np.sum(values**2))))
        if total_snr is None and total_error is not None:
            total_snr = signed_snr(total_rate, total_error)
    if channel_snr is not None and len(channel_snr) == 8:
        channel_snr = tuple(channel_snr) + (total_snr if total_snr is not None else float("nan"),)
    elif channel_snr is None or len(channel_snr) == 1:
        # A scalar SNR field in a CAT is usually the all-band diagnostic.  If
        # per-band rates/errors are available, derive the native eight-band
        # values as well; otherwise keep the unavailable channels explicit.
        native_snr = [float("nan")] * 8
        if channel_error is not None and len(channel_error) >= 8:
            native_snr = [
                signed_snr(channel_rate[index], channel_error[index])
                for index in range(8)
            ]
        scalar_fallback = (
            channel_snr[0]
            if channel_snr is not None and len(channel_snr) == 1
            else float("nan")
        )
        channel_snr = tuple(native_snr) + (
            total_snr if total_snr is not None else scalar_fallback,
        )
    return channel_rate, channel_error, channel_snr, 8, True


def _field(row: Any, names: Sequence[str]) -> Any:
    row_names = getattr(row, "colnames", ())
    if not row_names:
        row_names = getattr(getattr(row, "array", None), "names", ())
    if not row_names:
        row_names = getattr(getattr(row, "dtype", None), "names", ()) or ()
    available = {str(name).upper(): name for name in row_names}
    if isinstance(row, Mapping):
        available = {str(name).upper(): name for name in row}
    for name in names:
        actual = available.get(name.upper())
        if actual is not None:
            try:
                return row[actual]
            except (KeyError, IndexError):
                return None
    # CSV exports in the reference workflow mix spellings such as
    # ``FluxPos``/``FLUX_POS`` and ``RateErr``/``RATE_ERR``.  Keep exact FITS
    # names preferred, then use a punctuation-insensitive fallback so the
    # same reader can consume either representation without duplicating
    # column contracts.
    normalized = {
        re.sub(r"[^A-Z0-9]", "", str(key).upper()): value
        for key, value in available.items()
    }
    for name in names:
        actual = normalized.get(re.sub(r"[^A-Z0-9]", "", name.upper()))
        if actual is not None:
            try:
                return row[actual]
            except (KeyError, IndexError):
                return None
    return None


def signed_snr(rate: float | u.Quantity, error: float | u.Quantity) -> float:
    """Return signed rate/error.

    Rates normally have count/s units and the return value is dimensionless.
    Negative net rates remain negative; a non-positive error returns NaN.
    """
    # 方法：带符号信噪比 SNR = rate / error（净率为负时保留符号；error<=0 返回
    #       NaN）。全带合成 SNR 遵循 BAT 官方 batsurvey 的 TOTSNR 定义：
    #       TOTSNR = TOTAL(RATE) / sqrt(sum(BKG_VAR**2))（8 个原生能段的本底
    #       标准差平方和开根）；与 BatAnalysis get_count_rate 的
    #       snr_allband = rate_tot/sqrt(sum(bkg_var^2)) 逐式一致。
    # 参考：HEASoft batsurvey 官方文档，本地
    #       external_sources/heasoft-6.37/swift/bat/tasks/batsurvey/batsurvey.html
    #       （"TOTSNR ... computed as TOTAL(RATE)/SQRT(SUM(BKG_VAR**2))"）；
    #       本地 external_sources/BatAnalysis-main/batanalysis/bat_survey.py
    #       get_count_rate()（bkg_var_2_tot = sum(bkg_var^2)）；
    #       Barthelmy et al., 2005, Space Sci. Rev. 120, 143
    #       (doi:10.1007/s11214-005-5096-3)（BAT 仪器与 14-195 keV 巡天能段）。
    if isinstance(rate, u.Quantity):
        if not isinstance(error, u.Quantity):
            raise TypeError("rate and error must both be Quantity objects")
        value = (rate / error).to_value(u.dimensionless_unscaled)
    else:
        if isinstance(error, u.Quantity):
            raise TypeError("rate and error must both be Quantity objects")
        error_value = float(error)
        value = float(rate) / error_value if error_value > 0 else np.nan
    return float(value) if np.isfinite(value) else float("nan")


def _accepted_status_mask(values: Any) -> np.ndarray:
    """Normalize FITS ``IMAGE_STATUS`` values to a strict success mask."""
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return array.astype(bool)
    if array.dtype.kind in "iuf":
        return np.isfinite(array) & (array != 0)
    output = np.zeros(array.shape, dtype=bool)
    for index, value in np.ndenumerate(array):
        text = str(_decode(value)).strip().upper()
        output[index] = text in {"SUCCESS", "GOOD", "TRUE", "1", "OK", "PASS"}
    return output


@dataclass(frozen=True, slots=True)
class AreaRow:
    """One area-table row, in seconds and pixel squared."""

    start: float
    stop: float
    background_area: u.Quantity


def parse_area_table(path: str | Path) -> tuple[AreaRow, ...]:
    """Parse plain or gzip-compressed start/stop/background-area rows."""
    path = Path(path).expanduser().resolve()
    opener: Callable[..., Any] = gzip.open if path.suffix.lower() == ".gz" else open
    rows: list[AreaRow] = []
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                values = line.split()
                if len(values) < 3 or values[0].startswith(("#", "!")):
                    continue
                start, stop, area = (_float(item) for item in values[:3])
                if start is None or stop is None or area is None or stop <= start or area <= 0:
                    continue
                rows.append(AreaRow(start, stop, area * u.pixel**2))
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    return tuple(rows)


def calculate_background_scale(
    path: str | Path,
    *,
    source_area: u.Quantity,
    time_s: float | u.Quantity | Time | None = None,
) -> float:
    """Calculate alpha = source_area / background_area.

    A source area with units is required. Missing or invalid information
    raises instead of silently assuming alpha=1.
    """
    if not isinstance(source_area, u.Quantity):
        raise TypeError("source_area must be an astropy Quantity")
    source = source_area.to_value(u.pixel**2)
    if not np.isfinite(source) or source <= 0:
        raise ValueError("source_area must be finite and positive")
    rows = parse_area_table(path)
    if time_s is not None:
        value = (
            float(time_s.to_value(u.s))
            if isinstance(time_s, u.Quantity)
            else _time_value(time_s)
            if isinstance(time_s, Time)
            else float(time_s)
        )
        selected = tuple(
            row for row in rows
            if row.start <= value < row.stop
        )
        # Permit the final closed endpoint when it is the only exact match;
        # adjacent intervals otherwise follow the half-open convention.
        if not selected and rows and value == rows[-1].stop:
            selected = (rows[-1],)
        if not selected:
            raise ValueError(
                f"area table has no interval covering time_s={value:g}: {path}"
            )
        rows = selected
    if not rows:
        raise ValueError(f"area table has no valid rows: {path}")
    values = np.asarray(
        [source / row.background_area.to_value(u.pixel**2) for row in rows],
        dtype=float,
    )
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        raise ValueError(f"area table has no valid background area: {path}")
    return float(np.nanmedian(values))


@dataclass(frozen=True, slots=True)
class SurveyRatePoint:
    """Canonical one-pointing BAT survey rate measurement."""

    obsid: str | None
    pointing_id: str | None
    source: str | None
    time_start: float | None
    time_stop: float | None
    exposure_s: float | None
    rate: float | None
    rate_error: float | None
    snr: float | None
    pcode: float | None
    bad_bin: bool
    quality: str | None
    rate_unit: str
    error_source: str | None
    source_file: str
    rate_source: str | None = None
    channel_rate: tuple[float, ...] | None = None
    channel_error: tuple[float, ...] | None = None
    channel_snr: tuple[float, ...] | None = None
    background_variance: tuple[float, ...] | None = None
    systematic_error: tuple[float, ...] | None = None
    native_channel_count: int | None = None
    total_band_is_last: bool = False

    @property
    def local_background_std(self) -> tuple[float, ...] | None:
        """Per-channel BAT ``BKG_VAR`` standard deviation from survey products.

        The historical serialized field is named ``background_variance`` for
        compatibility, although BatAnalysis' column contains the local noise
        standard deviation used by TOTSNR.  It is not automatically combined
        with the PHA ``STAT_ERR``.
        """
        return self.background_variance

    def to_payload(self) -> dict[str, Any]:
        names = (
            "obsid", "pointing_id", "source", "time_start", "time_stop",
            "exposure_s", "rate", "rate_error", "snr", "pcode", "bad_bin",
            "quality", "rate_unit", "error_source", "source_file",
            "rate_source", "channel_rate", "channel_error", "channel_snr",
            "background_variance", "systematic_error",
            "native_channel_count", "total_band_is_last",
        )
        return {name: getattr(self, name) for name in names}


@dataclass(frozen=True, slots=True)
class BATSurveySensitivityAdapter:
    """Empirical fixed-position BAT survey sensitivity adapter.

    ``controls`` must be source-free, quality-screened survey measurements
    with the same native eight-channel contract as the target pointings.  The
    adapter evaluates the native TOTSNR statistic on those controls and on
    deterministic source injections.  It never substitutes PHA ``STAT_ERR``
    for the survey ``BKG_VAR`` noise field.
    """

    controls: tuple[SurveyRatePoint, ...]
    unit_source_rate: u.Quantity
    flux_per_amplitude: u.Quantity | None = None
    false_alarm_probability: float | None = None
    target_power: float = 0.90
    min_null_exceedances: int = 20

    def __post_init__(self) -> None:
        if not isinstance(self.unit_source_rate, u.Quantity):
            raise TypeError("unit_source_rate must be an astropy Quantity")
        if self.flux_per_amplitude is not None:
            if not isinstance(self.flux_per_amplitude, u.Quantity):
                raise TypeError("flux_per_amplitude must be an astropy Quantity")
            if not np.isfinite(float(self.flux_per_amplitude.value)) or float(
                self.flux_per_amplitude.value
            ) <= 0:
                raise ValueError("flux_per_amplitude must be finite and positive")
        power = float(self.target_power)
        if not 0.0 < power < 1.0:
            raise ValueError("target_power must be between 0 and 1")
        if int(self.min_null_exceedances) < 1:
            raise ValueError("min_null_exceedances must be positive")
        if self.false_alarm_probability is not None:
            alpha = float(self.false_alarm_probability)
            if not 0.0 < alpha < 1.0 or not np.isfinite(alpha):
                raise ValueError("false_alarm_probability must be between 0 and 1")

    def estimate(self, *, sigma: float = 3.0) -> Any:
        from jinwu.core.upperlimit import OneSidedLevel

        level = OneSidedLevel.from_sigma(sigma)
        return estimate_bat_survey_sensitivity(
            self.controls,
            unit_source_rate=self.unit_source_rate,
            level=level,
            target_power=self.target_power,
            false_alarm_probability=self.false_alarm_probability,
            flux_per_amplitude=self.flux_per_amplitude,
            min_null_exceedances=self.min_null_exceedances,
        )

    # Match the core adapter protocol so the object can be used by generic
    # response-aware code when a BAT observation carries a native template.
    def estimate_sensitivity(self, observations: Sequence[Any], **kwargs: Any) -> Any:
        del observations
        sigma_level = kwargs.get("level")
        sigma = 3.0 if sigma_level is None else float(sigma_level.sigma)
        policy = kwargs.get("policy")
        false_alarm_probability = self.false_alarm_probability
        target_power = self.target_power
        if policy is not None:
            false_alarm_probability = getattr(
                policy, "detection_false_alarm_probability", false_alarm_probability
            )
            target_power = float(getattr(policy, "detection_power", target_power))
        level = sigma_level
        return estimate_bat_survey_sensitivity(
            self.controls,
            unit_source_rate=self.unit_source_rate,
            level=level,
            sigma=sigma,
            target_power=target_power,
            false_alarm_probability=false_alarm_probability,
            flux_per_amplitude=self.flux_per_amplitude,
            min_null_exceedances=self.min_null_exceedances,
        )

    def calibrate(
        self,
        observations: Sequence[Any],
        *,
        level: Any,
        target_power: float,
        seed: int,
    ) -> Mapping[str, Any]:
        """Return auditable native-null/injection calibration metadata.

        The adapter's ``controls`` are the source-free null sample and the
        same TOTSNR calculation is reused for deterministic source injection.
        ``observations`` is accepted for the shared core protocol; BAT survey
        control rows already carry the native eight-channel template contract.
        """
        del observations
        result = estimate_bat_survey_sensitivity(
            self.controls,
            unit_source_rate=self.unit_source_rate,
            level=level,
            target_power=float(target_power),
            false_alarm_probability=self.false_alarm_probability,
            flux_per_amplitude=self.flux_per_amplitude,
            min_null_exceedances=self.min_null_exceedances,
        )
        return {
            "construction": "bat_native_totsnr_null_and_injection",
            "calibration_status": result.calibration_status,
            "status": result.status,
            "false_alarm_probability": result.false_alarm_probability,
            "target_power": result.target_power,
            "achieved_power": result.achieved_power,
            "null_trials": result.null_trials,
            "signal_trials": result.signal_trials,
            "trial_count": result.trial_count,
            "seed": int(seed),
            "min_null_exceedances": int(self.min_null_exceedances),
            "reason": result.reason,
        }


def estimate_bat_survey_sensitivity(
    controls: Sequence[SurveyRatePoint],
    *,
    unit_source_rate: u.Quantity,
    level: Any | None = None,
    sigma: float = 3.0,
    target_power: float = 0.90,
    false_alarm_probability: float | None = None,
    flux_per_amplitude: u.Quantity | None = None,
    min_null_exceedances: int = 20,
) -> Any:
    """Estimate fixed-position BAT sensitivity from blank survey controls.

    The statistic is the native fixed-position TOTSNR proxy
    ``RATE_TOTAL/sqrt(sum(BKG_VAR[:8]**2))``.  For an eight-channel export,
    ``RATE_TOTAL`` is the canonical sum supplied by the reader; when a
    product carries an explicit ninth all-band value, that value is used
    without re-summing it.  At least
    ``min_null_exceedances`` controls must populate the empirical upper tail;
    otherwise the result is explicitly ``unavailable`` rather than assigning
    an unjustified Gaussian-tail sensitivity.
    """
    # 方法：固定位置灵敏度用原生 TOTSNR 统计量在无源控制样本上的经验分布外推：
    #       TOTSNR = RATE_TOTAL/sqrt(sum(BKG_VAR[:8]**2))（batsurvey 官方定义，
    #       见 signed_snr 处参考）；灵敏度由控制样本的经验上尾 + 确定性源注入
    #       达到 (false_alarm_probability, target_power) 要求，控制样本不足时
    #       显式返回 unavailable，不用高斯尾外推。
    # 参考：HEASoft batsurvey 文档与 BatAnalysis bat_survey.py（同 signed_snr 注释）；
    #       注入-回收法的功效/虚警框架见 Cowan, Cranmer, Gross & Vitells, 2011,
    #       Eur. Phys. J. C 71, 1554 (arXiv:1007.1727)。
    from jinwu.core.upperlimit import DetectionSensitivity, OneSidedLevel

    items = tuple(controls)
    if level is None:
        level = OneSidedLevel.from_sigma(sigma)
    alpha = (
        0.5 * (1.0 - math.erf(float(level.sigma) / math.sqrt(2.0)))
        if false_alarm_probability is None
        else float(false_alarm_probability)
    )
    target_power = float(target_power)
    min_null_exceedances = int(min_null_exceedances)
    if min_null_exceedances < 1:
        raise ValueError("min_null_exceedances must be positive")
    template = _native_template_values(unit_source_rate)
    model_unit = str(unit_source_rate.unit)
    base_stats: list[float] = []
    control_rates: list[np.ndarray] = []
    control_totals: list[float] = []
    control_noise: list[float] = []
    for point in items:
        vector = _native_control_vector(point)
        if vector is None:
            continue
        rates, noise, total_rate = vector
        if rates.shape != template.shape:
            continue
        control_rates.append(rates)
        control_totals.append(total_rate)
        control_noise.append(noise)
        # TOTSNR uses the product's all-band total rate together with the
        # eight native-channel background variances.  Do not re-sum the
        # first eight entries when a CAT already carries an explicit ninth
        # total; that would silently change the statistic through rounding or
        # through the product's own total-band convention.
        base_stats.append(float(total_rate / noise))
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("false_alarm_probability must be between 0 and 1")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be between 0 and 1")
    if not control_rates:
        return DetectionSensitivity(
            amplitude=None,
            amplitude_unit=model_unit,
            flux=None,
            flux_unit=None if flux_per_amplitude is None else str(flux_per_amplitude.unit),
            fluence=None,
            fluence_unit=None,
            threshold=None,
            false_alarm_confidence=1.0 - alpha,
            target_power=target_power,
            achieved_power=None,
            null_trials=0,
            signal_trials=0,
            status="unavailable",
            reason="no valid blank-sky controls with native eight-channel rates",
            calibration_status="unavailable",
            false_alarm_probability=alpha,
            search_scope="fixed_position",
            trial_count=0,
        )
    null_stats = np.asarray(base_stats, dtype=float)
    if null_stats.size < 2 or not np.any(np.isfinite(null_stats)) or np.ptp(null_stats) <= 1e-15:
        return DetectionSensitivity(
            amplitude=None,
            amplitude_unit=model_unit,
            flux=None,
            flux_unit=None if flux_per_amplitude is None else str(flux_per_amplitude.unit),
            fluence=None,
            fluence_unit=None,
            threshold=None,
            false_alarm_confidence=1.0 - alpha,
            target_power=target_power,
            achieved_power=None,
            null_trials=int(null_stats.size),
            signal_trials=0,
            status="unavailable",
            empirical_false_alarm_probability=None,
            reason=(
                "blank-sky null statistic has no empirical tail variation; "
                "empirical exceedances cannot calibrate the requested tail"
            ),
            calibration_status="unavailable",
            false_alarm_probability=alpha,
            search_scope="fixed_position",
            trial_count=int(null_stats.size),
        )
    threshold = float(np.quantile(null_stats, 1.0 - alpha, method="higher"))
    empirical_alpha = float(np.mean(null_stats >= threshold))
    exceedances = int(np.sum(null_stats >= threshold))
    if exceedances < min_null_exceedances:
        return DetectionSensitivity(
            amplitude=None,
            amplitude_unit=model_unit,
            flux=None,
            flux_unit=None if flux_per_amplitude is None else str(flux_per_amplitude.unit),
            fluence=None,
            fluence_unit=None,
            threshold=threshold,
            false_alarm_confidence=1.0 - alpha,
            target_power=target_power,
            achieved_power=None,
            null_trials=int(null_stats.size),
            signal_trials=int(null_stats.size),
            status="unavailable",
            empirical_false_alarm_probability=empirical_alpha,
            reason=(
                f"empirical null tail has {exceedances} exceedances; "
                f"at least {min_null_exceedances} are required"
            ),
            calibration_status="unavailable",
            false_alarm_probability=alpha,
            search_scope="fixed_position",
            trial_count=int(null_stats.size),
        )
    # A finite empirical tail is only useful when it is compatible with the
    # requested false-alarm probability.  Requiring the target alpha to lie
    # inside a two-sided 95% binomial interval prevents a small or contaminated
    # control sample (for example, 20 high-SNR rows out of 100) from being
    # advertised as a 3-sigma sensitivity.  The interval is a diagnostic gate;
    # it does not turn the empirical threshold into a global-search result.
    alpha_interval = _binomial_proportion_interval(exceedances, int(null_stats.size))
    if not alpha_interval[0] <= alpha <= alpha_interval[1]:
        return DetectionSensitivity(
            amplitude=None,
            amplitude_unit=model_unit,
            flux=None,
            flux_unit=None if flux_per_amplitude is None else str(flux_per_amplitude.unit),
            fluence=None,
            fluence_unit=None,
            threshold=threshold,
            false_alarm_confidence=1.0 - alpha,
            target_power=target_power,
            achieved_power=None,
            null_trials=int(null_stats.size),
            signal_trials=0,
            status="unavailable",
            empirical_false_alarm_probability=empirical_alpha,
            reason=(
                "empirical null false-alarm rate "
                f"{empirical_alpha:.6g} is outside the 95% binomial interval "
                f"[{alpha_interval[0]:.6g}, {alpha_interval[1]:.6g}] for target "
                f"{alpha:.6g}"
            ),
            calibration_status="needs_review",
            false_alarm_probability=alpha,
            search_scope="fixed_position",
            trial_count=int(null_stats.size),
        )

    def power(amplitude: float) -> float:
        values = np.asarray(
            [
                (total + amplitude * float(np.sum(template))) / noise
                for total, noise in zip(control_totals, control_noise, strict=True)
            ],
            dtype=float,
        )
        return float(np.mean(values >= threshold))

    low, high = 0.0, max(1e-12, threshold * np.median(control_noise) / max(np.sum(template), 1e-30))
    high_power = power(high)
    for _ in range(60):
        if high_power >= target_power:
            break
        high *= 2.0
        high_power = power(high)
    else:
        return DetectionSensitivity(
            amplitude=None,
            amplitude_unit=model_unit,
            flux=None,
            flux_unit=None if flux_per_amplitude is None else str(flux_per_amplitude.unit),
            fluence=None,
            fluence_unit=None,
            threshold=threshold,
            false_alarm_confidence=1.0 - alpha,
            target_power=target_power,
            achieved_power=high_power,
            null_trials=int(null_stats.size),
            signal_trials=int(null_stats.size),
            status="failed",
            empirical_false_alarm_probability=empirical_alpha,
            reason="could not bracket target detection power",
            calibration_status="needs_review",
            false_alarm_probability=alpha,
            search_scope="fixed_position",
            trial_count=int(null_stats.size),
        )
    for _ in range(60):
        midpoint = 0.5 * (low + high)
        if power(midpoint) >= target_power:
            high = midpoint
        else:
            low = midpoint
    flux = None if flux_per_amplitude is None else float(high * flux_per_amplitude.value)
    flux_unit = None if flux_per_amplitude is None else str(flux_per_amplitude.unit)
    return DetectionSensitivity(
        amplitude=float(high),
        amplitude_unit=model_unit,
        flux=flux,
        flux_unit=flux_unit,
        fluence=None,
        fluence_unit=None,
        threshold=threshold,
        false_alarm_confidence=1.0 - alpha,
        target_power=target_power,
        achieved_power=power(high),
        null_trials=int(null_stats.size),
        signal_trials=int(null_stats.size),
        status="ready",
        empirical_false_alarm_probability=empirical_alpha,
        calibration_status="empirical_fixed_position",
        false_alarm_probability=alpha,
        search_scope="fixed_position",
        trial_count=int(null_stats.size * 2),
    )


def _binomial_proportion_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a two-sided Clopper--Pearson interval for a binomial rate.

    SciPy is an optional dependency of the Swift plugin, so the exact beta
    interval is imported lazily.  The Wilson interval is retained as a
    numerically stable fallback for lightweight installations; either branch
    is used only as a conservative calibration diagnostic, never as a source
    likelihood.
    """
    n = int(trials)
    k = int(successes)
    if n <= 0 or k < 0 or k > n:
        raise ValueError("binomial successes/trials are invalid")
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError("binomial interval confidence must be between 0 and 1")
    alpha = 1.0 - level
    try:
        from scipy.stats import beta

        lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
        upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
        if np.isfinite(lower) and np.isfinite(upper):
            return max(0.0, lower), min(1.0, upper)
    except Exception:  # pragma: no cover - exercised only without SciPy stats
        pass
    # Wilson score fallback.  It is deliberately widened slightly so a
    # missing SciPy installation cannot make the calibration gate optimistic.
    proportion = k / n
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _native_template_values(value: u.Quantity) -> np.ndarray:
    if not isinstance(value, u.Quantity):
        raise TypeError("unit_source_rate must be an astropy Quantity")
    array = np.asarray(value.value, dtype=float)
    if (
        array.ndim != 1
        or array.size != 8
        or np.any(~np.isfinite(array))
        or np.any(array < 0)
        or not np.any(array > 0)
    ):
        raise ValueError("unit_source_rate must contain eight finite non-negative channels")
    return array


def _native_control_vector(
    point: SurveyRatePoint,
) -> tuple[np.ndarray, float, float] | None:
    quality = str(point.quality or "").strip().upper()
    if quality and quality not in {"SUCCESS", "GOOD", "OK", "PASS", "TRUE", "1"}:
        return None
    if point.pcode is not None and (
        not np.isfinite(float(point.pcode)) or float(point.pcode) <= 0.0
    ):
        return None
    if point.bad_bin or point.channel_rate is None or point.background_variance is None:
        return None
    count = int(point.native_channel_count or 8)
    if (
        count != 8
        or len(point.channel_rate) < 8
        or len(point.background_variance) < 8
    ):
        return None
    rates = np.asarray(point.channel_rate[:8], dtype=float)
    noise_vector = np.asarray(point.background_variance[:8], dtype=float)
    if (
        np.any(~np.isfinite(rates))
        or np.any(~np.isfinite(noise_vector))
        or np.any(noise_vector <= 0)
    ):
        return None
    noise = float(np.sqrt(np.sum(noise_vector**2)))
    # ``SurveyRatePoint.rate`` is the canonical all-band value.  It is the
    # ninth product value when one was supplied, and is a derived sum only
    # for eight-channel exports.  Keep that distinction in the native
    # TOTSNR statistic rather than summing an attached total a second time.
    total_rate = (
        float(point.rate)
        if point.rate is not None and np.isfinite(point.rate)
        else float(np.sum(rates))
    )
    if not np.isfinite(total_rate):
        return None
    return rates, noise, total_rate


def _unavailable_bat_sensitivity(
    *,
    reason: str,
    amplitude_unit: str = "xspec_powerlaw_norm_ph_cm2_s_keV_at_1keV",
    flux_unit: str | None = "erg / (s cm2)",
    policy: Any | None = None,
) -> Any:
    from jinwu.core.upperlimit import DetectionSensitivity, OneSidedLevel

    sigma = float(
        getattr(
            policy,
            "upper_confidence_sigma",
            getattr(policy, "default_sigma", 3.0),
        )
        if policy is not None
        else 3.0
    )
    level = OneSidedLevel.from_sigma(sigma)
    alpha = getattr(policy, "detection_false_alarm_probability", None)
    alpha = (1.0 - level.confidence) if alpha is None else float(alpha)
    target_power = float(getattr(policy, "detection_power", 0.90))

    return DetectionSensitivity(
        amplitude=None,
        amplitude_unit=amplitude_unit,
        flux=None,
        flux_unit=flux_unit,
        fluence=None,
        fluence_unit=None,
        threshold=None,
        false_alarm_confidence=1.0 - alpha,
        target_power=target_power,
        achieved_power=None,
        null_trials=0,
        signal_trials=0,
        status="unavailable",
        reason=reason,
        calibration_status="unavailable",
        false_alarm_probability=alpha,
        search_scope="fixed_position",
        trial_count=0,
    )


def _unavailable_bat_observed(
    *,
    reason: str,
    energy_band_keV: tuple[float, float],
    model_name: str = "cflux*powerlaw",
    delta_stat: float = 9.0,
) -> dict[str, Any]:
    """Return the observed-bound object when no valid profile is available."""
    return {
        "value": None,
        "unit": "xspec_powerlaw_norm_ph_cm2_s_keV_at_1keV",
        "energy_band": [float(energy_band_keV[0]), float(energy_band_keV[1])],
        "confidence_level": float(NormalDist().cdf(math.sqrt(delta_stat))),
        "confidence_convention": "one_sided_gaussian_equivalent",
        "spectral_model": model_name,
        "signed_mle": None,
        "constrained_mle": None,
        "construction": "unavailable",
        "calibration_status": "unavailable",
        "profile_status": "unavailable",
        "background": {
            "likelihood": "gaussian_net_rate",
            "provenance": {
                "detector_normalization": "fully_illuminated_detector",
                "source": "unavailable",
            },
            "covariance_source": None,
            "residual_validation": None,
        },
        "reason": str(reason),
    }


def _bat_background_payload(
    profile: SurveyUpperLimitResult | None = None,
    *,
    pha: str | Path | None = None,
    response: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize the BAT background/variance contract for a result record.

    BAT survey PHA values are background-subtracted Gaussian rates.  The
    PHA ``STAT_ERR`` (and its optional fractional ``SYS_ERR``) defines the
    conditional spectral covariance; ``BKG_VAR`` belongs to the native
    survey detection statistic and is intentionally recorded as excluded
    from this spectral fit.  Keeping this block on every result prevents a
    downstream reader from confusing a conditional PHA profile with a
    calibrated fixed-position sensitivity measurement.
    """
    provenance: dict[str, Any] = {}
    if profile is not None:
        provenance.update(dict(profile.background_provenance))
    if pha is not None:
        provenance.setdefault("pha", str(pha))
    if response is not None:
        provenance.setdefault("response", str(response))
    provenance.setdefault("detector_normalization", "fully_illuminated_detector")
    provenance.setdefault("bkg_var_used_for_fit", False)
    likelihood = (
        profile.background_likelihood
        if profile is not None
        else "gaussian_net_rate"
    )
    covariance_source = profile.covariance_source if profile is not None else None
    residual_validation = profile.residual_validation if profile is not None else None
    return {
        "likelihood": likelihood,
        "provenance": provenance,
        "covariance_source": covariance_source,
        "residual_validation": residual_validation,
    }


def _bat_observed_payload(
    profile: SurveyUpperLimitResult,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Serialize a BAT profile result while retaining failed-state metadata."""
    band = profile.energy_band_keV or (14.0, 195.0)
    payload = {
        "value": profile.upper_norm,
        "unit": "xspec_powerlaw_norm_ph_cm2_s_keV_at_1keV",
        "energy_band": [float(band[0]), float(band[1])],
        "confidence_level": profile.confidence_level,
        "confidence_convention": profile.confidence_convention,
        "spectral_model": model_name,
        "signed_mle": profile.signed_mle,
        "constrained_mle": profile.constrained_mle,
        "construction": profile.construction,
        "calibration_status": profile.calibration_status,
        "profile_status": profile.profile_status,
        "flux": profile.flux_erg_cm2_s,
        "flux_unit": "erg cm-2 s-1",
        "background": _bat_background_payload(profile),
    }
    if profile.diagnostics:
        payload["diagnostics"] = list(profile.diagnostics)
    if profile.upper_norm is None:
        payload["reason"] = "; ".join(profile.diagnostics) or (
            f"BAT observed profile status is {profile.status}"
        )
    return payload


def _bat_sensitivity_for_profile(
    profile: SurveyUpperLimitResult,
    controls_path: Path | None,
    *,
    policy: Any | None = None,
) -> Any:
    from jinwu.core.upperlimit import OneSidedLevel

    sigma = float(
        getattr(
            policy,
            "upper_confidence_sigma",
            getattr(policy, "default_sigma", 3.0),
        )
        if policy is not None
        else 3.0
    )
    level = OneSidedLevel.from_sigma(sigma)
    false_alarm_probability = getattr(policy, "detection_false_alarm_probability", None)
    target_power = float(getattr(policy, "detection_power", 0.90))
    result_modes = tuple(getattr(policy, "result_modes", ()) or ())
    if result_modes and "detection_sensitivity" not in result_modes:
        return _unavailable_bat_sensitivity(
            reason=(
                "BAT detection sensitivity was disabled by upper-limit result_modes; "
                "the observed bound remains available."
            ),
            policy=policy,
        )
    if controls_path is None:
        return _unavailable_bat_sensitivity(
            reason=(
                "No blank-sky control path was supplied; BAT sensitivity requires "
                "native TOTSNR control statistics and source injections."
            ),
            policy=policy,
        )
    if profile.template_rate is None:
        return _unavailable_bat_sensitivity(
            reason="The profile result has no response-folded unit source template.",
            policy=policy,
        )
    try:
        controls = read_bat_survey_rates(controls_path, source_name=None)
        return estimate_bat_survey_sensitivity(
            controls,
            unit_source_rate=np.asarray(profile.template_rate, dtype=float) * (u.ct / u.s),
            level=level,
            target_power=target_power,
            false_alarm_probability=false_alarm_probability,
            flux_per_amplitude=_powerlaw_energy_flux_per_norm(
                profile.photon_index,
                *(profile.energy_band_keV or (14.0, 195.0)),
            ) * u.erg / (u.cm**2 * u.s),
        )
    except Exception as exc:
        return _unavailable_bat_sensitivity(
            reason=f"blank-sky sensitivity calibration failed: {type(exc).__name__}:{exc}",
            policy=policy,
        )


def _canonical_rate_row(
    row: Any,
    *,
    source_file: Path,
    header: Mapping[str, Any],
) -> SurveyRatePoint:
    raw_start = _field(row, _TIME_NAMES)
    raw_stop = _field(row, _STOP_NAMES)
    time_start = _scalar_value(raw_start)
    time_stop = _scalar_value(raw_stop)
    # CSV exports from the reference workflow use UTC ISO strings.  FITS
    # survey products use Swift MET seconds.  Preserve a single canonical
    # Swift-MET axis in the in-memory contract while retaining the source file.
    if time_start is None and raw_start is not None:
        try:
            time_start = _time_value(str(_decode(raw_start)))
        except (TypeError, ValueError, OverflowError):
            time_start = None
    if time_stop is None and raw_stop is not None:
        try:
            time_stop = _time_value(str(_decode(raw_stop)))
        except (TypeError, ValueError, OverflowError):
            time_stop = None
    # OGIP table times may be stored in units other than seconds and relative
    # to TIMEZERO.  Apply both cards to numeric mission times.  Textual UTC
    # values have already been converted by ``_time_value`` and must not
    # receive a second offset.  EXPOSURE is an elapsed duration in seconds in
    # the survey product contract, so a missing stop is derived after the
    # absolute start has been normalized.
    timezero = _float(header.get("TIMEZERO")) or 0.0
    unit_scale = _timeunit_scale(header.get("TIMEUNIT", "s"))

    def _is_textual_time(value: Any) -> bool:
        decoded = _decode(value)
        if isinstance(decoded, str):
            try:
                float(decoded.strip())
            except (TypeError, ValueError):
                return True
        return False

    numeric_start = time_start is not None and not _is_textual_time(raw_start)
    numeric_stop = time_stop is not None and not _is_textual_time(raw_stop)
    if numeric_start:
        time_start = time_start * unit_scale + timezero * unit_scale
    if numeric_stop:
        time_stop = time_stop * unit_scale + timezero * unit_scale
    exposure = _float(_field(row, _EXPOSURE_NAMES))
    if time_stop is None and time_start is not None and exposure is not None:
        time_stop = time_start + exposure
    rate_name = next(
        (name for name in _RATE_NAMES if _field(row, (name,)) is not None),
        None,
    )
    error_name = next(
        (name for name in _ERROR_NAMES if _field(row, (name,)) is not None),
        None,
    )
    raw_rate = _field(row, _RATE_NAMES)
    raw_error = _field(row, _ERROR_NAMES)
    raw_background_variance = _field(row, ("BKG_VAR",))
    raw_snr = _field(row, ("VECTSNR", "SNR", "SIGNIFICANCE", "TOTSNR"))
    channel_rate = _float_array(raw_rate)
    channel_error = _float_array(raw_error)
    channel_snr = _float_array(raw_snr)
    background_variance = _float_array(raw_background_variance)
    systematic_error = _float_array(
        _field(row, ("SYS_ERR", "SYSTEMATIC_ERROR", "SYSERR"))
    )
    rate = _scalar_value(raw_rate)
    error = _scalar_value(raw_error)
    snr = _scalar_value(raw_snr)
    if (
        (
            channel_error is None
            or not all(np.isfinite(value) and value > 0 for value in channel_error)
        )
        and background_variance
        and background_variance is not None
        and all(
            np.isfinite(value) and value > 0
            for value in (
                background_variance[:8]
                if len(background_variance) >= 8
                else background_variance
            )
        )
    ):
        # A few legacy exports retain the RATE_ERR column but fill it with
        # nulls.  Treat that as missing and use the documented BKG_VAR
        # fallback, while retaining the fallback's provenance label.
        raw_error = raw_background_variance
        channel_error = background_variance
        error_name = "BKG_VAR"
        error = _scalar_value(raw_error)
    rate_source = (
        {
            "RATE": "native_rate",
            "CENT_RATE": "native_central_rate",
            "COUNT_RATE": "native_count_rate",
            "COUNTRATE": "native_count_rate",
        }.get(rate_name or "", "native_rate")
        if rate_name
        else None
    )
    if rate is None:
        flux = _scalar_value(_field(row, _FLUX_NAMES))
        ecf = _scalar_value(_field(row, _ECF_NAMES))
        if flux is not None and ecf not in (None, 0.0):
            rate = flux / ecf
            rate_source = "flux_over_ecf"
            flux_high = _scalar_value(
                _field(
                    row,
                    (
                        "FLUX_POS",
                        "FLUXPOS",
                        "FLUX_ERR_HIGH",
                        "FLUXERRHIGH",
                        "FLUX_POS_WITH_ECF_ERR",
                        "FLUXPOSWITHECFERR",
                    ),
                )
            )
            flux_low = _scalar_value(
                _field(
                    row,
                    (
                        "FLUX_NEG",
                        "FLUXNEG",
                        "FLUX_ERR_LOW",
                        "FLUXERRLOW",
                        "FLUX_NEG_WITH_ECF_ERR",
                        "FLUXNEGWITHECFERR",
                    ),
                )
            )
            if flux_high is not None or flux_low is not None:
                error = max(abs(flux_high or 0.0), abs(flux_low or 0.0)) / abs(ecf)
                error_name = "flux_ecf_propagated"
            channel_rate = (rate,)
            channel_error = (error,) if error is not None else channel_error
    (
        channel_rate,
        channel_error,
        channel_snr,
        native_channel_count,
        total_band_is_last,
    ) = _totalize_native_channels(
        channel_rate,
        channel_error,
        channel_snr,
        background_variance,
    )
    # BatAnalysis appends the all-band value after the eight native bands;
    # make the scalar contract match that ``[-1]`` convention.
    if total_band_is_last:
        rate = _scalar_value(channel_rate)
        error = _scalar_value(channel_error)
        snr = _scalar_value(channel_snr)
    if snr is None and rate is not None and error is not None:
        snr = signed_snr(rate, error)
    bad = str(_decode(_field(row, ("BADBIN", "BAD_BIN", "QUALITY_FLAG"))) or "").strip().lower()
    quality_value = _decode(_field(row, ("IMAGE_STATUS", "STATUS", "QUALITY")))
    quality = str(quality_value).strip() if quality_value is not None else None
    rate_unit = _field(row, ("RATEUNIT", "RATE_UNIT", "TUNIT_RATE"))
    if rate_unit is None:
        rate_unit = header.get("RATEUNIT") or header.get("TUNIT_RATE") or "count/s"
    return SurveyRatePoint(
        obsid=str(_decode(_field(row, ("OBSID", "OBS_ID"))) or header.get("OBSID") or "") or None,
        pointing_id=str(_decode(_field(row, _POINTING_NAMES)) or "") or None,
        source=str(_decode(_field(row, _SOURCE_NAMES)) or "") or None,
        time_start=time_start,
        time_stop=time_stop,
        exposure_s=exposure,
        rate=rate,
        rate_error=error,
        snr=snr,
        pcode=_float(_field(row, _PCODE_NAMES)),
        bad_bin=bad in {"1", "true", "t", "yes", "y", "bad"},
        quality=quality,
        rate_unit=str(_decode(rate_unit) or "count/s"),
        error_source=error_name,
        source_file=str(source_file),
        rate_source=rate_source,
        channel_rate=channel_rate,
        channel_error=channel_error,
        channel_snr=channel_snr,
        background_variance=background_variance,
        systematic_error=systematic_error,
        native_channel_count=native_channel_count,
        total_band_is_last=total_band_is_last,
    )


def _read_fits_rates(path: Path, source_name: str | None) -> list[SurveyRatePoint]:
    points: list[SurveyRatePoint] = []
    with fits.open(path, memmap=False) as hdus:
        primary_header = dict(hdus[0].header) if hdus else {}
        for hdu in hdus:
            data = hdu.data
            columns = getattr(hdu, "columns", None)
            names = list(
                getattr(data, "names", ())
                or getattr(columns, "names", ())
                or ()
            )
            if data is None or not names or not ({str(name).upper() for name in names} & set(_RATE_NAMES)):
                continue
            # Rate tables follow the same OGIP convention as GTIs: timing
            # metadata may live on either the extension or the primary HDU.
            # Extension cards take precedence, including an explicit zero.
            header = dict(primary_header)
            header.update(dict(hdu.header))
            if columns is not None:
                # Astropy stores table units in TUNITn header cards.  Expose
                # the rate unit under one stable key for the row normalizer.
                for index, name in enumerate(names, start=1):
                    if str(name).upper() in _RATE_NAMES:
                        unit = getattr(columns[index - 1], "unit", None)
                        if unit:
                            # The extension column is authoritative when a
                            # stale primary-header card advertises a
                            # different rate unit.
                            header["TUNIT_RATE"] = str(unit)
                            break
            for row in data:
                point = _canonical_rate_row(row, source_file=path, header=header)
                if source_name and point.source and not _source_name_matches(point.source, source_name):
                    continue
                points.append(point)
    return points


def _read_csv_rates(path: Path, source_name: str | None) -> list[SurveyRatePoint]:
    points: list[SurveyRatePoint] = []
    opener: Callable[..., Any] = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            point = _canonical_rate_row(row, source_file=path, header={})
            if source_name and point.source and not _source_name_matches(point.source, source_name):
                continue
            points.append(point)
    return points


def _read_numeric_dat_rates(path: Path) -> list[SurveyRatePoint] | None:
    """Read a headerless Burst-Analyser/QDP-style 14/15-column table.

    This is deliberately a small compatibility reader.  Fourteen-column
    products carry ``Flux`` and ``ECF`` but no native rate; fifteen-column
    products carry a native rate and asymmetric rate errors.  Such rows have
    no trustworthy pointing interval, so the time is retained as a scalar and
    window selection will leave them out unless a caller supplies intervals.
    """
    data = None
    for delimiter in (",", None):
        try:
            candidate = np.genfromtxt(path, delimiter=delimiter, dtype=float, comments="#")
        except (OSError, ValueError):
            continue
        candidate = np.atleast_2d(candidate)
        if candidate.shape[1] in {14, 15}:
            data = candidate
            break
    if data is None:
        return None
    if data.size == 0:
        return []
    if not np.isfinite(data[:, 0]).any():
        return None
    points: list[SurveyRatePoint] = []
    for row in data:
        if not np.isfinite(row[0]):
            continue
        if data.shape[1] == 14:
            # Burst Analyser's real fourteen-column layout is
            # ``Time, TimePos, TimeNeg, Flux, FluxPos, FluxNeg,
            # FluxPosWithECFErr, FluxNegWithECFErr, Gamma, GammaPos,
            # GammaNeg, ECF, ECFPos, ECFNeg``.  The two ``WithECFErr``
            # columns include conversion-factor uncertainty and must not be
            # treated as native rates or divided by ECF.  Use the fixed-ECF
            # flux errors in columns 4/5 for the statistical rate error.
            ecf = row[11]
            if np.isfinite(ecf) and ecf > 0:
                rate = row[3] / ecf
                error = max(abs(row[4]), abs(row[5])) / abs(ecf)
            else:
                # Keep the time and provenance of an invalid row for QA, but
                # make the unavailable derived rate explicit.  Downstream
                # window selection excludes rows without a finite rate.
                rate = np.nan
                error = np.nan
            rate_source = "flux_over_ecf"
        else:
            rate = row[6]
            error = max(abs(row[7]), abs(row[8]))
            rate_source = "native_rate"
        points.append(
            SurveyRatePoint(
                obsid=None,
                pointing_id=None,
                source=None,
                time_start=float(row[0]),
                time_stop=None,
                exposure_s=None,
                rate=float(rate) if np.isfinite(rate) else None,
                rate_error=float(error) if np.isfinite(error) else None,
                snr=signed_snr(rate, error),
                pcode=None,
                bad_bin=False,
                quality=None,
                rate_unit="count/s",
                error_source="flux_ecf_propagated" if data.shape[1] == 14 else "native_rate_error",
                source_file=str(path),
                rate_source=rate_source,
                channel_rate=(float(rate),),
                channel_error=(float(error),),
                channel_snr=(signed_snr(rate, error),),
                native_channel_count=1,
                total_band_is_last=False,
            )
        )
    return points


def _read_qdp_rates(path: Path) -> list[SurveyRatePoint]:
    """Read a standard three-, four-, or six-column QDP count-rate curve.

    QDP files are not BAT survey CAT products, but reference Swift workflows
    use them as a last-resort native count-rate source.  Their time errors do
    not define a full pointing exposure, so ``time_stop`` remains unset and
    window selection will not mistake a point estimate for a complete survey
    spectrum.
    """
    opener: Callable[..., Any] = gzip.open if path.name.lower().endswith(".gz") else open
    points: list[SurveyRatePoint] = []
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                text = line.strip()
                if (
                    not text
                    or text.startswith("!")
                    or text.upper().startswith(("READ", "NO", "LABEL", "LAB", "TIME"))
                ):
                    continue
                parts = text.replace(",", " ").split()
                if len(parts) < 3:
                    continue
                try:
                    values = [float(value) for value in parts[:6]]
                except ValueError:
                    continue
                if len(values) >= 6:
                    time_value, rate_value = values[0], values[3]
                else:
                    # QDP ``READ SERR 1`` commonly writes ``time rate err``;
                    # four-column exports use the last two entries as
                    # asymmetric errors.  Neither form carries a complete
                    # survey pointing interval.
                    time_value, rate_value = values[0], values[1]
                if not np.isfinite(time_value) or not np.isfinite(rate_value):
                    continue
                if len(values) >= 6:
                    error = max(abs(values[4]), abs(values[5]))
                elif len(values) == 3:
                    error = abs(values[2])
                else:
                    error = max(abs(value) for value in values[2:])
                points.append(
                    SurveyRatePoint(
                        obsid=None,
                        pointing_id=None,
                        source=None,
                        time_start=float(time_value),
                        time_stop=None,
                        exposure_s=None,
                        rate=float(rate_value),
                        rate_error=float(error) if np.isfinite(error) else None,
                        snr=signed_snr(rate_value, error),
                        pcode=None,
                        bad_bin=False,
                        quality=None,
                        rate_unit="count/s",
                        error_source="native_qdp_error",
                        source_file=str(path),
                        rate_source="native_qdp_rate",
                        channel_rate=(float(rate_value),),
                        channel_error=(float(error),),
                        channel_snr=(signed_snr(rate_value, error),),
                        native_channel_count=1,
                        total_band_is_last=False,
                    )
                )
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    return points


def read_bat_survey_rates(
    path: str | Path,
    *,
    source_name: str | None = None,
) -> tuple[SurveyRatePoint, ...]:
    """Read FITS/CAT/CSV rates while preserving the error column semantics."""
    path = Path(path).expanduser().resolve()
    name = path.name.lower()
    if name.endswith((".qdp", ".qdp.gz")):
        return tuple(_read_qdp_rates(path))
    if any(
        name.endswith(suffix)
        for suffix in (".csv", ".csv.gz", ".txt", ".txt.gz", ".dat", ".dat.gz")
    ):
        if name.endswith((".dat", ".dat.gz", ".txt", ".txt.gz")):
            numeric = _read_numeric_dat_rates(path)
            if numeric is not None:
                return tuple(numeric)
        return tuple(_read_csv_rates(path, source_name))
    return tuple(_read_fits_rates(path, source_name))


def _mosaic_source_measurements(
    product: str | Path,
    *,
    source_name: str,
    detection_threshold: float,
) -> list[dict[str, Any]]:
    """Read a mosaic's own source catalogue for detection/upper-limit flags.

    Pointing survey rates are deliberately not consulted here.  A mosaic is a
    separate weighted measurement, so its detection label must come from the
    ``sources_tot.cat`` (or equivalent source catalogue) generated for that
    mosaic.  Missing catalogue rows are represented as an explicit
    ``not_detected`` measurement rather than borrowing a member pointing's
    SNR.
    """
    root = Path(product).expanduser().resolve()
    if root.is_file():
        candidates = [root] if _has_suffix(root, (".cat", ".fits", ".fit")) else []
    elif root.is_dir():
        candidates = sorted(
            (item for item in root.rglob("*") if item.is_file() and _has_suffix(item, (".cat", ".fits", ".fit"))),
            key=lambda item: (0 if item.name.lower() == "sources_tot.cat" else 1, str(item)),
        )
    else:
        candidates = []
    parse_errors: list[str] = []
    for catalog in candidates:
        try:
            points = read_bat_survey_rates(catalog, source_name=source_name)
        except (OSError, ValueError, TypeError) as exc:
            parse_errors.append(f"{catalog}:{type(exc).__name__}:{exc}")
            continue
        if not points:
            continue
        measurements: list[dict[str, Any]] = []
        for point in points:
            rate_ok = point.rate is not None and np.isfinite(point.rate)
            error_ok = point.rate_error is not None and np.isfinite(point.rate_error) and point.rate_error > 0
            snr_ok = point.snr is not None and np.isfinite(point.snr)
            if not rate_ok or not error_ok or not snr_ok:
                status = "invalid_measurement"
                diagnostics = ["mosaic_rate_or_error_invalid"]
            else:
                status = (
                    "detected"
                    if float(point.snr) >= float(detection_threshold)
                    else "not_detected"
                )
                diagnostics = []
            measurements.append(
                {
                    "source": point.source,
                    "source_file": str(catalog),
                    "status": status,
                    "rate": point.rate,
                    "rate_error": point.rate_error,
                    "snr": point.snr,
                    "rate_unit": point.rate_unit,
                    "detector_normalization": "fully_illuminated_detector",
                    "detection_threshold": float(detection_threshold),
                    "detection_basis": "mosaic_source_catalog_snr",
                    "diagnostics": diagnostics,
                }
            )
        return measurements
    if parse_errors:
        return [
            {
                "source": source_name,
                "status": "invalid_measurement",
                "source_file": None,
                "rate": None,
                "rate_error": None,
                "snr": None,
                "rate_unit": None,
                "detector_normalization": "fully_illuminated_detector",
                "detection_threshold": float(detection_threshold),
                "detection_basis": "mosaic_source_catalog_snr",
                "diagnostics": parse_errors,
            }
        ]
    return [
        {
            "source": source_name,
            "status": "not_detected",
            "source_file": None,
            "rate": None,
            "rate_error": None,
            "snr": None,
            "rate_unit": None,
            "detector_normalization": "fully_illuminated_detector",
            "detection_threshold": float(detection_threshold),
            "detection_basis": "mosaic_source_catalog_snr",
            "diagnostics": ["source_not_in_mosaic_catalog"],
        }
    ]


def _time_value(value: Any) -> float:
    if isinstance(value, u.Quantity):
        return float(value.to_value(u.s))
    if isinstance(value, Time):
        try:
            return float(value.to_value("swift"))
        except Exception:
            return float(value.unix)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            parsed = Time(value, scale="utc")
            try:
                return float(parsed.to_value("swift"))
            except Exception:
                return float(parsed.unix)
    return float(value)


def _timeunit_scale(value: Any) -> float:
    """Return the seconds multiplier for an OGIP ``TIMEUNIT`` card.

    Swift survey timing is stored on a mission-elapsed axis.  FITS permits
    seconds, milliseconds, microseconds, and days; unknown cards are rejected
    to seconds by the same conservative convention used for older products.
    ``TIMEZERO`` is expressed in the same unit and is scaled by the caller.
    """
    decoded = _decode(value)
    token = str(decoded if decoded is not None else "s").strip().lower()
    token = token.strip("'\"").replace(" ", "")
    return {
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1.0e-3,
        "millisecond": 1.0e-3,
        "milliseconds": 1.0e-3,
        "us": 1.0e-6,
        "microsecond": 1.0e-6,
        "microseconds": 1.0e-6,
        "d": 86400.0,
        "day": 86400.0,
        "days": 86400.0,
    }.get(token, 1.0)


def _mjd_value(value: Any) -> float | None:
    """Normalize a HEASARC table time to MJD without assuming its type."""
    value = _decode(value)
    if value is None:
        return None
    if isinstance(value, Time):
        return float(value.utc.mjd)
    if hasattr(value, "mjd"):
        try:
            return float(value.mjd)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try:
            value = value.value
        except Exception:
            pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        try:
            numeric = float(Time(str(value), scale="utc").mjd)
        except (TypeError, ValueError, OverflowError):
            return None
    return numeric if np.isfinite(numeric) else None


def _time_utc_iso(value: Any) -> str:
    """Render a Swift-MET/UTC input on the recorded UTC axis."""
    if isinstance(value, str):
        try:
            return Time(value, scale="utc").utc.isot
        except Exception:
            pass
    numeric = _time_value(value)
    return Time(numeric, format="swift").utc.isot


def select_overlapping_pointings(
    points: Iterable[SurveyRatePoint],
    start: Any,
    stop: Any,
    *,
    min_pcode: float = 0.0,
    strict_pcode: bool = True,
    require_good: bool = True,
    gti_intervals: Mapping[str | None, Iterable[tuple[float, float]]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select full pointings overlapping a window with overlap diagnostics."""
    start_value, stop_value = _time_value(start), _time_value(stop)
    if stop_value <= start_value:
        raise ValueError("selection stop must be after start")
    selected: list[dict[str, Any]] = []
    for point in points:
        # A malformed Flux/ECF row can still be retained by the reader for
        # provenance, but it is not a usable source measurement.
        if point.rate is None or not np.isfinite(point.rate):
            continue
        if point.time_start is None or point.time_stop is None or point.time_stop <= point.time_start:
            continue
        # A complete pointing can be selected only when the product carries
        # a positive live-time value.  The interval itself is still retained
        # in ``points`` for diagnostics; treating its geometric duration as
        # exposure would create a scientific measurement from missing data.
        if point.exposure_s is None or not np.isfinite(point.exposure_s) or point.exposure_s <= 0:
            continue
        if not (point.time_start < stop_value and point.time_stop > start_value):
            continue
        if point.pcode is not None:
            ok = point.pcode > min_pcode if strict_pcode else point.pcode >= min_pcode
            if not ok:
                continue
        elif min_pcode > 0:
            continue
        if require_good and (
            point.bad_bin
            or (
                point.quality
                and point.quality.upper()
                not in {"SUCCESS", "GOOD", "TRUE", "1", "OK", "PASS"}
            )
        ):
            continue
        overlap = max(0.0, min(point.time_stop, stop_value) - max(point.time_start, start_value))
        gti_overlap = None
        if gti_intervals is not None:
            intervals = gti_intervals.get(point.pointing_id)
            if intervals is None and point.obsid is not None:
                intervals = gti_intervals.get(point.obsid)
            if intervals is None:
                intervals = gti_intervals.get(None)
            gti_overlap = gti_overlap_duration(intervals or (), start_value, stop_value)
            if gti_overlap <= 0:
                continue
        selected.append({
            "point": point.to_payload(),
            "requested_start": start_value,
            "requested_stop": stop_value,
            "requested_start_utc": _time_utc_iso(start),
            "requested_stop_utc": _time_utc_iso(stop),
            "pointing_start_swift": float(point.time_start),
            "pointing_stop_swift": float(point.time_stop),
            "pointing_start_utc": _time_utc_iso(point.time_start),
            "pointing_stop_utc": _time_utc_iso(point.time_stop),
            "overlap_s": overlap,
            "overlap_fraction": overlap / (point.time_stop - point.time_start),
            "effective_exposure_s": point.exposure_s,
            "gti_overlap_s": gti_overlap,
            "selection": "overlap_full_pointing",
            "detected_snr3": bool(point.snr is not None and point.snr >= 3.0),
            "detected_snr5": bool(point.snr is not None and point.snr >= 5.0),
        })
    return tuple(selected)


def read_gti_intervals(path: str | Path) -> tuple[tuple[float, float], ...]:
    """Read ``START``/``STOP`` GTI rows as Swift-MET seconds."""
    path = Path(path).expanduser().resolve()
    intervals: list[tuple[float, float]] = []
    try:
        with fits.open(path, memmap=False) as hdus:
            primary_header = hdus[0].header if hdus else {}
            for hdu in hdus:
                data = hdu.data
                names = {
                    str(name).upper(): name
                    for name in (getattr(data, "names", ()) or ())
                }
                if data is None or not {"START", "STOP"}.issubset(names):
                    continue
                # OGIP writers normally put TIMEZERO/TIMEUNIT on the GTI
                # extension, but older Swift products leave one or both in
                # the primary header.  Use the extension value when present
                # and fall back only when it is absent.
                timezero = _float(
                    hdu.header["TIMEZERO"]
                    if "TIMEZERO" in hdu.header
                    else primary_header.get("TIMEZERO")
                ) or 0.0
                timeunit = str(
                    hdu.header["TIMEUNIT"]
                    if "TIMEUNIT" in hdu.header
                    else primary_header.get("TIMEUNIT", "s")
                ).strip().lower()
                unit_scale = {
                    "s": 1.0,
                    "sec": 1.0,
                    "second": 1.0,
                    "ms": 1.0e-3,
                    "millisecond": 1.0e-3,
                    "us": 1.0e-6,
                    "d": 86400.0,
                    "day": 86400.0,
                }.get(timeunit, 1.0)
                for row in data:
                    start = _scalar_value(row[names["START"]])
                    stop = _scalar_value(row[names["STOP"]])
                    if start is not None and stop is not None:
                        start = start * unit_scale + timezero * unit_scale
                        stop = stop * unit_scale + timezero * unit_scale
                    if start is not None and stop is not None and stop > start:
                        intervals.append((start, stop))
                if intervals:
                    break
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"cannot read Swift GTI file {path}: {exc}") from exc
    if not intervals:
        raise ValueError(f"Swift GTI file has no valid START/STOP rows: {path}")
    return tuple(intervals)


def gti_overlap_duration(
    intervals: Iterable[tuple[float, float]],
    start: Any,
    stop: Any,
) -> float:
    """Return union exposure overlapping a requested Swift/UTC interval."""
    start_value, stop_value = _time_value(start), _time_value(stop)
    if stop_value <= start_value:
        raise ValueError("GTI overlap stop must be after start")
    clipped = sorted(
        (
            max(start_value, float(left)),
            min(stop_value, float(right)),
        )
        for left, right in intervals
        if float(right) > float(left)
        and float(right) > start_value
        and float(left) < stop_value
    )
    total = 0.0
    current_start = current_stop = None
    for left, right in clipped:
        if current_start is None:
            current_start, current_stop = left, right
        elif left <= current_stop:
            current_stop = max(current_stop, right)
        else:
            total += current_stop - current_start
            current_start, current_stop = left, right
    if current_start is not None:
        total += current_stop - current_start
    return float(total)


@dataclass(frozen=True, slots=True)
class SurveyPhaValidation:
    path: Path
    valid: bool
    channels: int
    exposure_s: float | None
    response: Path | None
    ancillary_response: Path | None
    background: Path | None
    backscal: float | None
    columns: tuple[str, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SurveyUpperLimitResult:
    """Response-folded fixed-index upper-limit result."""

    status: str
    flux_erg_cm2_s: float | None
    upper_norm: float | None
    best_norm: float | None
    fit_statistic: float | None
    delta_stat: float
    photon_index: float
    parameter_status: str | None
    diagnostics: tuple[str, ...] = ()
    normalization_at_lower_bound: bool = False
    profile_delta_stat: float | None = None
    profile_delta_tolerance: float | None = None
    signed_mle: float | None = None
    constrained_mle: float | None = None
    construction: str = "xspec_profile"
    calibration_status: str = "conditional_model"
    profile_status: str = "unavailable"
    energy_band_keV: tuple[float, float] | None = None
    rate_unit: str | None = None
    template_rate: tuple[float, ...] | None = None
    confidence_level: float = NormalDist().cdf(3.0)
    confidence_convention: str = "one_sided_gaussian_equivalent"
    # Keep the background contract beside the numerical result.  This is
    # deliberately a mapping rather than a BatAnalysis object so that a
    # failed/conditional profile can still be written to the JSON report
    # without importing the optional backend at read time.
    background_likelihood: str = "gaussian_net_rate"
    background_provenance: Mapping[str, Any] = field(default_factory=dict)
    covariance_source: str | None = None
    residual_validation: Mapping[str, Any] | None = None


def _header_path(path: Path, value: Any) -> Path | None:
    text = str(_decode(value) or "").strip()
    if not text or text.upper() in {"NONE", "NULL", "-"}:
        return None
    candidate = Path(text)
    return candidate if candidate.is_absolute() else path.parent / candidate


def validate_survey_pha(
    path: str | Path,
    *,
    require_response: bool = True,
    require_matrix: bool = False,
) -> SurveyPhaValidation:
    """Validate a survey PHA and its response links without loading XSPEC.

    ``require_matrix`` is opt-in for callers that need a science-ready
    response.  The default keeps the historical EBOUNDS-only inspection
    useful for lightweight product inventory, while the pipeline's spectra
    gate enables it before any fit or upper-limit calculation.
    """
    path = Path(path).expanduser().resolve()
    diagnostics: list[str] = []
    response = ancillary = background = None
    channels = 0
    spectrum_channel_values: np.ndarray | None = None
    exposure = None
    backscal = None
    columns: tuple[str, ...] = ()
    try:
        with fits.open(path, memmap=False) as hdus:
            spectrum_hdu = next(
                (
                    hdu
                    for hdu in hdus
                    if getattr(hdu, "data", None) is not None
                    and bool(getattr(hdu.data, "names", None))
                    and "CHANNEL" in {str(name).upper() for name in hdu.data.names}
                    and bool(
                        {"RATE", "COUNTS", "CENT_RATE"}
                        & {str(name).upper() for name in hdu.data.names}
                    )
                ),
                None,
            )
            if spectrum_hdu is None:
                diagnostics.append("missing_spectrum_extension")
            else:
                data = spectrum_hdu.data
                primary_header = hdus[0].header
                def header_value(name: str) -> Any:
                    # An explicitly present zero/``NONE`` in the spectrum
                    # extension is a real product value; do not replace it
                    # with a potentially stale primary-header fallback.
                    return (
                        spectrum_hdu.header[name]
                        if name in spectrum_hdu.header
                        else primary_header.get(name)
                    )

                columns = tuple(str(item) for item in (data.names or ()))
                channels = len(data)
                column_names = {item.upper(): item for item in columns}
                upper = set(column_names)
                if channels <= 0:
                    diagnostics.append("empty_spectrum_table")
                if "CHANNEL" not in upper:
                    diagnostics.append("missing_CHANNEL")
                else:
                    try:
                        spectrum_channel_values = np.asarray(
                            data[column_names["CHANNEL"]], dtype=float
                        )
                        if spectrum_channel_values.shape != (channels,) or np.any(
                            ~np.isfinite(spectrum_channel_values)
                        ):
                            diagnostics.append("invalid_CHANNEL")
                    except (TypeError, ValueError):
                        diagnostics.append("invalid_CHANNEL")
                if not ({"RATE", "COUNTS", "CENT_RATE"} & upper):
                    diagnostics.append("missing_rate_or_counts_column")
                if "RATE" in upper and "STAT_ERR" not in upper:
                    diagnostics.append("missing_STAT_ERR")
                for value_name in ("RATE", "COUNTS", "CENT_RATE"):
                    actual_name = column_names.get(value_name)
                    if actual_name is None:
                        continue
                    try:
                        values = np.asarray(data[actual_name], dtype=float)
                        if values.shape[0] != channels or np.any(~np.isfinite(values)):
                            diagnostics.append(f"invalid_{value_name}")
                        if value_name == "COUNTS" and np.any(values < 0):
                            diagnostics.append("negative_COUNTS")
                    except (TypeError, ValueError):
                        diagnostics.append(f"invalid_{value_name}")
                for error_name in ("STAT_ERR", "SYS_ERR"):
                    actual_name = column_names.get(error_name)
                    if actual_name is not None:
                        try:
                            error_values = np.asarray(data[actual_name], dtype=float)
                            if error_values.shape != (channels,) or np.any(
                                ~np.isfinite(error_values)
                            ) or np.any(error_values < 0):
                                diagnostics.append(f"invalid_{error_name}")
                        except (TypeError, ValueError):
                            diagnostics.append(f"invalid_{error_name}")
                exposure = _float(
                    header_value("EXPOSURE")
                )
                if exposure is None or exposure <= 0:
                    diagnostics.append("missing_or_invalid_EXPOSURE")
                backscal = _float(
                    header_value("BACKSCAL")
                )
                if backscal is None and "BACKSCAL" in column_names:
                    try:
                        values = np.asarray(data[column_names["BACKSCAL"]], dtype=float)
                        finite = values[np.isfinite(values) & (values > 0)]
                        if finite.size:
                            backscal = float(np.nanmedian(finite))
                        if finite.size != values.size:
                            diagnostics.append("invalid_BACKSCAL")
                    except (TypeError, ValueError):
                        diagnostics.append("invalid_BACKSCAL")
                if backscal is None or backscal <= 0:
                    diagnostics.append("missing_or_invalid_BACKSCAL")
                response = _header_path(
                    path,
                    header_value("RESPFILE"),
                )
                ancillary = _header_path(
                    path,
                    header_value("ANCRFILE"),
                )
                background = _header_path(
                    path,
                    header_value("BACKFILE"),
                )
    except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
        diagnostics.append(f"fits_read_failed:{exc}")
    if require_response and (response is None or not response.is_file()):
        diagnostics.append("missing_response")
    if response is not None and response.is_file():
        try:
            response_channels = None
            response_channel_values: np.ndarray | None = None
            response_has_ebounds = False
            response_has_matrix = False
            matrix_detchans: int | None = None
            with fits.open(response, memmap=False) as response_hdus:
                for response_hdu in response_hdus:
                    response_data = response_hdu.data
                    response_columns = {
                        str(name).upper(): name
                        for name in (getattr(response_data, "names", ()) or ())
                    }
                    response_names = set(response_columns)
                    extension_name = str(getattr(response_hdu, "name", "")).upper()
                    hduclas2 = str(response_hdu.header.get("HDUCLAS2", "")).upper()
                    if (
                        response_data is not None
                        and "MATRIX" in response_names
                        and ("RSP_MATRIX" in hduclas2 or "MATRIX" in extension_name)
                    ):
                        response_has_matrix = True
                        detchans = response_hdu.header.get("DETCHANS")
                        try:
                            matrix_detchans = int(detchans) if detchans is not None else None
                        except (TypeError, ValueError):
                            matrix_detchans = None
                            diagnostics.append("response_invalid_DETCHANS")
                        if matrix_detchans is not None and matrix_detchans <= 0:
                            diagnostics.append("response_invalid_DETCHANS")
                        matrix_column = response_columns["MATRIX"]
                        raw_matrix = response_data[matrix_column]
                        # OGIP RMFs may store a dense matrix or a variable-length
                        # row for each incident-energy bin.  Validate the numeric
                        # contract without forcing sparse rows to share a width;
                        # ``N_CHAN``/``F_CHAN`` carry that layout in the latter
                        # representation.  Negative, non-finite or missing rows
                        # are never science-ready, even when EBOUNDS is present.
                        try:
                            matrix_array = np.asarray(raw_matrix, dtype=float)
                        except (TypeError, ValueError):
                            matrix_array = None
                        if matrix_array is not None and matrix_array.dtype != object:
                            matrix_rows = [matrix_array[index] for index in range(len(response_data))]
                        else:
                            try:
                                matrix_rows = [np.asarray(row, dtype=float) for row in raw_matrix]
                            except (TypeError, ValueError):
                                matrix_rows = []
                                diagnostics.append("response_invalid_MATRIX")
                        if len(matrix_rows) != len(response_data):
                            diagnostics.append("response_matrix_row_mismatch")
                        for row in matrix_rows:
                            if (
                                np.asarray(row).ndim != 1
                                or np.asarray(row).size == 0
                                or np.any(~np.isfinite(np.asarray(row, dtype=float)))
                                or np.any(np.asarray(row, dtype=float) < 0)
                            ):
                                diagnostics.append("response_invalid_MATRIX")
                                break
                        if (
                            matrix_array is not None
                            and matrix_array.ndim == 2
                            and matrix_detchans is not None
                            and matrix_array.shape[1] != matrix_detchans
                        ):
                            diagnostics.append(
                                "response_matrix_column_mismatch:"
                                f"{matrix_array.shape[1]}!={matrix_detchans}"
                            )
                    # A response MATRIX can also carry a CHANNEL column.  It
                    # is the EBOUNDS extension (or an equivalent E_MIN/E_MAX
                    # table) that establishes the channel-to-energy contract
                    # for a grouped survey PHA.
                    is_ebounds = (
                        "CHANNEL" in response_names
                        and {"E_MIN", "E_MAX"}.issubset(response_names)
                        and (
                            extension_name == "EBOUNDS"
                            or str(response_hdu.header.get("HDUCLAS2", "")).upper()
                            == "EBOUNDS"
                        )
                    )
                    if is_ebounds and response_data is not None:
                        response_has_ebounds = True
                        response_channels = len(response_data)
                        try:
                            response_channel_values = np.asarray(
                                response_data[response_columns["CHANNEL"]],
                                dtype=float,
                            )
                            if (
                                response_channel_values.shape != (response_channels,)
                                or np.any(~np.isfinite(response_channel_values))
                            ):
                                response_channel_values = None
                                diagnostics.append("response_invalid_CHANNEL")
                            for energy_name in ("E_MIN", "E_MAX"):
                                values = np.asarray(
                                    response_data[response_columns[energy_name]],
                                    dtype=float,
                                )
                                if (
                                    values.shape != (response_channels,)
                                    or np.any(~np.isfinite(values))
                                ):
                                    diagnostics.append(f"response_invalid_{energy_name}")
                            if all(
                                f"response_invalid_{energy_name}" not in diagnostics
                                for energy_name in ("E_MIN", "E_MAX")
                            ):
                                e_min = np.asarray(
                                    response_data[response_columns["E_MIN"]],
                                    dtype=float,
                                )
                                e_max = np.asarray(
                                    response_data[response_columns["E_MAX"]],
                                    dtype=float,
                                )
                                if np.any(e_max <= e_min):
                                    diagnostics.append("response_invalid_energy_bounds")
                        except (TypeError, ValueError, KeyError):
                            diagnostics.append("response_invalid_CHANNEL")
                        # Do not stop after EBOUNDS: OGIP responses may store
                        # EBOUNDS before the SPECRESP MATRIX extension.  Both
                        # contracts must be inspected before deciding that a
                        # response is science-ready.
            if not response_has_ebounds:
                diagnostics.append("response_missing_EBOUNDS")
            if require_matrix and not response_has_matrix:
                diagnostics.append("response_missing_MATRIX")
            if (
                require_matrix
                and matrix_detchans is not None
                and channels
                and matrix_detchans != channels
            ):
                diagnostics.append(
                    f"response_matrix_channel_mismatch:{matrix_detchans}!={channels}"
                )
            if response_channels is not None and channels and response_channels != channels:
                diagnostics.append(
                    f"response_channel_mismatch:{response_channels}!={channels}"
                )
            elif response_channel_values is not None and spectrum_channel_values is not None:
                channels_match = np.array_equal(
                    response_channel_values, spectrum_channel_values
                )
                if not channels_match and len(response_channel_values) > 1:
                    # OGIP products legitimately use different zero points:
                    # the BAT survey PHA commonly numbers channels 1..8
                    # while the response EBOUNDS uses 0..7.  Preserve that
                    # compatibility when both sequences are contiguous with
                    # the same step and differ only by a constant offset;
                    # reject reordered or partially missing channels.
                    response_step = np.diff(response_channel_values)
                    spectrum_step = np.diff(spectrum_channel_values)
                    same_grid = np.allclose(
                        response_step,
                        spectrum_step,
                        rtol=0.0,
                        atol=0.0,
                    ) and np.allclose(
                        response_channel_values - spectrum_channel_values,
                        response_channel_values[0] - spectrum_channel_values[0],
                        rtol=0.0,
                        atol=0.0,
                    )
                    channels_match = bool(same_grid)
                if not channels_match:
                    diagnostics.append("response_channel_values_mismatch")
        except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
            diagnostics.append(f"response_read_failed:{exc}")
    if ancillary is not None and not ancillary.is_file():
        diagnostics.append("missing_ancillary_response")
    if background is not None and not background.is_file():
        diagnostics.append("missing_background")
    return SurveyPhaValidation(
        path=path,
        valid=not diagnostics,
        channels=channels,
        exposure_s=exposure,
        response=response if response and response.is_file() else None,
        ancillary_response=ancillary if ancillary and ancillary.is_file() else None,
        background=background if background and background.is_file() else None,
        backscal=backscal,
        columns=columns,
        diagnostics=tuple(diagnostics),
    )


def _pha_time_interval(path: Path) -> tuple[float, float] | None:
    """Read a PHA's Swift-MET interval for rate-to-spectrum matching."""
    try:
        with fits.open(path, memmap=False) as hdus:
            header = dict(hdus[0].header) if hdus else {}
            if len(hdus) > 1:
                header.update(dict(hdus[1].header))
    except (OSError, ValueError):
        return None
    raw_start = header.get("TSTART")
    raw_stop = header.get("TSTOP")
    start = _scalar_value(raw_start)
    stop = _scalar_value(raw_stop)
    exposure = _scalar_value(header.get("EXPOSURE"))
    timezero = _float(header.get("TIMEZERO")) or 0.0
    unit_scale = _timeunit_scale(header.get("TIMEUNIT", "s"))

    def _is_textual_time(value: Any) -> bool:
        decoded = _decode(value)
        if isinstance(decoded, str):
            try:
                float(decoded.strip())
            except (TypeError, ValueError):
                return True
        return False

    if start is not None and not _is_textual_time(raw_start):
        start = start * unit_scale + timezero * unit_scale
    if stop is not None and not _is_textual_time(raw_stop):
        stop = stop * unit_scale + timezero * unit_scale
    if stop is None and start is not None and exposure is not None:
        stop = start + exposure
    if start is None or stop is None or stop <= start:
        return None
    return start, stop


_PHA_SCOPE_SOURCE_KEYS = (
    "OBJECT", "SOURCE", "SRCNAME", "SRC_NAME", "SOURCE_NAME", "TARGET",
    "TARGETID", "TARGET_ID",
)
_PHA_SCOPE_POINTING_KEYS = (
    "POINTING", "POINTING_ID", "PNT_ID", "IMAGE_ID", "IMAGEID",
)
_PHA_SCOPE_OBSID_KEYS = ("OBSID", "OBS_ID", "OBS_ID_", "SWIFTID")
_PHA_SCOPE_GENERIC_NAMES = frozenset(
    {"", "UNKNOWN", "UNDEFINED", "NONE", "NULL", "BAT", "SWIFT", "SURVEY", "SOURCE"}
)
_PHA_MANUAL_UPPER_LIMIT_RE = re.compile(
    r"(?:upper[_-]?lim|upper[_-]?limit|bkgnsigma|n[_-]?sigma|manual[_-]?upper)",
    re.IGNORECASE,
)


def _pha_scope_metadata(path: Path) -> dict[str, tuple[str, ...]]:
    """Read optional source/pointing identifiers from a PHA without XSPEC.

    Survey result directories often contain PHA files for several catalogue
    sources and old hand-built ``bkgnsigma_*_upperlim`` files.  FITS headers
    are the authoritative identifiers when present; filename tokens are used
    only as a conservative fallback for OBSID/pointing labels.  Missing
    metadata is represented by an empty tuple so legacy valid products are not
    rejected solely because an older writer omitted optional keywords.
    """
    values: dict[str, set[str]] = {key: set() for key in ("source", "pointing", "obsid")}
    try:
        with fits.open(path, memmap=False) as hdus:
            for hdu in hdus:
                header = getattr(hdu, "header", {})
                for key in _PHA_SCOPE_SOURCE_KEYS:
                    if key in header:
                        text = str(_decode(header[key]) or "").strip()
                        if text:
                            values["source"].add(text)
                for key in _PHA_SCOPE_POINTING_KEYS:
                    if key in header:
                        text = str(_decode(header[key]) or "").strip()
                        if text:
                            values["pointing"].add(text)
                for key in _PHA_SCOPE_OBSID_KEYS:
                    if key in header:
                        text = str(_decode(header[key]) or "").strip()
                        if text:
                            values["obsid"].add(text)
                data = getattr(hdu, "data", None)
                names = {
                    str(name).upper(): name
                    for name in (getattr(data, "names", ()) or ())
                }
                if data is None or len(data) == 0:
                    continue
                for kind, keys in (
                    ("source", _SOURCE_NAMES),
                    ("pointing", _POINTING_NAMES),
                    ("obsid", ("OBSID", "OBS_ID", "OBS_ID_")),
                ):
                    actual = next((names[key] for key in keys if key in names), None)
                    if actual is None:
                        continue
                    try:
                        for item in np.asarray(data[actual]).reshape(-1):
                            text = str(_decode(item) or "").strip()
                            if text:
                                values[kind].add(text)
                    except (TypeError, ValueError):
                        continue
    except (OSError, ValueError, IndexError, KeyError, TypeError):
        return {key: () for key in values}

    # A product's basename commonly carries ``<source>_point_<id>`` and
    # ``<obsid>``.  Use these only when the FITS header/table omitted the
    # corresponding identifier; a stale filename token must not override an
    # explicit header value.
    stem = path.name
    if not values["pointing"]:
        for match in re.finditer(
            r"(?:point(?:ing)?[_-]?)([A-Za-z0-9_-]+)", stem, re.IGNORECASE
        ):
            values["pointing"].add(match.group(1))
    if not values["obsid"]:
        for match in re.finditer(r"(?<!\d)(\d{8,11})(?!\d)", stem):
            values["obsid"].add(match.group(1))
    return {key: tuple(sorted(item)) for key, item in values.items()}


def _scope_identifier_matches(values: Sequence[str], allowed: set[str]) -> bool:
    """Return whether any explicit identifier belongs to an allowed set."""
    if not values or not allowed:
        return True
    def normalize(value: object) -> str:
        # BatAnalysis and the Swift survey catalogue use both ``123`` and
        # ``point_123``/``pointing_123`` for the same image identifier.  The
        # prefix is a label, not part of the identifier.  Strip it after
        # removing separators so a selected light-curve row can match the
        # token recovered from a PHA filename or header.
        token = re.sub(r"[^0-9A-Za-z.]", "", str(value)).lower()
        token = re.sub(r"^(?:pointing|point)", "", token)
        return token
    normalized_allowed = {normalize(item) for item in allowed}
    return any(normalize(value) in normalized_allowed for value in values)


def _core_survey_gaussian_profile(
    pha: Path,
    response: Path,
    *,
    ancillary_response_path: Path | None = None,
    photon_index: float,
    delta_stat: float,
    energy_range_keV: tuple[float, float],
) -> SurveyUpperLimitResult:
    """Run the core signed-rate GLS profile for a science-ready BAT PHA.

    The BAT survey PHA is a background-subtracted rate spectrum.  Its
    ``STAT_ERR`` is the statistical Gaussian error and ``SYS_ERR`` follows the
    OGIP fractional-error convention; the latter is converted to an absolute
    rate error exactly once.  A deterministic XSPEC ``fakeit`` template is
    folded through the same response and exposure, then converted to rate.
    """
    from jinwu.core.upperlimit import (
        CountTemplateModel,
        GaussianNetRateObservation,
        UpperLimitObservation,
        XspecCountPredictor,
        profile_gaussian_upper_bound,
    )

    emin, emax = (float(value) for value in energy_range_keV)
    with fits.open(pha, memmap=False) as hdus:
        spectrum = next(
            (
                hdu
                for hdu in hdus
                if getattr(hdu, "data", None) is not None
                and getattr(hdu.data, "names", None)
                and {str(name).upper() for name in hdu.data.names}
                >= {"CHANNEL", "RATE", "STAT_ERR"}
            ),
            None,
        )
        if spectrum is None:
            raise ValueError("BAT PHA lacks RATE and STAT_ERR columns")
        names = {str(name).upper(): name for name in spectrum.data.names}
        exposure = _float(spectrum.header.get("EXPOSURE"))
        if exposure is None or exposure <= 0:
            raise ValueError("BAT PHA exposure is missing or invalid")
        rate = np.asarray(spectrum.data[names["RATE"]], dtype=float)
        stat = np.asarray(spectrum.data[names["STAT_ERR"]], dtype=float)
        if rate.ndim != 1 or stat.shape != rate.shape or np.any(~np.isfinite(rate)):
            raise ValueError("BAT PHA RATE/STAT_ERR arrays are invalid")
        if np.any(~np.isfinite(stat)) or np.any(stat <= 0):
            raise ValueError("BAT PHA STAT_ERR must be finite and positive")
        sys_name = names.get("SYS_ERR")
        sys_fraction = (
            np.asarray(spectrum.data[sys_name], dtype=float)
            if sys_name is not None
            else np.zeros_like(rate)
        )
        if sys_fraction.shape != rate.shape or np.any(~np.isfinite(sys_fraction)) or np.any(sys_fraction < 0):
            raise ValueError("BAT PHA SYS_ERR is invalid")
        ebounds = next(
            (
                hdu
                for hdu in hdus
                if str(getattr(hdu, "name", "")).upper() == "EBOUNDS"
                and getattr(hdu, "data", None) is not None
            ),
            None,
        )
        if ebounds is None:
            raise ValueError("BAT PHA lacks EBOUNDS")
        elo = np.asarray(ebounds.data["E_MIN"], dtype=float)
        ehi = np.asarray(ebounds.data["E_MAX"], dtype=float)
        if elo.shape != rate.shape or ehi.shape != rate.shape:
            raise ValueError("BAT PHA RATE and EBOUNDS channel shapes differ")
        overlap = (elo < emax) & (ehi > emin)
        if not np.any(overlap):
            raise ValueError("BAT PHA has no channels in the requested energy band")
        rate = rate[overlap]
        stat = stat[overlap]
        sys_fraction = sys_fraction[overlap]

    # A combined ``.rsp`` carries both the redistribution matrix and area.
    # For split products the first path may instead be an ARF, with the RMF
    # supplied in ``ancillary_response_path``.  Keep this pairing explicit so
    # a valid RMF+ARF product cannot be folded as an ARF-only response.
    if _response_is_arf(response):
        if ancillary_response_path is None:
            raise ValueError("an ARF response requires an ancillary RMF path")
        rmf_path = ancillary_response_path
        arf_path = response
    else:
        rmf_path = response
        arf_path = ancillary_response_path

    # XspecCountPredictor reads the response EBOUNDS and applies the same
    # overlap selection.  The dummy source vector is never used for folding.
    dummy = UpperLimitObservation(
        name=pha.stem,
        source_counts=np.zeros(rate.size, dtype=float),
        exposure_s=exposure,
        response_path=rmf_path,
        arf_path=arf_path,
    )
    predictor = XspecCountPredictor(
        model_expression="powerlaw",
        parameters=(float(photon_index), 1.0),
    )
    template_counts = np.asarray(
        predictor.expected_counts(dummy, interval=(0.0, exposure), energy_band=(emin, emax)),
        dtype=float,
    )
    if template_counts.shape != rate.shape:
        raise ValueError(
            f"response template has {template_counts.size} channels, PHA has {rate.size}"
        )
    template_rate = template_counts / exposure
    absolute_sys = np.abs(rate) * sys_fraction
    covariance = np.diag(stat**2 + absolute_sys**2)
    rate_unit = u.ct / u.s
    observation = GaussianNetRateObservation(
        name=pha.stem,
        net_rate=rate * rate_unit,
        unit_source_rate=template_rate * rate_unit,
        covariance=covariance * rate_unit**2,
        exposure=exposure * u.s,
        energy_band_keV=(emin, emax),
        detector_normalization="fully_illuminated_detector",
        provenance={
            "pha": str(pha),
            "response": str(response),
            "stat_err_source": "PHA.STAT_ERR",
            "sys_err_source": "PHA.SYS_ERR_fractional",
            "sys_err_added_once": True,
            "bkg_var_used_for_fit": False,
            "detector_normalization": "fully_illuminated_detector",
            "template": "XSPEC fakeit(applyStats=False)",
        },
    )
    flux_factor = _powerlaw_energy_flux_per_norm(photon_index, emin, emax)
    profile = profile_gaussian_upper_bound(
        observation,
        delta_stat=delta_stat,
        amplitude_unit="xspec_powerlaw_norm_ph_cm2_s_keV_at_1keV",
        flux_per_amplitude=flux_factor * u.erg / (u.cm**2 * u.s),
    )
    diagnostics = list(profile.diagnostics)
    return SurveyUpperLimitResult(
        status="upper_limit_ready" if profile.status == "ready" else "failed",
        flux_erg_cm2_s=profile.flux_upper,
        upper_norm=profile.upper_bound,
        best_norm=profile.constrained_mle,
        fit_statistic=profile.fit_statistic,
        delta_stat=delta_stat,
        photon_index=float(photon_index),
        parameter_status="conditional_model" if profile.status == "ready" else None,
        diagnostics=tuple(diagnostics),
        normalization_at_lower_bound=profile.boundary == "lower_bound",
        # This is the value of the Gaussian statistic at the numerical root,
        # rather than a copy of the requested target.  The distinction keeps
        # the persisted recheck auditable when a response or covariance is
        # changed.
        profile_delta_stat=profile.profile_delta_at_upper,
        profile_delta_tolerance=0.02,
        signed_mle=profile.signed_mle,
        constrained_mle=profile.constrained_mle,
        construction=profile.construction,
        calibration_status=profile.calibration_status,
        profile_status="converged" if profile.converged else "failed",
        energy_band_keV=(emin, emax),
        rate_unit=str(rate_unit),
        template_rate=tuple(float(value) for value in template_rate),
        confidence_level=float(NormalDist().cdf(math.sqrt(delta_stat))),
        confidence_convention="one_sided_gaussian_equivalent",
        background_likelihood="gaussian_net_rate",
        background_provenance={
            "pha": str(pha),
            "response": str(response),
            "stat_err_source": "PHA.STAT_ERR",
            "sys_err_source": "PHA.SYS_ERR_fractional",
            "sys_err_added_once": True,
            "bkg_var_used_for_fit": False,
            "detector_normalization": "fully_illuminated_detector",
            "template": "XSPEC fakeit(applyStats=False)",
        },
        covariance_source=(
            "diag(PHA.STAT_ERR^2 + (abs(RATE)*PHA.SYS_ERR)^2)"
            if np.any(sys_fraction > 0)
            else "diag(PHA.STAT_ERR^2)"
        ),
        residual_validation=None,
    )


def _powerlaw_energy_flux_per_norm(gamma: float, emin: float, emax: float) -> float:
    """Energy flux in erg cm^-2 s^-1 for XSPEC powerlaw norm one."""
    exponent = 2.0 - float(gamma)
    integral = (
        np.log(emax / emin)
        if abs(exponent) < 1e-12
        else (emax**exponent - emin**exponent) / exponent
    )
    return float(1.602176634e-9 * integral)


def profile_survey_upper_limit(
    pha_path: str | Path,
    response_path: str | Path,
    *,
    ancillary_response_path: str | Path | None = None,
    photon_index: float = 2.0,
    delta_stat: float = 9.0,
    energy_range_keV: tuple[float, float] = (14.0, 195.0),
    output_dir: str | Path,
) -> SurveyUpperLimitResult:
    """Profile a fixed-index power law for a signed BAT survey PHA.

    Survey PHA values are background-subtracted Gaussian rates, so this uses
    XSPEC ``chi`` and a non-negative power-law normalization.  It intentionally
    does not call the Poisson ``core.upperlimit`` observation preparation path.
    The returned status includes the raw XSPEC error status and rejects a
    profile that ends at its hard upper bound.
    """
    emin, emax = (float(value) for value in energy_range_keV)
    if not np.isfinite(emin) or not np.isfinite(emax) or not 0 < emin < emax:
        raise ValueError("energy_range_keV must be finite and increasing")
    photon_index = float(photon_index)
    delta_stat = float(delta_stat)
    if not np.isfinite(photon_index) or photon_index <= 0:
        raise ValueError("photon_index must be finite and positive")
    if not np.isfinite(delta_stat) or delta_stat <= 0:
        raise ValueError("delta_stat must be finite and positive")
    try:
        from jinwu.core.upperlimit import UpperLimit
        import xspec
    except Exception as exc:  # pragma: no cover - depends on HEASoft/PyXspec
        return SurveyUpperLimitResult(
            status="unavailable",
            flux_erg_cm2_s=None,
            upper_norm=None,
            best_norm=None,
            fit_statistic=None,
            delta_stat=delta_stat,
            photon_index=photon_index,
            parameter_status=None,
            normalization_at_lower_bound=False,
            diagnostics=(f"xspec_unavailable:{exc}",),
            construction="xspec_profile",
            calibration_status="unavailable",
            profile_status="unavailable",
            energy_band_keV=(emin, emax),
            rate_unit="ct / s",
            confidence_level=float(NormalDist().cdf(math.sqrt(delta_stat))),
            confidence_convention="one_sided_gaussian_equivalent",
        )
    pha = Path(pha_path).expanduser().resolve()
    response = Path(response_path).expanduser().resolve()
    ancillary = (
        Path(ancillary_response_path).expanduser().resolve()
        if ancillary_response_path is not None
        else None
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostics: list[str] = []
    profile_delta_stat: float | None = None
    profile_delta_tolerance = 0.02
    # Science-ready survey PHA products use the shared signed-rate GLS
    # implementation.  Keep the historical XSPEC error-command path below as
    # a compatibility fallback for lightweight shims and older products that
    # cannot expose RATE/STAT_ERR cleanly; such a fallback is labelled in the
    # returned construction field rather than silently mixed with the core
    # result.
    core_candidate = False
    try:
        with fits.open(pha, memmap=False) as hdus:
            core_candidate = any(
                getattr(hdu, "data", None) is not None
                and bool(getattr(hdu.data, "names", None))
                and "CHANNEL" in {str(name).upper() for name in hdu.data.names}
                and "RATE" in {str(name).upper() for name in hdu.data.names}
                for hdu in hdus
            )
    except (OSError, ValueError, IndexError, TypeError):
        # Lightweight shims and pre-OGIP test doubles are kept on the
        # historical XSPEC compatibility path below.  A readable RATE PHA,
        # however, is a science product and must never be silently replaced
        # by a different likelihood when its response/profile fails.
        core_candidate = False
    if core_candidate:
        try:
            return _core_survey_gaussian_profile(
                pha,
                response,
                ancillary_response_path=ancillary,
                photon_index=photon_index,
                delta_stat=delta_stat,
                energy_range_keV=(emin, emax),
            )
        except Exception as exc:
            return SurveyUpperLimitResult(
                status="failed",
                flux_erg_cm2_s=None,
                upper_norm=None,
                best_norm=None,
                fit_statistic=None,
                delta_stat=delta_stat,
                photon_index=photon_index,
                parameter_status=None,
                normalization_at_lower_bound=False,
                diagnostics=(f"gaussian_profile_failed:{type(exc).__name__}:{exc}",),
                construction="gaussian_net_rate_gls_profile",
                calibration_status="conditional_model",
                profile_status="failed",
                energy_band_keV=(emin, emax),
                rate_unit="ct / s",
                confidence_level=float(NormalDist().cdf(math.sqrt(delta_stat))),
                confidence_convention="one_sided_gaussian_equivalent",
            )
    try:
        # ``Fit.error`` can use a loaded XSPEC chain instead of a fresh
        # profile search.  Survey upper limits are Gaussian chi-square
        # profiles and must not inherit a chain left by an earlier fit in the
        # same process, so clear that optional global state before preparing
        # the spectrum.  ``AllChains`` is absent in lightweight test shims.
        chains = getattr(xspec, "AllChains", None)
        clear_chains = getattr(chains, "clear", None)
        if callable(clear_chains):
            clear_chains()
        prepared = PreparedSpectrum(
            instrument="BATSurvey",
            obsid=None,
            module=None,
            detector="BAT",
            source_id=None,
            source_pha=pha,
            grouped_pha=pha,
            background_pha=None,
            arf=response if _response_is_arf(response) else ancillary,
            rmf=response if not _response_is_arf(response) else ancillary,
            group_min=1,
            energy_range_keV=(emin, emax),
        )
        fit_prepared(
            prepared,
            outdir=output,
            emin=emin,
            emax=emax,
            model_name="powerlaw",
            frozen_parameters={"powerlaw.PhoIndex": photon_index},
            stat_method="chi",
            calculate_errors=False,
            plot_formats=(),
        )
        parameter = xspec.AllModels(1)(2)
        best_norm = float(parameter.values[0])
        fit_statistic = float(xspec.Fit.statistic)
        result = UpperLimit(2).error(
            deltas={f"delta_stat_{delta_stat:g}": delta_stat}
        )
        point = next(iter(result.limits.values()))
        upper_norm = point.upper
        status_code = point.status
        normalization_at_lower_bound = bool(np.isclose(best_norm, 0.0, atol=1e-14, rtol=0.0))
        hard_max = float(parameter.values[5]) if len(parameter.values) >= 6 else None
        status_text = str(status_code or "").strip().upper()
        if not np.isfinite(best_norm) or best_norm < 0:
            diagnostics.append("best_normalization_invalid")
        elif normalization_at_lower_bound:
            # A one-sided profile remains meaningful when the best fit is at
            # the imposed zero boundary.  Record that boundary explicitly;
            # callers must not reinterpret it as a symmetric two-sided error.
            pass
        if upper_norm is None or not np.isfinite(upper_norm) or upper_norm <= 0:
            diagnostics.append("upper_profile_did_not_cross")
        if hard_max is not None and upper_norm is not None and np.isclose(
            upper_norm,
            hard_max,
            rtol=0,
            atol=max(abs(hard_max) * 1e-8, 1e-12),
        ):
            diagnostics.append("upper_profile_reached_hard_bound")

        # XSPEC's two-sided ``error`` command reports a failed negative
        # search when a non-negative normalization is pegged at zero.  That
        # status is expected for a one-sided upper limit, but it cannot be
        # accepted on its own.  Re-evaluate the model at the returned upper
        # point and require the actual statistic increase to equal the
        # requested delta.  This also protects against a finite value returned
        # by an incomplete search or a stale error result.
        if (
            upper_norm is not None
            and np.isfinite(upper_norm)
            and upper_norm > max(best_norm, 0.0)
            and np.isfinite(fit_statistic)
            and status_text
            and len(status_text) == 9
        ):
            saved_values = list(parameter.values)
            saved_frozen = bool(parameter.frozen)
            try:
                parameter.values = float(upper_norm)
                profile_delta_stat = float(xspec.Fit.statistic) - float(fit_statistic)
            except Exception as exc:
                diagnostics.append(f"upper_profile_recheck_failed:{type(exc).__name__}:{exc}")
            finally:
                try:
                    parameter.values = saved_values
                    parameter.frozen = saved_frozen
                except Exception:
                    pass
            if profile_delta_stat is not None and (
                not np.isfinite(profile_delta_stat)
                or abs(profile_delta_stat - delta_stat) > profile_delta_tolerance
            ):
                diagnostics.append(
                    "upper_profile_delta_stat_mismatch:"
                    f"{profile_delta_stat:g}!={delta_stat:g}"
                )

        fatal_status_indices = {0, 1, 2, 4, 8}
        fatal_status = any(
            index < len(status_text) and status_text[index] == "T"
            for index in fatal_status_indices
        )
        status_clean = status_text == "FFFFFFFFF"
        boundary_one_sided = normalization_at_lower_bound and not fatal_status
        if not status_clean and not boundary_one_sided:
            diagnostics.append(f"upper_profile_xspec_status:{status_code}")
        if (
            boundary_one_sided
            and profile_delta_stat is None
            and upper_norm is not None
        ):
            diagnostics.append("upper_profile_delta_stat_unchecked")
        flux = None
        if not diagnostics:
            parameter.values = float(upper_norm)
            parameter.frozen = True
            xspec.AllModels.calcFlux(
                f"{emin} {emax}"
            )
            flux = float(xspec.AllData(1).flux[0])
            if not np.isfinite(flux) or flux <= 0:
                diagnostics.append("upper_flux_invalid")
                flux = None
        return SurveyUpperLimitResult(
            status="upper_limit_ready" if flux is not None else "failed",
            flux_erg_cm2_s=flux,
            upper_norm=upper_norm,
            best_norm=best_norm,
            fit_statistic=fit_statistic,
            delta_stat=delta_stat,
            photon_index=photon_index,
            parameter_status=status_code,
            normalization_at_lower_bound=normalization_at_lower_bound,
            profile_delta_stat=profile_delta_stat,
            profile_delta_tolerance=profile_delta_tolerance,
            diagnostics=tuple(diagnostics),
            construction="xspec_profile",
            calibration_status="conditional_model",
            profile_status="converged" if flux is not None else "failed",
            energy_band_keV=(emin, emax),
            rate_unit="ct / s",
            confidence_level=float(NormalDist().cdf(math.sqrt(delta_stat))),
            confidence_convention="one_sided_gaussian_equivalent",
        )
    except Exception as exc:  # pragma: no cover - requires a real XSPEC session
        return SurveyUpperLimitResult(
            status="failed",
            flux_erg_cm2_s=None,
            upper_norm=None,
            best_norm=None,
            fit_statistic=None,
            delta_stat=delta_stat,
            photon_index=photon_index,
            parameter_status=None,
            normalization_at_lower_bound=False,
            profile_delta_stat=None,
            profile_delta_tolerance=profile_delta_tolerance,
            diagnostics=(f"profile_failed:{type(exc).__name__}:{exc}",),
            construction="xspec_profile",
            calibration_status="conditional_model",
            profile_status="failed",
            energy_band_keV=(emin, emax),
            rate_unit="ct / s",
            confidence_level=float(NormalDist().cdf(math.sqrt(delta_stat))),
            confidence_convention="one_sided_gaussian_equivalent",
        )
    finally:
        try:
            xspec.AllData.clear()
            xspec.AllModels.clear()
        except Exception:
            pass


def validate_observation_directory(path: str | Path) -> tuple[bool, tuple[str, ...]]:
    """Check BAT survey DPH/HK and auxiliary files needed by BatAnalysis."""
    path = Path(path).expanduser().resolve()
    diagnostics: list[str] = []
    fits_suffixes = (
        ".fits", ".fit", ".dph", ".hk", ".mkf", ".gti", ".sao",
        ".sat", ".orb", ".att",
    )

    def looks_like_fits(file_path: Path) -> bool:
        # Swift distribution files are commonly gzip-compressed (for
        # example ``*.dph.gz`` and ``*.hk.gz``).  ``Path.suffix`` alone sees
        # only ``.gz`` and would incorrectly reject otherwise valid raw
        # observations.
        name = file_path.name.lower()
        return any(
            name.endswith(suffix) or name.endswith(suffix + ".gz")
            for suffix in fits_suffixes
        )

    for relative in ("bat/survey", "bat/hk", "auxil"):
        if not (path / relative).is_dir():
            diagnostics.append(f"missing_directory:{relative}")
    checks = {
        "bat_survey_dph": path / "bat" / "survey",
        "bat_hk": path / "bat" / "hk",
        "auxil": path / "auxil",
    }
    for label, directory in checks.items():
        files = [
            item for item in directory.iterdir()
            if item.is_file() and item.stat().st_size > 0
        ] if directory.is_dir() else []
        if directory.is_dir() and not files:
            diagnostics.append(f"missing_files:{label}")
        if directory.is_dir():
            fits_candidates = [item for item in files if looks_like_fits(item)]
            if fits_candidates:
                valid_fits = False
                for candidate in fits_candidates:
                    try:
                        with fits.open(candidate, memmap=False) as hdus:
                            if len(hdus) > 0:
                                valid_fits = True
                                break
                    except (OSError, ValueError):
                        continue
                if not valid_fits:
                    diagnostics.append(f"invalid_fits:{label}")
            elif files:
                diagnostics.append(f"missing_fits:{label}")
    return not diagnostics, tuple(diagnostics)


def _has_product(path: Path) -> bool:
    """Return whether a path is an existing survey/mosaic product.

    A caller may point to one FITS/CSV/CAT/PHA file or to a BatAnalysis
    result directory.  Completion markers and pickle files alone are not
    accepted as scientific products.
    """
    if path.is_file():
        return _is_readable_survey_product(path)
    if not path.is_dir():
        return False
    return any(
        item.is_file() and _is_readable_survey_product(item)
        for item in path.rglob("*")
    )


def _is_readable_survey_product(path: Path) -> bool:
    """Check a candidate product's actual container before reusing it.

    A completion marker or a stale pickle is not enough to claim a survey
    result.  FITS/CAT files must open successfully and CAT tables must carry a
    rate-like column; text products need at least one non-comment byte.  PHA
    response completeness is intentionally left to ``validate_survey_pha``.
    """
    if not _has_suffix(path, _SURVEY_PRODUCT_SUFFIXES):
        return False
    name = path.name.lower()
    if name.endswith((".fits", ".fits.gz", ".fit", ".fit.gz", ".cat", ".cat.gz", ".pha", ".pha.gz")):
        try:
            with fits.open(path, memmap=False) as hdus:
                if not hdus:
                    return False
                if name.endswith((".cat", ".cat.gz")):
                    for hdu in hdus:
                        data = getattr(hdu, "data", None)
                        names = {
                            str(item).upper()
                            for item in (getattr(data, "names", ()) or ())
                        }
                        if names & set(_RATE_NAMES):
                            return True
                    return False
                if name.endswith((".pha", ".pha.gz")):
                    return any(getattr(hdu, "data", None) is not None for hdu in hdus[1:])
                # Generic FITS files in a BatAnalysis directory include
                # diagnostics and detector images (for example
                # ``stats_obs.fits`` and ``*_chi.fits``).  Their existence
                # must not make an incomplete survey cache look reusable.
                # Accept only a table carrying a source-rate-like column;
                # source catalogues with a ``RATE``/``CENT_RATE`` field are
                # the actual survey products consumed by the light-curve
                # stage.
                for hdu in hdus:
                    data = getattr(hdu, "data", None)
                    names = {
                        str(item).upper()
                        for item in (getattr(data, "names", ()) or ())
                    }
                    if names & set(_RATE_NAMES):
                        return True
                return False
        except (OSError, ValueError, IndexError, TypeError):
            return False
    try:
        if name.endswith((".dat", ".dat.gz", ".txt", ".txt.gz")):
            numeric = _read_numeric_dat_rates(path)
            # Arbitrary BatAnalysis diagnostics such as ``stats_obs.dat``
            # are not survey products.  Only accept a headerless table when
            # the positional reader recognizes the 14/15-column rate
            # contract and at least one finite rate is available.
            return numeric is not None and any(point.rate is not None for point in numeric)
        if name.endswith((".csv", ".csv.gz")):
            rows = _read_csv_rates(path, None)
            return any(point.rate is not None for point in rows)
        if name.endswith((".qdp", ".qdp.gz")):
            return bool(_read_qdp_rates(path))
        return False
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def _survey_product_diagnostics(path: Path) -> tuple[str, ...]:
    """Read per-pointing BatAnalysis status files without trusting markers."""
    if not path.is_dir():
        return ()
    diagnostics: list[str] = []
    for status_file in sorted(path.rglob("*_status.txt")):
        try:
            text = status_file.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            diagnostics.append(f"status_read_failed:{status_file.name}:{exc}")
            continue
        match = re.search(r"status\s*=\s*[\"']?([^;\"']+)", text, flags=re.IGNORECASE)
        status = match.group(1).strip().upper() if match else "UNKNOWN"
        if status not in {"SUCCESS", "OK", "TRUE", "1"}:
            reason = re.search(r"reason\s*=\s*[\"']?([^;\"']+)", text, flags=re.IGNORECASE)
            detail = reason.group(1).strip() if reason else status
            diagnostics.append(f"pointing_status_failure:{status_file.name}:{detail}")
    return tuple(diagnostics)


def safe_extract_archive(archive: str | Path, destination: str | Path) -> tuple[Path, ...]:
    """Extract tar/zip only after rejecting traversal and escaping links."""
    import tarfile
    import zipfile

    archive = Path(archive).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            infos = handle.infolist()
            for info in infos:
                member = info.filename
                target = (destination / member).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"archive member escapes destination: {member}")
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"archive symlink is not allowed: {member}")
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(handle.read(info))
                extracted.append(target)
    else:
        with tarfile.open(archive, "r:*") as handle:
            archive_members = handle.getmembers()
            for member in archive_members:
                if not (destination / member.name).resolve().is_relative_to(destination):
                    raise ValueError(f"archive member escapes destination: {member.name}")
                if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                    raise ValueError(f"archive special file is not allowed: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"archive links are not allowed: {member.name}")
            try:
                handle.extractall(destination, filter="data")
            except TypeError:  # Python 3.11 tarfile API
                handle.extractall(destination)
            extracted.extend(
                (destination / member.name).resolve()
                for member in archive_members
                if member.isfile() and (destination / member.name).resolve().is_file()
            )
    return tuple(extracted)


def _default_catalog_path() -> Path:
    try:
        import batanalysis as ba
        return Path(ba.__file__).resolve().parent / "data" / "survey6b_2.cat"
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("BatAnalysis is required to build a BAT survey source catalog") from exc


def build_source_catalog(
    path: str | Path,
    *,
    source_name: str,
    coordinate: SkyCoord,
    base_catalog: str | Path | None = None,
) -> Path:
    """Append one unambiguous source to a BatAnalysis survey catalog."""
    if not isinstance(coordinate, SkyCoord):
        raise TypeError("coordinate must be an astropy SkyCoord")
    path = Path(path).expanduser().resolve()
    base = Table.read(
        Path(base_catalog).expanduser().resolve() if base_catalog else _default_catalog_path(),
        format="fits",
    )
    required_columns = {"NAME", "RA_OBJ", "DEC_OBJ"}
    missing = required_columns.difference(base.colnames)
    if missing:
        raise ValueError(
            "BAT source catalog is missing required columns: "
            + ", ".join(sorted(missing))
        )
    names = [str(_decode(value)).strip() for value in base["NAME"]]
    existing = {
        name: SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        for name, ra, dec in zip(names, base["RA_OBJ"], base["DEC_OBJ"])
    }
    alias_name = next(
        (
            name
            for name in existing
            if _source_name_matches(name, source_name)
        ),
        None,
    )
    if alias_name is not None:
        separation = coordinate.separation(existing[alias_name]).to_value(u.arcmin)
        if separation > 5.0:
            raise ValueError(
                f"source name {source_name!r} (catalog alias {alias_name!r}) "
                f"exists {separation:.2f} arcmin from supplied coordinate"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        base.write(path, format="fits", overwrite=True)
        return path
    max_catnum = (
        int(np.max(base["CATNUM"]))
        if len(base) and "CATNUM" in base.colnames
        else 32000
    )
    row = {name: None for name in base.colnames}
    values = {
        "CATNUM": max_catnum + 1,
        "NAME": source_name,
        "RA_OBJ": float(coordinate.icrs.ra.deg),
        "DEC_OBJ": float(coordinate.icrs.dec.deg),
        "GLON_OBJ": float(coordinate.galactic.l.deg),
        "GLAT_OBJ": float(coordinate.galactic.b.deg),
        "ALWAYS_CLEAN": True,
    }
    row.update({name: value for name, value in values.items() if name in base.colnames})
    merged = vstack(
        [base, Table(rows=[row], names=base.colnames)],
        metadata_conflicts="silent",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.write(path, format="fits", overwrite=True)
    return path


def _call_supported(function: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return function(**kwargs)
    return function(
        **{key: value for key, value in kwargs.items() if key in signature.parameters}
    )


def _download_value_success(value: Any) -> bool:
    """Interpret the result shape used by BatAnalysis/swifttools downloaders."""
    if isinstance(value, Mapping):
        if "success" in value:
            raw = value.get("success")
            if isinstance(raw, str):
                return raw.strip().upper() in {"1", "TRUE", "YES", "OK", "SUCCESS", "COMPLETED"}
            return bool(raw)
        if "returncode" in value:
            return int(value.get("returncode") or 0) == 0
        if "status" in value:
            return str(value.get("status", "")).upper() in {
                "OK", "SUCCESS", "COMPLETED", "TRUE",
            }
    if isinstance(value, bool):
        return value
    attr = getattr(value, "success", None)
    if attr is not None:
        return bool(attr)
    return getattr(value, "returncode", 1) == 0


def _json_safe_external(value: Any) -> Any:
    """Summarize downloader return objects without serializing live tables.

    ``batanalysis.download_swiftdata`` returns ``Swift_Data`` objects under a
    ``data`` key.  Those objects contain network/client state and are not JSON
    serializable; the manifest only needs paths and status diagnostics.
    """
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe_external(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe_external(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_external(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe_external(value.item())
    summary: dict[str, Any] = {"type": f"{type(value).__module__}.{type(value).__name__}"}
    for name in ("obsid", "obs_id", "outdir", "obsoutdir", "localpath", "quicklook"):
        attr = getattr(value, name, None)
        if attr is not None and isinstance(attr, (str, int, float, bool, Path)):
            summary[name] = str(attr) if isinstance(attr, Path) else attr
    status = getattr(value, "status", None)
    status_value = getattr(status, "status", status)
    if status_value is not None and isinstance(status_value, (str, int, float, bool)):
        summary["status"] = status_value
    return summary


@contextmanager
def _math_thread_limit(threads: int = 1):
    """Temporarily cap common BLAS/OpenMP thread pools for backend work."""
    import os

    value = str(max(1, int(threads)))
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = value
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


@contextmanager
def _temporary_environment(values: Mapping[str, str]):
    """Apply a stage-local environment to in-process HEASoft adapters."""
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update({str(name): str(value) for name, value in values.items()})
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


@contextmanager
def _temporary_cwd(directory: str | Path):
    """Run a third-party adapter with its implicit files in a private tree."""
    previous = Path.cwd()
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(target)
        yield target
    finally:
        os.chdir(previous)


@contextmanager
def _temporary_sys_path(entries: Iterable[str]):
    """Temporarily expose HEASoft's Python modules to the current process."""
    previous = list(sys.path)
    try:
        for entry in reversed(tuple(str(item) for item in entries if item)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        yield
    finally:
        sys.path[:] = previous


def _query_tap_with_timeout(client: Any, query: str, timeout_s: float) -> Any:
    """Run an astroquery TAP request while honoring a finite timeout.

    Astroquery versions expose the timeout either as a keyword or as a
    ``TIMEOUT`` attribute on the TAP client.  The adapter handles both forms
    and restores the client's previous value after the request.
    """
    targets = [client]
    for name in ("_tap", "_tap_plus", "tap", "tap_plus"):
        target = getattr(client, name, None)
        if target is not None:
            targets.append(target)
    saved: list[tuple[Any, str, Any]] = []
    for target in targets:
        for name in ("TIMEOUT", "timeout"):
            if not hasattr(target, name):
                continue
            try:
                old = getattr(target, name)
                setattr(target, name, float(timeout_s))
                saved.append((target, name, old))
            except Exception:
                pass
    try:
        return _call_supported(
            client.query_tap,
            query=query,
            timeout=float(timeout_s),
            timeout_s=float(timeout_s),
        )
    finally:
        for target, name, value in reversed(saved):
            try:
                setattr(target, name, value)
            except Exception:
                pass


class BatAnalysisSurveyBackend:
    """Isolated BatAnalysis 2.x adapter for one or more survey observations."""

    def __init__(self, *, module: Any | None = None):
        if module is None:
            try:
                import batanalysis as module
            except ImportError as exc:
                raise RuntimeError("BatAnalysis is required for BAT survey processing") from exc
        self.module = module
        self.objects: dict[str, Any] = {}

    @staticmethod
    def _staged_observation_tree(
        obsid: str,
        obs_dir: Path,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Create a writable workspace shell whose data directories are links.

        BatAnalysis creates ``.local_pfile`` below the directory passed as
        ``obs_dir`` even when the actual survey ``outdir`` is elsewhere.  A
        raw mirror is an input and must remain read-only, so the adapter gives
        BatAnalysis a real workspace directory with read-only ``bat`` and
        ``auxil`` links into the source observation.  The returned pair is
        ``(staged_observation, staged_parent)`` for the constructor and keeps
        all generated parameter files outside the raw tree.
        """
        source = Path(obs_dir).expanduser().resolve()
        workspace = Path(output_dir).expanduser().resolve().parents[1] / "staged_observations"
        staged = workspace / str(obsid)
        staged.mkdir(parents=True, exist_ok=True)
        for name in ("bat", "auxil"):
            source_child = source / name
            target_child = staged / name
            if target_child.is_symlink():
                if target_child.resolve() != source_child:
                    raise ValueError(
                        f"staged {name} link points to {target_child.resolve()}, "
                        f"expected {source_child}"
                    )
            elif target_child.exists():
                raise ValueError(f"staged observation path is not a link: {target_child}")
            else:
                target_child.symlink_to(source_child, target_is_directory=True)
        return staged, workspace

    @staticmethod
    def _staged_result_cache(
        obsid: str,
        result_dir: Path,
        output_dir: Path | None,
    ) -> Path:
        """Copy a read-only BatAnalysis cache into the writable run workspace.

        ``BatSurvey(..., load_dir=...)`` creates ``.local_pfile`` below
        ``load_dir`` even for a completed pickle.  Pointing it directly at a
        reference ``*_surveyresult`` directory would therefore mutate the
        user's input tree.  The cache is copied (rather than symlinked) so
        every backend write remains inside the independent output workspace.
        """
        source = Path(result_dir).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"survey result directory does not exist: {source}")
        if output_dir is None:
            # Direct adapter callers still get read-only protection.  The
            # temporary directory is retained for the lifetime of the
            # returned BatSurvey object through its copied paths.
            workspace = Path(tempfile.mkdtemp(prefix=f"jinwu-bat-cache-{obsid}-"))
        else:
            workspace = Path(output_dir).expanduser().resolve().parents[1] / "staged_results"
            workspace.mkdir(parents=True, exist_ok=True)
        staged = workspace / str(obsid)
        if staged.is_symlink():
            staged.unlink()
        elif staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(source, staged, symlinks=False)
        return staged

    def load_observation(
        self,
        *,
        obsid: str,
        result_dir: Path,
        source_name: str | None = None,
        obs_dir: Path | None = None,
        data_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Load an existing BatAnalysis result without recalculating survey data."""
        result_dir = Path(result_dir).expanduser().resolve()
        if not result_dir.is_dir():
            raise FileNotFoundError(f"survey result directory does not exist: {result_dir}")
        staged_result_dir = self._staged_result_cache(obsid, result_dir, output_dir)
        # BatAnalysis keeps the result cache separate from the raw observation
        # tree, but its ``BatObservation`` base class still requires an
        # ``obs_dir`` parent containing ``<obsid>/``.  The old adapter passed
        # only ``load_dir`` and therefore failed whenever a caller supplied a
        # perfectly valid result cache alongside raw data.  Resolve the
        # parent explicitly, while retaining result-only parsing as a valid
        # fallback (the constructor will raise a clear error if raw data are
        # genuinely unavailable).
        obsid_text = str(obsid)
        candidates: list[Path] = []
        for value in (obs_dir, data_root, result_dir.parent):
            if value is None:
                continue
            candidate = Path(value).expanduser().resolve()
            if candidate.name == obsid_text and candidate.is_dir():
                candidates.append(candidate.parent)
            else:
                candidates.append(candidate)
        resolved_obs_parent = next(
            (candidate for candidate in candidates if (candidate / obsid_text).is_dir()),
            None,
        )
        load_kwargs: dict[str, Any] = {
            "obs_id": obsid_text,
            "obsid": obsid_text,
            "load_dir": str(staged_result_dir),
            "recalc": False,
            "verbose": False,
        }
        if resolved_obs_parent is not None:
            if output_dir is not None:
                staged_obs_dir, staged_parent = self._staged_observation_tree(
                    obsid_text,
                    resolved_obs_parent / obsid_text,
                    output_dir,
                )
                del staged_obs_dir
                load_kwargs["obs_dir"] = str(staged_parent)
            else:
                load_kwargs["obs_dir"] = str(resolved_obs_parent)
        with _temporary_cwd(staged_result_dir):
            observation = _call_supported(
                self.module.BatSurvey,
                **load_kwargs,
            )
        self.objects[str(obsid)] = observation
        if source_name:
            with _temporary_cwd(staged_result_dir):
                try:
                    merge_pointings = getattr(observation, "merge_pointings", None)
                    if callable(merge_pointings):
                        merge_pointings(verbose=False)
                except Exception:
                    # A saved result may already contain merged products; failure
                    # here should not make the source PHA unusable.
                    pass
                load_source_information = getattr(observation, "load_source_information", None)
                if callable(load_source_information):
                    load_source_information([source_name])
        return {
            "obsid": str(obsid),
            "result_dir": str(result_dir),
            "staged_result_dir": str(staged_result_dir),
            "loaded_existing": True,
            "pointing_ids": [
                str(item) for item in getattr(observation, "pointing_ids", ())
            ],
        }

    def process_observation(
        self,
        *,
        obsid: str,
        obs_dir: Path,
        output_dir: Path,
        source_name: str,
        catalog_path: Path,
        survey: SwiftBATSurveyConfig,
        recalc: bool = True,
        task_timeout_s: float | None = None,
        processes: int = 1,
        internal_threads: int = 1,
    ) -> dict[str, Any]:
        valid, diagnostics = validate_observation_directory(obs_dir)
        if not valid:
            raise ValueError(f"observation {obsid} is incomplete: {', '.join(diagnostics)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # BatAnalysis' ``BatObservation`` interprets ``obs_dir`` as the
        # parent containing the numbered OBSID directory, whereas batsurvey's
        # ``indir`` is the concrete observation tree.
        staged_obs_dir, staged_parent = self._staged_observation_tree(
            str(obsid), obs_dir, output_dir
        )
        with _temporary_cwd(output_dir):
            observation = _call_supported(
                self.module.BatSurvey,
                obs_id=str(obsid),
                obsid=str(obsid),
                obs_dir=str(staged_parent),
                input_dict={
                    "indir": str(staged_obs_dir),
                    "outdir": str(output_dir),
                    "incatalog": str(catalog_path),
                    "detthresh": str(survey.detthresh),
                    "detthresh2": str(survey.detthresh2),
                    "pcodethresh": str(survey.min_pcode),
                },
                recalc=recalc,
                verbose=False,
                timeout=task_timeout_s,
                timeout_s=task_timeout_s,
            )
            self.objects[str(obsid)] = observation
            observation.merge_pointings(verbose=False)
            observation.load_source_information([source_name])
            observation.calculate_pha(
                id_list=[source_name],
                output_dir=str(output_dir / "PHA_files"),
                calc_upper_lim=False,
                verbose=False,
                clean_dir=False,
            )
        pha_paths = [Path(item).resolve() for item in getattr(observation, "pha_file_names_list", ())]
        if not pha_paths:
            pha_paths = sorted((output_dir / "PHA_files").glob("*.pha"))
        response_paths: list[str] = []
        response_diagnostics: list[str] = []
        calc_response = getattr(self.module, "calc_response", None)
        if calc_response is not None:
            for pha in pha_paths:
                try:
                    with _temporary_cwd(output_dir):
                        result = calc_response(str(pha))
                    # ``batdrmgen`` returns a result object while writing the
                    # response next to the PHA.  Record the actual file when
                    # it exists; only fall back to a returned path-like value
                    # for older adapters that do not use the sibling naming
                    # convention.
                    sibling = pha.with_suffix(".rsp")
                    if sibling.is_file():
                        response_paths.append(str(sibling))
                    elif isinstance(result, (str, Path)):
                        response_paths.append(str(Path(result).expanduser().resolve()))
                    elif result is not None:
                        response_diagnostics.append(
                            f"response_result_without_file:{pha}:{type(result).__name__}"
                        )
                except Exception as exc:
                    message = f"response_generation_failed:{pha}:{type(exc).__name__}:{exc}"
                    response_diagnostics.append(message)
                    logger.warning("response generation failed for %s: %s", pha, exc)
        return {
            "obsid": str(obsid),
            "result_dir": str(output_dir),
            "pha_files": [str(path) for path in pha_paths],
            "response_files": response_paths,
            "response_diagnostics": response_diagnostics,
            "pointing_ids": [str(item) for item in getattr(observation, "pointing_ids", ())],
            "processes": max(1, int(processes)),
            "internal_threads": max(1, int(internal_threads)),
        }

    def process_mosaic(
        self,
        *,
        windows: Sequence[tuple[str, str]],
        output_dir: Path,
        source_name: str,
        catalog_path: Path,
        min_pcode: float,
        gti_intervals: Mapping[str | None, Iterable[tuple[float, float]]] | None = None,
        overlap_selection: bool = True,
        detection_threshold: float = 3.0,
        processes: int = 1,
        internal_threads: int = 1,
        task_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Create mosaics for UTC windows from objects made in this worker.

        Older BatAnalysis versions select inventory rows by TSTART only.  The
        temporary replacement below uses true interval overlap and is restored
        in finally; it is safe because this method runs in an isolated worker.
        """
        if not windows:
            return {
                "status": "no_windows",
                "products": [],
                "mosaic_measurements": [],
                "overlap_selection": bool(overlap_selection),
                "detection_threshold": float(detection_threshold),
                "processes": max(1, int(processes)),
                "internal_threads": max(1, int(internal_threads)),
            }
        objects = list(self.objects.values())
        if not objects:
            return {
                "status": "no_valid_exposure",
                "products": [],
                "mosaic_measurements": [],
                "overlap_selection": bool(overlap_selection),
                "detection_threshold": float(detection_threshold),
                "processes": max(1, int(processes)),
                "internal_threads": max(1, int(internal_threads)),
            }
        # Prefer an explicitly injected module in tests or downstream
        # adapters.  Importing the installed package first would silently
        # bypass that adapter whenever BatAnalysis happened to be present in
        # the environment.
        mosaic_module = getattr(self.module, "mosaic", None)
        if mosaic_module is None:
            try:
                import batanalysis.mosaic as mosaic_module
            except ImportError:
                mosaic_module = None
        create_mosaics = getattr(mosaic_module, "create_mosaics", None)
        merge_outventory = getattr(mosaic_module, "merge_outventory", None)
        if create_mosaics is None or merge_outventory is None:
            raise RuntimeError("installed BatAnalysis has no mosaic.create_mosaics API")
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory = _call_supported(
            merge_outventory,
            survey_list=objects,
            surveys=objects,
            savedir=output_dir,
        )
        source_pcodes = self._source_pcodes(objects, source_name)
        members_by_window = self._inventory_window_members(
            Path(inventory),
            windows,
            min_pcode=min_pcode,
            source_pcodes=source_pcodes,
            gti_intervals=gti_intervals if overlap_selection else None,
        )
        if not any(item["members"] for item in members_by_window):
            return {
                "status": "no_valid_exposure",
                "products": [],
                "mosaic_measurements": [],
                "windows": members_by_window,
                "shared_members": [],
                "overlap_selection": bool(overlap_selection),
                "detection_threshold": float(detection_threshold),
                "processes": max(1, int(processes)),
                "internal_threads": max(1, int(internal_threads)),
            }
        mosaic_impl = mosaic_module
        skyview_impl = getattr(self.module, "bat_skyview", None)
        if skyview_impl is None:
            try:
                import batanalysis.bat_skyview as skyview_impl
            except ImportError:
                skyview_impl = None
        original_selector = getattr(mosaic_impl, "select_outventory", None)
        original_threshold = getattr(mosaic_impl, "_pcodethresh", None)
        original_sky_threshold = (
            getattr(skyview_impl, "_pcodethresh", None)
            if skyview_impl is not None
            else None
        )

        def overlap_selector(outventory_file, start_met, end_met):
            def _first_scalar(value: Any) -> float:
                array = np.asarray(value, dtype=float).reshape(-1)
                if array.size != 1 or not np.isfinite(array[0]):
                    raise ValueError("mosaic time selector requires one finite MET edge")
                return float(array[0])

            start_met = _first_scalar(start_met)
            end_met = _first_scalar(end_met)
            selected_path = Path(outventory_file).with_name(
                Path(outventory_file).stem + "_sel.fits"
            )
            with fits.open(outventory_file, memmap=False) as source:
                table = source[1].data
                names = {str(name).upper(): name for name in (table.names or ())}
                if "TSTOP" in names:
                    tstop = np.asarray(table[names["TSTOP"]], dtype=float)
                elif "EXPOSURE" in names:
                    tstop = np.asarray(table[names["TSTART"]], dtype=float) + np.asarray(
                        table[names["EXPOSURE"]], dtype=float
                    )
                else:
                    # Without a stop time or live-time column there is
                    # no defensible complete-pointing interval.
                    tstop = np.full(len(table), np.nan, dtype=float)
                mask = (
                    (np.asarray(table[names["TSTART"]], dtype=float) < float(end_met))
                    & (tstop > float(start_met))
                )
                if "EXPOSURE" in names:
                    exposure = np.asarray(table[names["EXPOSURE"]], dtype=float)
                    mask &= np.isfinite(exposure) & (exposure > 0)
                else:
                    mask &= False
                if "IMAGE_STATUS" in names:
                    mask &= _accepted_status_mask(table[names["IMAGE_STATUS"]])
                pcode_name = next(
                    (name for name in ("PCODEFR", "PCODEAPP", "PCODE") if name in names),
                    None,
                )
                if pcode_name is not None:
                    pcode = np.asarray(table[names[pcode_name]], dtype=float)
                    mask &= np.isfinite(pcode) & (pcode > float(min_pcode))
                elif source_pcodes:
                    identifiers = []
                    for row in table:
                        identifier = None
                        for name in ("POINTING_ID", "POINTING", "IMAGE_ID", "OBSID", "OBS_ID", "ID"):
                            if name in names:
                                value = str(_decode(row[names[name]])).strip()
                                if value:
                                    identifier = value
                                    break
                        identifiers.append(identifier)
                    mask &= np.asarray(
                        [
                            identifier is not None
                            and np.isfinite(source_pcodes.get(identifier, np.nan))
                            and source_pcodes[identifier] > float(min_pcode)
                            for identifier in identifiers
                        ],
                        dtype=bool,
                    )
                elif float(min_pcode) > 0:
                    # The mosaic threshold is source-specific.  If the
                    # inventory carries no pcode and no per-source CAT
                    # lookup was available, retaining the row would turn
                    # missing coverage metadata into an apparent valid
                    # exposure.
                    mask &= False
                if overlap_selection and gti_intervals:
                    # Keep only rows whose pointing-level GTI intersects this
                    # mosaic window.  The inventory row's complete exposure
                    # remains unchanged; GTI is a coverage gate rather than
                    # a scaling factor.
                    normalized_gti = {
                        (None if key is None else str(key)): tuple(value)
                        for key, value in gti_intervals.items()
                    }
                    identifiers = []
                    for row in table:
                        identifier = None
                        for name in (
                            "POINTING_ID", "POINTING", "IMAGE_ID", "OBSID",
                            "OBS_ID", "ID",
                        ):
                            if name in names:
                                value = str(_decode(row[names[name]])).strip()
                                if value:
                                    identifier = value
                                    break
                        identifiers.append(identifier)
                    for index, identifier in enumerate(identifiers):
                        candidates = [identifier]
                        if identifier and not identifier.lower().startswith("point_"):
                            candidates.append(f"point_{identifier}")
                        candidates.append(None)
                        intervals = next(
                            (
                                normalized_gti[key]
                                for key in candidates
                                if key in normalized_gti
                            ),
                            (),
                        )
                        if not intervals or gti_overlap_duration(
                            intervals, float(start_met), float(end_met)
                        ) <= 0:
                            mask[index] = False
                selected_hdu = fits.BinTableHDU(
                    data=table[mask],
                    header=source[1].header,
                    name=source[1].name,
                )
                fits.HDUList(
                    [fits.PrimaryHDU(header=source[0].header), selected_hdu]
                ).writeto(selected_path, overwrite=True)
            return selected_path

        try:
            if overlap_selection and original_selector is not None:
                mosaic_impl.select_outventory = overlap_selector
            if original_threshold is not None:
                mosaic_impl._pcodethresh = float(min_pcode)
            if skyview_impl is not None and original_sky_threshold is not None:
                skyview_impl._pcodethresh = float(min_pcode)
            if len(windows) == 1:
                time_bins: Any = Time([windows[0][0], windows[0][1]], scale="utc")
            else:
                time_bins = [
                    Time([[start], [stop]], scale="utc")
                    for start, stop in windows
                ]
            # BatAnalysis' ``create_mosaics`` expects the grouped outventory
            # files produced by ``group_outventory`` to exist beforehand.
            # Calling only ``create_mosaics`` (as older adapters did) leaves
            # the grouped path absent and fails before any image is built.
            # Keep the grouping in this isolated output tree and use the
            # interval-aware selector above so a short window can include a
            # pointing that started before the window edge.
            group_outventory = getattr(mosaic_impl, "group_outventory", None)
            with _temporary_cwd(output_dir):
                if callable(group_outventory):
                    _call_supported(
                        group_outventory,
                        outventory_file=inventory,
                        custom_timebins=time_bins,
                        recalc=True,
                        mjd_savedir=True,
                        save_group_outventory=True,
                    )
                result = _call_supported(
                    create_mosaics,
                    outventory_file=inventory,
                    outventory=inventory,
                    inventory_file=inventory,
                    time_bins=time_bins,
                    survey_list=objects,
                    surveys=objects,
                    catalog_file=catalog_path,
                    total_mosaic_savedir=output_dir / "total_mosaic",
                    recalc=True,
                    verbose=False,
                    processes=int(processes),
                    nprocs=int(processes),
                    num_procs=int(processes),
                    detection_threshold=float(detection_threshold),
                    snr_threshold=float(detection_threshold),
                    timeout=task_timeout_s,
                    timeout_s=task_timeout_s,
                )
        finally:
            try:
                if original_selector is not None:
                    mosaic_impl.select_outventory = original_selector
                if original_threshold is not None:
                    mosaic_impl._pcodethresh = original_threshold
                if skyview_impl is not None and original_sky_threshold is not None:
                    skyview_impl._pcodethresh = original_sky_threshold
            except Exception:
                logger.exception("failed to restore BatAnalysis mosaic globals")
        result_metadata: dict[str, Any] = {}
        if isinstance(result, Mapping):
            result_metadata = dict(result)
            raw_mosaics = result.get("products") or result.get("mosaics") or ()
            mosaics = list(raw_mosaics) if isinstance(raw_mosaics, (list, tuple)) else [raw_mosaics]
        elif isinstance(result, tuple):
            mosaics = list(result[0] or ())
            if len(result) > 1 and result[1] is not None:
                mosaics.append(result[1])
        else:
            mosaics = list(result) if isinstance(result, (list, tuple)) else ([result] if result is not None else [])
        if not mosaics:
            return {
                "status": "no_valid_exposure",
                "products": [],
                "mosaic_measurements": [],
                "windows": members_by_window,
                "shared_members": self._shared_members(members_by_window),
                "overlap_selection": bool(overlap_selection),
                "detection_threshold": float(detection_threshold),
                "processes": max(1, int(processes)),
                "internal_threads": max(1, int(internal_threads)),
            }
        products = []
        for item in mosaics:
            result_dir = getattr(item, "result_dir", None)
            if result_dir is None and isinstance(item, Mapping):
                result_dir = item.get("result_dir") or item.get("path")
            if result_dir is None and isinstance(item, (str, Path)):
                result_dir = item
            if result_dir is not None:
                products.append(str(result_dir))
        measurements: list[dict[str, Any]] = []
        for product in products:
            measurements.extend(
                _mosaic_source_measurements(
                    product,
                    source_name=source_name,
                    detection_threshold=detection_threshold,
                )
            )
        return {
            **result_metadata,
            "status": "completed",
            "products": products,
            "mosaic_measurements": measurements,
            "windows": members_by_window,
            "shared_members": self._shared_members(members_by_window),
            "overlap_selection": bool(overlap_selection),
            "detection_threshold": float(detection_threshold),
            "processes": int(processes),
            "internal_threads": max(1, int(internal_threads)),
        }

    @staticmethod
    def _source_pcodes(
        objects: Sequence[Any],
        source_name: str,
    ) -> dict[str, float]:
        """Read source PCODEFR values from the per-pointing BAT CAT files."""
        result: dict[str, float] = {}
        for observation in objects:
            files = list(getattr(observation, "pointing_flux_files", ()) or ())
            pointings = list(getattr(observation, "pointing_ids", ()) or ())
            for index, path_value in enumerate(files):
                path = Path(path_value)
                if not path.is_file():
                    continue
                identifier = (
                    str(pointings[index])
                    if index < len(pointings)
                    else path.parent.name.rsplit("_", 1)[-1]
                )
                try:
                    with fits.open(path, memmap=False) as hdus:
                        data = hdus[1].data
                        names = {str(name).upper() for name in (data.names or ())}
                        pcode_name = next(
                            (name for name in ("PCODEFR", "PCODEAPP", "PCODE") if name in names),
                            None,
                        )
                        if pcode_name is None or "NAME" not in names:
                            continue
                        matches = []
                        for row in data:
                            candidate = str(_decode(row["NAME"])).strip()
                            if _source_name_matches(candidate, source_name):
                                matches.append(row)
                            else:
                                comparator = getattr(observation, "_compare_source_name", None)
                                try:
                                    if callable(comparator) and bool(comparator(source_name, candidate)):
                                        matches.append(row)
                                except Exception:
                                    pass
                        if matches:
                            value = _float(matches[0][pcode_name])
                            if value is not None:
                                result[identifier] = value
                                result.setdefault(f"point_{identifier}", value)
                except (OSError, ValueError, IndexError, KeyError, TypeError):
                    continue
        return result

    @staticmethod
    def _inventory_window_members(
        inventory: Path,
        windows: Sequence[tuple[str, str]],
        *,
        min_pcode: float,
        source_pcodes: Mapping[str, float] | None = None,
        gti_intervals: Mapping[str | None, Iterable[tuple[float, float]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Describe complete inventory rows overlapping each UTC window."""
        normalized_gti = (
            {
                (None if key is None else str(key)): tuple(value)
                for key, value in gti_intervals.items()
            }
            if gti_intervals is not None
            else None
        )
        with fits.open(inventory, memmap=False) as hdus:
            table = hdus[1].data
            names = {str(name).upper(): name for name in (table.names or ())}
            if "TSTART" not in names:
                return [
                    {
                        "requested_start": start,
                        "requested_stop": stop,
                        "requested_start_utc": _time_utc_iso(start),
                        "requested_stop_utc": _time_utc_iso(stop),
                        "members": [],
                        "full_exposure_s": 0.0,
                        "overlap_s": 0.0,
                    }
                    for start, stop in windows
                ]
            starts = np.asarray(table[names["TSTART"]], dtype=float)
            if "TSTOP" in names:
                stops = np.asarray(table[names["TSTOP"]], dtype=float)
            elif "EXPOSURE" in names:
                stops = starts + np.asarray(table[names["EXPOSURE"]], dtype=float)
            else:
                stops = np.full(starts.shape, np.nan, dtype=float)
            exposure_values = (
                np.asarray(table[names["EXPOSURE"]], dtype=float)
                if "EXPOSURE" in names
                else np.full(starts.shape, np.nan, dtype=float)
            )
            good = np.isfinite(starts) & np.isfinite(stops) & (stops > starts)
            good &= np.isfinite(exposure_values) & (exposure_values > 0)
            if "IMAGE_STATUS" in names:
                good &= _accepted_status_mask(table[names["IMAGE_STATUS"]])
            pcode_name = next(
                (name for name in ("PCODEFR", "PCODEAPP", "PCODE") if name in names),
                None,
            )
            if pcode_name is not None:
                pcode = np.asarray(table[names[pcode_name]], dtype=float)
                good &= np.isfinite(pcode) & (pcode > float(min_pcode))
            members: list[dict[str, Any]] = []
            for index, row in enumerate(table):
                if not good[index]:
                    continue
                identifier = None
                for name in ("POINTING_ID", "POINTING", "IMAGE_ID", "OBSID", "OBS_ID", "ID"):
                    if name in names:
                        identifier = str(_decode(row[names[name]])).strip()
                        if identifier:
                            break
                if pcode_name is None:
                    if source_pcodes is None:
                        if float(min_pcode) > 0:
                            continue
                        pcode = None
                    else:
                        pcode = source_pcodes.get(identifier or "", np.nan)
                        if not np.isfinite(pcode) or pcode <= float(min_pcode):
                            continue
                elif pcode_name is not None:
                    pcode = _float(row[names[pcode_name]])
                else:
                    pcode = None
                members.append(
                    {
                        "row_index": int(index),
                        "id": identifier or f"row-{index}",
                        "tstart_met": float(starts[index]),
                        "tstop_met": float(stops[index]),
                        # Inventory overlap is evaluated with the full
                        # TSTART/TSTOP interval, while the reported exposure
                        # remains the product's live-time value when present.
                        "exposure_s": float(exposure_values[index]),
                        "pcode": pcode,
                        "gti_key": (
                            next(
                                (
                                    candidate
                                    for candidate in (
                                        identifier,
                                        (
                                            f"point_{identifier}"
                                            if identifier
                                            and not identifier.lower().startswith("point_")
                                            else None
                                        ),
                                        None,
                                    )
                                    if normalized_gti is not None and candidate in normalized_gti
                                ),
                                None,
                            )
                            if normalized_gti is not None
                            else None
                        ),
                    }
                )
        output: list[dict[str, Any]] = []
        for start, stop in windows:
            start_met, stop_met = _time_value(start), _time_value(stop)
            selected = []
            for member in members:
                if not (
                    member["tstart_met"] < stop_met
                    and member["tstop_met"] > start_met
                ):
                    continue
                gti_overlap = None
                if normalized_gti is not None:
                    intervals = normalized_gti.get(member.get("gti_key"), ())
                    gti_overlap = gti_overlap_duration(
                        intervals, start_met, stop_met
                    ) if intervals else 0.0
                    if gti_overlap <= 0:
                        continue
                selected.append(
                    {
                        **member,
                        "gti_key": member.get("gti_key"),
                        "gti_overlap_s": gti_overlap,
                        "overlap_s": max(
                            0.0,
                            min(member["tstop_met"], stop_met)
                            - max(member["tstart_met"], start_met),
                        ),
                    }
                )
            output.append(
                {
                    "requested_start": start,
                    "requested_stop": stop,
                    "requested_start_utc": _time_utc_iso(start),
                    "requested_stop_utc": _time_utc_iso(stop),
                    "requested_start_swift": float(start_met),
                    "requested_stop_swift": float(stop_met),
                    "time_format": "swift",
                    "members": selected,
                    "full_exposure_s": float(sum(item["exposure_s"] for item in selected)),
                    "overlap_s": float(sum(item["overlap_s"] for item in selected)),
                }
            )
        return output

    @staticmethod
    def _shared_members(windows: Sequence[Mapping[str, Any]]) -> list[str]:
        counts: dict[str, int] = {}
        for window in windows:
            for member in window.get("members", ()):
                identifier = str(member.get("id"))
                counts[identifier] = counts.get(identifier, 0) + 1
        return sorted(identifier for identifier, count in counts.items() if count > 1)

    def download(
        self,
        *,
        obsids: Sequence[str],
        destination: Path,
        retries: int = 3,
        retry_wait_s: float = 5.0,
        timeout_s: float = 60.0,
        processes: int = 1,
    ) -> Mapping[str, Any]:
        downloader = getattr(self.module, "download_swiftdata", None)
        if downloader is None:
            raise RuntimeError("installed BatAnalysis has no download_swiftdata helper")
        destination.mkdir(parents=True, exist_ok=True)
        pending = list(obsids)
        collected: dict[str, Any] = {}
        for attempt in range(max(1, int(retries))):
            if not pending:
                break
            try:
                downloader_parameters = inspect.signature(downloader).parameters
            except (TypeError, ValueError):
                downloader_parameters = {}
            # BatAnalysis uses ``observations``; older project shims used
            # ``table``.  Select the explicit parameter name so a variadic
            # downloader does not receive an unexpected alias through
            # ``**kwargs``.
            observation_key = (
                "observations"
                if "observations" in downloader_parameters
                else "table"
                if "table" in downloader_parameters
                else "obsids"
                if "obsids" in downloader_parameters
                else "observations"
            )
            download_kwargs = {
                observation_key: pending,
                "reload": attempt > 0,
                "fetch": True,
                "jobs": max(1, int(processes)),
                "nprocs": max(1, int(processes)),
                "bat": True,
                "auxil": True,
                "tdrss": False,
                "save_dir": destination,
                "timeout": timeout_s,
                "timeout_s": timeout_s,
            }
            try:
                result = _call_supported(downloader, **download_kwargs)
            except Exception as exc:
                # Network and archive failures are retryable at the adapter
                # boundary.  Keep the failed attempt in the returned record,
                # but do not discard the original data or submit an unbounded
                # sequence of requests.
                result = {
                    str(obsid): {
                        "success": False,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                    for obsid in pending
                }
            if isinstance(result, Mapping):
                normalized_result = {str(key): value for key, value in result.items()}
                collected.update(normalized_result)
                missing_status = [
                    obsid for obsid in pending if str(obsid) not in normalized_result
                ]
                if missing_status:
                    # A mapping without a per-OBSID status is ambiguous: it
                    # may represent a submitted remote job whose response
                    # was truncated.  Preserve that record and stop rather
                    # than submitting the same observation again.
                    for obsid in missing_status:
                        collected.setdefault(
                            str(obsid),
                            {
                                "success": False,
                                "status": "unknown",
                                "error": "ambiguous_download_result",
                            },
                        )
                pending = [
                    obsid
                    for obsid in pending
                    if str(obsid) in normalized_result
                    and not _download_value_success(normalized_result[str(obsid)])
                ]
                if missing_status:
                    pending = []
            elif isinstance(result, bool):
                for obsid in pending:
                    collected[str(obsid)] = {"success": result}
                pending = [] if result else list(pending)
            else:
                # A non-mapping, non-boolean return gives no per-OBSID
                # completion state.  Treat it as ambiguous and do not issue
                # another remote submission automatically.
                for obsid in pending:
                    collected[str(obsid)] = {
                        "success": False,
                        "status": "unknown",
                        "error": "ambiguous_download_result",
                    }
                pending = []
            if pending and attempt + 1 < max(1, int(retries)) and retry_wait_s > 0:
                import time
                time.sleep(float(retry_wait_s))
        return collected


@dataclass(frozen=True, slots=True, kw_only=True)
class BATSurveyInput(PipelineInput):
    """Inputs for one target; local raw or processed products are supported."""

    source_name: str | None = None
    coord: tuple[float, float] | SkyCoord | None = None
    # ``coord`` is retained as the compact tuple form used by the command
    # line example.  Accepting ``skycoord=SkyCoord(...)`` as an InitVar keeps
    # the public API consistent with the other sky-position pipelines without
    # storing a second, potentially divergent coordinate representation.
    skycoord: InitVar[SkyCoord | None] = None
    time_windows: tuple[tuple[str, str], ...] = ()
    obsids: tuple[str, ...] = ()
    data_root: Path | str | None = None
    raw_products_dir: Path | str | None = None
    survey_products_dir: Path | str | None = None
    mosaic_products_dir: Path | str | None = None
    catalog_path: Path | str | None = None
    sensitivity_control_path: Path | str | None = None
    query: bool = False
    download: bool = False
    mosaic: bool = False
    detthresh: int | None = None
    detthresh2: int | None = None
    min_pcode: float | None = None
    fetcher: Any = field(default=None, compare=False, repr=False)
    backend: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self, skycoord: SkyCoord | None = None) -> None:
        coordinate = self.coord
        if skycoord is not None:
            if coordinate is not None:
                raise ValueError("provide either coord or skycoord, not both")
            coordinate = skycoord
        if isinstance(coordinate, SkyCoord):
            coordinate = (float(coordinate.icrs.ra.deg), float(coordinate.icrs.dec.deg))
        elif coordinate is not None:
            if len(coordinate) != 2:
                raise ValueError("coord must contain (ra_deg, dec_deg)")
            coordinate = (float(coordinate[0]), float(coordinate[1]))
        object.__setattr__(self, "coord", coordinate)
        windows: list[tuple[str, str]] = []
        raw_windows = self.time_windows or ()
        if (
            isinstance(raw_windows, (list, tuple))
            and len(raw_windows) == 2
            and not all(
                isinstance(value, (list, tuple)) and len(value) == 2
                for value in raw_windows
            )
        ):
            raw_windows = (raw_windows,)  # type: ignore[assignment]
        for start, stop in raw_windows:
            start_time = start.utc if isinstance(start, Time) else Time(start, scale="utc")
            stop_time = stop.utc if isinstance(stop, Time) else Time(stop, scale="utc")
            if stop_time <= start_time:
                raise ValueError("every BAT survey time window must have stop after start")
            windows.append((start_time.utc.isot, stop_time.utc.isot))
        object.__setattr__(self, "time_windows", tuple(windows))
        if self.obsids is None:
            obsids = ()
        elif isinstance(self.obsids, (str, int, np.integer)):
            obsids = (self.obsids,)
        else:
            obsids = self.obsids
        normalized_obsids = []
        for item in obsids:
            if isinstance(item, (int, np.integer)):
                text = f"{int(item):011d}"
            else:
                text = str(item).strip()
                # HEASARC/BatAnalysis accept the spacecraft's 8-digit target
                # form as well as the canonical 11-digit OBSID.  Normalize
                # numeric strings just like integer inputs while leaving the
                # synthetic ``existing`` record and other explicit labels
                # untouched.
                if text.isdigit():
                    text = f"{int(text):011d}"
            if text:
                normalized_obsids.append(text)
        object.__setattr__(
            self,
            "obsids",
            tuple(normalized_obsids),
        )
        for name in (
            "data_root", "raw_products_dir", "survey_products_dir",
            "mosaic_products_dir", "catalog_path", "sensitivity_control_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())

    def raw_root(self) -> Path:
        return Path(self.raw_products_dir or self.data_root or self.root).expanduser().resolve()

    def survey_root(self) -> Path | None:
        return (
            Path(self.survey_products_dir).expanduser().resolve()
            if self.survey_products_dir is not None
            else None
        )


def _bat_survey_input_skycoord(input_data: BATSurveyInput) -> SkyCoord | None:
    """Expose the normalized coordinate stored by ``BATSurveyInput``."""
    return (
        SkyCoord(*input_data.coord, unit="deg", frame="icrs")
        if input_data.coord is not None
        else None
    )


# ``skycoord`` is an InitVar in the dataclass (so it is accepted by the
# constructor but not stored twice); attach the read-only normalized property
# after dataclass processing so its default does not leak into ``__post_init__``.
BATSurveyInput.skycoord = property(_bat_survey_input_skycoord)  # type: ignore[attr-defined]


@dataclass(slots=True)
class BATSurveyResult:
    status: PipelineStatus
    science_status: str
    target_id: str
    workspace: Path
    completed_stages: tuple[str, ...]
    products: dict[str, dict[str, str]]
    quality_status: str = "ok"
    full_pipeline: bool = False
    warnings: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    message: str | None = None


def _config_for_input(config: InstrumentConfig, input_data: BATSurveyInput) -> BATSurvey:
    if isinstance(config, BATSurvey) and all(
        getattr(input_data, name) is None for name in ("detthresh", "detthresh2", "min_pcode")
    ):
        return config
    survey = (
        config.selection
        if isinstance(config.selection, SwiftBATSurveyConfig)
        else config.survey
        if isinstance(config.survey, SwiftBATSurveyConfig)
        else SwiftBATSurveyConfig()
    )
    values = {item.name: getattr(survey, item.name) for item in fields(survey)}
    for name in ("detthresh", "detthresh2", "min_pcode"):
        override = getattr(input_data, name)
        if override is not None:
            values[name] = override
    survey = SwiftBATSurveyConfig(**values)
    # Preserve the caller's instrument metadata and profile name while
    # replacing only the immutable survey/selection group with the resolved
    # per-input overrides.  Reconstructing just the fitting fields would make
    # a ``BATSurvey(profile="lmjagn")`` run silently look like the default
    # preset whenever one input threshold was overridden.
    base_values = {
        item.name: getattr(config, item.name)
        for item in fields(config)
        if item.name not in {"survey", "selection"}
    }
    base_values["survey"] = survey
    base_values["selection"] = survey
    return BATSurvey(**base_values)


@register_pipeline("swift.bat.survey")
class BATSurveyPipeline(InstrumentPipeline[BATSurveyInput, BATSurveyResult]):
    """Resumable one-target BAT survey workflow."""

    stages = (
        PipelineStage("preflight"),
        PipelineStage("discover", ("preflight",)),
        PipelineStage("download", ("discover",)),
        PipelineStage("survey", ("discover", "download")),
        PipelineStage("lightcurve", ("survey",)),
        PipelineStage("mosaic", ("lightcurve",)),
        PipelineStage("spectra", ("survey", "lightcurve", "mosaic")),
        PipelineStage("fit", ("lightcurve", "spectra")),
        PipelineStage(
            "report",
            (
                "preflight", "discover", "download", "survey", "lightcurve",
                "mosaic", "spectra", "fit",
            ),
        ),
    )

    def __init__(
        self,
        input_data: BATSurveyInput,
        *,
        config: InstrumentConfig | None = None,
        backend: Any | None = None,
    ):
        resolved = _config_for_input(config or BATSurvey(), input_data)
        super().__init__(input_data, config=resolved)
        self.survey_config = (
            resolved.selection
            or resolved.survey
            or SwiftBATSurveyConfig()
        )
        self.backend = backend if backend is not None else input_data.backend
        self._points: list[SurveyRatePoint] = []
        self._pha_exclusions: list[dict[str, Any]] = []

    def run(self, *, until: str | None = None, resume: bool | None = None) -> BATSurveyResult:
        # Conda's HEASoft packages are intentionally not auto-sourced by
        # ``conda run``.  Keep the core config import side-effect free, but
        # establish a process-local environment at the optional-instrument
        # boundary so BatAnalysis, heasoftpy and PyXspec see the same HEADAS,
        # PFILES and writable HOME for every stage.
        environment = self._headas_environment("pipeline")
        python_entries = tuple(
            item for item in environment.get("PYTHONPATH", "").split(os.pathsep)
            if item
        )
        with _temporary_environment(environment), _temporary_sys_path(python_entries):
            return super().run(until=until, resume=resume)

    def _headas_environment(self, stage_name: str) -> dict[str, str]:
        """Return a safe environment for in-process and external HEASoft work.

        Prefer an explicitly configured ``HEADAS`` tree, then the Conda
        environment's conventional ``heasoft`` subdirectory.  If neither is
        present the caller still receives the inherited environment, allowing
        offline/fake backends to run without making HEASoft a core dependency.
        The returned environment uses stage-local PFILES and HOME directories
        below the pipeline workspace, so an installed user's configuration is
        never modified.
        """
        inherited = os.environ.copy()
        headas_value = inherited.get("HEADAS")
        candidate: Path | None = None
        if headas_value:
            explicit = Path(headas_value).expanduser()
            if (explicit / "bin").is_dir():
                candidate = explicit.resolve()
        if candidate is None:
            prefix = inherited.get("CONDA_PREFIX")
            if prefix:
                conda_candidate = Path(prefix).expanduser() / "heasoft"
                if (conda_candidate / "bin").is_dir():
                    candidate = conda_candidate.resolve()
        if candidate is None:
            return inherited

        from ...core.ops import ensure_headas_env

        stage_token = safe_filename_token(stage_name)
        pfiles = self.workspace / ".pfiles" / stage_token
        home = self.workspace / ".heasoft_home" / stage_token
        environment = ensure_headas_env(
            headas=candidate,
            env=inherited,
            home=home,
            pfiles=pfiles,
        )
        # HEASoft accepts the local parameter directory followed by its
        # system directory separated by a semicolon.  ``ensure_headas_env``
        # uses the host path separator for portability, so normalize this
        # boundary explicitly for the HEASoft grammar.
        syspfiles = candidate / "syspfiles"
        environment["PFILES"] = (
            f"{pfiles};{syspfiles}" if syspfiles.is_dir() else str(pfiles)
        )
        environment["HEADASNOQUERY"] = "1"
        environment["HEADASPROMPT"] = "/dev/null"
        # ``heainit.sh`` normally exports only CALDB; many shell profiles
        # additionally export CALDBCONFIG/CALDBALIAS.  Supply the Conda
        # package's local pair when those variables are absent so a clean
        # ``conda run`` has the same calibration lookup contract without
        # sourcing a script that writes to the user's HOME.
        caldb_defaults = candidate / "lib" / "perl" / "aht" / "aht-caldb"
        environment.setdefault("CALDB", "https://heasarc.gsfc.nasa.gov/FTP/caldb")
        environment.setdefault(
            "CALDBCONFIG", str(caldb_defaults / "caldb.config")
        )
        environment.setdefault(
            "CALDBALIAS", str(caldb_defaults / "alias_config.fits")
        )
        lib = candidate / "lib"
        if lib.is_dir():
            previous_ld = environment.get("LD_LIBRARY_PATH", "")
            parts = [str(lib)] + [item for item in previous_ld.split(os.pathsep) if item]
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                item for index, item in enumerate(parts) if item not in parts[:index]
            )
        return environment

    def _input_fingerprint(self) -> str:
        """Fingerprint scientific inputs while excluding runtime adapters.

        ``fetcher`` and ``backend`` are deliberately injectable callables or
        third-party objects.  They are execution adapters rather than
        scientific input files and are not JSON-serializable in the base
        dataclass fingerprint.  Their implementation is covered by the
        stage-code hash; paths, windows and configuration remain immutable
        fingerprint inputs.
        """
        implementation = inspect.getsourcefile(type(self))
        implementation_hash = None
        if implementation is not None and Path(implementation).is_file():
            implementation_hash = _file_fingerprint(Path(implementation))
        from importlib.metadata import PackageNotFoundError, version

        backend_versions: dict[str, str | None] = {}
        for package in ("batanalysis", "swiftbat", "astroquery"):
            try:
                backend_versions[package] = version(package)
            except PackageNotFoundError:
                backend_versions[package] = None
        input_payload = {
            item.name: getattr(self.input, item.name)
            for item in fields(self.input)
            if item.name not in {"fetcher", "backend"}
        }
        return _fingerprint(
            {
                "input": input_payload,
                "pipeline_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "implementation_sha256": implementation_hash,
                "backend_versions": backend_versions,
            }
        )

    def include_config_in_input_fingerprint(self) -> bool:
        """Use stage-scoped config fingerprints for BAT resume.

        Survey extraction and fitting are separate products.  A change to the
        model or error convention must invalidate fitting, while a plotting or
        reporting change should leave the downloaded and cleaned survey data
        reusable.
        """
        return False

    def stage_config_dependencies(self, stage: PipelineStage) -> Any:
        config = self.config
        common = {
            "instrument": (config.name, config.pipeline),
            "execution": config.execution,
        }
        if stage.name == "preflight":
            return {**common, "downloads": config.downloads, "survey": config.survey}
        if stage.name in {"discover", "download"}:
            return {
                **common,
                "downloads": config.downloads,
                "survey": config.survey,
                "selection": config.selection,
            }
        if stage.name in {"survey", "lightcurve"}:
            return {
                **common,
                "survey": config.survey,
                "selection": config.selection,
                "downloads": config.downloads if stage.name == "survey" else None,
            }
        if stage.name == "mosaic":
            return {
                **common,
                "survey": config.survey,
                "selection": config.selection,
                "mosaic": config.mosaic,
            }
        if stage.name == "spectra":
            return {
                **common,
                "survey": config.survey,
                "selection": config.selection,
                "spectrum": config.spectrum,
            }
        if stage.name == "fit":
            return {
                **common,
                "survey": config.survey,
                "selection": config.selection,
                "spectrum": config.spectrum,
                "fitting": config.fitting,
                "upper_limit": config.upper_limit,
                "plotting": config.plotting,
            }
        if stage.name == "report":
            return {
                **common,
                "survey": config.survey,
                "selection": config.selection,
                "mosaic": config.mosaic,
                "spectrum": config.spectrum,
                "fitting": config.fitting,
                "upper_limit": config.upper_limit,
                "plotting": config.plotting,
                "reporting": config.reporting,
            }
        return config

    def _stage_input_fingerprint(self, stage: PipelineStage) -> str:
        """Include calibration-file content in the backend stage cache.

        ``CALDB`` is normally a directory, so the generic pipeline records
        its existence and directory metadata without recursively walking it.
        BAT response generation and XSPEC fitting depend on the files inside
        that directory; include their relative paths and content hashes for
        those stages so an in-place calibration replacement invalidates only
        the affected survey/mosaic/spectrum/fit work.
        """
        base = super()._stage_input_fingerprint(stage)
        caldb_value = os.environ.get("CALDB")
        if not caldb_value:
            return base
        caldb = Path(caldb_value).expanduser().resolve()
        if not caldb.is_dir() or stage.name not in {"survey", "mosaic", "spectra", "fit"}:
            return base
        files: list[dict[str, Any]] = []
        try:
            for path in sorted(item for item in caldb.rglob("*") if item.is_file()):
                resolved = path.resolve()
                files.append(
                    {
                        "path": str(path.relative_to(caldb)),
                        "sha256": _file_fingerprint(resolved),
                    }
                )
        except (OSError, ValueError) as exc:
            files.append({"error": f"{type(exc).__name__}:{exc}"})
        return _fingerprint({"base": base, "caldb_files": files})

    def _load_cached_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult | None:
        """Do not resume a stage explicitly marked as needing recovery.

        BAT survey stages intentionally return ``COMPLETED`` after recording
        per-observation failures so that the report can include the partial
        products.  A normal pipeline cache would then treat that partial
        result as final forever, even after HEASoft, CALDB, or a missing raw
        product becomes available.  Handlers set ``cache_reusable=False``
        for such results; the next invocation reruns only that stage and its
        downstream dependants while retaining the existing manifest and raw
        data.
        """
        result = super()._load_cached_stage(stage, context)
        if result is None:
            return None
        if result.data.get("cache_reusable") is False:
            return None
        return result

    def validate_input(self) -> None:
        if not self.input.target_id.strip():
            raise ValueError("target_id must not be empty")
        candidate_roots = [self.input.resolved_root()]
        for value in (
            self.input.data_root,
            self.input.raw_products_dir,
            self.input.survey_products_dir,
            self.input.mosaic_products_dir,
        ):
            if value is None:
                continue
            candidate = Path(value).expanduser().resolve()
            candidate_roots.append(candidate if candidate.is_dir() else candidate.parent)
        query_configured = bool(
            self.input.query
            or (self.config.downloads and self.config.downloads.query_enabled)
        )
        download_configured = bool(
            self.input.download
            or (self.config.downloads and self.config.downloads.download_enabled)
        )
        if not any(candidate.is_dir() for candidate in candidate_roots) and not (
            query_configured or download_configured
        ):
            raise FileNotFoundError(
                f"BAT survey input root does not exist: {self.input.resolved_root()}"
            )
        if self.input.coord is not None:
            ra, dec = self.input.coord
            if not (-360 <= ra <= 360 and -90 <= dec <= 90):
                raise ValueError("coord must be valid ICRS degrees")
        if query_configured and self.input.coord is None:
            raise ValueError("online BAT survey query requires coord")
        if query_configured and not self.input.time_windows:
            raise ValueError("online BAT survey query requires at least one time window")

    def stage_code_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        from ...core import config as config_module
        from ...core import fit as fit_module, products as products_module

        def _src(module) -> Path:
            # getsourcefile 失败时显式报错：退化成 Path("") 会让指纹静默失效（AUD-01）
            path = inspect.getsourcefile(module)
            if not path:
                raise RuntimeError(f"无法定位模块源文件: {module!r}")
            return Path(path)

        return (
            Path(__file__).resolve(),
            _src(config_module),
            _src(fit_module),
            _src(products_module),
        )

    def stage_input_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        paths: list[Path] = []

        def files_or_path(
            value: Path,
            *,
            include: Callable[[Path], bool] | None = None,
        ) -> list[Path]:
            """Expand product directories so content changes invalidate caches.

            ``include`` lets a downstream stage declare the product family it
            actually consumes.  In particular, a response replacement must
            invalidate PHA validation/fitting without needlessly re-reading a
            survey rate light curve.
            """
            if not value.is_dir():
                return [value] if include is None or include(value) else []
            files = sorted(
                item
                for item in value.rglob("*")
                if item.is_file() and (include is None or include(item))
            )
            return files or [value]

        def is_rate_product(path: Path) -> bool:
            """Files that can contribute a survey light-curve measurement."""
            if _has_suffix(path, (".pha", ".rsp", ".rmf", ".arf")):
                return False
            return _has_suffix(
                path,
                (".cat", ".csv", ".dat", ".txt", ".qdp", ".fits", ".fit"),
            )

        # Keep cache invalidation scoped to the stages that consume each
        # product family.  Downstream stages still inherit upstream output
        # fingerprints, so a raw-file change invalidates survey/lightcurve
        # transitively without forcing an unrelated mosaic-only stage to
        # rerun.
        raw_stages = {"discover", "download", "survey", "mosaic"}
        survey_stages = {"discover", "survey", "lightcurve", "spectra", "fit", "mosaic"}
        if stage.name in raw_stages:
            raw_value: Path | str | None = self.input.raw_products_dir
            if raw_value is None:
                raw_root = self.input.raw_root()
                # Hash only requested OBSID directories when possible.  A
                # Swift mirror can contain millions of unrelated files, and
                # hashing its entire root for every manifest would make resume
                # unusable.
                if self.input.obsids:
                    for obsid in self.input.obsids:
                        paths.extend(files_or_path(raw_root / str(obsid)))
                elif self.input.survey_products_dir is None or self.input.data_root is not None:
                    paths.extend(files_or_path(raw_root))
            else:
                paths.extend(files_or_path(Path(raw_value)))
        if stage.name in survey_stages and self.input.survey_products_dir is not None:
            # Mosaic selection consumes pointing intervals/rates and the
            # inventory, never a PHA response.  Keep response replacements
            # from invalidating an otherwise reusable mosaic stage, just as
            # they do not invalidate the light curve.
            product_filter = (
                is_rate_product
                if stage.name in {"survey", "lightcurve", "mosaic"}
                else None
            )
            paths.extend(
                files_or_path(Path(self.input.survey_products_dir), include=product_filter)
            )
        if stage.name in {"survey", "mosaic"} and self.input.catalog_path is not None:
            paths.extend(files_or_path(Path(self.input.catalog_path)))
        if stage.name == "mosaic" and self.input.mosaic_products_dir is not None:
            paths.extend(files_or_path(Path(self.input.mosaic_products_dir)))
        # Survey and mosaic adapters write their PHA, response and CAT files
        # below the workspace.  Those files are scientific inputs to the
        # downstream validation/fit stages even when the caller did not pass
        # an explicit ``survey_products_dir``.  Include them so replacing a
        # response or editing a generated CAT invalidates only the dependent
        # stages on the next resume.
        if stage.name in {"lightcurve", "spectra", "fit", "mosaic"}:
            generated_survey = self.workspace / "survey"
            product_filter = (
                is_rate_product
                if stage.name in {"lightcurve", "mosaic"}
                else None
            )
            paths.extend(files_or_path(generated_survey, include=product_filter))
        if stage.name in {"spectra", "fit"}:
            generated_mosaic = self.workspace / "mosaic"
            paths.extend(files_or_path(generated_mosaic))
        if stage.name in {"survey", "mosaic", "spectra", "fit"}:
            # CALDB normally points at a directory while CALDBCONFIG and
            # CALDBALIAS point at small text files.  Hash the concrete config
            # files here; the directory location itself is already part of
            # the run fingerprint and avoids recursively hashing a large
            # calibration tree on every resume.
            for variable in ("CALDBCONFIG", "CALDBALIAS"):
                value = os.environ.get(variable)
                if value and Path(value).expanduser().is_file():
                    paths.append(Path(value).expanduser())
            caldb = os.environ.get("CALDB")
            if caldb:
                # Keep the calibration directory scoped to the stages that
                # actually invoke BatAnalysis/XSPEC.  The base fingerprint
                # records directory metadata; concrete CALDBCONFIG/ALIAS
                # files above carry content hashes for in-place edits.
                paths.append(Path(caldb).expanduser())
        if stage.name in {"fit", "report"} and self.input.sensitivity_control_path is not None:
            paths.extend(files_or_path(Path(self.input.sensitivity_control_path)))
        return tuple(sorted(set(paths)))

    def execute_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult:
        handlers = {
            "preflight": self._stage_preflight,
            "discover": self._stage_discover,
            "download": self._stage_download,
            "survey": self._stage_survey,
            "lightcurve": self._stage_lightcurve,
            "mosaic": self._stage_mosaic,
            "spectra": self._stage_spectra,
            "fit": self._stage_fit,
            "report": self._stage_report,
        }
        return handlers[stage.name](context)

    def _output(self, name: str) -> Path:
        path = self.workspace / "products" / safe_filename_token(self.input.target_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _stage_preflight(self, _: Mapping[str, StageResult]) -> StageResult:
        import importlib

        import io
        import shutil

        # Probe the actual runtime environment rather than distribution
        # metadata.  In a Conda HEASoft install, ``heainit.sh`` is not sourced
        # by ``conda run``; ``run`` establishes the same paths in-process and
        # this local context keeps the preflight probe consistent with later
        # BatAnalysis/XSPEC calls.
        environment = self._headas_environment("preflight")
        python_entries = tuple(
            item for item in environment.get("PYTHONPATH", "").split(os.pathsep)
            if item
        )
        backend_import_error: str | None = None
        heasoftpy_import_error: str | None = None
        pyxspec_import_error: str | None = None
        runtime_modules: dict[str, Any] = {}
        with _temporary_environment(environment), _temporary_sys_path(python_entries):
            backend_available = self.backend is not None
            if not backend_available:
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        runtime_modules["batanalysis"] = importlib.import_module(
                            "batanalysis"
                        )
                    backend_available = True
                except Exception as exc:
                    backend_import_error = f"{type(exc).__name__}:{exc}"
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    runtime_modules["heasoftpy"] = importlib.import_module("heasoftpy")
                heasoftpy_available = True
            except Exception as exc:
                heasoftpy_available = False
                heasoftpy_import_error = f"{type(exc).__name__}:{exc}"
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    runtime_modules["xspec"] = importlib.import_module("xspec")
                pyxspec_available = True
            except Exception as exc:
                pyxspec_available = False
                pyxspec_import_error = f"{type(exc).__name__}:{exc}"
            headas = environment.get("HEADAS")
            xspec_executable = shutil.which("xspec", path=environment.get("PATH"))
            if xspec_executable is None and headas:
                candidate = Path(headas) / "bin" / "xspec"
                xspec_executable = str(candidate) if candidate.is_file() else None
            caldb = environment.get("CALDB")
            caldb_config = environment.get("CALDBCONFIG")
            caldb_alias = environment.get("CALDBALIAS")
            config_ok = bool(caldb_config and Path(caldb_config).expanduser().is_file())
            alias_ok = bool(caldb_alias and Path(caldb_alias).expanduser().is_file())
            if caldb and Path(caldb).expanduser().is_dir():
                caldb_mode = "local"
            elif caldb and urlparse(str(caldb)).scheme in {"http", "https", "ftp"}:
                # CALDB may intentionally be an HEASARC URL.  It is usable by
                # BatAnalysis when the local CALDBCONFIG/CALDBALIAS pair is
                # present; no network request is made by this check.
                caldb_mode = "remote" if config_ok and alias_ok else "unusable"
            elif caldb:
                caldb_mode = "unusable"
            else:
                caldb_mode = "missing"
            caldb_available = caldb_mode in {"local", "remote"}
            caldb_provenance: list[dict[str, Any]] = []
            caldb_checksum_warnings: list[str] = []
            for label, value in (
                ("CALDBCONFIG", caldb_config),
                ("CALDBALIAS", caldb_alias),
            ):
                if value and Path(value).expanduser().is_file():
                    concrete = Path(value).expanduser().resolve()
                    try:
                        checksum = _file_fingerprint(concrete)
                    except OSError as exc:
                        checksum = None
                        caldb_checksum_warnings.append(
                            f"caldb_checksum_failed:{label}:{type(exc).__name__}:{exc}"
                        )
                    caldb_provenance.append(
                        {
                            "name": label,
                            "path": str(concrete),
                            "sha256": checksum,
                            "source": "HEASoft_CALDB_configuration",
                        }
                    )
                else:
                    caldb_provenance.append(
                        {
                            "name": label,
                            "path": str(value) if value else None,
                            "sha256": None,
                            "source": "missing",
                        }
                    )
        runtime_versions: dict[str, str | None] = {}
        runtime_sources: dict[str, str | None] = {}
        for name, module in runtime_modules.items():
            source = getattr(module, "__file__", None)
            runtime_sources[name] = str(Path(source).resolve()) if source else None
            version = getattr(module, "__version__", None)
            if name == "xspec":
                # PyXspec exposes both its Python wrapper and XSPEC engine
                # versions through Xset.version rather than __version__.
                xset_version = getattr(getattr(module, "Xset", None), "version", None)
                if isinstance(xset_version, (tuple, list)) and xset_version:
                    version = "/".join(str(item) for item in xset_version)
                elif xset_version:
                    version = str(xset_version)
            runtime_versions[name] = str(version) if version is not None else None
        checks = {
            "batanalysis": backend_available,
            "heasoftpy": heasoftpy_available,
            "pyxspec": pyxspec_available,
            "xspec_executable": xspec_executable is not None,
            "caldb": caldb_available,
        }
        dependency_warnings = [
            f"missing_dependency:{name}"
            for name, available in checks.items()
            if not available
        ]
        if backend_import_error:
            dependency_warnings.append(f"batanalysis_import_failed:{backend_import_error}")
        if heasoftpy_import_error:
            dependency_warnings.append(f"heasoftpy_import_failed:{heasoftpy_import_error}")
        if pyxspec_import_error:
            dependency_warnings.append(f"pyxspec_import_failed:{pyxspec_import_error}")
        if caldb_mode == "unusable":
            dependency_warnings.append("caldb_configuration_unusable")
        dependency_warnings.extend(caldb_checksum_warnings)
        payload = {
            "target_id": self.input.target_id,
            "bat_survey_config": self.survey_config,
            "bat_analysis_available": backend_available or self.backend is not None,
            "dependency_checks": checks,
            "xspec_executable": xspec_executable,
            "caldb": caldb,
            "caldb_mode": caldb_mode,
            "caldb_config": caldb_config,
            "caldb_alias": caldb_alias,
            "caldb_provenance": caldb_provenance,
            "headas": headas,
            "raw_root": str(self.input.raw_root()),
            "runtime_versions": runtime_versions,
            "runtime_sources": runtime_sources,
            "source": "local_first",
            "warnings": dependency_warnings,
            # Environment failures are recoverable after the user installs
            # HEASoft/PyXspec or points CALDB at a writable calibration tree.
            "cache_reusable": not dependency_warnings,
        }
        path = self._output("preflight.json")
        write_json(path, payload)
        return StageResult(outputs={"preflight": str(path)}, data=payload)

    def _discover_local(self) -> list[dict[str, Any]]:
        root = self.input.raw_root()
        obsids = list(self.input.obsids)
        root_is_observation, _root_diagnostics = validate_observation_directory(root)
        root_observation_id = root.name
        if not obsids and root_is_observation:
            # ``raw_products_dir`` is often set directly to one OBSID tree in
            # small, target-specific workspaces.
            obsids = [root_observation_id]
        if not obsids and root.is_dir():
            obsids = sorted(
                path.name
                for path in root.iterdir()
                if path.is_dir() and path.name.isdigit() and len(path.name) in {8, 11}
            )
        records: list[dict[str, Any]] = []
        survey_root = self.input.survey_root()
        survey_by_obsid: dict[str, Path] = {}
        if survey_root is not None and survey_root.is_file():
            # A single precomputed CAT/PHA/FITS/CSV is a valid local input;
            # keep its exact path as the product source.  If an OBSID was
            # supplied, associate the file with that one explicit target.
            if not _has_suffix(survey_root, _SURVEY_PRODUCT_SUFFIXES):
                raise ValueError(f"unsupported survey product file: {survey_root}")
            if not obsids:
                obsids = ["existing"]
            elif len(obsids) == 1:
                survey_by_obsid[obsids[0]] = survey_root
        elif survey_root is not None and survey_root.is_dir():
            direct_match = re.fullmatch(r"(\d{8,11})_surveyresult", survey_root.name)
            if direct_match:
                survey_by_obsid[direct_match.group(1)] = survey_root
                if not obsids:
                    obsids = [direct_match.group(1)]
            children = [path for path in survey_root.iterdir() if path.is_dir()]
            named = [path for path in children if path.name.isdigit() or "survey" in path.name.lower()]
            if named and not direct_match:
                survey_by_obsid = {
                    path.name.removesuffix("_surveyresult"): path for path in named
                }
                if not obsids:
                    obsids = sorted(survey_by_obsid)
            elif not obsids and any(path.is_file() for path in survey_root.iterdir()):
                # A caller may provide one already-produced survey directory
                # without retaining the raw OBSID tree.
                obsids = ["existing"]
        for obsid in obsids:
            obs_dir = (
                root
                if obsid == "existing"
                or (root_is_observation and obsid == root_observation_id)
                else root / obsid
            )
            survey_dir = (
                survey_by_obsid[obsid]
                if obsid in survey_by_obsid
                else survey_root / obsid
                if survey_root is not None and obsid != "existing" and (survey_root / obsid).exists()
                else survey_root
                if obsid == "existing" and survey_root is not None
                else obs_dir.with_name(obs_dir.name + "_surveyresult")
            )
            valid, diagnostics = validate_observation_directory(obs_dir)
            existing_product = _has_product(survey_dir)
            survey_diagnostics = list(_survey_product_diagnostics(survey_dir))
            records.append({
                "obsid": obsid,
                "obs_dir": str(obs_dir),
                "survey_dir": str(survey_dir),
                "local_valid": valid or existing_product,
                "diagnostics": list(diagnostics) + survey_diagnostics,
            })
        return records

    def _query_online(self) -> list[dict[str, Any]]:
        if self.input.fetcher is not None:
            fetched = self.input.fetcher(self.input)
            if isinstance(fetched, Mapping):
                # A small adapter often returns one row as a mapping, while
                # download/query helpers may return ``{obsid: row}``.  Keep
                # both shapes lossless and attach the mapping key when it is
                # the only available OBSID.
                aliases = {
                    "OBSID", "OBS_ID", "RA", "RA_PNT", "DEC", "DEC_PNT",
                    "START_TIME", "STOP_TIME", "TSTART", "TSTOP",
                }
                normalized_keys = {str(key).upper() for key in fetched}
                if normalized_keys & aliases:
                    fetched = [fetched]
                elif not fetched:
                    fetched = []
                elif fetched and all(isinstance(value, Mapping) for value in fetched.values()):
                    fetched = [
                        {"obsid": key, **dict(value)}
                        if not any(str(name).upper() in {"OBSID", "OBS_ID"} for name in value)
                        else dict(value)
                        for key, value in fetched.items()
                    ]
                else:
                    fetched = [fetched]
            return [dict(row) for row in (fetched or ())]
        try:
            from astroquery.heasarc import Heasarc
            import swiftbat
        except ImportError as exc:
            raise RuntimeError(
                "online BAT survey query requires optional astroquery and swiftbat"
            ) from exc
        if not self.input.time_windows or self.input.skycoord is None:
            raise ValueError("online query requires coord and at least one time window")
        source = swiftbat.source(
            ra=float(self.input.skycoord.icrs.ra.deg),
            dec=float(self.input.skycoord.icrs.dec.deg),
            name=self.input.source_name or self.input.target_id,
        )
        heasarc = Heasarc()
        rows: dict[str, dict[str, Any]] = {}
        download_config = self.config.downloads
        margin = float(download_config.query_margin_s) if download_config is not None else 0.0
        timeout_s = float(download_config.timeout_s) if download_config is not None else 60.0
        for start, stop in self.input.time_windows:
            start_time, stop_time = Time(start, scale="utc"), Time(stop, scale="utc")
            if margin:
                from astropy.time import TimeDelta
                start_time -= TimeDelta(margin, format="sec")
                stop_time += TimeDelta(margin, format="sec")
            query = (
                "SELECT TOP 9999999 obsid,ra,dec,roll_angle,start_time,stop_time,bat_exposure "
                "FROM swiftmastr "
                f"WHERE stop_time >= {start_time.mjd:.10f} "
                f"AND start_time <= {stop_time.mjd:.10f}"
            )
            queried = _query_tap_with_timeout(heasarc, query, timeout_s)
            table = queried.to_table() if callable(getattr(queried, "to_table", None)) else queried
            for row in table:
                obsid = str(_decode(_field(row, ("OBSID", "OBS_ID"))) or "").strip()
                if not obsid:
                    continue
                if obsid.isdigit():
                    obsid = f"{int(obsid):011d}"
                try:
                    exposure = source.exposure(
                        ra=float(_field(row, ("RA", "RA_PNT"))),
                        dec=float(_field(row, ("DEC", "DEC_PNT"))),
                        roll=float(_field(row, ("ROLL_ANGLE", "PA_PNT", "ROLL"))),
                    )
                    if isinstance(exposure, (tuple, list, np.ndarray)):
                        exposure = exposure[0]
                    area = float(
                        exposure.to_value(u.cm**2)
                        if hasattr(exposure, "to_value")
                        else exposure
                    )
                except (TypeError, ValueError, IndexError, AttributeError):
                    # Keep the observation identity and timing for the
                    # discovery/download audit, but make missing coded-area
                    # geometry explicit so later selection cannot treat it as
                    # valid source coverage.
                    area = float("nan")
                rows[obsid] = {
                    "obsid": obsid,
                    "start_time_mjd": _mjd_value(_field(row, ("START_TIME",))),
                    "stop_time_mjd": _mjd_value(_field(row, ("STOP_TIME",))),
                    "bat_exposure_s": _float(_field(row, ("BAT_EXPOSURE",))),
                    "coded_area_cm2": area,
                    "query_timeout_s": timeout_s,
                    "source": "swiftmastr",
                }
                if rows[obsid]["start_time_mjd"] is not None:
                    rows[obsid]["start_time_utc"] = Time(
                        rows[obsid]["start_time_mjd"], format="mjd", scale="utc"
                    ).utc.isot
                if rows[obsid]["stop_time_mjd"] is not None:
                    rows[obsid]["stop_time_utc"] = Time(
                        rows[obsid]["stop_time_mjd"], format="mjd", scale="utc"
                    ).utc.isot
        return list(rows.values())

    def _stage_discover(self, _: Mapping[str, StageResult]) -> StageResult:
        records = self._discover_local()
        warnings: list[str] = []
        query_configured = bool(
            self.input.query
            or (self.config.downloads and self.config.downloads.query_enabled)
        )
        if query_configured:
            query_config = self.config.downloads
            retries = query_config.retries if query_config is not None else 3
            retry_wait = query_config.retry_wait_s if query_config is not None else 5.0
            online: list[dict[str, Any]] | None = None
            for attempt in range(max(1, int(retries))):
                try:
                    online = self._query_online()
                    break
                except Exception as exc:
                    if attempt + 1 >= max(1, int(retries)):
                        warnings.append(f"query_failed:{type(exc).__name__}:{exc}")
                    elif retry_wait > 0:
                        import time
                        time.sleep(float(retry_wait))
            if online is not None:
                known = {row["obsid"] for row in records}
                for row in online:
                    raw_obsid = row.get("obsid")
                    if isinstance(raw_obsid, (int, np.integer)):
                        obsid = f"{int(raw_obsid):011d}"
                    else:
                        obsid = str(raw_obsid or "").strip()
                        if obsid.isdigit():
                            obsid = f"{int(obsid):011d}"
                    if obsid and obsid not in known:
                        survey_root = self.input.survey_root()
                        if survey_root is not None and survey_root.is_file():
                            survey_dir = survey_root
                        elif survey_root is not None and survey_root.is_dir():
                            candidates = (
                                survey_root / f"{obsid}_surveyresult",
                                survey_root / obsid,
                            )
                            survey_dir = next(
                                (candidate for candidate in candidates if candidate.exists()),
                                survey_root / f"{obsid}_surveyresult",
                            )
                        else:
                            survey_dir = self.input.raw_root() / f"{obsid}_surveyresult"
                        records.append({
                            "obsid": obsid,
                            "obs_dir": str(self.input.raw_root() / obsid),
                            "survey_dir": str(survey_dir),
                            "online": row,
                            "local_valid": False,
                            "diagnostics": ["not_local"],
                        })
                        known.add(obsid)
        path = self._output("discovery.json")
        if not records:
            warnings.append("no_observations_found")
        query_config = self.config.downloads
        payload = {
            "records": records,
            "warnings": warnings,
            "query_requested": query_configured,
            "query_windows_utc": [list(window) for window in self.input.time_windows],
            "query_timeout_s": (
                float(query_config.timeout_s) if query_config is not None else 60.0
            ),
            "query_margin_s": (
                float(query_config.query_margin_s) if query_config is not None else 0.0
            ),
            "cache_reusable": not warnings,
        }
        write_json(path, payload)
        return StageResult(outputs={"discovery": str(path)}, data=payload)

    def _stage_download(self, context: Mapping[str, StageResult]) -> StageResult:
        records = list(context["discover"].data.get("records", []))
        download_configured = bool(
            self.input.download
            or (self.config.downloads and self.config.downloads.download_enabled)
        )
        if not download_configured:
            payload = {
                "enabled": False,
                "records": records,
                "warnings": ["download_not_requested"],
                "requested_obsids": [],
                "cache_reusable": True,
            }
        else:
            try:
                backend = (
                    self.backend
                    if self.backend is not None
                    else BatAnalysisSurveyBackend()
                )
                missing = [
                    row["obsid"]
                    for row in records
                    if not row.get("local_valid") and str(row.get("obsid")) != "existing"
                ]
                download_config = self.config.downloads
                with self.stage_environment("download") as environment:
                    with _temporary_environment(environment), _math_thread_limit(
                        self.survey_config.internal_threads
                    ):
                        result = (
                            _call_supported(
                                backend.download,
                                obsids=missing,
                                destination=self.input.raw_root(),
                                retries=download_config.retries if download_config else 3,
                                retry_wait_s=download_config.retry_wait_s if download_config else 5.0,
                                timeout_s=download_config.timeout_s if download_config else 60.0,
                                processes=self.survey_config.processes,
                            )
                            if missing
                            else {}
                        )
                download_warnings = []
                if isinstance(result, Mapping):
                    failed = sorted(
                        str(obsid)
                        for obsid, value in result.items()
                        if not _download_value_success(value)
                    )
                    if failed:
                        download_warnings.append(
                            "download_incomplete:" + ",".join(failed)
                        )
                else:
                    download_warnings = ["download_returned_no_status"]
                for row in records:
                    valid, diagnostics = validate_observation_directory(Path(row["obs_dir"]))
                    row["local_valid"], row["diagnostics"] = valid, list(diagnostics)
                payload = {
                    "enabled": True,
                    "records": records,
                    "download_result": _json_safe_external(result),
                    "warnings": download_warnings,
                    "requested_obsids": list(missing),
                    "download_timeout_s": (
                        float(download_config.timeout_s)
                        if download_config is not None
                        else 60.0
                    ),
                    "download_retries": (
                        int(download_config.retries)
                        if download_config is not None
                        else 3
                    ),
                    "download_processes": int(self.survey_config.processes),
                    "cache_reusable": not download_warnings,
                }
            except Exception as exc:
                payload = {
                    "enabled": True,
                    "records": records,
                    "warnings": [f"download_failed:{type(exc).__name__}:{exc}"],
                    "requested_obsids": list(
                        row.get("obsid")
                        for row in records
                        if not row.get("local_valid")
                        and str(row.get("obsid")) != "existing"
                    ),
                    "cache_reusable": False,
                }
        path = self._output("download.json")
        write_json(path, payload)
        return StageResult(outputs={"download": str(path)}, data=payload)

    def _catalog(self) -> Path:
        catalog = self.workspace / "inputs" / f"{safe_filename_token(self.input.target_id)}.cat"
        if self.input.catalog_path is not None:
            base_catalog = Path(self.input.catalog_path)
            if not base_catalog.is_file():
                raise FileNotFoundError(f"BAT source catalog does not exist: {base_catalog}")
        else:
            base_catalog = None
        if self.input.skycoord is None:
            selected = base_catalog or _default_catalog_path()
            try:
                with fits.open(selected, memmap=False) as hdus:
                    names = {
                        str(name).upper()
                        for name in (getattr(hdus[1].data, "names", ()) or ())
                    }
                if not {"NAME", "RA_OBJ", "DEC_OBJ"}.issubset(names):
                    raise ValueError("catalog must contain NAME, RA_OBJ and DEC_OBJ columns")
            except (OSError, IndexError, ValueError) as exc:
                raise ValueError(f"invalid BAT source catalog: {selected}") from exc
            return selected
        return build_source_catalog(
            catalog,
            source_name=self.input.source_name or self.input.target_id,
            coordinate=self.input.skycoord,
            base_catalog=base_catalog,
        )

    def _stage_survey(self, context: Mapping[str, StageResult]) -> StageResult:
        records = list(context["download"].data.get("records", context["discover"].data.get("records", [])))
        backend = self.backend
        warnings: list[str] = []
        processed: list[dict[str, Any]] = []
        for row in records:
            obsid = row["obsid"]
            obs_dir, survey_dir = Path(row["obs_dir"]), Path(row["survey_dir"])
            existing_product = _has_product(survey_dir)
            if not row.get("local_valid") and not existing_product:
                warnings.append(f"{obsid}:observation_not_ready:{','.join(row.get('diagnostics', []))}")
                processed.append({**row, "status": "not_ready"})
                continue
            try:
                if existing_product:
                    result = {"result_dir": str(survey_dir), "status": "existing_products"}
                    product_diagnostics = [
                        item
                        for item in row.get("diagnostics", ())
                        if str(item).startswith(("pointing_status_", "status_"))
                    ]
                    if product_diagnostics:
                        warnings.append(
                            f"{obsid}:survey_product_quality:{','.join(product_diagnostics)}"
                        )
                    # Reuse an existing BatAnalysis cache when available so
                    # an explicitly requested mosaic can include local
                    # products.  Parsing-only workflows continue when the
                    # optional adapter or its cache dependencies are absent.
                    if self.input.mosaic or (
                        self.config.mosaic and self.config.mosaic.enabled
                    ):
                        try:
                            if backend is None:
                                backend = BatAnalysisSurveyBackend()
                                self.backend = backend
                            if hasattr(backend, "load_observation") and obsid != "existing":
                                loaded = _call_supported(
                                    backend.load_observation,
                                    obsid=obsid,
                                    result_dir=survey_dir,
                                    source_name=self.input.source_name or self.input.target_id,
                                    obs_dir=obs_dir,
                                    data_root=self.input.raw_root(),
                                    output_dir=self.workspace / "survey" / obsid,
                                )
                                result.update(
                                    {
                                        key: value
                                        for key, value in dict(loaded).items()
                                        if key not in {"obsid", "result_dir"}
                                    }
                                )
                        except Exception as exc:
                            warnings.append(
                                f"{obsid}:existing_cache_load_failed:{type(exc).__name__}:{exc}"
                            )
                else:
                    if backend is None:
                        backend = BatAnalysisSurveyBackend()
                        self.backend = backend
                    with self.stage_environment(
                        f"survey_{safe_filename_token(str(obsid))}"
                    ) as environment:
                        with _temporary_environment(environment), _math_thread_limit(
                            self.survey_config.internal_threads
                        ):
                            result = _call_supported(
                                backend.process_observation,
                                obsid=obsid,
                                obs_dir=obs_dir,
                                output_dir=self.workspace / "survey" / obsid,
                                source_name=self.input.source_name or self.input.target_id,
                                catalog_path=self._catalog(),
                                survey=self.survey_config,
                                recalc=True,
                                task_timeout_s=self.survey_config.task_timeout_s,
                                processes=self.survey_config.processes,
                                internal_threads=self.survey_config.internal_threads,
                            )
                processed.append(
                    {
                        **row,
                        "status": "completed",
                        **dict(_json_safe_external(result)),
                    }
                )
            except Exception as exc:
                warnings.append(f"{obsid}:survey_failed:{type(exc).__name__}:{exc}")
                processed.append({**row, "status": "failed", "error": str(exc)})
        path = self._output("survey.json")
        payload = {
            "records": processed,
            "warnings": warnings,
            "detthresh": self.survey_config.detthresh,
            "detthresh2": self.survey_config.detthresh2,
            "min_pcode": self.survey_config.min_pcode,
            "energy_range_keV": list(self.survey_config.energy_range_keV),
            "cache_reusable": not warnings and all(
                str(row.get("status", "")).lower()
                not in {"failed", "not_ready"}
                for row in processed
            ),
        }
        write_json(path, payload)
        return StageResult(outputs={"survey": str(path)}, data=payload)

    def _rate_files(self, records: Sequence[Mapping[str, Any]]) -> list[Path]:
        paths: list[Path] = []
        for row in records:
            directory = Path(str(row.get("result_dir") or row.get("survey_dir") or ""))
            if directory.is_file():
                if _has_suffix(directory, (".fits", ".fit", ".cat", ".csv", ".dat", ".qdp")):
                    paths.append(directory)
            elif directory.is_dir():
                paths.extend(
                    path
                    for path in directory.rglob("*")
                    if _has_suffix(path, (".fits", ".fit", ".cat", ".csv", ".dat", ".qdp"))
                    and "pha" not in path.name.lower()
                )
        survey_root = self.input.survey_root()
        if survey_root is not None:
            if survey_root.is_file() and _has_suffix(
                survey_root, (".fits", ".fit", ".cat", ".csv", ".dat", ".qdp")
            ):
                paths.append(survey_root)
            elif survey_root.is_dir():
                paths.extend(
                    path
                    for path in survey_root.rglob("*")
                    if _has_suffix(path, (".fits", ".fit", ".cat", ".csv", ".dat", ".qdp"))
                )
        paths = sorted(set(paths))
        source_name = self.input.source_name or self.input.target_id
        preferred_token = safe_filename_token(source_name).lower()
        preferred = [
            path
            for path in paths
            if preferred_token and preferred_token in path.stem.lower()
        ]
        # BatAnalysis writes one merged CAT per source.  Prefer it when
        # present so a large observation result tree is not repeatedly opened;
        # fall back to the complete tree for legacy layouts.
        return preferred or paths

    @staticmethod
    def _gti_intervals_for_records(
        records: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str | None, tuple[tuple[float, float], ...]], tuple[str, ...]]:
        """Collect per-pointing GTIs from raw or BatAnalysis result trees.

        Survey CAT rows provide the complete pointing interval and live time,
        while ``*_pnt.gti`` files provide the quality-filtered GTI intersection.
        The latter is used only to decide whether a requested window is
        covered; the selected row's complete exposure is never rescaled by the
        short-window overlap.
        """
        intervals: dict[str | None, tuple[tuple[float, float], ...]] = {}
        sources: list[str] = []
        seen: set[Path] = set()
        for row in records:
            roots = []
            for key in ("result_dir", "survey_dir", "obs_dir"):
                value = row.get(key)
                if value:
                    path = Path(str(value)).expanduser().resolve()
                    if path.is_file():
                        path = path.parent
                    if path.is_dir():
                        roots.append(path)
            for root in roots:
                for path in sorted(
                    item
                    for item in root.rglob("*")
                    if item.is_file()
                    and (
                        item.name.lower().endswith(".gti")
                        or item.name.lower().endswith(".gti.gz")
                    )
                ):
                    if path in seen:
                        continue
                    seen.add(path)
                    try:
                        parsed = read_gti_intervals(path)
                    except (OSError, ValueError, TypeError):
                        continue
                    if not parsed:
                        continue
                    parent = path.parent.name
                    # Pointing GTIs are named ``point_<id>_pnt*.gti``.  Keep
                    # the parent key because CAT rows use the same
                    # ``point_<id>`` identifier; an observation-level GTI is
                    # retained under ``None`` only when no pointing key is
                    # available.
                    key: str | None = parent if parent.lower().startswith("point_") else None
                    if key is not None and key not in intervals:
                        intervals[key] = parsed
                    elif key is None and None not in intervals:
                        intervals[None] = parsed
                    sources.append(str(path))
        return intervals, tuple(sources)

    def _stage_lightcurve(self, context: Mapping[str, StageResult]) -> StageResult:
        records = context["survey"].data.get("records", [])
        gti_intervals, gti_sources = self._gti_intervals_for_records(records)
        points: list[SurveyRatePoint] = []
        read_errors: list[str] = []
        for path in self._rate_files(records):
            try:
                points.extend(
                    read_bat_survey_rates(
                        path,
                        source_name=self.input.source_name or self.input.target_id,
                    )
                )
            except (OSError, ValueError) as exc:
                read_errors.append(f"{path}:{type(exc).__name__}:{exc}")
                continue
        # A result tree may contain both per-pointing CAT files and the
        # merged source CAT.  Keep one measurement per pointing so a second
        # file cannot silently double the light curve.
        unique_points: dict[tuple[Any, ...], SurveyRatePoint] = {}
        for point in points:
            key = (
                point.obsid,
                point.pointing_id,
                point.source,
                point.time_start,
                point.time_stop,
            )
            if key not in unique_points or "merged_pointings_lc" in point.source_file:
                unique_points[key] = point
        points = list(unique_points.values())
        self._points = points
        finite_rates = [
            point.rate
            for point in points
            if point.rate is not None and np.isfinite(point.rate)
        ]
        quality_warnings: list[str] = []
        if finite_rates and all(value == 0.0 for value in finite_rates):
            # Keep the signed values in the light curve, but surface an
            # explicit review marker instead of treating an all-zero survey
            # table as a successful non-detection.  This commonly indicates a
            # missing/failed batsurvey image or an empty catalogue row.
            quality_warnings.append("all_zero_survey_rates")
        if any(
            point.rate is not None
            and np.isfinite(point.rate)
            and (point.rate_error is None or not np.isfinite(point.rate_error) or point.rate_error <= 0)
            for point in points
        ):
            quality_warnings.append("invalid_survey_rate_errors")
        quality_dirs = []
        for row in records:
            value = row.get("result_dir") or row.get("survey_dir")
            if value:
                quality_dirs.append(Path(str(value)))
        status_files_present = any(
            path.is_dir() and any(path.rglob("*_status.txt"))
            for path in quality_dirs
        )
        if points and all(point.quality is None for point in points) and not status_files_present:
            quality_warnings.append("missing_survey_quality_records")
        selected: list[dict[str, Any]] = []
        for start, stop in self.input.time_windows:
            try:
                selected_points = select_overlapping_pointings(
                    points,
                    start,
                    stop,
                    min_pcode=self.survey_config.min_pcode,
                    strict_pcode=True,
                    gti_intervals=gti_intervals or None,
                )
                for item in selected_points:
                    snr = item.get("point", {}).get("snr")
                    item["detected_snr3"] = bool(
                        snr is not None and snr >= self.survey_config.source_snr_threshold
                    )
                    item["detected_snr5"] = bool(
                        snr is not None and snr >= self.survey_config.strong_snr_threshold
                    )
                selected.extend(selected_points)
            except Exception as exc:
                selected.append({
                    "selection_error": str(exc),
                    "requested_start": start,
                    "requested_stop": stop,
                })
        if self.input.time_windows and not any("point" in item for item in selected):
            quality_warnings.append("no_valid_exposure_for_requested_windows")
        payload = {
            "points": [point.to_payload() for point in points],
            "selected": selected,
            "product_kind": "bat_survey_pointing_rate",
            "rate_unit": "count/s/fully_illuminated_detector",
            "source_rate_units": sorted({
                str(point.rate_unit) for point in points if point.rate_unit
            }),
            "detector_normalization": "fully_illuminated_detector",
            "rate_semantics": "background_subtracted_survey_rate; not an energy flux",
            "energy_band_keV": list(self.survey_config.energy_range_keV),
            "selection_min_pcode": self.survey_config.min_pcode,
            "detection_thresholds_snr": {
                "survey_source": self.survey_config.source_snr_threshold,
                "survey_strong": self.survey_config.strong_snr_threshold,
            },
            "time_format": "swift",
            "time_reference": "Swift mission elapsed time; requested windows are UTC",
            "gti_sources": list(gti_sources),
            "gti_overlap_policy": (
                "select only requested windows with positive per-pointing GTI overlap; "
                "retain complete survey exposure without overlap scaling"
            ),
            "snr_labels": {
                "3sigma": self.survey_config.source_snr_threshold,
                "5sigma": self.survey_config.strong_snr_threshold,
                "interpretation": "known-position measurement labels, not blind-search discovery significance",
            },
            "native_channel_contract": {
                "energy_edges_keV": list(self.survey_config.channel_edges_keV),
                "native_bands": 8,
                "total_band_is_last": True,
                "duplicate_total_columns_dropped": True,
            },
            "quality_status": "needs_review" if quality_warnings else "ok",
            "warnings": (
                ([] if points else ["no_survey_rate_products"])
                + quality_warnings
                + read_errors
            ),
        }
        payload["cache_reusable"] = not payload["warnings"]
        path = self._output("survey_lightcurve.json")
        write_json(path, payload)
        return StageResult(outputs={"lightcurve": str(path)}, data=payload)

    def _stage_mosaic(self, context: Mapping[str, StageResult]) -> StageResult:
        enabled = bool(
            self.input.mosaic
            or self.input.mosaic_products_dir is not None
            or (self.config.mosaic and self.config.mosaic.enabled)
        )
        if not enabled:
            payload = {"enabled": False, "status": "skipped", "warnings": ["mosaic_not_requested"]}
        elif self.input.mosaic_products_dir and _has_product(Path(self.input.mosaic_products_dir)):
            root = Path(self.input.mosaic_products_dir)
            product_paths = (
                [root]
                if root.is_file()
                else [path for path in root.rglob("*") if path.is_file()]
            )
            measurement_products = [
                path for path in product_paths
                if path.is_dir() or _is_readable_survey_product(path)
            ]
            payload = {
                "enabled": True,
                "status": "existing_products",
                "products": [str(path) for path in product_paths],
                "mosaic_measurements": [
                    measurement
                    for product in measurement_products
                    for measurement in _mosaic_source_measurements(
                        product,
                        source_name=self.input.source_name or self.input.target_id,
                        detection_threshold=(
                            self.config.mosaic.detection_threshold
                            if self.config.mosaic is not None
                            else 3.0
                        ),
                    )
                ],
                "product_kind": "bat_survey_mosaic",
                "rate_semantics": (
                    "mosaic rates are backend products and are not combined with "
                    "pointing survey rates"
                ),
            }
        else:
            try:
                backend = (
                    self.backend
                    if self.backend is not None
                    else BatAnalysisSurveyBackend()
                )
                self.backend = backend
                if not hasattr(backend, "process_mosaic"):
                    raise RuntimeError("configured BAT survey backend has no mosaic adapter")
                # A resumed run restores the JSON survey manifest but not the
                # in-memory BatSurvey objects held by the optional adapter.
                # Rehydrate those objects before mosaicing; otherwise a
                # previously successful survey stage would appear to have no
                # exposure after a process restart.  Result-only CAT inputs
                # without a loadable BatAnalysis cache remain a documented
                # ``no_valid_exposure`` case rather than being fabricated into
                # a mosaic.
                restore_warnings: list[str] = []
                load_observation = getattr(backend, "load_observation", None)
                objects = getattr(backend, "objects", None)
                if callable(load_observation) and isinstance(objects, dict) and not objects:
                    survey_records = context.get("survey", StageResult()).data.get("records", ())
                    for row in survey_records:
                        if str(row.get("status", "")).lower() not in {
                            "completed",
                            "existing_products",
                        }:
                            continue
                        obsid = str(row.get("obsid") or "").strip()
                        result_dir = Path(
                            str(row.get("result_dir") or row.get("survey_dir") or "")
                        )
                        if not obsid or not result_dir.is_dir():
                            continue
                        try:
                            _call_supported(
                                load_observation,
                                obsid=obsid,
                                result_dir=result_dir,
                                source_name=self.input.source_name or self.input.target_id,
                                obs_dir=Path(str(row.get("obs_dir") or self.input.raw_root() / obsid)),
                                data_root=self.input.raw_root(),
                                output_dir=self.workspace / "mosaic" / "cache" / obsid,
                            )
                        except Exception as exc:
                            restore_warnings.append(
                                f"mosaic_cache_load_failed:{obsid}:{type(exc).__name__}:{exc}"
                            )
                mosaic_config = self.config.mosaic
                mosaic_gti_intervals, mosaic_gti_sources = self._gti_intervals_for_records(
                    context.get("survey", StageResult()).data.get("records", ())
                )
                with self.stage_environment("mosaic") as environment:
                    with _temporary_environment(environment), _math_thread_limit(
                        self.survey_config.internal_threads
                    ):
                        payload = _call_supported(
                            backend.process_mosaic,
                            windows=self.input.time_windows,
                            output_dir=self.workspace / "mosaic",
                            source_name=self.input.source_name or self.input.target_id,
                            catalog_path=self._catalog(),
                            gti_intervals=(mosaic_gti_intervals or None),
                            min_pcode=(
                                mosaic_config.min_pcode
                                if mosaic_config is not None
                                else 0.15
                            ),
                            overlap_selection=(
                                mosaic_config.overlap_selection
                                if mosaic_config is not None
                                else True
                            ),
                            detection_threshold=(
                                mosaic_config.detection_threshold
                                if mosaic_config is not None
                                else 3.0
                            ),
                            processes=self.survey_config.processes,
                            internal_threads=self.survey_config.internal_threads,
                            task_timeout_s=self.survey_config.task_timeout_s,
                        )
                payload = {
                    "enabled": True,
                    **dict(payload),
                    "product_kind": "bat_survey_mosaic",
                    "rate_semantics": (
                        "mosaic rates are backend products and are not combined with "
                        "pointing survey rates"
                    ),
                    "gti_sources": list(mosaic_gti_sources),
                }
                if restore_warnings:
                    payload["warnings"] = list(payload.get("warnings", ())) + restore_warnings
            except Exception as exc:
                payload = {
                    "enabled": True,
                    "status": "failed",
                    "product_kind": "bat_survey_mosaic",
                    "warnings": [f"mosaic_failed:{type(exc).__name__}:{exc}"],
                }
        payload = dict(_json_safe_external(payload))
        measurements = payload.get("mosaic_measurements", ())
        if isinstance(measurements, list):
            invalid_measurements = [
                item for item in measurements
                if isinstance(item, Mapping)
                and str(item.get("status", "")) == "invalid_measurement"
            ]
            if invalid_measurements:
                payload.setdefault("warnings", [])
                payload["warnings"] = list(payload["warnings"]) + [
                    "mosaic_invalid_source_measurement"
                ]
        payload.setdefault(
            "detection_basis",
            "mosaic_source_catalog_snr",
        )
        if "cache_reusable" not in payload:
            payload["cache_reusable"] = bool(
                payload.get("status") in {"skipped", "existing_products", "completed"}
                and not payload.get("warnings")
            )
        path = self._output("mosaic.json")
        write_json(path, payload)
        return StageResult(outputs={"mosaic": str(path)}, data=payload)

    def _pha_files(
        self,
        records: Sequence[Mapping[str, Any]],
        extra_paths: Sequence[str | Path] = (),
        selected: Sequence[Mapping[str, Any]] = (),
    ) -> list[Path]:
        """Find science PHA products belonging to this source and window.

        BatAnalysis result trees can contain one PHA per catalogue source,
        multiple pointings and historical manually constructed upper-limit
        spectra.  The old recursive glob silently fitted all of them.  Apply
        only positive scope evidence here: explicit source/pointing/OBSID
        identifiers that disagree are rejected, and an explicit PHA time
        interval must overlap a requested window.  Products with no optional
        scope metadata remain eligible for backwards compatibility and are
        still required to pass the FITS/response validation below.
        """
        paths: list[Path] = []
        directories = [
            Path(str(row.get("result_dir") or row.get("survey_dir") or ""))
            for row in records
        ] + [Path(value).expanduser().resolve() for value in extra_paths]
        for directory in directories:
            if directory.is_file():
                if _has_suffix(directory, (".pha",)):
                    paths.append(directory)
            elif directory.is_dir():
                paths.extend(
                    path
                    for path in directory.rglob("*")
                    if _has_suffix(path, (".pha",))
                )
        candidates = sorted(set(path.resolve() for path in paths if path.is_file()))
        requested_source = self.input.source_name or self.input.target_id
        allowed_obsids = {
            str(row.get("obsid"))
            for row in records
            if row.get("obsid") not in (None, "", "existing")
        }
        # A selected light-curve row is the strongest scope link.  If no row
        # is available (for example an inventory-only run without windows), do
        # not discard otherwise valid local products merely because their
        # optional pointing keywords are absent.
        selected_pointings: set[str] = set()
        selected_obsids: set[str] = set()
        for item in selected:
            point = item.get("point") if isinstance(item, Mapping) else None
            if not isinstance(point, Mapping):
                continue
            if point.get("pointing_id") not in (None, ""):
                selected_pointings.add(str(point["pointing_id"]))
            if point.get("obsid") not in (None, "", "existing"):
                selected_obsids.add(str(point["obsid"]))
        window_intervals = tuple(
            (_time_value(start), _time_value(stop))
            for start, stop in self.input.time_windows
        )
        exclusions: list[dict[str, Any]] = []
        accepted: list[Path] = []
        for path in candidates:
            reason: str | None = None
            if _PHA_MANUAL_UPPER_LIMIT_RE.search(path.name):
                reason = "manual_or_legacy_upper_limit_product"
            else:
                metadata = _pha_scope_metadata(path)
                source_values = tuple(
                    value for value in metadata["source"]
                    if value.upper() not in _PHA_SCOPE_GENERIC_NAMES
                )
                if source_values and not any(
                    _source_name_matches(value, requested_source)
                    for value in source_values
                ):
                    reason = f"source_mismatch:{','.join(source_values)}"
                elif allowed_obsids and metadata["obsid"] and not _scope_identifier_matches(
                    metadata["obsid"], allowed_obsids
                ):
                    reason = f"obsid_mismatch:{','.join(metadata['obsid'])}"
                elif selected_obsids and metadata["obsid"] and not _scope_identifier_matches(
                    metadata["obsid"], selected_obsids
                ):
                    reason = f"selected_obsid_mismatch:{','.join(metadata['obsid'])}"
                elif selected_pointings and metadata["pointing"] and not _scope_identifier_matches(
                    metadata["pointing"], selected_pointings
                ):
                    reason = f"pointing_outside_selection:{','.join(metadata['pointing'])}"
                elif window_intervals:
                    interval = _pha_time_interval(path)
                    if interval is not None and not any(
                        interval[0] < stop and interval[1] > start
                        for start, stop in window_intervals
                    ):
                        reason = "pha_outside_requested_windows"
            if reason is None:
                accepted.append(path)
            else:
                exclusions.append({"path": str(path), "reason": reason})
        self._pha_exclusions = exclusions
        return accepted

    def _stage_spectra(self, context: Mapping[str, StageResult]) -> StageResult:
        mosaic_products = context.get("mosaic", StageResult()).data.get("products", ())
        validations = [
            validate_survey_pha(path, require_matrix=True)
            for path in self._pha_files(
                context["survey"].data.get("records", []),
                extra_paths=mosaic_products,
                selected=context.get("lightcurve", StageResult()).data.get("selected", ()),
            )
        ]
        warnings = [
            f"{item.path.name}:{','.join(item.diagnostics)}"
            for item in validations
            if not item.valid
        ]
        if not validations:
            warnings.append("no_survey_pha_products")
        # Exclusions are recorded in the product index for provenance.  A
        # catalogue directory normally contains other sources and historical
        # hand-built upper-limit PHAs; filtering those products is expected
        # selection work, not a failed science stage.  The absence of any
        # usable PHA is still promoted by ``no_survey_pha_products`` above.
        payload = {
            "spectra": [
                {
                    "path": str(item.path),
                    "valid": item.valid,
                    "channels": item.channels,
                    "exposure_s": item.exposure_s,
                    "backscal": item.backscal,
                    "response": str(item.response) if item.response else None,
                    "ancillary_response": (
                        str(item.ancillary_response) if item.ancillary_response else None
                    ),
                    "background": str(item.background) if item.background else None,
                    "has_stat_err": "STAT_ERR" in item.columns,
                    "has_sys_err": "SYS_ERR" in item.columns,
                    "columns": list(item.columns),
                    "diagnostics": list(item.diagnostics),
                }
                for item in validations
            ],
            "warnings": warnings,
            "excluded_products": list(self._pha_exclusions),
            "cache_reusable": not warnings,
        }
        path = self._output("spectra.json")
        write_json(path, payload)
        return StageResult(outputs={"spectra": str(path)}, data=payload)

    def _stage_fit(self, context: Mapping[str, StageResult]) -> StageResult:
        if not self._points and "lightcurve" in context:
            self._points = [
                SurveyRatePoint(**payload)
                for payload in context["lightcurve"].data.get("points", ())
                if isinstance(payload, Mapping)
            ]
        results: list[dict[str, Any]] = []
        warnings: list[str] = []
        if not self.config.fitting.enabled:
            payload = {
                "results": [],
                "warnings": ["fit_disabled_by_config"],
                "statistic": self.config.fitting.statistic,
                "model": self.config.fitting.model_name,
                "cache_reusable": True,
                "observed_upper_bound": _unavailable_bat_observed(
                    reason="BAT fitting is disabled by configuration.",
                    energy_band_keV=self.survey_config.energy_range_keV,
                    model_name=self.config.fitting.model_name or "cflux*powerlaw",
                ),
                "detection_sensitivity": _unavailable_bat_sensitivity(
                    reason="BAT fitting is disabled by configuration.",
                    policy=self.config.upper_limit,
                ),
                "background": [_bat_background_payload()],
            }
            path = self._output("fits.json")
            write_json(path, payload)
            return StageResult(outputs={"fit": str(path)}, data=payload)
        configured_fit_energy = self.config.spectrum.fit_energy_range_keV
        fit_energy = tuple(
            float(value)
            for value in (
                configured_fit_energy or self.survey_config.energy_range_keV
            )
        )
        if fit_energy != tuple(float(value) for value in self.survey_config.energy_range_keV):
            # The survey product is calibrated for the survey range.  Keep a
            # user override visible, but do not silently relabel the product.
            warnings.append(
                "fit_energy_range_differs_from_survey_product:"
                f"{fit_energy}!={self.survey_config.energy_range_keV}"
            )
        model_name = self.config.fitting.model_name or "cflux*powerlaw"
        stat_method = self.config.fitting.statistic or "chi"
        error_delta_stat = float(self.config.fitting.error_delta_stat)
        group_min = self.config.spectrum.group_min_counts or 1
        upper_policy = self.config.upper_limit
        upper_photon_index = float(upper_policy.spectral_index)
        upper_delta_stat = float(upper_policy.default_sigma) ** 2
        sensitivity_controls = (
            Path(self.input.sensitivity_control_path)
            if self.input.sensitivity_control_path is not None
            else None
        )
        for item in context["spectra"].data.get("spectra", []):
            if not item.get("valid"):
                warnings.append(
                    f"fit_skipped:{item.get('path')}:{','.join(item.get('diagnostics', []))}"
                )
                continue
            pha = Path(item["path"])
            response = Path(item["response"])
            ancillary = (
                Path(item["ancillary_response"])
                if item.get("ancillary_response")
                else None
            )
            arf = response if _response_is_arf(response) else ancillary
            rmf = response if not _response_is_arf(response) else ancillary
            prepared = PreparedSpectrum(
                instrument="BATSurvey",
                obsid=None,
                module=None,
                detector="BAT",
                source_id=self.input.source_name or self.input.target_id,
                source_pha=pha,
                grouped_pha=pha,
                background_pha=None,
                arf=arf,
                rmf=rmf,
                group_min=int(group_min),
                energy_range_keV=self.survey_config.energy_range_keV,
            )
            try:
                pha_interval = _pha_time_interval(pha)
                matching = [
                    point
                    for point in self._points
                    if (
                        point.pointing_id
                        and point.pointing_id in pha.name
                    )
                    or (
                        pha_interval is not None
                        and point.time_start is not None
                        and point.time_stop is not None
                        and point.time_start < pha_interval[1]
                        and point.time_stop > pha_interval[0]
                    )
                ]
                if not matching:
                    warnings.append(f"fit_deferred_no_rate_match:{pha}")
                    results.append({
                        "pha": str(pha),
                        "status": "fit_deferred_no_rate_match",
                        "observed_upper_bound": _unavailable_bat_observed(
                            reason="No matching survey rate row was found for this PHA.",
                            energy_band_keV=self.survey_config.energy_range_keV,
                            model_name=model_name,
                            delta_stat=upper_delta_stat,
                        ),
                        "detection_sensitivity": _unavailable_bat_sensitivity(
                            reason="No matching survey rate row was found for this PHA.",
                            policy=self.config.upper_limit,
                        ),
                    })
                    continue
                detected = any(
                    point.snr is not None
                    and point.snr >= self.survey_config.source_snr_threshold
                    for point in matching
                )
                if detected:
                    result = fit_prepared(
                        prepared,
                        outdir=self._output("fits") / safe_filename_token(pha.stem),
                        emin=fit_energy[0],
                        emax=fit_energy[1],
                        model_name=model_name,
                        stat_method=stat_method,
                        calculate_errors=self.config.fitting.calculate_errors,
                        error_delta_stat=error_delta_stat,
                        plot_formats=(
                            self.config.plotting.formats
                            if self.config.plotting.enabled
                            else ()
                        ),
                        plot_density=self.config.plotting.dpi,
                        plot_required=(
                            self.config.plotting.required
                            and self.config.plotting.enabled
                        ),
                    )
                    profile_result = profile_survey_upper_limit(
                        pha,
                        response,
                        ancillary_response_path=ancillary,
                        photon_index=upper_photon_index,
                        delta_stat=upper_delta_stat,
                        energy_range_keV=self.survey_config.energy_range_keV,
                        output_dir=self._output("upper_limits") / safe_filename_token(pha.stem),
                    )
                    results.append({
                        "pha": str(pha),
                        "status": "fit_complete",
                        "result": result,
                        "observed_upper_bound": _bat_observed_payload(
                            profile_result,
                            model_name=model_name,
                        ),
                        "detection_sensitivity": _bat_sensitivity_for_profile(
                            profile_result,
                            sensitivity_controls,
                            policy=self.config.upper_limit,
                        ),
                    })
                else:
                    if not self.config.upper_limit.enabled:
                        results.append({
                            "pha": str(pha),
                            "status": "upper_limit_disabled",
                            "observed_upper_bound": _unavailable_bat_observed(
                                reason="BAT upper-limit calculation is disabled by configuration.",
                                energy_band_keV=self.survey_config.energy_range_keV,
                                model_name=model_name,
                                delta_stat=upper_delta_stat,
                            ),
                            "detection_sensitivity": _unavailable_bat_sensitivity(
                                reason="BAT upper-limit calculation is disabled by configuration.",
                                policy=self.config.upper_limit,
                            ),
                        })
                    else:
                        result = profile_survey_upper_limit(
                            pha,
                            response,
                            ancillary_response_path=ancillary,
                            photon_index=upper_photon_index,
                            delta_stat=upper_delta_stat,
                            energy_range_keV=self.survey_config.energy_range_keV,
                            output_dir=self._output("upper_limits") / safe_filename_token(pha.stem),
                        )
                        results.append({
                            "pha": str(pha),
                            "status": result.status,
                            "result": result,
                            "observed_upper_bound": _bat_observed_payload(
                                result,
                                model_name=model_name,
                            ),
                            "detection_sensitivity": _bat_sensitivity_for_profile(
                                result,
                                sensitivity_controls,
                                policy=self.config.upper_limit,
                            ),
                        })
            except Exception as exc:
                warnings.append(f"fit_failed:{pha}:{type(exc).__name__}:{exc}")
                results.append({
                    "pha": str(pha),
                    "status": "fit_failed",
                    "error": str(exc),
                    "observed_upper_bound": _unavailable_bat_observed(
                        reason=f"BAT fit failed: {type(exc).__name__}: {exc}",
                        energy_band_keV=self.survey_config.energy_range_keV,
                        model_name=model_name,
                        delta_stat=upper_delta_stat,
                    ),
                    "detection_sensitivity": _unavailable_bat_sensitivity(
                        reason=f"BAT fit failed: {type(exc).__name__}: {exc}",
                        policy=self.config.upper_limit,
                    ),
                })
        if not results:
            warnings.append("no_fit_or_upper_limit_products")
        # Keep the fixed background contract at the same level as each fit
        # record.  ``observed_upper_bound`` already carries the block for
        # successful and unavailable branches; promote it here so consumers
        # do not have to know which result mode was produced.
        for item in results:
            if "background" in item:
                continue
            observed_payload = item.get("observed_upper_bound")
            if isinstance(observed_payload, Mapping) and isinstance(
                observed_payload.get("background"), Mapping
            ):
                item["background"] = observed_payload["background"]
            else:
                item["background"] = _bat_background_payload(pha=item.get("pha"))
        for result in results:
            if result.get("status") not in {"fit_complete", "upper_limit_ready"}:
                warnings.append(
                    f"fit_status:{result.get('pha')}:{result.get('status')}"
                )
        payload = {
            "results": results,
            "warnings": warnings,
            "statistic": stat_method,
            "model": model_name,
            "energy_range_keV": list(fit_energy),
            "group_min": int(group_min),
            "error_delta_stat": error_delta_stat,
            "background": [
                item.get("background", _bat_background_payload(pha=item.get("pha")))
                for item in results
            ],
            "observed_upper_bound": (
                results[0].get("observed_upper_bound") if results else _unavailable_bat_observed(
                    reason="No valid BAT PHA fit was produced.",
                    energy_band_keV=self.survey_config.energy_range_keV,
                    model_name=model_name,
                    delta_stat=upper_delta_stat,
                )
            ),
            "detection_sensitivity": (
                results[0].get("detection_sensitivity") if results else _unavailable_bat_sensitivity(
                    reason="No valid BAT PHA fit was produced.",
                    policy=self.config.upper_limit,
                )
            ),
            "cache_reusable": not warnings,
        }
        path = self._output("fits.json")
        write_json(path, payload)
        return StageResult(outputs={"fit": str(path)}, data=payload)

    def _stage_report(self, context: Mapping[str, StageResult]) -> StageResult:
        warnings = [
            warning
            for result in context.values()
            for warning in result.data.get("warnings", [])
            if warning not in {"download_not_requested", "mosaic_not_requested"}
        ]
        mosaic_data = context.get("mosaic", StageResult()).data
        if mosaic_data.get("enabled") and mosaic_data.get("status") not in {
            "completed", "existing_products", "skipped",
        }:
            status = mosaic_data.get("status", "unknown")
            marker = f"mosaic_status:{status}"
            if marker not in warnings:
                warnings.append(marker)
        missing = [
            f"{stage}:{result.message or 'no product'}"
            for stage, result in context.items()
            if result.status != PipelineStatus.COMPLETED
        ]
        # Most stages deliberately complete with a review marker in their
        # JSON payload so one bad OBSID does not abort the target.  Promote
        # the concrete absence/failure markers to the report's missing list;
        # callers can then distinguish an empty product from a successfully
        # completed stage without parsing free-form warning text.
        missing_prefixes = (
            "no_", "missing_", "fit_status:", "mosaic_status:",
        )
        for stage, result in context.items():
            for warning in result.data.get("warnings", ()):
                text = str(warning)
                if text in {"download_not_requested", "mosaic_not_requested"}:
                    continue
                if text.startswith(missing_prefixes) or any(
                    marker in text
                    for marker in (
                        ":not_ready", ":survey_failed", ":fit_failed",
                        ":existing_cache_load_failed", "_failed:", "query_failed:",
                        "download_incomplete:",
                    )
                ):
                    marker = f"{stage}:{text}"
                    if marker not in missing:
                        missing.append(marker)
        science_status = "complete" if not warnings and not missing else "partial"
        quality_status = "needs_review" if warnings else "ok"
        survey_records = context.get("survey", StageResult()).data.get("records", ())
        discovered_records = context.get("discover", StageResult()).data.get("records", ())
        reported_obsids = sorted(
            {
                str(row.get("obsid"))
                for row in (*discovered_records, *survey_records)
                if isinstance(row, Mapping)
                and row.get("obsid") not in (None, "", "existing")
            }
        )
        if not reported_obsids:
            reported_obsids = list(self.input.obsids)
        lightcurve_data = context.get("lightcurve", StageResult()).data
        spectra_data = context.get("spectra", StageResult()).data
        fit_data = context.get("fit", StageResult()).data
        fit_results = list(_json_safe_external(fit_data.get("results", ())))
        observed_bounds = [
            item.get("observed_upper_bound")
            for item in fit_results
            if isinstance(item, Mapping)
            and isinstance(item.get("observed_upper_bound"), Mapping)
            and item.get("observed_upper_bound", {}).get("value") is not None
        ]
        sensitivities = [
            item.get("detection_sensitivity")
            for item in fit_results
            if isinstance(item, Mapping) and item.get("detection_sensitivity") is not None
        ]
        mosaic_summary = _json_safe_external(
            {
                "status": mosaic_data.get("status"),
                "products": mosaic_data.get("products", ()),
                "windows": mosaic_data.get("windows", ()),
                "shared_members": mosaic_data.get("shared_members", ()),
                "mosaic_measurements": mosaic_data.get("mosaic_measurements", ()),
                "detection_basis": mosaic_data.get(
                    "detection_basis", "mosaic_source_catalog_snr"
                ),
            }
        )
        payload = {
            "target_id": self.input.target_id,
            "source_name": self.input.source_name or self.input.target_id,
            "coordinates_icrs_deg": (
                {"ra": float(self.input.coord[0]), "dec": float(self.input.coord[1])}
                if self.input.coord is not None
                else None
            ),
            "obsids": reported_obsids,
            "science_status": science_status,
            "quality_status": quality_status,
            "requested_time_windows": self.input.time_windows,
            "energy_band_keV": list(self.survey_config.energy_range_keV),
            "survey_config": {
                "detthresh": self.survey_config.detthresh,
                "detthresh2": self.survey_config.detthresh2,
                "min_pcode": self.survey_config.min_pcode,
                "source_snr_threshold": self.survey_config.source_snr_threshold,
                "strong_snr_threshold": self.survey_config.strong_snr_threshold,
                "upper_limit_photon_index": self.survey_config.upper_limit_photon_index,
                "upper_limit_delta_stat": self.survey_config.upper_limit_delta_stat,
                "processes": self.survey_config.processes,
                "internal_threads": self.survey_config.internal_threads,
                "task_timeout_s": self.survey_config.task_timeout_s,
            },
            "upper_limit_config": {
                "spectral_index": float(self.config.upper_limit.spectral_index),
                "confidence_sigma": float(self.config.upper_limit.default_sigma),
                "delta_stat": float(self.config.upper_limit.default_sigma) ** 2,
                "false_alarm_probability": self.config.upper_limit.detection_false_alarm_probability,
                "target_power": float(self.config.upper_limit.detection_power),
            },
            "rate_unit": "count/s/fully_illuminated_detector",
            "detector_normalization": "fully_illuminated_detector",
            "lightcurve": {
                "points": _json_safe_external(lightcurve_data.get("points", ())),
                "selected": _json_safe_external(lightcurve_data.get("selected", ())),
                "gti_sources": list(lightcurve_data.get("gti_sources", ())),
                "time_format": lightcurve_data.get("time_format", "swift"),
            },
            "spectra": _json_safe_external(spectra_data.get("spectra", ())),
            "fit_results": fit_results,
            "observed_upper_bounds": observed_bounds,
            "detection_sensitivities": sensitivities,
            "observed_upper_bound": _json_safe_external(
                observed_bounds[0]
                if observed_bounds
                else _unavailable_bat_observed(
                    reason="No finite BAT observed profile was produced.",
                    energy_band_keV=self.survey_config.energy_range_keV,
                    model_name=fit_data.get("model", "cflux*powerlaw"),
                    delta_stat=float(self.config.upper_limit.default_sigma) ** 2,
                )
            ),
            "detection_sensitivity": _json_safe_external(
                sensitivities[0]
                if sensitivities
                else _unavailable_bat_sensitivity(
                    reason="No BAT detection sensitivity calibration was produced.",
                    policy=self.config.upper_limit,
                )
            ),
            "background": _json_safe_external(
                fit_data.get("background", [item.get("background") for item in fit_results])
            ),
            "upper_limits": [
                item for item in fit_results
                if isinstance(item, Mapping)
                and item.get("status") == "upper_limit_ready"
            ],
            "mosaic": mosaic_summary,
            "warnings": warnings,
            "missing": missing,
            "cache_reusable": not warnings and not missing,
            "stages": {
                name: {"status": result.status.value, "outputs": result.outputs}
                for name, result in context.items()
            },
        }
        report = self._output("report.json")
        write_json(report, payload)
        text_path = self._output("report.txt")
        text_path.write_text(
            "BAT survey report\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        return StageResult(
            outputs={"report": str(report), "report_txt": str(text_path)},
            data=payload,
        )

    def build_result(self, context: Mapping[str, StageResult]) -> BATSurveyResult:
        report = context.get("report")
        payload = report.data if report else {}
        products = {
            name: {key: str(value) for key, value in result.outputs.items()}
            for name, result in context.items()
            if result.outputs
        }
        status = PipelineStatus.COMPLETED if report is not None else PipelineStatus.PENDING
        return BATSurveyResult(
            status=status,
            science_status=str(payload.get("science_status", "partial")),
            quality_status=str(payload.get("quality_status", "ok")),
            target_id=self.input.target_id,
            workspace=self.workspace,
            completed_stages=tuple(context),
            products=products,
            full_pipeline=report is not None,
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
            missing=tuple(str(item) for item in payload.get("missing", ())),
        )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Process one Swift/BAT survey target")
    parser.add_argument("target_id")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--raw-products", "--raw-products-dir", dest="raw_products", type=Path)
    parser.add_argument(
        "--survey-products",
        "--survey-products-dir",
        dest="survey_products",
        type=Path,
    )
    parser.add_argument(
        "--mosaic-products",
        "--mosaic-products-dir",
        dest="mosaic_products",
        type=Path,
    )
    parser.add_argument("--source-name")
    parser.add_argument("--catalog", type=Path, help="optional local BAT source catalog")
    parser.add_argument(
        "--sensitivity-controls",
        type=Path,
        help="local blank-sky BAT survey rate product for fixed-position sensitivity",
    )
    parser.add_argument("--ra", type=float)
    parser.add_argument("--dec", type=float)
    parser.add_argument("--obsid", action="append", default=[])
    parser.add_argument("--window", nargs=2, action="append", metavar=("START_UTC", "STOP_UTC"), default=[])
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--mosaic", action="store_true")
    parser.add_argument("--detthresh", type=int)
    parser.add_argument("--detthresh2", type=int)
    parser.add_argument("--min-pcode", type=float)
    parser.add_argument("--processes", type=int, help="backend worker processes (default: 1)")
    parser.add_argument(
        "--internal-threads",
        type=int,
        help="math threads per worker (default: 1)",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        help="external survey/mosaic task timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--network-timeout",
        type=float,
        help="HEASARC/download timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        help="maximum query/download attempts (default: 3)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        help="seconds between query/download retries (default: 5)",
    )
    parser.add_argument(
        "--query-margin",
        type=float,
        help="expand each online query window by this many seconds (default: 0)",
    )
    parser.add_argument("--profile", choices=("default", "lmjagn"), default="default")
    parser.add_argument("--until", choices=tuple(stage.name for stage in BATSurveyPipeline.stages))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    config_kwargs = {
        name: value
        for name, value in {
            "processes": args.processes,
            "internal_threads": args.internal_threads,
            "task_timeout_s": args.task_timeout,
        }.items()
        if value is not None
    }
    download_kwargs = {
        name: value
        for name, value in {
            "timeout_s": args.network_timeout,
            "retries": args.retries,
            "retry_wait_s": args.retry_wait,
            "query_margin_s": args.query_margin,
        }.items()
        if value is not None
    }
    if download_kwargs:
        config_kwargs["downloads"] = SwiftBATSurveyDownloadConfig(**download_kwargs)
    config = BATSurvey(profile=args.profile, **config_kwargs)
    input_data = BATSurveyInput(
        target_id=args.target_id,
        root=args.root,
        output_root=args.output,
        data_root=args.data_root,
        raw_products_dir=args.raw_products,
        survey_products_dir=args.survey_products,
        mosaic_products_dir=args.mosaic_products,
        source_name=args.source_name,
        coord=(args.ra, args.dec) if args.ra is not None and args.dec is not None else None,
        time_windows=tuple(tuple(item) for item in args.window),
        obsids=tuple(args.obsid),
        query=args.query,
        download=args.download,
        mosaic=args.mosaic,
        detthresh=args.detthresh,
        detthresh2=args.detthresh2,
        min_pcode=args.min_pcode,
        catalog_path=args.catalog,
        sensitivity_control_path=args.sensitivity_controls,
    )
    result = BATSurveyPipeline(input_data, config=config).run(
        until=args.until,
        resume=not args.no_resume,
    )
    print(json.dumps({
        "status": result.status.value,
        "science_status": result.science_status,
        "full_pipeline": result.full_pipeline,
        "report": result.products.get("report", {}),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
