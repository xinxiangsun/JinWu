"""Resumable BAT+XRT GRB sample pipeline for Swift.

Ported from the ``swift_highz_grb`` research workflow into the jinwu
instrument-pipeline framework. One pipeline instance processes one GRB
(``target_id``) inside a sample workspace that follows the research layout::

    <root>/                                # the research "output" workspace
    |   catalog/
    |   |   swift_grb_catalog_merged.csv   # merged SDC/UK_XRT/BAT_GRB catalog
    |   |   swift_bat_burst_durations.csv  # NASA BAT duration table
    |   raw/<GRB>/burst_analyser/parsed/   # UKSSDC Burst Analyser products
    `--- xrt_product_requests_maybe/downloads/<GRB>/{lc,spec}/

Stages: ``catalog`` then ``download`` then the parallel analysis branches
(``lightcurve``, ``prompt``, ``galactic_absorption``, ``bblocks``) then
``spectra``, ``fit`` and ``report``. All stages are manifest-backed and
resumable, mirroring ``jinwu.ep.wxt.pipeline.WXTPointingPipeline``.

Catalog- and data-derived names are passed through ``safe_filename_token``
and every workspace path is verified to stay inside the workspace
(``_contained_path``) before use. Text outputs are written through
``Path.write_text`` on contained paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import io
import inspect
import json as _json
import logging
import os
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from astropy.io import fits
from astropy.stats import bayesian_blocks
from astropy.time import Time, TimeDelta

from ...core.config import InstrumentConfig, SwiftGRB
from ...core.fit import fit_prepared
from ...core.galactic import resolve_galactic_absorption
from ...core.pipeline import (
    InstrumentPipeline,
    PipelineInput,
    PipelineStage,
    PipelineStatus,
    StageResult,
    register_pipeline,
)
from ...core.products import safe_filename_token, write_json
from ...core.spectrum_prep import PreparedSpectrum
from ...core.utils import li_ma_snr
from ...core.xselect import XSelectRunResult, extract_products_with_xselect
from ...ftools.grppha_hsp import grppha_hsp

logger = logging.getLogger(__name__)

__all__ = [
    "GRBRecord",
    "RedshiftCandidate",
    "SwiftBurstAnalyserFetcher",
    "SwiftGRBInput",
    "SwiftGRBPipeline",
    "SwiftGRBResult",
    "TimeResolvedSegment",
    "build_grb_record",
    "merge_low_snr_segments",
    "normalize_grb_name",
    "parse_bat_burst_durations",
    "parse_redshift_candidates",
    "preferred_redshift",
    "prompt_classification",
    "read_badbin_table",
    "read_burst_analyser_dat",
    "safe_extract_tar",
    "segment_from_payload",
]

Extractor = Callable[..., XSelectRunResult]
Fetcher = Callable[[str, Path], Mapping[str, Any]]
NhtotQuery = Callable[..., dict[str, Any]]

# Research workflow conventions (skill §2.2): spectroscopic absorption lines
# outrank emission lines, which outrank photometric and fuzzy estimates.
REDSHIFT_PRIORITY: dict[str, int] = {
    "manual_override": 0,
    "spectroscopic_absorption": 1,
    "spectroscopic_emission": 2,
    "spectroscopic": 3,
    "photometric": 4,
    "estimate_or_range": 5,
    "unspecified": 6,
}

_TRUE_STRINGS = {"true", "yes", "y", "1"}

# Burst Analyser DAT columns are positional; the first six (Time ±, Flux ±)
# are shared by every BAT/XRT product and are all the lightcurve needs.
_DAT_FLUX_COLUMNS = ("Time", "TimePos", "TimeNeg", "Flux", "FluxPos", "FluxNeg")

_RIGHT_EDGE_EPS_S = 1e-2  # include the right boundary event in a segment


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "INDEF", "NONE"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not np.isfinite(number):
        return None
    return number


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in _TRUE_STRINGS


def _contained_path(base: str | Path, *parts: Any) -> Path:
    """Join ``parts`` below ``base`` with sanitized names and verify containment."""
    base_resolved = Path(base).expanduser().resolve()
    sanitized = tuple(safe_filename_token(str(part)) for part in parts)
    candidate = base_resolved.joinpath(*sanitized).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ValueError(f"path escapes workspace base {base_resolved}: {candidate}")
    return candidate


def _manifest_path(value: Any) -> Path:
    """Resolve a path recorded in one of our own stage manifests."""
    return Path(str(value)).expanduser().resolve()


def safe_extract_tar(archive: str | Path, destination: str | Path) -> tuple[Path, ...]:
    """Safely expand an UKSSDC tar archive below ``destination``.

    Product downloads are external input.  Parent-path traversal and links are
    rejected before extraction, and the returned list records the verified
    files for the request manifest.
    """
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            path = (target / member.name).resolve()
            if not path.is_relative_to(target) or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(target, filter="data")
        for member in handle.getmembers():
            path = (target / member.name).resolve()
            if path.is_file():
                extracted.append(path)
    return tuple(extracted)


def normalize_grb_name(grb: str) -> str:
    """Normalize a GRB label to the catalog merge key (``GRB 050904A`` becomes ``050904A``)."""
    text = str(grb).upper().strip()
    text = text.replace("GRB", "").replace(" ", "")
    return text


@dataclass(frozen=True, slots=True)
class RedshiftCandidate:
    value: float
    method: str
    raw: str

    @property
    def priority(self) -> int:
        return REDSHIFT_PRIORITY.get(self.method, REDSHIFT_PRIORITY["unspecified"])


def _method_from_note(note: str) -> str:
    lowered = note.lower()
    if "absorption" in lowered:
        return "spectroscopic_absorption"
    if "emission" in lowered:
        return "spectroscopic_emission"
    if "photometric" in lowered:
        return "photometric"
    if "spectroscop" in lowered:
        return "spectroscopic"
    return "unspecified"


def parse_redshift_candidates(text: str) -> list[RedshiftCandidate]:
    """Parse ``RedshiftText`` entries such as ``6.29 (Subaru: absorption)``.

    Range-like entries (``6 < z < 8``) keep the first numeric as the estimate
    and are ranked as ``estimate_or_range``, mirroring the research audit.
    """
    candidates: list[RedshiftCandidate] = []
    for chunk in str(text or "").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.search(r"(\d+(?:\.\d+)?)", chunk)
        if match is None:
            continue
        method = "estimate_or_range" if re.search(r"[<>~]", chunk) else _method_from_note(chunk)
        candidates.append(RedshiftCandidate(value=float(match.group(1)), method=method, raw=chunk))
    return candidates


def preferred_redshift(row: Mapping[str, Any]) -> tuple[float | None, str, str, str]:
    """Return ``(z, method, source, note)`` from one merged-catalog row.

    Priority: research ``preferred_z`` columns (which already encode the
    manual overrides), then ``SDC_Redshift``, then ``RedshiftText`` parsing.
    """
    preferred = _to_float(row.get("preferred_z"))
    if preferred is not None:
        method = str(row.get("preferred_z_method") or "unspecified").strip() or "unspecified"
        source = str(row.get("preferred_z_source") or "merged_catalog").strip()
        note = str(row.get("preferred_z_note") or "").strip()
        return preferred, method, source, note
    sdc = _to_float(row.get("SDC_Redshift"))
    if sdc is not None:
        return sdc, "unspecified", "SDC_Redshift", ""
    candidates = parse_redshift_candidates(str(row.get("RedshiftText") or ""))
    if candidates:
        best = min(candidates, key=lambda candidate: (candidate.priority, abs(candidate.value)))
        return best.value, best.method, "RedshiftText", best.raw
    return None, "missing", "", ""


def parse_bat_burst_durations(text: str) -> list[dict[str, Any]]:
    """Parse the NASA ``summary_burst_durations.txt`` pipe table.

    Durations are seconds relative to the trigger; ``N/A`` becomes ``None``.
    """
    rows: list[dict[str, Any]] = []
    numeric = ("Trig_time_met", "T100_start", "T100_stop", "T90_start", "T90_stop",
               "T50_start", "T50_stop", "peak_1s_start", "peak_1s_stop")
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("\\") or line.startswith("|GRBname"):
            continue
        fields = [part.strip() for part in line.strip("|").split("|")]
        if len(fields) < 11 or fields[0] == "GRBname":
            continue
        row: dict[str, Any] = {"GRBname": fields[0], "Trig_ID": fields[1]}
        for name, raw in zip(numeric, fields[2:11]):
            row[name] = _to_float(raw)
        row["_merge_key"] = normalize_grb_name(fields[0])
        rows.append(row)
    return rows


def read_badbin_table(path: str | Path) -> dict[str, Any]:
    """Read one ``*_with_badbin.csv`` Burst Analyser table saved pre-download.

    ``saveBurstAnalyser`` drops the BAT BadBin column, so the research
    workflow persists these CSVs first; they are the authoritative BAT input.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty badbin table: {path}")
    names = tuple(rows[0].keys())
    columns: dict[str, list[str]] = {name: [] for name in names}
    for row in rows:
        for name in names:
            columns[name].append(row[name])
    arrays: dict[str, Any] = {
        name: np.asarray(values, dtype=float) for name, values in columns.items() if name != "BadBin"
    }
    if "BadBin" in columns:
        arrays["BadBin"] = np.asarray([_to_bool(value) for value in columns["BadBin"]], dtype=bool)
    arrays["n_total"] = len(rows)
    arrays["n_bad"] = int(arrays.get("BadBin", np.zeros(len(rows), dtype=bool)).sum())
    return arrays


def read_burst_analyser_dat(path: str | Path) -> dict[str, Any]:
    """Read one Burst Analyser ``.dat`` table.

    Only the first six positional columns (``Time``, ``TimePos``, ``TimeNeg``,
    ``Flux``, ``FluxPos``, ``FluxNeg``) are consumed; trailing column sets
    (Rate/Gamma/Aux) vary between BAT and XRT products.
    """
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    data = np.atleast_2d(data)
    if data.shape[1] < len(_DAT_FLUX_COLUMNS):
        raise ValueError(f"{path}: expected at least {len(_DAT_FLUX_COLUMNS)} columns, got {data.shape[1]}")
    out: dict[str, Any] = {name: data[:, index] for index, name in enumerate(_DAT_FLUX_COLUMNS)}
    # 14-column tables lack a native rate; their ECF maps flux to rate.
    # 15-column tables contain native rate and asymmetric rate uncertainties.
    if data.shape[1] == 14:
        ecf = data[:, 6]
        out["ECF"] = ecf
        out["rate"] = np.divide(data[:, 3], ecf, out=np.full(len(data), np.nan), where=ecf != 0)
        out["rate_err_low"] = np.divide(np.abs(data[:, 5]), np.abs(ecf), out=np.full(len(data), np.nan), where=ecf != 0)
        out["rate_err_high"] = np.divide(np.abs(data[:, 4]), np.abs(ecf), out=np.full(len(data), np.nan), where=ecf != 0)
        out["rate_source"] = "flux_over_ecf"
    elif data.shape[1] >= 15:
        out["rate"] = data[:, 6]
        out["rate_err_high"] = np.abs(data[:, 7])
        out["rate_err_low"] = np.abs(data[:, 8])
        out["rate_source"] = "native_rate"
    else:
        out["rate"] = np.full(len(data), np.nan)
        out["rate_err_low"] = np.full(len(data), np.nan)
        out["rate_err_high"] = np.full(len(data), np.nan)
        out["rate_source"] = "unavailable"
    out["column_count"] = data.shape[1]
    out["n_total"] = data.shape[0]
    return out


@dataclass(frozen=True, slots=True)
class GRBRecord:
    """One GRB as resolved from the merged catalog and duration tables."""

    merge_key: str
    name: str
    canonical_name: str
    trigger_utc: str | None
    ra_deg: float | None
    dec_deg: float | None
    redshift: float | None
    redshift_method: str
    redshift_source: str
    redshift_note: str
    redshift_manual: bool
    trigger_met: float | None
    bat_t100: tuple[float, float] | None
    bat_t90: tuple[float, float] | None
    bat_t50: tuple[float, float] | None
    xrt_start_s: float | None
    raw_dir: Path
    bat_trigger_utc: str | None = None
    sdc_trigger_utc: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "merge_key": self.merge_key,
            "name": self.name,
            "canonical_name": self.canonical_name,
            "trigger_utc": self.trigger_utc,
            "bat_trigger_utc": self.bat_trigger_utc,
            "sdc_trigger_utc": self.sdc_trigger_utc,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "redshift": self.redshift,
            "redshift_method": self.redshift_method,
            "redshift_source": self.redshift_source,
            "redshift_note": self.redshift_note,
            "redshift_manual": self.redshift_manual,
            "trigger_met": self.trigger_met,
            "xrt_start_s": self.xrt_start_s,
            "raw_dir": str(self.raw_dir),
        }
        for name in ("bat_t100", "bat_t90", "bat_t50"):
            window = getattr(self, name)
            payload[name] = None if window is None else list(window)
        return payload


def _resolve_raw_dir(raw_root: Path, merge_key: str, row: Mapping[str, Any]) -> Path:
    candidates = [
        _contained_path(raw_root, f"GRB_{merge_key}"),
        _contained_path(raw_root, merge_key),
    ]
    name = str(row.get("SDC_Name") or "").strip()
    if name:
        candidates.append(_contained_path(raw_root, name))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def build_grb_record(
    row: Mapping[str, Any],
    raw_root: str | Path,
    *,
    redshift: float | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
) -> GRBRecord:
    """Build a :class:`GRBRecord` from one merged-catalog CSV row."""
    merge_key = str(row.get("_merge_key") or "").strip() or normalize_grb_name(
        row.get("CanonicalName") or row.get("SDC_Name") or ""
    )
    if not merge_key:
        raise ValueError("catalog row lacks a usable GRB name")
    merge_key = safe_filename_token(merge_key)
    z_value, z_method, z_source, z_note = preferred_redshift(row)
    if redshift is not None:
        z_value, z_method, z_source, z_note = (
            float(redshift),
            "manual_override",
            "pipeline_input",
            "override via SwiftGRBInput.redshift",
        )
    raw_dir = _resolve_raw_dir(Path(raw_root).expanduser().resolve(), merge_key, row)
    ra = ra_deg if ra_deg is not None else _to_float(row.get("SDC_BAT_RA"))
    if ra is None:
        ra = _to_float(row.get("SDC_XRT_RA"))
    dec = dec_deg if dec_deg is not None else _to_float(row.get("SDC_BAT_Dec"))
    if dec is None:
        dec = _to_float(row.get("SDC_XRT_Dec"))

    def window(prefix: str) -> tuple[float, float] | None:
        start = _to_float(row.get(f"bat_{prefix}_start_s"))
        stop = _to_float(row.get(f"bat_{prefix}_stop_s"))
        if start is None or stop is None:
            return None
        return (start, stop)

    return GRBRecord(
        merge_key=merge_key,
        name=str(row.get("SDC_Name") or "").strip(),
        canonical_name=str(row.get("CanonicalName") or "").strip() or merge_key,
        trigger_utc=str(row.get("BAT_Trig_time_UTC") or row.get("SDC_TriggerTime") or "").strip() or None,
        bat_trigger_utc=str(row.get("BAT_Trig_time_UTC") or "").strip() or None,
        sdc_trigger_utc=str(row.get("SDC_TriggerTime") or "").strip() or None,
        ra_deg=ra,
        dec_deg=dec,
        redshift=z_value,
        redshift_method=z_method,
        redshift_source=z_source,
        redshift_note=z_note,
        redshift_manual=bool(_to_bool(row.get("preferred_z_is_manual_override")) or redshift is not None),
        trigger_met=_to_float(row.get("bat_duration_Trig_time_met")),
        bat_t100=window("t100"),
        bat_t90=window("t90"),
        bat_t50=window("t50"),
        xrt_start_s=_to_float(row.get("SDC_XRT_StartTime")),
        raw_dir=raw_dir,
    )


def prompt_classification(record: GRBRecord) -> dict[str, Any]:
    """Strict prompt-stage judgment: XRT start inside the BAT T90 window.

    Membership uses the catalog's trigger-relative seconds.  BAT UTC remains
    the preferred absolute reference, while any SDC timing mismatch is saved
    as provenance instead of changing the time origin.
    """
    missing: list[str] = []
    if record.trigger_utc is None:
        missing.append("missing_bat_trigger_time")
    if record.bat_t90 is None:
        missing.append("missing_bat_t90_window")
    if record.xrt_start_s is None:
        missing.append("missing_xrt_start")
    if missing:
        return {"in_prompt": "UNDECIDED", "reasons": missing}
    t0 = Time(record.trigger_utc, format="iso", scale="utc")
    prompt_start = t0 + TimeDelta(record.bat_t90[0], format="sec")
    prompt_stop = t0 + TimeDelta(record.bat_t90[1], format="sec")
    xrt_start = t0 + TimeDelta(record.xrt_start_s, format="sec")
    in_prompt = bool(prompt_start <= xrt_start <= prompt_stop)
    payload = {
        "in_prompt": "YES" if in_prompt else "NO",
        "reasons": [],
        "bat_t90_start_utc": prompt_start.utc.isot,
        "bat_t90_stop_utc": prompt_stop.utc.isot,
        "xrt_start_utc": xrt_start.utc.isot,
        "bat_t90_start_s": record.bat_t90[0],
        "bat_t90_stop_s": record.bat_t90[1],
        "xrt_start_s": record.xrt_start_s,
    }
    if record.bat_trigger_utc and record.sdc_trigger_utc:
        try:
            payload["bat_sdc_trigger_difference_s"] = float(
                (Time(record.sdc_trigger_utc, format="iso", scale="utc")
                 - Time(record.bat_trigger_utc, format="iso", scale="utc")).sec
            )
        except ValueError:
            payload["trigger_time_warning"] = "could_not_parse_bat_or_sdc_trigger_utc"
    return payload


@dataclass(frozen=True, slots=True)
class TimeResolvedSegment:
    """One Bayesian-block segment with background-scaled significance."""

    mode: str
    index: int
    start: float
    stop: float
    source_counts: float
    background_counts: float
    alpha: float
    snr: float
    merged: bool = False

    @property
    def key(self) -> str:
        return f"{self.mode.lower()}_seg{self.index:03d}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "mode": self.mode,
            "index": self.index,
            "start": self.start,
            "stop": self.stop,
            "source_counts": self.source_counts,
            "background_counts": self.background_counts,
            "alpha": self.alpha,
            "snr": self.snr,
            "merged": self.merged,
        }


def segment_from_payload(payload: Mapping[str, Any]) -> TimeResolvedSegment:
    """Rebuild a :class:`TimeResolvedSegment` from its manifest payload."""
    fields = {name: value for name, value in payload.items() if name != "key"}
    return TimeResolvedSegment(**fields)


def _segment_snr(source: float, background: float, alpha: float) -> float:
    return float(li_ma_snr(source, background, alpha, signed=True))


def merge_low_snr_segments(
    segments: Sequence[TimeResolvedSegment],
    *,
    snr_threshold: float = 3.0,
    min_source_counts: float = 0.0,
) -> list[TimeResolvedSegment]:
    """Iteratively merge the least-significant segment into its weaker neighbour.

    Mirrors the research rule: merge until every segment satisfies
    ``snr >= snr_threshold`` (and ``source_counts >= min_source_counts`` for
    PC-mode spectra) or a single segment remains.
    """

    def deficient(segment: TimeResolvedSegment) -> bool:
        return segment.snr < snr_threshold or segment.source_counts < min_source_counts

    blocks: list[TimeResolvedSegment] = list(segments)
    while len(blocks) > 1 and any(deficient(block) for block in blocks):
        candidates = [block for block in blocks if deficient(block)]
        if not candidates:
            break
        worst = min(candidates, key=lambda block: block.snr)
        index = blocks.index(worst)
        if index == 0:
            partner_index = 1
        elif index == len(blocks) - 1:
            partner_index = index - 1
        else:
            partner_index = index - 1 if blocks[index - 1].snr <= blocks[index + 1].snr else index + 1
        lo, hi = sorted((index, partner_index))
        left, right = blocks[lo], blocks[hi]
        background = left.background_counts + right.background_counts
        alpha = (
            (left.alpha * left.background_counts + right.alpha * right.background_counts) / background
            if background > 0
            else (left.alpha + right.alpha) / 2.0
        )
        source = left.source_counts + right.source_counts
        blocks[lo:hi + 1] = [
            TimeResolvedSegment(
                mode=left.mode,
                index=left.index,
                start=left.start,
                stop=right.stop,
                source_counts=source,
                background_counts=background,
                alpha=alpha,
                snr=_segment_snr(source, background, alpha),
                merged=True,
            )
        ]
    return [
        TimeResolvedSegment(
            mode=block.mode,
            index=position,
            start=block.start,
            stop=block.stop,
            source_counts=block.source_counts,
            background_counts=block.background_counts,
            alpha=block.alpha,
            snr=block.snr,
            merged=block.merged,
        )
        for position, block in enumerate(blocks)
    ]


def _read_event_times(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, gti_bounds)`` from one XRT event file."""
    with fits.open(path, memmap=False) as hdus:
        events = hdus[1].data
        times = np.asarray(events["TIME"], dtype=float)
        gti: list[float] = []
        for hdu in hdus[1:]:
            if hdu.name == "GTI" and hdu.data is not None:
                for start, stop in zip(hdu.data["START"], hdu.data["STOP"]):
                    gti.extend((float(start), float(stop)))
    return times, np.asarray(gti, dtype=float).reshape(-1, 2)


def _read_area_ratio(bkg_event: Path, start: float, stop: float) -> float:
    """Background-area ratio ``alpha`` from SRCAREA/BGAREA over one window."""
    try:
        with fits.open(bkg_event, memmap=False) as hdus:
            columns = hdus[1].columns.names or []
            if "SRCAREA" not in columns or "BGAREA" not in columns:
                return float("nan")
            data = hdus[1].data
            times = np.asarray(data["TIME"], dtype=float)
            mask = (times >= start) & (times <= stop)
            if not mask.any():
                mask = np.abs(times - (start + stop) / 2.0) <= max(stop - start, 1.0)
            src = np.asarray(data["SRCAREA"][mask], dtype=float)
            bkg = np.asarray(data["BGAREA"][mask], dtype=float)
            valid = bkg > 0
            if not valid.any():
                return float("nan")
            return float(np.nanmedian(src[valid] / bkg[valid]))
    except (OSError, KeyError):
        return float("nan")


def _source_area_from_event(bkg_event: Path, start: float, stop: float) -> float | None:
    """Return the local SRCAREA needed to interpret an UKSSDC area table."""
    try:
        with fits.open(bkg_event, memmap=False) as hdus:
            data = hdus[1].data
            names = hdus[1].columns.names or []
            if "SRCAREA" not in names:
                return None
            times = np.asarray(data["TIME"], dtype=float)
            mask = (times >= start) & (times <= stop)
            values = np.asarray(data["SRCAREA"][mask], dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            return float(np.nanmedian(values)) if len(values) else None
    except (OSError, KeyError):
        return None


def _event_trigger_met(event_path: Path) -> float | None:
    """Read the event product trigger origin, when provided by UKSSDC."""
    try:
        with fits.open(event_path, memmap=False) as hdus:
            return _to_float(hdus[1].header.get("TRIGTIME") or hdus[0].header.get("TRIGTIME"))
    except OSError:
        return None


def _alpha_from_area_file(
    area_path: Path,
    *,
    source_area: float | None = None,
    time_s: float | None = None,
) -> float:
    """Calculate ``alpha`` from ``start stop background-area`` rows.

    UKSSDC ``.area`` files do not store a source area.  A valid source area
    (normally SRCAREA from the event table) is therefore mandatory; returning
    NaN makes the caller stop a fit instead of assuming ``alpha=1``.
    """
    if source_area is None or not np.isfinite(source_area) or source_area <= 0:
        return float("nan")
    candidates: list[float] = []
    try:
        opener = __import__("gzip").open if area_path.suffix == ".gz" else open
        with opener(area_path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    start, stop, bkg = (_to_float(value) for value in parts[:3])
                    if start is None or stop is None or bkg is None or bkg <= 0:
                        continue
                    if time_s is None or start <= time_s <= stop:
                        candidates.append(float(source_area / bkg))
    except OSError:
        return float("nan")
    return float(np.nanmedian(candidates)) if candidates else float("nan")


def _event_chunks(
    times: np.ndarray, gtis: np.ndarray, mode: str, observation_windows: np.ndarray | None = None
) -> list[tuple[float, float]]:
    """Never form blocks across WT GTIs or PC observation identifiers.

    When ``obstimes.txt`` is unavailable PC falls back to GTIs, which is safer
    than treating gaps between observations as sampled exposure.
    """
    if mode.upper() == "WT" and len(gtis):
        return [(float(start), float(stop)) for start, stop in gtis if stop > start]
    if mode.upper() == "PC" and observation_windows is not None and len(observation_windows):
        return [(float(start), float(stop)) for start, stop in observation_windows if stop > start]
    if mode.upper() == "PC" and len(gtis):
        return [(float(start), float(stop)) for start, stop in gtis if stop > start]
    if len(times) == 0:
        return []
    return [(float(np.min(times)), float(np.max(times)))]


def _locate_first(*paths: Path | None, directory: bool = False) -> Path | None:
    for path in paths:
        if path is not None and (path.is_dir() if directory else path.is_file()):
            return path
    return None


@dataclass
class SwiftBurstAnalyserFetcher:
    """UKSSDC Burst Analyser downloader implementing the research 3-tier strategy.

    Tier 1 fetches everything at once; tier 2 falls back to chunked BAT+XRT
    requests (``BATBinning=["SNR5"]`` — the ``sinceT0`` variant alone breaks
    the API); tier 3 uses standard XRT lightcurves. Burst Analyser ``tar``
    downloads are skipped by default (unreliable endpoint). Requires the
    optional ``swifttools`` package (``pip install 'jinwu-swift[ukssdc]'``).
    """

    include_tar: bool = False
    retries: int = 3
    retry_wait_s: float = 5.0
    username: str | None = None

    def __call__(self, merge_key: str, raw_dir: Path) -> dict[str, Any]:
        try:
            from swifttools.ukssdc.data import GRB as udg
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "swifttools is required for Burst Analyser downloads; "
                "install with `pip install 'jinwu-swift[ukssdc]'`"
            ) from exc

        grb_name = f"GRB{merge_key}"
        parsed_dir = _contained_path(raw_dir, "burst_analyser", "parsed")
        parsed_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        base: dict[str, Any] = {
            "GRBName": grb_name,
            "returnData": True,
            "silent": True,
        }
        attempts: tuple[dict[str, Any], ...] = (
            {"instruments": "all", "BATBinning": ["SNR5"], "bands": "all"},
            {"instruments": ["BAT"], "BATBinning": ["SNR5"], "bands": ["BATBand", "XRTBand"]},
            {"instruments": ["XRT"], "bands": ["XRTBand"]},
        )
        data: Any = None
        for index, kwargs in enumerate(attempts, start=1):
            for attempt in range(self.retries):
                try:
                    data = udg.getBurstAnalyser(**base, **kwargs, downloadTar=self.include_tar)
                    break
                except TypeError:
                    data = udg.getBurstAnalyser(**base, **kwargs)
                    break
                except Exception as exc:  # noqa: BLE001 - network layer
                    message = str(exc).lower()
                    if "no confirmed grb" in message or "must supply a grb targetid" in message:
                        return {"ok": False, "error": f"missing_target:{grb_name}", "warnings": warnings}
                    if attempt < self.retries - 1:
                        time.sleep(self.retry_wait_s)
                        continue
                    warnings.append(f"tier{index}_failed:{exc}")
            if data is not None:
                break
        if data is None:
            return {"ok": False, "error": "all_tiers_failed", "warnings": warnings}
        try:
            udg.saveBurstAnalyser(data, destDir=str(_contained_path(raw_dir, "burst_analyser")), silent=True)
        except Exception as exc:  # noqa: BLE001 - network layer
            warnings.append(f"save_failed:{exc}")
        self._save_badbin_tables(data, parsed_dir, warnings)
        return {"ok": True, "parsed_dir": str(parsed_dir), "warnings": warnings}

    @staticmethod
    def _save_badbin_tables(data: Any, parsed_dir: Path, warnings: list[str]) -> None:
        """Persist BAT BadBin flags before ``saveBurstAnalyser`` drops them."""
        try:
            singles = data if isinstance(data, list) else [data]
            for single in singles:
                bat = (single or {}).get("BAT", {}) if isinstance(single, dict) else {}
                for binning, payload in bat.items():
                    if not isinstance(payload, dict):
                        continue
                    for dataset, frame in payload.items():
                        if not hasattr(frame, "columns") or "BadBin" not in getattr(frame, "columns", []):
                            continue
                        out = frame.copy()
                        out["BadBin"] = out["BadBin"].astype(bool)
                        name = f"BAT_{safe_filename_token(binning)}_{safe_filename_token(dataset)}_with_badbin.csv"
                        out.to_csv(_contained_path(parsed_dir, name), index=False)
        except Exception as exc:  # noqa: BLE001 - best effort provenance
            warnings.append(f"badbin_preserve_failed:{exc}")


@dataclass(frozen=True, slots=True, kw_only=True)
class SwiftGRBInput(PipelineInput):
    """Inputs for one GRB inside a sample workspace."""

    redshift: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    galactic_nh_1e22: float | None = None
    catalog_path: Path | str | None = None
    raw_products_dir: Path | str | None = None
    xrt_products_dir: Path | str | None = None
    # Explicit per-run overrides.  ``None`` means use :class:`SwiftGRB`.
    bat_binning: str | None = None
    bat_band: str | None = None
    xrt_band: str | None = None
    group_min_counts: int | None = None
    bblock_p0: float | None = None
    wt_snr_threshold: float | None = None
    pc_snr_threshold: float | None = None
    pc_min_source_counts: float | None = None
    max_segments_per_mode: int | None = None
    request_xrt: bool = False
    poll_xrt: bool = False
    xrt_user: str | None = None
    fetcher: Fetcher | None = field(default=None, compare=False)

    def raw_root(self) -> Path:
        if self.raw_products_dir is not None:
            return Path(self.raw_products_dir).expanduser().resolve()
        return _contained_path(self.resolved_root(), "raw")


@dataclass(slots=True)
class SwiftGRBResult:
    """Public pipeline result, including partial review outcomes."""

    status: PipelineStatus
    grb: str
    workspace: Path
    completed_stages: tuple[str, ...]
    redshift: float | None
    redshift_method: str | None
    galactic_nh_1e22: float | None
    prompt: dict[str, Any] | None
    products: dict[str, dict[str, str]]
    warnings: tuple[str, ...] = ()
    message: str | None = None


@register_pipeline("swift.grb")
class SwiftGRBPipeline(InstrumentPipeline[SwiftGRBInput, SwiftGRBResult]):
    """BAT+XRT GRB workflow: catalog, products, blocks, spectra, fit, report."""

    stages = (
        PipelineStage("catalog"),
        PipelineStage("download", ("catalog",)),
        PipelineStage("galactic_absorption", ("catalog",)),
        PipelineStage("lightcurve", ("catalog", "download")),
        PipelineStage("prompt", ("catalog",)),
        PipelineStage("bblocks", ("catalog", "download")),
        PipelineStage("spectra", ("bblocks", "download")),
        PipelineStage("fit", ("catalog", "galactic_absorption", "spectra")),
        PipelineStage(
            "report",
            ("catalog", "lightcurve", "prompt", "galactic_absorption", "bblocks", "spectra", "fit"),
        ),
    )

    def __init__(
        self,
        input_data: SwiftGRBInput,
        *,
        config: InstrumentConfig | None = None,
        extractor: Extractor | None = None,
        nhtot_query: NhtotQuery | None = None,
        fetcher: Fetcher | None = None,
    ):
        super().__init__(input_data, config=config or _default_config())
        self.extractor: Extractor = extractor or extract_products_with_xselect
        self.nhtot_query = nhtot_query
        self.fetcher = fetcher or input_data.fetcher
        self._cached_record: GRBRecord | None = None
        self._prepared_segments: list[PreparedSpectrum] = []

    # ------------------------------------------------------------------
    # Framework contract
    # ------------------------------------------------------------------
    def run(self, *, until: str | None = None, resume: bool | None = None) -> SwiftGRBResult:
        return cast(SwiftGRBResult, super().run(until=until, resume=resume))

    def validate_input(self) -> None:
        if not self.input.target_id.strip():
            raise ValueError("target_id (normalized GRB name) must not be empty")
        root = self.input.resolved_root()
        if not root.is_dir():
            raise FileNotFoundError(f"sample workspace root does not exist: {root}")

    def stage_code_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        from ...core import fit, galactic, pipeline as core_pipeline, products, spectrum_prep, utils, xselect
        from ...ftools import grppha_hsp

        def _src(module) -> Path:
            # getsourcefile 失败时显式报错：退化成 Path("") 会让指纹静默失效（AUD-01）
            path = inspect.getsourcefile(module)
            if not path:
                raise RuntimeError(f"无法定位模块源文件: {module!r}")
            return Path(path)

        core = {
            "pipeline.py": _src(core_pipeline),
            "galactic.py": _src(galactic),
            "products.py": _src(products),
            "spectrum_prep.py": _src(spectrum_prep),
            "utils.py": _src(utils),
            "xselect.py": _src(xselect),
            "fit.py": _src(fit),
            "grppha_hsp.py": _src(grppha_hsp),
        }
        dependencies: dict[str, tuple[Path, ...]] = {
            "catalog": (core["pipeline.py"],),
            "download": (),
            "galactic_absorption": (core["galactic.py"], core["utils.py"]),
            "lightcurve": (),
            "prompt": (),
            "bblocks": (core["utils.py"],),
            "spectra": (core["xselect.py"], core["grppha_hsp.py"]),
            "fit": (core["fit.py"], core["spectrum_prep.py"]),
            "report": (core["products.py"],),
        }
        return dependencies.get(stage.name, ())

    def stage_input_dependencies(self, stage: PipelineStage) -> tuple[Path, ...]:
        """Fingerprint catalogs and local science products used by each stage."""
        catalog = (
            Path(self.input.catalog_path).expanduser().resolve()
            if self.input.catalog_path is not None
            else _contained_path(self._catalog_dir, "swift_grb_catalog_merged.csv")
        )
        parsed = self._record.raw_dir / "burst_analyser" / "parsed"
        downloads = self._downloads_dir
        selected: list[Path] = [catalog]
        if stage.name in {"download", "lightcurve"}:
            selected.extend(sorted(parsed.glob("*.dat")))
            selected.extend(sorted(parsed.glob("*_with_badbin.csv")))
        if stage.name in {"bblocks", "spectra"}:
            selected.extend(sorted((downloads / "lc").glob("*")))
            selected.extend(sorted((downloads / "spec").glob("interval0/*")))
        return tuple(selected)

    def execute_stage(
        self,
        stage: PipelineStage,
        context: Mapping[str, StageResult],
    ) -> StageResult:
        handlers = {
            "catalog": self._stage_catalog,
            "download": self._stage_download,
            "galactic_absorption": self._stage_galactic_absorption,
            "lightcurve": self._stage_lightcurve,
            "prompt": self._stage_prompt,
            "bblocks": self._stage_bblocks,
            "spectra": self._stage_spectra,
            "fit": self._stage_fit,
            "report": self._stage_report,
        }
        return handlers[stage.name](context)

    def build_result(self, context: Mapping[str, StageResult]) -> SwiftGRBResult:
        catalog = context.get("catalog")
        record_payload = (catalog.data.get("record") if catalog else None) or {}
        prompt = context["prompt"].data.get("prompt") if "prompt" in context else None
        nh = context["galactic_absorption"].data.get("tbabs_nh_1e22") if "galactic_absorption" in context else None
        statuses = {name: result.status for name, result in context.items()}
        overall = PipelineStatus.COMPLETED
        if any(status is PipelineStatus.FAILED for status in statuses.values()):
            overall = PipelineStatus.FAILED
        elif any(status is not PipelineStatus.COMPLETED for status in statuses.values()):
            overall = PipelineStatus.NEEDS_REVIEW
        products = {
            name: {key: str(value) for key, value in result.outputs.items()}
            for name, result in context.items()
            if result.outputs
        }
        warnings: list[str] = []
        for name, result in context.items():
            warnings.extend(f"{name}:{item}" for item in result.data.get("warnings", []))
        failed_stage = next((name for name, status in statuses.items() if status is PipelineStatus.FAILED), None)
        message = None
        if failed_stage is not None:
            message = f"stage {failed_stage} failed: {context[failed_stage].message}"
        elif overall is PipelineStatus.NEEDS_REVIEW:
            review = [name for name, status in statuses.items() if status is not PipelineStatus.COMPLETED]
            message = f"stages need review: {', '.join(sorted(review))}"
        return SwiftGRBResult(
            status=overall,
            grb=str(record_payload.get("merge_key") or self.input.target_id),
            workspace=self.workspace,
            completed_stages=tuple(
                name for name, result in context.items() if result.status is PipelineStatus.COMPLETED
            ),
            redshift=record_payload.get("redshift"),
            redshift_method=record_payload.get("redshift_method"),
            galactic_nh_1e22=nh,
            prompt=prompt,
            products=products,
            warnings=tuple(warnings),
            message=message,
        )

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------
    @property
    def _catalog_dir(self) -> Path:
        if self.input.catalog_path is not None:
            return Path(self.input.catalog_path).expanduser().resolve().parent
        return _contained_path(self.input.resolved_root(), "catalog")

    @property
    def _downloads_dir(self) -> Path:
        if self.input.xrt_products_dir is not None:
            return Path(self.input.xrt_products_dir).expanduser().resolve()
        legacy = _contained_path(
            self.input.resolved_root(), "xrt_product_requests_maybe", "downloads", f"GRB_{self._record.merge_key}"
        )
        if legacy.is_dir():
            return legacy
        return _contained_path(
            self.workspace,
            "data",
            "xrt_products",
            f"GRB_{self._record.merge_key}",
        )

    @property
    def _record(self) -> GRBRecord:
        if self._cached_record is None:
            self._cached_record = build_grb_record(
                self._read_catalog_row(),
                self.input.raw_root(),
                redshift=self.input.redshift,
                ra_deg=self.input.ra_deg,
                dec_deg=self.input.dec_deg,
            )
        return self._cached_record

    def _read_catalog_row(self) -> dict[str, str]:
        merged = (
            Path(self.input.catalog_path).expanduser().resolve()
            if self.input.catalog_path is not None
            else _contained_path(self._catalog_dir, "swift_grb_catalog_merged.csv")
        )
        if not merged.is_file():
            raise FileNotFoundError(
                f"merged catalog not found: {merged}; build it with the catalog fetch "
                "(swifttools uq.GRBQuery) or point --root at the research workspace"
            )
        target = normalize_grb_name(self.input.target_id)
        with open(merged, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = str(row.get("_merge_key") or "").strip()
                if not key:
                    key = normalize_grb_name(row.get("CanonicalName") or row.get("SDC_Name") or "")
                if key == target:
                    return row
        raise KeyError(f"GRB {self.input.target_id!r} not found in {merged.name}")

    @property
    def _swift_config(self) -> SwiftGRB:
        """Return the Swift-specific settings without importing the plugin in core."""
        return cast(SwiftGRB, self.config) if isinstance(self.config, SwiftGRB) else SwiftGRB()

    def _setting(self, override: Any, section: str, name: str) -> Any:
        return override if override is not None else getattr(getattr(self._swift_config, section), name)

    @property
    def _bat_binning(self) -> str:
        return str(self._setting(self.input.bat_binning, "data", "bat_binning"))

    @property
    def _bat_band(self) -> str:
        return str(self._setting(self.input.bat_band, "data", "bat_band"))

    @property
    def _xrt_band(self) -> str:
        return str(self._setting(self.input.xrt_band, "data", "xrt_band"))

    @property
    def _group_min_counts(self) -> int:
        return int(self.input.group_min_counts if self.input.group_min_counts is not None else self.config.spectrum.group_min_counts)

    def _xselect_executable(self) -> Path | None:
        """Locate the conda-installed HEASoft binary without changing PATH."""
        headas = os.environ.get("HEADAS")
        candidates = [Path(headas) / "bin" / "xselect"] if headas else []
        candidates.append(Path(sys.prefix) / "heasoft" / "bin" / "xselect")
        return next((candidate for candidate in candidates if candidate.is_file()), None)

    @staticmethod
    def _prepared_from_manifest(item: Mapping[str, Any]) -> PreparedSpectrum | None:
        fields = item.get("prepared")
        if not isinstance(fields, Mapping):
            return None
        segment = item.get("segment", {})
        energy = fields.get("energy_range_keV") or (0.3, 10.0)

        def path_or_none(key: str) -> Path | None:
            value = fields.get(key)
            return _manifest_path(value) if value else None

        return PreparedSpectrum(
            instrument=str(fields.get("instrument") or "XRT"),
            obsid=fields.get("obsid"),
            module=fields.get("module"),
            detector=str(fields.get("detector") or segment.get("mode") or "PC"),
            source_id=str(fields.get("source_id") or segment.get("key") or ""),
            source_pha=path_or_none("source_pha"),
            grouped_pha=path_or_none("grouped_pha"),
            background_pha=path_or_none("background_pha"),
            arf=path_or_none("arf"),
            rmf=path_or_none("rmf"),
            group_min=int(fields.get("group_min") or 20),
            energy_range_keV=(float(energy[0]), float(energy[1])),
        )

    # ------------------------------------------------------------------
    # Stage: catalog
    # ------------------------------------------------------------------
    def _stage_catalog(self, context: Mapping[str, StageResult]) -> StageResult:
        record = self._record
        warnings: list[str] = []
        if record.redshift is None:
            warnings.append("no redshift resolved; spectral fits will assume z=0")
        if record.trigger_utc is None:
            warnings.append("missing BAT trigger time; prompt classification will be UNDECIDED")
        payload = record.to_payload()
        path = write_json(_contained_path(self.workspace, "catalog", "grb_record.json"), payload)
        return StageResult(
            status=PipelineStatus.COMPLETED if record.redshift is not None else PipelineStatus.NEEDS_REVIEW,
            outputs={"record": str(path)},
            data={"record": payload, "warnings": warnings},
            message=None if record.redshift is not None else "redshift unresolved",
        )

    # ------------------------------------------------------------------
    # Stage: download
    # ------------------------------------------------------------------
    def _missing_products(self) -> list[str]:
        parsed_dir = self._record.raw_dir / "burst_analyser" / "parsed"
        bat_pattern = f"BAT_*{self._bat_binning}_{self._bat_band}*"
        bat_ok = any(parsed_dir.glob(f"{bat_pattern}_with_badbin.csv")) or any(parsed_dir.glob(f"{bat_pattern}.dat"))
        xrt_ok = any(parsed_dir.glob(f"XRT_{self._xrt_band}_*_incbad.dat"))
        return [
            label
            for label, present in (
                ("bat_burst_analyser", bat_ok),
                ("xrt_burst_analyser", xrt_ok),
            )
            if not present
        ]

    def _xrt_request_state_path(self) -> Path:
        return _contained_path(self.workspace, "xrt_requests", f"{self._record.merge_key}.json")

    def _ensure_xrt_products(self) -> tuple[bool, list[str], str | None]:
        """Submit or resume a single UKSSDC XRT product request explicitly.

        The state file is intentionally retained for ``--no-resume`` runs: an
        ambiguous submit result must be reviewed or recovered, never submitted
        again automatically.
        """
        existing = self._downloads_dir / "lc"
        if existing.is_dir():
            return True, [], None
        state_path = self._xrt_request_state_path()
        state: dict[str, Any] = {}
        if state_path.is_file():
            try:
                state = _json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False, [], "saved XRT request state is unreadable"
        if state.get("status") == "submitting":
            return False, [], "prior XRT submission has unknown outcome; refusing duplicate submission"
        if not (self.input.request_xrt or self.input.poll_xrt):
            return False, [], "XRT products absent; pass --request-xrt or supply --xrt-products-dir"
        username = self.input.xrt_user or os.environ.get("SWIFT_XRT_USER")
        if not username:
            return False, [], "XRT request requires --xrt-user or SWIFT_XRT_USER"
        try:
            from swifttools.ukssdc import xrt_prods as xp
        except ImportError:
            return False, [], "swifttools XRT product API is unavailable"
        try:
            request = xp.XRTProductRequest(username)
            job_id = state.get("job_id")
            if job_id:
                request.copyOldJob(int(job_id), becomeThis=True)
            elif self.input.poll_xrt:
                return False, [], "--poll-xrt has no saved job id"
            else:
                state = {
                    "schema_version": 1,
                    "grb": self._record.merge_key,
                    "status": "submitting",
                    "request": {"username": username, "request_xrt": True},
                }
                write_json(state_path, state)
                if self._record.ra_deg is None or self._record.dec_deg is None:
                    return False, [], "XRT request requires catalog RA/Dec"
                request.setGlobalPars(
                    name=f"GRB{self._record.merge_key}",
                    RA=self._record.ra_deg,
                    Dec=self._record.dec_deg,
                    T0=self._record.trigger_utc,
                )
                request.addLightCurve(binMeth="counts", pcCounts=20, wtCounts=30, dynamic=True)
                request.addSpectrum(hasRedshift=self._record.redshift is not None, redshift=self._record.redshift)
                submitted = request.submit()
                job_id = getattr(request, "JobID", None) or getattr(request, "jobID", None)
                if not submitted or job_id is None:
                    state["status"] = "submitting"
                    write_json(state_path, state)
                    return False, [], "XRT submission outcome is unclear; inspect saved request state"
                state.update({"status": "submitted", "job_id": int(job_id)})
                write_json(state_path, state)
            complete = bool(request.complete if not callable(getattr(request, "complete", None)) else request.complete())
            if not complete:
                state["status"] = "submitted"
                write_json(state_path, state)
                return False, [], f"XRT job {state.get('job_id')} is not complete"
            self._downloads_dir.mkdir(parents=True, exist_ok=True)
            request.downloadProducts(str(self._downloads_dir), clobber=False)
            state["status"] = "downloaded"
            state["destination"] = str(self._downloads_dir)
            write_json(state_path, state)
            return (self._downloads_dir / "lc").is_dir(), [], None
        except Exception as exc:  # noqa: BLE001 - remote API failures are review states
            state["status"] = state.get("status", "submitted")
            state["error"] = repr(exc)
            write_json(state_path, state)
            return False, [], f"XRT request failed or is pending: {exc}"

    def _stage_download(self, context: Mapping[str, StageResult]) -> StageResult:
        record = self._record
        parsed_dir = _contained_path(record.raw_dir, "burst_analyser", "parsed")
        warnings: list[str] = []
        missing = self._missing_products()
        if missing and self.fetcher is not None:
            outcome = dict(self.fetcher(record.merge_key, record.raw_dir))
            warnings.extend(str(item) for item in outcome.get("warnings", []))
            if outcome.get("ok"):
                missing = self._missing_products()
        data = {
            "warnings": warnings,
            "raw_dir": str(record.raw_dir),
            "parsed_dir": str(parsed_dir),
            "downloads_dir": str(self._downloads_dir),
            "missing": missing,
        }
        if missing:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                data=data,
                message=(
                    f"missing products: {', '.join(missing)}; provide a fetcher "
                    "or pre-download (swifttools Burst Analyser / UKSSDC XRT products)"
                ),
            )
        xrt_ok = (self._downloads_dir / "lc").is_dir()
        xrt_message: str | None = None
        if not xrt_ok and (self.input.request_xrt or self.input.poll_xrt):
            xrt_ok, xrt_warnings, xrt_message = self._ensure_xrt_products()
            warnings.extend(xrt_warnings)
        if not xrt_ok and (self.input.request_xrt or self.input.poll_xrt):
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                data={**data, "warnings": warnings, "xrt_request_state": str(self._xrt_request_state_path())},
                message=xrt_message,
            )
        outputs: dict[str, str] = {}
        if parsed_dir.is_dir():
            outputs["parsed_dir"] = str(parsed_dir)
        return StageResult(status=PipelineStatus.COMPLETED, outputs=outputs, data=data)

    # ------------------------------------------------------------------
    # Stage: galactic_absorption
    # ------------------------------------------------------------------
    def _stage_galactic_absorption(self, context: Mapping[str, StageResult]) -> StageResult:
        record = self._record
        if self.input.galactic_nh_1e22 is not None:
            return StageResult(
                status=PipelineStatus.COMPLETED,
                data={"tbabs_nh_1e22": float(self.input.galactic_nh_1e22), "source": "pipeline_input"},
            )
        if record.ra_deg is None or record.dec_deg is None:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                data={"warnings": ["no RA/Dec in catalog row; provide SwiftGRBInput.ra_deg/dec_deg"]},
                message="cannot resolve Galactic NH without coordinates",
            )
        result = resolve_galactic_absorption(
            record.ra_deg,
            record.dec_deg,
            cache_path=_contained_path(self.workspace, "galactic_nh_cache.csv"),
            query=self.nhtot_query,
        )
        path = write_json(
            _contained_path(self.workspace, "galactic_absorption", "galactic_nh.json"),
            {
                "ra_deg": record.ra_deg,
                "dec_deg": record.dec_deg,
                "tbabs_nh_1e22": result.tbabs_nh_1e22,
            },
        )
        return StageResult(
            status=PipelineStatus.COMPLETED,
            outputs={"galactic_nh": str(path)},
            data={"tbabs_nh_1e22": float(result.tbabs_nh_1e22), "source": "nhtot"},
        )

    # ------------------------------------------------------------------
    # Stage: lightcurve
    # ------------------------------------------------------------------
    def _stage_lightcurve(self, context: Mapping[str, StageResult]) -> StageResult:
        record = self._record
        parsed_dir = _contained_path(record.raw_dir, "burst_analyser", "parsed")
        series: list[dict[str, Any]] = []
        warnings: list[str] = []

        bat_paths = sorted(parsed_dir.glob(f"BAT_*{self._bat_binning}_{self._bat_band}_with_badbin.csv"))
        if bat_paths:
            table = read_badbin_table(bat_paths[0])
            flux = table["Flux"]
            positive = np.isfinite(flux) & (flux > 0)
            good = positive & ~table["BadBin"]
            series.append(
                {
                    "instrument": "BAT",
                    "mode": self._bat_binning,
                    "time": table["Time"][good],
                    "flux": flux[good],
                    "flux_err_low": np.abs(table["FluxNeg"][good]),
                    "flux_err_high": np.abs(table["FluxPos"][good]),
                    "n_bins": int(good.sum()),
                    "n_bad_removed": int((positive & table["BadBin"]).sum()),
                    "n_total": int(table["n_total"]),
                }
            )
        else:
            warnings.append("no BAT badbin table; BAT series skipped")

        for xrt_path in sorted(parsed_dir.glob(f"XRT_{self._xrt_band}_*_incbad.dat")):
            match = re.search(r"_XRTBand_([A-Za-z0-9]+?)_incbad", xrt_path.name)
            mode = match.group(1).upper() if match else "UNK"
            table = read_burst_analyser_dat(xrt_path)
            flux = table["Flux"]
            good = np.isfinite(flux) & (flux > 0)
            series.append(
                {
                    "instrument": "XRT",
                    "mode": mode,
                    "time": table["Time"][good],
                    "flux": flux[good],
                    "flux_err_low": np.abs(table["FluxNeg"][good]),
                    "flux_err_high": np.abs(table["FluxPos"][good]),
                    "n_bins": int(good.sum()),
                    "n_bad_removed": 0,
                    "n_total": int(table["n_total"]),
                }
            )
        series = [item for item in series if item["n_bins"] > 0]
        if not series:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                data={"warnings": warnings},
                message="no Burst Analyser series available",
            )

        key = safe_filename_token(record.merge_key)
        out_dir = _contained_path(self.workspace, "lightcurve")
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = _contained_path(out_dir, f"{key}_lc.npz")
        np.savez_compressed(
            npz_path,
            time=np.concatenate([item["time"] for item in series]),
            flux=np.concatenate([item["flux"] for item in series]),
            flux_err_low=np.concatenate([item["flux_err_low"] for item in series]),
            flux_err_high=np.concatenate([item["flux_err_high"] for item in series]),
            mode=np.asarray([item["mode"] for item in series for _ in range(item["n_bins"])]),
            instrument=np.asarray([item["instrument"] for item in series for _ in range(item["n_bins"])]),
            grb_name=record.merge_key,
            redshift=record.redshift if record.redshift is not None else np.nan,
            t0_met=record.trigger_met if record.trigger_met is not None else np.nan,
            t0_utc=record.trigger_utc or "",
        )
        figure_path = _contained_path(out_dir, f"{key}_lc.png")
        self._plot_lightcurve(series, figure_path)
        return StageResult(
            status=PipelineStatus.COMPLETED,
            outputs={"npz": str(npz_path), "figure": str(figure_path)},
            data={
                "series": [
                    {
                        name: value
                        for name, value in item.items()
                        if name not in {"time", "flux", "flux_err_low", "flux_err_high"}
                    }
                    for item in series
                ],
                "warnings": warnings,
            },
        )

    def _plot_lightcurve(
        self,
        series: Sequence[Mapping[str, Any]],
        path: Path,
        spectral: Mapping[str, Any] | None = None,
    ) -> None:
        """Research figure (flux / N_H / Gamma panels); empty panels are dropped."""
        from ...core.plotstyle import PALETTE, apply_style, save_figure

        apply_style()
        import matplotlib.pyplot as plt

        colors = {"WT": PALETTE["band"], "PC": "#000000", "UNK": PALETTE["muted"]}
        markers = {"WT": "s", "PC": "o", "UNK": "o"}
        draw_order = sorted(series, key=lambda item: item["mode"] != "WT")

        def style_for(instrument: str, mode: str) -> tuple[str, str]:
            # Research spec §8.2: BAT plots as blue circles regardless of binning,
            # XRT by readout mode (WT orange squares, PC black circles).
            if str(instrument).upper() == "BAT":
                return PALETTE["data"], "o"
            if mode in colors:
                return colors[mode], markers[mode]
            return colors["UNK"], markers["UNK"]

        spectral_panels = spectral is not None and bool(spectral.get("time"))
        heights = [4.4, 1.45, 1.45] if spectral_panels else [4.4]
        fig, axes = plt.subplots(
            len(heights),
            1,
            figsize=(10, 8.8 if spectral_panels else 4.8),
            sharex=True,
            gridspec_kw={"height_ratios": heights},
        )
        axes = np.atleast_1d(axes)
        errorbar_style = {"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.85, "markeredgewidth": 0.6}
        for item in draw_order:
            color, marker = style_for(str(item["instrument"]), str(item["mode"]))
            axes[0].errorbar(
                item["time"],
                item["flux"],
                yerr=[item["flux_err_low"], item["flux_err_high"]],
                fmt=marker,
                color=color,
                markersize=4.0,
                label=f"{item['instrument']} {item['mode']}",
                **errorbar_style,
            )
        record = self._record
        if record.bat_t90 is not None:
            for bound in record.bat_t90:
                axes[0].axvline(bound, color=PALETTE["reference"], ls="--", lw=0.8)
        if record.xrt_start_s is not None:
            axes[0].axvline(record.xrt_start_s, color=PALETTE["secondary"], ls="-.", lw=0.8)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Flux (erg cm$^{-2}$ s$^{-1}$)")
        axes[0].legend(loc="upper left", fontsize=8, frameon=False)
        z_text = f" (z={record.redshift})" if record.redshift is not None else ""
        axes[0].set_title(f"{record.canonical_name} BAT+XRT Flux Lightcurve{z_text}")
        if spectral_panels and spectral is not None:
            labels = ("N$_{H,int}$ (10$^{22}$ cm$^{-2}$)", "$\\Gamma$")
            for axis, key, label in zip(axes[1:], ("nh", "gamma"), labels):
                values = np.asarray(spectral[key], dtype=float)
                times = np.asarray(spectral["time"], dtype=float)
                finite = np.isfinite(values)
                axis.errorbar(
                    times[finite],
                    values[finite],
                    fmt="o",
                    color=PALETTE["tertiary"],
                    markersize=4.0,
                    **errorbar_style,
                )
                axis.set_ylabel(label)
        for axis in axes:
            axis.set_xscale("symlog", linthresh=1.0)
        axes[-1].set_xlabel("Time since T0 (s)")
        fig.tight_layout()
        save_figure(fig, path.with_suffix(""), formats=("png",))
        plt.close(fig)

    # ------------------------------------------------------------------
    # Stage: prompt
    # ------------------------------------------------------------------
    def _stage_prompt(self, context: Mapping[str, StageResult]) -> StageResult:
        payload = prompt_classification(self._record)
        path = write_json(_contained_path(self.workspace, "prompt", "prompt_classification.json"), payload)
        return StageResult(
            status=PipelineStatus.COMPLETED,
            outputs={"prompt": str(path)},
            data={"prompt": payload},
        )

    # ------------------------------------------------------------------
    # Stage: bblocks
    # ------------------------------------------------------------------
    def _stage_bblocks(self, context: Mapping[str, StageResult]) -> StageResult:
        lc_dir = _contained_path(self._downloads_dir, "lc")
        out_dir = _contained_path(self.workspace, "bblocks")
        segments: list[TimeResolvedSegment] = []
        warnings: list[str] = []
        per_mode: dict[str, Any] = {}
        for mode in ("WT", "PC"):
            src = _locate_first(
                lc_dir / f"{mode.lower()}sourcetotal_incbad.evt.gz",
                lc_dir / f"{mode.lower()}sourcetotal.evt.gz",
            )
            bkg = _locate_first(
                lc_dir / f"{mode.lower()}backtotal_incbad.evt.gz",
                lc_dir / f"{mode.lower()}backtotal.evt.gz",
            )
            if src is None or bkg is None:
                warnings.append(f"{mode}: source/background event files not found under {lc_dir}")
                per_mode[mode] = None
                continue
            try:
                src_times, gtis = _read_event_times(src)
                bkg_times, _ = _read_event_times(bkg)
            except (OSError, KeyError) as exc:
                warnings.append(f"{mode}: unreadable event file: {exc}")
                per_mode[mode] = None
                continue
            t0_met = self._record.trigger_met or _event_trigger_met(src)
            if t0_met is None:
                warnings.append(f"{mode}: event time origin is unknown; cannot publish trigger-relative segments")
                per_mode[mode] = None
                continue
            mode_segments: list[TimeResolvedSegment] = []
            snr_threshold = float(
                self.input.wt_snr_threshold if mode == "WT" and self.input.wt_snr_threshold is not None
                else self.input.pc_snr_threshold if mode == "PC" and self.input.pc_snr_threshold is not None
                else self._swift_config.segmentation.wt_snr_threshold if mode == "WT"
                else self._swift_config.segmentation.pc_snr_threshold
            )
            min_counts = 0.0 if mode == "WT" else float(
                self.input.pc_min_source_counts
                if self.input.pc_min_source_counts is not None
                else self._swift_config.segmentation.pc_min_source_counts
            )
            area_file = _locate_first(
                lc_dir / f"all{mode.lower()}back.area",
                lc_dir / f"all{mode.lower()}back.area.gz",
            )
            for chunk_start, chunk_stop in _event_chunks(src_times, gtis, mode):
                chunk_times = src_times[(src_times >= chunk_start) & (src_times <= chunk_stop)]
                if len(chunk_times) < 2:
                    continue
                p0 = self.input.bblock_p0 if self.input.bblock_p0 is not None else self._swift_config.segmentation.p0
                edges = np.asarray(bayesian_blocks(chunk_times, p0=p0), dtype=float)
                blocks: list[TimeResolvedSegment] = []
                for start, stop in zip(edges[:-1], edges[1:]):
                    src_counts = float(((src_times >= start) & (src_times < stop)).sum())
                    bkg_counts = float(((bkg_times >= start) & (bkg_times < stop)).sum())
                    alpha = _read_area_ratio(bkg, start, stop)
                    if not np.isfinite(alpha) and area_file is not None:
                        t0_met = _event_trigger_met(bkg)
                        alpha = _alpha_from_area_file(
                            area_file,
                            source_area=_source_area_from_event(bkg, start, stop),
                            time_s=((start + stop) / 2.0 - t0_met) if t0_met is not None else None,
                        )
                    if not np.isfinite(alpha) or alpha <= 0:
                        warnings.append(f"{mode}: missing valid area scaling for {start:.6f}--{stop:.6f}")
                        continue
                    blocks.append(
                        TimeResolvedSegment(
                            mode=mode,
                            index=0,
                            start=float(start - t0_met),
                            stop=float(stop - t0_met),
                            source_counts=src_counts,
                            background_counts=bkg_counts,
                            alpha=float(alpha),
                            snr=_segment_snr(src_counts, bkg_counts, alpha),
                        )
                    )
                mode_segments.extend(
                    merge_low_snr_segments(blocks, snr_threshold=snr_threshold, min_source_counts=min_counts)
                )
            mode_segments = [
                TimeResolvedSegment(
                    mode=segment.mode,
                    index=position,
                    start=segment.start,
                    stop=segment.stop,
                    source_counts=segment.source_counts,
                    background_counts=segment.background_counts,
                    alpha=segment.alpha,
                    snr=segment.snr,
                    merged=segment.merged,
                )
                for position, segment in enumerate(mode_segments)
            ]
            per_mode[mode] = [segment.to_payload() for segment in mode_segments]
            segments.extend(mode_segments)
        path = write_json(out_dir / "segments.json", {"modes": per_mode, "warnings": warnings})
        if not segments:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                outputs={"segments": str(path)},
                data={"segments": [], "per_mode": per_mode, "warnings": warnings},
                message="no Bayesian-block segments derived (XRT event files missing?)",
            )
        return StageResult(
            status=PipelineStatus.COMPLETED,
            outputs={"segments": str(path)},
            data={
                "segments": [segment.to_payload() for segment in segments],
                "per_mode": per_mode,
                "warnings": warnings,
            },
        )

    # ------------------------------------------------------------------
    # Stage: spectra
    # ------------------------------------------------------------------
    def _stage_spectra(self, context: Mapping[str, StageResult]) -> StageResult:
        segments = [segment_from_payload(payload) for payload in context["bblocks"].data.get("segments", [])]
        if self.input.max_segments_per_mode is not None:
            selected: list[TimeResolvedSegment] = []
            for mode in ("WT", "PC"):
                selected.extend([item for item in segments if item.mode == mode][: self.input.max_segments_per_mode])
            segments = selected
        if not segments:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                data={"manifest": [], "warnings": ["no segments from bblocks stage"]},
                message="spectra extraction skipped: no segments",
            )
        lc_dir = _contained_path(self._downloads_dir, "lc")
        interval0 = _locate_first(
            _contained_path(self._downloads_dir, "spec", "interval0"),
            _contained_path(self._record.raw_dir, "spectra", "interval0"),
            _contained_path(self._record.raw_dir, "spectra", "extracted", "interval0"),
            directory=True,
        )
        out_dir = _contained_path(self.workspace, "spectra")
        manifest: list[dict[str, Any]] = []
        warnings: list[str] = []
        self._prepared_segments = []
        for segment in segments:
            mode = segment.mode.lower()
            src_evt = _locate_first(
                lc_dir / f"{mode}sourcetotal_incbad.evt.gz",
                lc_dir / f"{mode}sourcetotal.evt.gz",
            )
            bkg_evt = _locate_first(
                lc_dir / f"{mode}backtotal_incbad.evt.gz",
                lc_dir / f"{mode}backtotal.evt.gz",
            )
            if src_evt is None or bkg_evt is None:
                warnings.append(f"{segment.key}: event files missing")
                continue
            seg_dir = _contained_path(out_dir, segment.key)
            seg_dir.mkdir(parents=True, exist_ok=True)
            stop = segment.stop + _RIGHT_EDGE_EPS_S
            event_t0 = self._record.trigger_met
            if event_t0 is None:
                warnings.append(f"{segment.key}: missing trigger MET; cannot select absolute event interval")
                continue
            try:
                with self.stage_environment("spectra") as environment:
                    xselect_kwargs: dict[str, Any] = {
                        "time_range": (segment.start + event_t0, stop + event_t0),
                        "time_format": "scc",
                        "overwrite": True,
                    }
                    if self.extractor is extract_products_with_xselect:
                        executable = self._xselect_executable()
                        if executable is not None and not environment.get("HEADAS"):
                            headas = executable.parent.parent
                            environment["HEADAS"] = str(headas)
                            environment["LHEA_DATA"] = str(headas / "refdata")
                            pfiles = self.workspace / ".pfiles" / "spectra"
                            environment["PFILES"] = f"{pfiles};{headas / 'syspfiles'}"
                            environment["PATH"] = f"{headas / 'bin'}:{environment.get('PATH', '')}"
                        xselect_kwargs.update(
                            xselect_executable=executable,
                            env=environment,
                            timeout=self._swift_config.data.request_timeout_s,
                        )
                    src_run = self.extractor(
                        src_evt, seg_dir, products="spectrum", prefix=self._record.merge_key,
                        label=f"{segment.key}_src", role="src", **xselect_kwargs,
                    )
                    bkg_run = self.extractor(
                        bkg_evt, seg_dir, products="spectrum", prefix=self._record.merge_key,
                        label=f"{segment.key}_bkg", role="bkg", **xselect_kwargs,
                    )
            except Exception as exc:  # noqa: BLE001 - external tool
                warnings.append(f"{segment.key}: xselect failed: {exc}")
                continue
            src_pha = src_run.spectrum
            bkg_pha = bkg_run.spectrum
            if src_pha is None or bkg_pha is None:
                warnings.append(f"{segment.key}: xselect produced no spectrum")
                continue
            rmf = interval0 / f"interval0{mode}.rmf" if interval0 else None
            arf = interval0 / f"interval0{mode}.arf" if interval0 else None
            responses_ok = rmf is not None and arf is not None and rmf.is_file() and arf.is_file()
            if not responses_ok:
                warnings.append(f"{segment.key}: interval0 responses missing; extraction retained but fit is blocked")
            grouped = _contained_path(seg_dir, f"{segment.key}_grouped_g{self._group_min_counts}.pha")
            grouping = grppha_hsp(
                infile=src_pha,
                outfile=grouped,
                min_counts=self._group_min_counts,
                rmf=rmf if rmf is not None and rmf.is_file() else None,
                arf=arf if arf is not None and arf.is_file() else None,
                bkg=bkg_pha,
                clobber=True,
            )
            if not bool(grouping.get("success")):
                warnings.append(
                    f"{segment.key}: grppha failed: {grouping.get('error') or grouping.get('message')}"
                )
                continue
            prepared = PreparedSpectrum(
                instrument="XRT",
                obsid=None,
                module=None,
                detector=segment.mode,
                source_id=segment.key,
                source_pha=src_pha,
                grouped_pha=grouped,
                background_pha=bkg_pha,
                arf=arf if arf is not None and arf.is_file() else None,
                rmf=rmf if rmf is not None and rmf.is_file() else None,
                group_min=self._group_min_counts,
                energy_range_keV=(0.3, 10.0),
            )
            manifest.append(
                {
                    "segment": segment.to_payload(),
                    "prepared": {
                        "instrument": prepared.instrument,
                        "obsid": prepared.obsid,
                        "module": prepared.module,
                        "detector": prepared.detector,
                        "source_id": prepared.source_id,
                        "source_pha": str(prepared.source_pha),
                        "grouped_pha": str(prepared.grouped_pha),
                        "background_pha": str(prepared.background_pha),
                        "arf": str(prepared.arf) if prepared.arf else None,
                        "rmf": str(prepared.rmf) if prepared.rmf else None,
                        "group_min": prepared.group_min,
                        "energy_range_keV": list(prepared.energy_range_keV),
                        "response_policy": self._swift_config.segmentation.response_policy,
                        "response_caveat": (
                            "interval0 response reused for this time-resolved segment; "
                            "calibration compatibility must be verified before final science use"
                        ),
                    },
                }
            )
            self._prepared_segments.append(prepared)
        manifest_path = write_json(out_dir / "manifest.json", {"segments": manifest, "warnings": warnings})
        if not manifest:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                outputs={"manifest": str(manifest_path)},
                data={"manifest": [], "warnings": warnings},
                message="no spectra extracted",
            )
        return StageResult(
            status=PipelineStatus.COMPLETED,
            outputs={"manifest": str(manifest_path)},
            data={"manifest": manifest, "warnings": warnings, "count": len(manifest)},
        )

    # ------------------------------------------------------------------
    # Stage: fit
    # ------------------------------------------------------------------
    def _stage_fit(self, context: Mapping[str, StageResult]) -> StageResult:
        manifest = context["spectra"].data.get("manifest", [])
        if not manifest:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                data={"index": [], "failure_logs": [], "warnings": ["no prepared spectra"]},
                message="fit skipped: no prepared spectra",
            )
        nh = context["galactic_absorption"].data.get("tbabs_nh_1e22")
        redshift = self._record.redshift
        if redshift is None:
            return StageResult(
                status=PipelineStatus.NEEDS_REVIEW,
                data={"index": [], "failure_logs": [], "warnings": ["redshift unresolved; ztbabs fit not run"]},
                message="fit skipped: a redshift is required for tbabs*ztbabs*cflux*powerlaw",
            )
        out_root = _contained_path(self.workspace, "fit")
        index: list[dict[str, Any]] = []
        failure_logs: list[dict[str, Any]] = []
        warnings: list[str] = []
        cached = {prepared.source_id: prepared for prepared in self._prepared_segments}
        for item in manifest:
            key = str(item["segment"]["key"])
            prepared = cached.get(key) or self._prepared_from_manifest(item)
            if prepared is None or prepared.grouped_pha is None or not prepared.grouped_pha.is_file():
                warnings.append(f"{key}: prepared spectrum unavailable; re-run the spectra stage")
                continue
            seg_dir = _contained_path(out_root, key)
            try:
                results = fit_prepared(
                    prepared,
                    outdir=seg_dir,
                    redshift=redshift,
                    galactic_nh_1e22=nh,
                    freeze_galactic_nh=True,
                    calculate_errors=True,
                    model_name=self.config.fitting.model_name,
                    stat_method=self.config.fitting.statistic,
                    error_delta_stat=self.config.fitting.error_delta_stat,
                    emin=(self.config.spectrum.fit_energy_range_keV or (0.3, 10.0))[0],
                    emax=(self.config.spectrum.fit_energy_range_keV or (0.3, 10.0))[1],
                    abundance=self.config.fitting.abundance,
                    cross_section=self.config.fitting.cross_section,
                    srcname=f"XRT_{key}",
                    instname="XRT",
                    plot_format="png",
                )
                path = write_json(_contained_path(seg_dir, "fit.json"), results)
                index.append(
                    {
                        "key": key,
                        "status": "ok",
                        "fit_json": str(path),
                        "statistics": results.get("statistics"),
                        "parameters": results.get("parameters"),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - XSPEC runtime
                failure_logs.append({"key": key, "error": repr(exc)})
                warnings.append(f"{key}: fit failed: {exc}")
        index_path = write_json(
            _contained_path(out_root, "index.json"), {"fits": index, "failure_logs": failure_logs}
        )
        return StageResult(
            status=PipelineStatus.COMPLETED if index else PipelineStatus.NEEDS_REVIEW,
            outputs={"index": str(index_path)},
            data={"index": index, "failure_logs": failure_logs, "warnings": warnings},
            message=None if index else "all segment fits failed (XSPEC available?)",
        )

    # ------------------------------------------------------------------
    # Stage: report
    # ------------------------------------------------------------------
    def _stage_report(self, context: Mapping[str, StageResult]) -> StageResult:
        record = self._record
        warnings: list[str] = []
        fit_index = context["fit"].data.get("index", [])
        spectral: dict[str, Any] | None = None
        rows: list[dict[str, Any]] = []
        manifest = {item["segment"]["key"]: item for item in context["spectra"].data.get("manifest", [])}
        for item in fit_index:
            try:
                payload = _json.loads(_manifest_path(item["fit_json"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                warnings.append(f"{item['key']}: unreadable fit.json")
                continue
            parameters = payload.get("parameters") or {}
            statistics = payload.get("statistics") or {}
            row: dict[str, Any] = {
                "segment": item["key"],
                "statistic": statistics.get("effective_statistic") or statistics.get("statistic"),
                "statistic_value": statistics.get("value"),
                "nh_1e22": parameters.get("zTBabs.nH"),
                "gamma": parameters.get("powerlaw.PhoIndex"),
                "lg10flux": parameters.get("cflux.lg10Flux"),
                "error_status": {
                    key: value.get("error_status")
                    for key, value in parameters.items()
                    if isinstance(value, Mapping)
                },
            }
            for name in ("nh_1e22", "gamma", "lg10flux"):
                value = row[name]
                if isinstance(value, Mapping):
                    row[name] = value.get("value")
            if row["nh_1e22"] is not None and row["gamma"] is not None and item["key"] in manifest:
                segment = manifest[item["key"]]["segment"]
                row["t_start"] = segment["start"]
                row["t_stop"] = segment["stop"]
            rows.append(row)
        timed_rows = [row for row in rows if "t_start" in row]
        if timed_rows:
            spectral = {
                "time": [(row["t_start"] + row["t_stop"]) / 2.0 for row in timed_rows],
                "nh": [float(row["nh_1e22"]) for row in timed_rows],
                "gamma": [float(row["gamma"]) for row in timed_rows],
            }

        out_dir = _contained_path(self.workspace, "report")
        spectral_path: Path | None = None
        if rows:
            buffer = io.StringIO()
            fieldnames = sorted({name for row in rows for name in row})
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            spectral_path = _contained_path(out_dir, "spectral_parameters.csv")
            spectral_path.write_text(buffer.getvalue(), encoding="utf-8")

        # Keep upstream lightcurve products immutable.  Final overlays are a
        # separate deliverable so they cannot invalidate the lightcurve cache.
        if spectral is not None:
            key = safe_filename_token(record.merge_key)
            npz_path = _contained_path(self.workspace, "lightcurve", f"{key}_lc.npz")
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as existing:
                    payload = {name: existing[name] for name in existing.files}
                payload.update(
                    {
                        "spectral_time": np.asarray(spectral["time"], dtype=float),
                        "spectral_nh": np.asarray(spectral["nh"], dtype=float),
                        "spectral_gamma": np.asarray(spectral["gamma"], dtype=float),
                    }
                )
                final_npz = _contained_path(out_dir, f"{key}_lc_with_spectra.npz")
                np.savez_compressed(final_npz, **payload)
            figure_path = _contained_path(out_dir, f"{key}_lc_with_spectra.png")
            series = self._series_from_npz(npz_path)
            if series:
                self._plot_lightcurve(series, figure_path, spectral=spectral)

        summary: dict[str, Any] = {
            "grb": record.to_payload(),
            "prompt": context["prompt"].data.get("prompt"),
            "galactic_nh_1e22": context["galactic_absorption"].data.get("tbabs_nh_1e22"),
            "segments": context["bblocks"].data.get("per_mode"),
            "spectral_parameters": rows,
            "fit_failures": context["fit"].data.get("failure_logs", []),
            "warnings": warnings,
        }
        summary_path = write_json(_contained_path(out_dir, "grb_summary.json"), summary)
        text_path = _contained_path(out_dir, "summary.txt")
        text_path.write_text(self._render_summary(summary), encoding="utf-8")
        outputs = {"summary": str(summary_path), "summary_text": str(text_path)}
        if spectral_path is not None:
            outputs["spectral_parameters"] = str(spectral_path)
        return StageResult(
            status=PipelineStatus.COMPLETED,
            outputs=outputs,
            data={"summary": summary, "warnings": warnings},
        )

    def _series_from_npz(self, npz_path: Path) -> list[dict[str, Any]] | None:
        if not npz_path.is_file():
            return None
        with np.load(npz_path, allow_pickle=False) as payload:
            time = payload["time"]
            flux = payload["flux"]
            instrument = payload["instrument"]
            mode = payload["mode"]
            err_low = payload["flux_err_low"]
            err_high = payload["flux_err_high"]
        series: list[dict[str, Any]] = []
        for instrument_name in sorted({str(value) for value in instrument}):
            for mode_name in sorted({str(value) for value in mode}):
                mask = (instrument == instrument_name) & (mode == mode_name)
                if not mask.any():
                    continue
                series.append(
                    {
                        "instrument": instrument_name,
                        "mode": mode_name,
                        "time": time[mask],
                        "flux": flux[mask],
                        "flux_err_low": err_low[mask],
                        "flux_err_high": err_high[mask],
                    }
                )
        return series

    def _render_summary(self, summary: Mapping[str, Any]) -> str:
        grb = summary.get("grb", {})
        prompt = summary.get("prompt") or {}
        lines = [
            f"GRB {grb.get('canonical_name', self.input.target_id)}",
            f"  redshift: {grb.get('redshift')} ({grb.get('redshift_method')}, {grb.get('redshift_source')})",
            f"  trigger UTC: {grb.get('trigger_utc')}",
            f"  BAT T90 window (s): {grb.get('bat_t90')}",
            f"  prompt classification: {prompt.get('in_prompt', 'UNDECIDED')}",
            f"  Galactic NH (1e22 cm^-2): {summary.get('galactic_nh_1e22')}",
        ]
        for row in summary.get("spectral_parameters") or []:
            lines.append(
                f"  {row['segment']}: nH={row.get('nh_1e22')}  Gamma={row.get('gamma')}  "
                f"lg10Flux={row.get('lg10flux')}  {row.get('statistic')}={row.get('statistic_value')}"
            )
        for failure in summary.get("fit_failures") or []:
            lines.append(f"  fit failed: {failure['key']}: {failure['error']}")
        return "\n".join(lines) + "\n"


def _default_config(group_min_counts: int = 20) -> InstrumentConfig:
    config = SwiftGRB()
    if group_min_counts == config.spectrum.group_min_counts:
        return config
    # Explicit CLI grouping remains an override while all other defaults stay
    # in the reusable Swift preset.
    from dataclasses import replace

    return replace(config, group_min_counts=group_min_counts, spectrum=replace(config.spectrum, group_min_counts=group_min_counts))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m jinwu.swift.grb.pipeline --root <ws> --grb <key>``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the jinwu Swift BAT+XRT GRB pipeline for one GRB")
    parser.add_argument("--root", required=True, help="sample workspace root (contains catalog/ and raw/)")
    parser.add_argument("--grb", required=True, help="GRB name or merge key, e.g. 050904A or 'GRB 050904'")
    parser.add_argument("--redshift", type=float, default=None, help="override the catalog redshift")
    parser.add_argument("--ra", type=float, default=None, help="override right ascension (deg)")
    parser.add_argument("--dec", type=float, default=None, help="override declination (deg)")
    parser.add_argument("--nh", type=float, default=None, help="Galactic NH in 1e22 cm^-2 (skips the nhtot query)")
    parser.add_argument("--until", default=None, help="stop after this stage")
    parser.add_argument("--no-resume", action="store_true", help="ignore cached stage manifests")
    parser.add_argument("--fetch", action="store_true", help="enable the Burst Analyser fetcher (requires swifttools)")
    parser.add_argument("--output-dir", default=None, help="separate pipeline output directory (keeps inputs read-only)")
    parser.add_argument("--catalog", default=None, help="merged catalog CSV (overrides <root>/catalog)")
    parser.add_argument("--raw-products-dir", default=None, help="root containing local raw GRB products")
    parser.add_argument("--xrt-products-dir", default=None, help="local UKSSDC XRT product directory")
    parser.add_argument("--request-xrt", action="store_true", help="explicitly submit one UKSSDC XRT product request")
    parser.add_argument("--poll-xrt", action="store_true", help="resume and poll an existing saved UKSSDC request")
    parser.add_argument("--xrt-user", default=None, help="registered UKSSDC XRT-product account (or SWIFT_XRT_USER)")
    parser.add_argument("--max-segments-per-mode", type=int, default=None, help="extract only the first N WT and PC segments")
    args = parser.parse_args(argv)

    # Resolve and validate the workspace before it reaches the pipeline input.
    root = Path(args.root).expanduser().resolve(strict=True)
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    input_data = SwiftGRBInput(
        target_id=args.grb,
        root=root,
        output_root=args.output_dir,
        catalog_path=args.catalog,
        raw_products_dir=args.raw_products_dir,
        xrt_products_dir=args.xrt_products_dir,
        redshift=args.redshift,
        ra_deg=args.ra,
        dec_deg=args.dec,
        galactic_nh_1e22=args.nh,
        fetcher=SwiftBurstAnalyserFetcher() if args.fetch else None,
        request_xrt=args.request_xrt,
        poll_xrt=args.poll_xrt,
        xrt_user=args.xrt_user,
        max_segments_per_mode=args.max_segments_per_mode,
    )
    pipeline = SwiftGRBPipeline(input_data, config=_default_config())
    result = pipeline.run(until=args.until, resume=not args.no_resume)
    print(f"status: {result.status.value}")
    if result.message:
        print(result.message)
    for stage, outputs in result.products.items():
        for label, path in outputs.items():
            print(f"  {stage}.{label}: {path}")
    return 0 if result.status is PipelineStatus.COMPLETED else 1
