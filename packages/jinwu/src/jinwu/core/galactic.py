"""Coordinate-based Galactic absorption lookup with reproducible caching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable

__all__ = ["GalacticAbsorptionResult", "resolve_galactic_absorption"]


_SCHEMA_VERSION = 1
_SERVICE = "swift_ukssdc_nhtot"
_ALGORITHM_VERSION = "nhtot-weighted-v1"


@dataclass(frozen=True, slots=True)
class GalacticAbsorptionResult:
    """Validated Willingale total Galactic column for one ICRS position."""

    ra_deg: float
    dec_deg: float
    equinox: int
    nhi_weighted_cm2: float | None
    nh2_weighted_cm2: float | None
    nhtot_weighted_cm2: float
    tbabs_nh_1e22: float
    service: str
    algorithm_version: str
    queried_at_utc: str
    service_ra: str | None = None
    service_dec: str | None = None
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_coordinates(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    ra = float(ra_deg)
    dec = float(dec_deg)
    if not math.isfinite(ra) or not 0.0 <= ra < 360.0:
        raise ValueError("ra_deg must be finite and in [0, 360)")
    if not math.isfinite(dec) or not -90.0 <= dec <= 90.0:
        raise ValueError("dec_deg must be finite and in [-90, 90]")
    return ra, dec


def _same_query(
    payload: dict[str, Any],
    ra: float,
    dec: float,
    equinox: int,
    service: str,
) -> bool:
    return (
        payload.get("schema_version") == _SCHEMA_VERSION
        and payload.get("service") == service
        and payload.get("algorithm_version") == _ALGORITHM_VERSION
        and int(payload.get("equinox", -1)) == int(equinox)
        and math.isclose(float(payload.get("ra_deg", math.nan)), ra, abs_tol=1e-12)
        and math.isclose(float(payload.get("dec_deg", math.nan)), dec, abs_tol=1e-12)
    )


def _read_cache(
    path: Path,
    ra: float,
    dec: float,
    equinox: int,
    service: str,
) -> GalacticAbsorptionResult | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _same_query(payload, ra, dec, equinox, service):
            return None
        data = dict(payload["result"])
        data["cache_hit"] = True
        result = GalacticAbsorptionResult(**data)
        if not math.isfinite(result.nhtot_weighted_cm2) or result.nhtot_weighted_cm2 <= 0:
            return None
        return result
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, result: GalacticAbsorptionResult, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "service": result.service,
        "algorithm_version": result.algorithm_version,
        "ra_deg": result.ra_deg,
        "dec_deg": result.dec_deg,
        "equinox": result.equinox,
        "result": result.to_dict(),
        "raw_result": raw,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_galactic_absorption(
    ra_deg: float,
    dec_deg: float,
    *,
    cache_path: str | Path,
    equinox: int = 2000,
    service: str = _SERVICE,
    timeout_s: float = 30.0,
    use_cache: bool = True,
    query: Callable[..., dict[str, Any]] | None = None,
) -> GalacticAbsorptionResult:
    """Resolve and cache weighted total Galactic NH for one sky position.

    The service returns columns in atoms cm^-2. ``tbabs_nh_1e22`` is the
    corresponding XSPEC TBabs value after division by 1e22.
    """

    ra, dec = _validate_coordinates(ra_deg, dec_deg)
    service_name = str(service).strip()
    if not service_name:
        raise ValueError("service must be a non-empty identifier")
    cache = Path(cache_path).expanduser().resolve()
    if use_cache:
        cached = _read_cache(cache, ra, dec, int(equinox), service_name)
        if cached is not None:
            return cached

    if query is None:
        from .utils import nhtot

        query = nhtot
    raw = query(ra, dec, equinox=int(equinox), timeout=float(timeout_s))
    if not isinstance(raw, dict) or not raw.get("ok"):
        detail = raw.get("error", "unknown error") if isinstance(raw, dict) else repr(raw)
        raise RuntimeError(f"Galactic nhtot query failed for ({ra}, {dec}): {detail}")

    nhtot_value = raw.get("nhtot_weighted")
    try:
        nhtot = float(nhtot_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("nhtot_weighted is missing or non-numeric") from exc
    if not math.isfinite(nhtot) or not 1e17 <= nhtot <= 1e25:
        raise ValueError(f"nhtot_weighted is outside the accepted physical range: {nhtot}")

    def optional_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    result = GalacticAbsorptionResult(
        ra_deg=ra,
        dec_deg=dec,
        equinox=int(equinox),
        nhi_weighted_cm2=optional_float(raw.get("nhi_weighted")),
        nh2_weighted_cm2=optional_float(raw.get("nh2_weighted")),
        nhtot_weighted_cm2=nhtot,
        tbabs_nh_1e22=nhtot / 1e22,
        service=service_name,
        algorithm_version=_ALGORITHM_VERSION,
        queried_at_utc=datetime.now(timezone.utc).isoformat(),
        service_ra=str(raw.get("ra")) if raw.get("ra") is not None else None,
        service_dec=str(raw.get("dec")) if raw.get("dec") is not None else None,
    )
    _write_cache(cache, result, raw)
    return replace(result, cache_hit=False)
