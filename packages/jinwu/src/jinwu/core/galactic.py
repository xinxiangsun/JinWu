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
        """导出全部结果字段 / Export all result fields as a plain dictionary.

        保留柱密度单位约定、服务来源、查询时间和缓存命中标记。
        Preserve column-unit conventions, provenance, query time and cache flag.
        """
        return asdict(self)


def _validate_coordinates(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """校验度数坐标 / Validate numeric sky coordinates in degrees.

    返回 float 的 (RA, Dec)；RA 范围 [0, 360)，Dec 范围 [-90, 90]。
    不自动取模或转换坐标系；非有限值或越界值抛出 ValueError。
    Return float (RA, Dec), requiring RA in [0, 360) and Dec in [-90, 90].
    No wrapping or frame conversion; invalid/nonfinite values raise ValueError.
    """
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
    """判断缓存查询身份是否相符 / Compare a cache's query identity.

    比较 schema、服务、算法版本、分点年份及坐标；坐标使用 math.isclose
    的默认相对容差及 abs_tol=1e-12 度。格式损坏可能抛出转换异常，
    由调用缓存读取器处理。
    Match schema, service, algorithm, equinox and coordinates. Coordinates use
    math.isclose with its default relative tolerance and abs_tol=1e-12 degrees.
    Malformed numeric fields may raise conversion errors for the reader to catch.
    """
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
    """读取匹配且有效的缓存 / Read a matching, valid absorption cache.

    路径不存在、查询身份不同、格式损坏或总柱密度非正/非有限时返回 None。
    成功时返回 cache_hit=True 的结果；不联网，也不更新缓存。
    Return None for absent, mismatching, malformed or invalid caches. A valid
    result has cache_hit=True; no network request or cache write occurs here.
    """
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
    """保存查询结果与原始响应 / Save a result and the original service response.

    创建父目录，先写同目录 .tmp 文件再替换目标 JSON。无返回值；
    I/O 或 JSON 序列化异常向上传递。路径由调用者显式给定。
    Create parents, write a sibling .tmp file, then replace the destination
    JSON. Return None; I/O/serialization errors propagate. Path is explicit.
    """
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
    """查询并缓存银河系总氢柱密度 / Resolve and cache total Galactic NH.

    Parameters
    ----------
    ra_deg, dec_deg : float
        视线坐标，单位度；不隐式转换坐标系，范围见 _validate_coordinates。
        Sky coordinates in degrees; no implicit frame conversion.
    cache_path : str or Path
        显式 JSON 缓存位置；新查询成功后创建父目录并写入结果。
        Explicit JSON cache path; successful fresh queries create/update it.
    equinox : int
        传给服务的分点年份，默认 2000。不是自行执行坐标岁差转换。
        Equinox year passed to the service (default 2000), not a local precession.
    service : str
        用于结果与缓存匹配的非空服务标识，不用于选择查询后端。
        Nonempty provenance/cache identifier; does not select a query backend.
    timeout_s : float
        传给查询函数的超时秒数 / Timeout passed to the query, in seconds.
    use_cache : bool
        True 时优先读取身份匹配的缓存；False 跳过读取，仍写新结果。
        Read a matching cache first if True; False still writes fresh results.
    query : callable or None
        可注入查询函数，签名须接受 (ra, dec, equinox=..., timeout=...)；
        默认调用 core.utils.nhtot，返回含 ok 和 nhtot_weighted 的 dict。
        Optional query implementation; defaults to core.utils.nhtot and must
        accept those arguments and return a dict with ok/nhtot_weighted.

    Returns
    -------
    GalacticAbsorptionResult
        柱密度单位 atoms cm^-2；tbabs_nh_1e22 = nhtot_weighted / 1e22，
        对应 XSPEC TBabs 的数值约定，并保留来源与 UTC 查询时间。
        Columns in atoms cm^-2, plus the XSPEC TBabs value divided by 1e22,
        provenance and UTC query time. Optional unavailable components are None.

    Raises
    ------
    ValueError
        坐标或服务标识无效，或总柱密度缺失/超出 [1e17, 1e25] cm^-2。
        Invalid coordinates/service or missing/out-of-range total column.
    RuntimeError
        查询未返回成功状态 / The service query does not report success.

    Notes
    -----
    缓存按查询身份匹配，不设过期时限；读取失败会尝试重新查询。
    Cache identity is checked without an expiry policy; unreadable caches
    fall back to a fresh query. Service and file errors otherwise propagate.
    """

    # 方法：按视线方向查询 Swift/UKSSDC nH 服务，取 Willingale 口径的全银经柱密度
    #       NHTOT（中性氢 + 分子氢 H2 加权平均，atoms cm^-2），并把 XSPEC TBabs 的
    #       nH 换算为 NHTOT/1e22（TBabs.nH 单位 10^22 cm^-2）；结果按坐标+服务+
    #       算法版本缓存。注意：该口径包含 H2，与仅用 HI4PI 21cm 的 NH 不同。
    # 参考：Willingale, Hands, Warwick, Page, O'Brien & Aungier, 2013, MNRAS 431, 394
    #       (doi:10.1093/mnras/stt125, arXiv:1304.3330)；服务实现见本地
    #       jinwu/core/utils.py nhtot()（swift_ukssdc_nhtot）；
    #       HI 21cm 巡天口径另见 HI4PI Collaboration, 2016, A&A 594, A116
    #       (doi:10.1051/0004-6361/201629178)（不含 H2，仅作对照）。

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
        """保留有限可选值 / Return a finite optional value, otherwise None."""
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
