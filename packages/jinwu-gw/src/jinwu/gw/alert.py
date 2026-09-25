"""LVK alert and GraceDB sky-map acquisition."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from jinwu.core.time import Time

from .gracedb import GraceDBClient, normalize_superevent_id
from .models import GWEvent, scalar_time



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versioned_path(path: Path, digest: str) -> Path:
    """Keep a prior alert/map file when the same logical name is updated."""
    if not path.exists() or _sha256(path) == digest:
        return path
    name = path.name
    for suffix in (".fits.gz", ".fit.gz", ".fits", ".fit"):
        if name.lower().endswith(suffix):
            stem = name[: -len(suffix)]
            return path.with_name(f"{stem}_{digest[:12]}{suffix}")
    return path.with_name(f"{path.stem}_{digest[:12]}{path.suffix}")


# Fake-IP DNS ranges used by local transparent proxies (Clash/mihomo 等)：
# 公网主机在本机解析为 198.18.0.0/15 保留段地址，实际流量由代理接管转发。
# 仅在用户显式设置 JINWU_GW_ALLOW_PROXY_DNS=1 时放行该段；环回/私有/链路本地
# 地址在任何情况下都拒绝。
_PROXY_DNS_RANGE = ipaddress.ip_network("198.18.0.0/15")


def _allow_proxy_fakeip_dns() -> bool:
    import os

    return os.environ.get("JINWU_GW_ALLOW_PROXY_DNS", "").strip().lower() not in ("", "0", "false")


def _validate_http_url(url: str) -> None:
    """Reject non-public http(s) targets before any request is issued.

    Server-side fetches must only reach public hosts: the scheme is restricted
    to http/https and every address the hostname resolves to must be globally
    routable, which excludes localhost, loopback, private, link-local and
    reserved ranges.  Addresses inside ``198.18.0.0/15`` are rejected as
    reserved too, unless ``JINWU_GW_ALLOW_PROXY_DNS=1`` opts into a local
    transparent-proxy DNS that maps public hosts into that range.
    """
    parsed = urlparse(str(url))
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"URL must use http:// or https://: {url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"URL has no hostname: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve host {host!r}: {exc}") from exc
    allow_proxy_dns = _allow_proxy_fakeip_dns()
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_global:
            continue
        if allow_proxy_dns and address in _PROXY_DNS_RANGE:
            continue
        raise ValueError(
            f"host {host!r} resolves to the non-public address {address}; refused"
        )


def _event_from_notice(payload: dict[str, Any], *, source: str | None = None) -> GWEvent:
    event = payload.get("event") or {}
    sid = str(payload.get("superevent_id") or payload.get("superevent") or event.get("superevent_id") or "local")
    time_value = (
        event.get("time")
        or event.get("event_time")
        or payload.get("time")
        or payload.get("event_time")
    )
    alert_type = str(payload.get("alert_type") or ("SUPEREVENT" if payload.get("t_0") is not None else "UNKNOWN"))
    labels = payload.get("labels") or ()
    if isinstance(labels, str):
        labels = (labels,)
    normalized_labels = {str(label).upper() for label in labels}
    declared_status = str(payload.get("status") or event.get("status") or "").lower()
    status = (
        "retracted"
        if alert_type.upper() in {"RETRACTION", "RETRACTED", "WITHDRAWAL"}
        or normalized_labels.intersection({"RETRACTED", "RETRACTION", "WITHDRAWN", "WITHDRAWAL"})
        or declared_status in {"retracted", "withdrawn", "withdrawal"}
        else "active"
    )
    if time_value is None and payload.get("t_0") is not None:
        # GraceDB superevent serializations carry the event time as GPS
        # seconds under ``t_0`` instead of the GCN notice's ISO ``event.time``.
        time_value = Time(float(payload["t_0"]), format="ligo", scale="utc")
    if time_value is None and status != "retracted":
        raise ValueError("alert JSON does not contain an event time")
    return GWEvent(
        superevent_id=sid,
        event_time=scalar_time(time_value) if time_value is not None else None,
        alert_type=alert_type,
        status=status,
        notice_version=str(
            payload.get("version")
            or payload.get("alert_version")
            or payload.get("time_created")
            or payload.get("created")
            or ""
        ) or None,
        notice_source=source,
        metadata={key: value for key, value in payload.items() if key not in {"event", "skymap"}},
    )


def read_notice(path: str | Path) -> tuple[GWEvent, dict[str, Any]]:
    """Read one JSON alert notice from disk."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("alert notice must be a JSON object")
    return _event_from_notice(payload, source=str(path)), payload


def fetch_http_bytes(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    max_redirects: int = 5,
    retries: int = 3,
) -> tuple[bytes, str]:
    """Fetch public bytes while validating every redirect destination.

    Each request has a bounded timeout and transient connection failures are
    retried at the same already-validated URL.  Redirects are never delegated
    to Requests, so every hop receives an independent public-address check.
    """
    client = session or requests.Session()
    current = str(url)
    for _ in range(int(max_redirects) + 1):
        _validate_http_url(current)
        response = None
        last_error: requests.RequestException | None = None
        for _attempt in range(max(1, int(retries))):
            try:
                response = client.get(current, timeout=timeout, allow_redirects=False)
                break
            except requests.RequestException as exc:
                last_error = exc
        if response is None:
            assert last_error is not None
            raise last_error
        if 300 <= response.status_code < 400 and response.headers.get("Location"):
            current = urljoin(current, response.headers["Location"])
            continue
        response.raise_for_status()
        return bytes(response.content), current
    raise requests.TooManyRedirects(f"too many redirects while fetching {url!r}")


def fetch_skymap_url(
    url: str,
    destination: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    filename: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Download one FITS sky map and return its checksum provenance."""
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    content, final_url = fetch_http_bytes(url, session=client, timeout=timeout)
    name = filename or Path(urlparse(url).path).name or "skymap.fits"
    if not name.lower().endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        name = f"{name}.fits"
    digest = hashlib.sha256(content).hexdigest()
    path = _versioned_path(destination / name, digest)
    path.write_bytes(content)
    return path, {"source_url": final_url, "sha256": digest, "filename": path.name}


def fetch_superevent(
    superevent_id: str,
    destination: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    notice_version: str | None = None,
    client: GraceDBClient | None = None,
) -> tuple[GWEvent, Path | None, dict[str, Any]]:
    """Fetch a requested or newest public GraceDB alert and its sky map."""
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sid = normalize_superevent_id(superevent_id)
    api = client or GraceDBClient()
    notice = api.select_notice(sid, notice_version)
    event = _event_from_notice(notice.payload, source=notice.url)
    event = GWEvent(
        event.superevent_id, event.event_time, alert_type=event.alert_type,
        status=event.status, notice_version=notice.filename,
        notice_source=notice.url, metadata=event.metadata,
    )
    notice_provenance = {
        "notice_url": notice.url,
        "notice_filename": notice.filename,
        "notice_sha256": hashlib.sha256(notice.content).hexdigest(),
        "notice_downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if event.status == "retracted":
        return event, None, notice_provenance
    path, map_provenance = skymap_from_notice(event, notice.payload, destination, session=session, timeout=timeout)
    if path is None:
        raise FileNotFoundError(
            f"GraceDB notice {notice.filename} has no supported sky map; use --skymap"
        )
    return event, path, {**notice_provenance, **map_provenance}


def skymap_from_notice(
    event: GWEvent,
    payload: dict[str, Any],
    destination: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> tuple[Path | None, dict[str, Any]]:
    """Resolve an embedded, URL, or GraceDB-referenced map in a notice."""
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    event_data = payload.get("event") or {}
    value = event_data.get("skymap") or payload.get("skymap")
    if isinstance(value, dict):
        value = value.get("url") or value.get("data")
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return fetch_skymap_url(
            value,
            destination,
            session=session,
            timeout=timeout,
            filename=f"{event.superevent_id}_skymap.fits",
        )
    if isinstance(value, str) and value.strip():
        local_candidate = Path(value).expanduser()
        try:
            if local_candidate.is_file():
                path = local_candidate.resolve()
                return path, {"source": str(path), "sha256": _sha256(path), "local": True}
        except OSError:
            # Long Base64 payloads are valid notice values but cannot be
            # passed through ``Path.stat`` on filesystems with NAME_MAX.
            pass
        encoded = "".join(value.split(",", 1)[-1].split())
        try:
            data = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("notice sky map is neither a URL nor valid Base64") from exc
        digest = hashlib.sha256(data).hexdigest()
        path = _versioned_path(destination / f"{event.superevent_id}_skymap.multiorder.fits", digest)
        path.write_bytes(data)
        return path, {"embedded": True, "sha256": digest, "filename": path.name}
    return None, {}
