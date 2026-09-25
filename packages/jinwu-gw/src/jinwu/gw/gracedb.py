"""Read-only public GraceDB access used by :mod:`jinwu.gw`.

The wrapper deliberately exposes only read operations.  It uses the official
``ligo-gracedb`` client for GraceDB discovery while keeping notice selection
and provenance in jinwu's explicit data contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable
from urllib.parse import urlparse

from .models import scalar_time


_SUPEREVENT_PATTERN = re.compile(r"^(?:MS|TS|S)\d{6}[a-z]+$", re.IGNORECASE)
_NOTICE_PATTERN = re.compile(
    r"^(?P<sid>(?:MS|TS|S)\d{6}[a-z]+)-(?P<kind>preliminary|initial|update|retraction)\.json(?:,(?P<version>\d+))?$",
    re.IGNORECASE,
)


def normalize_superevent_id(value: str) -> str:
    """Return a validated GraceDB superevent ID from an ID or public URL."""
    text = str(value).strip()
    parsed = urlparse(text)
    if parsed.scheme:
        parts = [part for part in parsed.path.split("/") if part]
        try:
            text = parts[parts.index("superevents") + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f"not a GraceDB superevent URL: {value!r}") from exc
    if not _SUPEREVENT_PATTERN.fullmatch(text):
        raise ValueError(f"invalid GraceDB superevent ID: {value!r}")
    return text[:2].upper() + text[2:].lower() if text[:2].upper() in {"MS", "TS"} else "S" + text[1:].lower()


@dataclass(frozen=True, slots=True)
class GraceDBNotice:
    """One public JSON alert notice plus immutable file identity."""

    filename: str
    url: str
    payload: dict[str, Any]
    content: bytes

    @property
    def created(self):
        value = self.payload.get("time_created") or self.payload.get("created")
        if value is None:
            raise ValueError(f"GraceDB notice {self.filename} has no time_created")
        return scalar_time(value)


class GraceDBClient:
    """Small read-only façade over :class:`ligo.gracedb.rest.GraceDb`.

    ``backend`` is injectable for offline tests.  With its default, public
    anonymous access is used and no credential discovery is required.
    """

    def __init__(self, *, backend: Any | None = None, max_results: int = 100):
        if backend is None:
            try:
                from ligo.gracedb.rest import GraceDb
            except ImportError as exc:  # pragma: no cover - package dependency
                raise RuntimeError("jinwu-gw needs ligo-gracedb for GraceDB access") from exc
            backend = GraceDb(force_noauth=True, retries=3)
        self._backend = backend
        self.max_results = int(max_results)
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")

    def search(self, query: str = "", *, max_results: int | None = None) -> list[dict[str, Any]]:
        """Search public superevents, returning at most ``max_results`` rows."""
        limit = self.max_results if max_results is None else int(max_results)
        if limit <= 0:
            raise ValueError("max_results must be positive")
        response = self._backend.superevents(query=query, max_results=limit)
        if hasattr(response, "json"):
            payload = response.json()
            if isinstance(payload, dict):
                rows: Iterable[Any] = payload.get("superevents", payload.get("results", ()))
            else:
                rows = payload
        else:
            rows = response
        return [dict(row) for row in rows if isinstance(row, dict)][:limit]

    def metadata(self, superevent_id: str) -> dict[str, Any]:
        """Return the public superevent representation."""
        response = self._backend.superevent(normalize_superevent_id(superevent_id))
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("GraceDB superevent response is not an object")
        return dict(payload)

    def files(self, superevent_id: str) -> dict[str, str]:
        """Return public file names and their immutable download URLs."""
        sid = normalize_superevent_id(superevent_id)
        # GraceDB's public ``files`` decorator infers a superevent from its
        # S/MS/TS identifier.  Passing ``is_superevent`` again conflicts with
        # ligo-gracedb 2.x's wrapper, so keep this call positional.
        response = self._backend.files(sid)
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("GraceDB file listing is not an object")
        return {str(name): str(url) for name, url in payload.items()}

    def file_bytes(self, superevent_id: str, filename: str) -> bytes:
        """Read one public GraceDB file through the official client."""
        sid = normalize_superevent_id(superevent_id)
        response = self._backend.files(sid, filename)
        # ``ligo-gracedb`` returns urllib3's ``HTTPResponse`` for file bytes,
        # whereas injected clients and older releases expose ``content``.
        payload = getattr(response, "content", None)
        if payload is None:
            payload = getattr(response, "data", None)
        if payload is None and hasattr(response, "read"):
            payload = response.read()
        if payload is None:
            raise ValueError(f"GraceDB file response for {filename!r} has no byte payload")
        return bytes(payload)

    def notices(self, superevent_id: str) -> list[GraceDBNotice]:
        """Read every public JSON alert notice attached to a superevent."""
        sid = normalize_superevent_id(superevent_id)
        result: list[GraceDBNotice] = []
        for filename, url in self.files(sid).items():
            match = _NOTICE_PATTERN.fullmatch(filename)
            if match is None or normalize_superevent_id(match.group("sid")) != sid:
                continue
            content = self.file_bytes(sid, filename)
            try:
                payload = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid GraceDB notice {filename}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"GraceDB notice {filename} is not a JSON object")
            result.append(GraceDBNotice(filename, url, payload, content))
        return result

    def select_notice(self, superevent_id: str, notice_version: str | None = None) -> GraceDBNotice:
        """Choose a requested notice, or the newest by public creation time."""
        notices = self.notices(superevent_id)
        if notice_version is not None:
            for notice in notices:
                if notice.filename == notice_version:
                    return notice
            raise FileNotFoundError(f"GraceDB notice {notice_version!r} is not available")
        if not notices:
            raise FileNotFoundError(f"no public JSON notices for {superevent_id}")
        return max(notices, key=lambda notice: (notice.created, notice.filename))
