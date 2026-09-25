"""Offline security and provenance regressions for :mod:`jinwu.gw.alert`.

Covers the public-host SSRF gate (``_validate_http_url``) required for
server-side URL fetches, and the GraceDB superevent JSON contract (GPS
``t_0`` event times, RETRACTED labels).  All tests are offline: DNS is
monkeypatched and no HTTP request is attempted.

Run in the ``hea`` environment::

    conda run -n hea python -m pytest test/test_gw_alert_security.py -q -p no:cacheprovider
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from jinwu.gw.alert import _event_from_notice, _validate_http_url
from jinwu.gw.gracedb import GraceDBClient
from jinwu.gw.models import scalar_time
from jinwu.core.time import Time


def _fake_resolution(monkeypatch, address: str):
    parsed_host = None

    def fake_getaddrinfo(host, port, *args, **kwargs):
        nonlocal parsed_host
        parsed_host = host
        family = 6 if ":" in address else 2
        return [(family, socket.SOCK_STREAM, 6, "", (address, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return lambda: parsed_host


def test_validate_http_url_accepts_public_host(monkeypatch):
    host = _fake_resolution(monkeypatch, "93.184.216.34")
    _validate_http_url("https://example.org/sky.multiorder.fits")
    assert urlparse("https://example.org").hostname == "example.org"
    assert host() == "example.org"


@pytest.mark.parametrize("address", [
    "127.0.0.1",       # loopback
    "10.1.2.3",        # private
    "172.16.0.9",      # private
    "192.168.1.1",     # private
    "169.254.1.1",     # link-local
    "0.0.0.0",         # unspecified
    "100.64.0.1",      # shared address space
    "192.0.2.1",       # documentation (reserved)
    "::1",             # IPv6 loopback
    "fc00::1",         # IPv6 unique local
])
def test_validate_http_url_rejects_non_public_addresses(monkeypatch, address):
    _fake_resolution(monkeypatch, address)
    with pytest.raises(ValueError, match="non-public"):
        _validate_http_url(f"https://internal.example.test/sky.fits")


def test_validate_http_url_rejects_non_http_scheme():
    with pytest.raises(ValueError, match="http"):
        _validate_http_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="http"):
        _validate_http_url("ftp://example.org/sky.fits")


def test_superevent_t0_gps_time_becomes_event_time():
    gps = 1386502263.0  # arbitrary GPS epoch seconds
    payload = {
        "superevent_id": "S240422ed",
        "t_0": gps,
        "labels": ["GW150914LIKE"],
        "created": "2024-04-22T12:34:56Z",
    }
    event = _event_from_notice(payload, source="https://gracedb.ligo.org/api/superevents/S240422ed")
    assert event.event_time == Time(gps, format="ligo", scale="utc")
    assert event.alert_type == "SUPEREVENT"
    assert event.status == "active"
    assert event.notice_version == "2024-04-22T12:34:56Z"


def test_superevent_retracted_label_sets_status():
    payload = {"superevent_id": "S240101abc", "t_0": 1000000000.0, "labels": ["RETRACTED"]}
    event = _event_from_notice(payload)
    assert event.status == "retracted"


def test_retraction_notice_without_event_has_no_invented_event_time():
    event = _event_from_notice({
        "superevent_id": "S240101abc",
        "alert_type": "RETRACTION",
        "time_created": "2024-01-02T00:00:00Z",
        "event": None,
    })
    assert event.status == "retracted"
    assert event.event_time is None


def test_gracedb_client_selects_newest_notice_before_choosing_its_skymap():
    class Backend:
        def superevent(self, superevent_id):
            return type("Response", (), {"json": lambda self: {"superevent_id": superevent_id, "t_0": 1386502263.0}})()

        def files(self, superevent_id, filename="", **kwargs):
            listings = {
                "": {
                    "S240101abc-preliminary.json,0": "https://example.org/preliminary.json",
                    "S240101abc-update.json,0": "https://example.org/update.json",
                },
                "S240101abc-preliminary.json,0": {
                    "alert_type": "PRELIMINARY", "time_created": "2024-01-01T00:00:00Z",
                    "superevent_id": superevent_id, "event": {"time": "2024-01-01T00:00:00Z", "skymap": "old"},
                },
                "S240101abc-update.json,0": {
                    "alert_type": "UPDATE", "time_created": "2024-01-02T00:00:00Z",
                    "superevent_id": superevent_id, "event": {"time": "2024-01-01T00:00:00Z", "skymap": "new"},
                },
            }
            value = listings[filename]
            return type("Response", (), {
                "json": lambda self: value,
                "content": __import__("json").dumps(value).encode(),
            })()

    client = GraceDBClient(backend=Backend())
    notice = client.select_notice("S240101abc")
    assert notice.filename == "S240101abc-update.json,0"
    assert notice.payload["event"]["skymap"] == "new"


def test_notice_without_any_time_is_rejected():
    with pytest.raises(ValueError, match="event time"):
        _event_from_notice({"superevent_id": "S240101abc"})


def test_optional_gwpy_gps_time_interoperates_with_jinwu_time():
    """GWpy remains optional; when installed its GPS scalar is accepted."""
    gwpy_time = pytest.importorskip("gwpy.time")
    gps = gwpy_time.to_gps("2024-01-01T00:00:00Z")
    assert scalar_time(gps).to_value("ligo") == pytest.approx(float(gps), abs=1e-6)
