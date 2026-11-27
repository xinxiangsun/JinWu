"""Galactic-NH cache validation at the TA shared-fit boundary."""
import json
from types import SimpleNamespace

import pytest

from jinwu.core.galactic import load_galactic_absorption_cache
from jinwu.ep import ta
from jinwu.ep.fxt import FXTInput

RA, DEC = 1.0, 2.0


def _cache(*, nhtot=2e20, tbabs=None, schema=1, service="swift_ukssdc_nhtot",
           algorithm="nhtot-weighted-v1", ra=RA, dec=DEC, equinox=2000):
    tbabs = nhtot / 1e22 if tbabs is None else tbabs
    return {
        "schema_version": schema, "service": service, "algorithm_version": algorithm,
        "ra_deg": ra, "dec_deg": dec, "equinox": equinox,
        "result": {"ra_deg": ra, "dec_deg": dec, "equinox": equinox,
                   "nhi_weighted_cm2": None, "nh2_weighted_cm2": None,
                   "nhtot_weighted_cm2": nhtot, "tbabs_nh_1e22": tbabs,
                   "service": service, "algorithm_version": algorithm,
                   "queried_at_utc": "2026-01-19T00:00:00+00:00"},
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_cache_returns_validated_result(tmp_path):
    path = _write(tmp_path / "nhtot.json", _cache())
    result = load_galactic_absorption_cache(path, ra_deg=RA, dec_deg=DEC)
    assert result.cache_hit is True
    assert result.tbabs_nh_1e22 == pytest.approx(.02)


@pytest.mark.parametrize("changes", [
    {"schema": 2}, {"service": "other"}, {"algorithm": "nhtot-weighted-v0"},
    {"ra": RA + 1}, {"dec": DEC - 1}, {"equinox": 1950},
    {"nhtot": float("nan")}, {"nhtot": 0.0}, {"tbabs": float("nan")},
])
def test_load_cache_rejects_mismatch_or_invalid_column(tmp_path, changes):
    path = _write(tmp_path / "nhtot.json", _cache(**changes))
    with pytest.raises(ValueError, match="Invalid Galactic absorption cache"):
        load_galactic_absorption_cache(path, ra_deg=RA, dec_deg=DEC)


def test_load_cache_reports_missing_and_malformed(tmp_path):
    path = tmp_path / "nhtot.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_galactic_absorption_cache(path, ra_deg=RA, dec_deg=DEC)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid Galactic absorption cache"):
        load_galactic_absorption_cache(path, ra_deg=RA, dec_deg=DEC)


def _children(tmp_path, payload):
    children = []
    for obsid in ("001", "002"):
        workspace = tmp_path / obsid
        _write(workspace / "galactic_absorption" / "nhtot.json", payload)
        segment = {"label": "bin", "fit_eligible": True,
                   **{key: str(workspace / key) for key in
                      ("source_pha", "background_pha", "arf", "rmf")}}
        children.append(SimpleNamespace(
            workspace=workspace, spectrum_bins=[segment],
            files=SimpleNamespace(obsid=obsid, detector="CMOS1")))
    return children


def _rows():
    return [{"obsid": obsid, "detector": "CMOS1", "segment": "bin"}
            for obsid in ("001", "002")]


def _pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, "prepare_spectra", lambda manifest, **options:
                        SimpleNamespace(spectra=[SimpleNamespace(ready=True)]))
    return ta.TAPipeline(ta.TAInput(target_id="test", ra_deg=RA, dec_deg=DEC,
                                    output_root=tmp_path / "out",
                                    fxt=FXTInput(target_id="test", root=tmp_path / "input",
                                                 ra_deg=RA, dec_deg=DEC)))


def test_ta_shared_fit_passes_validated_cache_nh(tmp_path, monkeypatch):
    recorded = {}

    def fit(spectra, **options):
        recorded.update(options)
        return SimpleNamespace(adopted_key="powerlaw_free_nh", adopted_fit={
            "per_group": [{"spectrum_indices": [index + 1],
                           "flux_unabs": {"erg_cm2_s": 1e-12}} for index in range(2)],
            "xspec_settings": {}})

    monkeypatch.setattr(ta, "fit_xray_models", fit)
    pipeline = _pipeline(tmp_path, monkeypatch)
    pipeline._fit_all_wxt_bins(_children(tmp_path, _cache()), _rows(),
                               tmp_path / "out", 0.)
    assert recorded["galactic_nh_1e22"] == pytest.approx(.02)
    assert recorded["shared_absorption_only"] is True


def test_ta_shared_fit_rejects_cache_for_other_coordinates(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, "fit_xray_models", lambda *args, **options:
                        pytest.fail("fit must not run with an unvalidated cache"))
    pipeline = _pipeline(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Invalid Galactic absorption cache"):
        pipeline._fit_all_wxt_bins(_children(tmp_path, _cache(ra=RA + 1)), _rows(),
                                   tmp_path / "out", 0.)
