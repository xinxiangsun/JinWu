"""Offline adapter boundaries; native numerical equivalence is checked separately."""
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import astropy.units as u
from astropy.io import fits
import numpy as np
import pytest

from jinwu.core.pipeline import PipelineStatus, pipeline
from jinwu.core.time import Time
from jinwu.gecam import GECAMConfig, GECAMInput, GECAMPipeline, extract_lightcurve
from jinwu.gecam import backend


def config():
    return GECAMConfig(detectors=(17,), gain="high", time_range=[10, 20]*u.s,
                       time_bin=1*u.s, energy_edges=[40, 100]*u.keV)


def test_missing_backend_is_actionable(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("missing", name="gecam")
    monkeypatch.setattr(backend, "import_module", missing)
    with pytest.raises(ImportError, match="official GECAMTools"):
        backend._backend()


def test_internal_missing_dependency_is_not_disguised(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("missing", name="lmfit")
    monkeypatch.setattr(backend, "import_module", missing)
    with pytest.raises(ModuleNotFoundError) as exc:
        backend._backend()
    assert exc.value.name == "lmfit"


def test_old_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(backend, "import_module", lambda _: SimpleNamespace(get_software_version=lambda: "20230611"))
    with pytest.raises(ImportError, match="20240514"):
        backend._backend()


@pytest.mark.parametrize("change, error", [
    ({"time_range": [10, 20]}, TypeError),
    ({"time_range": [20, 10]*u.s}, ValueError),
    ({"time_bin": 0*u.s}, ValueError),
    ({"energy_edges": [100, 40]*u.keV}, ValueError),
    ({"gain": "both"}, ValueError),
])
def test_lightcurve_rejects_ambiguous_selections(change, error):
    args = dict(detector=17, gain="high", time_range=[10, 20]*u.s,
                time_bin=1*u.s, energy_edges=[40, 100]*u.keV)
    with pytest.raises(error):
        extract_lightcurve(None, **(args | change))


def test_resume_partial_and_changed_input(tmp_path, monkeypatch):
    source = tmp_path / "input"; source.mkdir()
    event_file = source / "event.fits"
    fits.PrimaryHDU(header=fits.Header({"MJDREFI": 58484, "MJDREFF": .00080074074,
                                       "TIMESYS": "TT", "TIMEUNIT": "s", "TIMEZERO": 2.})).writeto(event_file)
    info = SimpleNamespace(obs_met_range=(0, 100), satellite_full_name="GECAM-B",
                           instrument="GRD", trig_id="test", trig_met=12)
    monkeypatch.setattr(backend, "open_event", lambda p: SimpleNamespace(info=info))
    monkeypatch.setattr(backend, "_backend", lambda: SimpleNamespace(
        __file__=backend.__file__, get_software_version=lambda: "20240514"))
    module = import_module("jinwu.gecam.pipeline")
    real_version = module.version
    monkeypatch.setattr(module, "version", lambda name: "test" if name == "lmfit" else real_version(name))
    calls = []
    def lightcurve(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            get_data=lambda **kw: (np.array([10, 20]), np.array([[3.]]), np.array([[np.sqrt(3)]])),
            energy_bins=np.array([39., 101.]), channel_bins=np.array([3, 10]), dead_time_on_bins=np.array([0.]),
        )
    monkeypatch.setattr(backend, "extract_lightcurve", lightcurve)
    inp = GECAMInput("test", source, output_root=tmp_path / "output", event_file="event.fits")
    job = pipeline(config(), inp)
    assert isinstance(job, GECAMPipeline)
    first = job.run(until="lightcurve")
    assert job.status().status == PipelineStatus.PENDING
    assert first["pipeline_status"] == "pending"
    assert "spectra" not in first["stages"]
    result = GECAMPipeline(inp, config=config()).run(resume=True)
    assert len(calls) == 1
    assert result["pipeline_status"] == "completed"
    assert result["stages"]["spectra"]["data"] == {"enabled": False}
    with np.load(first["stages"]["lightcurve"]["outputs"]["GRD17_high"]) as data:
        assert data["telescope"].item() == "GECAM-B"
        assert data["met_clock_source"].item() == "FITS MJDREF/TIMESYS"
        assert abs((Time(data["met_reference_utc"].item()) - Time("2019-01-01T00:00:02")).to_value(u.s)) < 1e-6
    fits.setval(event_file, "COMMENT", value="changed")
    result = GECAMPipeline(inp, config=config()).run(resume=True)
    assert len(calls) == 2
    old_path = first["stages"]["lightcurve"]["outputs"]["GRD17_high"]
    new_path = result["stages"]["lightcurve"]["outputs"]["GRD17_high"]
    assert old_path != new_path
    assert __import__("pathlib").Path(old_path).is_file()
    # Units normalize into cache identity without losing scale.
    changed = replace(config(), time_bin=1000*u.ms)
    assert job.stage_config_dependencies(job.stages[0]) == GECAMPipeline(inp, config=changed).stage_config_dependencies(job.stages[0])
    changed = replace(config(), time_bin=2*u.s)
    GECAMPipeline(inp, config=changed).run(resume=True)
    assert len(calls) == 3
