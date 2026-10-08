"""Focused offline contracts for the public Swift single-GRB workflow."""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pytest

from jinwu.core.config import SwiftGRB, instrument
from jinwu.swift.grb.pipeline import _alpha_from_area_file, read_burst_analyser_dat, safe_extract_tar


def test_swift_grb_is_registered_core_only_preset() -> None:
    config = instrument("swift_grb")
    assert isinstance(config, SwiftGRB)
    assert config.pipeline == "swift.grb"
    assert config.data.bat_binning == "SNR5_sinceT0"
    assert config.data.bat_band == "BATBand"
    assert config.spectrum.fit_energy_range_keV == (0.3, 10.0)


def test_fourteen_column_rate_is_explicitly_flux_over_ecf(tmp_path) -> None:
    values = np.zeros((2, 14), dtype=float)
    values[:, 3] = (4.0, 8.0)  # flux
    values[:, 4] = 1.0
    values[:, 5] = -1.0
    values[:, 6] = 7.0  # FluxPosWithECFErr, deliberately distinct from ECF
    values[:, 11] = 2.0  # ECF
    source = tmp_path / "xrt.dat"
    np.savetxt(source, values, delimiter=",")
    table = read_burst_analyser_dat(source)
    assert table["rate_source"] == "flux_over_ecf"
    np.testing.assert_allclose(table["rate"], (2.0, 4.0))
    np.testing.assert_allclose(table["ECF"], (2.0, 2.0))
    np.testing.assert_allclose(table["rate_err_low"], (.5, .5))
    np.testing.assert_allclose(table["rate_err_high"], (.5, .5))


def test_area_file_requires_source_area_and_uses_start_stop_background(tmp_path) -> None:
    area = tmp_path / "allpcback.area"
    area.write_text("0 20 200\n20 40 100\n", encoding="utf-8")
    assert np.isnan(_alpha_from_area_file(area, time_s=10.0))
    assert _alpha_from_area_file(area, source_area=100.0, time_s=30.0) == pytest.approx(1.0)


def test_tar_rejects_parent_path_members(tmp_path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as handle:
        info = tarfile.TarInfo("../outside.txt")
        data = b"unsafe"
        info.size = len(data)
        handle.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_tar(archive, tmp_path / "products")


def test_area_fallback_keeps_source_trigger_time_when_background_has_none(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from jinwu.swift.grb import pipeline as module
    lc = tmp_path / 'downloads' / 'lc'
    lc.mkdir(parents=True)
    for name in ('wtsourcetotal.evt.gz', 'wtbacktotal.evt.gz', 'allwtback.area'):
        (lc / name).touch()
    monkeypatch.setattr(module, '_read_event_times', lambda path: (np.array([101., 102., 103.]), None))
    monkeypatch.setattr(module, '_event_chunks', lambda *args: [(101., 104.)])
    monkeypatch.setattr(module, 'bayesian_blocks', lambda *args, **kwargs: np.array([101., 104.]))
    monkeypatch.setattr(module, '_event_trigger_met', lambda path: None)
    monkeypatch.setattr(module, '_read_area_ratio', lambda *args: np.nan)
    monkeypatch.setattr(module, '_source_area_from_event', lambda *args: 100.)
    monkeypatch.setattr(module, '_alpha_from_area_file', lambda *args, **kwargs: .5)
    pipe = SimpleNamespace(
        _downloads_dir=tmp_path/'downloads', workspace=tmp_path/'output',
        _record=SimpleNamespace(trigger_met=100.),
        input=SimpleNamespace(wt_snr_threshold=0., pc_snr_threshold=0., pc_min_source_counts=0., bblock_p0=.05),
        _swift_config=SwiftGRB())
    result = module.SwiftGRBPipeline._stage_bblocks(pipe, {})
    assert result.data['segments'][0]['start'] == 1.
    assert result.data['segments'][0]['stop'] == 4.
