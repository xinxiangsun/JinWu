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
    values[:, 6] = 2.0  # ECF
    source = tmp_path / "xrt.dat"
    np.savetxt(source, values, delimiter=",")
    table = read_burst_analyser_dat(source)
    assert table["rate_source"] == "flux_over_ecf"
    np.testing.assert_allclose(table["rate"], (2.0, 4.0))


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
