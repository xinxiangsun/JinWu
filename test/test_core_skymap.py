"""Regression for the shared light sky-map reader (jinwu.core.skymap)."""
import re

import numpy as np
import pytest
from astropy.io import fits

from jinwu.core.skymap import SkyMapData, load_skymap, sky_map_pixel_vectors


def _flat_map(path, ordering="RING"):
    probability = np.arange(1, 13, dtype=float)
    probability /= probability.sum()
    cols = fits.ColDefs([fits.Column(name="PROB", format="D", array=probability)])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["NSIDE"] = 1
    hdu.header["ORDERING"] = ordering
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


def _nuniq_map(path):
    import astropy_healpix as ah
    # two level-2 cells and one level-3 cell; density uniform per sr
    uniq = np.array([4 * 4**2 + 0, 4 * 4**2 + 5, 4 * 4**3 + 9], dtype=np.int64)
    levels, ipix = ah.uniq_to_level_ipix(uniq)
    area = ah.nside_to_pixel_area(ah.level_to_nside(levels)).to_value("sr")
    density = np.full(uniq.size, 1.0 / area.sum())
    cols = fits.ColDefs([
        fits.Column(name="UNIQ", format="K", array=uniq),
        fits.Column(name="PROBDENSITY", format="D", array=density, unit="sr-1"),
    ])
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)


def test_flat_ring_map_normalizes_and_marks_ordering(tmp_path):
    p = tmp_path / "m.fits"
    _flat_map(p)
    sm = load_skymap(p)
    assert isinstance(sm, SkyMapData)
    assert sm.total_probability == pytest.approx(1.0)
    assert sm.ordering == "RING"
    assert np.all(sm.levels == 0)
    # RING index 0 (north pole) maps to NESTED 0..3; contract only needs nested ids
    assert np.all((sm.ipix >= 0) & (sm.ipix < 12))


def test_flat_nested_map(tmp_path):
    p = tmp_path / "m.fits"
    _flat_map(p, ordering="NESTED")
    sm = load_skymap(p)
    assert sm.ordering == "NESTED"
    assert np.array_equal(sm.ipix, np.arange(12))


def test_nuniq_multires_map(tmp_path):
    p = tmp_path / "m.fits"
    _nuniq_map(p)
    sm = load_skymap(p)
    assert sm.ordering == "NUNIQ"
    assert sm.total_probability == pytest.approx(1.0)
    assert set(np.unique(sm.levels)) == {2, 3}


def test_rejects_nonpositive_probability(tmp_path):
    p = tmp_path / "m.fits"
    cols = fits.ColDefs([fits.Column(name="PROB", format="D", array=np.zeros(12))])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["NSIDE"] = 1
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(p)
    with pytest.raises(ValueError):
        load_skymap(p)


def test_pixel_vectors_subdivide_and_conserve_mass(tmp_path):
    p = tmp_path / "m.fits"
    _flat_map(p)
    sm = load_skymap(p)
    vecs, masses = sky_map_pixel_vectors(sm)
    assert vecs.shape[1] == 3
    assert masses.sum() == pytest.approx(sm.total_probability)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0)


def test_reader_contract_agrees_between_core_and_gw(tmp_path):
    """jinwu.core.skymap and jinwu.gw.skymap must agree on the pixel contract."""
    from jinwu.gw.skymap import load_skymap as gw_load_skymap

    p = tmp_path / "m.fits"
    _flat_map(p)
    core_map = load_skymap(p)
    gw_map = gw_load_skymap(p)
    assert np.allclose(core_map.pixel_probability, np.asarray(gw_map.pixel_probability))
    assert np.allclose(core_map.pixel_area_sr, np.asarray(gw_map.pixel_area_sr))
    assert np.allclose(core_map.prob_density_sr, np.asarray(gw_map.prob_density_sr))
    assert np.array_equal(core_map.uniq, np.asarray(gw_map.uniq))


def test_fermi_sources_do_not_import_jinwu_gw():
    """Dependency-direction guard: jinwu-fermi must not depend on jinwu-gw."""
    import sys
    from pathlib import Path

    fermi_root = Path("packages/jinwu-fermi/src")
    if not fermi_root.is_dir():  # installed-wheel context without repo tree
        pytest.skip("jinwu-fermi source tree not available")
    offenders = [
        str(path)
        for path in fermi_root.rglob("*.py")
        if "_vendor" not in path.parts
        and re.search(r"^\s*(from|import)\s+jinwu\.gw\b", path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == [], f"jinwu-fermi imports jinwu.gw (forbidden): {offenders}"
