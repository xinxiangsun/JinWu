from dataclasses import replace
from pathlib import Path

import astropy.units as u
from astropy.io import fits
import numpy as np
import pytest

from jinwu.core.data import RmfData
from jinwu.core.fit import fit_prepared
from jinwu.core.io import write_rmf
from jinwu.core.spectrum_prep import prepare_ogip_spectrum


@pytest.fixture
def type2_products(tmp_path):
    edges = np.geomspace(10, 180, 25)
    channels = np.arange(edges.size - 1)
    count_model = 14000 * (1 / edges[:-1] - 1 / edges[1:])
    # Row 2 has twice the exposure and twice the source rate.
    counts = np.rint([count_model + 30, 4 * count_model + 40])
    background = np.broadcast_to(np.array([30.0, 40.123456789])[:, None], counts.shape)
    paths = [tmp_path / name for name in ("full.pha", "background.pha", "matrix.rsp")]
    for path, values, errors in (
        (paths[0], counts, np.sqrt(counts)),
        (paths[1], background, np.sqrt(background) + 0.0123456789),
    ):
        n = channels.size
        hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="SPEC_NUM", format="J", array=[7, 20]),
            fits.Column(name="CHANNEL", format=f"{n}J", array=np.tile(channels, (2, 1))),
            fits.Column(name="COUNTS", format=f"{n}D", array=values),
            fits.Column(name="STAT_ERR", format=f"{n}D", array=errors),
            fits.Column(name="GROUPING", format=f"{n}I", array=np.ones((2, n), dtype=np.int16)),
            fits.Column(name="QUALITY", format=f"{n}I", array=np.zeros((2, n), dtype=np.int16)),
            fits.Column(name="EXPOSURE", format="D", array=[5.0, 10.0]),
            fits.Column(name="RESPFILE", format="16A", array=["stale.rsp", "stale.rsp"]),
            fits.Column(name="BACKFILE", format="16A", array=["stale.pha", "stale.pha"]),
        ], name="SPECTRUM")
        hdu.header.update(DETCHANS=n, TELESCOP="SYNTHETIC", INSTRUME="TEST", CHANTYPE="PI")
        ebounds = fits.BinTableHDU.from_columns([
            fits.Column(name="CHANNEL", format="J", array=channels),
            fits.Column(name="E_MIN", format="E", unit="keV", array=edges[:-1]),
            fits.Column(name="E_MAX", format="E", unit="keV", array=edges[1:]),
        ], name="EBOUNDS")
        fits.HDUList([fits.PrimaryHDU(), hdu, ebounds]).writeto(path)

    response = RmfData(
        path=paths[2], header={"TELESCOP": "SYNTHETIC", "INSTRUME": "TEST", "CHANTYPE": "PI", "TLMIN4": 0},
        meta=None, headers_dump=None, energ_lo=edges[:-1], energ_hi=edges[1:],
        n_grp=np.ones(n, dtype=int), f_chan=channels[:, None], n_chan=np.ones((n, 1), dtype=int),
        matrix=np.full((n, 1), 100.), channel=channels, e_min=edges[:-1], e_max=edges[1:],
        columns=("ENERG_LO", "ENERG_HI", "N_GRP", "F_CHAN", "N_CHAN", "MATRIX"),
    )
    write_rmf(response, paths[2])
    return paths


def test_type2_preparation_keeps_row_data_and_replaces_stale_links(tmp_path, type2_products):
    source, background, response = type2_products
    originals = [path.read_bytes() for path in type2_products]
    settings = dict(
        outdir=tmp_path / "prepared", instrument="GECAM", response=response,
        energy_range=[15, 150] * u.keV, background=background,
        source_poisson=True, background_poisson=False, background_row=2,
    )
    with pytest.raises(ValueError, match="1-based"):
        prepare_ogip_spectrum(source, **settings)
    prepared = prepare_ogip_spectrum(source, source_row=2, **settings)
    for original, copied, poisson in (
        (source, prepared.grouped_pha, True), (background, prepared.background_pha, False),
    ):
        with fits.open(original) as native, fits.open(copied, checksum=True) as output:
            hdu = output["SPECTRUM"]
            assert hdu.header["HDUCLAS4"] == "TYPE:I"
            assert hdu.header["EXPOSURE"] == 10
            assert hdu.header["POISSERR"] is poisson
            assert hdu.header["RESPFILE"] == "response.rsp"
            assert hdu.header["BACKFILE"] == ("background.pha" if poisson else "NONE")
            assert hdu.header["CORRFILE"] == "NONE"
            assert hdu.header["CORRSCAL"] == 0
            for name in ("COUNTS", "STAT_ERR", "CHANNEL", "GROUPING", "QUALITY"):
                np.testing.assert_array_equal(hdu.data[name], native["SPECTRUM"].data[name][1])
    assert prepared.rmf.read_bytes() == response.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_ogip_spectrum(source, source_row=2, **settings)
    assert [path.read_bytes() for path in type2_products] == originals


def test_type2_preparation_rejects_wrong_response_grid_atomically(tmp_path, type2_products):
    source, background, response = type2_products
    wrong = tmp_path / "wrong.rsp"
    with fits.open(response) as hdus:
        hdus["EBOUNDS"].data["E_MAX"][3] *= 1.01
        hdus.writeto(wrong)
    output = tmp_path / "prepared"
    output.mkdir()
    with pytest.raises(ValueError, match="EBOUNDS"):
        prepare_ogip_spectrum(
            source, outdir=output, instrument="GECAM", response=wrong,
            energy_range=[15, 150] * u.keV, background=background,
            source_row=2, background_row=2, source_poisson=True, background_poisson=False,
        )
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("correction_target", [0, 1])
def test_type2_preparation_rejects_correction_files(tmp_path, type2_products, correction_target):
    source, background, response = type2_products
    with fits.open(type2_products[correction_target], mode="update") as hdus:
        hdus["SPECTRUM"].header["CORRFILE"] = "correction.pha"
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="correction file is unsupported"):
        prepare_ogip_spectrum(
            source, outdir=output, instrument="GECAM", response=response,
            energy_range=[15, 150] * u.keV, background=background,
            source_row=2, background_row=2, source_poisson=True, background_poisson=False,
        )
    assert not output.exists()


def test_pyxspec_mixed_statistics_and_calibration_replay(tmp_path, type2_products, monkeypatch):
    xspec = pytest.importorskip("xspec")
    from jinwu.core import plot

    monkeypatch.chdir(tmp_path)
    source, background, response = type2_products
    prepared = []
    for row in (1, 2):
        prepared.append(prepare_ogip_spectrum(
            source, outdir=tmp_path / f"prepared_{row}", instrument="GECAM", response=response,
            energy_range=[15, 150] * u.keV, background=background, source_row=row, background_row=row,
            source_poisson=True, background_poisson=(row == 1), spectrum_key=f"detector{row}",
        ))
    options = dict(
        model_name="constant * cflux * powerlaw", explicit_bindings=True,
        joint_normalization="cross_calibration", flux_band_keV=(5., 160.),
        statistics={"detector1": "cstat", "detector2": "pgstat"},
        calculate_errors=False, plot_formats=(),
    )
    try:
        with pytest.raises(ValueError, match="outside the XSPEC model energy grid"):
            fit_prepared(prepared, outdir=tmp_path / "short_grid", **options)
        xspec.AllModels.setEnergies("5 180 1000 log")
        with pytest.raises(ValueError, match="Reset the existing custom XSPEC model energy grid"):
            fit_prepared(prepared, outdir=tmp_path / "preexisting_grid", **options)
        assert xspec.AllModels(1).energies(1)[0] == pytest.approx(5)
        xspec.AllModels.setEnergies("reset")
        with pytest.raises(ValueError, match="Gaussian errors"):
            fit_prepared(
                [prepared[0], replace(prepared[1], background_pha=prepared[0].background_pha)],
                outdir=tmp_path / "wrong_background", **options,
            )
        figures = []
        plotfit = plot.plotfit

        def capture_plot(**kwargs):
            path, figure = plotfit(**kwargs)
            figures.append(figure)
            return path, figure

        monkeypatch.setattr(plot, "plotfit", capture_plot)
        result = fit_prepared(
            prepared, outdir=tmp_path / "fit", model_energy_grid=(5., 180., 2000),
            **(options | {"plot_formats": ("png",), "plot_required": True}),
        )
        assert result["effective_statistics_per_spectrum"] == ["wstat", "pgstat"]
        assert result["effective_statistic_verified"]
        assert result["free_parameter_count"] == 3
        stats = result["statistics"]
        assert stats["method"] == "mixed"
        assert stats["value"] == pytest.approx(sum(row["value"] for row in stats["per_spectrum"].values()))
        assert result["calibration_reference_spectrum"] == "detector1"
        assert result["per_group"][0]["parameters"]["constant.factor"]["value"] == 1
        assert result["per_group"][1]["parameters"]["constant.factor"]["value"] == pytest.approx(2, abs=0.2)
        assert result["per_group"][1]["cflux"] == result["per_group"][0]["cflux"]
        data, _, residual = figures[0].axes
        assert "Mixed (W-stat + PG-stat)" in data.get_title()
        assert {label.get_text() for label in data.get_legend().get_texts()} == {"detector1", "detector2"}
        data_colors = [container.lines[0].get_color() for container in data.containers]
        assert len(set(data_colors)) == 2
        assert data_colors == [container.lines[0].get_color() for container in residual.containers]
        dense_result = fit_prepared(
            prepared, outdir=tmp_path / "fit_dense", model_energy_grid=(5., 180., 4000), **options,
        )
        assert dense_result["statistics"]["value"] == pytest.approx(stats["value"], abs=0.001)
        for name in ("cflux.lg10Flux", "powerlaw.PhoIndex", "constant.factor"):
            for coarse, dense in zip(result["per_group"], dense_result["per_group"]):
                assert dense["parameters"][name]["value"] == pytest.approx(
                    coarse["parameters"][name]["value"], abs=0.002,
                )
        assert set(result["cflux_extrapolated_spectra"]) == {"detector1", "detector2"}
        assert xspec.AllModels(1).energies(1)[0] == pytest.approx(10)
        xcm = Path(result["fit_products"]["xcm"])
        assert "statistic pgstat 2" in xcm.read_text()
        assert "energies 5.0 180.0 2000 log" in xcm.read_text()
        xspec.AllData.clear()
        xspec.AllModels.clear()
        xspec.Xset.restore(str(xcm))
        assert xspec.AllModels(1).energies(1)[0] == pytest.approx(5)
        xspec.Fit.perform()
        assert xspec.Fit.statistic == pytest.approx(stats["value"], abs=1e-4)
    finally:
        xspec.AllData.clear()
        xspec.AllModels.clear()
        xspec.AllModels.setEnergies("reset")
