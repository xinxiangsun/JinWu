"""Regression contracts for scientific and data failures fixed before beta -> master."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import QTable, Table

from jinwu.background.backprior import BackgroundSpectralPrior
from jinwu.core.config import BATSurvey, GECAM, GBM, SwiftBATSurveyConfig, UpperLimitConfig, instrument
from jinwu.core.data import EventData, LightcurveData, PhaData, RmfData
from jinwu.core.io import read_evt, write_evt, write_pha, write_rmf
from jinwu.core.ops import BayesianBlocksBinner, rebin_pha, slice_pha
from jinwu.core.products import load_net_lightcurve
from jinwu.core.time import Time
from jinwu.core.xselect import extract_image
from jinwu.gw.gracedb import normalize_superevent_id
from jinwu.gw.models import scalar_time
from jinwu.fermi.gbm.pipeline import _as_scalar_time
from jinwu.fermi.gbm.subthreshold import GBMTargetedSearchConfig, GBMTargetedSearchInput, GBMTargetedSearchPipeline


def test_met_utc_epochs_and_scalar_calendar_boundaries():
    assert Time(0, format="gecam").utc.isot == "2019-01-01T00:00:00.000"
    assert Time(0, format="hxmt").utc.isot == "2012-01-01T00:00:00.000"
    boundary = Time("2023-12-23T23:59:50", scale="utc").tt
    assert _as_scalar_time(boundary).datetime.hour == 23
    assert scalar_time(boundary).datetime.hour == 23


@pytest.mark.parametrize("raw,canonical", [
    ("MS230615az", "MS230615az"),
    ("ts230615abc", "TS230615abc"),
    ("s230615az", "S230615az"),
])
def test_gracedb_id_preserves_lowercase_suffix(raw, canonical):
    assert normalize_superevent_id(raw) == canonical


def test_config_replace_and_science_defaults():
    policy = UpperLimitConfig()
    assert replace(policy, default_sigma=4).default_sigma == 4
    with pytest.raises(ValueError, match="empirical calibration_mode"):
        UpperLimitConfig(calibration="bootstrap", calibration_mode="empirical_global_search")
    selection = SwiftBATSurveyConfig(detthresh=9000)
    survey = SwiftBATSurveyConfig(detthresh=8000)
    bat = BATSurvey(survey=survey, selection=selection)
    assert bat.selection is selection and bat.survey is survey
    assert replace(GBM(detector="BGO_1"), group_min_counts=10).detector == "BGO_1"
    assert replace(GECAM(detector="GRD01", energy_range_keV=(10, 100)), group_min_counts=10).detector == "GRD01"
    wxt = instrument("WXT")
    wxt.response_type = "rsp"
    assert not wxt.response_requires_arf
    with pytest.raises(ValueError, match="energy_range"):
        wxt.energy_range_keV = (4, 0.5)
    assert instrument("WXT").spectrum.group_min_counts == 1
    assert instrument("FXT").spectrum.group_min_counts == 3


def test_slice_pha_keeps_ebounds_aligned_and_rebinnable():
    channels = np.arange(1, 7)
    pha = PhaData(path=Path("synthetic.pha"), header={}, meta=None, headers_dump=None,
                  channels=channels, counts=np.ones(6), exposure=10,
                  ebounds=(channels, channels.astype(float), channels.astype(float) + 1))
    selected = slice_pha(pha, ch_lo=3, ch_hi=4)
    np.testing.assert_array_equal(selected.ebounds[0], [3, 4])
    assert len(rebin_pha(selected, factor=2).counts) == 1


def test_bayesian_block_bin_membership_conserves_counts(monkeypatch):
    import astropy.stats

    monkeypatch.setattr(astropy.stats, "bayesian_blocks", lambda *args, **kwargs: np.array([0., 1.5, 4.]))
    counts = np.array([10., 10., 20., 20.])
    lc = LightcurveData(path=Path("synthetic.lc"), time=np.arange(4) + .5,
                        value=counts, error=np.sqrt(counts), dt=1., exposure=4.,
                        is_rate=False, counts=counts, counts_err=np.sqrt(counts),
                        header={}, meta={}, headers_dump={}, columns=("TIME", "COUNTS"))
    binner = BayesianBlocksBinner()
    result = binner.fit(lc)
    assert result.counts.sum() == 60
    assert sum(map(len, binner.last_merged_indices)) == 4


def test_gbm_effective_time_uses_full_windows(tmp_path, monkeypatch):
    import jinwu.fermi.gbm.subthreshold.pipeline as pipeline_module
    import jinwu.fermi.gbm.subthreshold.search as search_module

    config = GBMTargetedSearchConfig(min_duration=1.024 * u.s, max_duration=1.024 * u.s,
                                     num_steps=8, search_interval=[0, 2] * u.s)
    inp = GBMTargetedSearchInput(target_id="test", root=tmp_path / "data",
                                 output_root=tmp_path / "out", trigger_time="2017-08-17T12:41:04.429126",
                                 template_root=tmp_path / "templates")
    pipe = GBMTargetedSearchPipeline(inp, config=config)
    pipe.workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pipeline_module.PreparedSearchData, "open", lambda _: object())
    monkeypatch.setattr(pipeline_module, "MeasuredHistory", lambda _: object())
    table = QTable({"tstart": [0., .128, .256] * u.s,
                    "duration": [1.024] * 3 * u.s,
                    "atmospheric_response": [True] * 3,
                    "loglr": [1., 1., 1.], "prior_loglr": [1., 1., 1.]})
    monkeypatch.setattr(search_module, "run_search_grid", lambda *args, **kwargs: (table, {"failed_windows": []}))
    monkeypatch.setattr(search_module, "select_search_candidates", lambda tab, **kwargs: (tab[:1], ["kept"] * len(tab)))
    context = {"background": SimpleNamespace(outputs={"prepared": "stub"}, data={"quality_passed": True}),
               "data": SimpleNamespace(data={"poshist": []})}
    stage = pipe._stage_search(context)
    intervals = stage.data["effective_intervals_met"]
    assert len(intervals) == 1
    assert intervals[0][1] - intervals[0][0] == pytest.approx(1.280)


def test_event_timezero_survives_write_read(tmp_path):
    evt = EventData(path=tmp_path / "original.evt", header={"TIMEZERO": 0.0}, meta=None, headers_dump=None,
                    time=np.array([0., 1.]), timezero=100., x=np.array([2., 3.]), y=np.array([4., 5.]))
    path = tmp_path / "roundtrip.evt"
    write_evt(evt, path)
    out = read_evt(path)
    np.testing.assert_allclose(out.time + out.timezero, [100., 101.])
    with fits.open(path) as hdus:
        assert hdus["EVENTS"].header["TIMEZERO"] == 100.


def test_rate_only_pha_does_not_relabel_rates_as_counts(tmp_path):
    pha = PhaData(path=Path("synthetic.pha"), header={"EXPOSURE": float("nan")}, meta=None,
                  headers_dump=None, channels=np.array([1, 2]), counts=np.array([.2, .3]),
                  rate=np.array([.2, .3]), exposure=float("nan"))
    path = tmp_path / "rate.pha"
    write_pha(pha, path)
    with fits.open(path) as hdus:
        names = hdus["SPECTRUM"].columns.names
        assert "RATE" in names and "COUNTS" not in names
        assert "EXPOSURE" not in hdus["SPECTRUM"].header


def test_background_prior_rejects_missing_observed_channels(tmp_path):
    prior = BackgroundSpectralPrior(a0=np.array([1., 1.]), b0=10.,
                                    area_ratio=.2, channels=np.array([1, 2]))
    spectrum = fits.BinTableHDU.from_columns([
        fits.Column(name="CHANNEL", format="I", array=np.array([1])),
        fits.Column(name="COUNTS", format="J", array=np.array([5])),
    ], name="SPECTRUM")
    spectrum.header["EXPOSURE"] = 10.
    path = tmp_path / "partial.pha"
    fits.HDUList([fits.PrimaryHDU(), spectrum]).writeto(path)
    with pytest.raises(ValueError, match="missing prior channels"):
        prior.update_with_off_spectrum(str(path), use_ebounds=False, verbose=False)
    with pytest.raises(ValueError, match="missing prior channels"):
        prior.update_with_on_bg_spectrum(str(path), use_ebounds=False, verbose=False)


def test_rmf_first_channel_keyword_follows_actual_column(tmp_path):
    rmf = RmfData(path=Path("synthetic.rmf"), header={}, meta=None, headers_dump=None,
                  energ_lo=np.array([1.]), energ_hi=np.array([2.]),
                  f_chan=np.array([np.array([1])], dtype=object),
                  n_chan=np.array([np.array([2])], dtype=object),
                  matrix=np.array([np.array([.5, .5])], dtype=object), tlmin=1,
                  channel=np.array([1, 2]), e_min=np.array([1., 1.]), e_max=np.array([2., 2.]))
    path = tmp_path / "response.rmf"
    write_rmf(rmf, path)
    with fits.open(path) as hdus:
        mat = hdus["MATRIX"]
        assert mat.columns.names[2] == "F_CHAN"
        assert mat.header["TLMIN3"] == 1
        assert "TLMIN4" not in mat.header


def test_filtered_event_image_uses_selected_rows():
    event = EventData(path=None, header={}, meta=None, headers_dump=None,
                      time=np.array([0., 1., 2.]), x=np.array([1., 2., 3.]), y=np.array([1., 2., 3.]))
    image, _, _ = extract_image(event, bins=(4, 4), tmin=1, tmax=2)
    assert image.sum() == 2


def test_net_lightcurve_fits_uppercase_metadata(tmp_path):
    t = Table()
    for name in ("time", "bin_width", "fractional_exposure", "source_rate", "source_error",
                 "background_rate", "background_error", "net_rate", "net_error"):
        t[name] = [1.0]
    t.meta["ALPHA"] = 0.2
    t.meta["TIMEZERO"] = 100.0
    path = tmp_path / "net.fits"
    t.write(path)
    result = load_net_lightcurve(path)
    assert result.alpha == 0.2 and result.timezero == 100.0
