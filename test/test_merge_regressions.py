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
from jinwu.core.ops import BayesianBlocksBinner, bayesian_blocks_exposure, rebin_pha, slice_pha
from jinwu.core.products import load_net_lightcurve
from jinwu.core.spectrum_prep import _stage_ancillary_file
from jinwu.core.time import Time
from jinwu.core.time import extract_time_interval
from jinwu.core.xselect import extract_image
from jinwu.gw.gracedb import normalize_superevent_id
from jinwu.gw.models import scalar_time
from jinwu.fermi.gbm.pipeline import _as_scalar_time
from jinwu.fermi.gbm.subthreshold import GBMTargetedSearchConfig, GBMTargetedSearchInput, GBMTargetedSearchPipeline


def test_prepare_spectrum_replaces_stale_ancillary_link(tmp_path):
    old_source = tmp_path / "old" / "response.rmf"
    current_source = tmp_path / "current" / "response.rmf"
    staged_dir = tmp_path / "prepared"
    old_source.parent.mkdir(parents=True)
    current_source.parent.mkdir(parents=True)
    staged_dir.mkdir()
    old_source.write_bytes(b"old response")
    current_source.write_bytes(b"current response")
    staged = staged_dir / "response.rmf"
    staged.symlink_to(old_source)

    result = _stage_ancillary_file(
        staged_dir, current_source, name="response.rmf", overwrite=False
    )

    assert result == staged
    assert result.is_symlink()
    assert result.resolve() == current_source.resolve()


def test_met_utc_epochs_and_scalar_calendar_boundaries():
    assert Time(0, format="gecam").utc.isot == "2019-01-01T00:00:00.000"
    assert Time(0, format="hxmt").utc.isot == "2012-01-01T00:00:00.000"
    boundary = Time("2023-12-23T23:59:50", scale="utc").tt
    assert _as_scalar_time(boundary).datetime.hour == 23
    assert scalar_time(boundary).datetime.hour == 23


def test_swift_compatibility_label_uses_bundled_met_format(monkeypatch):
    monkeypatch.delitem(Time.FORMATS, "swift", raising=False)
    pha = SimpleNamespace(header={"TSTART": 100., "TSTOP": 200.})
    interval = extract_time_interval(pha, "BAT", time_format="swift")
    assert interval["start"].utc.isot == Time(100., format="swiftmet").utc.isot
    from jinwu.swift.bat.survey import _time_value, _time_utc_iso
    assert _time_value(interval["start"]) == pytest.approx(100.)
    assert _time_value(interval["start"].utc.isot) == pytest.approx(100., abs=.001)
    assert _time_utc_iso(100.) == interval["start"].utc.isot


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


def test_bayesian_blocks_exposure_matches_exhaustive_partitions():
    cases = [
        (np.array([5.0]), np.array([1.0]), 0.0),
        (np.array([0.0, 100.0]), np.ones(2), 0.0),
        (np.array([0.0, 100.0, 0.0]), np.ones(3), 20.0),
        (np.array([0.0, 100.0, 0.0, 100.0]), np.ones(4), 4.0),
        (np.array([1.0, 3.0, 11.0, 2.0]), np.ones(4), 6.0),
        (np.array([0.0, 5.0]), np.array([0.0, 1.0]), 0.0),
    ]

    def score(counts, exposure, edges, prior):
        total = 0.0
        for start, stop in zip(edges[:-1], edges[1:]):
            n_block = float(np.sum(counts[start:stop]))
            t_block = float(np.sum(exposure[start:stop]))
            if t_block == 0:
                assert n_block == 0
                fitness = 0.0
            else:
                fitness = n_block * np.log(n_block / t_block) if n_block > 0 else 0.0
            total += fitness - prior
        return total

    for counts, exposure, prior in cases:
        edges = bayesian_blocks_exposure(counts, exposure, ncp_prior=prior)
        assert edges[0] == 0
        assert edges[-1] == counts.size
        assert np.all(np.diff(edges) > 0)

        best_score = -np.inf
        n = counts.size
        for mask in range(1 << (n - 1)):
            interior = [i for i in range(1, n) if mask & (1 << (i - 1))]
            candidate = np.asarray([0, *interior, n], dtype=int)
            best_score = max(best_score, score(counts, exposure, candidate, prior))
        assert score(counts, exposure, edges, prior) == pytest.approx(best_score)


def test_bayesian_blocks_exposure_rejects_invalid_poisson_inputs():
    invalid_inputs = [
        ([-1.0], [1.0]),
        ([np.nan], [1.0]),
        ([1.0], [np.inf]),
        ([1.0], [0.0]),
        ([0.0], [-1.0]),
    ]
    for counts, exposure in invalid_inputs:
        with pytest.raises(ValueError):
            bayesian_blocks_exposure(counts, exposure)


def test_bayesian_blocks_binner_preserves_single_exposure_bin():
    counts = np.array([5.0])
    lc = LightcurveData(
        path=Path("single-bin.lc"), time=np.array([11.0]), value=counts,
        error=np.sqrt(counts), dt=2.0, exposure=2.0, is_rate=False,
        counts=counts, counts_err=np.sqrt(counts), bin_exposure=np.array([2.0]),
        bin_lo=np.array([10.0]), bin_hi=np.array([12.0]), bin_width=np.array([2.0]),
        header={}, meta={}, headers_dump={}, columns=("TIME", "COUNTS"),
    )
    binner = BayesianBlocksBinner(use_exposure=True)

    result = binner.fit(lc)

    assert len(result.time) == 1
    assert result.counts.sum() == pytest.approx(counts.sum())
    assert binner.last_edges == pytest.approx([10.0, 12.0])


def test_rebin_pha_factor_arguments_are_explicit():
    channels = np.arange(1, 5)
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    pha = PhaData(
        path=Path("grouped.pha"), channels=channels, counts=counts,
        stat_err=np.sqrt(counts), exposure=10.0,
        grouping=np.array([1, -1, 1, -1]), header={}, meta={},
        headers_dump={}, columns=("CHANNEL", "COUNTS"),
    )

    assert len(rebin_pha(pha).counts) == 2
    assert rebin_pha(pha, factor=1) is pha
    for factor in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            rebin_pha(pha, factor=factor)
    with pytest.raises(ValueError, match="互斥"):
        rebin_pha(pha, factor=2, min_counts=4)


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


def test_lightcurve_invalid_errors_are_rejected_without_mutation():
    from jinwu.core.fit import LightcurveFitter
    errors = np.array([0., 1., 1.])
    fitter = LightcurveFitter((np.arange(3.), np.array([0., 1., 2.]), errors))
    with pytest.raises(ValueError, match='positive measurement errors'):
        fitter.fit('linear')
    np.testing.assert_array_equal(errors, [0., 1., 1.])


def test_astropy_fit_reports_evaluation_limit_as_failure():
    from astropy.modeling import models
    from jinwu.core.fit import LightcurveFitter
    time = np.linspace(.1, 10., 40)
    values = 5 * np.exp(-time / 2.)
    fitter = LightcurveFitter((time, values, np.full(40, .1)))
    result = fitter.fit(models.Exponential1D, p0=[1., -20.], fitter_method='trf', maxiter=1)
    assert not result.success
    assert np.all(np.isnan(result.errors))


@pytest.mark.parametrize('delta', [1., 3., 2.706])
def test_xspec_error_delta_is_always_a_real_literal(delta):
    from jinwu.core.fit import _prepared_error_parameters
    model = SimpleNamespace(componentNames=[], powerlaw=SimpleNamespace(
        PhoIndex=SimpleNamespace(index=1), norm=SimpleNamespace(index=2)))
    command = _prepared_error_parameters(model, 'powerlaw', 'zero', delta_stat=delta)
    token = command.split()[0]
    assert '.' in token
    assert float(token) == delta


def test_sparse_rmf_mapping_handles_subset_duplicate_and_invalid_channels():
    from scipy.sparse import csr_matrix
    from jinwu.ftools.rmf_mapping import map_channels_to_energy
    matrix = np.array([[1., 0., 0.], [1., 1., 0.], [0., 1., 1.], [0., 0., 0.]])
    channels = np.array([2, 0, 2, 3, -1, 4])
    sparse = map_channels_to_energy(csr_matrix(matrix), np.array([1., 2., 3.]), channels)
    dense = map_channels_to_energy(matrix, np.array([1., 2., 3.]), channels, prefer_sparse=False)
    np.testing.assert_allclose(sparse, [2.5, 1., 2.5, np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(sparse, dense, equal_nan=True)


def test_ftselect_preserves_boolean_precedence_and_chained_comparisons():
    from jinwu.ftools.ftselect import expression_to_mask
    ev = SimpleNamespace(time=np.arange(4.), pha=np.array([0, 10, 20, 30]))
    np.testing.assert_array_equal(expression_to_mask(ev, 'PHA>5 && PHA<25 || PHA==0'), [1, 1, 1, 0])
    np.testing.assert_array_equal(expression_to_mask(ev, '!(5 < PHA < 25)'), [1, 0, 0, 1])
    np.testing.assert_array_equal(expression_to_mask(ev, 'TIME >= 1e0 && PHA / 10 < 3'), [0, 1, 1, 0])
    with pytest.raises(ValueError, match='Unsupported'):
        expression_to_mask(ev, 'PHA.__class__')


def test_teldef_alignment_is_invertible_for_tilted_detector():
    from scipy.spatial.transform import Rotation
    from jinwu.ftools.teldef import Teldef
    teldef = Teldef()
    teldef.align = Rotation.from_euler('xyz', [.05, -.02, .1], degrees=True).as_matrix()
    teldef.focal_length = 3500.
    teldef.det_xscl, teldef.det_yscl = .04, .04
    teldef.optaxis = (300., 300.)
    for ra, dec in [(50., 30.), (50.1, 30.1), (49.9, 29.9)]:
        pixels = teldef.sky_to_det_with_pointing(ra, dec, 50., 30.)
        recovered = teldef.det_to_sky_with_pointing(*pixels, 50., 30.)
        np.testing.assert_allclose(recovered, [ra, dec], rtol=0, atol=1e-10)


def test_simulation_removes_input_background_before_response_scaling(tmp_path, monkeypatch):
    from jinwu.lf import lcfake
    npz = tmp_path / 'onoff.npz'
    np.savez(npz, time=np.arange(4.), corrected_counts_src=np.full(4, 10.),
             corrected_counts_back=np.full(4, 20.))
    cfg = lcfake.XspecConfig('', '', None, 'powerlaw', (2., 1.), (.5, 4.), 1., None)
    monkeypatch.setattr(lcfake._K_FACTORY, 'get_K', lambda **kwargs: 1.)
    result = lcfake.build_fake_from_npz(str(npz), cfg, area_ratio=.5,
                                       add_poisson=False, background_rate=10., output_total_rate=True)
    np.testing.assert_array_equal(result.counts, np.full(4, 10))
    pair = lcfake.build_fake_on_off_from_npz(str(npz), cfg, alpha=.5,
                                           add_poisson=False, background_rate_on=10.)
    np.testing.assert_array_equal(pair.counts_on, np.full(4, 10))
    np.testing.assert_array_equal(pair.counts_off, np.full(4, 20))


def test_netdata_rejects_misaligned_absolute_time_and_bin_width():
    from jinwu.core.datasets import netdata
    def curve(time, width=1., origin=0.):
        return LightcurveData(path=Path('synthetic.lc'), time=np.asarray(time), value=np.ones(3),
                              error=np.ones(3), dt=width, is_rate=False, timezero=origin,
                              header={}, meta={}, headers_dump={}, columns=('TIME', 'COUNTS'))
    src = curve(np.arange(3.) + 100000.5)
    with pytest.raises(ValueError, match='align'):
        netdata(src, curve(src.time + .9), ratio=1.)
    with pytest.raises(ValueError, match='TIMEZERO'):
        netdata(src, curve(src.time, origin=1.), ratio=1.)
    with pytest.raises(ValueError, match='align'):
        netdata(src, curve(src.time, width=2.), ratio=1.)
    np.testing.assert_array_equal(netdata(src, curve(src.time), ratio=1.).value, np.zeros(3))


@pytest.mark.parametrize('flux_failure', [False, True])
def test_bxa_flux_chain_restores_model_even_on_failure(flux_failure):
    from jinwu.core.bxa_fit import _create_flux_chain_at_best_fit
    class Solver:
        current = 'best'
        def create_flux_chain(self, spectrum, erange):
            self.current = 'posterior_sample'
            if flux_failure:
                raise RuntimeError('flux evaluation failed')
            return np.array([1., 2.])
        def set_best_fit(self):
            self.current = 'best'
    solver = Solver()
    messages = []
    chain = _create_flux_chain_at_best_fit(solver, None, '.3 5.', messages)
    assert solver.current == 'best'
    assert bool(messages) == flux_failure
    if flux_failure:
        assert chain is None
    else:
        np.testing.assert_array_equal(chain, [1., 2.])
