"""Offline contracts for GBM subthreshold search and empirical calibration."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
import pytest

from jinwu.core.time import Time
from jinwu.fermi.gbm.subthreshold import (
    GBMTargetedSearchConfig, GBMTargetedSearchInput, GBMTargetedSearchPipeline,
    estimate_candidate_far, calibration_from_searches,
)
from jinwu.fermi.gbm.subthreshold.data import merge_tte_events, interval_exposure, latest_products, PreparedSearchData
from jinwu.fermi.gbm.subthreshold.search import (
    make_search_windows, spatial_prior_weights, select_search_candidates, evaluate_prepared_window,
    evaluate_search_likelihood,
)


def job(tmp_path, **kwargs):
    return GBMTargetedSearchInput(target_id="test", root=tmp_path / "data", output_root=tmp_path / "out",
                                 trigger_time="2017-08-17T12:41:04.429126", template_root=tmp_path / "templates", **kwargs)


def test_units_and_dyadic_grid():
    with pytest.raises(TypeError):
        GBMTargetedSearchConfig(min_duration=.064)
    with pytest.raises(ValueError):
        GBMTargetedSearchConfig(max_duration=.3 * u.s)
    config = GBMTargetedSearchConfig(search_interval=[0, 2] * u.s, max_duration=1.024 * u.s)
    windows = make_search_windows(config, [(0., .7), (.9, 2.)])
    assert len(windows)
    assert all(any(a >= x - 1e-9 and a + d <= y + 1e-9 for x, y in [(0., .7), (.9, 2.)]) for a, d in windows)
    assert set(np.round(windows[:, 1], 6)) == {.064, .128, .256, .512}
    assert 1.024 in make_search_windows(config, [(0, 2)])[:, 1]
    assert np.allclose(windows[:, 0] / .064, np.round(windows[:, 0] / .064))


def test_point_map_conflict_and_input_time(tmp_path):
    with pytest.raises(ValueError):
        job(tmp_path, position=SkyCoord(10, 20, unit="deg"), skymap="a.fits")
    assert job(tmp_path).trigger_time.utc.isot.startswith("2017-08-17T12:41:04")
    assert json.dumps(job(tmp_path).to_dict())
    assert json.loads(json.dumps(GBMTargetedSearchConfig().to_dict())) == GBMTargetedSearchConfig().to_dict()


def test_fermi_time_includes_leap_seconds_independent_of_gdt():
    from jinwu.core.time import TimeFermi
    from astropy.time import Time as AstroTime
    epoch = AstroTime(TimeFermi.epoch_val, scale=TimeFermi.epoch_scale)
    target = AstroTime('2017-08-17T12:41:04.429126', scale='utc')
    expected = (target.tt - epoch.tt).to_value(u.s)
    assert expected == pytest.approx(524666469.429126, abs=1e-6)
    assert Time(target).to_value('fermi') == pytest.approx(expected, abs=1e-6)
    from gdt.missions.fermi.time import FermiSecTime
    assert FermiSecTime.epoch_scale == TimeFermi.epoch_scale
    assert FermiSecTime.epoch_val == TimeFermi.epoch_val


def test_configuration_import_without_optional_search_stack():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'healpy', 'rich', 'astropy_healpix'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
from jinwu.fermi.gbm.subthreshold import GBMTargetedSearchConfig
assert GBMTargetedSearchConfig().pipeline == 'fermi.gbm.subthreshold'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_multiset_merge_preserves_coincident_channels_and_multiplicity():
    t, c = merge_tte_events([([1., 1., 1., 2.], [0, 0, 1, 3]), ([1., 1., 2., 3.], [0, 1, 3, 2])])
    assert list(zip(t, c)) == [(1., 0), (1., 0), (1., 1), (2., 3), (3., 2)]


def test_gti_exposure_union_and_latest_version(tmp_path):
    assert interval_exposure([0, 1, 2, 3], [(0, .5), (.4, 1.5), (2.5, 3)]).tolist() == [1, .5, .5]
    a, b = [tmp_path / f"glg_tte_n0_170817_12z_v{v:02d}.fit.gz" for v in (0, 1)]
    assert latest_products([a, b]) == (b,)


def test_prior_preserves_hidden_probability_and_narrow_map():
    grid = SkyCoord([0, 90, 180, 270], [0]*4, unit="deg")
    visible = np.array([True, False, True, True])
    weights, mass, offset = spatial_prior_weights(grid, visible, position=grid[1])
    assert mass == 0 and weights.sum() == 0 and offset == 0
    vectors = SkyCoord([2, 89], [0, 0], unit="deg").cartesian.xyz.value.T
    weights, mass, _ = spatial_prior_weights(grid, visible, map_vectors=vectors, map_probability=[.2, .8])
    assert mass == pytest.approx(.2) and weights[0] == 1


def test_prior_ranking_before_clustering_and_empty_selection():
    table = QTable({"tstart": [0, .1, 2] * u.s, "duration": [1, 1, .2] * u.s,
                    "eligible": [True]*3, "loglr": [8., 4., 3.], "prior_loglr": [3., 9., np.nan]})
    independent, _ = select_search_candidates(table, score="loglr", threshold=5, overlap_factor=.2)
    targeted, _ = select_search_candidates(table, score="prior_loglr", threshold=5, overlap_factor=.2)
    assert independent["tstart"][0] == 0 * u.s
    assert targeted["tstart"][0] == .1 * u.s
    empty, reasons = select_search_candidates(table, score="loglr", threshold=50, overlap_factor=.2)
    assert len(empty) == 0 and set(reasons) == {"score_floor"}


def test_full_variance_and_detector_live_exposure(monkeypatch):
    import jinwu.fermi.gbm.subthreshold.search as module
    prepared = PreparedSearchData(np.arange(5) * .064, np.ones((4, 2, 8), int),
                  np.full((4, 2), .032), np.full((4, 2, 8), 100.), np.full((4, 2, 8), 10.),
                  np.ones((4, 2, 8), bool), np.array([[0., .256]]), ("n0", "n1"))
    captured = {}
    def capture(counts, bkg, var, response, **kwargs):
        captured.update(counts=counts, bkg=bkg, var=var, response=response)
        return "result"
    monkeypatch.setattr(module, "evaluate_search_likelihood", capture)
    response = SimpleNamespace(load_response=lambda a,b: np.ones((3, 2, 16)), sky_mask=lambda: np.array([True, True]))
    result, _, _, _ = evaluate_prepared_window(prepared, response, SimpleNamespace(size=2), 0, .128)
    assert result == "result"
    assert np.all(captured["counts"] == 2)
    assert np.allclose(captured["bkg"], 6.4)
    assert np.allclose(captured["var"], .64**2)
    assert np.all(captured["response"] == .5)


def test_calibration_rates_and_zero_tail():
    result = estimate_candidate_far(5., [1, 5, 10], 100 * u.s, search_time=60 * u.s)
    assert result["far_hz"] == .02
    assert result["fap"] == pytest.approx(1 - np.exp(-1.2))
    result = estimate_candidate_far(20, [1, 5, 10], 100 * u.s)
    assert result["far_hz"] is None and result["kind"] == "upper_limit"
    assert result["far_upper_hz"] == pytest.approx(-np.log(.05) / 100)


def test_measured_attitude_rejects_gap_saa_and_extrapolation():
    from jinwu.fermi.gbm.subthreshold.data import MeasuredHistory
    history = MeasuredHistory.__new__(MeasuredHistory)
    history.times = np.array([0., 1., 2., 10., 11., 12.])
    history.good = np.array([True, True, True, True, False, True])
    assert history.valid_interval(.2, 1.8)
    assert not history.valid_interval(1.5, 10.2)
    assert not history.valid_interval(10.2, 11.8)
    assert not history.valid_interval(-.1, 1.)


def test_template_completeness_and_roundoff(tmp_path, monkeypatch):
    from jinwu.fermi.gbm.subthreshold.pipeline import validate_search_templates
    from jinwu.fermi.gbm.subthreshold._vendor import utils
    monkeypatch.setattr(utils, 'SkyGrid', lambda step: SimpleNamespace(size=2))
    for kind, detectors in [('nai', 12), ('bgo', 2)]:
        paths = [tmp_path / 'direct' / f'{kind}.npy'] + [
            tmp_path / f'atmo_{kind}' / f'atmrates_az{az}_zen130.npy' for az in range(0, 360, 5)]
        a = np.ones((3, 2, 8, detectors)); a.flat[0] = -1e-20
        for p in paths:
            p.parent.mkdir(exist_ok=True); np.save(p, a)
    records = validate_search_templates(tmp_path)
    assert len(records) == 146 and all(r['roundoff_clipped'] == 1 for r in records.values())
    bad = tmp_path / 'direct/nai.npy'
    a = np.load(bad); a.flat[0] = -1.; np.save(bad, a)
    with pytest.raises(ValueError, match='negative'):
        validate_search_templates(tmp_path)
    bad.unlink()
    with pytest.raises(ValueError, match='missing'):
        validate_search_templates(tmp_path)


def test_background_rate_is_per_live_second(monkeypatch):
    import jinwu.fermi.gbm.subthreshold.data as module
    from gdt.core.background import unbinned
    # Constant 20% dead time: fitted elapsed-time rate 8 becomes live-time 10.
    times = np.arange(-21.95, 22., .1)
    monkeypatch.setattr(module, 'read_detector_events', lambda *a: (
        times, np.full(len(times), 10), [(-22., 22.)], (None, .02, .02)))
    class Model:
        def __init__(self, events): pass
        def fit(self, **kwargs): pass
        def interpolate(self, start, stop):
            return np.full((len(start), 1), 8.), np.full((len(start), 1), 2.)
    monkeypatch.setattr(unbinned, 'NaivePoisson', Model)
    config = GBMTargetedSearchConfig(search_interval=[-1, 1]*u.s, max_duration=.512*u.s,
                  background_window=2*u.s, background_context=2*u.s, detectors=('n0',))
    prepared, _ = module.prepare_search_data({'n0': ['stub']}, Time('2017-01-01'), config)
    center = np.argmin(abs(prepared.edges[:-1]))
    assert prepared.rates[center, 0, 1] == pytest.approx(10., rel=.015)
    assert prepared.uncertainty[center, 0, 1] == pytest.approx(2.5, rel=.015)


def report(t0=0., interval=(0., 60.), contract=None):
    return {"search_complete": True, "quality_passed": True, "trigger_met": t0,
            "effective_intervals_met": [list(interval)], "calibration_contract": contract or {"x": 1},
            "calibration_scores": {"loglr": [6], "prior_loglr": [7]}}


def test_calibration_duplicate_overlap_and_contract():
    result = calibration_from_searches([report(), report(), report(100, (100, 160))])
    assert result["livetime_s"] == 120 and result["scores"]["loglr"] == [6, 6]
    with pytest.raises(ValueError, match="overlapping"):
        calibration_from_searches([report(), report(30, (30, 90))])
    with pytest.raises(ValueError, match="configurations"):
        calibration_from_searches([report(), report(100, (100, 160), {"x": 2})])
    with pytest.raises(ValueError, match="quality"):
        calibration_from_searches([{**report(), "quality_passed": False}])


def test_missing_templates_produces_partial_report(tmp_path):
    pipe = GBMTargetedSearchPipeline(job(tmp_path))
    result = pipe.run()
    assert result.science_status == "needs_review"
    assert result.status == "needs_review"
    assert json.loads(Path(result.products["report"]).read_text())["search_complete"] is False


def test_prepared_roundtrip_and_pipeline_code_dependencies(tmp_path):
    prepared = PreparedSearchData(np.arange(3), np.ones((2, 1, 8)), np.ones((2, 1)),
                  np.ones((2, 1, 8)), np.ones((2, 1, 8)), np.ones((2, 1, 8), bool), np.array([[0,2]]), ("n0",))
    path = tmp_path / "prepared.npz"
    prepared.save(path)
    restored = PreparedSearchData.open(path)
    assert restored.detectors == ("n0",) and np.array_equal(restored.counts, prepared.counts)
    pipe = GBMTargetedSearchPipeline(job(tmp_path))
    for stage in pipe.stages:
        paths = pipe.stage_code_dependencies(stage)
        assert all(p.is_file() for p in paths)
        assert any(p.name == "likelihood.py" for p in paths)


def test_likelihood_matches_attributed_kernel_and_detects_injection():
    pytest.importorskip("healpy")
    from jinwu.fermi.gbm.subthreshold._vendor.likelihood import Likelihood
    rng = np.random.default_rng(10)
    response = rng.uniform(.5, 1.5, (3, 6, 14))
    background = np.full(14, 100.)
    var = np.full(14, 4.)
    values = []
    for amplitude in (0., 10., 30.):
        counts = np.rint(background + amplitude * response[1, 2])
        expected = Likelihood(3, 6, prethresh=-np.inf)
        expected.calculate(counts, background, var, response)
        result = evaluate_search_likelihood(counts, background, var, response, sky_size=6)
        assert result.marginal_llr == pytest.approx(expected.marginal_llr, rel=1e-12)
        assert np.allclose(result.llr, expected.llr, rtol=1e-12)
        values.append(result.marginal_llr)
    assert values[0] < values[1] < values[2]
