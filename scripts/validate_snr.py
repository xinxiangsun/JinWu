"""Frozen-source and real-data acceptance: python -m scripts.validate_snr.

Uses original gv_significance function bodies, with unused Z_Bi/ncephes
imports omitted. No source algorithm is patched by the reference adapter.
"""
from __future__ import annotations

import argparse
import ast
from importlib.resources import files
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import warnings

import astropy
import astropy.units as u
import numpy as np
import scipy
from scipy import special

from jinwu.core import Time, read_pha, read_rmf, snr
from jinwu.core.products import sha256_file, write_json
from jinwu.core.significance import _pg_profile, _pp_gaussian_profile
from scripts.compare_gbm_mvt_snr import load_pg_reference


def load_gv_reference(root):
    """Load pinned original known/PP/PG functions without obsolete imports."""
    root = Path(root)
    provenance = json.loads(files('jinwu.core').joinpath('GV_SIGNIFICANCE_PROVENANCE.json').read_text())
    for name, digest in provenance['source_sha256'].items():
        if sha256_file(root.parent / name) != digest:
            raise ValueError(f'frozen source hash mismatch: {name}')
    pg = load_pg_reference(root)
    namespace = dict(pg.__globals__)
    namespace['scipy'] = scipy

    def extract(filename, names, target):
        tree = ast.parse((root / filename).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / filename), 'exec'), target)

    known = dict(namespace)
    extract('significance_from_pvalue.py', ('significance_from_pvalue',), known)
    known['tiny'] = np.finfo(float).tiny
    extract('ideal_case.py', ('significance',), known)
    pp = dict(namespace)
    extract('poisson_poisson.py', ('_li_and_ma', '_likelihood_with_sys',
                                  '_get_TS_by_numerical_optimization', 'significance'), pp)
    pp['_get_TS_by_numerical_optimization_v'] = np.vectorize(pp['_get_TS_by_numerical_optimization'])
    return {'known': known['significance'], 'pp': pp['significance'], 'pg': pg}, provenance


def _reference_call(function, n, b, kwargs):
    """Capture original intermediates at return using a scoped profile hook."""
    trace = {}

    def capture(frame, event, arg):
        if event != 'return' or frame.f_code.co_filename != function.__code__.co_filename:
            return
        local = frame.f_locals
        if frame.f_code.co_name == 'significance':
            for key in ('B0_mle', 'pvalue'):
                if key in local:
                    trace[key] = np.array(local[key], copy=True)
        elif frame.f_code.co_name == '_get_TS_by_numerical_optimization':
            fitted = local.get('res')
            if fitted is not None:
                trace.setdefault('profiles', []).append({
                    'k0': float(fitted.x[0]), 'B0': float((local['b_'] + local['n_']) /
                        (local['alpha'] * fitted.x[0] + local['alpha'] + 1)),
                    'negative_loglike0': float(fitted.fun), 'TS': float(local['TS']),
                    'success': bool(fitted.success), 'message': str(fitted.message),
                })
        elif frame.f_code.co_name == '_li_and_ma' and 'res' in local:
            trace.setdefault('TS', []).append(2 * np.array(local['res'], copy=True))

    method = kwargs['method']
    arguments = {}
    if method == 'pp':
        arguments = {'alpha': kwargs['alpha'], 'k': kwargs.get('systematic_fraction', 0),
                     'sigma': kwargs.get('systematic_sigma', 0)}
    elif method == 'pg':
        arguments = {'sigma': kwargs['background_error']}
    arrays = np.broadcast_arrays(n, b, *arguments.values())
    flat = [np.asarray(a, dtype=float).ravel() for a in arrays]
    arguments = dict(zip(arguments, flat[2:]))
    previous = sys.getprofile()
    try:
        sys.setprofile(capture)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            try:
                result = np.asarray(function(flat[0], flat[1], **arguments))
                error = None
            except Exception as exc:
                result, error = None, f'{type(exc).__name__}: {exc}'
            trace['warnings'] = list(dict.fromkeys(str(w.message) for w in recorded))
    finally:
        sys.setprofile(previous)
    # np.vectorize without otypes executes the first input once to infer dtype.
    profiles = trace.get('profiles', [])
    expected = np.count_nonzero(arguments.get('sigma', 0)) if method == 'pp' else 0
    if len(profiles) == expected + 1:
        trace['vectorize_dtype_probe'] = profiles[0]
        trace['profiles'] = profiles[1:]
    return result, error, trace


def compare_snr_case(label, n, b, kwargs, references):
    """Compare final values and source intermediates on identical count inputs."""
    method = kwargs['method']
    numerical = method == 'pp' and np.any(np.asarray(kwargs.get('systematic_sigma', 0)) > 0)
    tolerance = 1e-7 if numerical else 1e-10
    old, old_error, trace = _reference_call(references[method], n, b, kwargs)
    source_failed = old is None or np.any(~np.isfinite(old)) or any(
        not row['success'] for row in trace.get('profiles', []))
    zero_barrier = numerical and np.any((np.asarray(n) + np.asarray(b)) == 0)
    try:
        new, new_error = np.asarray(snr(n, b, **kwargs)), None
    except (ValueError, ArithmeticError) as exc:
        new, new_error = None, f'{type(exc).__name__}: {exc}'
    record = {'label': label, 'n_on_counts': n, 'background_counts': b, 'settings': kwargs,
              'reference': old, 'migrated': new, 'reference_error': old_error,
              'migrated_error': new_error, 'reference_intermediates': trace,
              'rtol': tolerance, 'atol': tolerance}
    if old is not None and np.any(~np.isfinite(old)):
        record['nonfinite_reference_indices'] = {
            'nan': np.flatnonzero(np.isnan(old.ravel())),
            'positive_infinity': np.flatnonzero(np.isposinf(old.ravel())),
            'negative_infinity': np.flatnonzero(np.isneginf(old.ravel())),
        }
    if source_failed or zero_barrier:
        if new_error is None:
            raise AssertionError(f'{label}: source failure was not guarded')
        record['status'] = 'guarded_upstream_failure'
        return record
    if new_error is not None:
        raise AssertionError(f'{label}: {new_error}')
    np.testing.assert_allclose(new.ravel(), old.ravel(), rtol=tolerance, atol=tolerance,
                               err_msg=f'{label}: final significance mismatch')
    record['maximum_z_difference'] = float(np.max(np.abs(new.ravel() - old.ravel()), initial=0))
    intermediate = {}
    if method == 'known':
        pvalue = special.pdtrc(np.asarray(n), np.asarray(b))
        np.testing.assert_array_equal(pvalue.ravel(), trace['pvalue'].ravel())
        intermediate['pvalue'] = pvalue
    elif method == 'pg':
        nn, bb, ss = (a.ravel() for a in np.broadcast_arrays(n, b, kwargs['background_error']))
        B0, half_ts = _pg_profile(nn, bb, ss)
        np.testing.assert_array_equal(B0, trace['B0_mle'].ravel())
        np.testing.assert_allclose(2 * half_ts, old.ravel()**2, rtol=tolerance, atol=tolerance)
        intermediate.update(B0=B0, TS=2*half_ts)
    elif numerical:
        nn, bb, aa, ss = (a.ravel() for a in np.broadcast_arrays(n, b, kwargs['alpha'], kwargs['systematic_sigma']))
        nn, bb, aa, ss = (a[ss > 0] for a in (nn, bb, aa, ss))
        profiles = []
        for i, old_profile in enumerate(trace['profiles']):
            ts, fitted = _pp_gaussian_profile(nn[i], bb[i], aa[i], ss[i])
            new_profile = {'k0': float(fitted.x[0]),
                           'B0': float((bb[i]+nn[i]) / (aa[i]*fitted.x[0]+aa[i]+1)),
                           'negative_loglike0': float(fitted.fun), 'TS': float(ts)}
            for key, value in new_profile.items():
                np.testing.assert_allclose(value, old_profile[key], rtol=tolerance, atol=tolerance)
            profiles.append(new_profile)
        intermediate['profiles'] = profiles
    else:
        expected_ts = np.concatenate([a.ravel() for a in trace['TS']])
        np.testing.assert_allclose(new.ravel()**2, expected_ts, rtol=tolerance, atol=tolerance)
        intermediate['TS'] = new**2
    record['migrated_intermediates'] = intermediate
    record['status'] = 'equivalent'
    return record


def validate_wxt(root, references):
    """Read real ON/OFF PHA and RMF, select the same observed 0.5--4 keV band."""
    root = Path(root)
    source_path, off_path = root/'ep06800001692wxt32s1.pha', root/'ep06800001692wxt32s1bk.pha'
    rmf_path = root/'ep06800001692wxt32.rmf'
    src, off, rmf = read_pha(source_path), read_pha(off_path), read_rmf(rmf_path)
    np.testing.assert_array_equal(src.channels, off.channels)
    np.testing.assert_array_equal(src.channels, rmf.channel)
    mask = (rmf.e_min >= .5) & (rmf.e_max <= 4.)
    n, b = float(src.counts[mask].sum()), float(off.counts[mask].sum())
    alpha = (src.exposure/off.exposure) * (src.backscal/off.backscal)
    if src.areascal != off.areascal:
        raise ValueError('validation expects matching AREASCAL')
    cases = [compare_snr_case('WXT '+label, n, b, {'method':'pp','alpha':alpha, **extra}, references)
             for label, extra in [('plain', {}), ('fixed 10%', {'systematic_fraction':.1}),
                                  ('Gaussian 10%', {'systematic_sigma':.1})]]
    return {'observation':'06800001692_32', 'data_source':'local EP/WXT mission products',
            'files_sha256':{str(p.resolve()):sha256_file(p) for p in (source_path,off_path,rmf_path)},
            'requested_observed_band_keV':[.5,4.], 'actual_band_keV':[rmf.e_min[mask].min(),rmf.e_max[mask].max()],
            'source_exposure_s':src.exposure, 'off_exposure_s':off.exposure,
            'source_backscal':src.backscal, 'off_backscal':off.backscal, 'cases':cases}


def validate_gbm(report_path, references):
    """TTE -> energy selection -> background refit/covariance -> 1-s local PG."""
    from jinwu.fermi.gbm.mvt.data import prepare_detector
    from jinwu.fermi.gbm.mvt.models import GBMMVTConfig
    from jinwu.fermi.gbm.tte import interval_exposure, read_detector_events

    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    trigger = Time(report['trigger_met'], format='fermi')
    src = np.array(report['source_interval_s'])
    off = np.array(report['background_intervals_s'])
    # Fixed complete bins; no peak/window search and no partial-bin extrapolation.
    edges = src[0] + np.arange(int(np.floor(src[1]-src[0]))+1, dtype=float)
    bins = np.column_stack((edges[:-1], edges[1:]))
    n, b, variance = (np.zeros(len(bins)) for _ in range(3))
    config = GBMMVTConfig(energy_range=report['energy_range_keV']*u.keV,
                         detectors=tuple(report['selected_detectors']),
                         background_order=report['settings']['background_order'],
                         background_bin_width=report['settings']['background_bin_width_s']*u.s)
    records = {}
    for detector in report['selected_detectors']:
        paths = [Path(p) for p in report['data_provenance'] if f'_tte_{detector}_' in p]
        if not paths:
            raise ValueError(f'no original TTE for {detector}')
        for path in paths:
            if sha256_file(path) != report['data_provenance'][str(path)]:
                raise ValueError(f'TTE hash mismatch: {path}')
        events, fit = prepare_detector(paths, trigger, src, off, config)
        original = report['background'][detector]
        for key in ('coefficients','covariance','background_counts','background_exposure_s'):
            np.testing.assert_allclose(fit[key], original[key], rtol=1e-10, atol=1e-10)
        times, channels, gtis, (bounds, deadtime, overflow) = read_detector_events(
            paths, trigger, [src[0],src[1]]*u.s)
        native = np.histogram(times, edges)[0]
        saturated = np.histogram(times[channels == len(bounds)-1], edges)[0]
        exposure = interval_exposure(edges, gtis) - (native-saturated)*deadtime - saturated*overflow
        if np.any(exposure <= 0):
            raise ValueError('nonpositive source livetime')
        basis = np.stack([(bins[:,1]**(j+1)-bins[:,0]**(j+1))/(j+1)
                          for j in range(fit['order']+1)], axis=1)
        coefficients = np.asarray(fit['coefficients'])[:,0]
        covariance = np.asarray(fit['covariance'])[:,:,0]
        # Width is 1 second; basis is also the bin-averaged polynomial basis.
        predicted = (basis @ coefficients) * exposure
        predicted_variance = np.einsum('ij,jk,ik->i',basis,covariance,basis) * exposure**2
        observed = np.histogram(events,edges)[0]
        n += observed; b += predicted; variance += predicted_variance
        records[detector] = {'observed_counts':observed, 'source_livetime_s':exposure,
            'expected_background_counts':predicted, 'background_variance_counts2':predicted_variance,
            'background_refit_matches':True, 'fit':fit}
    comparison = compare_snr_case(report_path.parent.name, n, b,
                                 {'method':'pg','background_error':np.sqrt(variance)},references)
    return {'input_report':str(report_path.resolve()),'report_sha256':sha256_file(report_path),
            'data_source':'HEASARC Fermi/GBM public TTE (URLs in data/PROVENANCE.json)',
            'tte_files_sha256':report['data_provenance'], 'trigger_met':report['trigger_met'],
            'requested_observed_band_keV':report['energy_range_keV'], 'source_interval_s':src,
            'background_intervals_s':off, 'fixed_complete_bins_s':bins,
            'exposure_policy':'GTI minus native-channel event deadtime, including overflow',
            'omitted_final_partial_bin_s':float(src[1]-edges[-1]),
            'background_covariance_policy':'independent detector fits, full polynomial coefficient covariance',
            'detectors':records,'cases':[comparison]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('external_sources/gv_significance-master/gv_significance'))
    parser.add_argument('--wxt', type=Path, default=Path('test/EP260809adata/EP260809a/06800001692_32'))
    parser.add_argument('--gbm-root', type=Path, default=Path('.runtime/gbm-mvt-validation'))
    parser.add_argument('--output', type=Path, default=Path('reviews/evidence/snr/validation.json'))
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    references, provenance = load_gv_reference(args.source)
    cases = []
    for n in (0, 1, 5, 20, 120, 1000):
        for b in (0, 1, 10, 80):
            for alpha in (.03, .1, 1.):
                for label, extra in [('plain', {}), ('fixed', {'systematic_fraction':.1}),
                                     ('Gaussian', {'systematic_sigma':.1})]:
                    cases.append(compare_snr_case(f'PP {label} n={n} b={b} alpha={alpha}', n, b,
                                                 {'method':'pp','alpha':alpha,**extra},references))
        for b in (-1, 0, 1, 80, 1000):
            for sigma in (.3, 1., 5.3, 30.):
                cases.append(compare_snr_case(f'PG n={n} b={b} sigma={sigma}', n, b,
                                             {'method':'pg','background_error':sigma},references))
        for b in (1., 2., 80.):
            cases.append(compare_snr_case(f'known n={n} b={b}', n, b, {'method':'known'},references))
    real = []
    for sigma in (0, .001, 1e-6):
        cases.append(compare_snr_case(f'PP small systematic sigma={sigma}', 20, 80,
                     {'method':'pp','alpha':.1,'systematic_sigma':sigma}, references))
    if not args.synthetic_only:
        real.append(validate_wxt(args.wxt,references))
        for name in ('real170817','real250919','blank170817'):
            print(f'Refitting GBM {name} ...', flush=True)
            real.append(validate_gbm(args.gbm_root/name/'report.json',references))
    all_cases = cases + [row for data in real for row in data['cases']]
    summary = {'equivalent_cases':sum(c['status']=='equivalent' for c in all_cases),
               'guarded_upstream_failures':sum(c['status']=='guarded_upstream_failure' for c in all_cases),
               'maximum_z_difference':max(c.get('maximum_z_difference',0) for c in all_cases),
               'real_datasets':len(real), 'scope':'local computation equivalence; no trials calibration or universal correctness claim'}
    import jinwu.core.significance as implementation
    packages = {}
    for name in ('jinwu','jinwu-fermi','astro-gdt','astro-gdt-fermi'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = 'unavailable'
    evidence = {'source':provenance,'source_root':str(args.source.resolve()),
        'environment':{'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__,
                       'scipy':scipy.__version__,'astropy':astropy.__version__,
                       'implementation':implementation.__file__, 'packages':packages, 'randomness':'none'},
        'implementation_sha256':sha256_file(implementation.__file__),
        'validation_script_sha256':sha256_file(__file__),
        'execution':'python -m scripts.validate_snr (default inputs; --output '+str(args.output)+')',
        'summary':summary,'synthetic_cases':cases,'real_data':real}
    write_json(args.output,evidence)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
