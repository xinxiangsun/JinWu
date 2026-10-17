"""Run custom cflux models on preserved EP260119a WXT T100 products.

This validates software contracts, not the physical interpretation of trial models.
"""
from pathlib import Path
import json
import xspec
from jinwu.core.fit import fit_prepared
from jinwu.core.products import sha256_file, write_json, collect_runtime_environment
from jinwu.core.spectrum_prep import PreparedSpectrum

out=Path(__file__).resolve().parent
repo=out.parents[2]
reference=repo/'artifacts/ta_validation/ta_ep260119a_20261017/wxt/obs000/fit/t100/model_comparison.json'
record=json.loads(reference.read_text())['candidates']['powerlaw_nh0']
inputs=json.loads(Path(record['fit_products']['result_json']).read_text())['input_files'][0]
path_keys={'source_pha','grouped_pha','background_pha','arf','rmf'}
prepared=PreparedSpectrum(**{key:Path(value) if key in path_keys else value for key,value in inputs.items()},energy_range_keV=(.5,4.))
hashes={str(getattr(prepared,key)):sha256_file(getattr(prepared,key)) for key in path_keys}
xspec.Xset.chatter=0
xspec.Xset.logChatter=10
xspec.Fit.nIterations=100
results={}
for label,expression in [('blackbody','tbabs*cflux*bbodyrad'),('line','tbabs*cflux*gauss'),
                         ('continuum_line','tbabs*cflux*(powerlaw+gauss)')]:
    frozen={'gaussian.LineE':1.0,'gaussian.Sigma':.1} if 'gauss' in expression else None
    result=fit_prepared(prepared,outdir=out/label,model_name=expression,
        galactic_nh_1e22=.0145,calculate_errors=False,flux_band_keV=(.5,4.),
        frozen_parameters=frozen,plot_formats=('png',),plot_required=True)
    norms={name:value for name,value in result['parameters'].items() if name.endswith('.norm')}
    assert sum(p['frozen'] for p in norms.values())==1
    assert next(p for p in norms.values() if p['frozen'])['value']==1.0
    assert result['flux_unabs']['erg_cm2_s']>0
    assert result['flux_abs']['erg_cm2_s']<=result['flux_unabs']['erg_cm2_s']*(1+1e-10)
    results[label]=result
rejected={}
for i,expression in enumerate(['cflux*tbabs*powerlaw','tbabs*(powerlaw+cflux*gauss)',
                               'cflux*powerlaw+gauss']):
    try:
        fit_prepared(prepared,outdir=out/f'rejected{i}',model_name=expression,
                     galactic_nh_1e22=.0145,calculate_errors=False,plot_formats=())
    except ValueError as exc:
        assert 'cflux' in str(exc)
        rejected[expression]=str(exc)
    else:
        raise AssertionError(f'Unsupported cflux expression accepted: {expression}')
assert all(sha256_file(Path(path))==digest for path,digest in hashes.items())
write_json(out/'validation.json',dict(environment=collect_runtime_environment(),input_reference=str(reference),
    source_sha256=sha256_file(repo/'packages/jinwu/src/jinwu/core/fit.py'),input_sha256=hashes,
    original_inputs_unchanged=True,custom_fits=results,rejected_expressions=rejected,
    limitations=['Trial source models validate fitting and export, not source classification.',
                 'Profile errors disabled; no shape-constraint or line-detection claim.']))
print('CUSTOM_VALIDATION: 3 real fits and 3 invalid-scope rejections passed',flush=True)
