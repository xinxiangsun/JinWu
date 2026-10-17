"""Real EP spectra: stellar APEC redshift and FXT epoch-matching regression.

Run in conda hea. Fits are software validation, not stellar classifications.
No original products or previous fit outputs are rewritten.
"""
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import xspec
from astropy.io import fits

from jinwu.core.config import FXT
from jinwu.core.fit import fit_prepared, fit_xray_models
from jinwu.core.products import collect_runtime_environment, sha256_file, write_json
from jinwu.core.spectrum_prep import PreparedSpectrum
from jinwu.ep import TAInput, TAPipeline
from jinwu.ep.fxt import FXTInput

repo = Path(__file__).resolve().parents[3]
output = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else 'run')
output.mkdir(exist_ok=False)
reference_path = repo / 'artifacts/ta_validation/ta_ep260119a_20261017/wxt/obs000/fit/t100/model_comparison.json'
reference = json.loads(reference_path.read_text())['candidates']['powerlaw_nh0']
source_record = json.loads(Path(reference['fit_products']['result_json']).read_text())['input_files'][0]
wxt = PreparedSpectrum(**{key: Path(value) if key in {'source_pha','grouped_pha','background_pha','arf','rmf'} else value
                          for key, value in source_record.items()}, energy_range_keV=(.5,4.))
source = Path('/home/xinxiang/research/EP260119a')
fx_spectra = []
for module, short in [('FXTA','a'),('FXTB','b')]:
    folder = source / 'fxtevt/06800001128_result1am'
    pha = next(folder.glob(f'fxt_{short}_06800001128_*_po_src_*.pha'))
    background = next(folder.glob(f'fxt_{short}_06800001128_*_po_bkg_*.pha'))
    grouped = folder / f'EP260119a_fxt{short}_groupmin3_06800001128.pha'
    fx_spectra.append(PreparedSpectrum('FXT','06800001128',module,None,None,
        pha,grouped,background,pha.with_suffix('.arf'),pha.with_suffix('.rmf'),3,(.5,10.)))
inputs = {Path(getattr(spectrum,key)) for spectrum in [wxt,*fx_spectra]
          for key in ('source_pha','grouped_pha','background_pha','arf','rmf')}
checksums = {str(path):sha256_file(path) for path in inputs}
redshift, nh = 5.461, .0145
atomic = repo / 'external_sources/heasoft-6.36/Xspec/src/spectral/modelData/apec_v3.1.3'
xspec.Xset.chatter = 0
xspec.Xset.logChatter = 10
xspec.Fit.nIterations = 100
comparison = fit_xray_models(wxt,outdir=output/'wxt',galactic_nh_1e22=nh,redshift=redshift,
    candidate_keys=('powerlaw_nh0','apec','apec_nh0','double_apec'),selection_metric='aic',
    apec_root=atomic,calculate_errors=False,flux_band_keV=(.5,4.),plot_formats=('png',))
assert not comparison.failures, comparison.failures
for key, candidate in comparison.candidates.items():
    assert 'cflux' in candidate['model']
    if key.startswith(('apec','double_apec')):
        redshifts = {name:parameter['value'] for name,parameter in candidate['parameters'].items()
                     if name.endswith('.Redshift')}
        assert redshifts and set(redshifts.values()) == {0.0}, redshifts
        assert candidate['xspec_settings']['redshift_absorbers'] == [0.0]
    norms = [parameter for name,parameter in candidate['parameters'].items() if name.endswith('.norm')]
    assert sum(parameter['frozen'] for parameter in norms) == 1
    assert next(parameter for parameter in norms if parameter['frozen'])['value'] == 1.0
powerlaw = comparison.candidates['powerlaw_nh0']
# The reference ran Fit.error, which can refine the optimum. Match that setting
# rather than comparing its final optimum with the errors-disabled warm fit.
control = fit_prepared(wxt,outdir=output/'powerlaw_control',galactic_nh_1e22=nh,redshift=redshift,
    intrinsic_nh_mode='zero',calculate_errors=True,flux_band_keV=(.5,4.),plot_formats=())
np.testing.assert_allclose(control['statistics']['value'],reference['statistics']['value'],atol=1e-7,rtol=0)
np.testing.assert_allclose(control['flux_unabs']['erg_cm2_s'],reference['flux_unabs']['erg_cm2_s'],rtol=1e-5)
config = FXT()
config.fitting = replace(config.fitting,apec_root=str(atomic),calculate_errors=False)
config.plotting = replace(config.plotting,formats=('png',))
result = TAPipeline(TAInput(target_id='EP260119a_software_validation',ra_deg=158.1018,dec_deg=65.4956,
    output_root=output/'ta',redshift=redshift,trigger_time_utc='2026-01-19T00:13:08.8249',
    fxt=FXTInput(target_id='EP260119a',root=source/'fxtevt',ra_deg=158.1018,dec_deg=65.4956,
                 spectra=tuple(fx_spectra))),
    fxt_options={'config':config,'galactic_nh_1e22':nh,'candidate_keys':('apec_nh0',)}).run()
assert result.status == 'completed', result.warnings
row = result.flux_rows[0]
assert row['redshift'] == 0.0 and row['target_redshift'] == redshift
assert row['redshift_status'] == 'fixed_stellar_zero'
assert row['extra_nh_interpretation'] == 'local_stellar_absorption'
assert all(sha256_file(Path(path)) == digest for path,digest in checksums.items())
coverage = {s.module:{key:fits.getheader(s.source_pha,1)[key] for key in ('TSTART','TSTOP','EXPOSURE')}
            for s in fx_spectra}
write_json(output/'validation.json',dict(environment=collect_runtime_environment(),
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    source_sha256={str(path):sha256_file(path) for path in [
        repo/'packages/jinwu/src/jinwu/core/fit.py',repo/'packages/jinwu-ep/src/jinwu/ep/fxt/pipeline.py',
        repo/'packages/jinwu-ep/src/jinwu/ep/ta.py']},
    inputs_sha256=checksums,original_inputs_unchanged=True,fx_module_coverage=coverage,
    observer_flux_band_keV=[.5,4.],input_target_redshift=redshift,galactic_nh_1e22=nh,
    apec_root=str(atomic),powerlaw_reference=str(reference_path),
    powerlaw_statistic_difference=control['statistics']['value']-reference['statistics']['value'],
    error_disabled_powerlaw_statistic=powerlaw['statistics']['value'],
    wxt_comparison=comparison.to_dict(),ta_result=result,
    limitations=['EP260119a spectra validate implementation; these trial APEC fits do not identify a nearby star.',
                 'Profile intervals disabled for bounded regression; no parameter-constraint claim.',
                 'Pile-up and slew calibration remain deferred; no new event reduction was needed.']))
print('VALIDATION',result.status,'APEC z=0; cflux retained; original inputs unchanged',flush=True)
