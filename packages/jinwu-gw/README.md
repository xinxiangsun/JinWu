# jinwu-gw

`jinwu-gw` turns a public LIGO/Virgo/KAGRA alert or a local HEALPix
localization into a reproducible sky-coverage report.  The first release is
deliberately focused on visualization: it combines the GW probability map
with Fermi/GBM geometry and optional EP/BAT footprints.  It does not claim
that a geometric coverage result is a valid GBM exposure or spectral product.

```bash
jinwu-gw plot --notice alert.json --output results/
jinwu-gw plot --event S240422ed --output results/
jinwu-gw plot --skymap sky.multiorder.fits --time 2024-04-22T00:00:00Z \
    --gbm-cache ~/.cache/jinwu-gw/gbm --output results/
```

`--skymap` may also be an HTTPS FITS URL.  The output directory always
contains the all-sky and GBM diagnostic PNG/PDF (the latter is labelled
`unknown` when no POSHIST is available), `coverage.json`, and `coverage.ecsv`.
The CLI uses `$GBM_POSHIST_DIR` or `~/.cache/jinwu-gw/gbm` for GBM
POSHIST products when `--gbm-cache` is omitted; use `--no-download` for a
strictly offline run.
The JSON records the alert/map checksum, credible-region areas, MOC-order
convergence, coverage status and the explicit `spectral_analysis: "not_run"`
state.  A geometric GBM result is not a validated exposure or spectrum.
