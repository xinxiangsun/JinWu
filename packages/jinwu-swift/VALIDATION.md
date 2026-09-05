# Swift GRB pipeline validation record

Date: 2026-09-01

- Offline contracts: `84 passed` with the Swift GRB, core-config, and core
  pipeline tests.  They cover the registered configuration preset, 14-column
  `Flux/ECF` rate provenance, `.area` start/stop/background parsing, safe
  archive rejection, stage caching, and the synthetic BAT/XRT chain.
- GRB 050904 local products: `catalog` through `bblocks` completed in an
  independent output directory.  The produced WT/PC segment times are seconds
  since the recorded trigger MET; rerunning with resume reused the manifests.
- Limited WT+PC extraction was attempted with one segment per mode.  The
  pipeline now discovers the HEASoft `xselect` executable and supplies
  `HEADAS`, `LHEA_DATA`, `PATH`, and stage-local `PFILES`.  XSELECT accumulated
  the selected events, but `jinwu.core.xselect` did not find the PHA after the
  command reported it was written.  The spectra stage therefore correctly
  returned `needs_review`; no fit or scientific spectral parameter is claimed.
- The GRB spectral fit was not reached because the limited XSELECT extraction
  did not produce a discoverable PHA.  The fit stage records the failure per
  segment and never substitutes a redshift of zero when the catalog redshift
  is absent.  The BAT survey section below separately validates PyXspec with a
  real local PHA/RSP and a writable stage-local cache.

## BAT survey single-target pipeline

- **Offline contracts:** `41 passed, 6 warnings` in
  `tests/test_bat_survey.py`.  The suite covers the default and explicit
  `lmjagn` presets, configuration propagation, signed eight-band rates with a
  separate total band, `RATE_ERR`/`BKG_VAR`/`SYS_ERR` provenance, QDP and
  Flux/ECF fallbacks, all-zero quality review, Swift-MET and GTI overlap
  boundaries, compressed FITS validation, safe archive extraction, response
  MATRIX/EBOUNDS and channel checks, bounded retry/no-resubmission behavior,
  isolated cache loading, stage-local HEASoft variables, input/calibration
  cache invalidation, `--until` versus full-run recovery, response-extension
  ordering, the synthetic PHA → survey → light curve → fit → report chain,
  and mosaic detection from the mosaic's own source catalogue.
- **Affected regression:** `1041 passed, 65 skipped, 22 warnings, 15 subtests`
  in the local suite run with `external_sources` and the historical
  `txx_iterbkg` validation module excluded. The focused upper-limit/GBM/BAT
  selection was `97 passed, 1 skipped` after the total-band,
  false-alarm-policy, and compressed-response regressions were added. Tests requiring real HEASoft or
  local products use the explicit `heasoft` and `real_data` markers; the
  default suite does not make a network request.
- **HEASoft smoke test:** after setting a writable temporary `HOME`, PFILES
  and matplotlib cache, sourcing
  `/home/xinxiang/miniconda3/envs/hea/bin/heainit.sh` found
  `batanalysis`, `heasoftpy`, PyXspec and
  `/home/xinxiang/miniconda3/envs/hea/heasoft/bin/xspec`.  An empty XSPEC
  session exited with return code 0.  The pipeline also reproduces these
  `heainit.sh` runtime variables in-process, while keeping each stage's
  `HOME`, PFILES and log directory private.  The installed CALDB pair is
  recorded as `CALDBCONFIG`/`CALDBALIAS` with SHA-256 checksums; the preflight
  manifest also stores each imported module's source path and version (`BatAnalysis 2.1.0`,
  `heasoftpy 1.5`, PyXspec `2.1.5`/XSPEC `12.15.1`).  No observation query,
  download or online product submission was enabled.
- **Existing AT20G spectrum:** with
  `JINWU_RUN_REAL_DATA=1`, the actual PHA/RSP for OBSID `00097302084`
  completed the full local pipeline and PyXspec upper-limit branch.  The
  eight-channel PHA has `EXPOSURE=828.0 s`, uses Gaussian `chi`, and is not an
  old `bkgnsigma_*_upperlim` product.  The fit returned
  `fit_statistic=3.985153073941337`, `upper_norm=0.13038704322781092`,
  `flux_erg_cm2_s=5.502386284604323e-10`, and a profile recheck of
  `Δstat=9.0` for the configured `Δstat=9`.  The constrained MLE is at the
  physical lower boundary (`signed_mle=-0.05711003812548571`); this is retained
  in the report and the finite upper bound is accepted only because the
  independent profile recheck crossed the requested level.  The opt-in test
  result was `1 passed, 6 warnings`.
- **Finite mosaic:** the existing local products for AT20G pointings
  `00097302084` and `00097556048` completed a two-member `lmjagn` mosaic in
  `/tmp/jinwu-bat-at20g-mosaic4`.  The output records full member exposures
  `828.0 s` and `823.0 s`, GTI overlaps `825.8416199684143 s` and
  `824.6006000041962 s`, and total overlap `1650.4422199726105 s`.  The
  requested target's own mosaic catalogue measurement has SNR about `0.267`
  and is therefore `not_detected` at 3σ; no member-pointing SNR is reused.
  Shared-member metadata is present for windows that reuse a pointing.
- **Raw Mrk 79 preparation:** the local `00092462003` raw tree passed the
  observation checks, and the repaired HEASoft environment produced all three
  14–195 keV rebinned DPH files, `gti/master.gti`, `gti/dph.gti`, and the
  `point_20253220835` survey image products in the isolated
  `/tmp/jinwu-bat-mrk79-envfix` tree.  The long image stage was interrupted
  before the pipeline wrote a completed `survey.json`/source PHA, so this is
  recorded as **preparation evidence**, not a successful raw-data science
  validation.  The reference sibling directory still contains only a pickle,
  marker, DPH intermediates and `stats_obs.dat`; it is not adopted as a valid
  survey product.
- The AT20G DPH date and derived PHA date differ by 29 s in the reference
  files.  The pipeline records the source time system and keeps the requested
  window's GTI overlap separate from the complete pointing exposure; it never
  scales a full-pointing PHA by a short-window overlap and does not promise
  arbitrary second-level DPH spectra.
- Reports retain `science_status=partial` and a missing-product list whenever
  a quality, dependency or product gate fails.  Such stages carry
  `cache_reusable=false`; changing an input, response, calibration file or
  fitting configuration invalidates only the dependent stages, leaving raw
  inputs and prior manifests untouched.

## 2026-09-01 upper-limit/statistics migration

- The shared core now provides the unit-aware `GaussianNetRateObservation` and
  `profile_gaussian_upper_bound` interfaces. BAT keeps signed net rates and
  returns both the unconstrained signed MLE and the physical `A >= 0` MLE;
  `STAT_ERR` and OGIP `SYS_ERR` are combined once. The profile result is
  labelled `conditional_model` until an instrument-native calibration is
  supplied.
- BAT fixed-position sensitivity is a separate product. The optional
  `BATSurveySensitivityAdapter` uses the native eight-channel TOTSNR control
  statistic and deterministic injections. A short or invalid blank-sky tail
  produces `calibration_status=unavailable`; no finite calibrated sensitivity
  is substituted.
- GBM uses a Poisson source likelihood with a Gaussian background nuisance.
  Background integration uses the exact requested interval and PHAII exposure,
  optional BAK covariance replaces (rather than supplements) diagonal
  `STAT_ERR`, and RSP2 weighting uses actual TTE GTI overlap. The residual gate
  now reports finite-sample binomial intervals for 68% and 95% predictive
  coverage. A failed residual/prediction check marks the final report `analysis_status=needs_review`
  while retaining the conditional profile.
- Offline regression after this migration covers the core, GBM, and BAT
  contracts. The opt-in HEASoft BAT acceptance test and the initialized GBM
  PHA/BAK/RSP background-exposure contract test both ran successfully. The GBM
  product still fails the independent off-source residual gate, so no
  calibrated GBM flux or upper limit is claimed. No online query, observation
  download, or GECAM science result is claimed here; reference
  `batsurvey_lmjagn` outputs remain read-only.
