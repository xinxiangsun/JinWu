GBM Haar 最小变化时标（MVT）
=======================================================

支持均匀总计数光变与 GBM TTE 的独立可恢复流程。
输出测量/上限/未分辨状态、Haar scaleogram、重采样与 bin-width / SNR 诊断。

``jinwu.fermi.gbm.mvt`` provides a standalone, resumable MVT workflow and a
light-curve API. It pins ``xinxiangsun/GBM_MVT_paper`` commit
``57e7a0fe7e5b27311acb161f622f4a68b2e0e8c7``. The Haar calculations retain
the original numerical statements, thresholds, normalization and optimizer.
Spectral responses, XSPEC and targeted-search templates are not required.

The method is described by `Golkhou & Butler (2014)
<https://arxiv.org/abs/1403.4254>`_ and the resampling/validation workflow by
`Bala et al. (2026) <https://arxiv.org/abs/2512.16204>`_. This port is a
versioned implementation, not a claim that every published scientific result
has been independently reproduced.

Light-curve API
---------------

.. code-block:: python

   import astropy.units as u
   from jinwu.fermi.gbm.mvt import compute_mvt

   # counts: uniform per-bin TOTAL counts, not counts/s or background-subtracted data.
   estimate = compute_mvt(counts, bin_width=0.1 * u.ms)
   print(estimate.estimator_status, estimate.mvt, estimate.error)
   print(estimate.raw_values)       # original seven-value Haar return
   print(estimate.diagnostics)      # denoising weights and scaleogram arrays

Without explicit errors, the estimator uses ``sqrt(counts)``. Signed data
require explicit count errors. Gaps must not be filled with zero counts.
``estimator_status`` separates ``measurement``, ``upper_limit``, ``unavailable``
and ``failed``. The original zero-error limit is never a zero-error detection.

Real GBM TTE
------------

.. code-block:: python

   import astropy.units as u
   from jinwu.fermi.gbm.mvt import GBMMVTInput, GBMMVTConfig, run_gbm_mvt

   inputs = GBMMVTInput(
       target_id="GRB170817A", trigger_id="bn170817529",
       root="/path/to/read-only-tte", output_root="/path/to/new-mvt-output",
       source_interval=[-2.2, 4.3] * u.s,
       background_intervals=[[-62.9, -12.9], [14.9, 64.9]] * u.s,
   )
   settings = GBMMVTConfig(
       detectors=("na", "n1", "nb", "n4", "n5", "n0"),
       energy_range=[8, 900] * u.keV, bin_widths=[1, .1] * u.ms,
       n_resamples=300, seed=0, background_order=1, workers=1,
   )
   result = run_gbm_mvt(inputs, config=settings)
   print(result.science_status, result.summary["interpretation"])
   print(result.products["report"])

For multiple workers in a standalone Python script, put the run under an
``if __name__ == "__main__":`` guard. The CLI already supplies that guard.

Triggered files supply their reference time from ``TRIGTIME``. Continuous
TTE requires ``trigger_time`` (a scalar JinWu ``Time`` or UTC string). Explicit
``tte_paths`` are also accepted. Data are local-first; ``download=True``
requests archive products into the output cache. Missing files, changing
energy calibration or incomplete source/background GTIs are reported, not
silently clipped. BGO is outside the pinned NaI calibration scope.

The shared TTE reader preserves within-file event multiplicities while
removing duplicate events across overlapping products. Energy selection uses
GDT's channel-boundary semantics. The real-data adapter fits an integrated
selected-band GDT polynomial with explicit order and dead-time exposure.
``background_order=0`` is the constant default; order 1 was used for the
recorded real-data acceptance runs. The source counts entering Haar remain
unchanged by background fitting.

``detectors="auto"`` ranks the available NaI detectors at 64 ms (16 ms when
an explicit ``t90<1 s`` is supplied), evaluates cumulative prefixes, and
refines the choice using the conditional median MVT. Its scope is the supplied
detectors; supplying only a subset does not establish a global twelve-detector
optimum. Each candidate set, SNR and iteration is recorded. Fixed detector
sets bypass optimization and are useful for reproducible comparisons.

CLI equivalent
~~~~~~~~~~~~~~

.. code-block:: bash

   python -m jinwu.fermi.gbm.mvt \
     --target GRB170817A --trigger-id bn170817529 \
     --root /path/to/tte --output /path/to/new-output \
     --source -2.2 4.3 --background -62.9 -12.9 --background 14.9 64.9 \
     --detectors na n1 nb n4 n5 n0 --bin-widths-ms 1 .1 \
     --resamples 300 --seed 0 --background-order 1 --workers 3

Results and interpretation
--------------------------

The report records the source/background windows, nominal and actual energy
channel bounds, observer frame, input SHA256 hashes, package versions, Haar
settings, stream identities, background fit and dead-time diagnostics.
``prepared_events.npz`` and per-resolution light curves, original scaleograms,
all resample returns, state counts, and diagnostic PNGs are retained. Core
pipeline manifests verify code, calibration, data and output hashes on resume.

Observed per-bin counts are the means of independent Poisson draws. Stream
``[seed, detector_iteration, resolution_index, sample_index]`` makes draws
independent of worker count. This does not reproduce an unspecified historical
random stream. As in the frozen wrapper, MVT/error values are rounded to
0.001 ms and only valid samples with positive rounded analytic error enter
the 16/50/84 percentiles. Raw unrounded values and all excluded states remain
available. The percentile interval is conditional on measured samples and
does not include background-fit uncertainty.

The paper's frozen SNR helper computes ``peak TOTAL counts / sqrt(mean
background cps * MVT seconds)``. It is not a detection sigma. This exact
definition is retained; its weak-source SNR differs from the published
GRB170817A pair. A numerical correction or a different SNR definition requires
a separately validated method version. Therefore classification here means
the frozen code's empirical-curve result, not verified reproduction of every
paper classification.

It differs from ``gv_significance.poisson_gaussian.significance(n,b,sigma)``:
that function uses the Poisson ON count and Gaussian fitted-background error
in a signed profile-likelihood statistic. Its value cannot replace the frozen
helper on the existing validation curve. The same-bin GBM comparison is
recorded in ``reviews/gbm-mvt-pg-snr.md``. The original GBM simulation helper
estimates background from all native-channel background events, then selects
the peak light-curve energy band; its default bin origin follows the first
event. The real-data adapter explicitly uses a selected-band fit and the
source-window origin. These input conventions are recorded; the same-input
Haar equivalence checks do not establish simulation-platform or published
SNR reproduction.

The pinned curve uses log-space interpolation. Its finite domain is recorded;
outside it, the result is ``uncalibrated``. The two finest conditional
16--84 percentile intervals are checked for overlap. No stable plateau means
``unresolved_upper_limit`` when a percentile estimate exists. Algorithmic
limits, empirical classifications, bin-width stability, and background
quality are distinct fields. ``science_status="completed"`` records available
conditional estimates, background diagnostics and detector convergence.
Empirical classification uses the frozen proxy and recorded input conventions.
The reported percentile upper bound is an
empirical conditional bound, not a guaranteed frequentist coverage statement.

Licensing and migration evidence
--------------------------------

The fork declares Apache-2.0 in its package metadata, README and initializer,
although it has no standalone LICENSE. The port includes Apache text,
attribution, modification notice and source hashes under ``mvt/_vendor``.
The original ``nrbutler/mvt`` repository has no explicit license declaration
located in this audit. That provenance question remains unresolved before
redistribution; local implementation and numerical acceptance do not settle it.

``scripts/validate_gbm_mvt.py`` takes the untouched upstream source as input.
It checks numerical function AST equality, seven return values, eight classes
of intermediate arrays and every requested real-data Poisson sample against
the saved migrated result. Evidence is in ``reviews/evidence/gbm-mvt/``.
See ``reviews/gbm-mvt-migration.md`` for current acceptance results and limits.
