Changelog
=========

Unreleased
----------

* Added a resumable Swift BAT+XRT GRB pipeline (:mod:`jinwu.swift.grb`,
  registered ``"swift.grb"``) with CLI ``python -m jinwu.swift.grb``:
  catalog resolution with redshift priority, Burst Analyser ingestion with
  BadBin handling, prompt-stage classification, Bayesian blocks, per-segment
  XSPEC fitting and a summary report.
* Added a single-target Swift/BAT survey pipeline (:mod:`jinwu.swift.bat.survey`,
  registered ``"swift.bat.survey"``, CLI ``python -m jinwu.swift.bat.survey``):
  local-first discovery/download, survey lightcurves, optional mosaicking,
  Gaussian-rate spectral upper limits with a separate detection-sensitivity
  calibration.  Network stages are opt-in via ``--query`` / ``--download`` /
  ``--mosaic``.
* Added a resumable Fermi/GBM continuous-data pipeline (:mod:`jinwu.fermi.gbm`,
  registered ``"fermi.gbm"``, CLI ``python -m jinwu.fermi.gbm``): coverage and
  detector selection, TTE spectral extraction with AICc background-order
  selection, dual response backends (official ``SA_GBM_RSP_Gen.pl`` or
  pure-Python ``gbm_drm_gen``) and fixed-shape profile upper limits.
  :class:`jinwu.fermi.gbm.GBMObservation` no longer hardcodes a macOS
  poshist directory; it falls back to the ``GBM_POSHIST_DIR`` environment
  variable or the working directory.
* Added :mod:`jinwu.core.plotpanel` — multi-object overlay helpers
  (:func:`jinwu.core.multi_panel`, :func:`jinwu.core.overlay`,
  :class:`jinwu.core.PanelSpec`) for lightcurve- and spectrum-like data.
* Added the Gaussian net-rate upper-limit API
  (:class:`jinwu.core.upperlimit.GaussianNetRateObservation`,
  :func:`jinwu.core.upperlimit.profile_gaussian_upper_bound`,
  :class:`jinwu.core.upperlimit.EmpiricalCalibrationAdapter`): signed
  background-subtracted rates with per-channel errors or a full covariance
  are profiled with a non-negative source amplitude, and detection
  sensitivity is always calibrated separately from the observed bound.
* Added optional distribution extras: ``jinwu[bxa]`` (BXA + UltraNest),
  ``jinwu-swift[ukssdc]`` (swifttools), ``jinwu-swift[survey]``
  (BatAnalysis + astroquery) and ``jinwu-fermi[rsp]`` (gbm_drm_gen response
  stack).
* Deprecated :func:`jinwu.ftools.grppha_hsp`; use ``ftgrouppha`` instead.
* ``jinwu.core.io`` now also accepts RMF HDUs named ``SPECRESP MATRIX``.
* ``jinwu.core.ops.ensure_headas_env`` now completes the full HEASoft/Perl
  runtime (``LHEAPERL``, ``PERL5LIB``, ``PGPLOT_*``), which the BAT survey
  backends rely on when launched under ``conda run``.
* Pipeline framework: custom pipelines can hook fingerprinting and stage
  dependencies via ``include_config_in_input_fingerprint()``,
  ``stage_config_dependencies()`` and ``stage_input_dependencies()``;
  ``register_pipeline`` is idempotent under ``python -m`` double imports and
  entry points resolve lazily (e.g. ``swift.bat.survey``).
* Added Bayesian spectral fitting via BXA / UltraNest
  (:mod:`jinwu.core.bxa_fit`) and a process-wide fit-settings API
  (:func:`jinwu.core.set_fit_method`, :func:`jinwu.core.fit_settings`, and the
  unified dispatch entry point :func:`jinwu.core.fit.fit_spectral`).
* Added ``FitConfig.method`` (``mle`` / ``chain`` / ``bxa``) and
  ``InstrumentConfig.bxa`` (:class:`jinwu.core.config.BXAConfig`).  These
  fields enter the pipeline configuration fingerprint, so existing workspace
  manifests are invalidated **once** and re-run on first use after upgrading:
  WXT re-runs the full pipeline, while Swift survey / GBM re-run only their
  fit and report stages (their fingerprints are narrower).  This one-time
  recomputation is expected behaviour, not data corruption.
* Fixed ``txx_iterbkg`` input normalization (AUD-02): event-file paths are
  detected by content (``guess_ogip_kind``) instead of always being read as
  lightcurves; path and object inputs share one code path; results are
  expressed on the absolute mission-time grid (source/background files with
  different ``TIMEZERO`` offsets are now projected consistently — previously
  both were silently re-based to their own first sample); the returned dict
  adds ``alpha`` and ``time_reference``.  The EP event regression is marked
  ``real_data`` and runs in the real-data acceptance gate.
* Fixed WXT stage cache-code fingerprints (AUD-01): ``stage_code_dependencies``
  locates core modules via ``inspect.getsourcefile`` (the old
  ``parents[2] / "core"`` concatenation silently pointed at a non-existent
  directory, so core-code changes never invalidated caches), declares the
  full fit chain (``spectrum_prep`` / ``bxa_fit`` / ``config``) and
  ``timescale`` for the duration stage (``ops`` only re-exports it).  The
  base class now raises on declared-but-missing files instead of silently
  hashing them to a constant; Swift GRB / BAT declarations fail loudly the
  same way.
* Fixed :class:`jinwu.physics.GeneralRelativity` (AUD-03): ``g.v = 1.0`` no
  longer raises ``NameError``; ``beta`` is the analytic ``v/c`` instead of a
  placeholder ``0.0``; ``v >= c`` raises ``ValueError``; the ``show_*``
  helpers no longer require IPython at import time and fall back to
  plain-text output when it is absent.
* Release gating (AUD-04): a new PR/push CI workflow runs the offline test
  suite (core regression + plugin contracts + executable tutorials) on
  Python 3.11–3.13, and the publish workflow now blocks PyPI upload on a
  wheel gate — wheels are installed into a clean venv, entry points are
  discovered, and the offline gate tests run against the installed wheels.
* Quick Start / index examples now match the real API
  (``EnergyBand(emin=..., emin_unit=..., emax=..., emax_unit=...)`` and
  ``channel_mask_from_ebounds`` instead of the non-existent
  ``EnergyBand(..., unit=...)`` / ``ChannelBand.from_energy_band``) and are
  executed as tests against a fixed synthetic OGIP sample (AUD-05).
* Documentation build (AUD-06): the Sphinx version is read from installed
  distribution metadata (no silent ``0.0.27`` fallback), the dead ``src/``
  path hack is removed, and Read the Docs installs the four monorepo
  distributions instead of the non-installable repo root;
  ``sphinx-automodapi>=0.21`` is required for Sphinx 8.2+/9 compatibility.
* ``WXTPointingResult.display()`` keeps the same output contract
  (【标题】 headers first, summary last) with and without IPython.

v0.0.27 (2026-06-03)
--------------------

* Added Sphinx documentation with Read the Docs support
* Documentation now auto-generates API reference from docstrings

v0.0.26 (2026-05)
-----------------

* Added :class:`jinwu.core.upperlimit.UpperLimit` with chain and fit.error dual routes
* Added :mod:`jinwu.core.instruments` — EP FXT/WXT data directory scanning
* Added :meth:`jinwu.core.fit.run_xspec_chain` for MCMC chain fitting
* Telescope definitions in :mod:`jinwu.core.config`

Earlier Releases
----------------

*See the Git history for earlier changes.*
