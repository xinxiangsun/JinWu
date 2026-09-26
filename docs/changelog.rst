Changelog
=========

v0.2.0 (unreleased)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~

* ``from jinwu.core import timescale`` now resolves to the
  :mod:`jinwu.core.timescale` module.  On master the name referred to the
  timescale analyzer class in ``jinwu.core.data``; the new module silently
  shadowed it.  Use the ``LightcurveData.timescale()`` method or
  ``jinwu.core.data.timescale`` for the class.
* :meth:`jinwu.core.fit.LightcurveFitter.plot_fit
  <jinwu.core.fit.LightcurveFitter.plot_fit>` returns ``(ax1, ax2)`` again.
  The beta tree briefly returned ``(fig, (ax1, ax2))``, which silently bound
  the figure to ``ax1`` in master-style unpacking.
* :func:`jinwu.core.fit.fit_prepared`: the ``plot_dpi`` keyword is renamed
  back to ``plot_density`` (it feeds ``plotfit(density=...)``, a sampling
  density, not a DPI), and ``stat_method`` / ``abundance`` /
  ``cross_section`` carry explicit defaults (``cstat`` / ``wilm`` /
  ``vern``) again.  The function no longer resolves ``None`` against the
  process-wide fit settings, so ``set_fit_settings()`` can no longer change
  its behaviour as a hidden side effect — pass values explicitly or route
  through :func:`jinwu.core.fit_spectral` with ``settings=``.
* XSPEC result conversion data now reports ``'total_counts'`` while keeping
  ``'counts'`` as a compatibility alias.  Recoverable flux, rate, conversion,
  and statistic failures are recorded in the result warnings.
* The vestigial :mod:`jinwu.response` package is removed; GBM response
  generation lives in :mod:`jinwu.fermi.gbm.response`.
  ``jinwu.core.utils.generate_download_url`` moved to
  :mod:`jinwu.fermi.gbm` (a deprecated delegating shim remains in
  ``jinwu.core.utils``).
* ``InstrumentConfig.response_type`` is validated against the OGIP response
  vocabulary ``rsp2 | drm | rsp | rmf`` and every instrument preset declares
  its file type: Swift GRB ``"rmf"`` (was ``"rmf_arf"``), GBM /
  GBMContinuous ``"rsp2"`` (was ``"rsp"``).  ``response_type="rmf"``
  automatically sets ``response_requires_arf=True`` (an RMF must be paired
  with an ARF file, OGIP CAL/GEN/92-002).  The upper-limit contract check
  accepts ``{'rsp','rsp2','drm'}`` for ``response_folding='rsp'``.
* GECAM and Insight-HXMT mission elapsed time zero now corresponds to
  2019-01-01 and 2012-01-01 00:00:00 UTC, respectively; both formats use
  TT internally.  Any saved UTC labels produced from the former epochs
  should be regenerated.  Fermi/GW calendar grouping explicitly converts
  ``Time`` values to UTC before reading ``datetime``.
* Swift/BAT survey time conversion now uses JinWu's registered ``swiftmet``
  format.  ``extract_time_interval(time_format="swift")`` remains accepted as
  a compatibility spelling and resolves to ``swiftmet`` without requiring an
  optional external package to register ``swift`` with Astropy.
* ``jinwu.ftools`` now matches HEASoft 6.37 semantics: ``rebin_pha``
  (ftrbnpha) requires the output channel count to divide the input exactly,
  renumbers output channels from the first input channel and folds the
  QUALITY column; ``group_min_counts`` / ``compute_grouping_by_min_counts``
  (ftgrouppha/grppha) turn incomplete tail channels into single-channel
  groups flagged ``QUALITY=2`` (``grouping::loadMin``) instead of good data;
  ``rebin_rmf`` (ftrbnrmf) divides merged energy rows by the merged row
  count for REDIST-type responses (``rmf::rebinEnergies``).
* Release gating fix: the three offline gate test files referenced by CI and
  the publish wheel-gate (``test_quickstart_examples.py``,
  ``test_stage_code_deps.py``, ``test_gr.py``) are now tracked in git —
  ``.gitignore`` excluded the whole ``test/`` directory, so the wheel-gate
  (and therefore publishing) referenced non-existent files and never ran.
  CI additionally collects ``packages/jinwu-swift/tests``.

New and improved
~~~~~~~~~~~~~~~~

* Added :mod:`jinwu.core.skymap`, a dependency-light HEALPix probability-map
  reader (:class:`~jinwu.core.skymap.SkyMapData`, ``load_skymap``,
  ``sky_map_pixel_vectors``; requires the new ``jinwu[skymap]`` extra).
  The Fermi/GBM subthreshold search now takes its sky-map prior from this
  shared layer, so ``jinwu-fermi[search-skymap]`` no longer depends on
  ``jinwu-gw`` and the former ``jinwu-fermi <-> jinwu-gw`` dependency cycle
  is removed.  ``jinwu.gw.skymap.load_skymap`` keeps its richer
  URL-fetching/provenance/MOC behaviour for GW workflows.
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
* Added the standalone ``jinwu-gw`` distribution (:mod:`jinwu.gw`, entry point
  ``jinwu-gw plot``) for gravitational-wave localizations: LVK alert reading
  (embedded Base64 FITS, local FITS, FITS URL and public GraceDB superevents
  with alert version/checksum provenance), multi-resolution and flat HEALPix
  sky maps with 50%/90% credible regions, MOC-refined coverage integrals
  (order 10 -> 13, |dP| < 1e-3 convergence recorded), Fermi/GBM geometry from
  real POSHIST or the RapidGBM 30-orbit historical reference
  (:func:`jinwu.fermi.gbm.find_gbm_poshist`,
  :func:`jinwu.fermi.gbm.read_gbm_geometry`), time-tagged EP/BAT MOC, polygon
  and circle layers, and static all-sky/GBM diagnostic PNG+PDF plots with
  JSON/ECSV reports.  The pipeline machinery config contract is narrowed to
  the ``PipelineConfigProtocol`` (``name``/``pipeline``/``execution``) so the
  GW workflow ships its own lightweight configuration.
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
* Under review (annotated in code, logic unchanged): the GECAM and HXMT
  epochs in :mod:`jinwu.core.time` look 69.184 s early relative to a
  "UTC zero point + TT counting" convention — see the ⚠️ comments on
  ``TimeGECAM`` / ``TimeHXMT`` and verify against official mission
  documentation and real event files before correcting.

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
