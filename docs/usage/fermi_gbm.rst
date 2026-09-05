Fermi GBM Continuous-Data Pipeline
==================================

:mod:`jinwu.fermi.gbm` provides a resumable, single-target pipeline for
Fermi/GBM continuous (TTE/CSPEC) data, registered as ``"fermi.gbm"`` and
built on `astro-gdt <https://astro-gdt.readthedocs.io/>`_ and the generic
:mod:`jinwu.core.pipeline` machinery.

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install jinwu-fermi
   pip install "jinwu-fermi[rsp]"   # optional: pure-Python response stack (gbm_drm_gen)

Two response generators are supported (see `Response backends`_); the
official one additionally needs the HEASARC ``SA_GBM_RSP_Gen.pl`` Perl
environment, while the ``rsp`` extra needs the BALROG detector database
(``BALROG_DB``).  The ``fit`` stage requires a working HEASoft/PyXspec
environment.

Stages
~~~~~~

::

   preflight → coverage → detectors → download → windows
                                                 ↙        ↘
                           lightcurve    spectra → response → fit → report

* **preflight** — check for ``gdt-data``, both response generators and
  PyXspec, and write ``preflight.json``; missing tools do not fail the
  run, they gate later stages.
* **coverage** — locate (optionally download) position-history files and
  run visibility / spacecraft-state / GTI coverage checks.  A ``none`` or
  ``data_missing`` outcome is recorded as ``needs_review``, never as a
  non-detection.
* **detectors** — select the best NaI (default angle limit 60°, at most
  3) and BGO (90°, at most 2) detectors geometrically.
* **download** — fetch TTE/CSPEC only for the selected detectors, with
  HTTP-range resume; the download range automatically covers the largest
  background window.  Local-first: nothing is fetched unless
  ``--download`` is given.
* **windows** — source window (from the covered segment or explicit
  ``--window``) plus tight polynomial background windows before and
  after the burst (10 s guard, 300–1800 s per side, truncated by data
  coverage).
* **lightcurve** — binned lightcurves per detector in the analysis band
  (NPZ + PNG); product failures degrade to warnings.
* **spectra** — per-detector OGIP PHA/BAK from TTE, with an AICc-based
  automatic choice among polynomial background orders 0–2.
* **response** — see `Response backends`_.
* **fit** — per detector group (nai/bgo) × photon-index grid (2.0 main
  value + 1.5/2.5 sensitivity) fixed-shape power-law profile upper limits
  via :func:`jinwu.core.upperlimit.estimate_upper_limit`.  Source counts
  stay Poisson; the BAK ``STAT_ERR`` (plus explicit covariance
  extensions) enters as a Gaussian background nuisance.  No uncalibrated
  5% systematic error is added by default.
* **report** — ``report/gbm_summary.json`` + ``summary_row.csv``.

Command line
~~~~~~~~~~~~

.. code-block:: bash

   python -m jinwu.fermi.gbm Mrk421 \
       --ra 166.1138 --dec 38.2088 \
       --start 2024-07-16T12:00:00 --stop 2024-07-16T12:00:30 \
       --root ~/data/gbm-cache --output /tmp/mrk421_gbm --download

``python -m jinwu.fermi.gbm.pipeline`` is an equivalent entry point.
Required arguments are ``--ra``, ``--dec``, ``--start``, ``--stop``
(UTC ISO 8601) and the local-first data search root ``--root``.  Other
useful options:

``--window START STOP``
    Explicit source window (otherwise taken from the covered segment).
``--background-window START STOP``
    Override an automatically chosen background window; repeatable.
``--poshist FILE``
    Explicit position-history file (repeatable).
``--response-backend {auto,official,gbm_drm_gen}``
    Override the configured response backend.
``--download``
    Fetch missing archive products.
``--until STAGE``
    Stop after the given stage (e.g. ``--until windows``).
``--no-resume``
    Ignore cached stage manifests.
``--verbose``
    More logging.

Exit codes: ``0`` = completed, ``2`` = finished with
``needs_review`` outcomes that require human judgement, ``1`` = failure.

Python API
~~~~~~~~~~

The :class:`jinwu.core.config.GBMContinuous` preset carries the
instrument configuration; analysis parameters (detector angle limits,
background windows, photon-index grid, energy band, significance
threshold, …) are centralised in ``GBMAnalysisConfig`` and can be
overridden with keywords such as
``GBMContinuous(lc_bin_s=0.5, response_backend="gbm_drm_gen")``.

.. code-block:: python

   from jinwu.core.config import GBMContinuous
   from jinwu.core.pipeline import pipeline
   from jinwu.fermi.gbm import GBMPipelineInput

   result = pipeline(
       GBMContinuous(),
       GBMPipelineInput(target_id="Mrk421", root="~/data/gbm-cache",
                        output_root="/tmp/mrk421_gbm",
                        source_name="Mrk421", ra_deg=166.1138, dec_deg=38.2088,
                        start_utc="2024-07-16T12:00:00",
                        stop_utc="2024-07-16T12:00:30",
                        download=True),
   ).run()
   print(result.science_status, result.science_result, result.products["report"])

``root`` is the read-only search root for already-downloaded GBM products
(for example a shared Fermi archive cache); new downloads always go to
``<output>/cache``.  Interrupted runs resume from the workspace
``.pipeline/`` manifest when the same command is rerun.

Response backends
~~~~~~~~~~~~~~~~~

Two interchangeable backends generate the detector response matrices:

* **official** — ``SA_GBM_RSP_Gen.pl`` with CSPEC + position history; the
  RSP2 matrices are weighted by the actual GTI exposure.
* **gbm_drm_gen** — the pure-Python ``DRMGenTTE`` stack (single matrix at
  the segment midpoint).  The midpoint approximation is only kept when
  its folded rate in the analysis band agrees with the GTI-weighted
  response to within 1%; otherwise the weighted DRM is written out.

Both branches produce normalised products (1-based ``CHANNEL``
renumbering, ``MATRIX`` extension name) that are consistency-checked
through :func:`jinwu.fermi.gbm.read_ogip_products`.

Scientific gating
~~~~~~~~~~~~~~~~~

No stage publishes a flux or an upper limit without a validated
PHA/BAK/RSP triple; a missing gate is recorded as ``needs_review`` with
the next-step requirement written into the report, instead of fabricating
a non-detection.  The background model is checked with Poisson deviance
on source-free time blocks (residual mean, trend, and 68/95% prediction
coverage); if any check fails the report keeps the conditional profile
values but marks ``analysis_status`` as ``needs_review``.  The BAK
``EXPOSURE`` must match the source PHA's actual GTI exposure (up to
floating-point rounding); short windows only change the GTI overlap
record and never rescale the full source spectrum.  ``sqrt(null_statistic)``
is reported as a fixed-position model-significance diagnostic, and the
conditional upper limit is only quoted below the significance threshold.

See also
~~~~~~~~

* :doc:`upperlimits` — the ``poisson_gaussian_profile`` statistics shared
  with BAT survey upper limits.
* :doc:`bxa_fitting` — Bayesian fitting of the extracted spectra.
* :doc:`../api` — auto-generated API reference for
  :mod:`jinwu.fermi.gbm`.
