Swift BAT+XRT GRB Pipeline
==========================

:mod:`jinwu.swift.grb` analyses a single gamma-ray burst using Swift BAT
Burst Analyser products together with XRT data.  It is a resumable,
per-GRB pipeline ported from the ``swift_highz_grb`` research workflow,
registered as ``"swift.grb"`` and built on the generic
:mod:`jinwu.core.pipeline` machinery: stage outputs are cached in a
workspace manifest, so reruns pick up at the first incomplete stage.

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install jinwu-swift
   pip install "jinwu-swift[ukssdc]"   # optional: UKSSDC catalog queries + Burst Analyser downloads

The catalog, download and prompt stages work without the optional
dependency.  The ``fit`` stage additionally requires a working
HEASoft/PyXspec environment.

Stages
~~~~~~

::

   catalog → download → {lightcurve, prompt, galactic_absorption, bblocks}
           → spectra → fit → report

* **catalog** — resolve the GRB in ``catalog/swift_grb_catalog_merged.csv``
  (a research merge of SDC, UK_XRT and BAT_GRB catalogs) with redshift
  priority selection and the NASA BAT duration windows.
* **download** — verify local Burst Analyser ``parsed/`` products; the
  ``SwiftBurstAnalyserFetcher`` (3-tier strategy with BadBin preservation,
  requires the ``ukssdc`` extra) is optional and local XRT product
  directories are always preferred.
* **lightcurve** — BadBin-filtered BAT + XRT WT/PC flux series written to
  an NPZ plus a three-panel figure (flux / N_H / Γ) in the research
  styling.
* **prompt** — strict prompt-stage judgment (XRT start inside the absolute
  BAT T90 window); missing inputs yield ``UNDECIDED``.
* **galactic_absorption** — weighted Galactic N_H via
  ``jinwu.core.galactic.resolve_galactic_absorption``, cached per workspace.
* **bblocks** — Bayesian blocks on XRT source/background events (WT
  per-GTI, PC per observation, GTI fallback), Li-Ma significance with
  SRCAREA/BGAREA alpha, and low-SNR segment merging.  A missing area scale
  stops that segment instead of assuming alpha = 1.
* **spectra** — XSELECT source/background PHA per segment, grouped with
  ``grppha_hsp`` and linked to the interval-0 RMF/ARF.
* **fit** — :func:`jinwu.core.fit.fit_prepared` per segment with the
  research model ``tbabs*ztbabs*cflux*powerlaw`` and frozen Galactic
  N_H (HEASoft/PyXspec required).
* **report** — ``grb_summary.json``, ``summary.txt`` and
  ``spectral_parameters.csv``.

Command line
~~~~~~~~~~~~

The CLI works against a research-style workspace:

.. code-block:: bash

   python -m jinwu.swift.grb \
       --root ~/research/swift_highz_grb/output \
       --grb 050904 --nh 0.09 --output-dir /tmp/grb050904_jinwu

``--grb`` accepts a GRB name or merge key such as ``050904A`` or
``"GRB 050904"``.  The most useful options:

``--until STAGE``
    Stop after the given stage (e.g. ``--until bblocks``).
``--no-resume``
    Ignore cached stage manifests and recompute everything.
``--redshift / --ra / --dec / --nh``
    Override catalog values; ``--nh`` (in 10²² cm⁻²) skips the nhtot
    query.
``--fetch``
    Enable the Burst Analyser fetcher (requires ``jinwu-swift[ukssdc]``).
``--output-dir DIR``
    Write pipeline products to a separate directory and keep the input
    workspace read-only.
``--catalog CSV``, ``--raw-products-dir DIR``, ``--xrt-products-dir DIR``
    Override the default product locations.
``--max-segments-per-mode N``
    Extract only the first N WT and PC segments (fast trial runs).

.. note::

   ``--request-xrt`` is the only switch that can submit a UKSSDC XRT
   product request, and it needs a registered account
   (``--xrt-user`` or ``SWIFT_XRT_USER``).  A submitted request is saved
   and later resumed by ``--poll-xrt``; an ambiguous submission is never
   sent twice automatically.

.. code-block:: bash

   # Only with a registered UKSSDC XRT-product account:
   python -m jinwu.swift.grb --root ~/research/swift_highz_grb/output --grb 050904 \
       --request-xrt --xrt-user "$SWIFT_XRT_USER"

Python API
~~~~~~~~~~

The :class:`jinwu.core.config.SwiftGRB` preset carries the instrument
configuration; the input dataclass is
:class:`jinwu.swift.grb.SwiftGRBInput`:

.. code-block:: python

   from jinwu.core.config import SwiftGRB
   from jinwu.core.pipeline import pipeline
   from jinwu.swift.grb import SwiftGRBInput

   result = pipeline(
       SwiftGRB(),
       SwiftGRBInput(target_id="050904", root=workspace,
                     output_root="/tmp/grb050904_jinwu",
                     xrt_products_dir=workspace / "xrt_product_requests_maybe/downloads/GRB_050904"),
   ).run()
   print(result.science_status, result.products["report"])

Preset notes
~~~~~~~~~~~~

* The BAT Burst Analyser product is ``SNR5_sinceT0:BATBand`` and
  **BATBand is 15–50 keV** — distinct from Swift/BAT's 15–150 keV
  instrument range.
* XRT fitting uses 0.3–10 keV, C-stat (reported as W-stat when a Poisson
  background is loaded), grouping of 20 counts, and Δstat = 1 profile
  errors.
* The fit stage honours the global fit settings described in
  :doc:`bxa_fitting`, so ``set_fit_method("bxa")`` also affects GRB
  segment fitting.

Resumability and review semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every completed stage stores a manifest entry under the workspace.  When a
stage cannot proceed scientifically (e.g. missing network products) it is
marked ``needs_review`` instead of failing the whole run; after the missing
data are added locally, rerunning the same command resumes from the
interrupted point.  Use ``--no-resume`` to force a full recompute.

See also
~~~~~~~~

* :doc:`bxa_fitting` — Bayesian fitting of the per-segment spectra.
* :doc:`upperlimits` — bound/sensitivity semantics used across pipelines.
* :doc:`../api` — auto-generated API reference for
  :mod:`jinwu.swift.grb`.
