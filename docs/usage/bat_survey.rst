Swift BAT Survey Pipeline
=========================

:mod:`jinwu.swift.bat.survey` processes one known-position target from
Swift/BAT survey data.  It is a small, testable adapter around
`BatAnalysis <https://github.com/parsotat/BatAnalysis>`_ that owns the
survey data contract (native energy channels, signed
background-subtracted rates, GTI overlap and provenance).  Importing the
module neither imports BatAnalysis or HEASoft nor starts a network
request; the pipeline is registered as ``"swift.bat.survey"``.

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install jinwu-swift
   pip install "jinwu-swift[survey]"    # BatAnalysis >= 2.1 + astroquery for HEASARC queries
   pip install "jinwu-swift[ukssdc]"    # optional: UKSSDC catalog queries

The ``fit`` stage and the spectral upper limits additionally require a
working HEASoft/PyXspec environment.

Stages
~~~~~~

::

   preflight → discover → download → survey → lightcurve → mosaic
             → spectra → fit → report

The ``download`` and ``mosaic`` stages are optional and only run when
explicitly requested.  Existing products can be injected directly with
``--survey-products`` / ``--mosaic-products`` (or ``--raw-products`` for
raw per-observation directories); directories that contain only a
``batsurvey.pickle`` or completion markers are *not* accepted as usable
science products — PHA, response and processing status are re-validated
in the spectra stage.

Local-first semantics
~~~~~~~~~~~~~~~~~~~~~

Network access is always opt-in:

* ``--query`` enables online target discovery via HEASARC;
* ``--download`` enables downloading missing survey/mosaic products;
* ``--mosaic`` enables time-window mosaicking.

Without these flags the pipeline works strictly from local data.

Command line
~~~~~~~~~~~~

.. code-block:: bash

   python -m jinwu.swift.bat.survey NGC4253 \
     --root /data/swift/batdata --output /tmp/ngc4253_bat_survey \
     --source-name NGC4253 --ra 183.5625 --dec 29.8125 \
     --obsid 00098092002 \
     --window 2024-07-16T12:00:00 2024-07-16T13:00:00

The positional argument is a ``target_id``.  Frequently used options:

``--ra / --dec``
    ICRS position in degrees (with ``--source-name`` for labels).
``--obsid``, ``--window START STOP``
    Observation IDs and UTC time windows; both can be repeated.
``--query``, ``--download``, ``--mosaic``
    Explicitly enable the online stages (see above).
``--detthresh``, ``--detthresh2``, ``--min-pcode``
    Survey detection thresholds and partial-coding cut.
``--profile lmjagn``
    Use the reference project's 8000/0.01 filtering configuration and
    0.01 mosaic partial-coding threshold (default: 10000/0.05).
``--sensitivity-controls FITS``
    Local blank-sky BAT survey rate product used to estimate the
    fixed-position detection sensitivity (see below).
``--until STAGE``, ``--no-resume``
    Stage stop and cache-control switches, as in every jinwu pipeline.

Resource and network controls: ``--processes`` (backend worker
processes), ``--internal-threads`` (math threads per worker),
``--task-timeout`` (external survey/mosaic task timeout, default 3600 s),
and ``--network-timeout`` / ``--retries`` / ``--retry-wait`` /
``--query-margin`` for the network stages.

Python API
~~~~~~~~~~

.. code-block:: python

   from jinwu.core.config import BATSurvey
   from jinwu.swift.bat.survey import BATSurveyInput, BATSurveyPipeline

   job = BATSurveyInput(
       target_id="NGC4253",
       root="/data/swift/batdata",
       output_root="/tmp/ngc4253_bat_survey",
       source_name="NGC4253",
       coord=(183.5625, 29.8125),       # ICRS degrees
       obsids=("00098092002",),
       time_windows=(("2024-07-16T12:00:00", "2024-07-16T13:00:00"),),
       download=False,                  # local-first; explicit opt-in
       mosaic=False,
   )
   result = BATSurveyPipeline(job, config=BATSurvey()).run()
   print(result.science_status, result.products["report"])

The annotated script
``examples/bat_survey_pipeline.py`` in the ``jinwu-swift`` distribution
accepts the same parameters as the CLI and is a good starting point for
custom drivers.

Survey data contract
~~~~~~~~~~~~~~~~~~~~

* All eight native survey CAT energy channels are preserved; a ninth
  total channel is derived by error-weighted summation only when the
  product ships without one (an existing ninth column is never summed
  again).
* Survey rates keep the sign of ``CENT_RATE``; ``RATE_ERR`` is preferred
  as the rate error, with ``BKG_VAR`` as a documented fallback.  They are
  Gaussian count s⁻¹ fully-illuminated-detector measurements, **not**
  energy fluxes.
* Requested windows and actual full-pointing exposures are reported
  separately: a short request window never rescales a full-pointing PHA.
  If per-pointing ``*.gti`` files exist, the lightcurve selection records
  the real GTI overlap, used only for window-coverage bookkeeping.
* QDP lightcurves flagged ``native_qdp_rate`` carry single-point time
  errors only and are excluded from window spectrum selection.

Upper limits and sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BAT survey spectral upper limits use a Gaussian profile on the signed
``RATE``: ``STAT_ERR`` is the statistical error, the OGIP ``SYS_ERR``
column is converted and added exactly once, and ``BKG_VAR`` is used only
to calibrate the native TOTSNR sensitivity — never mixed into the
spectral fitting errors.  The response template is generated with
``fakeit(applyStats=False)`` from the same PHA/RSP, by default with a
fixed Γ = 2 and Δχ² = 9 (one-sided Gaussian confidence 0.99865).  See
:doc:`upperlimits` for the bound-vs-sensitivity semantics.

Each result reports two separate quantities:

* ``observed_upper_bound`` — the conditional profile bound of this
  observation;
* ``detection_sensitivity`` — the 90% detection sensitivity from
  blank-control samples with fixed-position signal injection.  It is
  only estimated when ``--sensitivity-controls`` points at a
  blank-sky control table from an inactive source region processed with
  consistent quality cuts; otherwise it is reported as
  ``calibration_status=unavailable`` rather than fabricated.

See also
~~~~~~~~

* :doc:`upperlimits` — statistical semantics shared with the GBM
  pipeline.
* :doc:`bxa_fitting` — Bayesian fitting of the prepared survey spectra.
* :doc:`../api` — auto-generated API reference for
  :mod:`jinwu.swift.bat` and :mod:`jinwu.swift.bat.survey`.
