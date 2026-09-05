
Bayesian Spectral Fitting (BXA)
===============================

JinWu can infer X-ray spectra either by maximum likelihood (MLE), by an
XSPEC MCMC chain, or by **Bayesian nested sampling** through
`BXA <https://bxagroup.github.io/bxa/>`_ (Bayesian X-ray Analysis, built on
UltraNest).  The BXA path lives in :mod:`jinwu.core.bxa_fit` and produces a
log-evidence (``logZ``) per model, enabling rigorous Bayesian model comparison
in addition to posterior sampling.

The whole BXA surface is **lazily imported**: ``import jinwu.core`` never
requires ``bxa`` or HEASoft/PyXspec to be installed.  Dependencies are only
touched when you actually run a ``method="bxa"`` fit.

.. note::

   The BXA path reuses the *exact* XSPEC session-building helpers of
   :func:`jinwu.core.fit.fit_prepared` (spectrum loading, absorption handling,
   group linking, best-fit reporting), so an MLE fit and a BXA fit of the same
   ``model_name`` describe the same likelihood surface.


Environment & Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~

JinWu's BXA fitting needs three things at run time: ``jinwu`` itself, the
``bxa`` extra (``bxa`` + ``ultranest``), and an importable HEASoft/PyXspec.
There is no ``python``/``pip`` on the bare ``PATH``; always activate a conda
environment first:

.. code-block:: bash

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate <env>

    # BXA + UltraNest (xspec is provided by HEASoft, not by pip)
    pip install jinwu[bxa]

Two conda environments are set up for this project.  **Pick by purpose**:

``hea`` -- development & mock unit tests (HEASoft **6.36**)
    The only environment with ``jinwu`` installed *editable* (pointing at the
    workspace source, so code edits take effect immediately) **and** ``pytest``.
    Use it for anything that does not touch a real ``xspec`` session:

    .. code-block:: bash

        conda activate hea
        python -m pytest test/ -q

    ``xspec`` (12.15.1) is present but **not** auto-initialized; initialize it
    manually before any real BXA/XSPEC run:

    .. code-block:: bash

        export HEADAS=$CONDA_PREFIX/heasoft
        source $HEADAS/BUILD_DIR/headas-init.sh

``jiasui`` -- real-machine XSPEC/BXA end-to-end on HEASoft **6.37**
    Activating it auto-initializes ``xspec`` 13.0.0 (no manual ``headas-init``).
    It ships ``bxa``/``ultranest`` but **lacks ``jinwu`` and ``pytest``**, so
    install them first:

    .. code-block:: bash

        conda activate jiasui
        pip install -e packages/jinwu -e packages/jinwu-ep \
            -e packages/jinwu-fermi -e packages/jinwu-swift
        pip install pytest pytest-mock

**Rule of thumb:** mock unit tests / development -> ``hea``.  Real XSPEC/BXA
end-to-end on HEASoft 6.37 -> ``jiasui`` (after the ``pip install -e`` step).
If HEASoft 6.36 is acceptable, ``hea`` with the manual ``headas-init`` is the
least-effort route since ``jinwu`` + ``pytest`` + ``bxa`` + ``xspec`` are all
already present.


``method`` vs ``backend``
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`jinwu.core.FitConfig` carries two orthogonal selectors:

``method`` -- *how* the spectrum is inferred
    One of ``"mle"`` (maximum likelihood), ``"chain"`` (XSPEC MCMC), or
    ``"bxa"`` (UltraNest nested sampling).  This is the switch you change to
    run a Bayesian fit.

``backend`` -- *which likelihood engine* evaluates the model
    Currently only ``"xspec"``.  A native ``jinwu.model`` engine is reserved
    for the future (see `jinwu.model placeholder`_ below).  You normally never
    change this.

Both fields are validated on construction: an unknown ``method`` or ``backend``
raises ``ValueError``.


Choosing the fit method (four ways)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are four levels at which you can select ``method="bxa"``, from the most
local to the most persistent.  All of them funnel through
:func:`jinwu.core.fit.fit_spectral`, whose **effective method** resolves with
the following precedence (highest first):

1. Explicit keyword arguments passed at the fit call site.
2. ``fit_spectral(method=...)`` / ``fit_spectral(settings=...)`` arguments.
3. The pipeline's per-instrument ``config.fitting.method`` (explicitly
   destructured and passed to the fit call).
4. The process-wide global settings (:func:`jinwu.core.set_fit_method` /
   :func:`jinwu.core.set_fit_settings`).
5. The :class:`jinwu.core.FitConfig` dataclass default (``method="mle"``).

.. note::

   A pipeline cannot tell an *explicitly chosen* ``method="mle"`` apart from
   the *default* ``"mle"``.  So when an instrument's ``config.fitting.method``
   is left at its default ``"mle"`` (never explicitly set), a process-wide
   ``set_fit_method("bxa")`` **will** drive that pipeline's fit stage to BXA --
   the global setting (level 4) is the only method signal available.  Pin
   ``config.fitting.method="mle"`` explicitly on the instrument config if you
   want that pipeline to stay MLE regardless of the global setting.  (The
   pipeline freezes the *effective* method into its configuration fingerprint;
   this precedence is the intended behaviour.)

**(1) Single call** (highest priority, temporary) -- affects only this call:

.. code-block:: python

    from jinwu.core import fit_spectral

    result = fit_spectral(prepared, outdir="fit/bxa", method="bxa")
    # -> a BXAFitResult

**(2) Process-wide** (persistent) -- set once at the top of a script/notebook;
every subsequent bare call defaults to BXA until you reset it:

.. code-block:: python

    from jinwu.core import set_fit_method, set_fit_settings, reset_fit_settings

    set_fit_method("bxa")                       # shortcut for method only
    # ...or update several basic settings at once:
    set_fit_settings(method="bxa", statistic="cstat")

    result = fit_spectral(prepared, outdir="fit/bxa")   # routes to BXA

    reset_fit_settings()                        # restore packaged defaults

**(3) Context manager** (locally temporary) -- overrides inside the ``with``
block and restores the previous settings on exit, even if the block raises:

.. code-block:: python

    from jinwu.core import fit_settings, fit_spectral

    with fit_settings(method="bxa"):
        result = fit_spectral(prepared, outdir="fit/bxa")
    # settings are back to whatever they were before the block

**(4) Pipeline level** -- pass a :class:`jinwu.core.FitConfig` into the
instrument config, or rely on the global default:

.. code-block:: python

    from jinwu.core import WXT, FitConfig

    cfg = WXT(fitting=FitConfig(method="bxa"))
    # cfg.fitting.method == "bxa" -> the pipeline routes its fit stage to BXA

.. note::

   ``model_name`` is deliberately **not** resolved from the global settings:
   the packaged :class:`jinwu.core.FitConfig` default (``"tbabs*powerlaw"``)
   differs from ``fit_prepared``'s own hard-coded default, so pulling it from
   the global singleton would silently change existing behaviour.  Set
   ``model_name`` explicitly per call (or via ``fit_spectral``/pipeline config)
   when you need a non-default model.


Scope of the global settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process-wide settings are **process-level and cross-call-site** -- they are
not scoped to BXA.  Calling ``set_fit_settings(statistic=..., abundance=...,
cross_section=...)`` changes the defaults seen by **every** later
:func:`jinwu.core.fit.fit_prepared`, :func:`jinwu.core.fit.fit`, and
:func:`jinwu.core.fit.fit_xray_models` call in the same process that **omits**
those arguments: the resolved values are silently applied to the XSPEC session
(``Xset.abund``, ``Xset.xsect`` and ``Fit.statMethod``).  A bare
``fit_prepared(...)`` therefore behaves exactly as before *unless* you changed
the global settings.

Two ways to keep a change from leaking across call sites:

* **Isolate it** with the :func:`jinwu.core.fit_settings` context manager,
  which restores the previous settings on exit (even if the block raises).
* **Override at the call site** -- passing ``stat_method=`` / ``abundance=`` /
  ``cross_section=`` explicitly always wins over the global default (see the
  precedence above).

Reserve the persistent :func:`jinwu.core.set_fit_settings` /
:func:`jinwu.core.set_fit_method` for when you intend the whole
process/session to adopt the new defaults, and call
:func:`jinwu.core.reset_fit_settings` to restore the packaged defaults.


Running a BXA fit
~~~~~~~~~~~~~~~~~

Single spectrum / joint spectrum -- :func:`jinwu.core.bxa_fit.fit_prepared_bxa`.
The input contract is the *same* ``PreparedSpectrum`` /
``PreparedJointSpectrum`` produced by ``prepare_spectra`` (see
:doc:`spectral`):

.. code-block:: python

    from jinwu.core.spectrum_prep import prepare_spectra
    from jinwu.core import fit_prepared_bxa, BXAPriorSpec

    catalog = ...                     # built by scan() or by hand
    prepared_catalog = prepare_spectra(catalog, outdir="fit/prepared")
    prepared = prepared_catalog.spectra[0]

    result = fit_prepared_bxa(
        prepared,
        outdir="fit/bxa",
        model_name="tbabs*ztbabs*cflux*powerlaw",
        galactic_nh_1e22=0.05,
        stat_method="cstat",          # must be cash / cstat / pstat
        # --- BXA / UltraNest tuning (BXAConfig contract) ---
        n_live_points=400,            # "publication"; use fewer to screen fast
        evidence_tolerance=0.5,
        speed="safe",
        resume=False,                 # True -> continue from the cached run
        calculate_flux_chain=True,
        flux_erange="2.0 10.0",
    )

    print(result.logz, result.logzerr)     # Bayesian log-evidence
    print(result.summary())                # compact, human-facing surface
    print(result.chain_path)               # <outdir>/chains/chain.fits
    payload = result.to_dict()             # JSON-safe (ndarray -> list)

``fit_prepared_bxa`` creates ``outdir`` and its ``chains/`` sub-directory up
front (BXA's ``outputfiles_basename`` must already exist), runs a
maximum-likelihood warm start, samples the posterior with
``vectorized=False``, and writes ``bxa_summary.json`` plus the standard XSPEC
session products (result JSON / report / ``.xcm`` / transcript).  The
``chains/chain.fits`` it produces can be read back by
:meth:`jinwu.core.upperlimit.UpperLimit.from_chain`.


Declaring priors (``BXAPriorSpec``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every thawed, unlinked parameter is sampled.  By default each gets a **uniform**
prior over its existing soft bounds.  Override individual parameters with
:class:`jinwu.core.bxa_fit.BXAPriorSpec`, keyed by ``"component.parameter"``:

.. code-block:: python

    from jinwu.core import BXAPriorSpec

    priors = {
        # uniform over an explicit range (replaces the parameter's soft bounds)
        "powerlaw.PhoIndex": BXAPriorSpec(kind="uniform", low=0.5, high=4.0),
        # log-uniform (requires low > 0) -- natural for normalization / flux
        "cflux.lg10Flux": BXAPriorSpec(kind="loguniform", low=1e-14, high=1e-6),
        # gaussian with mean / std
        "ztbabs.nH": BXAPriorSpec(kind="gaussian", mean=0.05, std=0.01),
        # custom transform (u in [0,1] -> physical value), optional inverse
        "powerlaw.norm": BXAPriorSpec(
            kind="custom",
            transform=lambda u: 10 ** (u * 4 - 8),
            aftertransform=lambda x: (x + 8) / 4 if x else x,
        ),
    }

    result = fit_prepared_bxa(prepared, outdir="fit/bxa", prior_specs=priors)

Internally :func:`jinwu.core.bxa_fit.resolve_priors` discovers the thawed
parameters exactly as :mod:`jinwu.core.fit` counts them, **unifies each
parameter's soft/hard bounds** so that ``pmin == pbottom`` and ``ptop == pmax``
(a BXA v5 requirement), and builds the transformations.  ``loguniform`` priors
raise if the resolved lower bound is not ``> 0``; ``gaussian`` priors require a
positive ``std``.


Multi-candidate Bayesian comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`jinwu.core.bxa_fit.fit_xray_models_bxa` mirrors
:func:`jinwu.core.fit.fit_xray_models`: it fits a set of model candidates and
ranks them -- but by **descending ``logZ``** instead of AICc.  It returns the
same :class:`jinwu.core.fit.XRayModelComparisonResult`, so downstream pipeline
stages and reports need no changes; only ``selection_metric`` becomes
``"logz"`` and each candidate's :class:`jinwu.core.fit.ModelFitMetrics` carries
``logz`` / ``logzerr`` (built by
:func:`jinwu.core.fit.calculate_bayesian_model_metrics`, which leaves
AIC/AICc/BIC ``None``).

.. code-block:: python

    from jinwu.core import fit_xray_models_bxa

    comparison = fit_xray_models_bxa(
        prepared,
        outdir="fit/bxa_models",
        galactic_nh_1e22=0.05,
        # per-candidate priors: {candidate_key: {parameter: BXAPriorSpec}}
        prior_specs=None,
        # forwarded to fit_prepared_bxa:
        stat_method="cstat",
        n_live_points=400,
    )

    print(comparison.adopted_key)          # highest logZ
    print(comparison.adopted_reason)
    print(comparison.to_dict()["ranking"]) # keys ordered by descending logZ

**logZ vs AICc.**  AICc ranks by goodness-of-fit penalized for the number of
free parameters at the maximum-likelihood point; it is a *frequentist*
information criterion.  ``logZ`` is the log of the Bayesian marginal
likelihood -- the prior-weighted average of the likelihood over the whole
parameter volume -- and automatically applies an Occam penalty to models with
wasted prior volume.  The two can therefore order the same candidates
differently, especially when a more complex model fits marginally better but
spreads its prior thinly.

**Bayes factor.**  To compare two evidences directly, use
:func:`jinwu.core.model_comparison.summarize_bayes_factor`, which takes the
natural-log evidences (and optional errors) and returns the Bayes factor
``Z1/Z0`` with an interpretation:

.. code-block:: python

    from jinwu.core.model_comparison import summarize_bayes_factor

    bf = summarize_bayes_factor(
        log_evidence_h0=result_a.logz,
        log_evidence_h1=result_b.logz,
        log_evidence_error_h0=result_a.logzerr,
        log_evidence_error_h1=result_b.logzerr,
    )
    print(bf.log_bayes_factor, bf.strength)


Pipeline orchestrator
~~~~~~~~~~~~~~~~~~~~~

:func:`jinwu.core.bxa_fit.run_bxa_pipeline` chains prepare -> BXA fit ->
aggregate for a whole scanned dataset.  Pass a ``Catalog``/``Manifest``; each
ready prepared spectrum is fitted with ``fit_prepared_bxa`` and the returned
``dict`` collects per-spectrum ``logz`` / ``logzerr``, product paths and any
failures (also written to ``bxa_pipeline.json``):

.. code-block:: python

    from jinwu.core import run_bxa_pipeline

    summary = run_bxa_pipeline(
        data,                               # Catalog / Manifest from scan()
        outdir="fit/bxa_pipeline",
        group_min=20,
        model_name="tbabs*ztbabs*cflux*powerlaw",
        backend_kwargs={"stat_method": "cstat", "n_live_points": 400},
    )
    print(summary["n_fits"], summary["failures"])


``jinwu.model`` placeholder
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A form-compatible seam exists for a future **native** (non-XSPEC) likelihood
engine: ``JinwuModelBXASolver`` (a ``bxa.xspec.BXASolver`` subclass overriding
``log_likelihood``) and ``fit_prepared_jinwu_model`` (same signature as
``fit_prepared_bxa``).  **Both currently raise ``NotImplementedError``.**  They
are reserved so that a future ``jinwu.model`` backend can be dropped in without
changing call sites, and are deliberately excluded from ``jinwu.core.__all__``
to prevent accidental use.  For now, use the XSPEC-backed path
(``model_name=...``).


Key constraints
~~~~~~~~~~~~~~~

Statistic must be Poisson
    ``Fit.statMethod`` must be one of ``cstat`` / ``cash`` / ``pstat`` (BXA's
    allowed statistics).  Any other value raises ``ValueError`` before sampling
    starts.  ``stat_method`` defaults to the global
    :class:`jinwu.core.FitConfig` ``statistic`` (``"cstat"``) via a ``None``
    sentinel.

No thread / process-shared parallelism
    ``BXASolver.vectorized`` is hard-coded ``False`` and PyXspec is a
    **non-reentrant global singleton** (it holds ``AllData``/``AllModels``
    state).  Therefore thread pools, ``joblib`` threads and
    ``multiprocessing`` sessions that share the XSPEC session are **forbidden**
    -- they corrupt the global state.  ``fit_prepared_bxa`` clears
    ``AllData``/``AllModels`` on entry *and* exit; do not interleave BXA and
    manual XSPEC fitting in the same session.

MPI (optional, advanced)
    The only safe parallel path is **process-level MPI**: UltraNest
    auto-detects ``mpi4py`` and distributes live points across ranks.  Launch
    with ``mpirun -n N python your_script.py``; each rank loads the same
    prepared spectra independently.  The first release defaults to a single
    process.

Resume / caching
    ``resume=True`` passes ``resume='resume'`` to UltraNest, continuing from
    the cached run in ``outputfiles_basename`` (the ``chains/`` directory)
    instead of overwriting it.  The result records ``cache_status``
    (``"resume"`` or ``"overwrite"``).  Keep the ``chains/`` directory between
    runs to benefit from it.

Performance presets
    There is no separate preset object -- tune directly via the ``BXAConfig``
    keyword arguments.  A fast *screening* run uses a low ``n_live_points``
    with an integer ``speed`` (SliceSampler steps); a *publication* run uses
    ``n_live_points >= 400`` with ``speed="safe"``.


See also
~~~~~~~~

* :doc:`spectral` -- prepare/fit flow, MLE fitting and chain analysis.
* :mod:`jinwu.core.bxa_fit` -- BXA nested-sampling implementation.
* :func:`jinwu.core.fit.fit_spectral` -- unified method-routing entry point.
* :class:`jinwu.core.config.FitConfig` and the process-wide settings API
  (:func:`jinwu.core.config.get_fit_settings`,
  :func:`jinwu.core.config.set_fit_settings`,
  :func:`jinwu.core.config.set_fit_method`,
  :func:`jinwu.core.config.fit_settings`).
