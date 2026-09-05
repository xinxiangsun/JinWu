=============================================================
JinWu Documentation
=============================================================

**Joint Inference for high-energy transient light-curve & spectral analysis
with Unifying physical modeling.**

JinWu (金乌) is a comprehensive Python toolkit for X-ray and gamma-ray
astrophysics, bringing together spectral and temporal analysis with
unified physical modeling.  Named after the mythical three-legged
golden crow dwelling in the sun — a fitting symbol for a package that
shines light on the most energetic transients in the Universe.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api
   usage/lightcurve
   usage/spectral
   usage/bxa_fitting
   usage/upperlimits
   usage/swift_grb
   usage/bat_survey
   usage/fermi_gbm
   usage/nhtot
   RedshiftExtrapolator
   changelog

----

Key Features
============

* **OGIP FITS I/O** — Read and write ARF, RMF, PHA, lightcurve, and event files
* **Lightcurve & Spectral Analysis** — Background modeling, trigger evaluation, XSPEC-inspired components
* **Multi-mission Support** — Einstein Probe (EP), Fermi/GBM, Swift/BAT, and more
* **Resumable Instrument Pipelines** — Registered per-target pipelines for EP/WXT pointing,
  Fermi/GBM continuous data, Swift GRB and BAT survey analysis (see the usage pages below)
* **Bayesian Spectral Fitting** — BXA/UltraNest nested sampling alongside MLE and MCMC chains
* **Unified Physical Modeling** — Consistent framework across bands and messengers
* **Pure-Python ftools** — HEASOFT-compatible tools written entirely in Python
* **Upper Limit Computation** — Bayesian & frequentist upper limits for faint transients

Installation
============

JinWu's spectral fitting and XSPEC integration depend on
**HEASoft/PyXspec** and XSPEC model data.  We recommend installing
these via the HEASARC conda channel, though any working HEASoft
installation is fine.

.. code-block:: bash

   # 1. Install HEASoft + XSPEC + model data via conda
   conda create -n hea -c https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/conda \\
       heasoft xspec xspec-data
   conda activate hea

   # 2. Install JinWu
   pip install jinwu

Optional extras pull in heavier dependencies only when needed:

.. code-block:: bash

   pip install "jinwu[bxa]"          # BXA + UltraNest Bayesian spectral fitting
   pip install jinwu-ep              # Einstein Probe (WXT) support
   pip install "jinwu-fermi[rsp]"    # Fermi/GBM + pure-Python response generation
   pip install "jinwu-swift[ukssdc]" # Swift/BAT + UKSSDC catalog & Burst Analyser
   pip install "jinwu-swift[survey]" # + BatAnalysis for the BAT survey pipeline

If you already have HEASoft installed outside conda (e.g. from source),
JinWu can still use it — call :class:`jinwu.core.heasoft.HeasoftEnvManager`
to initialize the environment:

.. code-block:: python

   from jinwu.core.heasoft import HeasoftEnvManager
   mgr = HeasoftEnvManager()      # auto-detects HEADAS from env / rc files
   mgr.init_heasoft()

For development:

.. code-block:: bash

   conda activate hea
   git clone https://github.com/xinxiangsun/jinwu
   cd jinwu
   # Monorepo: install distributions from packages/ (the repo root is not installable)
   pip install -e "packages/jinwu[docs]"   # core + docs build dependencies
   pip install -e packages/jinwu-ep        # instrument packages referenced by the API docs
   pip install -e packages/jinwu-swift
   pip install -e packages/jinwu-fermi

.. note::

   The ``HeasoftEnvManager`` also works in Jupyter notebooks —
   use :meth:`HeasoftEnvManager.init_heasoft_in_notebook`.

Quick Start
===========

.. code-block:: python

   import jinwu.core as jw

   # Read OGIP FITS files
   pha = jw.read_pha("source.pha")
   lc  = jw.read_lc("lightcurve.fits")

   # Work with energy bands (emin/emax each carry their own unit string)
   band = jw.EnergyBand(emin=0.3, emin_unit="keV", emax=10.0, emax_unit="keV")

   # General net data computation
   net = jw.netdata(src_lc, bkg_lc)


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
