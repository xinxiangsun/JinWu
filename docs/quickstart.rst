
Quickstart
==========

Prerequisites
-------------

JinWu requires Python 3.11+ and several standard scientific Python packages
(numpy, scipy, astropy).  Some optional features need additional dependencies:

- **XSPEC / PyXspec** — for spectral fitting (requires HEASoft installation)
- **swiftbat** — for Swift/BAT data processing
- **batanalysis** — for BAT survey data analysis

Basic Usage
-----------

Reading Data
~~~~~~~~~~~~

JinWu can read standard OGIP FITS files and mission-specific data products:

.. code-block:: python

    from jinwu.core.data import read_arf, read_pha, read_lightcurve

    # Read an ARF (auxiliary response file)
    arf = read_arf("path/to/swtmkarf_ex.img")

    # Read a PHA spectrum
    pha = read_pha("path/to/swtpo_ex.pi")

    # Read a light curve
    lc = read_lightcurve("path/to/swtmbrxlc_ex.lc")

Each reader returns a dataclass with validated fields — no need to manually
parse FITS headers.

Galactic NH
~~~~~~~~~~~

The ``nhtot`` function (Willingale et al. 2013) retrieves the total Galactic
hydrogen column density — including both atomic (HI) and molecular (H₂)
components — for use in XSPEC absorption models:

.. code-block:: python

    from jinwu.core.utils import nhtot

    result = nhtot(ra=159.386, dec=56.171)
    print(result["nhtot_weighted"])   # NH,tot in atoms cm⁻²
    print(result["ebv_weighted"])     # E(B-V) in magnitudes

For ``tbabs`` in XSPEC, always use ``nhtot_weighted`` (not HEASoft's ``nh``,
which gives HI only).

Light-curve Fitting
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from jinwu.core.fit import LightcurveFitter
    from jinwu.core.data import LightcurveData

    # Load data
    lc = LightcurveData.from_fits("lightcurve.fits")

    # Fit a broken power-law
    fitter = LightcurveFitter(lc, model="broken_powerlaw")
    result = fitter.fit()
    print(result)

Spectral Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from jinwu.core.fit import run_xspec_fit

    result = run_xspec_fit(
        pha="spectrum.pi",
        model="tbabs*ztbabs*powerlaw",
        nh_gal=0.0313,
        z=4.045,
    )
    print(f"Γ = {result['Gamma']:.2f} ± {result['Gamma_err']:.2f}")
    print(f"NH,intr = {result['NH_intr']:.2e} cm⁻²")

Upper Limits
~~~~~~~~~~~~

.. code-block:: python

    from jinwu.core.upperlimit import UpperLimit

    ul = UpperLimit(delta_fit_stat=2.706)  # 90% CL
    result = ul.from_chain("chain.fits", param_index=4)
    print(f"90% upper limit: {result.limit:.2e}")

Next Steps
----------

- :doc:`usage/spectral` — detailed spectral fitting guide
- :doc:`usage/lightcurve` — light-curve analysis
- :doc:`usage/upperlimits` — upper limit methods
- :doc:`api` — full API reference
