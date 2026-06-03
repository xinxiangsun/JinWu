
JinWu Documentation
====================

**JinWu** (金乌) is a comprehensive Python package for X-ray and gamma-ray
astrophysics, combining spectral and temporal analysis with unified physical
modeling.  It provides a modular, extensible framework for processing data from
missions such as *Swift*, *Fermi*/GBM, and the *Einstein Probe* (EP).

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   usage/spectral
   usage/lightcurve
   usage/upperlimits
   usage/nhtot

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   GitHub <https://github.com/xinxiangsun/jinwu>

----

Key Features
------------

- **OGIP FITS I/O** — read/write ARF, RMF, PHA, lightcurves, and event files
  with full validation.
- **Spectral fitting** — wrapper around XSPEC/PyXspec with Bayesian chain
  analysis, model comparison, and upper limit computation.
- **Light-curve fitting** — built-in models (power-law, broken power-law,
  exponential, Gaussian) with custom expression support.
- **Upper limits** — Feldman-Cousins, Bayesian, and inverse Li-Ma methods,
  with joint multi-observation significance.
- **Galactic NH** — the ``nhtot`` tool wrapping Willingale et al. (2013)
  for NHI + H₂ column densities.
- **Multi-mission** — dedicated readers for Swift/BAT, Swift/XRT, Fermi/GBM,
  and Einstein Probe data products.

Installation
------------

.. code-block:: bash

   pip install jinwu

Or from source:

.. code-block:: bash

   git clone https://github.com/xinxiangsun/jinwu.git
   cd jinwu
   pip install -e .

Quick Example
-------------

Get the Galactic hydrogen column density for a sky position:

.. code-block:: python

   >>> from jinwu.core.utils import nhtot
   >>> result = nhtot(159.386, 56.171)  # RA, Dec in degrees
   >>> print(f"N_H,tot = {result['nhtot_weighted']:.2e} cm⁻²")
   N_H,tot = 5.26e+19 cm⁻²

Fit a light curve with a broken power-law:

.. code-block:: python

   >>> from jinwu.core.data import LightcurveData
   >>> from jinwu.core.fit import LightcurveFitter
   >>> ...

License
-------

JinWu is released under the `GNU General Public License v3.0 or later
<https://www.gnu.org/licenses/gpl-3.0.html>`_.

Cite
----

If you use JinWu in your research, please cite the repository and the
underlying methods (Willingale+2013, Li & Ma 1983, Feldman & Cousins 1998, etc.).

