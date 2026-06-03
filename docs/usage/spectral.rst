
Spectral Fitting
================

.. warning::

   This page is a work in progress.  See the module docstrings in
   :mod:`jinwu.core.fit` and :mod:`jinwu.core.upperlimit` for detailed
   API documentation.

JinWu provides a Pythonic wrapper around XSPEC/PyXspec for spectral fitting,
supporting:

- Standard models (power-law, absorbed power-law, blackbody, etc.)
- Bayesian MCMC chain analysis with PyMultiNest / emcee
- Model comparison with Bayes factors and AIC
- Upper limit computation (Feldman-Cousins, Bayesian, inverse Li-Ma)

Basic Fit
~~~~~~~~~

.. code-block:: python

    from jinwu.core.fit import run_xspec_fit

    result = run_xspec_fit(
        pha="spectrum.pi",
        model="tbabs*ztbabs*powerlaw",
        nh_gal=0.05,
        z=2.0,
    )

For batch processing multiple GRBs, see the EP260119a analysis workflow in the
project repository.

Chain Analysis
~~~~~~~~~~~~~~

.. code-block:: python

    from jinwu.core.fit import run_xspec_chain, XspecChainResult
    
    chain = run_xspec_chain(
        pha="spectrum.pi",
        model="tbabs*ztbabs*powerlaw",
        nh_gal=0.05,
        z=2.0,
        n_live=600,
    )
    chain.summary()
