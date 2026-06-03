
Light-curve Fitting
===================

.. warning::

   This page is a work in progress.  See :mod:`jinwu.core.fit` for the
   complete API.

JinWu's light-curve fitting module supports:

- Built-in models: power-law, broken power-law, exponential decay, Gaussian
- Custom expression via string input
- Astropy-based fitting backends (Levenberg-Marquardt, TRFLSQP)
- Unified interface for ``LightcurveData`` and ``LightcurveDataset``

Example
~~~~~~~

.. code-block:: python

    from jinwu.core.data import LightcurveData
    from jinwu.core.fit import LightcurveFitter

    lc = LightcurveData(
        time=[0.1, 0.2, 0.5, 1.0, 2.0],
        flux=[10.0, 8.5, 6.2, 3.1, 1.5],
        flux_err=[0.5, 0.4, 0.3, 0.2, 0.1],
    )

    fitter = LightcurveFitter(lc, model="broken_powerlaw")
    result = fitter.fit()
    print(f"Break time: {result.params['t_break']:.2f} s")
