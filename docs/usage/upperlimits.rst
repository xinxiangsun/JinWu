
Upper Limits
============

.. warning::

   This page is a work in progress.  See :mod:`jinwu.core.upperlimit` for
   the complete API.

JinWu provides a unified upper limit framework:

- ``UpperLimit`` class wrapping chain-based and error-based methods
- Supports multiple confidence levels (68%, 90%, 95%, 99%)
- Feldman-Cousins and Bayesian approaches
- Joint multi-observation significance via Li & Ma (1983)

Example
~~~~~~~

.. code-block:: python

    from jinwu.core.upperlimit import UpperLimit

    ul = UpperLimit(delta_fit_stat=2.706)  # 90% CL
    result = ul.from_chain("chain.fits", param_index=4)
    print(f"90% CL upper limit: {result.limit:.2e}")
