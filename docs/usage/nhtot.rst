
Galactic NH (nhtot)
===================

.. warning::

   This page is a work in progress.  See :func:`jinwu.core.utils.nhtot` for
   the complete API.

The ``nhtot`` function queries the Swift UKSSDC nhtot web service to obtain
the **total** Galactic hydrogen column density using the method of
Willingale et al. (2013, MNRAS, 431, 394).

This includes:
- **NHI** — atomic hydrogen (from 21-cm surveys)
- **NH₂** — molecular hydrogen (estimated from dust reddening E(B-V))
- **NH,tot** = NHI + 2×NH₂

Why nhtot?
~~~~~~~~~~

HEASoft's ``nh`` tool returns NHI **only** (from the HI4PI map).
At low Galactic latitudes (|b| < 20°), molecular hydrogen contributes
significantly, and using NHI-only values in ``tbabs`` (which assumes
20% molecular fraction) systematically underestimates Galactic absorption.

The Swift community (including Valan et al. 2023) uses ``nhtot`` for the
Galactic component of X-ray spectral fitting.

Usage
~~~~~

.. code-block:: python

    from jinwu.core.utils import nhtot

    # Decimal degrees
    result = nhtot(ra=159.386, dec=56.171)
    print(f"N_HI       = {result['nhi_weighted']:.2e} cm⁻²")
    print(f"N_H₂       = {result['nh2_weighted']:.2e} cm⁻²")
    print(f"N_H,tot    = {result['nhtot_weighted']:.2e} cm⁻²")
    print(f"E(B-V)     = {result['ebv_weighted']:.3f} mag")

    # Sexagesimal coordinates also accepted
    result = nhtot("10:37:32.6", "+56:10:15.6")

    # For XSPEC tbabs:
    nh_gal = result["nhtot_weighted"]  # Use this, not nh!

Reference
~~~~~~~~~

Willingale, R., Starling, R. L. C., Beardmore, A. P., Tanvir, N. R., &
O'Brien, P. T. 2013, MNRAS, 431, 394
(`arXiv:1303.0843 <https://arxiv.org/abs/1303.0843>`_)
