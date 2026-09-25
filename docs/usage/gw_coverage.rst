GW localization and coverage plots
==================================

The optional ``jinwu-gw`` distribution creates a reproducible static report
from an LVK notice or a local HEALPix map.  It uses the GBM position-history
geometry already provided by ``jinwu-fermi`` and accepts time-tagged EP/BAT
footprints as a JSON layer file.

.. code-block:: console

   jinwu-gw plot --notice alert.json --gbm-cache ~/.cache/jinwu-gw/gbm \
       --layers layers.json --output results/

   jinwu-gw plot --event S251014cn --output results/

   jinwu-gw plot --event https://gracedb.ligo.org/superevents/S251014cn/view/ \
       --notice-version S251014cn-update.json,0 --output results/

   jinwu-gw plot --skymap bayestar.multiorder.fits \
       --time 2024-04-22T00:00:00Z --output results/

For a GraceDB event, the newest public JSON notice is selected by its
``time_created`` and its own embedded map is used.  ``--notice-version`` locks
a historical ``filename,version``.  A standard retraction with ``event: null``
produces only a provenance report; it never reuses an older map or coverage.

Each resolved source fingerprint is archived under
``results/<event-id>/runs/<run-id>/``.  ``current.json`` is atomically updated
only for an active event; a retraction removes it and writes ``RETRACTED.json``.
The report always contains ``sky_map.png/.pdf``,
``gbm_diagnostic.png/.pdf`` (the diagnostic is labelled ``unknown`` when no
POSHIST is available), and ``coverage.json/.ecsv``.  The JSON includes source
checksums, 50/90% credible-region areas and actual enclosed probabilities,
per-instrument unions, GBM/EP intersections, and MOC-order convergence.  A
GBM result marked ``predicted_30_orbit`` is a geometric visibility estimate
based on a historical POSHIST and is not a validated TTE exposure or spectral
result.  The spectral-analysis field is explicitly ``not_run`` in this stage.
When ``--gbm-cache`` is omitted, the CLI uses ``$GBM_POSHIST_DIR`` or
``~/.cache/jinwu-gw/gbm``; add ``--no-download`` for an offline run.

The optional layer JSON has this shape::

   {"layers": [
     {"name": "WXT", "kind": "circle", "ra_deg": 120.0,
      "dec_deg": 30.0, "radius_deg": 5.0,
      "time": "2024-04-22T00:00:00Z", "source": "local WXT product"}
   ]}
