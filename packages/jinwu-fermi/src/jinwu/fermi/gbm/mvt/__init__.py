"""Frozen paper Haar MVT algorithm and GBM TTE workflow.

References: Golkhou & Butler (2014), arXiv:1403.4254; Golkhou et al.
(2015), arXiv:1501.05948; Bala et al. (2026), arXiv:2512.16204.
"""
from .models import GBMMVTConfig, GBMMVTInput, GBMMVTResult, MVTResult
from .pipeline import GBMMVTPipeline, run_gbm_mvt

__all__ = ["GBMMVTConfig", "GBMMVTInput", "GBMMVTResult", "MVTResult",
           "compute_mvt", "classify_mvt", "GBMMVTPipeline", "run_gbm_mvt"]


def __getattr__(name):
    if name in {"compute_mvt", "classify_mvt"}:
        from . import engine
        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
