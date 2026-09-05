"""Swift BAT+XRT GRB sample pipeline.

Ported from the ``swift_highz_grb`` research workflow: catalog merging with
redshift priority selection, Burst Analyser ingestion with BadBin handling,
per-GRB flux lightcurve products, strict prompt-stage classification,
Bayesian-block time-resolved spectroscopy, and XSPEC fitting through
``jinwu.core.fit.fit_prepared``.

The concrete pipeline is registered as ``"swift.grb"`` via
``jinwu.instruments`` entry points.
"""

from __future__ import annotations

from .pipeline import (
    GRBRecord,
    RedshiftCandidate,
    SwiftBurstAnalyserFetcher,
    SwiftGRBInput,
    SwiftGRBPipeline,
    SwiftGRBResult,
    build_grb_record,
    merge_low_snr_segments,
    normalize_grb_name,
    parse_bat_burst_durations,
    parse_redshift_candidates,
    preferred_redshift,
    prompt_classification,
    read_burst_analyser_dat,
    read_badbin_table,
    segment_from_payload,
)

__all__ = [
    "GRBRecord",
    "RedshiftCandidate",
    "SwiftBurstAnalyserFetcher",
    "SwiftGRBInput",
    "SwiftGRBPipeline",
    "SwiftGRBResult",
    "build_grb_record",
    "merge_low_snr_segments",
    "normalize_grb_name",
    "parse_bat_burst_durations",
    "parse_redshift_candidates",
    "preferred_redshift",
    "prompt_classification",
    "read_burst_analyser_dat",
    "read_badbin_table",
    "segment_from_payload",
]
