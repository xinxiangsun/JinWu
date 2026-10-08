"""Reproducible Poisson samples with execution-independent stream identities."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing

import astropy.units as u
import numpy as np
from .engine import compute_mvt, summarize_resamples


def _chunk(args):
    counts, width, settings, indices, stream = args
    records = []
    for i in indices:
        seed = [settings.seed, *stream, i]
        draw = np.random.default_rng(seed).poisson(counts)
        result = compute_mvt(draw, bin_width=width * u.s, config=settings, diagnostics=False)
        records.append((i, result))
    return records


def resample_mvt(counts, bin_width, *, config, stream=(0, 0)):
    """Poisson(mean=observed counts), preserving paper conditional percentiles.

    ``stream`` explicitly identifies detector iteration and resolution. Sample
    i uses default_rng([seed, iteration, resolution, i]); changing workers
    preserves every draw and result. Inputs are count arrays and time Quantity.
    """
    settings = replace(config, workers=1)
    n = config.n_resamples
    chunks = [list(range(i, min(i + 10, n))) for i in range(0, n, 10)]
    args = [(counts, float(bin_width.to_value(u.s)), settings, indices, tuple(stream)) for indices in chunks]
    if config.workers == 1:
        records = [record for arg in args for record in _chunk(arg)]
    else:
        with ProcessPoolExecutor(max_workers=config.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            records = [record for chunk in pool.map(_chunk, args) for record in chunk]
    ordered = [r for _, r in sorted(records, key=lambda item: item[0])]
    summary = summarize_resamples(ordered)
    summary["random_stream"] = {"generator": "numpy.default_rng", "entropy": [config.seed, *stream, "sample_index"]}
    return ordered, summary
