"""Compare frozen upstream source with migrated MVT, including real products.

Run in hea. Upstream source and real reports are explicit read-only inputs;
the source is not imported by production code. No network or result overwrite.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import warnings

import astropy.units as u
import numpy as np
from jinwu.core.products import write_json, sha256_file
from jinwu.fermi.gbm.mvt import compute_mvt, GBMMVTConfig


def original(root):
    modules = {}
    for name in ("haar_denoise", "haar_nondec_regular_err_wt", "haar_power_mod"):
        spec = importlib.util.spec_from_file_location(name, Path(root) / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        with warnings.catch_warnings():
            spec.loader.exec_module(module)
        modules[name] = module
    return modules


class RemoveAdapter(ast.NodeTransformer):
    def visit_If(self, node):
        # Remove only the two added diagnostic blocks and lazy plot import.
        if any(isinstance(n, ast.Name) and n.id == "_diagnostics" for n in ast.walk(node.test)):
            return None
        if len(node.body) == 1 and isinstance(node.body[0], ast.ImportFrom) and node.body[0].module == "pylab":
            return None
        return self.generic_visit(node)


def normalized(path, function):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    if function == "haar_power_mod":
        while node.args.args[-1].arg.startswith("_"):
            node.args.args.pop(); node.args.defaults.pop()
        node = RemoveAdapter().visit(node)
    return ast.dump(node, include_attributes=False)


def reference_with_intermediates(function, counts, errors, width):
    state = {}
    def profile(frame, event, arg):
        if event == "return" and frame.f_code is function.__code__:
            for name in ("wt", "tau", "pspec", "pspec0", "dpspec", "g2", "dta", "dta1"):
                if name in frame.f_locals:
                    state[name] = frame.f_locals[name].copy()
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = function(counts.copy(), errors.copy(), min_dt=width, doplot=False, verbose=False)
    finally:
        sys.setprofile(previous)
    return np.asarray(raw), state


def compare(root, counts, errors, width, name):
    ref = original(root)["haar_power_mod"].haar_power_mod
    raw, state = reference_with_intermediates(ref, counts, errors, width)
    port = compute_mvt(counts, errors, width * u.s)
    np.testing.assert_array_equal(raw, port.raw_values)
    aliases = {"wt": "weights", "tau": "tau", "pspec": "power", "pspec0": "noise",
               "dpspec": "power_error", "g2": "significant", "dta": "scale_start", "dta1": "scale_stop"}
    for old, new in aliases.items():
        np.testing.assert_array_equal(state[old], port.diagnostics[new])
    return {"case": name, "bins": len(counts), "bin_width_s": width, "raw_values": raw.tolist(),
            "estimator_status": port.estimator_status, "intermediates_exact": list(aliases.values()),
            "max_absolute_final_difference": 0.}


def sample_chunk(args):
    root, lc_path, width, entropy, records = args
    ref = original(root)["haar_power_mod"].haar_power_mod
    with np.load(lc_path, allow_pickle=False) as lc:
        counts = lc["counts"]
    n = 0
    for i, record in records:
        draw = np.random.default_rng([*entropy, i]).poisson(counts)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw = ref(draw.copy(), np.sqrt(draw), min_dt=width, doplot=False, verbose=False)
        except (ValueError, IndexError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
            assert record["estimator_status"] == "failed"
        else:
            # JSON stores nonfinite elements as null; normalize symmetrically.
            normalized_raw = [float(x) if np.isfinite(x) else None for x in raw]
            assert normalized_raw == record["raw_values"], (i, raw, record)
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--report", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--samples-per-resolution", nargs="+", type=int,
                        help="Explicit bounded comparisons per resolution, e.g. 300 5")
    args = parser.parse_args()
    from jinwu.fermi.gbm.mvt import _vendor
    vendor = Path(_vendor.__file__).parent
    evidence = {"upstream_commit": GBMMVTConfig().to_dict()["upstream_commit"], "ast_equality": {},
                "upstream_source_sha256": {}, "cases": [], "real_resampling": []}
    for file, func in (("haar_denoise.py", "haar_denoise"),
                       ("haar_nondec_regular_err_wt.py", "haar_nondec"), ("haar_power_mod.py", "haar_power_mod")):
        assert normalized(args.upstream / file, func) == normalized(vendor / file, func), file
        evidence["ast_equality"][func] = True
        evidence["upstream_source_sha256"][file] = sha256_file(args.upstream / file)
    sample = np.loadtxt(args.upstream / "test_mvt_grb_lc.txt")
    evidence["cases"].append(compare(args.upstream, sample[:, 2], sample[:, 3], .0001, "upstream provided light curve"))
    for report_path in args.report:
        report = json.loads(report_path.read_text())
        for resolution_index, row in enumerate(report["resolutions"]):
            with np.load(row["products"]["lightcurve"], allow_pickle=False) as archive:
                counts = archive["counts"]
            width = row["bin_width_s"]
            evidence["cases"].append(compare(args.upstream, counts, np.sqrt(counts), width, f"{report_path.parent.name} observed"))
            saved = json.loads(Path(row["products"]["resamples"]).read_text())
            entropy = saved["random_stream"]["entropy"][:-1]
            limit = args.max_samples if args.samples_per_resolution is None else args.samples_per_resolution[resolution_index]
            records = list(enumerate(saved["samples"]))[:limit]
            chunks = [(str(args.upstream), row["products"]["lightcurve"], width, entropy, records[i:i + 10])
                      for i in range(0, len(records), 10)]
            if args.workers == 1:
                total = sum(sample_chunk(c) for c in chunks)
            else:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    total = sum(pool.map(sample_chunk, chunks))
            valid = [r for r in saved["samples"] if r["raw_values"] is not None and
                     r["estimator_status"] == "measurement" and round(r["raw_values"][3] * 1000, 3) > 0]
            q = (np.percentile([round(r["raw_values"][2] * 1000, 3) for r in valid], [16, 50, 84]) / 1000).tolist() if len(valid) >= 2 else None
            if q is None:
                assert row["resampling"]["percentiles_s"] is None
            else:
                np.testing.assert_allclose(q, row["resampling"]["percentiles_s"], rtol=0, atol=1e-15)
            evidence["real_resampling"].append({"report": str(report_path.resolve()), "width_s": width,
                    "original_vs_saved_samples_exact": total, "total_samples": len(saved["samples"]),
                    "percentiles_match_frozen_wrapper": True, "percentiles_s": q})
            print(f"Validated {report_path.parent.name} {width}s: {total} exact samples", flush=True)
    write_json(args.output, evidence)
    print(args.output)


if __name__ == "__main__":
    main()
