"""Compare frozen MVT proxy and untouched gv_significance PG on identical bins.

Background b/sigma are expected counts/error, not counts/s. This is a local
profile-likelihood diagnostic, without time/bin/detector trials calibration.
The legacy z_bi dependency is irrelevant to PG and is not imported here.
"""
import argparse
import ast
from pathlib import Path
import json

import numpy as np
from jinwu.core.products import write_json, sha256_file
from jinwu.fermi.gbm.mvt.data import bin_events


def load_pg_reference(root):
    root = Path(root)
    namespace = {"np": np, "sqrt": np.sqrt, "squeeze": np.squeeze}
    # Execute exact original function bodies, avoiding only unused legacy
    # absolute imports (z_bi requires the unavailable ncephes extension).
    for filename, names in (("xlogy.py", ("xlogy", "xlogyv")),
                            ("size_one_or_n.py", ("size_one_or_n",)),
                            ("poisson_gaussian.py", ("significance",))):
        tree = ast.parse((root / filename).read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(root / filename), "exec"), namespace)
    # xlogy scalar helper needs math.log; vector helper and PG use np.log.
    import math
    namespace["log"] = math.log
    return namespace["significance"]


def compare_report(report_path, pg):
    summary = json.loads(Path(report_path).read_text())
    with np.load(Path(report_path).parent / "prepared_events.npz", allow_pickle=False) as data:
        events = np.concatenate([data[d] for d in summary["selected_detectors"]])
    records = []
    for row in summary["resolutions"]:
        quantiles = row["resampling"]["percentiles_s"]
        if quantiles is None:
            continue
        dt = quantiles[1]
        counts, edges = bin_events(events, summary["source_interval_s"], dt)
        index = int(np.argmax(counts))
        n = int(counts[index])
        b = sum(summary["background"][d]["source_average_rate_cps"] for d in summary["selected_detectors"]) * dt
        sigma = np.sqrt(sum(summary["background"][d]["source_average_rate_error_cps"] ** 2 for d in summary["selected_detectors"])) * dt
        center = (edges[index] + edges[index + 1]) / 2
        local_b, local_variance = 0., 0.
        for d in summary["selected_detectors"]:
            model = summary["background"][d]
            # Bin-averaged polynomial basis matches GDT Polynomial evaluation.
            a, z = edges[index:index + 2]
            basis = np.array([(z ** (j + 1) - a ** (j + 1)) / ((j + 1) * dt)
                              for j in range(model["order"] + 1)])
            coefficients = np.asarray(model["coefficients"])[:, 0]
            covariance = np.asarray(model["covariance"])[:, :, 0]
            local_b += float(basis @ coefficients) * dt
            local_variance += float(basis @ covariance @ basis) * dt ** 2
        local_sigma = np.sqrt(local_variance)
        records.append({"input_report": str(Path(report_path).resolve()), "mvt_s": dt,
            "peak_interval_s": edges[index:index + 2].tolist(), "observed_total_counts": n,
            "mean_expected_background_counts": b, "mean_background_error_counts": sigma,
            "frozen_mvt_snr": float(n / np.sqrt(b)), "gv_pg_z_same_mean_background": float(pg(n, b, sigma)),
            "net_gaussian_approx": float((n - b) / np.sqrt(b + sigma * sigma)),
            "local_expected_background_counts": local_b, "local_background_error_counts": float(local_sigma),
            "gv_pg_z_local_polynomial": float(pg(n, local_b, local_sigma)),
            "scope": "same total-peak bin, nominal exposure; Gaussian fitted covariance; no trials correction"})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-root", type=Path, default=Path("external_sources/gv_significance-master/gv_significance"))
    parser.add_argument("--report", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pg = load_pg_reference(args.pg_root)
    examples = [{"n": n, "b": b, "sigma_b": sigma, "pg_z": float(pg(n, b, sigma)),
                 "frozen_mvt_snr": float(n / np.sqrt(b))}
                for n, b, sigma in ((100, 90, 2.4), (90, 90, 2.4), (120, 80, 5.3))]
    evidence = {"pg_source": str(args.pg_root.resolve()),
                "pg_source_sha256": sha256_file(args.pg_root / "poisson_gaussian.py"),
                "execution_adapter": "original PG/xlogyv/size_one_or_n function bodies, without unused z_bi/ncephes imports",
                "examples": examples, "real_data": [row for report in args.report for row in compare_report(report, pg)]}
    write_json(args.output, evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
