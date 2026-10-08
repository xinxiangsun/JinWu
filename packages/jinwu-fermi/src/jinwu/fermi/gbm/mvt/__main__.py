"""python -m jinwu.fermi.gbm.mvt --help."""
import argparse
import json
import astropy.units as u
from .models import GBMMVTInput, GBMMVTConfig
from .pipeline import run_gbm_mvt


def main():
    parser = argparse.ArgumentParser(description="Standalone frozen-paper GBM Haar MVT")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="GBM-MVT")
    parser.add_argument("--trigger-id")
    parser.add_argument("--trigger-time", help="UTC reference; inferred from triggered TTE if omitted")
    parser.add_argument("--tte", action="append", default=[])
    parser.add_argument("--source", nargs=2, type=float, required=True, metavar=("START_S", "STOP_S"))
    parser.add_argument("--background", nargs=2, type=float, action="append", required=True)
    parser.add_argument("--energy-kev", nargs=2, type=float, default=[8., 900.])
    parser.add_argument("--bin-widths-ms", nargs="+", type=float, default=[1., .1, .01])
    parser.add_argument("--detectors", nargs="+", default=["auto"])
    parser.add_argument("--t90-s", type=float)
    parser.add_argument("--resamples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--background-order", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    inputs = GBMMVTInput(target_id=args.target, root=args.root, output_root=args.output,
        trigger_id=args.trigger_id, trigger_time=args.trigger_time, tte_paths=tuple(args.tte),
        source_interval=args.source * u.s, background_intervals=args.background * u.s, download=args.download)
    config = GBMMVTConfig(energy_range=args.energy_kev * u.keV, bin_widths=args.bin_widths_ms * u.ms,
        detectors="auto" if args.detectors == ["auto"] else tuple(args.detectors),
        t90=None if args.t90_s is None else args.t90_s * u.s, n_resamples=args.resamples,
        seed=args.seed, workers=args.workers, background_order=args.background_order)
    result = run_gbm_mvt(inputs, config=config, resume=not args.no_resume)
    print(json.dumps({"science_status": result.science_status, "interpretation": result.summary.get("interpretation"),
                      "report": result.products["report"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
