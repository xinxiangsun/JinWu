"""Command-line entry point for GBM subthreshold searches and calibration."""
import argparse
import logging
import sys
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord

from . import GBMTargetedSearchInput, GBMTargetedSearchConfig, run_targeted_search, calibrate_targeted_search


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("search", "calibrate"))
    parser.add_argument("target_id")
    parser.add_argument("--time", required=True, help="external trigger UTC ISO time")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--templates", required=True, type=Path, help="directory containing direct/ and atmo_* /")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", nargs=2, type=float, default=(-30, 30), metavar=("START_S", "STOP_S"))
    parser.add_argument("--min-duration", type=float, default=.064)
    parser.add_argument("--max-duration", type=float, default=8.192)
    parser.add_argument("--min-step", type=float, default=.064)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--background-context", type=float, default=500)
    parser.add_argument("--background-window", type=float, default=125)
    parser.add_argument("--min-score", type=float, default=5)
    parser.add_argument("--position", nargs=2, type=float, metavar=("RA_DEG", "DEC_DEG"))
    parser.add_argument("--skymap", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--off-times", type=Path, help="one UTC ISO time per line; calibrate command")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--until", choices=("preflight", "data", "background", "search", "report"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        job = GBMTargetedSearchInput(target_id=args.target_id, root=args.root, output_root=args.output,
              trigger_time=args.time, template_root=args.templates,
              position=None if args.position is None else SkyCoord(*args.position, unit="deg"),
              skymap=args.skymap, calibration=args.calibration, download=args.download)
        config = GBMTargetedSearchConfig(search_interval=args.interval * u.s, min_duration=args.min_duration * u.s,
                 max_duration=args.max_duration * u.s, min_step=args.min_step * u.s,
                 num_steps=args.num_steps, background_context=args.background_context * u.s,
                 background_window=args.background_window * u.s, min_score=args.min_score)
        if args.command == "calibrate":
            if args.off_times is None:
                parser.error("calibrate requires --off-times")
            times = [line.strip() for line in args.off_times.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
            calibrate_targeted_search(job, times, config=config)
            print(args.output / "calibration.json")
            return 0
        result = run_targeted_search(job, config=config, until=args.until, resume=not args.no_resume)
        print(result.products["report"])
        return 2 if result.science_status == "needs_review" else 0
    except (ValueError, OSError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
