"""Command-line interface for the first-stage GW coverage report."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from jinwu.core.config import ExecutionConfig

from .pipeline import GWConfig, GWPipelineInput, run_gw_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jinwu-gw", description="Plot a GW localization and high-energy coverage")
    sub = parser.add_subparsers(dest="command", required=True)
    plot = sub.add_parser("plot", help="create a static sky-coverage report")
    source = plot.add_mutually_exclusive_group(required=True)
    source.add_argument("--notice", type=Path, help="local LVK JSON notice")
    source.add_argument("--event", help="public GraceDB superevent ID or event-page URL")
    plot.add_argument("--notice-version", help="lock GraceDB notice as filename,version")
    source.add_argument("--skymap", type=str, help="local path or URL to a HEALPix/multi-order FITS map")
    plot.add_argument("--time", help="event time (required with --skymap)")
    plot.add_argument("--at", help="display/query time; defaults to event time")
    plot.add_argument("--layers", type=Path, help="JSON EP/BAT MOC, polygon, or circle layer list")
    plot.add_argument("--gbm-cache", type=Path, help="GBM daily cache directory")
    plot.add_argument("--gbm-mode", choices=("auto", "observed", "predicted", "none"), default="auto")
    plot.add_argument("--no-download", action="store_true", help="do not fetch missing GBM POSHIST files")
    plot.add_argument("--output", type=Path, required=True, help="output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "plot":
        return 2
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_id = args.event or (args.notice.stem if args.notice else "local")
    # Keep the CLI one-command path useful while allowing deployments to
    # point at the existing jinwu-fermi cache.  Programmatic callers can pass
    # ``gbm_cache=None`` explicitly to record an unknown GBM result without a
    # network attempt.
    gbm_cache = args.gbm_cache or Path(
        os.environ.get("GBM_POSHIST_DIR", "~/.cache/jinwu-gw/gbm")
    ).expanduser()
    input_data = GWPipelineInput(
        target_id=target_id,
        root=output,
        output_root=output,
        notice=args.notice,
        event=args.event,
        notice_version=args.notice_version,
        skymap=args.skymap,
        time=args.time,
        at=args.at,
        layers=args.layers,
        gbm_cache=gbm_cache,
        gbm_mode=args.gbm_mode,
        no_download=args.no_download,
    )
    config = GWConfig(execution=ExecutionConfig(workspace=output, resume=False))
    result = run_gw_pipeline(input_data, config=config)
    print(f"GW report written to {output}")
    print(f"event={result.event.superevent_id} skymap={result.skymap_source}")
    return 0
