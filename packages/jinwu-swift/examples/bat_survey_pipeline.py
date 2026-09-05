"""Runnable single-target Swift/BAT survey example.

Run inside the HEASoft environment after installing the optional survey extra::

    conda run -n hea python examples/bat_survey_pipeline.py \
        --root /data/swift/batdata --obsid 00098092002

The example never enables network access unless --query or --download is
given. Replace the coordinates and UTC window with the target being studied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jinwu.core.config import BATSurvey, SwiftBATSurveyDownloadConfig
from jinwu.swift.bat.survey import BATSurveyInput, BATSurveyPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", default="NGC4253")
    parser.add_argument("--source-name", default="NGC4253")
    parser.add_argument("--ra", type=float, default=183.5625)
    parser.add_argument("--dec", type=float, default=29.8125)
    parser.add_argument("--obsid", action="append", default=[])
    parser.add_argument("--window", nargs=2, action="append", default=[], metavar=("START_UTC", "STOP_UTC"))
    parser.add_argument("--raw-products", type=Path)
    parser.add_argument("--survey-products", type=Path)
    parser.add_argument("--mosaic-products", type=Path)
    parser.add_argument("--query", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--mosaic", action="store_true")
    parser.add_argument("--profile", choices=("default", "lmjagn"), default="default")
    parser.add_argument("--detthresh", type=int)
    parser.add_argument("--detthresh2", type=int)
    parser.add_argument("--min-pcode", type=float)
    parser.add_argument("--network-timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=5.0)
    parser.add_argument("--query-margin", type=float, default=0.0)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--internal-threads", type=int, default=1)
    parser.add_argument("--task-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--until",
        choices=tuple(stage.name for stage in BATSurveyPipeline.stages),
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    job = BATSurveyInput(
        target_id=args.target,
        root=args.root,
        output_root=args.output,
        source_name=args.source_name,
        coord=(args.ra, args.dec),
        obsids=tuple(args.obsid),
        time_windows=tuple(tuple(item) for item in args.window),
        raw_products_dir=args.raw_products,
        survey_products_dir=args.survey_products,
        mosaic_products_dir=args.mosaic_products,
        query=args.query,
        download=args.download,
        mosaic=args.mosaic,
        detthresh=args.detthresh,
        detthresh2=args.detthresh2,
        min_pcode=args.min_pcode,
    )
    download = SwiftBATSurveyDownloadConfig(
        timeout_s=args.network_timeout,
        retries=args.retries,
        retry_wait_s=args.retry_wait,
        query_margin_s=args.query_margin,
    )
    result = BATSurveyPipeline(
        job,
        config=BATSurvey(
            profile=args.profile,
            processes=args.processes,
            internal_threads=args.internal_threads,
            task_timeout_s=args.task_timeout,
            downloads=download,
        ),
    ).run(until=args.until, resume=not args.no_resume)
    print(f"pipeline={result.status.value} science={result.science_status}")
    print(f"report={result.products.get('report', {}).get('report')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
