"""EP/WXT normal-pointing demonstration with an explicit region-review pause.

Run in the ``hea`` environment. Set JINWU_WXT_OBS_ROOT for another observation.
The script never writes into the input observation directory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from jinwu.core.config import instrument
from jinwu.ep.wxt import WXTPointingInput, WXTPointingPipeline


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATION = REPO_ROOT / "test/EP260809adata/EP260809a/06800001692_32"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and run one EP/WXT pointing observation")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("JINWU_WXT_OBS_ROOT", DEFAULT_OBSERVATION)))
    parser.add_argument("--output-base", type=Path, default=Path(os.environ.get("JINWU_EXAMPLE_OUTPUT_ROOT", REPO_ROOT / "examples/_outputs")))
    parser.add_argument("--ra", type=float, default=328.573, help="source right ascension in degrees")
    parser.add_argument("--dec", type=float, default=17.607, help="source declination in degrees")
    parser.add_argument("--source-id", default="s1")
    parser.add_argument("--target-id", default="EP260809a")
    parser.add_argument("--approve-regions", action="store_true", help="offer an interactive approval prompt after QC")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"WXT observation directory does not exist: {root}")
    output = args.output_base.expanduser().resolve() / datetime.now(timezone.utc).strftime("wxt_%Y%m%dT%H%M%S%fZ")
    inp = WXTPointingInput(
        target_id=args.target_id, root=root, output_root=output,
        source_id=args.source_id, ra_deg=args.ra, dec_deg=args.dec,
        auto_approve_regions=False,
    )
    config = instrument("WXT")
    config.reporting = replace(config.reporting, print_summary=False)
    pipe = WXTPointingPipeline(inp, config=config)

    preview = pipe.run(until="exposure_arm_qc", resume=False)
    print("Status / 状态:", preview.status)
    print("Workspace / 产物目录:", preview.workspace)
    region_dir = preview.workspace / "regions"
    regions = json.loads((region_dir / "regions.json").read_text(encoding="utf-8"))
    qc = json.loads((region_dir / "exposure_qc.json").read_text(encoding="utf-8"))
    for label in ("source", "background", "background_effective", "arm"):
        print(f"{label} region:", regions.get(label))
    print("Exposure QC / 曝光诊断:", json.dumps(qc, ensure_ascii=False, indent=2))
    print("Inspect regions against the observation image and read every QC warning before approval.")

    if not args.approve_regions:
        print("Paused for review. Inspect the saved workspace and use the Notebook for the interactive walkthrough.")
        return 0
    if input("After inspecting the regions, type APPROVE to continue: ").strip() != "APPROVE":
        print("Approval withheld; no downstream analysis was run.")
        return 0
    pipe.approve_regions(note="reviewed interactively during WXT teaching demonstration")
    result = pipe.run(resume=True)
    print("Final status / 最终状态:", result.status)
    print("Alpha / ON-OFF 缩放:", result.alpha)
    print("T90 source PHA:", result.t90_source_pha)
    print("T90 background PHA:", result.t90_background_pha)
    print("Report / 报告:", result.report)
    print(result.summary_text())
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
