"""Version-retention and retraction semantics of the GW report workspace.

规范：警报更新保留版本；撤回警报生成撤回状态，不继续把旧图作为当前结果。
Two notice versions of the same event run into one workspace: every run appends
a history entry and ``current.json`` follows the latest active alert; a
retraction removes the pointer and writes ``RETRACTED.json`` while keeping all
history entries.

Run in the ``hea`` environment::

    conda run -n hea python -m pytest test/test_gw_pipeline_archive.py -q -p no:cacheprovider
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from jinwu.gw.pipeline import GWPipelineInput, run_gw_pipeline


def _notice(tmp_path: Path, name: str, *, time_created: str, alert_type: str = "INITIAL") -> Path:
    npix = 12
    probability = np.full(npix, 1.0 / npix)
    columns = fits.ColDefs([fits.Column(name="PROB", format="D", array=probability)])
    table = fits.BinTableHDU.from_columns(columns)
    table.header["NSIDE"] = 1
    table.header["ORDERING"] = "RING"
    fits_path = tmp_path / f"{name}.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(fits_path, overwrite=True)
    notice = {
        "superevent_id": "S240101abc",
        "alert_type": alert_type,
        "time_created": time_created,
        "event": None if alert_type == "RETRACTION" else {
            "time": "2024-01-01T00:00:00Z",
            "skymap": base64.b64encode(fits_path.read_bytes()).decode(),
        },
    }
    notice_path = tmp_path / f"{name}.json"
    notice_path.write_text(json.dumps(notice), encoding="utf-8")
    return notice_path


def _run(output: Path, notice: Path) -> object:
    return run_gw_pipeline(
        GWPipelineInput(
            target_id="S240101abc",
            root=output,
            output_root=output,
            notice=notice,
            gbm_mode="none",
        )
    )


def test_alert_versions_are_retained_and_current_follows_latest(tmp_path: Path):
    output = tmp_path / "workspace"
    notice_v1 = _notice(tmp_path, "v1", time_created="2024-01-01T01:00:00Z")
    notice_v2 = _notice(tmp_path, "v2", time_created="2024-01-01T02:00:00Z")

    _run(output, notice_v1)
    event_root = output / "S240101abc"
    pointer = json.loads((event_root / "current.json").read_text(encoding="utf-8"))
    assert pointer["notice_version"] == "2024-01-01T01:00:00Z"

    _run(output, notice_v2)
    history = json.loads((event_root / "runs_history.json").read_text(encoding="utf-8"))
    assert [entry["notice_version"] for entry in history] == [
        "2024-01-01T01:00:00Z",
        "2024-01-01T02:00:00Z",
    ]
    pointer = json.loads((event_root / "current.json").read_text(encoding="utf-8"))
    assert pointer["notice_version"] == "2024-01-01T02:00:00Z"
    runs = sorted((event_root / "runs").iterdir())
    assert len(runs) == 2
    assert all((run / "coverage.json").is_file() for run in runs)


def test_retraction_removes_current_pointer_but_keeps_history(tmp_path: Path):
    output = tmp_path / "workspace"
    notice_active = _notice(tmp_path, "active", time_created="2024-01-01T01:00:00Z")
    notice_retracted = _notice(
        tmp_path,
        "retracted",
        time_created="2024-01-01T03:00:00Z",
        alert_type="RETRACTION",
    )

    _run(output, notice_active)
    event_root = output / "S240101abc"
    assert (event_root / "current.json").is_file()

    result = _run(output, notice_retracted)
    assert result.event.status == "retracted"
    assert not (event_root / "current.json").exists()
    retraction = json.loads((event_root / "RETRACTED.json").read_text(encoding="utf-8"))
    assert retraction["alert_type"] == "RETRACTION"
    history = json.loads((event_root / "runs_history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[-1]["status"] == "retracted"
