"""WXT single-frame diagnostics adapted from EPTA ``cosmicray2.py``.

The local reference is ``/home/xinxiang/research/epta/cosmicray2.py``:
match TIME/X/Y/PI to CMOSFRAM, screen source frame counts, inspect RAWXY.
Screening alone never removes events or establishes a cosmic-ray origin.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.io import fits

from ...core.products import write_json


def screen_cosmic_ray_frames(
    cleaned_event: Path,
    source_event: Path,
    unfiltered_event: Path | None,
    output_dir: Path,
    *,
    threshold: int = 5,
    confirmed_frames: Sequence[int] = (),
    make_plots: bool = True,
) -> dict:
    """Screen frames and optionally exclude explicitly human-confirmed frames.

    ``threshold`` is the strict source-count screening threshold. Frame removal
    requires CMOSFRAM, a unique frame timestamp, TIMEDEL, TIMEPIXR, and native
    GTIs; the independent output gets reduced GTIs and live exposure. Matching
    is exact on TIME/X/Y/PI, following EPTA, with ambiguous joins rejected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "missing_cmosfram", "screening_threshold": int(threshold),
              "confirmed_frames": list(map(int, confirmed_frames)),
              "screening_is_classification": False, "outputs": {},
              "reference": "EPTA cosmicray2.py; TIME/X/Y/PI to CMOSFRAM; RAWXY review"}
    if unfiltered_event is None:
        result["reason"] = "No unfiltered event product containing CMOSFRAM"
        result["requires_review"] = bool(confirmed_frames)
        return result
    with fits.open(unfiltered_event, memmap=False) as hdul:
        raw = hdul["EVENTS"].data.copy()
        header = hdul["EVENTS"].header.copy()
    required = {"TIME", "X", "Y", "PI", "CMOSFRAM"}
    if not required.issubset(raw.names):
        result["reason"] = f"Unfiltered event lacks {sorted(required.difference(raw.names))}"
        result["requires_review"] = bool(confirmed_frames)
        return result
    lookup = {}
    for row in raw:
        key = tuple(row[name] for name in ("TIME", "X", "Y", "PI"))
        frame = int(row["CMOSFRAM"])
        if key in lookup and lookup[key] != frame:
            result.update(status="ambiguous_frame_join", requires_review=bool(confirmed_frames))
            return result
        lookup[key] = frame

    def frame_ids(events):
        return np.asarray([lookup.get(tuple(row[n] for n in ("TIME", "X", "Y", "PI")), -1)
                           for row in events], dtype=np.int64)

    with fits.open(cleaned_event, memmap=False) as hdul:
        clean = hdul["EVENTS"].data.copy()
        clean_header = hdul["EVENTS"].header.copy()
    with fits.open(source_event, memmap=False) as hdul:
        source = hdul["EVENTS"].data.copy()
    frames, source_frames = frame_ids(clean), frame_ids(source)
    if np.any(frames < 0) or np.any(source_frames < 0):
        result.update(status="incomplete_frame_join", unmatched_clean=int(np.sum(frames < 0)),
                      unmatched_source=int(np.sum(source_frames < 0)),
                      requires_review=bool(confirmed_frames))
        return result
    all_ids, all_counts = np.unique(frames, return_counts=True)
    src_ids, src_counts = np.unique(source_frames, return_counts=True)
    suspicious = src_ids[src_counts > threshold]
    result.update(status="screened", candidate_frames=suspicious.tolist(),
                  frame_ids=all_ids.tolist(), frame_counts=all_counts.tolist(),
                  source_frame_ids=src_ids.tolist(), source_frame_counts=src_counts.tolist(),
                  matched_clean_events=len(clean), matched_source_events=len(source))
    result["pointing_mode"] = clean_header.get("OBS_MODE")
    result["position_type"] = clean_header.get("XPITYPE")
    if make_plots and all_ids.size:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(all_ids - all_ids[0], all_counts, lw=0.6, label="All cleaned events")
        ax.plot(src_ids - all_ids[0], src_counts, ".", ms=3, label="Source region")
        ax.axhline(threshold, color="0.5", ls="--")
        ax.set(xlabel="CMOS frame offset", ylabel="Counts per frame")
        ax.legend()
        fig.tight_layout()
        path = output_dir / "frame_counts.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        result["outputs"]["frame_counts_plot"] = str(path)
        for frame in sorted(set(suspicious.tolist()).union(map(int, confirmed_frames))):
            mask = frames == frame
            if not np.any(mask) or not {"RAWX", "RAWY"}.issubset(clean.names):
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(clean["RAWX"][mask], clean["RAWY"][mask], s=8)
            ax.set(xlabel="RAWX [pixel]", ylabel="RAWY [pixel]",
                   title=f"Frame {frame}: {int(np.sum(mask))} cleaned events")
            ax.set_aspect("equal", adjustable="datalim")
            path = output_dir / f"frame_{frame}_rawxy.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            result["outputs"][f"frame_{frame}_rawxy"] = str(path)
    if confirmed_frames:
        if (str(clean_header.get("OBS_MODE", "")).upper() != "POINTING" or
                str(clean_header.get("XPITYPE", "")).upper() != "NOMINAL"):
            result.update(status="pointing_exposure_scaling_unverified", requires_review=True)
            return result
        intervals = []
        dt, pixel = header.get("TIMEDEL"), header.get("TIMEPIXR")
        if dt is None or pixel is None or not np.isfinite(dt) or dt <= 0 or not 0 <= pixel <= 1:
            result.update(status="frame_time_unavailable", requires_review=True)
            return result
        for frame in sorted(set(map(int, confirmed_frames))):
            times = np.asarray(raw["TIME"][raw["CMOSFRAM"] == frame], dtype=float)
            if not times.size or np.ptp(times) > 1e-6:
                result.update(status="frame_time_ambiguous", requires_review=True)
                return result
            start = float(times[0] - float(pixel) * float(dt))
            intervals.append((start, start + float(dt)))
        with fits.open(cleaned_event, memmap=False) as hdul:
            hdul = fits.HDUList([hdu.copy() for hdu in hdul])
            gti = hdul["GTI"]
            original = [(float(a), float(b)) for a, b in zip(gti.data["START"], gti.data["STOP"])]
            kept = original
            for start, stop in intervals:
                kept = [(lo, hi) for left, right in kept
                        for lo, hi in ((left, min(right, start)), (max(left, stop), right))
                        if hi > lo]
            old_live = sum(b-a for a,b in original)
            new_live = sum(b-a for a,b in kept)
            if old_live <= 0 or new_live <= 0:
                result.update(status="no_live_exposure_after_exclusion", requires_review=True)
                return result
            mask = ~np.isin(frames, confirmed_frames)
            hdul["EVENTS"].data = clean[mask]
            new_gti = fits.BinTableHDU.from_columns([
                fits.Column(name="START", format="D", array=[a for a,b in kept]),
                fits.Column(name="STOP", format="D", array=[b for a,b in kept]),
            ], header=gti.header, name="GTI")
            hdul[hdul.index_of("GTI")] = new_gti
            fraction = new_live / old_live
            for hdu in hdul:
                if "EXPOSURE" in hdu.header:
                    hdu.header["EXPOSURE"] = float(hdu.header["EXPOSURE"]) * fraction
                if "ONTIME" in hdu.header:
                    hdu.header["ONTIME"] = new_live
                hdu.header.add_history("JinWu: human-confirmed CMOS frames excluded; GTI/live exposure updated")
            path = output_dir / cleaned_event.name
            hdul.writeto(path, overwrite=True)
        result.update(status="confirmed_frames_excluded", exclusion_intervals_s=intervals,
                      removed_clean_events=int(np.sum(~mask)), live_exposure_fraction=fraction,
                      original_live_s=old_live, retained_live_s=new_live,
                      exposure_assumption="Pointing exposure maps scale by retained native-GTI live fraction")
        result["outputs"]["filtered_event"] = str(path)
    manifest = write_json(output_dir / "cosmic_ray_qc.json", result)
    result["outputs"]["manifest"] = str(manifest)
    return result
