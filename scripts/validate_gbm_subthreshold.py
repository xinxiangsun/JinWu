"""Reproduce two public GBM events; does not certify FAR or science quality.

Run in hea. Data are discovered under ROOT/{GW170817,GRB140606A}; --download
fetches missing archive products into each pipeline workspace.
Event references: https://arxiv.org/abs/1710.05834 (GW170817 UTC),
https://arxiv.org/abs/1806.02378 (GRB140606A MET 423745096.496).
"""
import argparse
import json
from pathlib import Path

import astropy.units as u
from jinwu.core.time import Time
from jinwu.fermi.gbm.subthreshold import (
    GBMTargetedSearchConfig, GBMTargetedSearchInput, run_targeted_search,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--short', action='store_true', help='±5 s smoke run instead of default ±30 s')
    args = parser.parse_args()
    config = GBMTargetedSearchConfig(search_interval=([-5, 5] if args.short else [-30, 30])*u.s)
    events = {'GW170817': Time('2017-08-17T12:41:04.429126', scale='utc'),
              'GRB140606A': Time(423745096.496, format='fermi')}
    summary = {}
    for name, time in events.items():
        result = run_targeted_search(GBMTargetedSearchInput(
            target_id=name, trigger_time=time, root=args.root / name,
            template_root=args.templates, output_root=args.output / name,
            download=args.download), config=config)
        summary[name] = {'status': result.status, 'science_status': result.science_status,
                         'candidate_count': len(result.candidates), 'report': result.products['report']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'validation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
