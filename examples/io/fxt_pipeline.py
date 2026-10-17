"""Fit local FXT L2/L3 products and produce observer-frame 0.5--4 keV flux.

Pass one observation directory or a collection of observation/module folders.
The default fits PL with zero/free additional NH, independently per epoch.
Pile-up processing is deferred for supplied L2/L3 products.
"""

from jinwu.ep.fxt import FXTInput, FXTPipeline


def run_fxt(root, output, *, name, ra_deg, dec_deg, t0_utc, redshift=0.0):
    return FXTPipeline(FXTInput(
        target_id=name, root=root, output_root=output, ra_deg=ra_deg, dec_deg=dec_deg,
        trigger_time_utc=t0_utc, redshift=redshift,
    )).run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--t0", required=True)
    parser.add_argument("--redshift", type=float, default=0)
    args = parser.parse_args()
    result = run_fxt(args.root, args.output, name=args.name, ra_deg=args.ra,
                     dec_deg=args.dec, t0_utc=args.t0, redshift=args.redshift)
    print(result.status, result.products.get("report"))
