"""Run one Swift BAT+XRT GRB with local products.

Set ``workspace`` to the research workspace holding the merged catalog and
``xrt_products`` to an already downloaded UKSSDC single-target product.  The
analysis output is deliberately separate from both inputs.
"""

from pathlib import Path

from jinwu.core.config import SwiftGRB
from jinwu.core.pipeline import pipeline
from jinwu.swift.grb import SwiftGRBInput


workspace = Path("/path/to/swift_highz_grb/output")
xrt_products = workspace / "xrt_product_requests_maybe/downloads/GRB_050904"

result = pipeline(
    SwiftGRB(),
    SwiftGRBInput(
        target_id="050904",
        root=workspace,
        output_root=Path("/tmp/grb050904_jinwu"),
        xrt_products_dir=xrt_products,
        # Leave this unset to use the catalog redshift; no fit assumes z=0.
        galactic_nh_1e22=0.09,
    ),
).run()

print(result.status.value)
for stage, products in result.products.items():
    for name, filename in products.items():
        print(f"{stage}.{name}: {filename}")
