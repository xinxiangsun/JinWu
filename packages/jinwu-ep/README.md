# jinwu-ep

Einstein Probe (WXT) instrument support for the
[jinwu](https://pypi.org/project/jinwu/) analysis toolkit: observation
discovery, WXT pointing-mode data reduction orchestration, event/light-curve
products and spectral extraction helpers.

```bash
pip install jinwu-ep
```

Import the normal-pointing workflow as
`from jinwu.ep.wxt import WXTPointingInput, WXTPointingPipeline`.
Run through `exposure_arm_qc`, inspect the proposed source/background regions
and exposure diagnostics, call `approve_regions()`, then resume. Supply an
independent `output_root` so products do not land beside the input observation.
