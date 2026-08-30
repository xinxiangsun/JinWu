# jinwu (core)

Core analysis layer of the jinwu (金乌) toolkit for high-energy transient
light-curve and spectral analysis: OGIP FITS I/O, time/GTI utilities, spectral
fitting helpers, background modeling, response utilities and plotting.

Instrument support lives in separate distributions that depend on this one
and contribute to the same `jinwu` import namespace:

- [`jinwu-ep`](https://pypi.org/project/jinwu-ep/) — Einstein Probe (WXT)
- [`jinwu-swift`](https://pypi.org/project/jinwu-swift/) — Swift/BAT
- [`jinwu-fermi`](https://pypi.org/project/jinwu-fermi/) — Fermi/GBM

```bash
pip install jinwu              # core only
pip install "jinwu[ep]"        # core + Einstein Probe support
pip install "jinwu[swift]"     # core + Swift/BAT support
pip install "jinwu[fermi]"     # core + Fermi/GBM support
pip install "jinwu[crossmatch]"  # host-galaxy cross-matching tools
```

See the [repository root](https://github.com/xinxiangsun/jinwu) for the full
documentation and roadmap.
