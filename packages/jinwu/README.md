# jinwu (core)

Core analysis layer of the jinwu (金乌) toolkit for high-energy transient
light-curve and spectral analysis: OGIP FITS I/O, time/GTI utilities, spectral
fitting helpers, background modeling, response utilities and plotting.

Instrument support lives in separate distributions that depend on this one
and contribute to the same `jinwu` import namespace:

- [`jinwu-ep`](https://pypi.org/project/jinwu-ep/) — Einstein Probe (WXT)
- [`jinwu-swift`](https://pypi.org/project/jinwu-swift/) — Swift/BAT
- [`jinwu-fermi`](https://pypi.org/project/jinwu-fermi/) — Fermi/GBM

GECAMTools 接口由源码中的 `jinwu-gecam` 提供（`jinwu.gecam`）；官方后端需另装。

当前源码的 `all` 包含除 Swift 外的全部可选依赖；Swift 开发暂缓，
安装时需显式选择 `[swift]` 或 `[all,swift]`。此配置将在后续发布版中生效。

```bash
pip install jinwu              # core only
pip install "jinwu[ep]"        # core + Einstein Probe support
pip install "jinwu[swift]"     # core + Swift/BAT support
pip install "jinwu[fermi]"     # core + Fermi/GBM support
pip install "jinwu[crossmatch]"  # host-galaxy cross-matching tools
```

See the [repository root](https://github.com/xinxiangsun/jinwu) for the full
documentation and roadmap.
