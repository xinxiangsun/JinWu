"""
Executable Quick Start examples（AUD-05 回归）
===============================================

--- 中文说明 ---
docs/quickstart.rst 与 docs/index.rst 的代码示例曾引用不存在的接口
（``EnergyBand(0.3, 2.0, unit="keV")``、``ChannelBand.from_energy_band``），
新用户照文档操作即失败（AUD-05）。本文件按教程顺序，用固定合成 OGIP
小样例执行文档中的每个关键调用，保证示例真实可执行——不依赖真实数据、
网络或 HEASoft。

覆盖教程段落：
  Part 1: Reading OGIP FITS Files（read_pha / read_lc / read_arf / readfits）
  Part 2: Working with Energy Bands（EnergyBand / channel_mask_from_ebounds /
          ChannelBand / band_from_arf_bins）
  Part 3: Computing Net Data（netdata / src - bkg）

EP/WXT 流水线段落需要真实观测目录，由相应的真实数据回归覆盖，此处不重复。

--- English ---
The Quick Start examples used to reference non-existent interfaces and
failed for new users (AUD-05).  This module executes every key call of
the tutorials in order against a fixed synthetic OGIP sample, so the
documentation stays executable.

运行方式 / Run:
    python -m pytest test/test_quickstart_examples.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jinwu.core import read_arf, read_lc, read_pha, readfits
from jinwu.core.base import RegionArea
from jinwu.core.data import ArfData, LightcurveData, PhaData
from jinwu.core.io import write_arf, write_lc, write_pha


# ============================================================================
# 合成 OGIP 小样例 / Synthetic OGIP fixtures
# ============================================================================

_N_CH = 20   # 20 个通道，每通道 0.5 keV，覆盖 0-10 keV
_DE = 0.5


def _make_lc_file(out: Path, *, rate_scale: float, role: str, seed: int) -> Path:
    """合成光变文件（构造字段与 test_datasets.py 的助手保持一致）。

    注意：LightcurveWriter 只落盘 ``rate``/``counts`` 字段，因此这里同时
    填充 rate/rate_err，使写出的文件能被 read_lc 读回。
    """
    rng = np.random.default_rng(seed)
    n = 50
    rate = rate_scale + rng.normal(0.0, 0.2, n)
    err = np.sqrt(np.maximum(rate, 0.0))
    region = RegionArea(role=role, shape="circle", area=1.0, component=0)
    lc = LightcurveData(
        time=np.arange(1.0, n + 1.0),
        value=rate,
        error=err,
        dt=1.0,
        exposure=float(n),
        bin_exposure=np.full(n, 1.0),
        is_rate=True,
        region=region,
        timezero=0.0,
        # OgipFitsBase 必需字段 / Required OgipFitsBase fields
        path=Path("<synthetic>"),
        header={},
        meta=None,
        headers_dump=None,
        time_raw=None,
        time_rel=None,
        timezero_obj=None,
        bin_lo=None,
        bin_hi=None,
        bin_width=None,
        binning="uniform",
        tstart=0.0,
        tseg=float(n),
        counts=None,
        rate=rate,
        counts_err=None,
        rate_err=err,
        err_dist=None,
        gti_start=None,
        gti_stop=None,
        quality=None,
        fracexp=None,
        backscal=None,
        areascal=None,
        telescop=None,
        timesys=None,
        mjdref=None,
        columns=(),
        ratio=None,
    )
    write_lc(lc, out)
    # 读取器依据 TELESCOP 构建 timezero_obj（EP 时钟格式已在 core.time 注册）
    from astropy.io import fits

    with fits.open(out, mode="update") as hdul:
        hdul["LIGHTCURVE"].header["TELESCOP"] = "EP"
        hdul.flush()
    return out


@pytest.fixture(scope="module")
def ogip_dir(tmp_path_factory):
    """写出教程引用的全部合成文件并返回其目录。"""
    from astropy.io import fits

    d = tmp_path_factory.mktemp("quickstart")

    # --- source.pha：SPECTRUM 主表 + EBOUNDS 扩展 ---
    channels = np.arange(_N_CH, dtype=int)
    pha = PhaData(
        path="<synthetic>",
        header={},
        meta={},
        headers_dump=None,
        channels=channels,
        counts=np.linspace(50.0, 5.0, _N_CH),
        exposure=1000.0,
        columns=("CHANNEL", "COUNTS"),
    )
    pha_path = d / "source.pha"
    write_pha(pha, pha_path)
    # 教程能段示例依赖 EBOUNDS：通道-能量映射存于 PHA/RMF，而非 ARF
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="J", array=channels),
            fits.Column(name="E_MIN", format="E", array=channels * _DE),
            fits.Column(name="E_MAX", format="E", array=(channels + 1) * _DE),
        ],
        name="EBOUNDS",
    )
    with fits.open(pha_path, mode="update") as hdul:
        hdul.append(ebounds)
        hdul.flush()

    # --- source.lc / background.lc ---
    _make_lc_file(d / "source.lc", rate_scale=10.0, role="src", seed=1)
    _make_lc_file(d / "background.lc", rate_scale=1.0, role="bkg", seed=2)

    # --- source.arf ---
    elo = np.arange(0.0, 10.0, _DE)
    write_arf(
        ArfData(
            path="<synthetic>",
            header={},
            meta={},
            headers_dump=None,
            energ_lo=elo,
            energ_hi=elo + _DE,
            specresp=np.ones(elo.size),
            columns=("ENERG_LO", "ENERG_HI", "SPECRESP"),
        ),
        d / "source.arf",
    )
    return d


# ============================================================================
# Part 1 — Reading OGIP FITS Files
# ============================================================================

def test_reading_ogip_files(ogip_dir):
    """教程第一段：read_pha / read_lc / read_arf 与 readfits 自动判型。"""
    pha = read_pha(str(ogip_dir / "source.pha"))
    assert float(pha.exposure) == pytest.approx(1000.0)
    assert len(pha.channels) == _N_CH

    lc = read_lc(str(ogip_dir / "source.lc"))
    # 读取器将 TIME 重定基到 0 起，绝对时刻保存在 timezero（相对+零点=绝对）
    assert float(lc.time.max()) - float(lc.time.min()) == pytest.approx(49.0)
    assert float(lc.timezero) == pytest.approx(1.0)

    arf = read_arf(str(ogip_dir / "source.arf"))
    assert arf.specresp.shape == (20,)

    data = readfits(str(ogip_dir / "source.pha"))
    assert data.kind == "pha"


# ============================================================================
# Part 2 — Working with Energy Bands
# ============================================================================

def test_working_with_energy_bands(ogip_dir):
    """教程第二段：四字段 EnergyBand 构造 + EBOUNDS 通道掩码 + ARF 反查。"""
    from jinwu.core import ChannelBand, EnergyBand, band_from_arf_bins
    from jinwu.core import channel_mask_from_ebounds

    # Define an energy band (emin/emax each carry their own unit string)
    soft_band = EnergyBand(emin=0.3, emin_unit="keV", emax=2.0, emax_unit="keV")
    hard_band = EnergyBand(emin=2.0, emin_unit="keV", emax=10.0, emax_unit="keV")

    # Map an energy band onto detector channels via the EBOUNDS extension
    pha = read_pha(str(ogip_dir / "source.pha"))
    assert pha.ebounds is not None
    mask_soft = channel_mask_from_ebounds(pha.ebounds, soft_band)
    mask_hard = channel_mask_from_ebounds(pha.ebounds, hard_band)

    # 0.5 keV/通道：soft 与通道 0-3（0-2.0 keV）相交，hard 与通道 4-19 相交
    assert int(mask_soft.sum()) == 4
    assert int(mask_hard.sum()) == _N_CH - 4
    assert int((mask_soft & mask_hard).sum()) == 0

    # Restrict to a channel range on top of the energy selection
    ch_band = ChannelBand(ch_lo=0, ch_hi=3)
    mask_soft_ch = channel_mask_from_ebounds(pha.ebounds, soft_band, ch_band)
    assert np.array_equal(mask_soft_ch, mask_soft)

    # Reverse direction: the energy range covered by ARF bin indices
    band = band_from_arf_bins(str(ogip_dir / "source.arf"), bin_lo=1, bin_hi=4)
    assert float(band.emin) == pytest.approx(0.0)
    assert float(band.emax) == pytest.approx(2.0)


def test_index_page_energy_band_example():
    """docs/index.rst 顶部示例：四字段 EnergyBand 构造。"""
    import jinwu.core as jw

    band = jw.EnergyBand(emin=0.3, emin_unit="keV", emax=10.0, emax_unit="keV")
    assert band.emin_unit == "keV"
    assert band.emax_unit == "keV"


# ============================================================================
# Part 3 — Computing Net Data
# ============================================================================

def test_computing_net_data(ogip_dir):
    """教程第三段：netdata 减除与 `src - bkg` 速记等价。

    合成文件不携带区域面积信息（LightcurveWriter 不落盘 region），
    这里按流水线的做法在读回对象上显式附加 RegionArea，
    使自动 ratio（面积×曝光加权）可以推断。
    """
    import jinwu.core as jw

    src = jw.read_lc(str(ogip_dir / "source.lc"))
    bkg = jw.read_lc(str(ogip_dir / "background.lc"))
    src.region = RegionArea(role="src", shape="circle", area=100.0, component=0)
    bkg.region = RegionArea(role="bkg", shape="circle", area=400.0, component=0)

    # 自动 ratio = (100*50)/(400*50) = 0.25
    net_auto = src - bkg
    np.testing.assert_allclose(
        np.asarray(net_auto.value, dtype=float),
        np.asarray(src.value, dtype=float) - 0.25 * np.asarray(bkg.value, dtype=float),
        rtol=1e-9,
    )

    # 手动 ratio 与自动 ratio 一致
    net_manual = jw.netdata(src, bkg, ratio=0.25)
    np.testing.assert_allclose(
        np.asarray(net_manual.value, dtype=float),
        np.asarray(net_auto.value, dtype=float),
        rtol=1e-9,
    )
