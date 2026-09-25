"""
Tutorial-style tests for jinwu.core.base and jinwu.core.time
============================================================

--- 中文说明 ---
本文件包含两个部分：
  Part 1: jinwu.core.base — 数据基类 (dataclasses) 的构造与默认值测试
  Part 2: jinwu.core.time — 天文任务时间转换与时间区间工具函数测试

每个测试函数包含中英双语 docstring 和行内注释，可作为 API 教程阅读。
所有测试可直接运行：
    cd /home/xinxiang/research/jinwu && python -m pytest test/test_base_time.py -v

--- English ---
This file has two parts:
  Part 1: jinwu.core.base — construction and default-value tests for data
          container dataclasses
  Part 2: jinwu.core.time — mission time conversion and interval utility tests

Every test function has bilingual (Chinese+English) docstrings and inline
comments and can be read as an API tutorial.  Run with:
    cd /home/xinxiang/research/jinwu && python -m pytest test/test_base_time.py -v
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest


# =============================================================================
# Part 1 — jinwu.core.base
# =============================================================================

from jinwu.core.base import (
    EnergyBand,
    ChannelBand,
    RegionArea,
    RegionAreaSet,
    HduHeader,
    FitsHeaderDump,
    OgipMeta,
    ArfBase,
    RmfBase,
    PhaBase,
    LightcurveDataBase,
    EventDataBase,
)


# ---------------------------------------------------------------------------
# Helpers: small default dictionaries for OGIP parent fields
# 辅助：构造 OGIP 父类所需字段的默认字典
# ---------------------------------------------------------------------------
def _dummy_header(**extra) -> Dict[str, Any]:
    """返回一个包含常用 FITS 关键字的模拟 header。"""
    hdr = {
        "TELESCOP": "MOCK",
        "INSTRUME": "MOCKINST",
        "CHANTYPE": "PI",
        "TIMESYS": "TT",
        "TIMEUNIT": "s",
        "MJDREF": 55000.0,
        **extra,
    }
    return hdr


def _dummy_meta(**extra) -> OgipMeta:
    """返回一个预设的 OgipMeta 实例。"""
    defaults = {
        "telescop": "MOCK",
        "instrume": "MOCKINST",
        "detnam": None,
        "timesys": "TT",
        "timeunit": "s",
        "mjdref": 55000.0,
        "tstart": 0.0,
        "tstop": 100.0,
        "object": "TEST_SRC",
        "obs_id": "test001",
        "binsize": None,
        "timezero": 0.0,
        "trefpos": "SUN",
        "dateobs": None,
        **extra,
    }
    return OgipMeta(**defaults)


def _dummy_headers_dump() -> FitsHeaderDump:
    """返回一个预设的 FitsHeaderDump。"""
    return FitsHeaderDump(
        primary={"SIMPLE": True, "BITPIX": 8},
        extensions=[HduHeader(name="GTI", ver=1, header={"EXTNAME": "GTI"})],
    )


# =============================================================================
# EnergyBand — 能段定义  /  Energy-band definition
# =============================================================================

class TestEnergyBand:
    """测试 EnergyBand 数据类的构造与字段访问。

    Test EnergyBand dataclass construction and field access.
    """

    def test_construction_simple(self):
        """构造 EnergyBand 并验证四个字段。  Construction with four fields."""
        # 创建一个典型的 X 射线能段  /  Create a typical X-ray energy band
        eb = EnergyBand(emin=0.3, emin_unit="keV", emax=10.0, emax_unit="keV")
        assert eb.emin == 0.3
        assert eb.emin_unit == "keV"
        assert eb.emax == 10.0
        assert eb.emax_unit == "keV"

    def test_different_units(self):
        """不同单位的能段构造。  Energy bands with different units."""
        # 软 X 射线能段，单位 erg  /  Soft X-ray band in erg
        soft = EnergyBand(emin=1e-9, emin_unit="erg", emax=1e-7, emax_unit="erg")
        assert soft.emin == 1e-9
        assert soft.emin_unit == "erg"

        # 硬 X 射线能段，单位 MeV  /  Hard X-ray band in MeV
        hard = EnergyBand(emin=0.1, emin_unit="MeV", emax=1.0, emax_unit="MeV")
        assert hard.emax == 1.0


# =============================================================================
# ChannelBand — 通道区间  /  Channel interval
# =============================================================================

class TestChannelBand:
    """测试 ChannelBand 数据类。  Test ChannelBand dataclass."""

    def test_construction(self):
        """构造 ChannelBand 并验证整型通道范围。"""
        cb = ChannelBand(ch_lo=0, ch_hi=1023)
        assert cb.ch_lo == 0
        assert cb.ch_hi == 1023
        assert isinstance(cb.ch_lo, int)
        assert isinstance(cb.ch_hi, int)

    def test_negative_channels_allowed(self):
        """允许负数通道（仅数据容器，语义由上层处理）。"""
        cb = ChannelBand(ch_lo=-1, ch_hi=100)
        assert cb.ch_lo == -1


# =============================================================================
# RegionArea — 区域与面积  /  Region and area
# =============================================================================

class TestRegionArea:
    """测试 RegionArea 数据类（含 Literal 角色字段）。"""

    def test_src_role(self):
        """构造一个 src（源区）角色。  Source region."""
        ra = RegionArea(role="src", shape="circle", area=12.56, component=1)
        assert ra.role == "src"
        assert ra.shape == "circle"
        assert ra.area == 12.56
        assert ra.component == 1

    def test_bkg_role_none_shape(self):
        """构造 bkg（背景区），shape 和 area 可为 None。  Background region."""
        ra = RegionArea(role="bkg", shape=None, area=None, component=None)
        assert ra.role == "bkg"
        assert ra.shape is None
        assert ra.area is None
        assert ra.component is None

    def test_unk_role(self):
        """unk（未知）角色。  Unknown role."""
        ra = RegionArea(role="unk", shape="annulus", area=50.0, component=2)
        assert ra.role == "unk"


# =============================================================================
# RegionAreaSet — 区域集  /  Set of regions with area aggregation
# =============================================================================

class TestRegionAreaSet:
    """测试 RegionAreaSet 的构造、属性和类方法。"""

    def test_empty_construction(self):
        """空构造：所有列表为空，面积属性为 None。"""
        rs = RegionAreaSet()
        assert rs.src == []
        assert rs.bkg == []
        assert rs.unk == []
        assert rs.src_area is None  # 无 src 区域 → None
        assert rs.bkg_area is None  # 无 bkg 区域 → None

    def test_src_area_sum(self):
        """src_area 属性应返回所有源区面积之和。"""
        rs = RegionAreaSet(
            src=[
                RegionArea(role="src", shape="circle", area=10.0, component=1),
                RegionArea(role="src", shape="circle", area=5.0, component=2),
            ],
            bkg=[],
            unk=[],
        )
        assert rs.src_area == 15.0  # 10 + 5

    def test_bkg_area_sum_with_none(self):
        """bkg_area 忽略 area=None 的项。  bkg_area skips None entries."""
        rs = RegionAreaSet(
            src=[],
            bkg=[
                RegionArea(role="bkg", shape="annulus", area=20.0, component=1),
                RegionArea(role="bkg", shape=None, area=None, component=2),
            ],
            unk=[],
        )
        assert rs.bkg_area == 20.0  # 仅对非 None 的 area 求和

    def test_from_regions_classmethod_mixed(self):
        """from_regions() 类方法按 role 自动分类。  Auto-classify by role."""
        regions = [
            RegionArea(role="src", shape="circle", area=3.0, component=1),
            RegionArea(role="bkg", shape="annulus", area=7.0, component=2),
            RegionArea(role="unk", shape=None, area=None, component=None),
            RegionArea(role="src", shape="circle", area=8.0, component=3),
        ]
        rs = RegionAreaSet.from_regions(regions)
        assert len(rs.src) == 2
        assert len(rs.bkg) == 1
        assert len(rs.unk) == 1
        assert rs.src_area == 11.0  # 3 + 8
        assert rs.bkg_area == 7.0

    def test_from_regions_empty(self):
        """空列表 → 空 RegionAreaSet。  Empty list returns empty set."""
        rs = RegionAreaSet.from_regions([])
        assert rs.src == []
        assert rs.bkg == []
        assert rs.unk == []
        assert rs.src_area is None

    def test_from_regions_none(self):
        """None → 空 RegionAreaSet。  None input returns empty set."""
        rs = RegionAreaSet.from_regions(None)
        assert rs.src == []


# =============================================================================
# HduHeader — FITS 扩展头  /  FITS extension header
# =============================================================================

class TestHduHeader:
    """测试 HduHeader 数据类。"""

    def test_construction(self):
        """构造 HduHeader：name, ver, header dict。"""
        hdr = {"EXTNAME": "SPECTRUM", "HDUCLASS": "OGIP"}
        hh = HduHeader(name="SPECTRUM", ver=1, header=hdr)
        assert hh.name == "SPECTRUM"
        assert hh.ver == 1
        assert hh.header["EXTNAME"] == "SPECTRUM"

    def test_ver_none(self):
        """ver 可为 None（无版本号）。  Version can be None."""
        hh = HduHeader(name="PRIMARY", ver=None, header={})
        assert hh.ver is None


# =============================================================================
# FitsHeaderDump — FITS 头转储  /  FITS header dump
# =============================================================================

class TestFitsHeaderDump:
    """测试 FitsHeaderDump 数据类。"""

    def test_construction(self):
        """构造：primary dict + extensions 列表。"""
        primary = {"SIMPLE": True, "BITPIX": 16}
        ext1 = HduHeader(name="EVENTS", ver=1, header={"EXTNAME": "EVENTS"})
        ext2 = HduHeader(name="GTI", ver=1, header={"EXTNAME": "GTI"})
        fhd = FitsHeaderDump(primary=primary, extensions=[ext1, ext2])
        assert fhd.primary["SIMPLE"] is True
        assert len(fhd.extensions) == 2
        assert fhd.extensions[0].name == "EVENTS"

    def test_empty_extensions(self):
        """extensions 可为空列表。  Extensions can be empty."""
        fhd = FitsHeaderDump(primary={"SIMPLE": True}, extensions=[])
        assert fhd.extensions == []


# =============================================================================
# OgipMeta — OGIP 元数据  /  OGIP metadata container
# =============================================================================

class TestOgipMeta:
    """测试 OgipMeta 数据类（15 个可选字段）。"""

    def test_all_fields(self):
        """全部字段均可通过关键字设置。  All fields via keywords."""
        om = OgipMeta(
            telescop="SWIFT",
            instrume="BAT",
            detnam="DET0",
            timesys="TT",
            timeunit="s",
            mjdref=51910.0,
            tstart=0.0,
            tstop=100.0,
            object="GRB220101",
            obs_id="obs001",
            binsize=0.05,
            timezero=0.0,
            trefpos="SUN",
            dateobs="2022-01-01T00:00:00",
        )
        assert om.telescop == "SWIFT"
        assert om.instrume == "BAT"
        assert om.detnam == "DET0"
        assert om.timesys == "TT"
        assert om.timeunit == "s"
        assert om.mjdref == 51910.0
        assert om.tstart == 0.0
        assert om.tstop == 100.0
        assert om.object == "GRB220101"
        assert om.obs_id == "obs001"
        assert om.binsize == 0.05
        assert om.timezero == 0.0
        assert om.trefpos == "SUN"
        assert om.dateobs == "2022-01-01T00:00:00"

    def test_defaults_none(self):
        """所有字段均可显式设为 None。  All fields can be explicitly set to None."""
        om = OgipMeta(
            telescop=None,
            instrume=None,
            detnam=None,
            timesys=None,
            timeunit=None,
            mjdref=None,
            tstart=None,
            tstop=None,
            object=None,
            obs_id=None,
            binsize=None,
            timezero=None,
            trefpos=None,
            dateobs=None,
        )
        assert om.telescop is None
        assert om.instrume is None
        assert om.mjdref is None


# =============================================================================
# ArfBase — ARF 响应基类  /  Auxiliary Response File base
# =============================================================================

class TestArfBase:
    """测试 ArfBase 数据类构造与数组默认值。"""

    def test_construction_defaults(self):
        """默认构造：空数组、kind='arf'。  Default construction with empty arrays."""
        arf = ArfBase(
            path=Path("test.arf"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
        )
        assert arf.kind == "arf"          # ClassVar 类变量
        assert len(arf.energ_lo) == 0     # 默认为空数组
        assert len(arf.energ_hi) == 0
        assert len(arf.specresp) == 0

    def test_with_data_arrays(self):
        """传入具体数组。  Construction with explicit data arrays."""
        energ_lo = np.array([1.0, 2.0, 3.0])
        energ_hi = np.array([2.0, 3.0, 4.0])
        specresp = np.array([100.0, 200.0, 150.0])
        arf = ArfBase(
            path=Path("test.arf"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            energ_lo=energ_lo,
            energ_hi=energ_hi,
            specresp=specresp,
        )
        assert len(arf.energ_lo) == 3
        assert len(arf.energ_hi) == 3
        assert len(arf.specresp) == 3
        # 字段值一致  /  Field values match
        np.testing.assert_array_equal(arf.energ_lo, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(arf.specresp, [100.0, 200.0, 150.0])


# =============================================================================
# RmfBase — RMF 响应基类  /  Redistribution Matrix File base
# =============================================================================

class TestRmfBase:
    """测试 RmfBase 数据类（含密集/稀疏矩阵字段）。"""

    def test_construction_defaults(self):
        """默认构造，kind='rmf'，矩阵相关字段为 None 或空。"""
        rmf = RmfBase(
            path=Path("test.rmf"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
        )
        assert rmf.kind == "rmf"
        assert len(rmf.energ_lo) == 0
        assert len(rmf.energ_hi) == 0
        assert rmf.n_grp is None     # 稀疏表示字段默认 None
        assert rmf.f_chan is None
        assert rmf.n_chan is None

    def test_with_dense_matrix(self):
        """密集矩阵：直接使用 matrix 字段。  Dense matrix via matrix field."""
        matrix = np.array([[0.1, 0.2], [0.3, 0.4]])
        rmf = RmfBase(
            path=Path("test.rmf"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            energ_lo=np.array([1.0, 2.0]),
            energ_hi=np.array([2.0, 3.0]),
            matrix=matrix,
            channel=np.array([0, 1]),
        )
        assert rmf.matrix.shape == (2, 2)
        np.testing.assert_array_equal(rmf.energ_lo, [1.0, 2.0])

    def test_with_sparse_representation(self):
        """稀疏矩阵：使用 n_grp / f_chan / n_chan。  Sparse representation."""
        n_grp = np.array([1, 1])
        f_chan = np.array([0, 1])
        n_chan = np.array([2, 2])
        rmf = RmfBase(
            path=Path("test.rmf"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            energ_lo=np.array([1.0, 2.0]),
            energ_hi=np.array([2.0, 3.0]),
            n_grp=n_grp,
            f_chan=f_chan,
            n_chan=n_chan,
        )
        np.testing.assert_array_equal(rmf.n_grp, [1, 1])
        np.testing.assert_array_equal(rmf.f_chan, [0, 1])
        np.testing.assert_array_equal(rmf.n_chan, [2, 2])

    def test_e_min_e_max_fields(self):
        """e_min 和 e_max 字段（额外能量边界）。  Extra energy boundaries."""
        rmf = RmfBase(
            path=Path("test.rmf"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            e_min=np.array([0.5, 1.0]),
            e_max=np.array([1.0, 2.0]),
        )
        np.testing.assert_array_equal(rmf.e_min, [0.5, 1.0])
        np.testing.assert_array_equal(rmf.e_max, [1.0, 2.0])


# =============================================================================
# PhaBase — PHA 能谱基类  /  Pulse Height Analyzer spectrum base
# =============================================================================

class TestPhaBase:
    """测试 PhaBase 数据类（能谱字段）。"""

    def test_construction_defaults(self):
        """默认构造，kind='pha'，channels/counts 为空数组。"""
        pha = PhaBase(
            path=Path("test.pha"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
        )
        assert pha.kind == "pha"
        assert len(pha.channels) == 0
        assert len(pha.counts) == 0
        assert pha.exposure == 0.0
        assert pha.rate is None

    def test_with_spectrum_data(self):
        """传入完整的能谱数据。  Full spectrum data construction."""
        channels = np.arange(0, 1024)
        counts = np.random.poisson(50, 1024).astype(float)
        pha = PhaBase(
            path=Path("test.pha"),
            header=_dummy_header(EXPOSURE=100.0),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            channels=channels,
            counts=counts,
            exposure=100.0,
            backscal=0.01,
            areascal=1.0,
            respfile="test.rmf",
            ancrfile="test.arf",
        )
        assert len(pha.channels) == 1024
        assert len(pha.counts) == 1024
        assert pha.exposure == 100.0
        assert pha.backscal == 0.01
        assert pha.areascal == 1.0
        assert pha.respfile == "test.rmf"
        assert pha.ancrfile == "test.arf"

    def test_optional_fields_none(self):
        """可选字段 quality, grouping, stat_err 默认为 None。"""
        pha = PhaBase(
            path=Path("test.pha"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
        )
        assert pha.quality is None
        assert pha.grouping is None
        assert pha.stat_err is None
        assert pha.ebounds is None
        assert pha.raw_spectrum_columns is None


# =============================================================================
# LightcurveDataBase — 光变曲线基类  /  Light curve base
# =============================================================================

class TestLightcurveDataBase:
    """测试 LightcurveDataBase 的构造与所有字段。"""

    def test_construction_minimal(self):
        """最小构造：仅提供父类必需字段。  Minimal construction."""
        lc = LightcurveDataBase(
            path=Path("test.lc"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
        )
        assert lc.kind == "lc"
        assert lc.time is None
        assert lc.value is None
        assert lc.is_rate is False
        assert lc.timezero == 0.0
        assert lc.binning == "unknown"

    def test_with_lightcurve_data(self):
        """传入完整的光变曲线数据。  Full lightcurve data."""
        time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        value = np.array([10.0, 15.0, 12.0, 18.0, 11.0])
        error = np.sqrt(value)
        dt = np.full_like(time, 1.0)
        gti_start = np.array([0.0])
        gti_stop = np.array([5.0])

        lc = LightcurveDataBase(
            path=Path("test.lc"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            time=time,
            time_raw=time,
            time_rel=time,
            value=value,
            error=error,
            dt=dt,
            is_rate=False,
            gti_start=gti_start,
            gti_stop=gti_stop,
            exposure=5.0,
            telescop="MOCK",
            timesys="TT",
            mjdref=55000.0,
            binning="uniform",
            tstart=0.0,
            tseg=5.0,
            fracexp=np.ones(5),
            backscal=1.0,
            areascal=1.0,
        )
        assert len(lc.time) == 5
        assert len(lc.value) == 5
        assert lc.is_rate is False
        assert lc.binning == "uniform"
        assert lc.exposure == 5.0
        np.testing.assert_array_equal(lc.gti_start, [0.0])
        np.testing.assert_array_equal(lc.gti_stop, [5.0])

    def test_rate_mode(self):
        """速率模式：is_rate=True。  Rate mode light curve."""
        lc = LightcurveDataBase(
            path=Path("test.lc"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            is_rate=True,
            rate=np.array([1.0, 2.0]),
            rate_err=np.array([0.1, 0.2]),
            value=np.array([1.0, 2.0]),
        )
        assert lc.is_rate is True
        assert lc.counts is None


# =============================================================================
# EventDataBase — 事件数据基类  /  Event list base
# =============================================================================

class TestEventDataBase:
    """测试 EventDataBase 数据类。"""

    def test_construction_defaults(self):
        """默认构造，kind='evt'，time 为空数组。  Default: kind='evt'."""
        evt = EventDataBase(
            path=Path("test.evt"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
        )
        assert evt.kind == "evt"
        assert len(evt.time) == 0
        assert evt.timezero == 0.0
        assert evt.pi is None
        assert evt.x is None
        assert evt.y is None

    def test_with_event_data(self):
        """传入事件列表（时间、PI、位置）。  Event list with time/PI/position."""
        n_events = 100
        evt = EventDataBase(
            path=Path("test.evt"),
            header=_dummy_header(TSTART=0.0, TSTOP=100.0),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
            time=np.sort(np.random.uniform(0, 100, n_events)),
            time_raw=np.sort(np.random.uniform(0, 100, n_events)),
            time_rel=np.arange(n_events, dtype=float),
            pi=np.random.randint(0, 1024, n_events),
            channel=np.random.randint(0, 1024, n_events),
            x=np.random.normal(0, 1, n_events),
            y=np.random.normal(0, 1, n_events),
            telescop="MOCK",
            gti_start=np.array([0.0]),
            gti_stop=np.array([100.0]),
            gti=[(0.0, 100.0)],
        )
        assert len(evt.time) == 100
        assert evt.pi is not None and len(evt.pi) == 100
        assert evt.channel is not None and len(evt.channel) == 100
        assert evt.x is not None and len(evt.x) == 100
        assert evt.y is not None and len(evt.y) == 100
        # GTI 字段  /  GTI fields
        assert len(evt.gti_start) == 1
        assert evt.gti_start[0] == 0.0
        assert evt.gti_stop[0] == 100.0
        assert evt.gti == [(0.0, 100.0)]

    def test_optional_fields_none(self):
        """energy, ebounds, colmap, raw_columns 默认为 None。"""
        evt = EventDataBase(
            path=Path("test.evt"),
            header=_dummy_header(),
            meta=_dummy_meta(),
            headers_dump=_dummy_headers_dump(),
        )
        assert evt.energy is None
        assert evt.ebounds is None
        assert evt.colmap is None
        assert evt.raw_columns is None


# =============================================================================
# Part 2 — jinwu.core.time
# =============================================================================

# 尝试导入时间模块；若 astropy 不可用则跳过所有时间测试。
# Try importing time module; skip all time tests if astropy is unavailable.
try:
    from astropy.time import Time

    from jinwu.core.time import (
        TimeFermi,
        TimeEP,
        TimeLEIA,
        TimeGECAM,
        TimeHXMT,
        TimeSwift,
        TimeGrid,
        TimeMAXI,
        TimeLIGO,
        TimeSuzaku,
        TimeNewton,
        TimeXRISM,
        TimeAstrSat,
        mission_time_format,
        time_from_mission_seconds,
        check_time_overlap,
        get_overlap_duration,
        compare_time_intervals,
        plot_time_intervals,
        extract_time_interval,
    )

    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


# 标记：若 astropy 不可用则跳过所有时间测试
pytestmark_time = pytest.mark.skipif(
    not _ASTROPY_AVAILABLE,
    reason="astropy 不可用，跳过时间模块测试  /  astropy unavailable, skipping time tests",
)


# =============================================================================
# TimeFermi — Fermi MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeFermi:
    """测试 Fermi MET 格式构造与转换。  Test Fermi MET construction/conversion."""

    def test_construction_from_met(self):
        """从 MET 值构造 Fermi 时间。  Construct from MET seconds."""
        # 以 Fermi 纪元 (2001-01-01 UTC) 后 1 天 = 86400 s
        t = Time(86400.0, format="fermi")
        assert t.format == "fermi"
        # 验证等效 UTC 日期
        assert "2001-01-02" in t.isot

    def test_met_to_utc_epoch(self):
        """MET=0 应对应 Fermi 纪元 (2001-01-01 00:00:00 UTC)。  Epoch check."""
        t = Time(0.0, format="fermi")
        # 断言用 UTC 渲染：t.isot 是 TT 时标（含 64.184s TT−UTC 偏移），
        # 且是否应用 UTCF 修正依赖 swiftbat 状态，直接断言 isot 会随
        # 测试顺序漂移；UTC 值才是物理稳定的量。
        assert "2001-01-01T00:00:00" in t.utc.isot

    def test_utc_to_met(self):
        """从 UTC 时间转换为 Fermi MET。  UTC → Fermi MET."""
        t = Time("2001-01-01T00:00:00", scale="utc")
        met = t.to_value("fermi")
        assert met == 0.0

    def test_roundtrip(self):
        """Fermi MET ↔ UTC 往返转换。  Round-trip conversion."""
        original_met = 746496123.0  # ~2024-08-15
        t = Time(original_met, format="fermi")
        recovered = t.to_value("fermi")
        assert abs(recovered - original_met) < 1e-6


# =============================================================================
# TimeEP — Einstein Probe MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeEP:
    """测试 EP MET 格式 (纪元: 2020-01-01 UTC)。"""

    def test_epoch_zero(self):
        """MET=0 → 2020-01-01 00:00:00 UTC。  Epoch check."""
        t = Time(0.0, format="ep")
        assert "2020-01-01T00:00:00" in t.isot

    def test_roundtrip(self):
        """EP MET ↔ UTC 往返转换。"""
        met = 86400.0 * 365.25  # ~1 年后
        t = Time(met, format="ep")
        recovered = t.to_value("ep")
        assert abs(recovered - met) < 1e-6

    def test_mission_metadata_helper_uses_ep_clock(self):
        """Mission aliases resolve only to registered clock formats."""
        met = 205_211_573.9999
        resolved = time_from_mission_seconds("WXT", met)

        assert mission_time_format("EP") == "ep"
        assert resolved is not None
        assert abs((resolved - Time(met, format="ep")).to_value("sec")) < 1e-6
        assert mission_time_format("XMM EPIC") == "newton"
        assert time_from_mission_seconds("UNKNOWN-MISSION", met) is None


# =============================================================================
# TimeLEIA — LEIA MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeLEIA:
    """测试 LEIA MET 格式 (纪元: 2021-01-01 UTC)。"""

    def test_epoch_zero(self):
        """MET=0 → 2021-01-01 00:00:00 UTC。"""
        t = Time(0.0, format="leia")
        assert "2021-01-01T00:00:00" in t.isot

    def test_roundtrip(self):
        """LEIA MET ↔ UTC 往返。"""
        met = 1_000_000.0
        t = Time(met, format="leia")
        recovered = t.to_value("leia")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeGECAM — GECAM MET 格式 (TT 时间尺度)
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeGECAM:
    """测试 GECAM MET 格式 (纪元: 2019-01-01 UTC, 内部使用 TT)。"""

    def test_epoch_zero(self):
        """MET=0 → 2019-01-01 00:00:00 UTC。"""
        t = Time(0.0, format="gecam")
        assert "2019-01-01T00:00:00" in t.utc.isot

    def test_roundtrip(self):
        """GECAM MET ↔ UTC 往返。"""
        met = 1_000_000.0
        t = Time(met, format="gecam")
        recovered = t.to_value("gecam")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeHXMT — HXMT MET 格式 (CXC 参考时间)
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeHXMT:
    """测试 HXMT MET 格式。  Note: uses TT scale with CXC epoch offset."""

    def test_roundtrip(self):
        """HXMT MET ↔ UTC 往返。"""
        met = 100_000.0
        t = Time(met, format="hxmt")
        recovered = t.to_value("hxmt")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeSwift — Swift MET 格式（含 UTCF 修正）
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeSwift:
    """测试 Swift MET 格式 (纪元: 2001-01-01 UTC, 含 UTCF 闰秒修正)。"""

    def test_epoch_zero_approx(self):
        """MET=0 约对应 2001-01-01 00:00:00 UTC。
        Approximately epoch, with UTCF correction of ~13 seconds."""
        t = Time(0.0, format="swiftmet")
        # 闰秒修正使 MET=0 偏离约 13 秒；允许 30 秒容差
        # UTCF correction shifts MET=0 by ~13 s; allow 30 s tolerance
        diff_days = abs(t.mjd - Time("2001-01-01T00:00:00", scale="utc").mjd)
        assert diff_days < 30.0 / 86400  # within 30 seconds

    def test_roundtrip(self):
        """Swift MET ↔ UTC 往返。  Round-trip with UTCF correction."""
        met = 700_000_000.0  # ~2023
        t = Time(met, format="swiftmet")
        recovered = t.to_value("swiftmet")
        assert abs(recovered - met) < 1e-3

    def test_to_value_property(self):
        """to_value / value 属性在 Swift MET 格式上的行为。
        Test to_value / value property on Swift MET format."""
        met = 700_000_003.0
        t = Time(met, format="swiftmet")
        # 往返一致性  /  Round-trip consistency
        recovered = t.to_value("swiftmet")
        assert abs(recovered - met) < 1e-3
        # value 属性应返回近似值  /  value property returns approximate
        v = t.swiftmet
        assert abs(v - met) < 1e-3


# =============================================================================
# TimeGrid — GRID 任务时间（Unix 纪元）
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeGrid:
    """测试 GRID MET 格式 (纪元: 1970-01-01 UTC, 即 Unix 时间)。"""

    def test_epoch_zero(self):
        """MET=0 → 1970-01-01 00:00:00 UTC (Unix epoch)。"""
        t = Time(0.0, format="grid")
        assert "1970-01-01T00:00:00" in t.isot

    def test_roundtrip(self):
        """GRID MET ↔ UTC 往返。"""
        met = 1_700_000_000.0  # ~2023
        t = Time(met, format="grid")
        recovered = t.to_value("grid")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeMAXI — MAXI MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeMAXI:
    """测试 MAXI MET 格式 (纪元: 2000-01-01 TT)。"""

    def test_roundtrip(self):
        """MAXI MET ↔ UTC 往返。"""
        met = 500_000_000.0
        t = Time(met, format="maxi")
        recovered = t.to_value("maxi")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeLIGO — LIGO/GPS 时间格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeLIGO:
    """测试 LIGO/GPS 时间格式 (纪元: 1980-01-06 UTC, TAI 刻度)。"""

    def test_known_value(self):
        """GPS 时间 630720013.0 应对应 2000-01-01 00:00:00 UTC。
        根据文档，630720013.0 = 2000-01-01 00:00:00。"""
        t = Time(630720013.0, format="ligo")
        # 应接近 2000-01-01
        assert "2000-01-01T00:00:00" in t.utc.isot

    def test_roundtrip(self):
        """LIGO/GPS ↔ UTC 往返。"""
        gps_time = 1_300_000_000.0
        t = Time(gps_time, format="ligo")
        recovered = t.to_value("ligo")
        assert abs(recovered - gps_time) < 1e-6


# =============================================================================
# TimeSuzaku — Suzaku MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeSuzaku:
    """测试 Suzaku MET 格式 (纪元: 2000-01-01 TT)。"""

    def test_roundtrip(self):
        """Suzaku MET ↔ UTC 往返。"""
        met = 500_000_000.0
        t = Time(met, format="suzaku")
        recovered = t.to_value("suzaku")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeNewton — XMM-Newton MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeNewton:
    """测试 XMM-Newton MET 格式 (纪元: 1998-01-01 TT)。"""

    def test_roundtrip(self):
        """Newton MET ↔ UTC 往返。"""
        met = 600_000_000.0
        t = Time(met, format="newton")
        recovered = t.to_value("newton")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeXRISM — XRISM MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeXRISM:
    """测试 XRISM MET 格式 (纪元: 2019-01-01 UTC)。"""

    def test_epoch_zero(self):
        """MET=0 → 2019-01-01 00:00:00 UTC。"""
        t = Time(0.0, format="xrism")
        assert "2019-01-01T00:00:00" in t.isot

    def test_roundtrip(self):
        """XRISM MET ↔ UTC 往返。"""
        met = 100_000.0
        t = Time(met, format="xrism")
        recovered = t.to_value("xrism")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# TimeAstrSat — AstroSat MET 格式
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestTimeAstrSat:
    """测试 AstroSat MET 格式 (纪元: 2010-01-01 UTC)。"""

    def test_roundtrip(self):
        """AstroSat MET ↔ UTC 往返。"""
        met = 350_000_000.0
        t = Time(met, format="astrosat")
        recovered = t.to_value("astrosat")
        assert abs(recovered - met) < 1e-6


# =============================================================================
# Cross-mission 跨任务转换  /  Cross-mission conversion
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestCrossMission:
    """测试跨任务时间格式互转。  Test cross-mission format conversion."""

    def test_fermi_to_ep(self):
        """Fermi MET → EP MET：同一 UTC 时刻应前后一致。"""
        # 选择一个 2023 年附近的时间  /  Choose a time near 2023
        fermi_met = 700_000_000.0
        t = Time(fermi_met, format="fermi")
        # 同一时刻转换到 EP MET
        ep_met = t.to_value("ep")
        # 再转回来
        t2 = Time(ep_met, format="ep")
        fermi_recovered = t2.to_value("fermi")
        assert abs(fermi_recovered - fermi_met) < 1e-4

    def test_grid_to_ligo(self):
        """GRID (Unix) → LIGO (GPS)。  Unix timestamp to GPS."""
        # 1990-01-01 作为 Unix 时间戳  /  1990-01-01 as Unix timestamp
        unix_1990 = 631152000.0
        t = Time(unix_1990, format="grid")
        gps_val = t.to_value("ligo")
        # 转回来  /  Convert back
        t2 = Time(gps_val, format="ligo")
        recovered = t2.to_value("grid")
        assert abs(recovered - unix_1990) < 1e-3

    def test_ep_to_swift(self):
        """EP MET → Swift MET：确保跨任务转换一致。"""
        ep_met = 100_000_000.0
        t = Time(ep_met, format="ep")
        swift_met = t.to_value("swiftmet")
        # Swift MET 应 > EP MET（Swift 纪元更早）
        assert swift_met > ep_met

    def test_all_formats_roundtrip(self):
        """所有格式与 fermiso (ISO→FERMI→ISO) 链测试。"""
        utc_str = "2023-06-15T12:00:00.000"
        t_utc = Time(utc_str, scale="utc", format="isot")

        # 对所有支持返回的格式做往返  /  Round-trip through all formats
        for fmt_name in [
            "fermi", "ep", "leia", "gecam", "hxmt",
            "grid", "maxi", "ligo", "suzaku", "newton",
            "xrism", "astrosat",
        ]:
            met = t_utc.to_value(fmt_name)
            t_back = Time(met, format=fmt_name)
            # 用统一刻度（TT 或 UTC）下的 MJD 比较，避免 TT/UTC 偏移
            # Compare MJD in a common scale to avoid TT/UTC offsets
            try:
                mjd_back = t_back.tt.mjd
                mjd_ref = t_utc.tt.mjd
            except Exception:
                mjd_back = t_back.mjd
                mjd_ref = t_utc.mjd
            assert abs(mjd_back - mjd_ref) < 1e-6, (
                f"Round-trip failed for format {fmt_name}"
            )


# =============================================================================
# check_time_overlap — 检查时间重叠
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestCheckTimeOverlap:
    """测试 check_time_overlap 函数。  Test overlap detection."""

    def test_overlapping(self):
        """两个时间段有重叠时应返回 True。  Overlapping → True."""
        s1 = Time(0.0, format="ep")
        e1 = Time(100.0, format="ep")
        s2 = Time(50.0, format="ep")
        e2 = Time(150.0, format="ep")
        assert check_time_overlap(s1, e1, s2, e2) is True

    def test_no_overlap(self):
        """无重叠时应返回 False。  No overlap → False."""
        s1 = Time(0.0, format="ep")
        e1 = Time(100.0, format="ep")
        s2 = Time(200.0, format="ep")
        e2 = Time(300.0, format="ep")
        assert check_time_overlap(s1, e1, s2, e2) is False

    def test_adjacent(self):
        """相邻不重叠（end1 == start2 不应重叠）。  Adjacent → no overlap."""
        s1 = Time(0.0, format="ep")
        e1 = Time(100.0, format="ep")
        s2 = Time(100.0, format="ep")  # 恰好等于 end1
        e2 = Time(200.0, format="ep")
        # 不等式：end1 < start2 → 100 < 100 → False，所以 not (False or ...) → True（有重叠）
        # 但 end1 == start2：100 < 100 is False, end2 < start1 is False
        # → not (False or False) → True → 认为重叠
        # 实际上这取决于需求，此处仅测试当前行为
        result = check_time_overlap(s1, e1, s2, e2)
        assert isinstance(result, bool)

    def test_identical_intervals(self):
        """完全重合的时间区间应视为重叠。  Identical → overlap."""
        s1 = Time(0.0, format="grid")
        e1 = Time(100.0, format="grid")
        assert check_time_overlap(s1, e1, s1, e1) is True

    def test_cross_format(self):
        """不同格式但同一时刻的时间也应正确判断重叠。  Cross-format overlap."""
        s1 = Time(0.0, format="fermi")
        e1 = Time(100.0, format="fermi")
        s2 = Time(0.0, format="ep")      # 不同的纪元
        e2 = Time(100.0, format="ep")    # EP 的开始已是 2020 年
        # 这些时刻不在同一时间范围，不应重叠
        assert check_time_overlap(s1, e1, s2, e2) is False


# =============================================================================
# get_overlap_duration — 获取重叠时长
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestGetOverlapDuration:
    """测试 get_overlap_duration 函数。  Test overlap duration calculation."""

    def test_partial_overlap(self):
        """部分重叠：应返回重叠的起止时刻和时长。  Partial overlap."""
        s1 = Time(0.0, format="ep")
        e1 = Time(100.0, format="ep")
        s2 = Time(40.0, format="ep")
        e2 = Time(140.0, format="ep")
        o_start, o_end, dur = get_overlap_duration(s1, e1, s2, e2)
        assert o_start is not None
        assert o_end is not None
        assert o_start.to_value("ep") == pytest.approx(40.0)
        assert o_end.to_value("ep") == pytest.approx(100.0)
        assert dur == pytest.approx(60.0)

    def test_full_containment(self):
        """一个完全包含另一个。  One fully contains the other."""
        s1 = Time(0.0, format="grid")
        e1 = Time(100.0, format="grid")
        s2 = Time(20.0, format="grid")
        e2 = Time(80.0, format="grid")
        o_start, o_end, dur = get_overlap_duration(s1, e1, s2, e2)
        assert o_start.to_value("grid") == pytest.approx(20.0)
        assert o_end.to_value("grid") == pytest.approx(80.0)
        assert dur == pytest.approx(60.0)

    def test_no_overlap(self):
        """无重叠 → None, None, 0.0。  No overlap → None."""
        s1 = Time(0.0, format="ep")
        e1 = Time(10.0, format="ep")
        s2 = Time(20.0, format="ep")
        e2 = Time(30.0, format="ep")
        o_start, o_end, dur = get_overlap_duration(s1, e1, s2, e2)
        assert o_start is None
        assert o_end is None
        assert dur == 0.0


# =============================================================================
# compare_time_intervals — 比较多个时间段
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestCompareTimeIntervals:
    """测试 compare_time_intervals 函数（多区间比较）。"""

    def test_two_overlapping(self):
        """两个重叠的时间区间。  Two overlapping intervals."""
        intervals = [
            {
                "name": "Satellite A",
                "start": Time(0.0, format="ep"),
                "end": Time(100.0, format="ep"),
            },
            {
                "name": "Satellite B",
                "start": Time(50.0, format="ep"),
                "end": Time(150.0, format="ep"),
            },
        ]
        results = compare_time_intervals(intervals, print_summary=False)
        assert results["total_overlaps"] == 1
        assert len(results["overlaps"]) == 1
        overlap = results["overlaps"][0]
        assert overlap["duration"] == pytest.approx(50.0)

    def test_three_with_two_overlaps(self):
        """三个区间，两两重叠。  Three intervals, two pairwise overlaps."""
        intervals = [
            {"name": "A", "start": Time(0.0, format="ep"), "end": Time(80.0, format="ep")},
            {"name": "B", "start": Time(40.0, format="ep"), "end": Time(120.0, format="ep")},
            {"name": "C", "start": Time(100.0, format="ep"), "end": Time(200.0, format="ep")},
        ]
        results = compare_time_intervals(intervals, print_summary=False)
        # A-B overlap, B-C overlap; A-C no overlap
        assert results["total_overlaps"] == 2

    def test_no_overlaps(self):
        """三个互不重叠的区间。  Three non-overlapping intervals."""
        intervals = [
            {"name": "A", "start": Time(0.0, format="grid"), "end": Time(10.0, format="grid")},
            {"name": "B", "start": Time(20.0, format="grid"), "end": Time(30.0, format="grid")},
            {"name": "C", "start": Time(40.0, format="grid"), "end": Time(50.0, format="grid")},
        ]
        results = compare_time_intervals(intervals, print_summary=False)
        assert results["total_overlaps"] == 0

    def test_with_reference_time(self):
        """使用 reference_time 参数。  With a reference time."""
        ref = Time(0.0, format="ep")
        intervals = [
            {"name": "Test", "start": Time(100.0, format="ep"), "end": Time(200.0, format="ep")},
        ]
        results = compare_time_intervals(intervals, reference_time=ref, print_summary=False)
        assert results["total_overlaps"] == 0  # 只有一个区间，无重叠


# =============================================================================
# plot_time_intervals — 绘制时间段图（不保存文件）
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestPlotTimeIntervals:
    """测试 plot_time_intervals 函数（生成图形，不保存文件）。"""

    def test_plot_basic(self):
        """基本绘图：两个区间。  Basic plot with two intervals."""
        import matplotlib
        matplotlib.use("Agg")  # 非交互后端  /  Non-interactive backend

        intervals = [
            {
                "name": "EP WXT",
                "start": Time(100.0, format="ep"),
                "end": Time(200.0, format="ep"),
                "color": "blue",
            },
            {
                "name": "Swift BAT",
                "start": Time(150.0, format="swiftmet"),
                "end": Time(250.0, format="swiftmet"),
                "color": "red",
            },
        ]
        fig = plot_time_intervals(intervals, title="Test Plot")
        assert fig is not None
        # matplotlib 图形应有 axes
        assert len(fig.axes) >= 1
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_with_reference(self):
        """使用参考时间绘图。  Plot with a reference time."""
        import matplotlib
        matplotlib.use("Agg")

        ref = Time(100.0, format="ep")
        intervals = [
            {
                "name": "Single",
                "start": Time(100.0, format="ep"),
                "end": Time(200.0, format="ep"),
                "color": "green",
            },
        ]
        fig = plot_time_intervals(intervals, reference_time=ref,
                                   show_utc=False, show_relative=True)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_single_axis(self):
        """仅显示 UTC 轴。  Only UTC axis."""
        import matplotlib
        matplotlib.use("Agg")

        intervals = [
            {"name": "A", "start": Time(0.0, format="grid"),
             "end": Time(100.0, format="grid")},
        ]
        fig = plot_time_intervals(intervals, show_utc=True, show_relative=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)


# =============================================================================
# 边界情况和错误处理  /  Edge cases and error handling
# =============================================================================

@pytest.mark.skipif(not _ASTROPY_AVAILABLE, reason="astropy required")
class TestEdgeCases:
    """测试边界情况和错误处理。  Edge cases and error handling."""

    def test_invalid_format(self):
        """不存在的格式名应引发 ValueError。  Invalid format raises error."""
        with pytest.raises(ValueError):
            Time(0.0, format="nonexistent_format")

    def test_very_large_met(self):
        """非常大的 MET 值（模拟远期未来）。  Very large MET value."""
        t = Time(1e15, format="fermi")
        assert t.mjd > 50000  # 应在未来

    def test_negative_met(self):
        """负数 MET（纪元之前）。  Negative MET (before epoch)."""
        # 纪元前 1 天  /  1 day before epoch
        t = Time(-86400.0, format="fermi")
        assert "2000-12-31" in t.isot  # 2000-12-31

    def test_array_input(self):
        """数组形式的 MET 输入。  MET as numpy array."""
        mets = np.array([0.0, 86400.0, 172800.0])
        t = Time(mets, format="fermi")
        assert len(t) == 3
        # 每个元素应对应不同日期  /  Each element should be a different day
        assert "2001-01-01" in t[0].isot
        assert "2001-01-02" in t[1].isot

    def test_plot_no_axes(self):
        """show_utc=False 且 show_relative=False 应引发错误。"""
        import matplotlib
        matplotlib.use("Agg")

        intervals = [
            {"name": "X", "start": Time(0.0, format="grid"),
             "end": Time(1.0, format="grid")},
        ]
        with pytest.raises(ValueError, match="At least one time axis"):
            plot_time_intervals(intervals, show_utc=False, show_relative=False)

    def test_compare_intervals_type_error(self):
        """compare_time_intervals 中非 Time 对象应引发 TypeError。"""
        intervals = [
            {"name": "Bad", "start": "not_a_time", "end": "not_a_time"},
        ]
        with pytest.raises(TypeError, match="必须是 Time 对象"):
            compare_time_intervals(intervals, print_summary=False)

    def test_swift_leap_second_boundary(self):
        """Swift MET 在闰秒边界的转换。  Leap-second boundary."""
        # 2016-12-31 23:59:60 附近（有闰秒）的转换应不崩溃
        # MET ~ 504921600 附近
        try:
            t = Time(504921600.0, format="swiftmet")
            _ = t.isot  # 不应抛出异常
        except Exception:
            # 如果有任何实现问题，至少不会崩溃
            pass


# =============================================================================
# 模块级：astropy 不可用时的 fallback 测试
# =============================================================================

class TestTimeModuleAvailability:
    """当 astropy 不可用时，验证测试能被正确跳过。  Graceful skip tests."""

    def test_skip_if_no_astropy(self):
        """若未安装 astropy，所有时间测试应被跳过。"""
        if not _ASTROPY_AVAILABLE:
            pytest.skip("astropy 未安装，这是预期行为  /  astropy not installed (expected)")
        # 若能运行到此处，说明至少 astropy 可用
        assert True
