"""
Tutorial-style tests for jinwu.core.utils and jinwu.core.units
================================================================================

--- 中文说明 ---
本文件包含三个部分：
  Part 1: jinwu.core.utils — 信噪比、误差转换、文件解压等工具函数测试
  Part 3: jinwu.core.units — 星等/流量转换、滤光片系统测试

每个测试函数包含中英双语 docstring 和行内注释，可作为 API 教程阅读。
所有测试可直接运行：
    cd /home/xinxiang/research/jinwu && python -m pytest test/test_utils_txx_units.py -v

--- English ---
This file has three parts:
  Part 1: jinwu.core.utils — SNR, error conversion, file decompression utilities
  Part 3: jinwu.core.units — magnitude/flux conversion, filter system

Every test function has bilingual (Chinese+English) docstrings and inline
comments and can be read as an API tutorial.  Run with:
    cd /home/xinxiang/research/jinwu && python -m pytest test/test_utils_txx_units.py -v
"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# =============================================================================
# Part 1 — jinwu.core.utils
# =============================================================================

from jinwu.core.utils import (
    li_ma_snr,
    flux_err_from_log10,
    get_asym_err,
    generate_download_url,
    gunzip,
    extract_all_gz_recursive,
)


# ---------------------------------------------------------------------------
# 1a. li_ma_snr — Li & Ma 信噪比公式 / Li & Ma SNR formula
# ---------------------------------------------------------------------------

class TestSnrLiMa:
    """Test the Li & Ma SNR formula used in gamma-ray astronomy.

    测试伽马射线天文中使用的 Li & Ma 信噪比公式。
    """

    def test_basic_high_snr(self):
        """
        Test li_ma_snr with strong source detection.

        测试强源探测下的 li_ma_snr 计算。

        强源信号 (n_src=100, n_bkg=10, alpha=1) 应产生一个较大的正 SNR。
        / Strong source signal should yield a large positive SNR.
        """
        snr = li_ma_snr(100, 10, 1.0)
        assert isinstance(snr, float), "SNR should be a float"
        assert snr > 0, "SNR should be positive for strong source"
        assert np.isfinite(snr), "SNR should be finite"
        # Li & Ma 17 confirmed: S=100, B=10, alpha=1 gives SNR ~ 9.5
        # Li & Ma Eq.17 验证: S=100, B=10, alpha=1 给出 SNR ≈ 9.5
        assert 8.0 < snr < 12.0, (
            f"Expected SNR ~9.5 for S=100, B=10, alpha=1, got {snr:.2f}"
        )

    def test_alpha_not_one(self):
        """
        Test li_ma_snr with non-unit alpha (unequal area ratio).

        测试非单位 alpha (不等面积比) 下的 li_ma_snr。

        alpha=0.5 表示源区域是背景区域面积的一半。
        / alpha=0.5 means the source region is half the background area.
        """
        snr = li_ma_snr(100, 50, 0.5)
        assert isinstance(snr, float)
        assert snr > 0, "SNR should be positive"

    def test_alpha_equal_one(self):
        """
        Test li_ma_snr with alpha=1 (equal source/background area).

        测试 alpha=1 (源/背景面积相等) 的情况。

        这是最常见的场景，源区域与背景区域面积相同。
        / Most common case where source and background regions have equal area.
        """
        snr = li_ma_snr(50, 5, 1.0)
        assert snr > 0
        # Verify the formula gives reasonable values
        # 验证公式给出合理的值
        assert np.isfinite(snr)

    def test_zero_source_counts(self):
        """
        Test li_ma_snr with zero source counts.

        测试源计数为零的情况。

        With signed=True (default), n_on=0, n_off>0 is a background deficit:
        the signed Li & Ma limit is -sqrt(2 * n_off * ln(1+alpha)).
        """
        import math
        snr = li_ma_snr(0, 10, 1.0)
        expected = -math.sqrt(2 * 10 * math.log(2.0))
        assert snr == pytest.approx(expected)

    def test_zero_background_counts(self):
        """
        Test li_ma_snr with zero background counts.

        测试背景计数为零的情况。

        When n_bkg=0, returns the correct asymptotic limit
        sqrt(2 * n_on * log((1+alpha)/alpha)) instead of inf.
        """
        import math
        snr = li_ma_snr(100, 0, 1.0)
        expected = math.sqrt(2 * 100 * math.log(2.0 / 1.0))
        assert snr == pytest.approx(expected)

    def test_both_zero(self):
        """
        Test li_ma_snr with both source and background zero.

        测试源和背景计数都为零的情况。

        When both are zero, the function returns 0.0 (no information
        → no significance), not inf.
        """
        snr = li_ma_snr(0, 0, 1.0)
        assert snr == 0.0

    def test_weak_source(self):
        """
        Test li_ma_snr with weak source signal (low SNR).

        测试弱源信号 (低 SNR) 的情况。

        源计数仅略高于背景时，SNR 应该较小。
        / When source counts are only slightly above background, SNR should be low.
        """
        snr = li_ma_snr(15, 10, 1.0)
        assert snr > 0, "SNR should be positive even for weak source"
        assert snr < 5.0, f"Weak source should have SNR < 5, got {snr:.2f}"

    def test_background_dominates(self):
        """
        Test that background deficit returns negative significance.

        n_on=5, n_off=100, alpha=1 means ON region is severely below
        background expectation.  With signed=True (the default),
        li_ma_snr returns a negative value ~-10.26 — this is a
        background deficit, not a source detection.
        """
        import math
        snr = li_ma_snr(5, 100, 1.0)
        assert isinstance(snr, float)
        assert snr < 0
        assert snr == pytest.approx(-10.26, rel=0.01)


class TestLiMaSigned:
    """Verify signed Li & Ma behaviour for source-detection use."""

    def test_excess_positive(self):
        """Source excess → positive signed SNR."""
        s = li_ma_snr(100, 10, 1.0, signed=True)
        assert s > 0

    def test_deficit_negative(self):
        """Background deficit → negative signed SNR — not a source trigger."""
        s = li_ma_snr(5, 100, 1.0, signed=True)
        assert s < 0

    def test_equal_zero(self):
        """n_on == alpha * n_off → exactly zero."""
        s = li_ma_snr(50, 50, 1.0, signed=True)
        assert s == 0.0

    def test_unsigned_unchanged(self):
        """signed=False preserves legacy unsigned magnitude."""
        s = li_ma_snr(5, 100, 1.0, signed=False)
        assert s > 0  # magnitude, not sign

    def test_trigger_rejects_deficit(self):
        """A background deficit should never pass a positive threshold."""
        s = li_ma_snr(5, 100, 1.0, signed=True)
        assert s < 7.0  # would not trigger with threshold 7

    def test_n_on_zero_signed_negative(self):
        """n_on=0, n_off>0 → signed Li & Ma returns negative deficit."""
        import math
        s = li_ma_snr(0, 10, 1.0)
        expected = -math.sqrt(2 * 10 * math.log(2.0))
        assert s == pytest.approx(expected)
        assert s < 0


class TestTriggerDeciderTarget:
    """Verify that decide() propagates the target parameter correctly."""

    def test_decide_respects_target(self):
        from jinwu.core.utils import BackgroundSimple, TriggerDecider

        bg = BackgroundSimple(area_ratio=1.0, t_off_ref=1.0, n_off_ref=1)
        # Single bin: n_on=15, n_off=1, alpha=1.0 → SNR ≈ 3.83
        time = np.array([0.0], dtype=float)
        counts = np.array([15.0], dtype=float)
        td = TriggerDecider(time, counts, dt=1.0, bg=bg)

        # With target=5, SNR 3.83 should NOT trigger
        result = td.decide(window=1.0, target=5.0)
        assert result["triggered"] is False

        # With default target=7, SNR 3.83 should NOT trigger either
        result7 = td.decide(window=1.0)
        assert result7["triggered"] is False

        # With target=3, SNR 3.83 SHOULD trigger
        result3 = td.decide(window=1.0, target=3.0)
        assert result3["triggered"] is True


# ---------------------------------------------------------------------------
# 1b. flux_err_from_log10 — 对数误差到线性误差的转换
#     / Convert logarithmic errors to linear
# ---------------------------------------------------------------------------

class TestFluxErrFromLog10:
    """Test flux error conversion from log10 space to linear space.

    测试从 log10 空间到线性空间的通量误差转换。
    """

    def test_symmetric_log_errors(self):
        """
        Test flux_err_from_log10 with symmetric log errors.

        测试对称 log 误差的转换。

        当高低误差相等时，线性误差应基本对称。
        / When upper and lower log errors are equal, linear errors should be nearly symmetric.
        """
        # log10 flux = -12, log errors = ±0.1
        # log10 通量 = -12, log 误差 = ±0.1
        err_low, err_high = flux_err_from_log10(-12.0, 0.1, 0.1)
        assert isinstance(err_low, float)
        assert isinstance(err_high, float)
        assert err_low > 0, "Lower error should be positive"
        assert err_high > 0, "Upper error should be positive"
        # Both should be similar magnitude for symmetric log errors
        # 对于对称 log 误差，两者应量级相近
        ratio = err_high / err_low
        assert 0.5 < ratio < 2.0, (
            f"Symmetric log errors should give similar linear errors, "
            f"got ratio={ratio:.3f}"
        )

    def test_asymmetric_log_errors(self):
        """
        Test flux_err_from_log10 with asymmetric log errors.

        测试非对称 log 误差的转换。

        较大的对数误差应导致相应方向上较大的线性误差。
        / Larger log error should lead to larger linear error in that direction.
        """
        # log10 flux = -13, log_err_low=0.05, log_err_high=0.3
        # log10 通量 = -13, log_err_low=0.05, log_err_high=0.3
        err_low, err_high = flux_err_from_log10(-13.0, 0.05, 0.3)
        assert err_high > err_low, (
            "Larger log error on high side should give larger linear error, "
            f"got err_low={err_low:.2e}, err_high={err_high:.2e}"
        )

    def test_small_log_errors(self):
        """
        Test flux_err_from_log10 with very small log errors (precise measurement).

        测试极小 log 误差的情况 (精密测量)。

        对于小误差，线性误差应近似等于 dF = F * ln(10) * d(logF)。
        / For small errors, linear errors should approximate dF = F * ln(10) * d(logF).
        """
        F = 10.0 ** (-14.0)  # F = 1e-14
        err_low, err_high = flux_err_from_log10(-14.0, 0.001, 0.001)
        # Approx: err ≈ F * ln(10) * dlogF = 1e-14 * 2.3026 * 0.001 ≈ 2.3e-17
        # 近似: err ≈ F * ln(10) * dlogF = 1e-14 * 2.3026 * 0.001 ≈ 2.3e-17
        expected_approx = F * np.log(10) * 0.001
        assert abs(err_low - expected_approx) / expected_approx < 0.1, (
            f"Small-error approximation should hold, "
            f"expected ~{expected_approx:.2e}, got {err_low:.2e}"
        )
        assert abs(err_high - expected_approx) / expected_approx < 0.1

    def test_none_inputs_return_none(self):
        """
        Test flux_err_from_log10 with None inputs returns None, None.

        测试 None 输入返回 None, None。

        当任何输入为 None 时，函数应安全返回 (None, None)。
        / When any input is None, function should safely return (None, None).
        """
        result = flux_err_from_log10(None, 0.1, 0.1)
        assert result == (None, None), f"Should return (None, None), got {result}"

        result = flux_err_from_log10(-12.0, None, 0.1)
        assert result == (None, None)

        result = flux_err_from_log10(-12.0, 0.1, None)
        assert result == (None, None)

    def test_non_numeric_input_returns_none(self):
        """
        Test flux_err_from_log10 with non-numeric inputs returns None.

        测试非数值输入安全返回 None。

        异常处理确保无效输入不会导致程序崩溃。
        / Exception handling ensures invalid inputs don't crash.
        """
        result = flux_err_from_log10("abc", 0.1, 0.1)
        assert result == (None, None), "Should gracefully return None on bad input"


# ---------------------------------------------------------------------------
# 1c. generate_download_url — GBM poshist 文件下载链接
#     / Generate GBM poshist file download URL
# ---------------------------------------------------------------------------

class TestGenerateDownloadUrl:
    """Test generate_download_url for Fermi GBM poshist file URLs.

    测试 Fermi GBM poshist 文件 URL 生成。
    """

    def test_returns_valid_url(self):
        """
        Test generate_download_url returns a valid URL string.

        测试 generate_download_url 返回有效的 URL 字符串。

        传入 astropy Time 对象应生成指向 NASA HEASARC 的 URL。
        / Passing an astropy Time object should generate a URL pointing
        to the NASA HEASARC server.
        """
        from astropy.time import Time

        t = Time("2024-06-15T12:00:00", format="isot")
        url = generate_download_url(t)

        assert isinstance(url, str), "URL should be a string"
        assert url.startswith("https://"), f"URL should start with https://, got {url}"
        assert "heasarc.gsfc.nasa.gov" in url, (
            "URL should point to NASA HEASARC"
        )
        assert "fermi" in url.lower(), "URL should reference Fermi data"
        assert "2024" in url, "URL should contain the year"
        assert "06" in url, "URL should contain the month"
        assert "15" in url, "URL should contain the day"

    def test_different_date(self):
        """
        Test generate_download_url with a different date.

        测试不同日期的 URL 生成。

        不同日期应生成指向不同目录的 URL。
        / Different dates should generate URLs pointing to different directories.
        """
        from astropy.time import Time

        t1 = Time("2023-01-01T00:00:00", format="isot")
        t2 = Time("2025-12-31T23:59:59", format="isot")

        url1 = generate_download_url(t1)
        url2 = generate_download_url(t2)

        assert url1 != url2, "URLs for different dates should differ"
        assert "2023" in url1
        assert "2025" in url2
        assert "01" in url1
        assert "12" in url2


# ---------------------------------------------------------------------------
# 1d. get_asym_err — XSPEC 参数非对称误差提取
#     / Extract asymmetric errors from XSPEC parameters
# ---------------------------------------------------------------------------

class TestGetAsymErr:
    """Test extraction of asymmetric parameter errors (XSPEC-style).

    测试从 XSPEC 参数对象中提取非对称误差。
    """

    def test_basic_extraction(self):
        """
        Test get_asym_err with a mock XSPEC parameter object.

        用模拟 XSPEC 参数对象测试 get_asym_err。

        创建一个具有 .error 和 .values 属性的简单模拟对象，
        验证能正确提取非对称误差。
        / Creates a simple mock with .error and .values attributes to
        verify asymmetric error extraction.
        """
        # Mock an XSPEC parameter: value=1.0, error range [-0.1, +0.2]
        # 模拟 XSPEC 参数: value=1.0, 误差范围 [-0.1, +0.2]
        class MockParam:
            def __init__(self):
                self.values = [1.0]
                self.error = [0.9, 1.2]  # lower bound, upper bound
                self.name = "test_param"

        param = MockParam()
        err_lo, err_hi = get_asym_err(param)
        assert err_lo == pytest.approx(0.1), f"Expected err_lo=0.1, got {err_lo}"
        assert err_hi == pytest.approx(0.2), f"Expected err_hi=0.2, got {err_hi}"

    def test_symmetric_error(self):
        """
        Test get_asym_err with symmetric errors.

        测试对称误差的情况。

        当上下界与中心值距离相等时，返回的误差应相同。
        / When upper/lower bounds are equidistant from central value,
        returned errors should be equal.
        """
        class MockParam:
            def __init__(self):
                self.values = [5.0]
                self.error = [4.7, 5.3]  # ±0.3 symmetric
                self.name = "sym_param"

        param = MockParam()
        err_lo, err_hi = get_asym_err(param)
        assert err_lo == pytest.approx(0.3)
        assert err_hi == pytest.approx(0.3)
        assert err_lo == err_hi, "Symmetric errors should be equal"

    def test_large_error(self):
        """
        Test get_asym_err with large asymmetric error range.

        测试大范围非对称误差的情况。

        验证能正确处理大范围的非对称误差值。
        / Verifies correct handling of large asymmetric error ranges.
        """
        class MockParam:
            def __init__(self):
                self.values = [100.0]
                self.error = [50.0, 200.0]  # -50, +100
                self.name = "wide_param"

        param = MockParam()
        err_lo, err_hi = get_asym_err(param)
        assert err_lo == pytest.approx(50.0)
        assert err_hi == pytest.approx(100.0)

    def test_missing_error_raises(self):
        """
        Test get_asym_err raises RuntimeError on missing error attribute.

        测试缺少 error 属性时抛出 RuntimeError。

        如果模拟对象没有 error 属性，应抛出异常。
        / If mock object lacks error attribute, should raise RuntimeError.
        """
        class BadParam:
            values = [1.0]
            name = "bad"

        with pytest.raises(RuntimeError, match="Error getting asymmetric error"):
            get_asym_err(BadParam())


# ---------------------------------------------------------------------------
# 1e. gunzip & extract_all_gz_recursive — gzip 解压工具
#     / gzip decompression utilities
# ---------------------------------------------------------------------------

class TestGunzip:
    """Test gzip decompression utilities (gunzip / extract_all_gz_recursive).

    测试 gzip 解压工具函数。
    """

    def test_gunzip_single_file(self):
        """
        Test gunzip on a single .gz file.

        测试单个 .gz 文件的解压。

        创建临时 .gz 文件 → 解压 → 验证输出文件存在且内容正确。
        / Create temp .gz file → decompress → verify output exists and matches.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .gz file with known content
            # 创建包含已知内容的 .gz 文件
            original_content = b"Hello, jinwu! This is a test file.\n"
            gz_path = Path(tmpdir) / "test_data.txt.gz"

            with gzip.open(gz_path, "wb") as f:
                f.write(original_content)

            # Decompress
            # 解压
            count = gunzip(tmpdir, remove_gz=True, verbose=False)
            assert count == 1, f"Should decompress 1 file, got {count}"

            # Verify output
            # 验证输出
            output_path = Path(tmpdir) / "test_data.txt"
            assert output_path.exists(), f"Output file {output_path} should exist"
            assert not gz_path.exists(), "Original .gz should be removed"

            content = output_path.read_bytes()
            assert content == original_content, (
                "Decompressed content should match original"
            )

    def test_gunzip_keep_original(self):
        """
        Test gunzip with remove_gz=False to keep the original .gz file.

        测试 remove_gz=False 保留原始 .gz 文件。

        解压后原 .gz 文件应保留不变。
        / After decompression, the original .gz file should remain.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            original_content = b"Keep me!\n"
            gz_path = Path(tmpdir) / "keep_test.dat.gz"

            with gzip.open(gz_path, "wb") as f:
                f.write(original_content)

            count = gunzip(tmpdir, remove_gz=False, verbose=False)
            assert count == 1
            assert gz_path.exists(), "Original .gz should be kept when remove_gz=False"
            output_path = Path(tmpdir) / "keep_test.dat"
            assert output_path.exists(), "Decompressed file should exist"

    def test_extract_all_gz_recursive_nested(self):
        """
        Test extract_all_gz_recursive on nested directories.

        测试嵌套目录中的递归解压。

        在多层子目录中放置 .gz 文件 → rglob 应找到并解压所有文件。
        / Place .gz files in nested subdirectories → rglob should find and
        decompress all.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            # 创建嵌套目录结构
            sub1 = Path(tmpdir) / "sub1"
            sub2 = Path(tmpdir) / "sub1" / "sub2"
            sub1.mkdir()
            sub2.mkdir()

            # Create .gz files in different directories
            # 在不同目录中创建 .gz 文件
            for i, d in enumerate([Path(tmpdir), sub1, sub2]):
                gz_path = d / f"data_{i}.fits.gz"
                with gzip.open(gz_path, "wb") as f:
                    f.write(f"content_{i}".encode())

            count = extract_all_gz_recursive(tmpdir, remove_gz=True, verbose=False)
            assert count == 3, f"Should decompress 3 files recursively, got {count}"

            # Verify all outputs exist
            # 验证所有输出都存在
            for i in range(3):
                for d in [Path(tmpdir), sub1, sub2]:
                    f = d / f"data_{i}.fits"
                    if f.exists():
                        break  # found it
                else:
                    pytest.fail(f"Output data_{i}.fits not found in any directory")

    def test_extract_all_gz_empty_directory(self):
        """
        Test extract_all_gz_recursive on directory with no .gz files.

        测试不含 .gz 文件的目录。

        空目录 (无 .gz 文件) 应返回计数 0 且不报错。
        / Empty directory (no .gz files) should return count 0 without error.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a regular file (not .gz)
            # 创建普通文件 (非 .gz)
            (Path(tmpdir) / "regular.txt").write_text("hello")
            (Path(tmpdir) / "sub").mkdir()

            count = extract_all_gz_recursive(tmpdir, verbose=False)
            assert count == 0, f"Empty dir should return 0, got {count}"

    def test_extract_all_gz_nonexistent_path_raises(self):
        """
        Test extract_all_gz_recursive raises FileNotFoundError on bad path.

        测试不存在的路径抛出 FileNotFoundError。

        传入不存在的路径应抛出友好的错误信息。
        / Passing a nonexistent path should raise a clear error.
        """
        with pytest.raises(FileNotFoundError, match="路径不存在"):
            extract_all_gz_recursive("/nonexistent/path/12345", verbose=False)

    def test_extract_all_gz_not_a_directory_raises(self):
        """
        Test extract_all_gz_recursive raises NotADirectoryError on file path.

        测试传入文件路径 (非目录) 时抛出 NotADirectoryError。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "not_a_dir.txt"
            file_path.write_text("test")
            with pytest.raises(NotADirectoryError, match="不是目录"):
                extract_all_gz_recursive(str(file_path), verbose=False)


# =============================================================================
# Part 2 — jinwu.core.units
# =============================================================================

from astropy import units as u
from astropy import constants as const

from jinwu.core.units import (
    FilterInfo,
    Magnitude,
    magnitude_to_flux,
    InstrumentFilterLibrary,
    TELESCOPES,
    FILTERS,
)


# ---------------------------------------------------------------------------
# 3a. FilterInfo — 滤光片信息数据类
#     / Filter properties dataclass
# ---------------------------------------------------------------------------

class TestFilterInfo:
    """Test the FilterInfo dataclass for photometric filter properties.

    测试光度滤光片属性的 FilterInfo 数据类。
    """

    @pytest.fixture
    def r_band_filter(self):
        """
        Fixture: create a sample R-band filter for testing.

        创建测试用的 R 波段滤光片。
        """
        return FilterInfo(
            name="Test/R_band",
            wavelength=6400 * u.Angstrom,
            weff=1580 * u.Angstrom,
            zero_point_vega=3060 * u.Jy,
            lambda_pivot=6400 * u.Angstrom,
        )

    def test_construction_basic(self, r_band_filter):
        """
        Test basic FilterInfo construction.

        测试 FilterInfo 的基本构造。

        验证所有属性正确存储并可访问。
        / Verify all attributes stored and accessible correctly.
        """
        f = r_band_filter
        assert f.name == "Test/R_band"
        assert f.wavelength.value == pytest.approx(6400)
        assert f.weff.value == pytest.approx(1580)
        assert f.zero_point_vega.value == pytest.approx(3060)
        # Default AB zero-point should be 3631 Jy
        # 默认 AB 零点应为 3631 Jy
        assert f.zero_point_ab.value == pytest.approx(3631)

    def test_default_lambda_pivot(self):
        """
        Test FilterInfo defaults lambda_pivot to wavelength when not provided.

        测试未提供 lambda_pivot 时默认使用 wavelength。

        当不传入 lambda_pivot 时，应自动设为 wavelength。
        / When lambda_pivot is not provided, it should default to wavelength.
        """
        f = FilterInfo(
            name="Simple",
            wavelength=5500 * u.Angstrom,
            weff=890 * u.Angstrom,
            zero_point_vega=3640 * u.Jy,
        )
        assert f.lambda_pivot == f.wavelength, \
            "lambda_pivot should default to wavelength"

    def test_get_zero_point_vega(self, r_band_filter):
        """
        Test get_zero_point() for Vega system.

        测试 Vega 系统的 get_zero_point()。

        system='Vega' 应返回 zero_point_vega 值。
        / system='Vega' should return the zero_point_vega value.
        """
        zp = r_band_filter.get_zero_point('Vega')
        assert zp.value == pytest.approx(3060)
        assert zp.unit == u.Jy

    def test_get_zero_point_ab(self, r_band_filter):
        """
        Test get_zero_point() for AB system.

        测试 AB 系统的 get_zero_point()。

        system='AB' 应返回 zero_point_ab (默认 3631 Jy)。
        / system='AB' should return zero_point_ab (default 3631 Jy).
        """
        zp = r_band_filter.get_zero_point('AB')
        assert zp.value == pytest.approx(3631)
        assert zp.unit == u.Jy

    def test_get_zero_point_invalid_system(self, r_band_filter):
        """
        Test get_zero_point() raises ValueError for invalid system.

        测试无效 photometric system 时 get_zero_point() 抛出 ValueError。
        """
        with pytest.raises(ValueError, match="system must be 'AB' or 'Vega'"):
            r_band_filter.get_zero_point('SDSS')

    def test_wavelength_must_be_quantity(self):
        """
        Test FilterInfo requires wavelength to be an astropy Quantity.

        测试 wavelength 必须为 astropy Quantity。

        传入纯数值应抛出 ValueError。
        / Passing a plain number should raise ValueError.
        """
        with pytest.raises(ValueError, match="wavelength must be an astropy Quantity"):
            FilterInfo(
                name="Bad",
                wavelength=5500,  # no units!
                weff=890 * u.Angstrom,
                zero_point_vega=3640 * u.Jy,
            )

    def test_str_representation(self, r_band_filter):
        """
        Test FilterInfo string representation.

        测试 FilterInfo 的字符串表示。
        """
        s = str(r_band_filter)
        assert "Test/R_band" in s
        assert "λ=" in s

    def test_mul_operator_creates_magnitude(self, r_band_filter):
        """
        Test FilterInfo * float creates a Magnitude object.

        测试 FilterInfo * float 创建 Magnitude 对象。

        filter * 20.5 应生成一个 Vega 系统的 Magnitude。
        / filter * 20.5 should create a Magnitude with Vega system.
        """
        mag = r_band_filter * 20.5
        assert isinstance(mag, Magnitude)
        assert mag.magnitude == pytest.approx(20.5)
        assert mag.system == 'Vega'
        assert mag.filter_info == r_band_filter

    def test_rmul_operator(self, r_band_filter):
        """
        Test right multiplication (float * FilterInfo) also creates Magnitude.

        测试右乘 (float * FilterInfo) 也创建 Magnitude。

        20.5 * filter 应与 filter * 20.5 等效。
        / 20.5 * filter should be equivalent to filter * 20.5.
        """
        mag = 18.3 * r_band_filter
        assert isinstance(mag, Magnitude)
        assert mag.magnitude == pytest.approx(18.3)
        assert mag.system == 'Vega'


# ---------------------------------------------------------------------------
# 3b. Magnitude — 星等转换类
#     / Magnitude conversion class
# ---------------------------------------------------------------------------

class TestMagnitude:
    """Test Magnitude class for magnitude ↔ flux conversion.

    测试 Magnitude 星等 ↔ 流量转换类。
    """

    @pytest.fixture
    def r_filter(self):
        """Fixture: R-band filter."""
        return FilterInfo(
            name="Test/R",
            wavelength=6400 * u.Angstrom,
            weff=1580 * u.Angstrom,
            zero_point_vega=3060 * u.Jy,
            lambda_pivot=6400 * u.Angstrom,
        )

    def test_construction_basic(self, r_filter):
        """
        Test basic Magnitude construction.

        测试 Magnitude 的基本构造。

        用数值和 FilterInfo 创建 Vega 系统 Magnitude。
        / Create a Vega-system Magnitude with numeric value and FilterInfo.
        """
        mag = Magnitude(20.5, r_filter, system='Vega')
        assert mag.magnitude == pytest.approx(20.5)
        assert mag.system == 'Vega'
        assert mag.filter_info == r_filter

    def test_construction_with_error(self, r_filter):
        """
        Test Magnitude construction with error.

        测试带误差的 Magnitude 构造。
        """
        mag = Magnitude(18.7, r_filter, system='AB', error=0.05)
        assert mag.magnitude == pytest.approx(18.7)
        assert mag.error == pytest.approx(0.05)
        assert mag.system == 'AB'

    def test_to_fnu_vega(self, r_filter):
        """
        Test to_fnu() conversion for Vega system.

        测试 Vega 系统的 to_fnu() 转换。

        f_ν = ZP_Vega × 10^(-m/2.5)
        验证输出的单位和合理的通量值。
        / Verify output units and reasonable flux values.
        """
        mag = Magnitude(20.0, r_filter, system='Vega')
        fnu = mag.to_fnu()

        assert fnu.unit == u.Unit('erg/(cm2 s Hz)') or u.Unit('erg/(cm2 s Hz)').is_equivalent(fnu.unit)
        # fnu should be positive
        # fnu 应为正值
        assert fnu.value > 0, "Flux density should be positive"

        # Manual check: ZP=3060 Jy, m=20 → fnu_Jy = 3060 * 10^(-20/2.5)
        # 手工验证: ZP=3060 Jy, m=20 → fnu_Jy = 3060 * 10^(-20/2.5)
        expected_jy = 3060 * 10 ** (-20.0 / 2.5)
        # Convert to erg/cm2/s/Hz: 1 Jy = 1e-23 erg/cm2/s/Hz
        # 转换为 erg/cm2/s/Hz: 1 Jy = 1e-23 erg/cm2/s/Hz
        expected_fnu = expected_jy * 1e-23
        assert fnu.value == pytest.approx(expected_fnu, rel=1e-6), \
            f"Expected {expected_fnu:.3e}, got {fnu.value:.3e}"

    def test_to_fnu_ab(self, r_filter):
        """
        Test to_fnu() for AB system.

        测试 AB 系统的 to_fnu()。

        AB 零点为 3631 Jy，验证转换结果。
        / AB zero-point is 3631 Jy, verify conversion result.
        """
        mag = Magnitude(20.0, r_filter, system='AB')
        fnu = mag.to_fnu()
        expected_jy = 3631 * 10 ** (-20.0 / 2.5)
        expected_fnu = expected_jy * 1e-23
        assert fnu.value == pytest.approx(expected_fnu, rel=1e-6)

    def test_to_Jy(self, r_filter):
        """
        Test to_Jy() returns flux density in Jansky.

        测试 to_Jy() 返回 Jansky 单位。

        to_Jy() 应等同于 to_fnu(unit='Jy')。
        / to_Jy() should be equivalent to to_fnu(unit='Jy').
        """
        mag = Magnitude(15.0, r_filter, system='Vega')
        fnu_jy = mag.to_Jy()
        assert fnu_jy.unit == u.Jy
        # ZP=3060, m=15
        expected = 3060 * 10 ** (-15.0 / 2.5)
        assert fnu_jy.value == pytest.approx(expected, rel=1e-6)

    def test_to_flam(self, r_filter):
        """
        Test to_flam() conversion to wavelength flux density.

        测试 to_flam() 转换为波长通量密度。

        f_λ = f_ν × c / λ²
        验证转换关系的正确性。
        / Verify the conversion relation f_λ = f_ν × c / λ².
        """
        mag = Magnitude(18.0, r_filter, system='Vega')
        flam = mag.to_flam()

        assert flam.value > 0, "Wavelength flux density should be positive"
        # Default unit is erg/(cm2 s Angstrom)
        # 默认单位是 erg/(cm2 s Angstrom)
        assert 'Angstrom' in str(flam.unit) or 'angstrom' in str(flam.unit), \
            f"Expected Angstrom in unit, got {flam.unit}"

        # Cross-check: f_λ should be consistent with f_ν via c/λ²
        # 交叉验证: f_λ 应与 f_ν 通过 c/λ² 一致
        fnu = mag.to_fnu()
        flam_check = (fnu * const.c / (r_filter.lambda_pivot ** 2)).to('erg/(cm2 s Angstrom)')
        assert flam.value == pytest.approx(flam_check.value, rel=1e-3)

    def test_to_flux(self, r_filter):
        """
        Test to_flux() for integrated flux within filter bandwidth.

        测试 to_flux() 计算滤光片带宽内积分流量。

        F = f_λ × Weff
        验证积分流量的合理性。
        / Verify integrated flux F = f_λ × Weff.
        """
        mag = Magnitude(18.0, r_filter, system='Vega')
        flux = mag.to_flux()

        assert flux.value > 0, "Integrated flux should be positive"
        assert u.Unit('erg/(cm2 s)').is_equivalent(flux.unit)

        # Manual check
        # 手工验证
        flam = mag.to_flam()
        expected = (flam * r_filter.weff).to('erg/(cm2 s)')
        assert flux.value == pytest.approx(expected.value, rel=1e-3)

    def test_fnu_with_error_propagation(self, r_filter):
        """
        Test error propagation in to_fnu().

        测试 to_fnu() 中的误差传播。

        当创建带误差的 Magnitude 时，转换后的通量应携带误差。
        / When Magnitude is created with error, converted flux should
        carry propagated error.
        """
        mag = Magnitude(18.0, r_filter, system='Vega', error=0.1)
        fnu = mag.to_fnu()

        # The error should be attached as fnu.error
        # 误差应附加为 fnu.error
        assert hasattr(fnu, 'error') or fnu.error is not None, \
            "Flux density should have propagated error"
        assert fnu.error.value > 0, "Error should be positive"

    def test_flam_with_error_propagation(self, r_filter):
        """
        Test error propagation in to_flam().

        测试 to_flam() 中的误差传播。
        """
        mag = Magnitude(18.0, r_filter, system='Vega', error=0.1)
        flam = mag.to_flam()

        assert hasattr(flam, 'error') or flam.error is not None, \
            "Wavelength flux density should have propagated error"

    def test_flux_with_error_propagation(self, r_filter):
        """
        Test error propagation in to_flux().

        测试 to_flux() 中的误差传播。
        """
        mag = Magnitude(18.0, r_filter, system='Vega', error=0.1)
        flux = mag.to_flux()

        assert hasattr(flux, 'error') or flux.error is not None, \
            "Integrated flux should have propagated error"

    def test_str_representation(self, r_filter):
        """
        Test Magnitude string representation.

        测试 Magnitude 的字符串表示。
        """
        mag = Magnitude(20.5, r_filter, system='Vega', error=0.1)
        s = str(mag)
        assert "20.50" in s
        assert "0.10" in s
        assert "Test/R" in s
        assert "Vega" in s

    def test_default_system_is_vega(self, r_filter):
        """
        Test Magnitude defaults to Vega system.

        测试 Magnitude 默认使用 Vega 系统。
        """
        mag = Magnitude(15.0, r_filter)
        assert mag.system == 'Vega', \
            "Default photometric system should be Vega"


# ---------------------------------------------------------------------------
# 3c. magnitude_to_flux — 便捷函数
#     / Convenience function
# ---------------------------------------------------------------------------

class TestMagnitudeToFlux:
    """Test the convenience function magnitude_to_flux.

    测试便捷函数 magnitude_to_flux。
    """

    def test_fnu_conversion_with_filter_name(self):
        """
        Test magnitude_to_flux using a predefined filter by name.

        测试使用预定义滤光片名称的 magnitude_to_flux。

        传入字符串滤光片名称 → 通过 FILTERS 字典解析。
        / Pass string filter name → resolved through FILTERS dictionary.
        """
        fnu = magnitude_to_flux(
            20.0, 'Cousins.R', system='Vega',
            flux_type='fnu', error=0.1
        )
        assert fnu.value > 0
        assert u.Unit('erg/(cm2 s Hz)').is_equivalent(fnu.unit)

    def test_flam_conversion(self):
        """
        Test magnitude_to_flux with flux_type='flam'.

        测试 flux_type='flam' 的 magnitude_to_flux。
        """
        flam = magnitude_to_flux(
            18.0, 'Cousins.V', system='Vega',
            flux_type='flam'
        )
        assert flam.value > 0
        assert 'Angstrom' in str(flam.unit) or 'angstrom' in str(flam.unit).lower()

    def test_integrated_flux_F(self):
        """
        Test magnitude_to_flux with flux_type='F' (integrated).

        测试 flux_type='F' (积分流量) 的 magnitude_to_flux。
        """
        F = magnitude_to_flux(
            18.0, 'Cousins.R', system='Vega',
            flux_type='F'
        )
        assert F.value > 0
        assert u.Unit('erg/(cm2 s)').is_equivalent(F.unit)

    def test_ab_system(self):
        """
        Test magnitude_to_flux with AB system.

        测试 AB 系统的 magnitude_to_flux。
        """
        fnu_ab = magnitude_to_flux(
            20.0, 'Cousins.R', system='AB',
            flux_type='fnu'
        )
        assert fnu_ab.value > 0

    def test_unknown_filter_raises(self):
        """
        Test magnitude_to_flux raises ValueError for unknown filter name.

        测试未知滤光片名称抛出 ValueError。
        """
        with pytest.raises(ValueError, match="Unknown filter"):
            magnitude_to_flux(20.0, 'NonExistent/Filter.X', system='Vega')

    def test_invalid_flux_type_raises(self):
        """
        Test magnitude_to_flux raises ValueError for invalid flux_type.

        测试无效 flux_type 抛出 ValueError。
        """
        with pytest.raises(ValueError, match="Unknown flux_type"):
            magnitude_to_flux(20.0, 'Cousins.R', system='Vega', flux_type='invalid')


# ---------------------------------------------------------------------------
# 3d. InstrumentFilterLibrary — 仪器滤光片库
#     / Instrument filter library
# ---------------------------------------------------------------------------

class TestInstrumentFilterLibrary:
    """Test the InstrumentFilterLibrary class.

    测试 InstrumentFilterLibrary 类。
    """

    def test_list_filters(self):
        """
        Test list_filters() returns all filter names.

        测试 list_filters() 返回所有滤光片名称。

        从 NOT/ALFOSC 获取滤光片列表，验证包含已知滤光片。
        / Get filter list from NOT/ALFOSC, verify known filters are present.
        """
        alfosc = TELESCOPES['NOT']['ALFOSC']
        filters = alfosc.list_filters()

        assert isinstance(filters, list)
        assert len(filters) > 0, "ALFOSC should have filters"
        assert 'NOT/ALFOSC.Bes_R' in filters, "ALFOSC should have R-band"
        assert 'NOT/ALFOSC.Bes_V' in filters, "ALFOSC should have V-band"

    def test_getitem_access(self):
        """
        Test __getitem__ bracket access to filters.

        测试 __getitem__ 方括号访问滤光片。
        """
        alfosc = TELESCOPES['NOT']['ALFOSC']
        r_filter = alfosc['NOT/ALFOSC.Bes_R']
        assert isinstance(r_filter, FilterInfo)
        assert r_filter.name == 'NOT/ALFOSC.Bes_R'

    def test_getitem_key_error(self):
        """
        Test __getitem__ raises KeyError for unknown filter.

        测试 __getitem__ 对未知滤光片抛出 KeyError。
        """
        alfosc = TELESCOPES['NOT']['ALFOSC']
        with pytest.raises(KeyError, match="Filter 'BadFilter' not found"):
            _ = alfosc['BadFilter']

    def test_getattr_dot_access(self):
        """
        Test __getattr__ dot-notation access to filters.

        测试 __getattr__ 点号访问滤光片。

        instrument.bes_r 应通过短名称匹配到 NOT/ALFOSC.Bes_R。
        / instrument.bes_r should match NOT/ALFOSC.Bes_R via short name.
        """
        alfosc = TELESCOPES['NOT']['ALFOSC']
        # Access by short name (case-insensitive)
        # 通过短名称访问 (不区分大小写)
        r_filter = alfosc.bes_r
        assert isinstance(r_filter, FilterInfo)
        assert 'Bes_R' in r_filter.name

    def test_getattr_attribute_error(self):
        """
        Test __getattr__ raises AttributeError for unknown filter.

        测试 __getattr__ 对未知滤光片抛出 AttributeError。
        """
        alfosc = TELESCOPES['NOT']['ALFOSC']
        with pytest.raises(AttributeError, match="not found"):
            _ = alfosc.nonexistent_filter_xyz

    def test_str_representation(self):
        """
        Test InstrumentFilterLibrary string representation.

        测试 InstrumentFilterLibrary 的字符串表示。
        """
        alfosc = TELESCOPES['NOT']['ALFOSC']
        s = str(alfosc)
        assert 'NOT/ALFOSC' in s
        assert 'filters' in s.lower()

    def test_telescopes_structure(self):
        """
        Test TELESCOPES dictionary structure.

        测试 TELESCOPES 字典结构。

        验证 TELESCOPES 包含预期的大望远镜和仪器。
        / Verify TELESCOPES contains expected telescopes and instruments.
        """
        assert 'NOT' in TELESCOPES
        assert 'Swift' in TELESCOPES
        assert 'Generic' in TELESCOPES

        assert 'ALFOSC' in TELESCOPES['NOT']
        assert 'UVOT' in TELESCOPES['Swift']
        assert 'Cousins' in TELESCOPES['Generic']

    def test_filters_flat_dict(self):
        """
        Test FILTERS flat dictionary contains all filters.

        测试 FILTERS 扁平字典包含所有滤光片。

        FILTERS 是 TELESCOPES 的扁平化版本，方便直接通过名称访问。
        / FILTERS is a flattened version of TELESCOPES for direct name access.
        """
        assert 'NOT/ALFOSC.Bes_R' in FILTERS
        assert 'Swift/UVOT.white' in FILTERS
        assert 'Cousins.R' in FILTERS
        assert 'SDSS.r' in FILTERS


# =============================================================================
# End of test file
# =============================================================================
