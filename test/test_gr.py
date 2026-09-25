"""
Tests for jinwu.physics.gr — GeneralRelativity
===============================================

--- 中文说明 ---
本文件验证 GeneralRelativity 的单位处理、解析解与适用边界（AUD-03 回归）：
  Part 1: v 的单位处理 —— Quantity 自动换算、裸数值按 m/s 解释
  Part 2: beta 与洛伦兹因子的解析解对照
  Part 3: 适用边界 —— v >= c 与负速度报错、未设速度读 beta 报错
  Part 4: show_* 公式展示可调用且 LaTeX 转义正确

AUD-03 背景：`g.v = 1.0` 曾触发 `NameError: name 'u' is not defined`，
`beta` 固定返回占位值 0.0，`lorentz_factor` 因缺 numpy 导入不可用。

--- English ---
This file validates unit handling, analytic values and applicability
boundaries of GeneralRelativity (AUD-03 regression: assigning `g.v = 1.0`
used to raise NameError, `beta` returned a placeholder 0.0 and
`lorentz_factor` was unusable due to a missing numpy import).

Run with:
    python -m pytest test/test_gr.py -v
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from jinwu.physics import GeneralRelativity
from jinwu.physics.gr import C_M_S


# =============================================================================
# Part 1 — v 的单位处理
# =============================================================================

def test_v_accepts_bare_number_si():
    """裸数值按 SI (m/s) 解释；AUD-03 回归：此处曾抛 NameError。"""
    g = GeneralRelativity()
    g.v = 1.0
    assert g.v.unit == u.meter / u.second
    assert g.v.value == pytest.approx(1.0)


def test_v_accepts_quantity_and_converts():
    """Quantity 输入自动换算到 m/s（含单位换算 km/s -> m/s）。"""
    g = GeneralRelativity()
    g.v = 150000.0 * (u.kilometer / u.second)
    assert g.v.unit == u.meter / u.second
    assert g.v.value == pytest.approx(1.5e8)


def test_v_roundtrip_preserves_value():
    """setter/getter 往返保持数值。"""
    g = GeneralRelativity()
    g.v = 0.5 * C_M_S * (u.meter / u.second)
    assert g.v.to_value(u.meter / u.second) == pytest.approx(0.5 * C_M_S)


# =============================================================================
# Part 2 — beta 与洛伦兹因子的解析解
# =============================================================================

def test_beta_is_v_over_c():
    """beta = v/c（不再是占位值 0.0）。"""
    g = GeneralRelativity()
    g.v = 0.5 * C_M_S * (u.meter / u.second)
    assert g.beta == pytest.approx(0.5)


def test_lorentz_factor_analytic():
    """beta = 0.5 时 gamma = 1/sqrt(1 - 0.25) = 2/sqrt(3)。"""
    g = GeneralRelativity()
    g.v = 0.5 * C_M_S * (u.meter / u.second)
    assert g.lorentz_factor == pytest.approx(2.0 / np.sqrt(3.0))


def test_small_v_gives_gamma_near_one():
    """低速极限 gamma -> 1。"""
    g = GeneralRelativity()
    g.v = 1.0
    assert g.lorentz_factor == pytest.approx(1.0, abs=1e-17)


def test_time_dilation_and_length_contraction():
    """时间膨胀 gamma*t 与长度收缩 t/gamma。"""
    g = GeneralRelativity()
    g.v = 0.5 * C_M_S * (u.meter / u.second)
    gamma = g.lorentz_factor
    assert g.time_dilation(2.0) == pytest.approx(2.0 * gamma)
    assert g.length_contraction(2.0) == pytest.approx(2.0 / gamma)


# =============================================================================
# Part 3 — 适用边界
# =============================================================================

def test_v_at_or_above_c_raises():
    """v >= c 超出适用边界，必须报错而不是给出无效结果。"""
    g = GeneralRelativity()
    with pytest.raises(ValueError):
        g.v = C_M_S * (u.meter / u.second)
    with pytest.raises(ValueError):
        g.v = 1.1 * C_M_S * (u.meter / u.second)


def test_negative_v_raises():
    g = GeneralRelativity()
    with pytest.raises(ValueError):
        g.v = -1.0
    with pytest.raises(ValueError):
        g.v = -1.0 * (u.meter / u.second)


def test_beta_without_v_raises():
    """未设置速度时读取 beta 必须显式报错。"""
    g = GeneralRelativity()
    with pytest.raises(ValueError):
        _ = g.beta


# =============================================================================
# Part 4 — show_* 公式展示
# =============================================================================

def test_show_helpers_are_callable(capsys):
    """三个 show_* 方法均可调用（纯展示，不依赖 v）。"""
    GeneralRelativity.show_formula("lorentz")
    GeneralRelativity.show_radiation_transform("flux1")
    GeneralRelativity.show_grmhd_equations()
    assert capsys.readouterr().out != ""


def test_show_formula_all(capsys):
    GeneralRelativity.show_formula("all")
    assert capsys.readouterr().out != ""


def test_show_formula_single_backslash_latex():
    """raw string 双反斜杠（r"\\text"）会按 LaTeX 换行渲染，必须为单反斜杠。"""
    import inspect

    source = inspect.getsource(GeneralRelativity.show_formula)
    assert "\\\\text" not in source
    assert "\\text{" in source
