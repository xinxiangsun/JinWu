"""
Tests for pipeline stage_code_dependencies（AUD-01 回归）
==========================================================

--- 中文说明 ---
阶段缓存的代码指纹依赖各管线声明的 ``stage_code_dependencies``。AUD-01：
WXT 旧实现用 ``Path(__file__).parents[2] / "core"`` 拼路径，monorepo 拆分后
指向不存在的 ``packages/jinwu-ep/src/jinwu/core/``，指纹退化为常量、核心算法
改动永不触发缓存失效。本文件验证：
  Part 1: 各管线（ep.wxt / swift.grb / swift.bat survey / gw）声明的依赖文件都存在
  Part 2: WXT fit 阶段登记完整算法链（含延迟导入的 spectrum_prep/bxa_fit），
          duration 阶段登记 timescale（ops 只是重导出垫片）
  Part 3: 指纹对文件内容变化敏感
  Part 4: 声明了但不存在的依赖必须显式报错（基类不再静默 sha256=None）

--- English ---
Validates that every pipeline stage declares existing code dependencies,
that WXT declares the full algorithm chain, and that the base-class
fingerprint is content-sensitive and fails loudly on missing files.

运行方式 / Run:
    python -m pytest test/test_stage_code_deps.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jinwu.core.pipeline import InstrumentPipeline, PipelineStage
from jinwu.ep.wxt.pipeline import WXTPointingPipeline
from jinwu.swift.bat.survey import BATSurveyPipeline
from jinwu.swift.grb.pipeline import SwiftGRBPipeline

try:
    from jinwu.gw.pipeline import GWPipeline
except ImportError:  # pragma: no cover - jinwu-gw 是独立发行包，可选安装
    GWPipeline = None

_PIPELINE_CLASSES = [WXTPointingPipeline, SwiftGRBPipeline, BATSurveyPipeline]
if GWPipeline is not None:
    _PIPELINE_CLASSES.append(GWPipeline)


# =============================================================================
# Part 1 — 声明的依赖文件必须真实存在
# =============================================================================

@pytest.mark.parametrize("cls", _PIPELINE_CLASSES)
def test_all_declared_code_dependencies_exist(cls):
    """每条管线的每个阶段，其声明的代码依赖文件都必须真实存在（AUD-01）。"""
    for stage in cls.stages:
        deps = cls.stage_code_dependencies(None, stage)
        for path in deps:
            assert path.is_file(), (
                f"{cls.__name__} stage {stage.name!r}: 缺失依赖 {path}"
            )


# =============================================================================
# Part 2 — WXT 声明完整性
# =============================================================================

def test_wxt_fit_stage_declares_full_algorithm_chain():
    """fit 阶段必须登记实际执行路径上的核心模块（含延迟导入）。"""
    deps = {
        stage.name: WXTPointingPipeline.stage_code_dependencies(None, stage)
        for stage in WXTPointingPipeline.stages
    }
    fit_names = {p.name for p in deps["fit"]}
    assert {"fit.py", "spectrum_prep.py", "bxa_fit.py", "config.py"} <= fit_names


def test_wxt_duration_stage_declares_timescale():
    """duration 经 ``...core.ops.txx`` 使用 timescale 的真实算法（ops 只是垫片）。"""
    deps = {
        stage.name: WXTPointingPipeline.stage_code_dependencies(None, stage)
        for stage in WXTPointingPipeline.stages
    }
    duration_names = {p.name for p in deps["duration"]}
    assert "timescale.py" in duration_names
    assert "ops.py" in duration_names


# =============================================================================
# Part 3/4 — 基类指纹行为
# =============================================================================

def test_stage_code_fingerprint_is_content_sensitive(tmp_path):
    """依赖文件内容变化必须改变指纹。"""
    f = tmp_path / "dep.py"
    f.write_text("x = 1\n")
    ns = SimpleNamespace(stage_code_dependencies=lambda stage: (f,))
    stage = PipelineStage("any")
    h1 = InstrumentPipeline._stage_code_fingerprint(ns, stage)
    f.write_text("x = 2\n")
    h2 = InstrumentPipeline._stage_code_fingerprint(ns, stage)
    assert h1 != h2


def test_stage_code_fingerprint_raises_on_missing_dependency(tmp_path):
    """声明了但不存在的依赖必须显式报错，而不是静默写 sha256=None。"""
    missing = tmp_path / "does_not_exist.py"
    ns = SimpleNamespace(stage_code_dependencies=lambda stage: (missing,))
    stage = PipelineStage("any")
    with pytest.raises(RuntimeError, match="声明的代码依赖不存在"):
        InstrumentPipeline._stage_code_fingerprint(ns, stage)
