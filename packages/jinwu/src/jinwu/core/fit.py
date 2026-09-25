

"""通用光变曲线拟合模块（Lightcurve Fitting Module）

本模块提供灵活、可扩展的光变曲线拟合框架，支持：
- 多种内置模型（幂律、broken power-law、指数衰减、高斯等）
- 自定义函数表达式
- 统一的拟合接口，接受 LightcurveData 或 LightcurveDataset
- 完整的拟合结果与误差估计

English
-------
General-purpose lightcurve fitting with built-in models (power-law, exponential,
Gaussian, etc.), custom expression support, and unified interface for both
LightcurveData and LightcurveDataset inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Optional, Dict, Any, Literal, Mapping, Union, Sequence
from pathlib import Path
import math
import json
import os
import re
import warnings

import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import LMLSQFitter, TRFLSQFitter

from jinwu.core.data import LightcurveData
from jinwu.core.config import FitConfig, get_fit_settings

if TYPE_CHECKING:
    from jinwu.core.datasets import LightcurveDataset
    from jinwu.core.instruments import Catalog

__all__ = [
    "FitResult",
    "LightcurveFitter",
    "ModelRegistry",
    "XspecChainParameter",
    "XspecChainResult",
    "fit",
    "fit_prepared",
    "fit_spectral",
    "fit_xray_models",
    "resolve_xray_model_specs",
    "calculate_model_fit_metrics",
    "calculate_bayesian_model_metrics",
    "XRayModelSpec",
    "ModelFitMetrics",
    "XRayModelComparisonResult",
    "run_xspec_chain",
]


# ---------- 数据容器 ----------

@dataclass(slots=True)
class FitResult:
    """光变曲线拟合结果容器

    字段
    ----
    model_name : str
        模型名称
    params : np.ndarray
        最优参数值
    param_names : tuple[str, ...]
        参数名称列表
    covariance : np.ndarray | None
        协方差矩阵（若可用）
    errors : np.ndarray | None
        参数 1-sigma 误差（对称情况）
    errors_lower : np.ndarray | None
        参数下误差（非对称情况）
    errors_upper : np.ndarray | None
        参数上误差（非对称情况）
    chisq : float
        卡方值
    dof : int
        自由度
    reduced_chisq : float
        约化卡方
    success : bool
        拟合是否成功
    message : str
        拟合状态信息
    time : np.ndarray
        拟合所用时间数据
    data : np.ndarray
        拟合所用观测值
    data_err : np.ndarray | None
        拟合所用误差
    fitted_curve : np.ndarray
        模型预测值（对应 time）
    residuals : np.ndarray
        残差 (data - fitted_curve)
    model : Callable
        模型评估器，签名为 model(t, *params)，对 astropy 模型进行评估

    English
    -------
    Container for lightcurve fit results including parameters, errors,
    goodness-of-fit statistics, and residuals.
    """
    model_name: str
    params: np.ndarray
    param_names: tuple[str, ...]
    covariance: Optional[np.ndarray]
    errors: Optional[np.ndarray]
    errors_lower: Optional[np.ndarray] = None
    errors_upper: Optional[np.ndarray] = None
    chisq: float = 0.0
    dof: int = 0
    reduced_chisq: float = 0.0
    success: bool = False
    message: str = ""
    time: Optional[np.ndarray] = None
    data: Optional[np.ndarray] = None
    data_err: Optional[np.ndarray] = None
    fitted_curve: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    model: Optional[Callable] = None
    
    def summary(self) -> str:
        """返回拟合结果的文本摘要"""
        lines = [
            f"=== Fit Result: {self.model_name} ===",
            f"Success: {self.success}",
            f"Message: {self.message}",
            f"Chi-squared: {self.chisq:.4f}",
            f"DOF: {self.dof}",
            f"Reduced chi-squared: {self.reduced_chisq:.4f}",
            "",
            "Parameters:",
        ]
        for i, name in enumerate(self.param_names):
            val = self.params[i]
            if self.errors_lower is not None and self.errors_upper is not None:
                # 显示非对称误差
                err_str = f" +{self.errors_upper[i]:.4g} -{self.errors_lower[i]:.4g}"
            elif self.errors is not None:
                # 显示对称误差
                err_str = f" ± {self.errors[i]:.4g}"
            else:
                err_str = ""
            lines.append(f"  {name}: {val:.4g}{err_str}")
        return "\n".join(lines)
    
    def evaluate(self, time: np.ndarray | float, model_func: Optional[Callable] = None) -> np.ndarray | float:
        """在给定时间点评估拟合模型
        
        参数
        ----
        time : array or float
            时间点
        model_func : callable, optional
            模型函数，签名为 model_func(t, *params)
            如果为 None，使用 self.model
        
        返回
        ----
        array or float : 模型预测值
        """
        func = model_func if model_func is not None else self.model
        if func is None:
            raise ValueError("No model function available. Provide model_func or set self.model.")
        
        # 确保输入是数组（模型函数通常期望数组）
        time_array = np.atleast_1d(time)
        result = func(time_array, *self.params)
        
        # 如果输入是标量，返回标量
        if np.isscalar(time):
            return float(result[0]) if result.size > 0 else float(result)
        return result

    def to_dict(self) -> dict:
        """JSON 安全的字典形式，用于结果落盘或结构化上报。"""
        def _optional_array(value: Optional[np.ndarray]) -> list | None:
            return None if value is None else np.asarray(value).tolist()

        return {
            "model_name": self.model_name,
            "success": self.success,
            "message": self.message,
            "param_names": list(self.param_names),
            "params": np.asarray(self.params).tolist(),
            "errors": _optional_array(self.errors),
            "errors_lower": _optional_array(self.errors_lower),
            "errors_upper": _optional_array(self.errors_upper),
            "covariance": _optional_array(self.covariance),
            "chisq": float(self.chisq),
            "dof": int(self.dof),
            "reduced_chisq": float(self.reduced_chisq),
        }


@dataclass(frozen=True, slots=True)
class XRayModelSpec:
    """Stable definition of one XSPEC candidate used in model comparison."""

    key: str
    expression: str
    family: str
    absorption_mode: Literal["free", "zero", "none"]
    critical_parameters: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelFitMetrics:
    """Information criteria for one fit to one fixed dataset.

    ``logz`` / ``logzerr`` carry the Bayesian log-evidence produced by the BXA
    nested-sampling path; they stay ``None`` for maximum-likelihood fits so the
    existing AIC/AICc/BIC surface is unchanged.  ``ranking_metric`` may be
    ``"logz"`` when candidates are ordered by evidence.
    """

    statistic: float
    dof: int
    free_parameters: int
    effective_bins: int
    aic: float
    aicc: float | None
    bic: float
    delta: float | None = None
    delta_aic: float | None = None
    delta_aicc: float | None = None
    delta_bic: float | None = None
    akaike_weight: float | None = None
    logz: float | None = None
    logzerr: float | None = None
    ranking_metric: str = "aicc"


@dataclass(slots=True)
class XRayModelComparisonResult:
    """All candidate fits and the model adopted for downstream products."""

    candidates: dict[str, dict[str, Any]]
    metrics: dict[str, ModelFitMetrics]
    failures: dict[str, str]
    ranking: tuple[str, ...]
    adopted_key: str
    adopted_reason: str
    selection_metric: str
    warnings: tuple[str, ...] = ()
    absorption_comparisons: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparison_json: str | None = None
    comparison_txt: str | None = None

    @property
    def adopted_fit(self) -> dict[str, Any]:
        return self.candidates[self.adopted_key]

    def to_dict(self, *, include_candidates: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "adopted_key": self.adopted_key,
            "adopted_reason": self.adopted_reason,
            "selection_metric": self.selection_metric,
            "ranking": list(self.ranking),
            "metrics": {
                key: {
                    "statistic": value.statistic,
                    "dof": value.dof,
                    "free_parameters": value.free_parameters,
                    "effective_bins": value.effective_bins,
                    "aic": value.aic,
                    "aicc": value.aicc,
                    "bic": value.bic,
                    "delta": value.delta,
                    "delta_aic": value.delta_aic,
                    "delta_aicc": value.delta_aicc,
                    "delta_bic": value.delta_bic,
                    "akaike_weight": value.akaike_weight,
                    "ranking_metric": value.ranking_metric,
                }
                for key, value in self.metrics.items()
            },
            "failures": dict(self.failures),
            "warnings": list(self.warnings),
            "absorption_comparisons": self.absorption_comparisons,
            "comparison_json": self.comparison_json,
            "comparison_txt": self.comparison_txt,
        }
        if include_candidates:
            payload["candidates"] = {
                key: _compact_xray_fit(value) for key, value in self.candidates.items()
            }
        return payload


_XRAY_MODEL_SPECS: tuple[XRayModelSpec, ...] = (
    XRayModelSpec(
        "powerlaw_free_nh", "tbabs*ztbabs*cflux*powerlaw", "powerlaw", "free",
        ("powerlaw.PhoIndex",),
    ),
    XRayModelSpec(
        "powerlaw_nh0", "tbabs*ztbabs*cflux*powerlaw", "powerlaw", "zero",
        ("powerlaw.PhoIndex",),
    ),
    XRayModelSpec("apec", "cflux*apec", "apec", "none", ("apec.kT",)),
    XRayModelSpec(
        "bbody_free_nh", "tbabs*ztbabs*cflux*bbody", "bbody", "free",
        ("bbody.kT",),
    ),
    XRayModelSpec(
        "bbody_nh0", "tbabs*ztbabs*cflux*bbody", "bbody", "zero",
        ("bbody.kT",),
    ),
    XRayModelSpec(
        "bknpower_free_nh", "tbabs*ztbabs*cflux*bknpower", "bknpower", "free",
        ("bknpower.PhoIndx1", "bknpower.BreakE", "bknpower.PhoIndx2"),
    ),
    XRayModelSpec(
        "bknpower_nh0", "tbabs*ztbabs*cflux*bknpower", "bknpower", "zero",
        ("bknpower.PhoIndx1", "bknpower.BreakE", "bknpower.PhoIndx2"),
    ),
)


def resolve_xray_model_specs(
    *,
    model_class: str = "auto",
    absorption_mode: str = "auto",
    candidate_keys: Sequence[str] | None = None,
) -> tuple[XRayModelSpec, ...]:
    """Resolve a deterministic candidate set without importing XSPEC."""

    by_key = {spec.key: spec for spec in _XRAY_MODEL_SPECS}
    if candidate_keys is not None:
        unknown = [key for key in candidate_keys if key not in by_key]
        if unknown:
            raise ValueError(f"Unknown X-ray model candidates: {', '.join(unknown)}")
        if not candidate_keys:
            raise ValueError("candidate_keys cannot be empty")
        return tuple(by_key[key] for key in candidate_keys)

    model_class = str(model_class).lower()
    if model_class not in {"auto", "powerlaw"}:
        raise ValueError("model_class must be 'auto' or 'powerlaw'")
    absorption_mode = str(absorption_mode).lower()
    if absorption_mode not in {"auto", "free", "zero"}:
        raise ValueError("absorption_mode must be 'auto', 'free', or 'zero'")

    allowed_families = {"powerlaw", "bknpower"}
    if model_class == "auto":
        allowed_families.update({"apec", "bbody"})
    selected = []
    for spec in _XRAY_MODEL_SPECS:
        if spec.family not in allowed_families:
            continue
        if spec.absorption_mode == "none" or absorption_mode == "auto":
            selected.append(spec)
        elif spec.absorption_mode == absorption_mode:
            selected.append(spec)
    return tuple(selected)


# 方法：以似然拟合统计量（对 cstat/wstat 即 deviance=-2lnL 加仅依赖数据的常数，模型比较时常数相消）加复杂度惩罚做模型选择；关键式：AIC=stat+2k；AICc=AIC+2k(k+1)/(n-k-1)；BIC=stat+k*ln(n)；其中 k=自由参数数、n=dof+k=拟合 bin 数（与 XSPEC dof=bin数-自由参数数一致）
# 参考：Akaike 1974, IEEE Trans. Autom. Control 19, 716；Burnham & Anderson 2002, Model Selection and Multimodel Inference, 2nd ed., Springer (AICc 校正项)；Schwarz 1978, Ann. Statist. 6, 461；dof 定义见本地 HEASoft 6.37 源码 Xspec/src/XSFit/Fit/StatManager.cxx:702
def calculate_model_fit_metrics(
    statistic: float,
    dof: int,
    free_parameters: int,
) -> ModelFitMetrics:
    """Calculate AIC, AICc, and BIC from one likelihood fit."""

    statistic = float(statistic)
    dof = int(dof)
    free_parameters = int(free_parameters)
    if not math.isfinite(statistic):
        raise ValueError("fit statistic must be finite")
    if dof < 0 or free_parameters < 0:
        raise ValueError("dof and free_parameters must be non-negative")
    n = dof + free_parameters
    if n <= 0:
        raise ValueError("effective bin count must be positive")
    aic = statistic + 2.0 * free_parameters
    aicc = None
    if n > free_parameters + 1:
        aicc = aic + (
            2.0 * free_parameters * (free_parameters + 1)
            / (n - free_parameters - 1)
        )
    bic = statistic + free_parameters * math.log(n)
    return ModelFitMetrics(
        statistic=statistic,
        dof=dof,
        free_parameters=free_parameters,
        effective_bins=n,
        aic=aic,
        aicc=aicc,
        bic=bic,
    )


# 方法：以嵌套采样得到的贝叶斯对数证据 logZ=ln∫L(θ)π(θ)dθ 作为模型排名指标（Poisson 似然与先验下模型的全域支持度），ΔlogZ（含 logzerr 误差棒）替代 AIC/BIC 用于贝叶斯模型比较
# 参考：Buchner et al. 2014, A&A 564, A125 (arXiv:1402.0004, BXA)；Buchner 2021, J. Open Source Softw. 6, 3001 (UltraNest)；Feroz, Hobson & Bridges 2009, MNRAS 398, 1601 (MultiNest/嵌套采样基础)
def calculate_bayesian_model_metrics(
    logz: float,
    logzerr: float | None,
    free_parameters: int,
    dof: int,
) -> ModelFitMetrics:
    """Build :class:`ModelFitMetrics` from a Bayesian log-evidence.

    Used by the BXA nested-sampling path where model comparison ranks by
    ``logz`` rather than AIC/AICc/BIC.  The information criteria are left
    ``None`` because they are undefined without a maximum-likelihood statistic;
    ``statistic`` mirrors ``dof``-agnostic placeholders set to ``nan``-free
    values so downstream serialization stays JSON safe.
    """
    logz_value = float(logz)
    if not math.isfinite(logz_value):
        raise ValueError("logz must be finite")
    logzerr_value = None if logzerr is None else float(logzerr)
    if logzerr_value is not None and (
        not math.isfinite(logzerr_value) or logzerr_value < 0
    ):
        raise ValueError("logzerr must be finite and non-negative")
    free_parameters = int(free_parameters)
    dof = int(dof)
    if free_parameters < 0 or dof < 0:
        raise ValueError("dof and free_parameters must be non-negative")
    return ModelFitMetrics(
        statistic=logz_value,
        dof=dof,
        free_parameters=free_parameters,
        effective_bins=dof + free_parameters,
        aic=None,
        aicc=None,
        bic=None,
        logz=logz_value,
        logzerr=logzerr_value,
        ranking_metric="logz",
    )


# ---------- Astropy 自定义模型类 ----------

class PowerLawModel(Fittable1DModel):
    """幂律模型: norm * (t/t0)^index"""
    norm = Parameter(default=1.0)
    index = Parameter(default=-1.0)
    t0 = Parameter(default=1.0, fixed=True)
    
    @staticmethod
    def evaluate(t, norm, index, t0):
        return norm * np.power(t / t0, index)

class BrokenPowerLawModel(Fittable1DModel):
    """分段幂律模型"""
    norm = Parameter(default=1.0)
    index1 = Parameter(default=-1.0)
    index2 = Parameter(default=-2.0)
    t_break = Parameter(default=100.0)
    
    @staticmethod
    def evaluate(t, norm, index1, index2, t_break):
        result = np.empty_like(t)
        mask1 = t < t_break
        mask2 = ~mask1
        result[mask1] = norm * np.power(t[mask1] / t_break, index1)
        result[mask2] = norm * np.power(t[mask2] / t_break, index2)
        return result

class SmoothlyBrokenPowerLawModel(Fittable1DModel):
    """平滑分段幂律（Willingale 2007 风格）"""
    norm = Parameter(default=1.0)
    index1 = Parameter(default=-1.0)
    index2 = Parameter(default=-2.0)
    t_break = Parameter(default=100.0)
    smoothness = Parameter(default=0.3)
    
    @staticmethod
    def evaluate(t, norm, index1, index2, t_break, smoothness):
        x = t / t_break
        s = smoothness
        term1 = np.power(x, -index1 * s)
        term2 = np.power(x, -index2 * s)
        return norm * np.power(term1 + term2, -1.0 / s)

class DoubleBrokenPowerLawModel(Fittable1DModel):
    """三段幂律（双折断）"""
    norm = Parameter(default=1.0)
    index1 = Parameter(default=0.0)
    index2 = Parameter(default=-2.0)
    index3 = Parameter(default=-1.0)
    t_break1 = Parameter(default=50.0)
    t_break2 = Parameter(default=200.0)
    
    @staticmethod
    def evaluate(t, norm, index1, index2, index3, t_break1, t_break2):
        result = np.empty_like(t)
        mask1 = t < t_break1
        mask2 = (t >= t_break1) & (t < t_break2)
        mask3 = t >= t_break2
        result[mask1] = norm * np.power(t[mask1] / t_break1, index1)
        result[mask2] = norm * np.power(t[mask2] / t_break1, index2)
        norm_late = norm * np.power(t_break2 / t_break1, index2)
        result[mask3] = norm_late * np.power(t[mask3] / t_break2, index3)
        return result

class SmoothlyDoubleBrokenPowerLawModel(Fittable1DModel):
    """平滑三段幂律（两次平滑折断）"""
    norm = Parameter(default=1e-9)
    index1 = Parameter(default=0.0)
    index2 = Parameter(default=-2.0)
    index3 = Parameter(default=-1.0)
    t_break1 = Parameter(default=50.0)
    t_break2 = Parameter(default=200.0)
    smoothness1 = Parameter(default=0.3)
    smoothness2 = Parameter(default=0.3)
    
    @staticmethod
    def evaluate(t, norm, index1, index2, index3, t_break1, t_break2, smoothness1, smoothness2):
        x1 = t / t_break1
        x2 = t / t_break2
        s1 = smoothness1
        s2 = smoothness2
        term1 = np.power(np.power(x1, -index1 * s1) + np.power(x1, -index2 * s1), -1.0 / s1)
        term2 = np.power(np.power(x2, -index2 * s2) + np.power(x2, -index3 * s2), -1.0 / s2)
        return norm * term1 * term2

class ExponentialModel(Fittable1DModel):
    """指数衰减模型"""
    norm = Parameter(default=1.0)
    decay = Parameter(default=0.1)
    t0 = Parameter(default=0.0)
    
    @staticmethod
    def evaluate(t, norm, decay, t0):
        return norm * np.exp(-decay * (t - t0))

class GaussianModel(Fittable1DModel):
    """高斯脉冲模型"""
    amplitude = Parameter(default=1.0)
    mean = Parameter(default=0.0)
    sigma = Parameter(default=1.0)
    
    @staticmethod
    def evaluate(t, amplitude, mean, sigma):
        return amplitude * np.exp(-0.5 * np.power((t - mean) / sigma, 2))

class ConstantModel(Fittable1DModel):
    """常数模型"""
    level = Parameter(default=1.0)
    
    @staticmethod
    def evaluate(t, level):
        return np.full_like(t, level)

class LinearModel(Fittable1DModel):
    """线性模型"""
    slope = Parameter(default=0.0)
    intercept = Parameter(default=0.0)
    
    @staticmethod
    def evaluate(t, slope, intercept):
        return slope * t + intercept


# 纯 astropy 模型实现，删除了函数式模型以统一接口


# ---------- 模型注册表 ----------

class ModelRegistry:
    """模型注册表：管理内置与自定义模型
    
    用法
    ----
    >>> registry = ModelRegistry()
    >>> registry.register("powerlaw", powerlaw, ["norm", "index", "t0"])
    >>> func, names = registry.get("powerlaw")
    """
    
    def __init__(self):
        # 注册表: 名称 -> (AstropyModelClass, param_names)
        self._models: Dict[str, tuple[type[Fittable1DModel], tuple[str, ...]]] = {}
        self._register_builtin()
    
    def _register_builtin(self):
        """注册内置 astropy 模型类"""
        self.register("powerlaw", PowerLawModel, ("norm", "index", "t0"))
        self.register("broken_powerlaw", BrokenPowerLawModel, ("norm", "index1", "index2", "t_break"))
        self.register("double_broken_powerlaw", DoubleBrokenPowerLawModel,
                      ("norm", "index1", "index2", "index3", "t_break1", "t_break2"))
        self.register("smoothly_broken_powerlaw", SmoothlyBrokenPowerLawModel,
                      ("norm", "index1", "index2", "t_break", "smoothness"))
        self.register("smoothly_double_broken_powerlaw", SmoothlyDoubleBrokenPowerLawModel,
                      ("norm", "index1", "index2", "index3", "t_break1", "t_break2", "smoothness1", "smoothness2"))
        self.register("exponential", ExponentialModel, ("norm", "decay", "t0"))
        self.register("gaussian", GaussianModel, ("amplitude", "mean", "sigma"))
        self.register("constant", ConstantModel, ("level",))
        self.register("linear", LinearModel, ("slope", "intercept"))
    
    def register(self, name: str, model_class: type[Fittable1DModel], param_names: tuple[str, ...]):
        """注册新 astropy 模型类"""
        self._models[name] = (model_class, param_names)
    
    def get(self, name: str) -> tuple[type[Fittable1DModel], tuple[str, ...]]:
        """获取 astropy 模型类与参数名"""
        if name not in self._models:
            raise ValueError(f"Unknown model '{name}'. Available: {list(self._models.keys())}")
        return self._models[name]
    
    def list_models(self) -> list[str]:
        """列出所有已注册模型"""
        return list(self._models.keys())


# 全局默认注册表
_default_registry = ModelRegistry()


# ---------- 拟合器主类 ----------

class LightcurveFitter:
    """通用光变曲线拟合器
    
    支持输入类型
    --------------
    - LightcurveData（来自 readfits 或 read_lc）
    - LightcurveDataset（来自 netdata 或手动构造）
    
    使用示例
    --------
    >>> from jinwu import readfits, netdata
    >>> from jinwu.core.fit import LightcurveFitter
    >>> 
    >>> # 方式1：直接拟合 LightcurveData
    >>> lc = readfits("example.lc", kind='lc')
    >>> fitter = LightcurveFitter(lc)
    >>> result = fitter.fit("powerlaw", p0=[1.0, -1.0, 1.0])
    >>> print(result.summary())
    >>> 
    >>> # 方式2：拟合 Dataset（自动处理背景减除）
    >>> src = readfits("source.lc", kind='lc')
    >>> bkg = readfits("background.lc", kind='lc')
    >>> ds = netdata(source=src, background=bkg, label='WXT')
    >>> fitter = LightcurveFitter(ds)
    >>> result = fitter.fit("exponential", p0=[10.0, 0.1, 0.0])
    >>> 
    >>> # 方式3：使用自定义模型
    >>> def my_model(t, a, b, c):
    ...     return a * np.sin(b * t + c)
    >>> fitter.fit(my_model, p0=[1.0, 0.5, 0.0], param_names=["a", "b", "c"])
    """
    
    def __init__(
        self,
        data: Union[
            LightcurveData,
            LightcurveDataset,
            tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],
            tuple[Sequence[float], Sequence[float]],
            tuple[Sequence[float], Sequence[float], Optional[Sequence[float]]],
        ],
        registry: Optional[ModelRegistry] = None,
    ):
        """
        参数
        ----
        data : LightcurveData | LightcurveDataset
            光变曲线数据
        registry : ModelRegistry, optional
            模型注册表（默认使用全局注册表）
        """
        self.registry = registry or _default_registry
        from jinwu.core.datasets import LightcurveDataset
        
        # 统一提取时间、值、误差
        if isinstance(data, LightcurveDataset):
            self._dataset = data
            self.time = data.time
            self.value = data.value
            self.error = data.error
        elif isinstance(data, LightcurveData):
            self._dataset = None
            self.time = data.time
            self.value = data.value
            self.error = data.error
        elif isinstance(data, tuple):
            # 支持 (time, value) 或 (time, value, error) 原始数组输入
            if len(data) == 2:
                t, v = data
                e = None
            elif len(data) == 3:
                t, v, e = data
            else:
                raise TypeError("Tuple data must be (time, value) or (time, value, error)")
            self._dataset = None
            self.time = np.asarray(t, dtype=float)
            self.value = np.asarray(v, dtype=float)
            self.error = None if e is None else np.asarray(e, dtype=float)
        else:
            raise TypeError(
                f"data must be LightcurveData or LightcurveDataset, got {type(data).__name__}"
            )
        
        # 检查数据有效性
        if self.time.size == 0:
            raise ValueError("Empty lightcurve data")
        if self.time.size != self.value.size:
            raise ValueError("Time and value arrays must have the same length")
        if self.error is not None and self.error.size != self.time.size:
            raise ValueError("Error array must match time/value length")
    
    def fit(
        self,
        model: Union[str, Callable],
        p0: Optional[list[float] | np.ndarray] = None,
        param_names: Optional[tuple[str, ...]] = None,
        bounds: Optional[tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]] = None,
        sigma: Optional[np.ndarray] = None,
        absolute_sigma: bool = False,
        fitter_method: Literal["lm", "trf"] = "lm",
        **kwargs,
    ) -> FitResult:
        """执行拟合

        参数
        ----
        model : str | callable
            - 若为 str：从注册表获取模型（如 "powerlaw"）
            - 若为 callable：astropy Fittable1DModel 子类
        p0 : array-like, optional
            初始参数猜测；若为 None 则尝试自动估计
        param_names : tuple[str, ...], optional
            参数名称（仅用于自定义模型类）
        bounds : 2-tuple of array-like, optional
            参数边界 (lower, upper)
        sigma : array, optional
            覆盖数据误差（默认使用 self.error）；内部会复制，不会修改原数组
        absolute_sigma : bool
            是否将 sigma 视为绝对误差（影响协方差缩放）
        fitter_method : {"lm", "trf"}
            'lm' - Levenberg-Marquardt (LMLSQFitter，默认)
            'trf' - Trust Region Reflective (TRFLSQFitter，可选)
        **kwargs : 传递给底层拟合函数的额外参数

        返回
        ----
        FitResult : 拟合结果对象

        注记
        ----
        - 全面使用 astropy.modeling 拟合；支持 LM（默认）与 TRF
        """
        # 解析模型
        if isinstance(model, str):
            model_class, pnames = self.registry.get(model)
            model_name = model
        elif callable(model):
            # 支持用户传入自定义 astropy 模型类或可调用；若为函数，需提供 param_names
            if isinstance(model, type) and issubclass(model, Fittable1DModel):
                model_class = model
                model_name = getattr(model_class, "__name__", "custom_astropy")
                if param_names is None:
                    # 尝试从模型属性推断参数名
                    attrs = [a for a in dir(model_class) if isinstance(getattr(model_class, a), Parameter)]
                    pnames = tuple(attrs)
                else:
                    pnames = param_names
            else:
                raise TypeError("仅支持 astropy Fittable1DModel 子类或注册名称作为模型输入")
        else:
            raise TypeError("model must be str or astropy model class")
        
        # 准备误差
        if sigma is None:
            sigma = self.error if self.error is not None else np.ones_like(self.value)
        # 防止 0 或 NaN 权重导致发散。
        # 注意必须复制：np.asarray 对 ndarray 不拷贝，原地替换非法值
        # 会污染调用者的 LightcurveData.error，导致二次拟合结果错误。
        if sigma is not None:
            sigma = np.array(sigma, dtype=float, copy=True)
            # 用数据的 10% 或极小值替换非法/非正误差
            bad = ~np.isfinite(sigma) | (sigma <= 0)
            if np.any(bad):
                fallback = 0.1 * np.maximum(np.abs(self.value), np.finfo(float).eps)
                sigma[bad] = fallback[bad] if fallback.shape == sigma.shape else np.nan_to_num(fallback, nan=1.0)
        
        # 初值估计
        if p0 is None:
            p0 = self._guess_initial_params(model_name, pnames)
        p0 = np.asarray(p0, dtype=float)
        
        # 边界估计：如果未提供，使用智能默认边界
        if bounds is None:
            bounds = self._get_default_bounds(model_name, pnames)
        
        # 统一使用 astropy 拟合
        return self._fit_astropy(
            model_class, model_name, pnames, p0, sigma, bounds,
            fitter_method, absolute_sigma, **kwargs
        )
    
    # 移除 SciPy 路径
    
    def _fit_astropy(
        self,
        model_class: type[Fittable1DModel],
        model_name: str,
        pnames: tuple[str, ...],
        p0: np.ndarray,
        sigma: np.ndarray,
        bounds: Optional[tuple],
        astropy_method: str = "lm",
        absolute_sigma: bool = False,
        **kwargs,
    ) -> FitResult:
        """使用 astropy.modeling 进行拟合
        
        参数
        ----
        astropy_method : str
            'lm' - Levenberg-Marquardt (LMLSQFitter, 默认)
            'trf' - Trust Region Reflective (TRFLSQFitter，可选)
        """
        try:
            # 创建模型实例并设置初始参数/边界
            model_instance = model_class()
            for i, name in enumerate(pnames):
                setattr(model_instance, name, p0[i])
                if bounds is not None:
                    param = getattr(model_instance, name)
                    min_val = bounds[0][i] if i < len(bounds[0]) else None
                    max_val = bounds[1][i] if i < len(bounds[1]) else None
                    if min_val is not None or max_val is not None:
                        param.bounds = (min_val, max_val)
            
            # 选择拟合器（默认 LM，可切换 TRF）
            if astropy_method == "trf":
                fitter = TRFLSQFitter()
            else:
                fitter = LMLSQFitter()
                astropy_method = "lm"
            
            # 执行拟合
            weights = 1.0 / sigma if sigma is not None else None
            fitted_model = fitter(model_instance, self.time, self.value, weights=weights, **kwargs)
            
            # 提取结果
            popt = np.array([getattr(fitted_model, name).value for name in pnames])

            # 计算误差（从协方差矩阵）。astropy 只对自由参数给出协方差，
            # 且顺序与模型参数定义顺序一致；必须按各参数的 fixed 标志
            # 对位填入，固定参数在中间时不能假设自由参数都排在前面。
            if hasattr(fitter, 'fit_info') and 'param_cov' in fitter.fit_info:
                pcov_raw = fitter.fit_info['param_cov']
                if pcov_raw is not None:
                    errors_raw = np.sqrt(np.diag(pcov_raw))
                    free_indices = [
                        i for i, name in enumerate(pnames)
                        if not getattr(fitted_model, name).fixed
                    ]
                    if errors_raw.size < len(pnames) and errors_raw.size == len(free_indices):
                        errors = np.zeros(len(pnames))
                        for free_pos, param_index in enumerate(free_indices):
                            errors[param_index] = errors_raw[free_pos]
                        pcov = np.zeros((len(pnames), len(pnames)))
                        for a, ia in enumerate(free_indices):
                            for b, ib in enumerate(free_indices):
                                pcov[ia, ib] = pcov_raw[a, b]
                    else:
                        errors = errors_raw
                        pcov = pcov_raw
                else:
                    errors = np.zeros(len(pnames))
                    pcov = None
            else:
                errors = np.zeros(len(pnames))
                pcov = None
            
            fitted = fitted_model(self.time)
            residuals = self.value - fitted
            
            # 计算卡方
            # 方法：加权最小二乘 χ²=Σ((data-fitted)/σ)²，dof=N-k，约化卡方=χ²/dof；astropy 拟合 weights=1/σ 即最小化该 χ²（absolute_sigma=False 时协方差按约化卡方缩放）
            # 参考：Bevington & Robinson 2003, Data Reduction and Error Analysis for the Physical Sciences, 3rd ed., McGraw-Hill（最小二乘与协方差缩放惯例）
            if sigma is not None:
                chisq = np.sum((residuals / sigma) ** 2)
            else:
                chisq = np.sum(residuals ** 2)
            
            dof = len(self.time) - len(popt)
            reduced_chisq = chisq / dof if dof > 0 else np.inf
            
            success = True
            message = f"astropy {astropy_method} converged successfully"
            
        except Exception as e:
            warnings.warn(f"astropy fitting failed: {e}")
            popt = p0
            pcov = None
            errors = np.zeros(len(pnames))
            # 失败时用初值评估
            fitted = model_instance(self.time)
            residuals = self.value - fitted
            
            if sigma is not None:
                chisq = np.sum((residuals / sigma) ** 2)
            else:
                chisq = np.sum(residuals ** 2)
            
            dof = len(self.time) - len(p0)
            reduced_chisq = chisq / dof if dof > 0 else np.inf
            success = False
            message = f"astropy fitting error: {str(e)}"
        
        return FitResult(
            model_name=model_name,
            params=popt,
            param_names=pnames,
            covariance=pcov,
            errors=errors,
            errors_lower=None,
            errors_upper=None,
            chisq=chisq,
            dof=dof,
            reduced_chisq=reduced_chisq,
            success=success,
            message=message,
            time=self.time.copy(),
            data=self.value.copy(),
            data_err=sigma.copy() if sigma is not None else None,
            fitted_curve=fitted,
            residuals=residuals,
            # 提供一个评估器，使得 FitResult.evaluate(t, *params) 工作
            model=lambda t, *params: self._evaluate_astropy_model(model_class, pnames, params, t),
        )
    
    # 移除 MCMC 路径

    def _evaluate_astropy_model(self, model_class: type[Fittable1DModel], pnames: tuple[str, ...], params: Sequence[float], t: np.ndarray) -> np.ndarray:
        """辅助: 使用给定参数评估 astropy 模型类"""
        m = model_class()
        for i, name in enumerate(pnames):
            setattr(m, name, params[i])
        return m(t)
    
    def _guess_initial_params(self, model_name: str, param_names: tuple[str, ...]) -> np.ndarray:
        """智能初值猜测启发式
        
        基于数据统计量为各模型提供合理的初值估计
        """
        n = len(param_names)
        p0 = np.ones(n)
        
        # 数据统计量
        t_min, t_max = self.time.min(), self.time.max()
        t_mid = np.median(self.time)
        t_span = t_max - t_min
        v_mean = np.mean(self.value)
        v_max = np.max(self.value)
        v_min = np.min(self.value)
        
        # 估计衰减趋势（用于判断是上升还是下降）
        if len(self.time) > 1:
            # 用前20%和后20%的均值比较
            n_pts = len(self.time)
            n_early = max(1, n_pts // 5)
            early_mean = np.mean(self.value[:n_early])
            late_mean = np.mean(self.value[-n_early:])
            is_declining = early_mean > late_mean
        else:
            is_declining = True
        
        if model_name == "powerlaw":
            # norm, index, t0
            # index: 通常负值表示衰减，正值表示上升
            index_guess = -1.5 if is_declining else 0.5
            p0 = np.array([v_max, index_guess, 1.0])
            
        elif model_name == "broken_powerlaw":
            # norm, index1, index2, t_break
            # 早期较平，后期较陡
            idx1 = -0.5 if is_declining else 0.5
            idx2 = -2.0 if is_declining else 1.0
            p0 = np.array([v_max, idx1, idx2, t_mid])
            
        elif model_name == "double_broken_powerlaw":
            # norm, index1, index2, index3, t_break1, t_break2
            # 典型GRB: 平台期(~0) -> 快衰减(-1.5~-2.5) -> 慢衰减(-1.0)
            t1 = np.percentile(self.time, 25)
            t2 = np.percentile(self.time, 75)
            if is_declining:
                p0 = np.array([v_max, 0.0, -2.0, -1.0, t1, t2])
            else:
                p0 = np.array([v_min, 0.5, 1.5, 0.5, t1, t2])
                
        elif model_name == "smoothly_broken_powerlaw":
            # norm, index1, index2, t_break, smoothness
            idx1 = -0.5 if is_declining else 0.5
            idx2 = -2.0 if is_declining else 1.0
            p0 = np.array([v_max, idx1, idx2, t_mid, 0.5])
            
        elif model_name == "smoothly_double_broken_powerlaw":
            # norm, index1, index2, index3, t_break1, t_break2, smoothness1, smoothness2
            t1 = np.percentile(self.time, 25)
            t2 = np.percentile(self.time, 75)
            if is_declining:
                p0 = np.array([v_max, 0.0, -2.0, -1.0, t1, t2, 0.5, 0.5])
            else:
                p0 = np.array([v_min, 0.5, 1.5, 0.5, t1, t2, 0.5, 0.5])
                
        elif model_name == "exponential":
            # norm, decay, t0
            decay_guess = 2.0 / t_span if t_span > 0 else 0.1
            p0 = np.array([v_max, decay_guess, t_min])
            
        elif model_name == "gaussian":
            # amplitude, mean, sigma
            sigma_guess = t_span / 6.0  # ~3-sigma覆盖
            p0 = np.array([v_max - v_min, t_mid, sigma_guess])
            
        elif model_name == "constant":
            # level
            p0 = np.array([v_mean])
            
        elif model_name == "linear":
            # slope, intercept
            if len(self.time) > 1:
                slope = (self.value[-1] - self.value[0]) / (self.time[-1] - self.time[0])
            else:
                slope = 0.0
            intercept = v_mean - slope * t_mid
            p0 = np.array([slope, intercept])
        
        return p0
    
    def _get_default_bounds(self, model_name: str, param_names: tuple[str, ...]) -> tuple:
        """为各模型生成智能默认边界
        
        参数
        ----
        model_name : str
            模型名称
        param_names : tuple[str, ...]
            参数名称列表
        
        返回
        ----
        bounds : tuple of (lower, upper)
            参数下界和上界的元组
        
        注记
        ----
        幂律指数范围统一设为 [-10, 3]，覆盖常见天体物理情况
        """
        t_min, t_max = self.time.min(), self.time.max()
        v_min, v_max = self.value.min(), self.value.max()
        v_range = v_max - v_min
        t_range = t_max - t_min
        
        # 振幅/归一化的通用边界：从 0 到无穷
        norm_lower = 0.0
        norm_upper = np.inf
        
        # 时间参数的通用边界
        t_lower = max(t_min * 0.1, t_min - t_range)
        t_upper = min(t_max * 10, t_max + t_range)
        
        # 幂律指数边界：[-10, 3] 覆盖绝大多数情况
        index_lower = -10.0
        index_upper = 3.0
        
        if model_name == "powerlaw":
            # norm, index, t0
            lower = [norm_lower, index_lower, 1e-10]
            upper = [norm_upper, index_upper, np.inf]
            
        elif model_name == "broken_powerlaw":
            # norm, index1, index2, t_break
            lower = [norm_lower, index_lower, index_lower, t_min]
            upper = [norm_upper, index_upper, index_upper, t_max]
            
        elif model_name == "double_broken_powerlaw":
            # norm, index1, index2, index3, t_break1, t_break2
            # 确保 t_break2 > t_break1
            t_mid = (t_min + t_max) / 2
            lower = [norm_lower, index_lower, index_lower, index_lower, t_min, t_mid]
            upper = [norm_upper, index_upper, index_upper, index_upper, t_mid, t_max]
            
        elif model_name == "smoothly_broken_powerlaw":
            # norm, index1, index2, t_break, smoothness
            lower = [norm_lower, index_lower, index_lower, t_min, 0.01]
            upper = [norm_upper, index_upper, index_upper, t_max, 5.0]
            
        elif model_name == "smoothly_double_broken_powerlaw":
            # norm, index1, index2, index3, t_break1, t_break2, smoothness1, smoothness2
            t_mid = (t_min + t_max) / 2
            lower = [norm_lower, index_lower, index_lower, index_lower, t_min, t_mid, 0.01, 0.01]
            upper = [norm_upper, index_upper, index_upper, index_upper, t_mid, t_max, 5.0, 5.0]
            
        elif model_name == "exponential":
            # norm, decay, t0
            lower = [0.0, 0.0, t_lower]
            upper = [norm_upper, 100.0 / (t_range if t_range > 0 else 1.0), t_upper]
            
        elif model_name == "gaussian":
            # amplitude, mean, sigma
            lower = [0.0, t_min, 0.0]
            upper = [norm_upper * 2, t_max, t_range * 2]
            
        elif model_name == "constant":
            # level
            lower = [-np.inf]
            upper = [np.inf]
            
        elif model_name == "linear":
            # slope, intercept
            # 允许任意斜率和截距
            lower = [-np.inf, -np.inf]
            upper = [np.inf, np.inf]
            
        else:
            # 未知模型：返回无约束边界
            n = len(param_names)
            lower = [-np.inf] * n
            upper = [np.inf] * n
        
        return (lower, upper)
    
    # 删除对数空间拟合（SciPy）路径，全面改为 astropy
    
    def plot_fit(
        self,
        result: FitResult,
        *,
        model_func: Optional[Callable] = None,
        n_samples: int = 200,
        ax=None,
        show_residuals: bool = True,
        source_name: Optional[str] = None,
        xlabel: str = "Time (s)",
        ylabel: str = "Flux (erg/cm2/s)",
        annotate_params: bool = True,
        annotation_loc: str = "upper right",
        annotation_alpha: float = 0.6,
        # 智能标签放置（可选，默认关闭以保持兼容）
        smart_labels: bool = False,
        show_norm_in_labels: bool = False,
        label_num_candidates: int = 80,
        label_padding_px: int = 3,
        label_min_gap_px: int = 2,
        label_fontsize: int = 10,
        **plot_kwargs,
    ):
        """绘制拟合结果（需要 matplotlib）
        
        参数
        ----
        result : FitResult
            拟合结果
        model_func : callable, optional
            模型函数（若为 None 则尝试从注册表获取）
        n_samples : int
            拟合曲线采样点数
        ax : matplotlib axes, optional
            绘图轴（若为 None 则创建新图）
        show_residuals : bool
            是否在下方子图显示残差
        source_name : str, optional
            源名称（若提供，将添加到标题前）
        xlabel : str
            横轴标签（默认 Time (s)）
        ylabel : str
            纵轴标签（默认 Flux (erg/cm2/s)）
        **plot_kwargs : 传递给 errorbar 的参数

        返回
        ----
        ``(ax1, ax2)``：主图与残差图坐标轴（无残差时 ax2 为 None）；
        figure 经 ``ax1.figure`` 获取。与 master 保持同一返回结构——beta 曾改为
        ``return fig, (ax1, ax2)``，master 风格的 ``ax1, ax2 = plot_fit(...)`` 调用
        会把 Figure 静默解包进 ax1（已修复，0.2.0 回归项）。
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting; install it with `pip install matplotlib`"
            ) from exc
        from jinwu.core.plotstyle import PALETTE, apply_style

        apply_style()
        if model_func is None:
            # 使用结果内的评估器（astropy 统一）
            model_func = result.model

        if ax is None:
            if show_residuals:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                               gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax2 = None
        else:
            ax1 = ax
            fig = ax1.get_figure()
            ax2 = None
        
        # 数据点（对数坐标要求正数，做掩码以避免报错）
        t_arr = np.asarray(result.time)
        y_arr = np.asarray(result.data)
        yerr_arr = np.asarray(result.data_err) if result.data_err is not None else None
        mask_pos = np.isfinite(t_arr) & (t_arr > 0)
        mask_pos &= np.isfinite(y_arr) & (y_arr > 0)
        if yerr_arr is not None:
            mask_pos &= np.isfinite(yerr_arr) & (yerr_arr >= 0)

        t_plot = t_arr[mask_pos] if np.any(mask_pos) else t_arr
        y_plot = y_arr[mask_pos] if np.any(mask_pos) else y_arr
        yerr_plot = (yerr_arr[mask_pos] if (yerr_arr is not None and np.any(mask_pos)) else yerr_arr)

        ax1.errorbar(
            t_plot,
            y_plot,
            yerr=yerr_plot,
            fmt='o',
            color=PALETTE["data"],
            label='Data',
            alpha=0.7,
            **plot_kwargs,
        )
        
        # 拟合曲线（对数均匀采样以匹配对数坐标）
        # 仅在 t>0 的范围内取样
        t_min_pos = np.min(t_plot) if t_plot.size > 0 else np.min(t_arr[t_arr > 0])
        t_max = np.max(t_plot) if t_plot.size > 0 else np.max(t_arr)
        if np.isfinite(t_min_pos) and np.isfinite(t_max) and t_min_pos > 0 and t_max > t_min_pos:
            t_fine = np.logspace(np.log10(t_min_pos), np.log10(t_max), n_samples)
        elif result.time is not None:
            t_fine = np.linspace(np.maximum(1e-12, result.time.min()), result.time.max(), n_samples)
        else:
            t_fine = np.linspace(np.maximum(1e-12, t_arr.min()), t_arr.max(), n_samples)
        y_fine = result.evaluate(t_fine, model_func)
        ax1.plot(t_fine, y_fine, '-', color=PALETTE["model"], label=f'Fit: {result.model_name}', linewidth=2)

        # 坐标轴改为对数
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylabel(ylabel)
        ax1.legend()
        title_core = f'{result.model_name} Fit (χ²/dof = {result.reduced_chisq:.2f})'
        title = f'{source_name} - {title_core}' if source_name else title_core
        ax1.set_title(title)

        # 可选：在图内添加一个不重叠的参数面板（mathtext 上/下标误差）
        if annotate_params and result.params is not None and result.param_names is not None:
            def _map_math_name(name: str) -> str:
                # 映射常用参数到数学符号
                if name == 'norm':
                    return r'\mathrm{norm}'
                if name.startswith('index'):
                    # index, index1, index2, index3 -> \alpha, \alpha_1, ...
                    if name == 'index':
                        return r'\alpha'
                    suffix = name.replace('index', '')
                    return rf'\alpha_{suffix}' if suffix else r'\alpha'
                if name == 't_break':
                    return r't_{\mathrm{break}}'
                if name.startswith('t_break'):
                    # t_break1, t_break2
                    suffix = name.replace('t_break', '')
                    return rf't_{{\mathrm{{break}}{suffix}}}'
                if name == 'smoothness':
                    return r'n'
                if name == 't0':
                    return r't_0'
                return rf'\mathrm{{{name}}}'

            def _format_math(name_math: str, val: float, lo: float | None, up: float | None) -> str:
                # 统一指数：当需要科学计数法时，将值与误差共同提取 10^e
                def _needs_sci(x: float) -> bool:
                    ax = abs(x)
                    if ax == 0 or not np.isfinite(ax):
                        return False
                    e = int(np.floor(np.log10(ax)))
                    return (e <= -3) or (e >= 4)

                use_sci = _needs_sci(val)
                if (lo is not None and up is not None and np.isfinite(lo) and np.isfinite(up) and (lo>0 or up>0)):
                    if use_sci:
                        if val == 0:
                            e = 0
                        else:
                            e = int(np.floor(np.log10(abs(val))))
                        scale = 10.0 ** e
                        m = val / scale
                        um = up / scale
                        lm = lo / scale
                        return f"${name_math} = {m:.3g}^{{+{um:.2g}}}_{{-{lm:.2g}}} \\times 10^{{{e}}}$"
                    else:
                        return f"${name_math} = {val:.3g}^{{+{up:.2g}}}_{{-{lo:.2g}}}$"
                else:
                    if use_sci:
                        if val == 0:
                            return f"${name_math} = 0$"
                        e = int(np.floor(np.log10(abs(val))))
                        scale = 10.0 ** e
                        m = val / scale
                        return f"${name_math} = {m:.3g} \\times 10^{{{e}}}$"
                    return f"${name_math} = {val:.3g}$"

            vals = np.asarray(result.params)
            names = list(result.param_names)
            if result.errors_lower is not None and result.errors_upper is not None:
                lo = np.asarray(result.errors_lower)
                up = np.asarray(result.errors_upper)
            elif result.errors is not None:
                lo = up = np.asarray(result.errors)
            else:
                lo = np.array([None]*len(vals), dtype=object)
                up = np.array([None]*len(vals), dtype=object)

            # 组装每行的 mathtext 文本
            lines = []
            for i, name in enumerate(names):
                mname = _map_math_name(name)
                v = vals[i]
                lo_i = (lo[i] if i < lo.size else None)
                up_i = (up[i] if i < up.size else None)
                lines.append(_format_math(mname, v, lo_i, up_i))
            panel_text = "\n".join(lines)

            # 放置位置
            ha = 'right' if 'right' in annotation_loc else 'left'
            va = 'top' if 'upper' in annotation_loc else 'bottom'
            x = 0.98 if ha == 'right' else 0.02
            y = 0.98 if va == 'top' else 0.02
            ax1.text(
                x, y, panel_text,
                transform=ax1.transAxes,
                ha=ha, va=va,
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=annotation_alpha),
                clip_on=False,
            )
        
        # 智能标签：为幂律类模型添加段内斜率 α_i 与折断时间 t_break_i 标签
        # 规则：
        # - 横坐标固定（斜率标签：各段几何中心；折断标签：恰在折断时间处）
        # - 仅沿纵向搜索多个候选，避免与数据点、误差棒、已放置标签重叠，且不超过轴范围
        # - 默认不展示 norm，仅展示 α 与 t_break（可通过 show_norm_in_labels 控制）
        if smart_labels and result.params is not None and result.param_names is not None:
            try:
                import matplotlib.pyplot as plt  # already imported above, keep local for clarity
                from matplotlib.transforms import Bbox
            except Exception:
                smart_labels = False
            
            if smart_labels:
                fig = ax1.figure
                # 构建障碍物（数据点与误差棒的显示空间包围盒）
                def _build_obstacles(ax, xs, ys, yerrs, pad_px: int):
                    obstacles = []
                    if xs is None or ys is None or len(xs) == 0:
                        return obstacles
                    # 确保已绘制以获得有效度量
                    try:
                        fig.canvas.draw()
                    except Exception:
                        pass
                    # 以像素度量，给每个点一个固定的水平半宽
                    half_w = max(pad_px, 4)
                    for i in range(len(xs)):
                        x = xs[i]
                        y = ys[i]
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue
                        # 误差棒竖向范围
                        if yerrs is None:
                            y_low, y_high = y, y
                        else:
                            err_i = yerrs[i]
                            if np.ndim(err_i) == 0:
                                y_low, y_high = y - err_i, y + err_i
                            else:
                                # 允许 [lo, hi] 非对称
                                if len(err_i) == 2:
                                    y_low, y_high = y - err_i[0], y + err_i[1]
                                else:
                                    y_low, y_high = y, y
                        # 转到显示坐标
                        p_low = ax.transData.transform((x, max(y_low, np.finfo(float).tiny)))
                        p_mid = ax.transData.transform((x, max(y, np.finfo(float).tiny)))
                        p_high = ax.transData.transform((x, max(y_high, np.finfo(float).tiny)))
                        # 垂直包络
                        y0 = min(p_low[1], p_high[1])
                        y1 = max(p_low[1], p_high[1])
                        # 水平给定固定半宽
                        x0 = p_mid[0] - half_w
                        x1 = p_mid[0] + half_w
                        # padding
                        x0 -= pad_px
                        x1 += pad_px
                        y0 -= pad_px
                        y1 += pad_px
                        obstacles.append((x0, y0, x1, y1))
                    return obstacles
                
                def _overlap(a, b):
                    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])
                
                def _area(a):
                    return max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
                
                def _intersect(a, b):
                    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
                    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
                    if x1 <= x0 or y1 <= y0:
                        return (0,0,0,0)
                    return (x0, y0, x1, y1)
                
                def _text_bbox(ax, s: str, xdata: float, ydata: float):
                    # 获取在数据坐标 (xdata,ydata) 放置文本 s 的显示坐标包围盒
                    txt = ax.text(xdata, ydata, s, fontsize=label_fontsize, transform=ax.transData)
                    try:
                        fig.canvas.draw()
                        bb = txt.get_window_extent()
                        bbox = (bb.x0, bb.y0, bb.x1, bb.y1)
                    except Exception:
                        bbox = (0,0,0,0)
                    finally:
                        txt.remove()
                    # 加 padding
                    return (bbox[0]-label_padding_px, bbox[1]-label_padding_px,
                            bbox[2]+label_padding_px, bbox[3]+label_padding_px)
                
                def _score(bbox, obstacles, occupied, axes_win):
                    # 评分：优先无重叠；其次窗口溢出像素最小；再按重叠面积最小
                    # 统计与障碍/已占用重叠
                    overlap_cnt = 0
                    overlap_area = 0.0
                    for ob in obstacles:
                        if _overlap(bbox, ob):
                            overlap_cnt += 1
                            overlap_area += _area(_intersect(bbox, ob))
                    for oc in occupied:
                        if _overlap(bbox, oc):
                            overlap_cnt += 1
                            overlap_area += _area(_intersect(bbox, oc))
                    # 轴窗口外溢像素
                    overflow = 0.0
                    if bbox[0] < axes_win.x0:
                        overflow += (axes_win.x0 - bbox[0])
                    if bbox[2] > axes_win.x1:
                        overflow += (bbox[2] - axes_win.x1)
                    if bbox[1] < axes_win.y0:
                        overflow += (axes_win.y0 - bbox[1])
                    if bbox[3] > axes_win.y1:
                        overflow += (bbox[3] - axes_win.y1)
                    # 返回元组用于排序
                    return (overlap_cnt, overflow, overlap_area)
                
                def _find_best_y(ax, s: str, x_fixed: float, y_candidates: np.ndarray, obstacles, occupied, axes_win):
                    best = None
                    for y in y_candidates:
                        bbox = _text_bbox(ax, s, x_fixed, y)
                        # 与已占用最小间隔要求
                        gap_ok = True
                        for oc in occupied:
                            # 扩张对比框实现最小间隙
                            exp = (oc[0]-label_min_gap_px, oc[1]-label_min_gap_px, oc[2]+label_min_gap_px, oc[3]+label_min_gap_px)
                            if _overlap(bbox, exp):
                                gap_ok = False
                                break
                        if not gap_ok:
                            continue
                        score = _score(bbox, obstacles, occupied, axes_win)
                        if (best is None) or (score < best[0]):
                            best = (score, y, bbox)
                    return best[1:] if best is not None else (None, None)
                
                # 仅使用正值数据构建障碍
                obstacles = _build_obstacles(ax1, t_plot, y_plot, yerr_plot, label_padding_px)
                occupied_bboxes = []
                
                # 当前轴窗口（显示坐标）
                try:
                    fig.canvas.draw()
                except Exception:
                    pass
                axes_win = ax1.get_window_extent()
                
                # y 候选（对数均匀）
                y_min, y_max = ax1.get_ylim()
                y_min = max(y_min, np.finfo(float).tiny)
                if y_max <= y_min:
                    y_max = y_min * 10.0
                y_candidates = np.logspace(np.log10(y_min*1.02), np.log10(y_max/1.02), max(5, label_num_candidates))
                
                # 构造标签文本（仅 α 与 t_break，norm 可选）
                vals = np.asarray(result.params)
                names = list(result.param_names)
                if result.errors_lower is not None and result.errors_upper is not None:
                    lo = np.asarray(result.errors_lower)
                    up = np.asarray(result.errors_upper)
                elif result.errors is not None:
                    lo = up = np.asarray(result.errors)
                else:
                    lo = np.array([None]*len(vals), dtype=object)
                    up = np.array([None]*len(vals), dtype=object)
                
                def _find_param(name):
                    if name in names:
                        i = names.index(name)
                        return vals[i], (lo[i] if i < lo.size else None), (up[i] if i < up.size else None)
                    return None
                
                # 构造模型特异的段与折断
                tmin_seg = np.min(t_plot) if t_plot.size>0 else np.min(t_arr[t_arr>0])
                tmax_seg = np.max(t_plot) if t_plot.size>0 else np.max(t_arr)
                tmin_seg = max(tmin_seg, np.finfo(float).tiny)
                
                def _geom_mean(a, b):
                    a = max(a, np.finfo(float).tiny)
                    b = max(b, np.finfo(float).tiny)
                    return 10**((np.log10(a)+np.log10(b))/2.0)
                
                model_name = result.model_name
                # labels_to_place: list of tuples (text, x_fixed)
                labels_to_place = []
                
                # 内部 mathtext 格式化（不含外层 $...$，用单反斜杠）
                def _fmt(name_math_inner: str, val: float, lo: float | None, up: float | None) -> str:
                    def _needs_sci(x: float) -> bool:
                        ax = abs(x)
                        if ax == 0 or not np.isfinite(ax):
                            return False
                        e = int(np.floor(np.log10(ax)))
                        return (e <= -3) or (e >= 4)
                    use_sci = _needs_sci(val)
                    if (lo is not None and up is not None and np.isfinite(lo) and np.isfinite(up) and (lo>0 or up>0)):
                        if use_sci:
                            e = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
                            scale = 10.0 ** e
                            m, um, lm = val/scale, up/scale, lo/scale
                            return f"${name_math_inner} = {m:.3g}^{{+{um:.2g}}}_{{-{lm:.2g}}} \\times 10^{{{e}}}$"
                        else:
                            return f"${name_math_inner} = {val:.3g}^{{+{up:.2g}}}_{{-{lo:.2g}}}$"
                    else:
                        if use_sci and val != 0:
                            e = int(np.floor(np.log10(abs(val))))
                            m = val / (10.0**e)
                            return f"${name_math_inner} = {m:.3g} \\times 10^{{{e}}}$"
                        return f"${name_math_inner} = {val:.3g}$"
                
                def _alpha_label(idx_name, subscript: str | None = None):
                    # 构造 mathtext：单反斜杠在最终字符串中
                    mname = "\\alpha" if not subscript else f"\\alpha_{{{subscript}}}"
                    p = _find_param(idx_name)
                    if p is None:
                        return None
                    v, lo_i, up_i = p
                    return _fmt(mname, v, lo_i, up_i)
                
                def _tbreak_label(n: int | None = None):
                    if n is None:
                        key = 't_break'
                        sub = '\\mathrm{break}'
                    else:
                        key = f't_break{n}'
                        sub = f'\\mathrm{{break}}{n}'
                    p = _find_param(key)
                    if p is None:
                        return None
                    v, lo_i, up_i = p
                    return _fmt(f't_{{{sub}}}', v, lo_i, up_i)
                
                # Powerlaw: 仅一个斜率 α
                if model_name == 'powerlaw':
                    s = _alpha_label('index')
                    if s is not None:
                        x_fixed = _geom_mean(tmin_seg, tmax_seg)
                        labels_to_place.append((s, x_fixed))
                    if show_norm_in_labels:
                        p = _find_param('norm')
                        if p is not None:
                            s = _fmt(r'\mathrm{norm}', p[0], p[1], p[2])
                            labels_to_place.append((s, _geom_mean(tmin_seg, tmax_seg)))
                
                # Broken powerlaw: 两段 α1, α2 + t_break
                if model_name == 'broken_powerlaw':
                    tb = _find_param('t_break')
                    if tb is not None:
                        tb_val = tb[0]
                        # α1
                        s1 = _alpha_label('index1', '1')
                        if s1 is not None and np.isfinite(tb_val) and tb_val>tmin_seg:
                            labels_to_place.append((s1, _geom_mean(tmin_seg, tb_val)))
                        # α2
                        s2 = _alpha_label('index2', '2')
                        if s2 is not None and np.isfinite(tb_val) and tb_val<tmax_seg:
                            labels_to_place.append((s2, _geom_mean(tb_val, tmax_seg)))
                        # t_break 标签（x 固定为 tb）
                        s_tb = _tbreak_label()
                        if s_tb is not None and np.isfinite(tb_val):
                            labels_to_place.append((s_tb, max(tb_val, np.finfo(float).tiny)))
                    if show_norm_in_labels:
                        p = _find_param('norm')
                        if p is not None:
                            s = _fmt(r'\mathrm{norm}', p[0], p[1], p[2])
                            labels_to_place.append((s, _geom_mean(tmin_seg, tmax_seg)))
                
                # Smoothly broken powerlaw: 两段 α1, α2 + t_break（平滑参数可不显示）
                if model_name == 'smoothly_broken_powerlaw':
                    tb = _find_param('t_break')
                    if tb is not None:
                        tb_val = tb[0]
                        s1 = _alpha_label('index1', '1')
                        if s1 is not None and np.isfinite(tb_val) and tb_val>tmin_seg:
                            labels_to_place.append((s1, _geom_mean(tmin_seg, tb_val)))
                        s2 = _alpha_label('index2', '2')
                        if s2 is not None and np.isfinite(tb_val) and tb_val<tmax_seg:
                            labels_to_place.append((s2, _geom_mean(tb_val, tmax_seg)))
                        s_tb = _tbreak_label()
                        if s_tb is not None and np.isfinite(tb_val):
                            labels_to_place.append((s_tb, max(tb_val, np.finfo(float).tiny)))
                    # 可选显示平滑度
                    # if 'smoothness' in names: ...（按需开启）
                    if show_norm_in_labels:
                        p = _find_param('norm')
                        if p is not None:
                            s = _fmt(r'\mathrm{norm}', p[0], p[1], p[2])
                            labels_to_place.append((s, _geom_mean(tmin_seg, tmax_seg)))
                
                # Double broken powerlaw 及其平滑版: 三段 α1,α2,α3 + t_break1, t_break2
                if model_name in ('double_broken_powerlaw', 'smoothly_double_broken_powerlaw'):
                    tb1 = _find_param('t_break1')
                    tb2 = _find_param('t_break2')
                    tb1_val = tb1[0] if tb1 is not None else None
                    tb2_val = tb2[0] if tb2 is not None else None
                    # α1
                    s1 = _alpha_label('index1', '1')
                    if s1 is not None and tb1_val is not None and np.isfinite(tb1_val) and tb1_val>tmin_seg:
                        labels_to_place.append((s1, _geom_mean(tmin_seg, tb1_val)))
                    # α2
                    s2 = _alpha_label('index2', '2')
                    if s2 is not None and (tb1_val is not None) and (tb2_val is not None) \
                        and np.isfinite(tb1_val) and np.isfinite(tb2_val) and (tb2_val>tb1_val):
                        labels_to_place.append((s2, _geom_mean(tb1_val, tb2_val)))
                    # α3
                    s3 = _alpha_label('index3', '3')
                    if s3 is not None and tb2_val is not None and np.isfinite(tb2_val) and (tmax_seg>tb2_val):
                        labels_to_place.append((s3, _geom_mean(tb2_val, tmax_seg)))
                    # t_break1, t_break2 标签
                    s_tb1 = _tbreak_label(1)
                    if s_tb1 is not None and tb1_val is not None and np.isfinite(tb1_val):
                        labels_to_place.append((s_tb1, max(tb1_val, np.finfo(float).tiny)))
                    s_tb2 = _tbreak_label(2)
                    if s_tb2 is not None and tb2_val is not None and np.isfinite(tb2_val):
                        labels_to_place.append((s_tb2, max(tb2_val, np.finfo(float).tiny)))
                    if show_norm_in_labels:
                        p = _find_param('norm')
                        if p is not None:
                            s = _fmt(r'\mathrm{norm}', p[0], p[1], p[2])
                            labels_to_place.append((s, _geom_mean(tmin_seg, tmax_seg)))
                
                # 分离 α 标签和 t_break 标签进行不同处理
                alpha_labels = []
                tbreak_labels = []
                other_labels = []
                for s, x_fixed in labels_to_place:
                    if '\\alpha' in s:
                        alpha_labels.append((s, x_fixed))
                    elif 'break' in s:
                        tbreak_labels.append((s, x_fixed))
                    else:
                        other_labels.append((s, x_fixed))
                
                # 处理 norm 等其他标签（保持原逻辑）
                for s, x_fixed in other_labels:
                    y_best, bbox_best = _find_best_y(ax1, s, x_fixed, y_candidates, obstacles, occupied_bboxes, axes_win)
                    if y_best is None:
                        y_best = y_candidates[len(y_candidates)//2]
                        bbox_best = _text_bbox(ax1, s, x_fixed, y_best)
                    txt = ax1.text(
                        x_fixed, y_best, s,
                        transform=ax1.transData,
                        fontsize=label_fontsize,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75),
                        clip_on=True,
                    )
                    occupied_bboxes.append(bbox_best)
                
                # 处理 t_break 标签：绘制竖直虚线，标签固定在下方 1/5 纵轴高度
                y_min_log, y_max_log = ax1.get_ylim()
                y_span_log = np.log10(y_max_log) - np.log10(y_min_log)
                y_tbreak_offset = 0.2  # 从底部 20% (1/5) 位置
                y_tbreak = 10**(np.log10(y_min_log) + y_tbreak_offset * y_span_log)
                
                for s, x_fixed in tbreak_labels:
                    # 绘制竖直虚线
                    ax1.axvline(x_fixed, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
                    # 标签放在固定 y 位置
                    bbox_best = _text_bbox(ax1, s, x_fixed, y_tbreak)
                    txt = ax1.text(
                        x_fixed, y_tbreak, s,
                        transform=ax1.transData,
                        fontsize=label_fontsize,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75),
                        clip_on=True,
                    )
                    occupied_bboxes.append(bbox_best)
                
                # 处理 α 标签：放在拟合线附近，距离在 1/8~1/4 纵轴高度之间
                # 对每个 α 标签，在拟合线上下 1/8~1/4 纵轴范围内搜索
                offset_min = 0.125  # 1/8
                offset_max = 0.25   # 1/4
                
                for s, x_fixed in alpha_labels:
                    # 计算拟合线在 x_fixed 处的 y 值（使用 result.evaluate）
                    fit_y = None
                    if result.model is not None:
                        try:
                            fit_y_raw = result.evaluate(x_fixed)
                            # 确保返回标量
                            fit_y = float(np.asarray(fit_y_raw).item()) if np.asarray(fit_y_raw).size == 1 else None
                        except Exception:
                            pass
                    
                    if fit_y is None or not np.isfinite(fit_y) or fit_y <= 0:
                        # 回退：使用全局 y 候选
                        y_best, bbox_best = _find_best_y(ax1, s, x_fixed, y_candidates, obstacles, occupied_bboxes, axes_win)
                        if y_best is None:
                            y_best = y_candidates[len(y_candidates)//2]
                            bbox_best = _text_bbox(ax1, s, x_fixed, y_best)
                    else:
                        # 在拟合线附近搜索：对数空间偏移 ±(1/8~1/4)纵轴高度
                        fit_y_log = np.log10(fit_y)
                        # 生成候选 y：拟合线上下各 offset_min~offset_max 范围
                        offset_range = np.linspace(-offset_max, -offset_min, label_num_candidates//4).tolist() + \
                                       np.linspace(offset_min, offset_max, label_num_candidates//4).tolist()
                        alpha_y_candidates = [10**(fit_y_log + off * y_span_log) for off in offset_range]
                        # 确保在轴范围内
                        alpha_y_candidates = [y for y in alpha_y_candidates if y_min_log <= y <= y_max_log]
                        if not alpha_y_candidates:
                            alpha_y_candidates = [fit_y]
                        
                        y_best, bbox_best = _find_best_y(ax1, s, x_fixed, np.array(alpha_y_candidates), 
                                                         obstacles, occupied_bboxes, axes_win)
                        if y_best is None:
                            # 回退：直接放在拟合线位置
                            y_best = fit_y
                            bbox_best = _text_bbox(ax1, s, x_fixed, y_best)
                    
                    txt = ax1.text(
                        x_fixed, y_best, s,
                        transform=ax1.transData,
                        fontsize=label_fontsize,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75),
                        clip_on=True,
                    )
                    occupied_bboxes.append(bbox_best)
        
        # 残差
        if show_residuals and ax2 is not None:
            ax2.errorbar(
                result.time,
                result.residuals,
                yerr=result.data_err,
                fmt='o',
                color=PALETTE["residual"],
                alpha=0.6,
            )
            ax2.axhline(0, color=PALETTE["reference"], linestyle='--', alpha=0.6)
            # 残差图：横轴用对数，纵轴保留线性以显示正负
            ax2.set_xscale('log')
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel('Residuals')
        elif not show_residuals:
            ax1.set_xlabel(xlabel)

        try:
            plt.tight_layout()
        except (ValueError, RuntimeError) as e:
            # tight_layout 可能在某些情况下失败（特别是智能标签放置后）
            # 静默忽略，布局可能略有偏差但不影响使用
            pass
        # 返回结构与 master 一致：(ax1, ax2)；figure 用 ax1.figure 获取。
        return ax1, ax2


# =============================================================================
# XSPEC 光谱拟合模块
# =============================================================================

def _require_xspec():
    """检查XSPEC是否可用"""
    try:
        import xspec
        return xspec
    except ImportError:
        raise ImportError(
            "xspec (PyXspec) module is required for spectral fitting. "
            "Install HEASOFT with XSPEC and make sure its python/xspec "
            "package is importable in this environment."
        )


@dataclass(slots=True)
class XspecChainParameter:
    """Metadata for a parameter sampled by an XSPEC chain session."""

    index: int
    component: str | None
    name: str
    value: float | None
    frozen: bool
    unit: str | None


@dataclass(slots=True)
class XspecChainResult:
    """Result payload for an XSPEC MCMC chain run."""

    chain_path: str
    fit_statistic: float | None
    fit_dof: int | None
    stat_method: str | None
    free_parameters: list[XspecChainParameter]
    all_parameters: list[XspecChainParameter]
    source_counts: float | None
    background_counts: float | None
    chain_settings: dict
    parallel_settings: dict
    warnings: list[str]
    status: str

    def to_dict(self) -> dict:
        """JSON 安全的字典形式，用于结果落盘或结构化上报。"""
        def _parameter(parameter: XspecChainParameter) -> dict:
            return {
                "index": parameter.index,
                "component": parameter.component,
                "name": parameter.name,
                "value": parameter.value,
                "frozen": parameter.frozen,
                "unit": parameter.unit,
            }

        return {
            "chain_path": self.chain_path,
            "status": self.status,
            "stat_method": self.stat_method,
            "fit_statistic": self.fit_statistic,
            "fit_dof": self.fit_dof,
            "source_counts": self.source_counts,
            "background_counts": self.background_counts,
            "free_parameters": [_parameter(item) for item in self.free_parameters],
            "all_parameters": [_parameter(item) for item in self.all_parameters],
            "chain_settings": dict(self.chain_settings),
            "parallel_settings": dict(self.parallel_settings),
            "warnings": list(self.warnings),
        }


def _default_xspec_parallel_processes(fraction: float = 0.75) -> int:
    """Return the default XSPEC parallel process count for this host."""
    if fraction <= 0:
        raise ValueError("parallel fraction must be greater than 0")

    n_cpu = os.cpu_count() or 1
    return max(1, math.floor(n_cpu * fraction))


def _xspec_chain_models(xspec) -> list[Any]:
    """Return one model copy per active XSPEC source for the first data group."""
    models = []
    sources = getattr(xspec.AllModels, "sources", None)

    if isinstance(sources, dict):
        for _, model_name in sorted(sources.items()):
            try:
                models.append(xspec.AllModels(1, model_name))
            except TypeError:
                models.append(xspec.AllModels(1))
            except Exception:
                continue

    if not models:
        try:
            models.append(xspec.AllModels(1))
        except Exception:
            pass

    return models


def _xspec_parameter_value(param) -> float | None:
    try:
        return float(param.values[0])
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _xspec_chain_parameters(models: Sequence[Any]) -> list[XspecChainParameter]:
    """Collect component-aware parameter metadata from XSPEC model objects."""
    parameters = []

    for model in models:
        for component_name in getattr(model, "componentNames", ()):
            component = getattr(model, component_name)

            for parameter_name in getattr(component, "parameterNames", ()):
                parameter = getattr(component, parameter_name)
                parameters.append(
                    XspecChainParameter(
                        index=int(parameter.index),
                        component=str(component_name),
                        name=str(getattr(parameter, "name", parameter_name)),
                        value=_xspec_parameter_value(parameter),
                        frozen=bool(parameter.frozen),
                        unit=getattr(parameter, "unit", None) or None,
                    )
                )

    return parameters


def _xspec_chain_spectra(xspec, spectra) -> list[Any]:
    if spectra is None:
        return [xspec.AllData(index) for index in range(1, int(xspec.AllData.nSpectra) + 1)]

    if isinstance(spectra, (str, bytes)):
        raise TypeError("spectra must be Spectrum objects or an iterable of Spectrum objects")

    try:
        return list(spectra)
    except TypeError:
        return [spectra]


def _xspec_spectrum_counts(spectra: Sequence[Any], warnings_list: list[str]) -> tuple[float | None, float | None]:
    source_counts = 0.0
    background_counts = 0.0
    source_seen = False
    background_seen = False

    for spectrum in spectra:
        try:
            source_counts += float(np.sum(spectrum.values)) * float(spectrum.exposure)
            source_seen = True
        except (AttributeError, TypeError, ValueError):
            warnings_list.append("Could not calculate source counts for one XSPEC spectrum.")

        try:
            background = spectrum.background
        except Exception:
            background = None

        if background is None:
            continue

        try:
            background_counts += float(np.sum(background.values)) * float(background.exposure)
            background_seen = True
        except (AttributeError, TypeError, ValueError):
            warnings_list.append("Could not calculate background counts for one XSPEC spectrum.")

    return (
        source_counts if source_seen else None,
        background_counts if background_seen else None,
    )


def _xspec_fit_value(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _xspec_fit_dof(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# 方法：调用 XSPEC 内置 MCMC（chain 命令）对当前全部自由参数采样：algorithm='gw' 为 XSPEC 默认的 Goodman-Weare 仿射不变集成采样器（多游走者）、'mh' 为 Metropolis-Hastings；burn/runLength/walkers 默认值与 XSPEC 一致（walkers=10、gw）
# 参考：Goodman & Weare 2010, Commun. Appl. Math. Comput. Sci. 5, 65（GW 集成采样）；XSPEC manual chain 命令；本地 HEASoft 6.37 源码 Xspec/src/XSUser/Handler/xsChain.cxx:41,47（默认 walkers=10、type='gw'）与 636-641（mh=Metropolis-Hastings / gw=Goodman-Weare）
def run_xspec_chain(
    *,
    chain_path: str | Path,
    refit: bool = True,
    chain_length: int = 50000,
    chain_burn: int = 10000,
    chain_algorithm: Literal["gw", "mh"] = "gw",
    chain_walkers: int = 10,
    parallel_fraction: float = 0.75,
    parallel_processes: int | None = None,
    overwrite: bool = False,
    spectra=None,
) -> XspecChainResult:
    """Run an XSPEC MCMC chain for all currently thawed model parameters."""
    xspec = _require_xspec()

    if int(getattr(xspec.AllData, "nSpectra", 0) or 0) < 1:
        raise RuntimeError("run_xspec_chain requires at least one loaded XSPEC spectrum.")

    models = _xspec_chain_models(xspec)
    if not models:
        raise RuntimeError("run_xspec_chain requires a loaded XSPEC model.")

    all_parameters = _xspec_chain_parameters(models)
    free_parameters = [parameter for parameter in all_parameters if not parameter.frozen]
    if not free_parameters:
        raise RuntimeError("run_xspec_chain requires at least one thawed XSPEC parameter.")

    if chain_algorithm not in {"gw", "mh"}:
        raise ValueError("chain_algorithm must be 'gw' or 'mh'.")
    if chain_length < 1:
        raise ValueError("chain_length must be at least 1.")
    if chain_burn < 0:
        raise ValueError("chain_burn must be non-negative.")
    if chain_walkers < 1:
        raise ValueError("chain_walkers must be at least 1.")

    cpu_count = os.cpu_count() or 1
    if parallel_processes is None:
        n_parallel = _default_xspec_parallel_processes(parallel_fraction)
    else:
        if parallel_processes < 1:
            raise ValueError("parallel_processes must be at least 1.")
        n_parallel = int(parallel_processes)

    xspec.Xset.parallel.walkers = n_parallel
    if refit:
        xspec.Xset.parallel.leven = n_parallel
        xspec.Fit.perform()

    chain_file = Path(chain_path).expanduser()
    if not chain_file.parent.exists():
        raise FileNotFoundError(f"XSPEC chain parent directory does not exist: {chain_file.parent}")
    if chain_file.exists():
        if not overwrite:
            raise FileExistsError(f"XSPEC chain file already exists: {chain_file}")
        chain_file.unlink()

    warnings_list = []
    selected_spectra = _xspec_chain_spectra(xspec, spectra)
    source_counts, background_counts = _xspec_spectrum_counts(selected_spectra, warnings_list)

    status = "chain_ready"
    try:
        xspec.Chain(
            str(chain_file),
            burn=chain_burn,
            runLength=chain_length,
            algorithm=chain_algorithm,
            walkers=chain_walkers,
        )
    except Exception as exc:
        status = "failed"
        warnings_list.append(f"XSPEC chain run failed: {exc}")

    return XspecChainResult(
        chain_path=str(chain_file),
        fit_statistic=_xspec_fit_value(getattr(xspec.Fit, "statistic", None)),
        fit_dof=_xspec_fit_dof(getattr(xspec.Fit, "dof", None)),
        stat_method=getattr(xspec.Fit, "statMethod", None),
        free_parameters=free_parameters,
        all_parameters=all_parameters,
        source_counts=source_counts,
        background_counts=background_counts,
        chain_settings={
            "refit": refit,
            "length": chain_length,
            "burn": chain_burn,
            "algorithm": chain_algorithm,
            "walkers": chain_walkers,
            "overwrite": overwrite,
        },
        parallel_settings={
            "cpu_count": cpu_count,
            "fraction": parallel_fraction,
            "processes": n_parallel,
            "contexts": {
                "walkers": n_parallel,
                "leven": n_parallel if refit else None,
            },
        },
        warnings=warnings_list,
        status=status,
    )


# 方法：XSPEC 误差搜索的 9 位状态串按位（bit0=1, bit1=2, ..., bit8=256）对应九种异常，'T' 置位、全 'F' 表示搜索未报异常；从未计算时初始值即 "FFFFFFFFF"
# 参考：本地 HEASoft 6.37 源码 Xspec/src/XSFit/Fit/FitErrorCalc.h:55（ErrorCalcCodes: NEWMIN=1..TOOLARGE=256）与 FitErrorCalc.cxx errCodeToString；Fit.cxx:1264（TOOLARGE=约化卡方超上限拒绝误差计算）；ModParam.cxx:51（初始 "FFFFFFFFF"）
#: XSPEC 九位误差状态串各位的含义（tclout error / Parameter.error[2]）。
#: 每位 T 表示该异常发生，全 'F' 表示搜索本身未报异常。
_XSPEC_ERROR_STATUS_FLAGS: tuple[str, ...] = (
    "new minimum found",
    "non-monotonicity detected",
    "minimization may have run into problem",
    "hit hard lower limit",
    "hit hard upper limit",
    "parameter was frozen",
    "search failed in -ve direction",
    "search failed in +ve direction",
    "reduced chi-squared too high",
)


def _decode_xspec_error_status(status: str) -> list[str]:
    """把 XSPEC 九位误差状态串解码为触发异常的原因列表。"""
    reasons: list[str] = []
    for index, flag in enumerate(_XSPEC_ERROR_STATUS_FLAGS):
        if index < len(status) and status[index].upper() == "T":
            reasons.append(flag)
    return reasons


# 方法：区分"轮廓误差从未计算"与"真实轮廓区间"——HEASoft 6.36+ 未运行 error 时 PyXspec Parameter.error 返回数值哨兵 (0.0, 0.0) 且状态串 'FFFFFFFFF'（ModParam 初始化 m_emn=m_epo=0），旧版哨兵为 (2v, 2v)；真实区间取 |bound-value|（非对称）
# 参考：本地 HEASoft 6.37 源码 Xspec/src/XSModel/Parameter/ModParam.cxx:47-65（m_epo(0.)/m_emn(0.) 初始化）；Xspec/src/XSUser/Python/xspec/parameter.py:_getError（getParTuple 索引 6-8 = 下界,上界,状态串）
def _classify_parameter_error(param, param_val: float, *, errors_computed: bool = True) -> tuple[dict[str, Any], str]:
    """读取 PyXspec ``Parameter.error`` 三元组并按参数归类误差状态。

    PyXspec 返回 ``(lower_bound, upper_bound, status_code_string)``。
    实测（HEASOFT 6.36 / PyXspec）：误差从未计算时三元组为
    ``(0.0, 0.0, 'FFFFFFFFF')``——状态串与成功时相同，因此必须同时
    检查数值哨兵；旧版 PyXspec 的未计算哨兵为 ``(2v, 2v)``，同样检测。
    ``errors_computed=False``（error 命令未执行或整体失败）时跳过读取，
    直接标记 ``uncomputed``——命令级失败意味着任何逐参数数值都不可信。

    返回 ``(写入参数字典的字段, 报告文本后缀)``。
    """
    if not errors_computed:
        return {"error_status": "uncomputed"}, " (profile error not computed)"

    status = ""
    try:
        raw = param.error
        err_lo_raw = float(raw[0])
        err_hi_raw = float(raw[1])
        status = str(raw[2]) if len(raw) > 2 else ""
    except Exception as exc:
        return (
            {"error_status": "unavailable", "error_reason": str(exc)},
            " (error unavailable)",
        )

    base_fields = {"xspec_error_status": status}
    reasons = _decode_xspec_error_status(status)

    # 数值哨兵：从未跑过 error 命令时边界为 (0, 0)（新版）或 (2v, 2v)（旧版）。
    uncomputed_sentinel = (
        (err_lo_raw == 0.0 and err_hi_raw == 0.0)
        or (
            param_val != 0.0
            and math.isclose(err_lo_raw, 2.0 * param_val, rel_tol=1e-12, abs_tol=0.0)
            and math.isclose(err_hi_raw, 2.0 * param_val, rel_tol=1e-12, abs_tol=0.0)
        )
    )
    if uncomputed_sentinel:
        fields = dict(base_fields, error_status="uncomputed")
        if reasons:
            fields["error_reasons"] = reasons
        return fields, " (profile error not computed)"

    err_lo = abs(err_lo_raw - param_val)
    err_hi = abs(err_hi_raw - param_val)
    if not all(math.isfinite(value) and value >= 0.0 for value in (err_lo, err_hi)):
        return (
            dict(base_fields, error_status="failed", error_reasons=reasons or ["non-finite bounds"]),
            " (profile error failed)",
        )

    if reasons:
        # 触硬边界时数值仍可用（搜索停在硬限上），保留数值并标注。
        if all(reason in ("hit hard lower limit", "hit hard upper limit", "new minimum found") for reason in reasons):
            fields = dict(
                base_fields,
                error_status="boundary",
                error_lo=err_lo,
                error_hi=err_hi,
                error_reasons=reasons,
            )
            return fields, f" (-{err_lo:.4g}, +{err_hi:.4g}) (profile interval at hard limit)"

        fields = dict(base_fields, error_status="failed", error_reasons=reasons)
        return fields, f" (profile error failed: {'; '.join(reasons)})"

    return (
        dict(base_fields, error_status="ok", error_lo=err_lo, error_hi=err_hi),
        f" (-{err_lo:.4g}, +{err_hi:.4g}) (profile interval)",
    )


def _generate_xspec_result(
    model,
    spectrum,
    *,
    flux_range_keV: tuple[float, float] | None = None,
    warnings_list: list[str] | None = None,
    errors_computed: bool = True,
) -> dict:
    """
    根据XSPEC模型和光谱自动生成结果字典

    参数:
        model: XSPEC模型对象
        spectrum: XSPEC光谱对象
        warnings_list: 组件级异常会追加到该列表（而非静默吞掉）
        errors_computed: error 命令是否成功完成；False 时所有自由参数
            标记为 ``uncomputed``，不读取任何误差数值

    返回:
        包含模型参数、flux、rate等信息的字典
    """
    xspec = _require_xspec()

    def warn(component: str, exc: Exception) -> None:
        message = f"XSPEC {component} unavailable: {type(exc).__name__}: {exc}"
        if warnings_list is not None:
            warnings_list.append(message)
        else:
            import warnings
            warnings.warn(message, RuntimeWarning, stacklevel=2)

    lines = []
    result = {}
    result['model'] = model.expression

    result['parameters'] = {}
    lines.append(f"Model: {model.expression}")

    processed_params = set()

    for comp_name in model.componentNames:
        try:
            comp = getattr(model, comp_name)
        except Exception as exc:
            message = f"parameter extraction failed for component {comp_name}: {exc}"
            if warnings_list is None:
                warnings_list = []
            warnings_list.append(message)
            continue

        for param_name in comp.parameterNames:
            param_key = f"{comp_name}.{param_name}"
            if param_key in processed_params:
                continue
            processed_params.add(param_key)

            param = getattr(comp, param_name)
            param_val = param.values[0]

            param_dict = {
                'value': param_val,
                'frozen': bool(param.frozen),
                'index': _model_parameter_index(model, param),
                'link': str(getattr(param, "link", "") or ""),
            }
            lower = getattr(param, "min", None)
            upper = getattr(param, "max", None)
            parameter_values = getattr(param, "values", ())
            if lower is None and len(parameter_values) >= 6:
                lower = parameter_values[2]
            if upper is None and len(parameter_values) >= 6:
                upper = parameter_values[5]
            if lower is not None:
                param_dict["min"] = float(lower)
            if upper is not None:
                param_dict["max"] = float(upper)

            if not param.frozen:
                error_fields, error_note = _classify_parameter_error(
                    param, param_val, errors_computed=errors_computed
                )
                param_dict.update(error_fields)
                lines.append(f"{comp_name}.{param_name}: {param_val:.4f}{error_note}")
            else:
                lines.append(f"{comp_name}.{param_name}: {param_val:.4f} (fixed)")

            result['parameters'][param_key] = param_dict

    try:
        if hasattr(model, 'cflux'):
            emin = model.cflux.Emin.values[0]
            emax = model.cflux.Emax.values[0]
        elif flux_range_keV is not None:
            emin, emax = flux_range_keV
        else:
            raise ValueError("No flux energy range is defined")
        xspec.AllModels.calcFlux(f"{emin} {emax}")
        flux_erg = float(spectrum.flux[0])
        flux_photons = float(spectrum.flux[3])
    except Exception as exc:
        warn("flux", exc)
        flux_erg = None
        flux_photons = None
        emin = None
        emax = None

    result['flux_abs'] = {
        'erg_cm2_s': flux_erg,
        'photons_cm2_s': flux_photons,
        'energy_range_keV': (emin, emax) if emin is not None and emax is not None else None,
    }

    if flux_erg is not None and emin is not None and emax is not None:
        lines.append(f"Absorbed Flux ({emin:.1f}-{emax:.1f} keV): {flux_erg:.4e} erg/cm²/s")
        lines.append(f"Absorbed Photon Flux ({emin:.1f}-{emax:.1f} keV): {flux_photons:.4e} photons/cm²/s")

    try:
        rate = float(spectrum.rate[0])
        rate_err = float(spectrum.rate[1]) if len(spectrum.rate) > 1 else None
    except Exception as exc:
        warn("rate", exc)
        rate = None
        rate_err = None

    result['rate'] = {
        'value': rate,
        'error': rate_err
    }

    if rate is not None:
        if rate_err is not None:
            lines.append(f"Rate: {rate:.4f} ± {rate_err:.4f} cts/s")
        else:
            lines.append(f"Rate: {rate:.4f} cts/s")

    exposure = spectrum.exposure if hasattr(spectrum, 'exposure') else None

    if rate is not None and rate > 0 and flux_erg is not None and flux_erg > 0:
        try:
            conv_factor = 10**model.cflux.lg10Flux.values[0] / rate
        except Exception as exc:
            warn("conversion factor", exc)
            conv_factor = None
    else:
        conv_factor = None
    photon_counts = rate * exposure if rate is not None and exposure is not None else None
    result['conversion'] = {
        'exposure_s': exposure,
        'erg_per_count': conv_factor,
        'counts': photon_counts,
        'total_counts': photon_counts,
    }

    if exposure is not None:
        lines.append(f"Exposure: {exposure:.1f} s")

    if conv_factor is not None:
        lines.append(f"Conversion factor: {conv_factor:.4e} erg/cm²/s per cts/s")
    if photon_counts is not None:
        lines.append(f"Total counts: {photon_counts:.2f} counts")

    try:
        statistic = xspec.Fit.statistic
        dof = xspec.Fit.dof
        stat_method = xspec.Fit.statMethod
        statdof = statistic / dof if dof > 0 else None
        null_prob = xspec.Fit.nullhyp

        lines.append(f"Stat/dof: {stat_method}={statistic:.2f}/{dof}={statdof:.2f}" if statdof else f"Stat/dof: {stat_method}={statistic:.2f}/{dof}")
        lines.append(f"Null hypothesis probability: {null_prob:.4f}")

        result['statistics'] = {
            'method': stat_method,
            'value': statistic,
            'dof': dof,
            'reduced': statdof,
            'null_hypothesis_probability': null_prob
        }
    except Exception as exc:
        warn("fit statistics", exc)
        result['statistics'] = {}

    result['text'] = "\n".join(lines)

    return result


def _count_free_xspec_parameters(models: Sequence[Any]) -> int:
    """Count thawed, unlinked XSPEC parameters across data groups."""
    count = 0
    for model in models:
        for component_name in getattr(model, "componentNames", ()):
            component = getattr(model, component_name, None)
            if component is None:
                continue
            for parameter_name in getattr(component, "parameterNames", ()):
                parameter = getattr(component, parameter_name, None)
                if parameter is None or bool(getattr(parameter, "frozen", False)):
                    continue
                if str(getattr(parameter, "link", "") or "").strip():
                    continue
                count += 1
    return count


def _model_parameter_index(model, parameter) -> int:
    start = getattr(model, "startParIndex", None)
    index = int(parameter.index)
    if start in (None, 1):
        return index
    return int(start) + index - 1


def _prepared_error_parameters(
    model,
    model_name: str,
    models=None,
    *,
    intrinsic_nh_mode: str = "free",
    delta_stat: float = 1.0,
) -> str:
    parameter_indices = []
    prefix = "1." if float(delta_stat) == 1.0 else f"{float(delta_stat):g}"
    if intrinsic_nh_mode == "free" and "ztbabs" in model_name.lower():
        for _, component in _prepared_ztbabs_components(model):
            if hasattr(component, "nH"):
                parameter_indices.append(
                    str(_model_parameter_index(model, component.nH))
                )
    for group_model in models or [model]:
        if hasattr(group_model, "cflux") and hasattr(group_model.cflux, "lg10Flux"):
            parameter_indices.append(
                str(_model_parameter_index(group_model, group_model.cflux.lg10Flux))
            )
    if hasattr(model, "powerlaw") and hasattr(model.powerlaw, "PhoIndex"):
        parameter_indices.append(str(_model_parameter_index(model, model.powerlaw.PhoIndex)))
        if not hasattr(model, "cflux") and hasattr(model.powerlaw, "norm"):
            parameter_indices.append(str(_model_parameter_index(model, model.powerlaw.norm)))
    elif hasattr(model, "zpowerlw") and hasattr(model.zpowerlw, "PhoIndex"):
        parameter_indices.append(str(_model_parameter_index(model, model.zpowerlw.PhoIndex)))
        if not hasattr(model, "cflux") and hasattr(model.zpowerlw, "norm"):
            parameter_indices.append(str(_model_parameter_index(model, model.zpowerlw.norm)))
    elif hasattr(model, "apec") and hasattr(model.apec, "kT"):
        parameter_indices.append(str(_model_parameter_index(model, model.apec.kT)))
    elif hasattr(model, "bbody") and hasattr(model.bbody, "kT"):
        parameter_indices.append(str(_model_parameter_index(model, model.bbody.kT)))
    elif hasattr(model, "bknpower"):
        for name in ("PhoIndx1", "BreakE", "PhoIndx2"):
            parameter = getattr(model.bknpower, name, None)
            if parameter is not None:
                parameter_indices.append(str(_model_parameter_index(model, parameter)))
    return " ".join((prefix, *parameter_indices)) if parameter_indices else ""


# 方法：XSPEC error 命令的单参数轮廓（profile）置信区间：Δstat=stat(θ)-stat_min 达到阈值处即区间端点；关键式：Δ=1.0 对应单参数 1σ/68.3%（置信度 0.682689492137=erf(1/√2)），Δ=2.706 对应 90%（χ²1 分布上分位数），3σ 对应 Δ=9.0
# 参考：Cash 1979, ApJ 228, 939；Baker & Cousins 1984, Nucl. Instrum. Methods Phys. Res. A 221, 437（2ΔlnL↔χ² 等价）；XSPEC manual "Fits and Confidence Intervals"/error 命令（2.706→单参数 90%）
def _profile_error_metadata(delta_stat: float) -> dict[str, Any]:
    """Describe the one-parameter XSPEC profile interval for ``delta_stat``."""

    try:
        value = float(delta_stat)
    except (TypeError, ValueError) as exc:
        raise ValueError("error_delta_stat must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("error_delta_stat must be finite and positive")
    metadata: dict[str, Any] = {
        "error_delta_stat": value,
        "profile_error_kind": "single_parameter_profile",
        "profile_confidence": None,
        "profile_sigma": None,
        "profile_error_label": f"single-parameter profile (delta statistic={value:g})",
    }
    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9):
        metadata.update(
            {
                "profile_confidence": 0.682689492137,
                "profile_sigma": 1.0,
                "profile_error_label": "single-parameter profile 1 sigma (68.3%)",
            }
        )
    elif math.isclose(value, 2.706, rel_tol=0.0, abs_tol=1e-6):
        metadata.update(
            {
                "profile_confidence": 0.90,
                "profile_error_label": "single-parameter profile 90%",
            }
        )
    return metadata


def _set_prepared_parameter_bounds(parameter, value: float, lower: float, upper: float) -> None:
    """Set bounds on fake test parameters and real PyXspec parameters."""
    try:
        parameter.min = lower
        parameter.max = upper
    except Exception:
        parameter.values = f"{value},,{lower},{lower},{upper},{upper}"


def _freeze_prepared_parameters(model, frozen_parameters: Mapping[str, float] | None) -> None:
    """Set and freeze named XSPEC parameters on a prepared model.

    Keys use the stable ``component.parameter`` spelling exposed by
    :func:`_xspec_chain_parameters`, for example ``powerlaw.PhoIndex``.  The
    helper deliberately rejects unknown or ambiguous names so an upper-limit
    calculation cannot silently profile the wrong parameter.
    """
    if not frozen_parameters:
        return
    components = {
        str(component_name).lower(): getattr(model, component_name)
        for component_name in getattr(model, "componentNames", ())
    }
    for key, value in frozen_parameters.items():
        name = str(key).strip()
        if "." not in name:
            raise ValueError(
                f"frozen parameter {name!r} must use 'component.parameter' spelling"
            )
        component_name, parameter_name = (part.strip() for part in name.split(".", 1))
        component = components.get(component_name.lower())
        if component is None:
            raise ValueError(f"unknown XSPEC model component in frozen parameter {name!r}")
        parameter_names = {
            str(candidate).lower(): str(candidate)
            for candidate in getattr(component, "parameterNames", ())
        }
        actual_name = parameter_names.get(parameter_name.lower())
        if actual_name is None or not hasattr(component, actual_name):
            raise ValueError(f"unknown XSPEC model parameter {name!r}")
        try:
            resolved = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frozen parameter {name!r} must be finite") from exc
        if not math.isfinite(resolved):
            raise ValueError(f"frozen parameter {name!r} must be finite")
        parameter = getattr(component, actual_name)
        parameter.values = resolved
        parameter.frozen = True


def _prepared_ztbabs_components(model) -> list[tuple[str, Any]]:
    """Return every zTBabs component, including XSPEC's suffixed duplicates."""
    names = [
        str(name)
        for name in (getattr(model, "componentNames", None) or ())
        if str(name).lower().startswith("ztbabs")
    ]
    if not names and hasattr(model, "zTBabs"):
        names = ["zTBabs"]
    return [(name, getattr(model, name)) for name in names]


def _configure_prepared_model(
    model,
    *,
    model_name: str,
    emin: float,
    emax: float,
    redshift: float,
    redshift_absorbers: Sequence[float] | None = None,
    galactic_nh_1e22: float | None,
    freeze_galactic_nh: bool,
    intrinsic_nh_mode: str = "free",
) -> None:
    if intrinsic_nh_mode not in {"free", "zero"}:
        raise ValueError("intrinsic_nh_mode must be 'free' or 'zero'")
    if hasattr(model, "TBabs") and hasattr(model.TBabs, "nH"):
        if galactic_nh_1e22 is not None:
            if not np.isfinite(galactic_nh_1e22) or galactic_nh_1e22 < 0:
                raise ValueError("galactic_nh_1e22 must be finite and non-negative")
            model.TBabs.nH = float(galactic_nh_1e22)
            model.TBabs.nH.frozen = bool(freeze_galactic_nh)

    if "ztbabs" in model_name.lower():
        absorbers = _prepared_ztbabs_components(model)
        absorber_redshifts = (
            tuple(float(value) for value in redshift_absorbers)
            if redshift_absorbers is not None
            else (float(redshift),)
        )
        if len(absorber_redshifts) != len(absorbers):
            raise ValueError(
                "redshift_absorbers must provide exactly one redshift for each "
                f"zTBabs component; got {len(absorber_redshifts)} for {len(absorbers)}"
            )
        if any(not np.isfinite(value) or value < 0 for value in absorber_redshifts):
            raise ValueError("redshift_absorbers must be finite and non-negative")
        for (_, absorber), absorber_redshift in zip(absorbers, absorber_redshifts):
            if hasattr(absorber, "nH"):
                initial_nh = 0.0 if intrinsic_nh_mode == "zero" else 0.5
                absorber.nH = initial_nh
                _set_prepared_parameter_bounds(absorber.nH, initial_nh, 0.0, 100.0)
                absorber.nH.frozen = intrinsic_nh_mode == "zero"
            if hasattr(absorber, "Redshift"):
                absorber.Redshift = absorber_redshift
                absorber.Redshift.frozen = True

    if hasattr(model, "cflux"):
        if hasattr(model.cflux, "Emin"):
            model.cflux.Emin = emin
            model.cflux.Emin.frozen = True
        if hasattr(model.cflux, "Emax"):
            model.cflux.Emax = emax
            model.cflux.Emax.frozen = True
        if hasattr(model.cflux, "lg10Flux"):
            model.cflux.lg10Flux = -10.0
            _set_prepared_parameter_bounds(model.cflux.lg10Flux, -10.0, -20.0, 10.0)

    if hasattr(model, "powerlaw") and hasattr(model.powerlaw, "PhoIndex"):
        model.powerlaw.PhoIndex = 2.0
        _set_prepared_parameter_bounds(model.powerlaw.PhoIndex, 2.0, 0.0, 9.0)
        if hasattr(model.powerlaw, "norm"):
            if hasattr(model, "cflux"):
                model.powerlaw.norm = 1.0
                model.powerlaw.norm.frozen = True
            else:
                model.powerlaw.norm = 1e-3
                _set_prepared_parameter_bounds(model.powerlaw.norm, 1e-3, 0.0, 1e24)
                model.powerlaw.norm.frozen = False

    if hasattr(model, "zpowerlw") and hasattr(model.zpowerlw, "PhoIndex"):
        model.zpowerlw.PhoIndex = 2.0
        _set_prepared_parameter_bounds(model.zpowerlw.PhoIndex, 2.0, 0.0, 9.0)
        if redshift > 0 and hasattr(model.zpowerlw, "Redshift"):
            model.zpowerlw.Redshift = redshift
            model.zpowerlw.Redshift.frozen = True
        if hasattr(model.zpowerlw, "norm") and hasattr(model, "cflux"):
            model.zpowerlw.norm = 1.0
            model.zpowerlw.norm.frozen = True

    if hasattr(model, "apec"):
        if hasattr(model.apec, "kT"):
            model.apec.kT = 1.0
            _set_prepared_parameter_bounds(model.apec.kT, 1.0, 0.008, 64.0)
        if hasattr(model.apec, "Abundanc"):
            model.apec.Abundanc = 1.0
            model.apec.Abundanc.frozen = True
        if hasattr(model.apec, "Redshift"):
            model.apec.Redshift = redshift
            model.apec.Redshift.frozen = True
        if hasattr(model.apec, "norm"):
            model.apec.norm = 1.0
            model.apec.norm.frozen = True

    if hasattr(model, "bbody"):
        if hasattr(model.bbody, "kT"):
            model.bbody.kT = min(max(0.3, emin), emax)
            _set_prepared_parameter_bounds(model.bbody.kT, model.bbody.kT.values[0], 0.01, max(10.0, emax))
        if hasattr(model.bbody, "norm"):
            model.bbody.norm = 1.0
            model.bbody.norm.frozen = True

    if hasattr(model, "bknpower"):
        if hasattr(model.bknpower, "PhoIndx1"):
            model.bknpower.PhoIndx1 = 1.5
            _set_prepared_parameter_bounds(model.bknpower.PhoIndx1, 1.5, 0.0, 9.0)
        if hasattr(model.bknpower, "BreakE"):
            break_energy = math.sqrt(emin * emax)
            model.bknpower.BreakE = break_energy
            _set_prepared_parameter_bounds(model.bknpower.BreakE, break_energy, emin, emax)
        if hasattr(model.bknpower, "PhoIndx2"):
            model.bknpower.PhoIndx2 = 2.5
            _set_prepared_parameter_bounds(model.bknpower.PhoIndx2, 2.5, 0.0, 9.0)
        if hasattr(model.bknpower, "norm"):
            model.bknpower.norm = 1.0
            model.bknpower.norm.frozen = True


def _prepared_input_dict(prepared) -> dict[str, str | int | None]:
    return {
        "instrument": prepared.instrument,
        "obsid": prepared.obsid or "unknown",
        "module": prepared.module,
        "detector": prepared.detector,
        "source_id": prepared.source_id,
        "source_pha": str(prepared.source_pha) if prepared.source_pha else None,
        "grouped_pha": str(prepared.grouped_pha) if prepared.grouped_pha else None,
        "background_pha": str(prepared.background_pha) if prepared.background_pha else None,
        "arf": str(prepared.arf) if prepared.arf else None,
        "rmf": str(prepared.rmf) if prepared.rmf else None,
        "group_min": prepared.group_min,
    }


def _prepared_spectrum_key(prepared) -> str:
    if prepared.module:
        return f"{prepared.obsid}:{prepared.module}"
    if prepared.detector:
        suffix = f":{prepared.source_id}" if prepared.source_id else ""
        return f"{prepared.obsid}:{prepared.detector}{suffix}"
    return f"{prepared.obsid}:{prepared.instrument}"


def _prepared_energy_ranges(
    prepared_spectra,
    *,
    emin: float | None,
    emax: float | None,
    energy_ranges: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    if energy_ranges and (emin is not None or emax is not None):
        raise ValueError("energy_ranges cannot be combined with global emin/emax")

    resolved = {
        _prepared_spectrum_key(spectrum): (
            float(emin if emin is not None else spectrum.energy_range_keV[0]),
            float(emax if emax is not None else spectrum.energy_range_keV[1]),
        )
        for spectrum in prepared_spectra
    }
    if not energy_ranges:
        return resolved

    unknown = set(energy_ranges) - set(resolved)
    if unknown:
        choices = ", ".join(sorted(resolved))
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown prepared energy range keys: {names}. Available: {choices}")
    for key, limits in energy_ranges.items():
        lower, upper = limits
        resolved[key] = (float(lower), float(upper))
    return resolved


def _prepared_data_groups(prepared_spectra, fit_ranges) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, float], dict[str, Any]] = {}
    ordered_groups = []
    for spectrum_index, spectrum in enumerate(prepared_spectra, start=1):
        spectrum_key = _prepared_spectrum_key(spectrum)
        emin, emax = fit_ranges[spectrum_key]
        key = (spectrum.instrument, emin, emax)
        group = groups.get(key)
        if group is None:
            group = {
                "group_index": len(ordered_groups) + 1,
                "instrument": spectrum.instrument,
                "energy_range": {"emin": emin, "emax": emax},
                "spectrum_keys": [],
                "spectrum_indices": [],
                "spectra": [],
            }
            groups[key] = group
            ordered_groups.append(group)
        group["spectrum_keys"].append(spectrum_key)
        group["spectrum_indices"].append(spectrum_index)
        group["spectra"].append(spectrum)
    return ordered_groups


def _xspec_model_for_group(xspec, model, group_index: int):
    if group_index == 1:
        return model
    try:
        return xspec.AllModels(group_index)
    except TypeError:
        return xspec.AllModels(group_index, "")


def _link_parameter(parameter, reference) -> None:
    try:
        parameter.link = reference
    except Exception:
        parameter.link = str(getattr(reference, "index", reference))


def _link_default_prepared_model_groups(models, model_name: str) -> None:
    if len(models) <= 1:
        return
    reference = models[0]
    for model in models[1:]:
        reference_absorbers = _prepared_ztbabs_components(reference)
        absorbers = _prepared_ztbabs_components(model)
        if len(reference_absorbers) != len(absorbers):
            raise RuntimeError("Prepared XSPEC model groups have different zTBabs counts")
        for (_, reference_absorber), (_, absorber) in zip(reference_absorbers, absorbers):
            if hasattr(reference_absorber, "nH") and hasattr(absorber, "nH"):
                _link_parameter(absorber.nH, reference_absorber.nH)
        if hasattr(reference, "powerlaw") and hasattr(model, "powerlaw"):
            if hasattr(reference.powerlaw, "PhoIndex") and hasattr(model.powerlaw, "PhoIndex"):
                _link_parameter(model.powerlaw.PhoIndex, reference.powerlaw.PhoIndex)
        for component_name, parameter_names in (
            ("apec", ("kT", "Abundanc", "Redshift")),
            ("bbody", ("kT",)),
            ("bknpower", ("PhoIndx1", "BreakE", "PhoIndx2")),
        ):
            reference_component = getattr(reference, component_name, None)
            component = getattr(model, component_name, None)
            if reference_component is None or component is None:
                continue
            for parameter_name in parameter_names:
                reference_parameter = getattr(reference_component, parameter_name, None)
                parameter = getattr(component, parameter_name, None)
                if reference_parameter is None or parameter is None:
                    continue
                if not bool(getattr(parameter, "frozen", False)):
                    _link_parameter(parameter, reference_parameter)


def _set_prepared_xspec_links(spectrum, prepared) -> None:
    if prepared.background_pha is not None:
        try:
            background = spectrum.background
        except Exception:
            background = None
        if background is None:
            spectrum.background = str(prepared.background_pha)

    try:
        response = spectrum.response
    except Exception:
        response = None
    if prepared.rmf is not None and response is None:
        spectrum.response = str(prepared.rmf)
        try:
            response = spectrum.response
        except Exception:
            response = None

    if prepared.arf is not None:
        if response is not None and hasattr(response, "arf"):
            if not getattr(response, "arf", None):
                response.arf = str(prepared.arf)
        elif hasattr(spectrum, "arf"):
            spectrum.arf = str(prepared.arf)


def _load_prepared_xspec_spectrum(
    xspec,
    prepared,
    *,
    index: int | None = None,
    data_group: int | None = None,
):
    """Load one grouped PHA where XSPEC can resolve its local response links."""
    grouped = Path(prepared.grouped_pha).expanduser().resolve()
    original = Path.cwd()
    os.chdir(grouped.parent)
    try:
        if index is None:
            return xspec.Spectrum(str(grouped))
        xspec.AllData(f"{data_group or 1}:{index} {grouped}")
        return xspec.AllData(index)
    finally:
        os.chdir(original)


def _capture_xspec_log(xspec, path: Path, action, warnings_list: list[str]) -> str:
    """执行 XSPEC 动作并捕获其日志文本。

    成功路径：读完日志后删除临时文件。
    失败路径：保留临时文件（供 ``_write_xray_failure_log`` 收集转录），
    由调用方决定如何消费后清理。
    """
    opener = getattr(getattr(xspec, "Xset", None), "openLog", None)
    closer = getattr(getattr(xspec, "Xset", None), "closeLog", None)
    if not callable(opener) or not callable(closer):
        action()
        return ""

    try:
        opener(str(path))
        action()
    finally:
        try:
            closer()
        except Exception as exc:
            warnings_list.append(f"Could not close XSPEC log {path.name}: {exc}")

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        warnings_list.append(f"Could not read XSPEC log {path.name}: {exc}")
        return ""
    try:
        path.unlink()
    except OSError:
        pass
    return text


def _xspec_show_text(xspec, outdir: Path, label: str, warnings_list: list[str]) -> str:
    def show() -> None:
        xspec.AllModels.show()
        xspec.Fit.show()

    try:
        return _capture_xspec_log(xspec, outdir / f"{label}.xspec_show.tmp.log", show, warnings_list)
    except Exception as exc:
        warnings_list.append(f"Could not capture XSPEC show output: {exc}")
        return ""


def _write_prepared_report(
    *,
    path: Path,
    prepared_inputs: list[dict[str, str | int | None]],
    results: dict,
    show_text: str,
    error_command: str | None,
    error_text: str,
    warnings_list: list[str],
    data_groups=None,
    per_group=None,
) -> Path:
    lines = ["[prepared_inputs]"]
    for index, item in enumerate(prepared_inputs, start=1):
        lines.append(f"Spectrum {index}")
        lines.extend(f"{key}: {value}" for key, value in item.items())
    if data_groups:
        lines.extend(["", "[data_groups]"])
        for group in data_groups:
            lines.extend(
                [
                    f"Group {group['group_index']}",
                    f"instrument: {group['instrument']}",
                    f"energy_range: {group['energy_range']}",
                    f"spectrum_indices: {group['spectrum_indices']}",
                    f"spectrum_keys: {group['spectrum_keys']}",
                ]
            )
    if per_group:
        lines.extend(["", "[per_group]"])
        for group in per_group:
            lines.extend(
                [
                    f"Group {group['group_index']}",
                    f"instrument: {group['instrument']}",
                    f"energy_range: {group['energy_range']}",
                    f"flux_abs: {group.get('flux_abs')}",
                    f"cflux: {group.get('cflux')}",
                    f"member_spectra: {group['member_spectra']}",
                ]
            )
    lines.extend(["", "[fit_summary]", str(results.get("text", ""))])
    if error_command:
        lines.extend(["", "[xspec.Fit.error]", f"command: {error_command}", error_text])
    if show_text:
        lines.extend(["", "[xspec.AllModels.show / xspec.Fit.show]", show_text])
    if warnings_list:
        lines.extend(["", "[warnings]"])
        lines.extend(warnings_list)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def fit_prepared(
    prepared,
    *,
    outdir: str | Path,
    emin: float | None = None,
    emax: float | None = None,
    energy_ranges: Mapping[str, tuple[float, float]] | None = None,
    model_name: str = "tbabs*ztbabs*cflux*powerlaw",
    frozen_parameters: Mapping[str, float] | None = None,
    redshift: float = 0.0,
    redshift_absorbers: Sequence[float] | None = None,
    # 与 master 一致恢复显式默认值（statistic="cstat", abundance="wilm",
    # cross_section="vern"）。beta 曾默认 None 并隐式回退到进程级
    # get_fit_settings()，导致同一进程内其它代码调用 set_fit_settings()/
    # set_fit_method() 会静默改变本函数行为（全局状态副作用，0.2.0 已修复）。
    # 需要"进程级拟合设置"的调用方请走 fit_spectral(settings=...)，由它把
    # FitConfig 解析成显式实参传入本函数；直接调用本函数时行为只取决于实参。
    stat_method: str = "cstat",
    abundance: str = "wilm",
    cross_section: str = "vern",
    galactic_nh_1e22: float | None = None,
    freeze_galactic_nh: bool = True,
    intrinsic_nh_mode: Literal["free", "zero"] = "free",
    calculate_errors: bool = True,
    error_command: str | None = None,
    error_delta_stat: float = 1.0,
    srcname: str | None = None,
    instname: str | None = None,
    plot_backend: str = "matplotlib",
    plot_format: str = "png",
    plot_formats: Sequence[str] | None = None,
    # 参数名与 master 一致为 plot_density（传给 plotfit 的 density=，是绘图
    # 采样密度而非 DPI；beta 曾更名为 plot_dpi，属静默破坏性改名，已回退）。
    plot_density: int = 300,
    plot_required: bool = False,
) -> dict:
    """Fit one prepared spectrum or multiple prepared spectra with XSPEC."""
    from jinwu.core.plot import plotfit
    from jinwu.core.spectrum_prep import PreparedJointSpectrum, PreparedSpectrum

    error_metadata = _profile_error_metadata(error_delta_stat)
    if isinstance(prepared, PreparedSpectrum):
        prepared_spectra = [prepared]
    elif isinstance(prepared, PreparedJointSpectrum):
        prepared_spectra = list(prepared.spectra)
    elif isinstance(prepared, Sequence) and not isinstance(prepared, (str, bytes)):
        prepared_spectra = list(prepared)
        if not all(isinstance(spectrum, PreparedSpectrum) for spectrum in prepared_spectra):
            raise TypeError("prepared sequences must contain PreparedSpectrum items")
    else:
        raise TypeError(
            "prepared must be PreparedSpectrum, PreparedJointSpectrum, "
            "or a sequence of PreparedSpectrum items"
        )
    if not prepared_spectra or any(not spectrum.ready for spectrum in prepared_spectra):
        raise RuntimeError("fit_prepared requires ready prepared spectra")
    if any(spectrum.grouped_pha is None for spectrum in prepared_spectra):
        raise RuntimeError("fit_prepared requires grouped PHA paths")

    xspec = _require_xspec()
    output = Path(outdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    warnings_list: list[str] = []
    fit_ranges = _prepared_energy_ranges(
        prepared_spectra,
        emin=emin,
        emax=emax,
        energy_ranges=energy_ranges,
    )
    first_key = _prepared_spectrum_key(prepared_spectra[0])
    fit_emin, fit_emax = fit_ranges[first_key]

    xspec.AllData.clear()
    xspec.AllModels.clear()
    xspec.Xset.abund = abundance
    xspec.Xset.xsect = cross_section
    xspec.Fit.query = "yes"
    xspec.Fit.statMethod = stat_method

    data_groups = _prepared_data_groups(prepared_spectra, fit_ranges)
    group_for_index = {
        spectrum_index: group
        for group in data_groups
        for spectrum_index in group["spectrum_indices"]
    }
    xspec_spectra = []
    if len(prepared_spectra) == 1:
        xspec_spectrum = _load_prepared_xspec_spectrum(xspec, prepared_spectra[0])
        _set_prepared_xspec_links(xspec_spectrum, prepared_spectra[0])
        xspec_spectra.append(xspec_spectrum)
    else:
        for index, spectrum in enumerate(prepared_spectra, start=1):
            xspec_spectrum = _load_prepared_xspec_spectrum(
                xspec,
                spectrum,
                index=index,
                data_group=group_for_index[index]["group_index"],
            )
            _set_prepared_xspec_links(xspec_spectrum, spectrum)
            xspec_spectra.append(xspec_spectrum)

    xspec.AllData.ignore("bad")
    for xspec_spectrum, spectrum in zip(xspec_spectra, prepared_spectra):
        spectrum_emin, spectrum_emax = fit_ranges[_prepared_spectrum_key(spectrum)]
        xspec_spectrum.ignore(f"**-{spectrum_emin} {spectrum_emax}-**")

    model = xspec.Model(model_name)
    group_models = []
    for group in data_groups:
        group_model = _xspec_model_for_group(xspec, model, group["group_index"])
        group_models.append(group_model)
        _configure_prepared_model(
            group_model,
            model_name=model_name,
            emin=group["energy_range"]["emin"],
            emax=group["energy_range"]["emax"],
            redshift=redshift,
            redshift_absorbers=redshift_absorbers,
            galactic_nh_1e22=galactic_nh_1e22,
            freeze_galactic_nh=freeze_galactic_nh,
            intrinsic_nh_mode=intrinsic_nh_mode,
        )
        _freeze_prepared_parameters(group_model, frozen_parameters)
    _link_default_prepared_model_groups(group_models, model_name)
    perform_text = _capture_xspec_log(
        xspec,
        output / "fit_prepared.xspec_perform.tmp.log",
        xspec.Fit.perform,
        warnings_list,
    )

    command = error_command or _prepared_error_parameters(
        model,
        model_name,
        group_models,
        intrinsic_nh_mode=intrinsic_nh_mode,
        delta_stat=error_delta_stat,
    )
    error_text = ""
    profile_errors_succeeded = False
    error_log_path = output / "fit_prepared.xspec_error.tmp.log"
    if calculate_errors and command:
        try:
            error_text = _capture_xspec_log(
                xspec,
                error_log_path,
                lambda: xspec.Fit.error(command),
                warnings_list,
            )
            profile_errors_succeeded = True
            for line in error_text.splitlines():
                if "warning" in line.lower() or "pegged" in line.lower():
                    warnings_list.append(f"XSPEC profile error: {line.strip()}")
        except Exception as exc:
            warnings_list.append(f"XSPEC error calculation failed: {exc}")
            # error 命令失败但拟合本身成功：把转录并入报告后清理临时日志
            try:
                transcript = error_log_path.read_text(encoding="utf-8", errors="ignore")
                error_text = transcript or str(exc)
                error_log_path.unlink()
            except OSError:
                error_text = str(exc)

    results = _generate_xspec_result(
        model,
        xspec_spectra[0],
        flux_range_keV=fit_ranges[first_key],
        warnings_list=warnings_list,
        errors_computed=profile_errors_succeeded,
    )
    prepared_inputs = [_prepared_input_dict(spectrum) for spectrum in prepared_spectra]
    if srcname is None:
        first = prepared_spectra[0]
        srcname = "_".join(
            str(value)
            for value in (first.obsid, first.module or first.detector, first.source_id)
            if value
        ) or "prepared"
    if instname is None:
        instname = "+".join(
            spectrum.module or spectrum.detector or spectrum.instrument
            for spectrum in prepared_spectra
        )
    label = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{srcname}_{instname}").strip("_")

    per_group = []
    for group, group_model in zip(data_groups, group_models):
        spectrum_index = group["spectrum_indices"][0]
        group_result = _generate_xspec_result(
            group_model,
            xspec_spectra[spectrum_index - 1],
            flux_range_keV=(
                group["energy_range"]["emin"],
                group["energy_range"]["emax"],
            ),
            warnings_list=warnings_list,
            errors_computed=profile_errors_succeeded,
        )
        cflux = {}
        if hasattr(group_model, "cflux"):
            for name in ("Emin", "Emax", "lg10Flux"):
                parameter = getattr(group_model.cflux, name, None)
                if parameter is not None:
                    cflux[name] = _xspec_parameter_value(parameter)
        per_group.append(
            {
                "group_index": group["group_index"],
                "instrument": group["instrument"],
                "energy_range": group["energy_range"],
                "member_spectra": list(group["spectrum_keys"]),
                "spectrum_indices": list(group["spectrum_indices"]),
                "model": group_result.get("model"),
                "parameters": group_result.get("parameters"),
                "flux_abs": group_result.get("flux_abs"),
                "cflux": cflux,
            }
        )

    show_text = _xspec_show_text(xspec, output, label, warnings_list)
    report = _write_prepared_report(
        path=output / f"{label}_fit.txt",
        prepared_inputs=prepared_inputs,
        results=results,
        show_text=show_text,
        error_command=command if calculate_errors else None,
        error_text=error_text,
        warnings_list=warnings_list,
        data_groups=data_groups,
        per_group=per_group,
    )

    results["energy_range"] = {"emin": fit_emin, "emax": fit_emax}
    results["energy_ranges"] = {
        key: {"emin": limits[0], "emax": limits[1]}
        for key, limits in fit_ranges.items()
    }
    results["data_groups"] = [
        {
            key: value
            for key, value in group.items()
            if key != "spectra"
        }
        for group in data_groups
    ]
    results["per_group"] = per_group
    results["input_files"] = prepared_inputs
    results["report_txt"] = str(report)
    results["warnings"] = warnings_list
    results["error_parameters"] = (
        [int(token) for token in str(command).split()[1:] if token.isdigit()]
        if (calculate_errors and command)
        else []
    )
    results["group_min"] = prepared_spectra[0].group_min
    results["xspec_settings"] = {
        "abundance": abundance,
        "cross_section": cross_section,
        "requested_statistic": stat_method,
        "galactic_nh_1e22": galactic_nh_1e22,
        "freeze_galactic_nh": freeze_galactic_nh,
        "intrinsic_nh_mode": intrinsic_nh_mode,
        "redshift_absorbers": (
            list(redshift_absorbers) if redshift_absorbers is not None else [redshift]
        ),
        **error_metadata,
        "calculate_errors": calculate_errors,
        "frozen_parameters": (
            {str(key): float(value) for key, value in frozen_parameters.items()}
            if frozen_parameters
            else {}
        ),
        "error_command": command if calculate_errors else None,
        "profile_errors_succeeded": profile_errors_succeeded,
    }
    # 方法：XSPEC 统计命令选 cstat 且光谱挂载 Poisson 背景文件时，实际计算的是 W 统计量——对每 bin 背景率 f̂ 做 profile likelihood 解析消去背景阈值参数（s>0,b>0 分支 f̂ 由二次方程 ti*f^2+(ti*y-s-b)*f-b*y=0 的正根给出，总统计量=2Σ半贡献），故此处把 (cstat+背景) 报告为 wstat
    # 参考：Cash 1979, ApJ 228, 939；Humphrey, Liu & Buote 2009, ApJ 693, 822（W 统计量推导）；XSPEC manual Appendix B "Statistics in XSPEC"；本地 HEASoft 6.37 源码 Xspec/src/XSStat/CstatVariants.cxx（specificPerformB 五分支闭式）与 Cstat.h:307（statValues*=2.0）
    results["effective_statistic"] = (
        "wstat"
        if stat_method.lower() == "cstat"
        and any(spectrum.background_pha is not None for spectrum in prepared_spectra)
        else stat_method.lower()
    )
    results["free_parameter_count"] = _count_free_xspec_parameters(group_models)
    results["group_mins"] = {
        _prepared_spectrum_key(spectrum): spectrum.group_min
        for spectrum in prepared_spectra
    }

    requested_formats = (
        tuple(plot_formats) if plot_formats is not None else (plot_format,)
    )
    plot_paths = []
    plot_errors = []
    for requested_format in requested_formats:
        try:
            figure_path, figure = plotfit(
                srcname=srcname,
                instname=instname,
                group_min=prepared_spectra[0].group_min,
                modelname=model_name,
                redshift=redshift,
                outputdir=output,
                backend=plot_backend,
                output_format=requested_format,
                density=plot_density,
            )
            if figure_path is not None:
                plot_paths.append(str(figure_path))
            if figure is not None:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:
                    pass
        except Exception as exc:
            plot_errors.append(f"{requested_format}: {exc}")
    results["plot_fit"] = plot_paths[0] if plot_paths else None
    results["plot_fits"] = plot_paths
    if plot_errors:
        results["plot_fit_error"] = "; ".join(plot_errors)
        warnings_list.extend(f"fit plot failed: {message}" for message in plot_errors)
    if plot_required and len(plot_paths) != len(requested_formats):
        raise RuntimeError(
            "Required XSPEC fit plots were not produced: " + "; ".join(plot_errors)
        )

    from jinwu.core.products import save_xspec_session

    products = save_xspec_session(
        xspec,
        output_dir=output,
        label=label,
        result=results,
        report_txt=report,
        transcript="\n\n".join(
            part
            for part in (perform_text.strip(), error_text.strip(), show_text.strip())
            if part
        ),
        plots=plot_paths,
        input_paths=[
            value
            for item in prepared_inputs
            for value in item.values()
            if isinstance(value, str) and Path(value).exists()
        ],
    )
    results["fit_products"] = {
        "result_json": str(products.result_json),
        "report_txt": str(products.report_txt),
        "xspec_log": str(products.xspec_log),
        "xcm": str(products.xcm),
        "plots": [str(path) for path in products.plots],
        "replay_cwd": str(products.replay_cwd),
    }
    return results


def fit_spectral(
    prepared,
    *,
    outdir: str | Path,
    method: Literal["mle", "chain", "bxa"] | None = None,
    settings: FitConfig | None = None,
    chain_path: str | Path | None = None,
    chain_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
):
    """Unified entry point that routes a prepared fit to the selected method.

    The effective ``method`` resolves, from highest to lowest priority, as:
    explicit ``method`` argument > ``settings.method`` >
    :func:`jinwu.core.config.get_fit_settings`.  Routing:

    - ``"mle"``   -> :func:`fit_prepared` (returns its result ``dict``).
    - ``"chain"`` -> :func:`fit_prepared` then :func:`run_xspec_chain` on the
      still-loaded XSPEC session; the :class:`XspecChainResult` is attached under
      ``result["xspec_chain"]`` and the ``dict`` is returned.
    - ``"bxa"``   -> :func:`jinwu.core.bxa_fit.fit_prepared_bxa` (imported lazily
      inside the body to avoid a circular import, since ``bxa_fit`` reuses this
      module's private helpers); returns a ``BXAFitResult``.

    ``kwargs`` are forwarded to the underlying fit function.  Chain-specific
    options are supplied via ``chain_path`` / ``chain_kwargs`` so they never
    collide with ``fit_prepared`` arguments.
    """
    if settings is not None and not isinstance(settings, FitConfig):
        raise TypeError("settings must be a FitConfig instance or None")
    if method is None:
        method = settings.method if settings is not None else get_fit_settings().method
    resolved_method = str(method).lower()
    if resolved_method not in {"mle", "chain", "bxa"}:
        raise ValueError(
            f"Unknown fit method: {method!r}; expected 'mle', 'chain' or 'bxa'"
        )

    output = Path(outdir).expanduser().resolve()
    fit_kwargs: dict[str, Any] = dict(kwargs)
    if settings is not None:
        # An explicit per-call ``settings`` supplies defaults for the three
        # sentinel parameters, but never overrides values the caller passed.
        fit_kwargs.setdefault("stat_method", settings.statistic)
        fit_kwargs.setdefault("abundance", settings.abundance)
        fit_kwargs.setdefault("cross_section", settings.cross_section)

    if resolved_method == "bxa":
        from jinwu.core.bxa_fit import fit_prepared_bxa

        return fit_prepared_bxa(prepared, outdir=output, **fit_kwargs)

    result = fit_prepared(prepared, outdir=output, **fit_kwargs)
    if resolved_method == "mle":
        return result

    resolved_chain_path = (
        Path(chain_path).expanduser().resolve()
        if chain_path is not None
        else output / "chain.fits"
    )
    chain_result = run_xspec_chain(
        chain_path=resolved_chain_path,
        **(dict(chain_kwargs) if chain_kwargs else {}),
    )
    result["xspec_chain"] = chain_result
    return result


def _compact_xray_fit(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the serializable scientific surface of a candidate fit."""

    keys = (
        "model_key", "model_family", "absorption_mode", "model", "parameters",
        "statistics", "effective_statistic", "free_parameter_count", "energy_range", "energy_ranges",
        "flux_abs", "rate", "fit_products", "report_txt", "plot_fit", "plot_fits",
        "warnings", "xspec_settings", "derived_parameters", "metrics",
    )
    return {key: result.get(key) for key in keys if key in result}


def _parameter_by_name(
    parameters: Mapping[str, Mapping[str, Any]],
    requested: str,
) -> Mapping[str, Any] | None:
    requested_lower = requested.lower()
    for name, value in parameters.items():
        if name.lower() == requested_lower:
            return value
    return None


def _metric_value(metrics: ModelFitMetrics, metric: str) -> float:
    value = getattr(metrics, metric)
    if value is None:
        raise ValueError(f"Metric {metric} is unavailable")
    return float(value)


# 方法：Akaike 权重 w_i=exp(-0.5Δ_i)/Σ_j exp(-0.5Δ_j)（Δ 为相对最优模型的 AICc 差，AICc 不可用时退回 AIC），表示"模型 i 为给定候选集中最优近似"的相对可信度
# 参考：Burnham & Anderson 2004, Sociol. Methods Res. 33, 261；Burnham & Anderson 2002, Model Selection and Multimodel Inference, 2nd ed., Springer
def _decorate_model_metrics(
    metrics: Mapping[str, ModelFitMetrics],
    requested_metric: str,
) -> tuple[dict[str, ModelFitMetrics], tuple[str, ...], str, list[str]]:
    metric = requested_metric.lower()
    if metric not in {"aic", "aicc", "bic"}:
        raise ValueError("selection_metric must be 'aic', 'aicc', or 'bic'")
    warnings_list: list[str] = []
    if metric == "aicc" and any(item.aicc is None for item in metrics.values()):
        metric = "aic"
        warnings_list.append(
            "AICc is unavailable for at least one candidate because n <= k + 1; "
            "ranking fell back to AIC."
        )
    ranking = tuple(sorted(metrics, key=lambda key: _metric_value(metrics[key], metric)))
    best = _metric_value(metrics[ranking[0]], metric)
    deltas = {key: _metric_value(value, metric) - best for key, value in metrics.items()}
    best_aic = min(item.aic for item in metrics.values())
    best_bic = min(item.bic for item in metrics.values())
    delta_aic = {key: item.aic - best_aic for key, item in metrics.items()}
    delta_bic = {key: item.bic - best_bic for key, item in metrics.items()}
    if all(item.aicc is not None for item in metrics.values()):
        best_aicc = min(float(item.aicc) for item in metrics.values())
        delta_aicc = {
            key: float(item.aicc) - best_aicc for key, item in metrics.items()
        }
        weight_deltas = delta_aicc
    else:
        delta_aicc = {key: None for key in metrics}
        weight_deltas = delta_aic
    normalizer = sum(math.exp(-0.5 * delta) for delta in weight_deltas.values())
    decorated = {
        key: replace(
            value,
            delta=deltas[key],
            delta_aic=delta_aic[key],
            delta_aicc=delta_aicc[key],
            delta_bic=delta_bic[key],
            akaike_weight=(math.exp(-0.5 * weight_deltas[key]) / normalizer),
            ranking_metric=metric,
        )
        for key, value in metrics.items()
    }
    return decorated, ranking, metric, warnings_list


def _nh_interval_touches_zero(result: Mapping[str, Any]) -> tuple[bool, str]:
    parameter = _parameter_by_name(result.get("parameters") or {}, "zTBabs.nH")
    if parameter is None:
        return True, "zTBabs.nH is missing"
    value = float(parameter.get("value", 0.0))
    error_low = parameter.get("error_lo")
    if error_low is None or not math.isfinite(float(error_low)):
        return True, "lower profile error is unavailable"
    return value - float(error_low) <= 0.0, "90% profile interval reaches zero"


def _candidate_is_well_constrained(
    spec: XRayModelSpec,
    result: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    parameters = result.get("parameters") or {}
    problems: list[str] = []
    critical_parameters = list(spec.critical_parameters)
    if spec.absorption_mode == "free":
        critical_parameters.append("zTBabs.nH")
    for name in critical_parameters:
        parameter = _parameter_by_name(parameters, name)
        if parameter is None:
            problems.append(f"missing critical parameter {name}")
            continue
        value = float(parameter.get("value", math.nan))
        error_low = parameter.get("error_lo")
        error_high = parameter.get("error_hi")
        if not math.isfinite(value):
            problems.append(f"non-finite {name}")
        if error_low is None or error_high is None:
            problems.append(f"profile error unavailable for {name}")
        elif not all(math.isfinite(float(item)) and float(item) > 0 for item in (error_low, error_high)):
            problems.append(f"invalid profile error for {name}")
        lower = parameter.get("min")
        upper = parameter.get("max")
        scale = max(abs(value), 1.0)
        span = (
            float(upper) - float(lower)
            if lower is not None and upper is not None
            else 0.0
        )
        boundary_tolerance = max(1e-5 * scale, 1e-3 * max(span, 0.0))
        if lower is not None and value - float(lower) <= boundary_tolerance:
            problems.append(f"{name} is at its lower bound")
        if upper is not None and float(upper) - value <= boundary_tolerance:
            problems.append(f"{name} is at its upper bound")
        if error_low is not None and lower is not None:
            if value - float(error_low) <= float(lower):
                problems.append(f"{name} profile interval reaches its lower bound")
        if error_high is not None and upper is not None:
            if value + float(error_high) >= float(upper):
                problems.append(f"{name} profile interval reaches its upper bound")
    return not problems, problems


# 方法：ΔC 的朴素似然比 p 值按 χ²1 上尾计算：p=P(χ²1>ΔC)=0.5*erfc(sqrt(ΔC/2))；因检验对象 nH=0 位于参数空间边界，朴素 χ² LRT p 值不成立（正确零分布为 0.5χ²0+0.5χ²1 混合），故该值仅作描述性参考（字段名 boundary_lrt_p_reference_only）并显式警告、不使用 F 检验
# 参考：Baker & Cousins 1984, Nucl. Instrum. Methods Phys. Res. A 221, 437（2ΔlnL↔χ²）；Protassov et al. 2002, ApJ 571, 545（边界参数 LRT 禁用约定）
def _absorption_comparisons(
    candidates: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, ModelFitMetrics],
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for family in ("powerlaw", "bbody", "bknpower"):
        free_key = f"{family}_free_nh"
        zero_key = f"{family}_nh0"
        if free_key not in candidates or zero_key not in candidates:
            continue
        free_metric = metrics[free_key]
        zero_metric = metrics[zero_key]
        parameter = _parameter_by_name(
            candidates[free_key].get("parameters") or {}, "zTBabs.nH"
        ) or {}
        xspec_settings = candidates[free_key].get("xspec_settings") or {}
        delta_c = zero_metric.statistic - free_metric.statistic
        boundary_p = (
            1.0
            if delta_c <= 0.0
            else 0.5 * math.erfc(math.sqrt(delta_c / 2.0))
        )
        comparisons[family] = {
            "free_key": free_key,
            "zero_key": zero_key,
            "delta_c_zero_minus_free": delta_c,
            "delta_aic_zero_minus_free": zero_metric.aic - free_metric.aic,
            "delta_aicc_zero_minus_free": (
                None
                if zero_metric.aicc is None or free_metric.aicc is None
                else zero_metric.aicc - free_metric.aicc
            ),
            "delta_bic_zero_minus_free": zero_metric.bic - free_metric.bic,
            "nh_best": parameter.get("value"),
            "nh_error_lo": parameter.get("error_lo"),
            "nh_error_hi": parameter.get("error_hi"),
            "profile_error": {
                key: xspec_settings.get(key)
                for key in (
                    "error_delta_stat",
                    "profile_error_kind",
                    "profile_confidence",
                    "profile_sigma",
                    "profile_error_label",
                    "profile_errors_succeeded",
                )
            },
            "boundary_lrt_p_reference_only": boundary_p,
            "warning": (
                "The boundary likelihood-ratio p-value is descriptive only; "
                "no F-test is used for intrinsic absorption."
            ),
        }
    return comparisons


# 方法：模型采用的决策规则：信息准则差 Δ<2 视为统计不可区分并按简约性选自由参数更少者；挑战模型需较基线改进 Δ≥6（"相当支持"量级）且关键参数受约束良好才被采纳（2.0/6.0 为约定经验阈值）
# 参考：Burnham & Anderson 2004, Sociol. Methods Res. 33, 261（Δ 阈值经验标尺：≤2 等价、4-7 相当差异、>10 实质无支持）
def _choose_xray_model(
    specs: Mapping[str, XRayModelSpec],
    candidates: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, ModelFitMetrics],
    ranking: Sequence[str],
    metric: str,
) -> tuple[str, str, list[str]]:
    warnings_list: list[str] = []

    def score(key: str) -> float:
        return _metric_value(metrics[key], metric)

    def best_family(family: str) -> str | None:
        keys = [key for key in ranking if specs[key].family == family]
        if not keys:
            return None
        free = next((key for key in keys if specs[key].absorption_mode == "free"), None)
        zero = next((key for key in keys if specs[key].absorption_mode == "zero"), None)
        if free and zero:
            touches_zero, _ = _nh_interval_touches_zero(candidates[free])
            if touches_zero and score(zero) - score(free) <= 2.0:
                return zero
            if abs(score(zero) - score(free)) < 2.0:
                return min((free, zero), key=lambda key: metrics[key].free_parameters)
        best = keys[0]
        equivalent = [key for key in keys if score(key) - score(best) < 2.0]
        return min(equivalent, key=lambda key: metrics[key].free_parameters)

    baseline = best_family("powerlaw")
    if baseline is None:
        raw = ranking[0]
        return raw, "No power-law baseline succeeded; adopted the best available candidate.", warnings_list

    adopted = baseline
    reason = (
        f"Adopted the preferred power-law candidate {baseline}; models within "
        "Delta criterion < 2 were treated as indistinguishable and simplified."
    )
    baseline_constrained, baseline_problems = _candidate_is_well_constrained(
        specs[baseline], candidates[baseline]
    )
    if not baseline_constrained:
        warnings_list.append(
            f"Adopted power-law baseline {baseline} has constrained-parameter "
            f"diagnostics requiring review: {'; '.join(baseline_problems)}."
        )
    challengers = []
    for family in ("apec", "bbody", "bknpower"):
        key = best_family(family)
        if key is None:
            continue
        improvement = score(baseline) - score(key)
        constrained, problems = _candidate_is_well_constrained(specs[key], candidates[key])
        if improvement >= 6.0 and constrained:
            challengers.append((improvement, key))
        elif improvement >= 6.0:
            warnings_list.append(
                f"{key} improved {metric.upper()} by {improvement:.3g} but was not "
                f"auto-adopted: {'; '.join(problems)}."
            )
    if challengers:
        improvement, adopted = max(challengers)
        reason = (
            f"{adopted} replaced the preferred power-law model because it improved "
            f"{metric.upper()} by {improvement:.3g} and its critical parameters had "
            "valid profile errors away from configured bounds."
        )
    return adopted, reason, warnings_list


def _write_xray_candidate_summary(
    path: Path,
    spec: XRayModelSpec,
    result: Mapping[str, Any] | None,
    metrics: ModelFitMetrics | None,
    failure: str | None = None,
) -> None:
    lines = [f"候选模型：{spec.key}", f"XSPEC 表达式：{spec.expression}"]
    if spec.family == "apec":
        lines.append("吸收假设：按配置完全不包含银河系或本征吸收。")
    else:
        lines.append(
            "吸收假设：银河系 TBabs 固定；"
            + ("本征 zTBabs.nH 自由。" if spec.absorption_mode == "free" else "本征 zTBabs.nH 固定为 0。")
        )
    if failure is not None:
        lines.append(f"拟合失败：{failure}")
    elif result is not None and metrics is not None:
        settings = result.get("xspec_settings") or {}
        if not settings.get("calculate_errors", True):
            lines.append("XSPEC error 未启用，以下仅报告 best-fit 参数。")
        else:
            lines.append(
                "XSPEC error："
                + str(settings.get("profile_error_label") or "single-parameter profile interval")
                + f"；delta statistic={settings.get('error_delta_stat', 'N/A')}。"
                + ("" if settings.get("profile_errors_succeeded", True) else "（error 命令整体执行失败，以下误差不可用）")
            )
        lines.append(
            f"统计量：{result.get('effective_statistic', 'unknown')}="
            f"{metrics.statistic:.6g}/{metrics.dof}；k={metrics.free_parameters}；"
            f"AIC={metrics.aic:.6g}；AICc={metrics.aicc if metrics.aicc is not None else 'N/A'}；"
            f"BIC={metrics.bic:.6g}；Delta={metrics.delta if metrics.delta is not None else 'N/A'}；"
            f"weight={metrics.akaike_weight if metrics.akaike_weight is not None else 'N/A'}。"
        )
        lines.append("参数：")
        for name, parameter in (result.get("parameters") or {}).items():
            value = parameter.get("value")
            if parameter.get("frozen"):
                lines.append(f"  {name}={value}（固定）")
            else:
                error_lo = parameter.get("error_lo")
                error_hi = parameter.get("error_hi")
                if error_lo is not None and error_hi is not None:
                    lines.append(f"  {name}={value} (-{error_lo}/+{error_hi})")
                else:
                    status = parameter.get("error_status")
                    if status and status not in ("ok",):
                        lines.append(f"  {name}={value}（误差不可用：{status}）")
                    else:
                        lines.append(f"  {name}={value}")
        flux_parameter = _parameter_by_name(
            result.get("parameters") or {}, "cflux.lg10Flux"
        )
        if flux_parameter is not None:
            lg_flux = float(flux_parameter["value"])
            lines.append(f"未吸收 flux={10.0 ** lg_flux:.6g} erg s^-1 cm^-2。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_xray_failure_log(
    candidate_dir: Path,
    spec: XRayModelSpec,
    failure: str,
) -> Path:
    """Promote any partial XSPEC transcript into one stable failure log."""

    log_path = candidate_dir / "fit_failure.log"
    lines = [
        "[jinwu_xray_candidate_failure]",
        f"model_key: {spec.key}",
        f"model_expression: {spec.expression}",
        f"error: {failure}",
    ]
    temporary_logs = sorted(candidate_dir.glob("*.tmp.log"))
    if temporary_logs:
        lines.append("")
        lines.append("[captured_xspec_transcript]")
    for path in temporary_logs:
        try:
            transcript = path.read_text(encoding="utf-8", errors="replace").rstrip()
        except OSError as exc:
            transcript = f"<could not read {path.name}: {exc}>"
        lines.extend((f"--- {path.name} ---", transcript))
        try:
            path.unlink()
        except OSError:
            pass
    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return log_path


def fit_xray_models(
    prepared,
    *,
    outdir: str | Path,
    model_class: str = "auto",
    absorption_mode: str = "auto",
    candidate_keys: Sequence[str] | None = None,
    selection_metric: str = "aicc",
    galactic_nh_1e22: float | None = None,
    redshift: float = 0.0,
    error_delta_stat: float = 1.0,
    **fit_kwargs: Any,
) -> XRayModelComparisonResult:
    """Fit and compare a controlled set of XSPEC X-ray spectral models."""

    specs_tuple = resolve_xray_model_specs(
        model_class=model_class,
        absorption_mode=absorption_mode,
        candidate_keys=candidate_keys,
    )
    specs = {spec.key: spec for spec in specs_tuple}
    output = Path(outdir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, dict[str, Any]] = {}
    raw_metrics: dict[str, ModelFitMetrics] = {}
    failures: dict[str, str] = {}
    warning_messages: list[str] = []

    forbidden = {
        "model_name", "intrinsic_nh_mode", "error_command", "error_delta_stat",
        "freeze_galactic_nh",
    }
    overlap = forbidden.intersection(fit_kwargs)
    if overlap:
        raise TypeError(f"fit_xray_models controls these arguments: {', '.join(sorted(overlap))}")
    if not math.isfinite(float(redshift)) or float(redshift) < 0:
        raise ValueError("redshift must be finite and non-negative")
    error_metadata = _profile_error_metadata(error_delta_stat)
    if any("tbabs" in spec.expression.lower() for spec in specs_tuple):
        if galactic_nh_1e22 is None:
            raise ValueError(
                "galactic_nh_1e22 is required when any candidate contains TBabs"
            )
        if not math.isfinite(float(galactic_nh_1e22)) or float(galactic_nh_1e22) < 0:
            raise ValueError("galactic_nh_1e22 must be finite and non-negative")

    for spec in specs_tuple:
        candidate_dir = output / "models" / spec.key
        candidate_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = fit_prepared(
                prepared,
                outdir=candidate_dir,
                model_name=spec.expression,
                intrinsic_nh_mode=(
                    "zero" if spec.absorption_mode == "zero" else "free"
                ),
                error_delta_stat=float(error_delta_stat),
                galactic_nh_1e22=galactic_nh_1e22,
                freeze_galactic_nh=True,
                redshift=redshift,
                **fit_kwargs,
            )
            statistics = result.get("statistics") or {}
            metrics = calculate_model_fit_metrics(
                statistics["value"],
                statistics["dof"],
                result.get("free_parameter_count", 0),
            )
            result["model_key"] = spec.key
            result["model_family"] = spec.family
            result["absorption_mode"] = spec.absorption_mode
            result["derived_parameters"] = {}
            if spec.family == "bbody":
                parameter = _parameter_by_name(result.get("parameters") or {}, "bbody.kT")
                if parameter is not None:
                    result["derived_parameters"]["rest_kT_keV"] = (
                        (1.0 + float(redshift)) * float(parameter["value"])
                    )
            if spec.family == "bknpower":
                parameter = _parameter_by_name(result.get("parameters") or {}, "bknpower.BreakE")
                if parameter is not None:
                    result["derived_parameters"]["rest_break_energy_keV"] = (
                        (1.0 + float(redshift)) * float(parameter["value"])
                    )
            if spec.family == "apec":
                message = (
                    "APEC candidate intentionally omits both Galactic and intrinsic "
                    "absorption; its information criterion tests that distinct assumption."
                )
                result.setdefault("warnings", []).append(message)
            candidates[spec.key] = result
            raw_metrics[spec.key] = metrics
            _write_xray_candidate_summary(
                candidate_dir / "summary_zh.txt", spec, result, metrics
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            failures[spec.key] = failure
            failure_log = _write_xray_failure_log(candidate_dir, spec, failure)
            # A failed comparison candidate must not advertise a replayable
            # XSPEC session, even if the exception occurred after XSPEC saved it.
            for xcm in candidate_dir.glob("*.xcm"):
                try:
                    xcm.unlink()
                except OSError:
                    pass
            failure_payload = {
                "model_key": spec.key,
                "model_expression": spec.expression,
                "status": "failed",
                "error": failure,
                "xspec_log": str(failure_log),
            }
            (candidate_dir / "fit_failure.json").write_text(
                json.dumps(failure_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_xray_candidate_summary(
                candidate_dir / "summary_zh.txt", spec, None, None, failure
            )

    if not candidates:
        raise RuntimeError(
            "All X-ray model candidates failed: "
            + "; ".join(f"{key}: {value}" for key, value in failures.items())
        )

    metrics, ranking, effective_metric, metric_warnings = _decorate_model_metrics(
        raw_metrics, selection_metric
    )
    warning_messages.extend(metric_warnings)
    adopted_key, adopted_reason, selection_warnings = _choose_xray_model(
        specs, candidates, metrics, ranking, effective_metric
    )
    warning_messages.extend(selection_warnings)
    absorption = _absorption_comparisons(candidates, metrics)
    for key, result in candidates.items():
        result["metrics"] = {
            "statistic": metrics[key].statistic,
            "dof": metrics[key].dof,
            "free_parameters": metrics[key].free_parameters,
            "effective_bins": metrics[key].effective_bins,
            "aic": metrics[key].aic,
            "aicc": metrics[key].aicc,
            "bic": metrics[key].bic,
            "delta": metrics[key].delta,
            "delta_aic": metrics[key].delta_aic,
            "delta_aicc": metrics[key].delta_aicc,
            "delta_bic": metrics[key].delta_bic,
            "akaike_weight": metrics[key].akaike_weight,
            "ranking_metric": metrics[key].ranking_metric,
        }
        result_json = (result.get("fit_products") or {}).get("result_json")
        if result_json:
            Path(result_json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        _write_xray_candidate_summary(
            output / "models" / key / "summary_zh.txt",
            specs[key],
            result,
            metrics[key],
        )

    comparison = XRayModelComparisonResult(
        candidates=candidates,
        metrics=metrics,
        failures=failures,
        ranking=ranking,
        adopted_key=adopted_key,
        adopted_reason=adopted_reason,
        selection_metric=effective_metric,
        warnings=tuple(warning_messages),
        absorption_comparisons=absorption,
    )
    json_path = output / "model_comparison.json"
    text_path = output / "model_comparison.txt"
    comparison.comparison_json = str(json_path)
    comparison.comparison_txt = str(text_path)
    json_path.write_text(
        json.dumps(comparison.to_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    profile_error_line = (
        "Profile errors: " + str(error_metadata["profile_error_label"])
        if fit_kwargs.get("calculate_errors", True)
        else "Profile errors: disabled"
    )
    text_lines = [
        f"Adopted model: {adopted_key}",
        f"Reason: {adopted_reason}",
        f"Ranking metric: {effective_metric}",
        profile_error_line,
        f"Profile error delta statistic: {error_metadata['error_delta_stat']:g}",
        "",
        "key statistic/dof k AIC AICc BIC delta weight",
    ]
    for key in ranking:
        item = metrics[key]
        text_lines.append(
            f"{key} {item.statistic:.6g}/{item.dof} {item.free_parameters} "
            f"{item.aic:.6g} {item.aicc if item.aicc is not None else 'N/A'} "
            f"{item.bic:.6g} {item.delta:.6g} {item.akaike_weight:.6g}"
        )
    if failures:
        text_lines.extend(["", "Failed candidates:"])
        text_lines.extend(f"{key}: {value}" for key, value in failures.items())
    if warning_messages:
        text_lines.extend(["", "Warnings:", *warning_messages])
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return comparison


def _validate_fit_catalogs(catalogs) -> None:
    from jinwu.core.instruments import Catalog

    if not catalogs:
        raise TypeError("fit() requires at least one Catalog")
    for catalog in catalogs:
        if not isinstance(catalog, Catalog):
            raise TypeError("fit() accepts Catalog inputs from jinwu.core.instruments.scan()")
        if not catalog.bundles:
            raise RuntimeError(f"Catalog has no spectrum bundles ready for fit: {catalog.root}")
        for manifest in catalog.manifests:
            if manifest.instrument.upper() != "WXT":
                continue
            source_ids = {
                bundle.source_id
                for bundle in manifest.bundles
                if bundle.source_id is not None
            }
            if len(source_ids) > 1:
                raise ValueError(
                    "fit() requires one WXT source per Catalog; "
                    "call catalog.select_source('sN') first"
                )


def _fit_prepare_outdir(catalogs, prepare_outdir: str | Path | None) -> Path | None:
    if prepare_outdir is not None:
        return Path(prepare_outdir).expanduser().resolve()
    if len(catalogs) == 1:
        return None

    parents = {catalog.root.expanduser().resolve().parent for catalog in catalogs}
    if len(parents) != 1:
        raise ValueError(
            "Joint fit Catalog roots must share one parent directory, "
            "or pass prepare_outdir explicitly"
        )
    return parents.pop() / "jointfit"


def _fit_catalog_input(catalogs, *, prepare_outdir: Path | None):
    from jinwu.core.instruments import Catalog

    if len(catalogs) == 1:
        return catalogs[0]
    root = prepare_outdir if prepare_outdir is not None else catalogs[0].root.parent
    return Catalog(
        root=root,
        manifests=[
            manifest
            for catalog in catalogs
            for manifest in catalog.manifests
        ],
        warnings=[
            warning
            for catalog in catalogs
            for warning in catalog.warnings
        ],
    )


def fit(
    *catalogs: Catalog,
    outdir: str | Path | None = None,
    prepare_outdir: str | Path | None = None,
    overwrite: bool = True,
    energy_ranges: Mapping[str, tuple[float, float]] | None = None,
    **fit_kwargs,
) -> dict:
    """Prepare scanned spectrum catalogs and run one XSPEC fit."""
    from jinwu.core.spectrum_prep import prepare_spectra

    if "group_min" in fit_kwargs:
        raise TypeError("fit() uses InstrumentConfig.group_min_counts and has no group_min override")

    _validate_fit_catalogs(catalogs)
    resolved_prepare_outdir = _fit_prepare_outdir(catalogs, prepare_outdir)
    prepared = prepare_spectra(
        _fit_catalog_input(catalogs, prepare_outdir=resolved_prepare_outdir),
        outdir=resolved_prepare_outdir,
        overwrite=overwrite,
    )
    if not prepared.spectra or any(not spectrum.ready for spectrum in prepared.spectra):
        raise RuntimeError("fit() requires all prepared spectra to be ready")

    prepared_input = prepared.spectra[0] if len(prepared.spectra) == 1 else tuple(prepared.spectra)
    fit_outdir = Path(outdir).expanduser().resolve() if outdir is not None else prepared.root / "fit"
    results = fit_prepared(
        prepared_input,
        outdir=fit_outdir,
        energy_ranges=energy_ranges,
        **fit_kwargs,
    )
    results["catalogs"] = tuple(catalogs)
    results["prepared_catalog"] = prepared
    results["prepared_spectra"] = tuple(prepared.spectra)
    results["prepare_root"] = str(prepared.root)
    return results
