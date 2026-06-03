

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Dict, Any, Literal, Mapping, Union, Sequence
from pathlib import Path
import math
import os
import re
import warnings

import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import LMLSQFitter, TRFLSQFitter

from jinwu.core.data import LightcurveData

if TYPE_CHECKING:
    from jinwu.core.datasets import LightcurveDataset
    from jinwu.core.instruments import Catalog

__all__ = [
    "FitResult",
    "LightcurveFitter",
    "ModelRegistry",
    "XspecChainParameter",
    "XspecChainResult",
    "fit_spectrum",
    "fit_spectrum_from_files",
    "fit",
    "fit_prepared",
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
    mcmc_samples : np.ndarray | None
        MCMC 采样结果 (n_samples, n_params)，仅在使用贝叶斯方法时可用
    mcmc_logprob : np.ndarray | None
        MCMC 对数概率，仅在使用贝叶斯方法时可用
    
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
    mcmc_samples: Optional[np.ndarray] = None
    mcmc_logprob: Optional[np.ndarray] = None
    
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
            - 若为 callable：直接使用自定义函数 model(t, *params)
        p0 : array-like, optional
            初始参数猜测；若为 None 则尝试自动估计
        param_names : tuple[str, ...], optional
            参数名称（仅用于自定义函数）
        bounds : 2-tuple of array-like, optional
            参数边界 (lower, upper)
        method : {"curve_fit", "least_squares", "lmfit"}
            拟合方法
        sigma : array, optional
            覆盖数据误差（默认使用 self.error）
        absolute_sigma : bool
            是否将 sigma 视为绝对误差（影响协方差缩放）
        bayesian : bool
            是否使用贝叶斯 MCMC 方法（需要 emcee）
        mcmc_nwalkers : int
            MCMC walker 数量（仅 bayesian=True 时）
        mcmc_nsteps : int
            MCMC 采样步数（仅 bayesian=True 时）
        mcmc_burn : int
            MCMC burn-in 步数（仅 bayesian=True 时）
        mcmc_thin : int
            MCMC 采样间隔（仅 bayesian=True 时）
        use_lmfit : bool
            是否使用 astropy.modeling 进行拟合（默认 True）
        lmfit_method : str
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
        # 防止 0 或 NaN 权重导致发散
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
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
            
            # 计算误差（从协方差矩阵），并对固定参数补零以匹配参数总数
            if hasattr(fitter, 'fit_info') and 'param_cov' in fitter.fit_info:
                pcov_raw = fitter.fit_info['param_cov']
                if pcov_raw is not None:
                    errors_raw = np.sqrt(np.diag(pcov_raw))
                    # 若存在固定参数（如 powerlaw.t0 固定），协方差矩阵维度会小于参数总数
                    # 此时对误差与协方差进行零填充以匹配 pnames 长度
                    if errors_raw.size < len(pnames):
                        errors = np.zeros(len(pnames))
                        # 将前面对应的自由参数误差填入（astropy 通常按可变参数顺序）
                        errors[:errors_raw.size] = errors_raw
                        pcov = np.zeros((len(pnames), len(pnames)))
                        pcov[:errors_raw.size, :errors_raw.size] = pcov_raw
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
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
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
        ax1.plot(t_fine, y_fine, '-', label=f'Fit: {result.model_name}', linewidth=2)
        
        # 坐标轴改为对数
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylabel(ylabel)
        ax1.legend()
        ax1.grid(alpha=0.3)
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
                alpha=0.6,
            )
            ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
            # 残差图：横轴用对数，纵轴保留线性以显示正负
            ax2.set_xscale('log')
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel('Residuals')
            ax2.grid(alpha=0.3)
        elif not show_residuals:
            ax1.set_xlabel(xlabel)
        
        try:
            plt.tight_layout()
        except (ValueError, RuntimeError) as e:
            # tight_layout 可能在某些情况下失败（特别是智能标签放置后）
            # 静默忽略，布局可能略有偏差但不影响使用
            pass
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
            "xspec module is required for spectral fitting. "
            "Please install heasoftpy and set up XSPEC environment."
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


def _generate_xspec_result(model, spectrum) -> dict:
    """
    根据XSPEC模型和光谱自动生成结果字典

    参数:
        model: XSPEC模型对象
        spectrum: XSPEC光谱对象

    返回:
        包含模型参数、flux、rate等信息的字典
    """
    xspec = _require_xspec()

    lines = []
    result = {}
    result['model'] = model.expression

    result['parameters'] = {}
    lines.append(f"Model: {model.expression}")

    processed_params = set()

    for comp_name in model.componentNames:
        try:
            comp = getattr(model, comp_name)

            for param_name in comp.parameterNames:
                param_key = f"{comp_name}.{param_name}"
                if param_key in processed_params:
                    continue
                processed_params.add(param_key)

                param = getattr(comp, param_name)
                param_val = param.values[0]

                param_dict = {
                    'value': param_val,
                    'frozen': param.frozen
                }

                if not param.frozen:
                    try:
                        array = np.array(param.error[:2]) - param_val
                        err_lo = abs(array[0])
                        err_hi = abs(array[1])
                        param_dict['error_lo'] = err_lo
                        param_dict['error_hi'] = err_hi
                        lines.append(f"{comp_name}.{param_name}: {param_val:.4f} (-{err_lo:.4f}, +{err_hi:.4f})(1sigma error)")
                    except Exception:
                        lines.append(f"{comp_name}.{param_name}: {param_val:.4f} (error calculation failed)")
                else:
                    lines.append(f"{comp_name}.{param_name}: {param_val:.4f} (fixed)")

                result['parameters'][param_key] = param_dict

        except Exception as e:
            continue

    try:
        emin = model.cflux.Emin.values[0] if hasattr(model, 'cflux') else None
        emax = model.cflux.Emax.values[0] if hasattr(model, 'cflux') else None
        xspec.AllModels.calcFlux(f"{emin} {emax}")
        flux_erg = float(spectrum.flux[0])
        flux_photons = float(spectrum.flux[3])
    except Exception:
        flux_erg = None
        flux_photons = None
        emin = None
        emax = None

    result['flux_abs'] = {
        'erg_cm2_s': flux_erg,
        'photons_cm2_s': flux_photons
    }

    if flux_erg is not None and emin is not None and emax is not None:
        lines.append(f"Absorbed Flux ({emin:.1f}-{emax:.1f} keV): {flux_erg:.4e} erg/cm²/s")
        lines.append(f"Absorbed Photon Flux ({emin:.1f}-{emax:.1f} keV): {flux_photons:.4e} photons/cm²/s")

    try:
        rate = float(spectrum.rate[0])
        rate_err = float(spectrum.rate[1]) if len(spectrum.rate) > 1 else None
    except Exception:
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
        except Exception:
            conv_factor = None
    else:
        conv_factor = None
    photon_counts = rate * exposure if rate is not None and exposure is not None else None
    result['conversion'] = {
        'exposure_s': exposure,
        'erg_per_count': conv_factor,
        'counts': photon_counts
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
    except Exception:
        result['statistics'] = {}

    result['text'] = "\n".join(lines)

    return result


def fit_spectrum(
    phapath: str | Path,
    rmfpath: str | Path,
    outdir: str | Path,
    *,
    bkgpath: Optional[str | Path] = None,
    arfpath: Optional[str | Path] = None,
    group_min: int = 1,
    emin: float = 0.5,
    emax: float = 4.0,
    model_name: str = "tbabs*ztbabs*cflux*powerlaw",
    redshift: float = 0.0,
    save_pha: bool = True,
    clobber: bool = True,
) -> dict:
    """
    执行XSPEC光谱拟合的自动化流程

    参数:
        phapath: 输入的PHA文件路径
        rmfpath: RMF响应文件路径
        outdir: 输出目录
        bkgpath: 背景文件路径（可选）
        arfpath: ARF有效面积文件路径（可选）
        group_min: 最小计数分组数（默认1）
        emin: 能量下限 keV（默认0.5）
        emax: 能量上限 keV（默认4.0）
        model_name: XSPEC模型名称（默认'tbabs*ztbabs*cflux*powerlaw'）
        redshift: 源红移（默认0.0）
        save_pha: 是否保存分组后的PHA文件
        clobber: 是否覆盖已存在的文件

    返回:
        包含拟合结果的字典

    示例:
        >>> result = fit_spectrum(
        ...     phapath="source.pha",
        ...     rmfpath="response.rmf",
        ...     outdir="output",
        ...     group_min=1,
        ...     emin=0.5,
        ...     emax=4.0,
        ...     model_name="tbabs*ztbabs*cflux*powerlaw",
        ...     redshift=0.1
        ... )
    """
    import xspec
    from jinwu.ftools.grppha import grppha as jinwu_grppha

    phapath = Path(phapath)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grouped_pha = outdir / f"grouped_g{group_min}.pha"

    jinwu_grppha(
        input_pha=str(phapath),
        outfile=str(grouped_pha) if save_pha else None,
        min_counts=group_min,
        overwrite=clobber,
    )

    xspec.AllData.clear()
    xspec.AllModels.clear()
    xspec.Xset.abund = "wilm"
    xspec.Fit.query = "yes"
    xspec.Fit.statMethod = "cstat"

    s1 = xspec.Spectrum(str(grouped_pha))

    if bkgpath is not None:
        s1.background = str(bkgpath)
    if arfpath is not None:
        s1.arf = str(arfpath)
    s1.response = str(rmfpath)

    xspec.AllData.ignore("bad")
    s1.ignore(f"**-{emin} {emax}-**")

    m = xspec.Model(model_name)

    m.TBabs.nH = 1.0
    m.TBabs.nH.frozen = True

    if "ztbabs" in model_name.lower():
        m.zTBabs.nH = 0.5
        m.zTBabs.nH.min = 0.0
        m.zTBabs.nH.max = 100.0
        if redshift > 0:
            m.zTBabs.Redshift = redshift
            m.zTBabs.Redshift.frozen = True

    if hasattr(m, "cflux"):
        m.cflux.Emin = emin
        m.cflux.Emax = emax
        if hasattr(m.cflux, "lg10Flux"):
            m.cflux.lg10Flux = -10.0
            m.cflux.lg10Flux.min = -20.0
            m.cflux.lg10Flux.max = 10.0

    if hasattr(m, "powerlaw"):
        m.powerlaw.PhoIndex = 2.0
        m.powerlaw.PhoIndex.min = 0.0
        m.powerlaw.PhoIndex.max = 9.0

    if hasattr(m, "zpowerlw"):
        m.zpowerlw.PhoIndex = 2.0
        m.zpowerlw.PhoIndex.min = 0.0
        m.zpowerlw.PhoIndex.max = 9.0
        if redshift > 0:
            m.zpowerlw.Redshift = redshift
            m.zpowerlw.Redshift.frozen = True

    xspec.Fit.perform()

    param_str = ""
    if "ztbabs" in model_name.lower() and hasattr(m, "zTBabs") and hasattr(m, "cflux"):
        param_str += f"1. {m.zTBabs.nH.index} "
    if hasattr(m, "cflux") and hasattr(m.cflux, "lg10Flux"):
        param_str += f"1. {m.cflux.lg10Flux.index} "
    if hasattr(m, "powerlaw"):
        param_str += f"1. {m.powerlaw.PhoIndex.index}"
    elif hasattr(m, "zpowerlw"):
        param_str += f"1. {m.zpowerlw.PhoIndex.index}"

    if param_str.strip():
        xspec.Fit.error(param_str.strip())

    results = _generate_xspec_result(m, s1)

    results["group_min"] = group_min
    results["energy_range"] = {"emin": emin, "emax": emax}
    results["input_files"] = {
        "pha": str(phapath),
        "grouped_pha": str(grouped_pha),
        "rmf": str(rmfpath),
        "bkg": str(bkgpath) if bkgpath else None,
        "arf": str(arfpath) if arfpath else None,
    }

    return results


def fit_spectrum_from_files(
    phafile: str | Path,
    rmfobj: Any,
    outdir: str | Path,
    *,
    bkgfile: Optional[str | Path] = None,
    arffile: Optional[str | Path] = None,
    group_min: int = 1,
    emin: float = 0.5,
    emax: float = 4.0,
    model_name: str = "tbabs*ztbabs*cflux*powerlaw",
    redshift: float = 0.0,
    save_pha: bool = True,
    clobber: bool = True,
    srcname: Optional[str] = None,
    instname: Optional[str] = None,
    nhgal_kw: Optional[dict] = None,
    plottype: str = "ldata_eeufspec_delchi",
    plot_backend: str = "matplotlib",
    plot_outdir: Optional[str | Path] = None,
    plot_format: str = "png",
    plot_density: int = 300,
) -> dict:
    """
    从文件路径执行完整的XSPEC光谱拟合，包括绘图

    这是一个高级封装，组合了光谱分组、XSPEC拟合和结果绘图

    参数:
        phafile: 输入的PHA文件路径
        rmfobj: RMF对象（Path字符串或RMFData对象）
        outdir: 输出目录
        bkgfile: 背景文件路径（可选）
        arffile: ARF有效面积文件路径（可选）
        group_min: 最小计数分组数（默认1）
        emin: 能量下限 keV（默认0.5）
        emax: 能量上限 keV（默认4.0）
        model_name: XSPEC模型名称
        redshift: 源红移（默认0.0）
        save_pha: 是否保存分组后的PHA文件
        clobber: 是否覆盖已存在的文件
        srcname: 源名称（用于绘图标题）
        instname: 仪器名称（用于绘图标题）
        nhgal_kw: nhgal函数的关键字参数（可选）
        plottype: 兼容参数；当前拟合图固定为 ldata + eeufspec + delchi 三联图
        plot_backend: 绘图后端，'matplotlib'或'xspec'
        plot_outdir: 绘图输出目录（默认与outdir相同）
        plot_format: 绘图输出格式
        plot_density: 绘图分辨率

    返回:
        包含拟合结果和图表路径的字典

    示例:
        >>> result = fit_spectrum_from_files(
        ...     phafile="source.pha",
        ...     rmfobj="response.rmf",
        ...     outdir="output",
        ...     srcname="MySource",
        ...     instname="WXT"
        ... )
    """
    import xspec
    from jinwu.core.plot import plotfit

    rmfpath = str(rmfobj) if isinstance(rmfobj, (str, Path)) else None

    results = fit_spectrum(
        phapath=phafile,
        rmfpath=rmfpath,
        outdir=outdir,
        bkgpath=bkgfile,
        arfpath=arffile,
        group_min=group_min,
        emin=emin,
        emax=emax,
        model_name=model_name,
        redshift=redshift,
        save_pha=save_pha,
        clobber=clobber,
    )

    if srcname is None:
        srcname = Path(phafile).stem
    if instname is None:
        instname = "INST"

    plotdir = plot_outdir if plot_outdir is not None else outdir

    try:
        figurefit, _ = plotfit(
            srcname=srcname,
            instname=instname,
            group_min=group_min,
            modelname=model_name,
            redshift=redshift,
            plottype=plottype,
            outputdir=plotdir,
            backend=plot_backend,
            output_format=plot_format,
            density=plot_density,
        )
        results["plot_fit"] = str(figurefit) if figurefit else None
    except Exception as e:
        results["plot_fit"] = None
        results["plot_fit_error"] = str(e)

    return results


def _model_parameter_index(model, parameter) -> int:
    start = getattr(model, "startParIndex", None)
    index = int(parameter.index)
    if start in (None, 1):
        return index
    return int(start) + index - 1


def _prepared_error_parameters(model, model_name: str, models=None) -> str:
    parameters = []
    if "ztbabs" in model_name.lower() and hasattr(model, "zTBabs"):
        if hasattr(model.zTBabs, "nH"):
            parameters.append(f"1. {_model_parameter_index(model, model.zTBabs.nH)}")
    for group_model in models or [model]:
        if hasattr(group_model, "cflux") and hasattr(group_model.cflux, "lg10Flux"):
            parameters.append(
                f"1. {_model_parameter_index(group_model, group_model.cflux.lg10Flux)}"
            )
    if hasattr(model, "powerlaw") and hasattr(model.powerlaw, "PhoIndex"):
        parameters.append(f"1. {_model_parameter_index(model, model.powerlaw.PhoIndex)}")
    elif hasattr(model, "zpowerlw") and hasattr(model.zpowerlw, "PhoIndex"):
        parameters.append(f"1. {_model_parameter_index(model, model.zpowerlw.PhoIndex)}")
    return " ".join(parameters)


def _set_prepared_parameter_bounds(parameter, value: float, lower: float, upper: float) -> None:
    """Set bounds on fake test parameters and real PyXspec parameters."""
    try:
        parameter.min = lower
        parameter.max = upper
    except Exception:
        parameter.values = f"{value},,{lower},{lower},{upper},{upper}"


def _configure_prepared_model(model, *, model_name: str, emin: float, emax: float, redshift: float) -> None:
    if hasattr(model, "TBabs") and hasattr(model.TBabs, "nH"):
        model.TBabs.nH = 1.0
        model.TBabs.nH.frozen = True

    if "ztbabs" in model_name.lower() and hasattr(model, "zTBabs"):
        if hasattr(model.zTBabs, "nH"):
            model.zTBabs.nH = 0.5
            _set_prepared_parameter_bounds(model.zTBabs.nH, 0.5, 0.0, 100.0)
        if redshift > 0 and hasattr(model.zTBabs, "Redshift"):
            model.zTBabs.Redshift = redshift
            model.zTBabs.Redshift.frozen = True

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
            model.powerlaw.norm = 1.0
            model.powerlaw.norm.frozen = True

    if hasattr(model, "zpowerlw") and hasattr(model.zpowerlw, "PhoIndex"):
        model.zpowerlw.PhoIndex = 2.0
        _set_prepared_parameter_bounds(model.zpowerlw.PhoIndex, 2.0, 0.0, 9.0)
        if redshift > 0 and hasattr(model.zpowerlw, "Redshift"):
            model.zpowerlw.Redshift = redshift
            model.zpowerlw.Redshift.frozen = True


def _prepared_input_dict(prepared) -> dict[str, str | int | None]:
    return {
        "instrument": prepared.instrument,
        "obsid": prepared.obsid,
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
    if model_name.lower().replace(" ", "") != "tbabs*ztbabs*cflux*powerlaw":
        return
    if len(models) <= 1:
        return
    reference = models[0]
    for model in models[1:]:
        if hasattr(reference, "zTBabs") and hasattr(model, "zTBabs"):
            if hasattr(reference.zTBabs, "nH") and hasattr(model.zTBabs, "nH"):
                _link_parameter(model.zTBabs.nH, reference.zTBabs.nH)
        if hasattr(reference, "powerlaw") and hasattr(model, "powerlaw"):
            if hasattr(reference.powerlaw, "PhoIndex") and hasattr(model.powerlaw, "PhoIndex"):
                _link_parameter(model.powerlaw.PhoIndex, reference.powerlaw.PhoIndex)


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
    redshift: float = 0.0,
    stat_method: str = "cstat",
    calculate_errors: bool = True,
    error_command: str | None = None,
    srcname: str | None = None,
    instname: str | None = None,
    plot_backend: str = "matplotlib",
    plot_format: str = "png",
    plot_density: int = 300,
) -> dict:
    """Fit one prepared spectrum or multiple prepared spectra with XSPEC."""
    from jinwu.core.plot import plotfit
    from jinwu.core.spectrum_prep import PreparedJointSpectrum, PreparedSpectrum

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
    xspec.Xset.abund = "wilm"
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
        )
    _link_default_prepared_model_groups(group_models, model_name)
    xspec.Fit.perform()

    command = error_command or _prepared_error_parameters(model, model_name, group_models)
    error_text = ""
    if calculate_errors and command:
        try:
            error_text = _capture_xspec_log(
                xspec,
                output / "fit_prepared.xspec_error.tmp.log",
                lambda: xspec.Fit.error(command),
                warnings_list,
            )
        except Exception as exc:
            warnings_list.append(f"XSPEC error calculation failed: {exc}")

    results = _generate_xspec_result(model, xspec_spectra[0])
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
        group_result = _generate_xspec_result(group_model, xspec_spectra[spectrum_index - 1])
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
    results["prepared"] = prepared
    results["report_txt"] = str(report)
    results["warnings"] = warnings_list
    results["group_min"] = prepared_spectra[0].group_min
    results["group_mins"] = {
        _prepared_spectrum_key(spectrum): spectrum.group_min
        for spectrum in prepared_spectra
    }

    try:
        figure_path, _ = plotfit(
            srcname=srcname,
            instname=instname,
            group_min=prepared_spectra[0].group_min,
            modelname=model_name,
            redshift=redshift,
            outputdir=output,
            backend=plot_backend,
            output_format=plot_format,
            density=plot_density,
        )
        results["plot_fit"] = str(figure_path) if figure_path else None
    except Exception as exc:
        results["plot_fit"] = None
        results["plot_fit_error"] = str(exc)
        warnings_list.append(f"fit plot failed: {exc}")
    return results


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
