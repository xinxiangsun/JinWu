

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
from typing import Callable, Optional, Dict, Any, Literal, Union, Sequence
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import curve_fit, minimize

from jinwu.core.file import LightcurveData
from jinwu.core.datasets import LightcurveDataset

__all__ = [
    "FitResult",
    "LightcurveFitter",
    "ModelRegistry",
    # 内置模型函数
    "powerlaw", "broken_powerlaw", "double_broken_powerlaw", "exponential", "gaussian", 
    "constant", "linear", "smoothly_broken_powerlaw",
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
        模型函数引用，签名为 model(t, *params)
    
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


# ---------- 内置模型函数 ----------

def powerlaw(t: np.ndarray, norm: float, index: float, t0: float = 1.0) -> np.ndarray:
    """幂律模型: norm * (t/t0)^index
    
    参数
    ----
    t : 时间
    norm : 归一化常数
    index : 幂律指数
    t0 : 参考时间（默认1.0）
    """
    return norm * np.power(t / t0, index)


def broken_powerlaw(t: np.ndarray, norm: float, index1: float, index2: float, t_break: float) -> np.ndarray:
    """分段幂律: 在 t_break 处从 index1 转为 index2
    
    参数
    ----
    norm : t_break 处的归一化
    index1 : t < t_break 的指数
    index2 : t >= t_break 的指数
    t_break : 转折时间
    """
    result = np.empty_like(t)
    mask1 = t < t_break
    mask2 = ~mask1
    result[mask1] = norm * np.power(t[mask1] / t_break, index1)
    result[mask2] = norm * np.power(t[mask2] / t_break, index2)
    return result


def smoothly_broken_powerlaw(
    t: np.ndarray, norm: float, index1: float, index2: float, t_break: float, smoothness: float = 0.3
) -> np.ndarray:
    """平滑分段幂律（Willingale 2007 风格）
    
    F(t) = norm * [(t/t_break)^(-index1*s) + (t/t_break)^(-index2*s)]^(-1/s)
    
    参数
    ----
    smoothness : 平滑参数（越小越接近硬转折）
    """
    x = t / t_break
    s = smoothness
    term1 = np.power(x, -index1 * s)
    term2 = np.power(x, -index2 * s)
    return norm * np.power(term1 + term2, -1.0 / s)


def double_broken_powerlaw(
    t: np.ndarray, 
    norm: float, 
    index1: float, 
    index2: float, 
    index3: float,
    t_break1: float,
    t_break2: float
) -> np.ndarray:
    """三段幂律（双折断）: 在 t_break1 和 t_break2 处发生两次转折
    
    适用于早期平台→中期过渡→后期陡衰减的典型 GRB 光变曲线
    
    参数
    ----
    norm : 第一个折断处 (t=t_break1) 的通量归一化
    index1 : t < t_break1 的幂律指数（早期，通常接近0表示平台）
    index2 : t_break1 <= t < t_break2 的幂律指数（中期，通常为负陡衰减）
    index3 : t >= t_break2 的幂律指数（后期，通常为负缓衰减）
    t_break1 : 第一个转折时间（早期→中期）
    t_break2 : 第二个转折时间（中期→后期）
    
    返回
    ----
    flux : 三段连续幂律，在 t_break1 和 t_break2 处保持连续
    
    注记
    ----
    - 在 t=t_break1 处: F = norm
    - 在 t=t_break2 处: 自动匹配以保持连续性
    - 各段采用相对该段起点的归一化，确保折断处连续
    """
    result = np.empty_like(t)
    
    # 分段：早期、中期、后期
    mask1 = t < t_break1
    mask2 = (t >= t_break1) & (t < t_break2)
    mask3 = t >= t_break2
    
    # 早期（相对 t_break1，斜率 index1）
    # F(t) = norm * (t/t_break1)^index1
    result[mask1] = norm * np.power(t[mask1] / t_break1, index1)
    
    # 中期（相对 t_break1，斜率 index2）
    # 在 t=t_break1 处与早期段连续: F(t_break1) = norm
    result[mask2] = norm * np.power(t[mask2] / t_break1, index2)
    
    # 后期（相对 t_break2，斜率 index3）
    # 在 t=t_break2 处与中期段连续: F(t_break2) = norm * (t_break2/t_break1)^index2
    norm_late = norm * np.power(t_break2 / t_break1, index2)
    result[mask3] = norm_late * np.power(t[mask3] / t_break2, index3)
    
    return result


def exponential(t: np.ndarray, norm: float, decay: float, t0: float = 0.0) -> np.ndarray:
    """指数衰减: norm * exp(-decay * (t - t0))
    
    参数
    ----
    norm : 初始幅度
    decay : 衰减率
    t0 : 起始时间
    """
    return norm * np.exp(-decay * (t - t0))


def gaussian(t: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    """高斯脉冲: amplitude * exp(-0.5*((t-mean)/sigma)^2)"""
    return amplitude * np.exp(-0.5 * np.power((t - mean) / sigma, 2))


def constant(t: np.ndarray, level: float) -> np.ndarray:
    """常数模型"""
    return np.full_like(t, level)


def linear(t: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """线性模型: slope * t + intercept"""
    return slope * t + intercept


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
        self._models: Dict[str, tuple[Callable, tuple[str, ...]]] = {}
        self._register_builtin()
    
    def _register_builtin(self):
        """注册内置模型"""
        self.register("powerlaw", powerlaw, ("norm", "index", "t0"))
        self.register("broken_powerlaw", broken_powerlaw, ("norm", "index1", "index2", "t_break"))
        self.register("double_broken_powerlaw", double_broken_powerlaw, 
                     ("norm", "index1", "index2", "index3", "t_break1", "t_break2"))
        self.register("smoothly_broken_powerlaw", smoothly_broken_powerlaw, 
                     ("norm", "index1", "index2", "t_break", "smoothness"))
        self.register("exponential", exponential, ("norm", "decay", "t0"))
        self.register("gaussian", gaussian, ("amplitude", "mean", "sigma"))
        self.register("constant", constant, ("level",))
        self.register("linear", linear, ("slope", "intercept"))
    
    def register(self, name: str, func: Callable, param_names: tuple[str, ...]):
        """注册新模型
        
        参数
        ----
        name : 模型名称
        func : 模型函数，签名 func(t, *params)
        param_names : 参数名称元组
        """
        self._models[name] = (func, param_names)
    
    def get(self, name: str) -> tuple[Callable, tuple[str, ...]]:
        """获取模型函数与参数名"""
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
        # SciPy 接受任何 array-like；这里放宽类型并允许 None 作为“无界”
        bounds: Optional[tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]] = None,
        method: Literal["curve_fit", "least_squares"] = "curve_fit",
        sigma: Optional[np.ndarray] = None,
        absolute_sigma: bool = False,
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
        method : {"curve_fit", "least_squares"}
            拟合方法
        sigma : array, optional
            覆盖数据误差（默认使用 self.error）
        absolute_sigma : bool
            是否将 sigma 视为绝对误差（影响协方差缩放）
        **kwargs : 传递给 scipy.optimize.curve_fit 的额外参数
        
        返回
        ----
        FitResult : 拟合结果对象
        """
        # 解析模型
        if isinstance(model, str):
            model_func, pnames = self.registry.get(model)
            model_name = model
        elif callable(model):
            model_func = model
            model_name = getattr(model, "__name__", "custom")
            if param_names is None:
                raise ValueError("param_names must be provided for custom model functions")
            pnames = param_names
        else:
            raise TypeError("model must be str or callable")
        
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
        
        # 检查是否为幂律模型，若是则在对数空间拟合
        powerlaw_models = {"powerlaw", "broken_powerlaw", "double_broken_powerlaw", "smoothly_broken_powerlaw"}
        use_log_fitting = model_name in powerlaw_models
        
        errors_lower = None
        errors_upper = None
        
        if use_log_fitting:
            # 对数空间拟合
            popt, pcov, success, message, errors_lower, errors_upper = self._fit_powerlaw_logspace(
                model_name, model_func, pnames, p0, sigma, absolute_sigma, bounds, **kwargs
            )
        else:
            # 标准拟合
            # 处理边界：None 代表无界，相当于 (-inf, inf)
            bounds_to_use: tuple[Any, Any]
            if bounds is None:
                bounds_to_use = (-np.inf, np.inf)
            else:
                bounds_to_use = bounds

            try:
                popt, pcov = curve_fit(
                    model_func,
                    self.time,
                    self.value,
                    p0=p0,
                    sigma=sigma,
                    absolute_sigma=absolute_sigma,
                    bounds=bounds_to_use,
                    **kwargs,
                )
                success = True
                message = "Fit converged successfully"
            except Exception as e:
                warnings.warn(f"Fit failed: {e}")
                popt = p0
                pcov = None
                success = False
                message = str(e)
        
        # 计算拟合曲线与统计量
        fitted = model_func(self.time, *popt)
        residuals = self.value - fitted
        chisq = np.sum((residuals / sigma) ** 2)
        dof = len(self.time) - len(popt)
        reduced_chisq = chisq / dof if dof > 0 else np.inf
        
        # 提取参数误差
        errors = None
        if pcov is not None:
            try:
                errors = np.sqrt(np.diag(pcov))
            except Exception:
                errors = None
        
        return FitResult(
            model_name=model_name,
            params=popt,
            param_names=pnames,
            covariance=pcov,
            errors=errors,
            errors_lower=errors_lower,
            errors_upper=errors_upper,
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
            model=model_func,
        )
    
    def _guess_initial_params(self, model_name: str, param_names: tuple[str, ...]) -> np.ndarray:
        """简单的初值猜测启发式"""
        n = len(param_names)
        p0 = np.ones(n)
        
        # 数据统计量
        t_mid = np.median(self.time)
        t_span = self.time.max() - self.time.min()
        v_mean = np.mean(self.value)
        v_max = np.max(self.value)
        
        if model_name == "powerlaw":
            p0 = np.array([v_mean, -1.0, t_mid])
        elif model_name == "broken_powerlaw":
            p0 = np.array([v_mean, -1.0, -2.0, t_mid])
        elif model_name == "double_broken_powerlaw":
            # 三段: 早期平台(~0)、中期(-1.5)、后期(-2.5), 折断在 1/3 和 2/3 处
            t1 = np.percentile(self.time, 33)
            t2 = np.percentile(self.time, 67)
            p0 = np.array([v_max, 0.0, -1.5, -2.5, t1, t2])
        elif model_name == "smoothly_broken_powerlaw":
            p0 = np.array([v_mean, -1.0, -2.0, t_mid, 0.3])
        elif model_name == "exponential":
            p0 = np.array([v_max, 1.0 / t_span, self.time.min()])
        elif model_name == "gaussian":
            p0 = np.array([v_max, t_mid, t_span / 4.0])
        elif model_name == "constant":
            p0 = np.array([v_mean])
        elif model_name == "linear":
            # 简单线性回归斜率估计
            slope = (self.value[-1] - self.value[0]) / (self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0
            p0 = np.array([slope, v_mean])
        
        return p0
    
    def _fit_powerlaw_logspace(
        self,
        model_name: str,
        model_func: Callable,
        param_names: tuple[str, ...],
        p0: np.ndarray,
        sigma: np.ndarray,
        absolute_sigma: bool,
        bounds: Optional[tuple],
        **kwargs,
    ) -> tuple[np.ndarray, Optional[np.ndarray], bool, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """在对数空间拟合幂律模型（内部线性拟合）
        
        参数
        ----
        model_name : str
            模型名称
        model_func : callable
            原始模型函数（用于最终评估）
        param_names : tuple
            参数名称
        p0 : ndarray
            初始猜测（原始空间）
        sigma : ndarray
            误差
        absolute_sigma : bool
            是否使用绝对误差
        bounds : tuple or None
            参数边界
        **kwargs : dict
            额外参数
        
        返回
        ----
        popt : ndarray
            优化后的参数（原始空间）
        pcov : ndarray or None
            协方差矩阵
        success : bool
            是否成功
        message : str
            状态消息
        """
        # 转换到对数空间
        log_t = np.log10(self.time)
        log_y = np.log10(self.value)
        
        # 误差传播: σ(log y) ≈ σ(y) / (y * ln(10))
        log_sigma = sigma / (self.value * np.log(10))
        
        # 定义对数空间的线性模型
        if model_name == "powerlaw":
            # log(F) = log(norm) + index * log(t/t0)
            #        = [log(norm) - index*log(t0)] + index * log(t)
            #        = a + b * log(t)
            def log_model_pl(log_t, a, b):
                return a + b * log_t
            
            # 转换初值: p0 = [norm, index, t0]
            # a = log(norm) - index * log(t0), b = index
            log_p0 = [np.log10(p0[0]) - p0[1] * np.log10(p0[2]), p0[1]]
            
            # 转换边界
            if bounds is not None:
                # bounds = ([norm_min, index_min, t0_min], [norm_max, index_max, t0_max])
                # 简化：只限制 index
                log_bounds = ([- np.inf, bounds[0][1]], [np.inf, bounds[1][1]])
            else:
                log_bounds = (-np.inf, np.inf)
            
        elif model_name == "broken_powerlaw":
            # 分段线性模型
            def log_model_bpl(log_t, a, b1, b2, log_tb):
                result = np.zeros_like(log_t)
                mask1 = log_t < log_tb
                mask2 = log_t >= log_tb
                result[mask1] = a + b1 * log_t[mask1]
                result[mask2] = a + b1 * log_tb + b2 * (log_t[mask2] - log_tb)
                return result
            
            # 转换初值: p0 = [norm, index1, index2, t_break]
            log_p0 = [np.log10(p0[0]), p0[1], p0[2], np.log10(p0[3])]
            
            # 转换边界
            if bounds is not None:
                log_bounds = (
                    [-np.inf, bounds[0][1], bounds[0][2], np.log10(bounds[0][3])],
                    [np.inf, bounds[1][1], bounds[1][2], np.log10(bounds[1][3])]
                )
            else:
                log_bounds = (-np.inf, np.inf)
            
        elif model_name == "double_broken_powerlaw":
            # 三段线性模型
            def log_model_dbpl(log_t, a, b1, b2, b3, log_tb1, log_tb2):
                result = np.zeros_like(log_t)
                mask1 = log_t < log_tb1
                mask2 = (log_t >= log_tb1) & (log_t < log_tb2)
                mask3 = log_t >= log_tb2
                result[mask1] = a + b1 * log_t[mask1]
                result[mask2] = a + b1 * log_tb1 + b2 * (log_t[mask2] - log_tb1)
                result[mask3] = (a + b1 * log_tb1 + b2 * (log_tb2 - log_tb1) +
                                b3 * (log_t[mask3] - log_tb2))
                return result
            
            # 转换初值: p0 = [norm, index1, index2, index3, t_break1, t_break2]
            log_p0 = [np.log10(p0[0]), p0[1], p0[2], p0[3], np.log10(p0[4]), np.log10(p0[5])]
            
            # 转换边界
            if bounds is not None:
                log_bounds = (
                    [-np.inf, bounds[0][1], bounds[0][2], bounds[0][3], 
                     np.log10(bounds[0][4]), np.log10(bounds[0][5])],
                    [np.inf, bounds[1][1], bounds[1][2], bounds[1][3],
                     np.log10(bounds[1][4]), np.log10(bounds[1][5])]
                )
            else:
                log_bounds = (-np.inf, np.inf)
                
        elif model_name == "smoothly_broken_powerlaw":
            # 平滑折断幂律不适合简单线性化，使用标准拟合
            warnings.warn(f"{model_name} 无法线性化，使用标准拟合")
            bounds_to_use = bounds if bounds is not None else (-np.inf, np.inf)
            try:
                popt, pcov = curve_fit(
                    model_func, self.time, self.value,
                    p0=p0, sigma=sigma, absolute_sigma=absolute_sigma,
                    bounds=bounds_to_use, **kwargs
                )
                return popt, pcov, True, "Fit converged (standard fitting)", None, None
            except Exception as e:
                warnings.warn(f"Fit failed: {e}")
                return p0, None, False, str(e), None, None
        else:
            raise ValueError(f"Unknown powerlaw model: {model_name}")
        
        # 执行对数空间拟合
        try:
            log_popt, log_pcov = curve_fit(
                log_model_pl if model_name == "powerlaw" else (
                    log_model_bpl if model_name == "broken_powerlaw" else log_model_dbpl
                ),
                log_t, log_y,
                p0=log_p0, sigma=log_sigma, absolute_sigma=absolute_sigma,
                bounds=log_bounds, **kwargs
            )
            
            # 转换回原始空间参数
            if model_name == "powerlaw":
                # log_popt = [a, b] -> [norm, index, t0]
                # 固定 t0=1 简化：norm = 10^a, index = b
                popt = np.array([10**log_popt[0], log_popt[1], 1.0])
                
                # 计算非对称误差
                log_errors = np.sqrt(np.diag(log_pcov))  # 对数空间的对称误差
                
                # norm参数: 从 10^(a ± σ_a) 得到非对称误差
                norm_center = 10**log_popt[0]
                norm_upper = 10**(log_popt[0] + log_errors[0]) - norm_center
                norm_lower = norm_center - 10**(log_popt[0] - log_errors[0])
                
                # index参数: 对数空间为线性,误差对称
                index_err = log_errors[1]
                
                # t0固定为1,误差为0
                errors_lower = np.array([norm_lower, index_err, 0.0])
                errors_upper = np.array([norm_upper, index_err, 0.0])
                
                # 扩展协方差矩阵以匹配3参数
                # 对数空间参数 [a, b] 对应原空间 [norm, index]
                # norm = 10^a => σ(norm) = norm * ln(10) * σ(a)
                # t0固定为1，误差为0
                pcov_expanded = np.zeros((3, 3))
                pcov_expanded[0, 0] = (10**log_popt[0] * np.log(10))**2 * log_pcov[0, 0]  # var(norm)
                pcov_expanded[1, 1] = log_pcov[1, 1]  # var(index)
                pcov_expanded[0, 1] = pcov_expanded[1, 0] = (10**log_popt[0] * np.log(10)) * log_pcov[0, 1]
                pcov = pcov_expanded
                
            elif model_name == "broken_powerlaw":
                # log_popt = [a, b1, b2, log_tb] -> [norm, index1, index2, t_break]
                popt = np.array([10**log_popt[0], log_popt[1], log_popt[2], 10**log_popt[3]])
                
                # 计算非对称误差
                log_errors = np.sqrt(np.diag(log_pcov))
                
                # norm参数: 从 10^(a ± σ_a) 得到非对称误差
                norm_center = 10**log_popt[0]
                norm_upper = 10**(log_popt[0] + log_errors[0]) - norm_center
                norm_lower = norm_center - 10**(log_popt[0] - log_errors[0])
                
                # index1, index2: 线性参数,误差对称
                index1_err = log_errors[1]
                index2_err = log_errors[2]
                
                # t_break: 从 10^(log_tb ± σ_log_tb) 得到非对称误差
                tb_center = 10**log_popt[3]
                tb_upper = 10**(log_popt[3] + log_errors[3]) - tb_center
                tb_lower = tb_center - 10**(log_popt[3] - log_errors[3])
                
                errors_lower = np.array([norm_lower, index1_err, index2_err, tb_lower])
                errors_upper = np.array([norm_upper, index1_err, index2_err, tb_upper])
                
                # 扩展协方差矩阵
                pcov_expanded = np.zeros((4, 4))
                # norm = 10^a
                pcov_expanded[0, 0] = (10**log_popt[0] * np.log(10))**2 * log_pcov[0, 0]
                # index1, index2 直接对应
                pcov_expanded[1, 1] = log_pcov[1, 1]
                pcov_expanded[2, 2] = log_pcov[2, 2]
                # t_break = 10^log_tb
                pcov_expanded[3, 3] = (10**log_popt[3] * np.log(10))**2 * log_pcov[3, 3]
                # 协方差项（简化处理，仅对角元素）
                pcov = pcov_expanded
                
            elif model_name == "double_broken_powerlaw":
                # log_popt = [a, b1, b2, b3, log_tb1, log_tb2]
                popt = np.array([
                    10**log_popt[0], log_popt[1], log_popt[2], log_popt[3],
                    10**log_popt[4], 10**log_popt[5]
                ])
                
                # 计算非对称误差
                log_errors = np.sqrt(np.diag(log_pcov))
                
                # norm参数: 从 10^(a ± σ_a) 得到非对称误差
                norm_center = 10**log_popt[0]
                norm_upper = 10**(log_popt[0] + log_errors[0]) - norm_center
                norm_lower = norm_center - 10**(log_popt[0] - log_errors[0])
                
                # index参数: 线性,误差对称
                index1_err = log_errors[1]
                index2_err = log_errors[2]
                index3_err = log_errors[3]
                
                # t_break参数: 从 10^(log_tb ± σ_log_tb) 得到非对称误差
                tb1_center = 10**log_popt[4]
                tb1_upper = 10**(log_popt[4] + log_errors[4]) - tb1_center
                tb1_lower = tb1_center - 10**(log_popt[4] - log_errors[4])
                
                tb2_center = 10**log_popt[5]
                tb2_upper = 10**(log_popt[5] + log_errors[5]) - tb2_center
                tb2_lower = tb2_center - 10**(log_popt[5] - log_errors[5])
                
                errors_lower = np.array([norm_lower, index1_err, index2_err, index3_err, 
                                         tb1_lower, tb2_lower])
                errors_upper = np.array([norm_upper, index1_err, index2_err, index3_err,
                                         tb1_upper, tb2_upper])
                
                # 扩展协方差矩阵
                pcov_expanded = np.zeros((6, 6))
                pcov_expanded[0, 0] = (10**log_popt[0] * np.log(10))**2 * log_pcov[0, 0]
                pcov_expanded[1, 1] = log_pcov[1, 1]
                pcov_expanded[2, 2] = log_pcov[2, 2]
                pcov_expanded[3, 3] = log_pcov[3, 3]
                pcov_expanded[4, 4] = (10**log_popt[4] * np.log(10))**2 * log_pcov[4, 4]
                pcov_expanded[5, 5] = (10**log_popt[5] * np.log(10))**2 * log_pcov[5, 5]
                pcov = pcov_expanded
            
            success = True
            message = "Fit converged successfully (log-space linear fitting)"
            
        except Exception as e:
            warnings.warn(f"Log-space fit failed: {e}, trying standard fitting")
            # 回退到标准拟合
            bounds_to_use = bounds if bounds is not None else (-np.inf, np.inf)
            try:
                popt, pcov = curve_fit(
                    model_func, self.time, self.value,
                    p0=p0, sigma=sigma, absolute_sigma=absolute_sigma,
                    bounds=bounds_to_use, **kwargs
                )
                success = True
                message = "Fit converged (fallback to standard fitting)"
                errors_lower = None  # 标准拟合不产生非对称误差
                errors_upper = None
            except Exception as e2:
                popt = p0
                pcov = None
                success = False
                message = f"Both log-space and standard fitting failed: {e}, {e2}"
                errors_lower = None
                errors_upper = None
        
        return popt, pcov, success, message, errors_lower, errors_upper
    
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
            try:
                model_func, _ = self.registry.get(result.model_name)
            except ValueError:
                raise ValueError("model_func must be provided for custom models")
        
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
                
                # Double broken powerlaw: 三段 α1,α2,α3 + t_break1, t_break2
                if model_name == 'double_broken_powerlaw':
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
