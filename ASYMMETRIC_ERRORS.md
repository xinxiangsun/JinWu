# 非对称误差实现说明 (Asymmetric Error Implementation)

## 概述

`jinwu.core.fit` 模块现在支持非对称误差估计,特别适用于在对数空间拟合的幂律模型。

## 实现原理

### 1. 对数空间拟合与误差转换

对于幂律模型,我们在对数空间进行线性拟合:

```
log(F) = log(norm) + index * log(t)
```

设对数空间参数为 `a = log(norm)`,其1-sigma误差为 `σ_a`,则:

- **中心值**: `norm = 10^a`
- **上界**: `norm_upper = 10^(a + σ_a)`
- **下界**: `norm_lower = 10^(a - σ_a)`

由于指数函数的非线性,误差不对称:

```
Δnorm_upper = norm_upper - norm = 10^a * (10^σ_a - 1)
Δnorm_lower = norm - norm_lower = 10^a * (1 - 10^(-σ_a))
```

通常 `Δnorm_upper > Δnorm_lower`,非对称性约3-5%。

### 2. FitResult数据结构更新

```python
@dataclass
class FitResult:
    params: np.ndarray           # 拟合参数
    errors: np.ndarray | None    # 对称误差(向后兼容)
    errors_lower: np.ndarray | None  # 下误差(新增)
    errors_upper: np.ndarray | None  # 上误差(新增)
    # ...其他字段
```

### 3. 误差显示格式

#### 对称误差:

```
norm: 104.4 ± 3.094
```

#### 非对称误差:

```
norm: 104.4 +3.14 -3.048
```

`summary()` 方法自动选择合适的格式。

## 支持的模型

| 模型                     | 非对称参数               | 对称参数               |
| ------------------------ | ------------------------ | ---------------------- |
| powerlaw                 | norm                     | index, t0              |
| broken_powerlaw          | norm, t_break            | index1, index2         |
| double_broken_powerlaw   | norm, t_break1, t_break2 | index1, index2, index3 |
| smoothly_broken_powerlaw | 无(标准拟合)             | 所有参数               |

## 使用示例

```python
from jinwu.core.fit import LightcurveFitter

# 准备数据
fitter = LightcurveFitter((time, flux, flux_err))

# 拟合幂律模型(自动使用对数空间线性拟合)
result = fitter.fit("powerlaw")

# 查看非对称误差
print(result.summary())

# 输出示例:
# === Fit Result: powerlaw ===
# Parameters:
#   norm: 104.4 +3.14 -3.048
#   index: -1.515 +0.007535 -0.007535
#   t0: 1 +0 -0

# 手动访问非对称误差
if result.errors_lower is not None:
    for i, name in enumerate(result.param_names):
        print(f"{name}: {result.params[i]:.4g} "
              f"+{result.errors_upper[i]:.4g} "
              f"-{result.errors_lower[i]:.4g}")
```

## 技术细节

### 对数空间拟合流程

1. **数据转换**: `log_y = log10(y)`, `log_t = log10(t)`
2. **误差传播**: `σ_log = σ_y / (y * ln(10))`
3. **线性拟合**: 在 `(log_t, log_y)` 空间使用 `curve_fit`
4. **参数转换**:
   - norm: `10^a`
   - t_break: `10^log_tb`
   - index: 直接使用
5. **非对称误差计算**:
   - 对数参数: `10^(x ± σ)` 产生非对称
   - 线性参数: `±σ` 保持对称

### 协方差矩阵处理

对数空间协方差矩阵 `log_pcov` 转换为原空间:

```python
# norm = 10^a 的方差
var(norm) = (norm * ln(10))^2 * var(a)

# t_break = 10^log_tb 的方差  
var(t_break) = (t_break * ln(10))^2 * var(log_tb)

# index 的方差直接对应
var(index) = var(b)
```

## 对比XSPEC

**XSPEC**:

- 默认线性空间拟合,χ² = Σ((O-M)/σ)²
- 使用Cash/C-stat处理低计数
- 可通过MCMC获取非对称误差

**jinwu (log-space fitting)**:

- 对数空间拟合,χ² = Σ((log O - log M)/σ_log)²
- 自动产生非对称误差(通过变换)
- 不同的χ²定义,结果不直接可比

## 测试验证

运行测试脚本验证功能:

```bash
conda run -n hea python test_asymmetric_errors.py
conda run -n hea python test_broken_powerlaw_errors.py
```

或查看演示notebook:

```bash
jupyter notebook demo_asymmetric_errors.ipynb
```

## 向后兼容性

- `errors` 字段保留,存储对称误差(协方差矩阵对角元的平方根)
- 旧代码仍可使用 `result.errors`
- 新代码优先使用 `result.errors_lower` 和 `result.errors_upper`

## 参考

1. Bevington & Robinson, "Data Reduction and Error Analysis for the Physical Sciences"
2. XSPEC Manual: https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/manual.html
3. Press et al., "Numerical Recipes"
