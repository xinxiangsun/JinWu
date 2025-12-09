# Jinwu 拟合功能使用指南

## 概述

`jinwu.core.fit` 模块现已集成三种拟合方法：

1. **SciPy** - 标准最小二乘拟合（默认）
2. **lmfit** - 增强的拟合框架，提供更好的参数管理和多种优化算法
3. **emcee** - 贝叶斯 MCMC 采样，获取完整的参数后验分布

## ✨ 新功能：自动初值和边界

**从现在开始，所有模型都有智能的默认初值和边界！**

- **幂律指数范围**: -10 到 3（覆盖所有常见情况）
- **初值自动判断**: 根据数据趋势自动选择上升/下降初值
- **边界智能生成**: 基于数据统计量自动设置合理边界

## 安装依赖

```bash
# 基础功能（仅 scipy）
pip install numpy scipy

# 使用 lmfit（推荐）
pip install lmfit

# 使用贝叶斯方法
pip install emcee

# 可视化 MCMC 结果（可选）
pip install corner
```

## 使用示例

### 0. 最简单的调用方式（完全自动）✨ 新！

```python
from jinwu.core.fit import LightcurveFitter
import numpy as np

# 准备数据
t = np.array([...])  # 时间
y = np.array([...])  # 通量
yerr = np.array([...])  # 误差

# 创建拟合器
fitter = LightcurveFitter((t, y, yerr))

# ✓ 完全自动：初值和边界都自动生成
result = fitter.fit('smoothly_broken_powerlaw')

# ✓ 也可以只提供初值，边界自动生成
result = fitter.fit('smoothly_broken_powerlaw', p0=[1e-9, -1.0, -2.0, 100, 0.5])

# ✓ 或只提供边界，初值自动生成
result = fitter.fit('smoothly_broken_powerlaw', 
                   bounds=([0, -10, -10, t.min(), 0.1], 
                          [np.inf, 3, 3, t.max(), 2.0]))

print(result.summary())
```

### 1. 标准 SciPy 拟合

```python
# 传统方式：手动指定初值和边界（仍然支持）
result = fitter.fit(
    'smoothly_broken_powerlaw',
    p0=[1e-9, -1.0, -2.0, 100, 0.5],
    bounds=([0, -5, -5, t.min(), 0.1], 
            [np.inf, 5, 5, t.max(), 2.0])
)

print(result.summary())
```

### 2. lmfit 拟合（推荐用于复杂模型）

```python
# ✨ 使用自动初值和边界
result_lmfit = fitter.fit(
    'smoothly_double_broken_powerlaw',
    use_lmfit=True,
    lmfit_method='least_squares'  # 可选: leastsq, nelder, powell, differential_evolution
)

# 或手动指定（传统方式）
result_lmfit = fitter.fit(
    'smoothly_double_broken_powerlaw',
    p0=[1e-9, 0, -1.5, -2.5, 30, 400, 0.5, 0.5],
    bounds=(lower, upper),
    use_lmfit=True,
    lmfit_method='least_squares'
)

# lmfit 自动提供更准确的误差估计
print(result_lmfit.summary())
```

**lmfit 的优势：**
- 自动计算置信区间
- 支持 15+ 种优化算法
- 参数约束更灵活（可设置表达式约束）
- 更好的收敛性和诊断信息

### 3. 贝叶斯 MCMC 拟合（最稳健）

⚠️ **重要：MCMC 必须提供合理的初值！**

**推荐两步法：**

```python
# 第一步：用 lmfit 快速获得好的初值
result_init = fitter.fit('model', use_lmfit=True)

# 第二步：用 lmfit 结果作为 MCMC 初值
result_mcmc = fitter.fit(
    'model',
    p0=result_init.params,  # ✓ 使用优化结果
    bounds=bounds,           # ✓ 必须提供边界
    bayesian=True,
    mcmc_nwalkers=32,
    mcmc_nsteps=3000,
    mcmc_burn=1000,
    mcmc_thin=2,
    progress=True
)
```

**错误示范（会导致 χ² → ∞）：**

```python
# ✗ 错误：MCMC 缺少初值和边界
result = fitter.fit('model', bayesian=True, mcmc_nsteps=5000)
```

```python
result_bayes = fitter.fit(
    'smoothly_double_broken_powerlaw',
    p0=[1e-9, 0, -1.5, -2.5, 30, 400, 0.5, 0.5],
    bounds=(lower, upper),
    bayesian=True,
    mcmc_nwalkers=32,      # walker 数量（建议 2*ndim 以上）
    mcmc_nsteps=5000,      # 采样步数
    mcmc_burn=1000,        # burn-in 步数
    mcmc_thin=1,           # 采样间隔
    progress=True          # 显示进度条
)

print(result_bayes.summary())

# 访问 MCMC 样本
samples = result_bayes.mcmc_samples  # shape: (n_samples, n_params)
logprob = result_bayes.mcmc_logprob  # 对数概率

# 参数不确定度（非对称）
print(f"参数下误差: {result_bayes.errors_lower}")
print(f"参数上误差: {result_bayes.errors_upper}")
```

**贝叶斯方法的优势：**
- 完整的参数后验分布
- 非对称误差自然处理
- 参数间相关性分析
- 不依赖初值选择（在合理先验下）
- 适合参数高度相关或退化的情况

### 4. 可视化 MCMC 结果

```python
import corner
import matplotlib.pyplot as plt

if result_bayes.mcmc_samples is not None:
    fig = corner.corner(
        result_bayes.mcmc_samples,
        labels=result_bayes.param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    plt.suptitle('参数后验分布', y=1.02)
    plt.show()
```

## 默认参数范围详解 ✨

### 幂律指数（所有幂律模型）

**范围**: -10 到 3

| 指数范围 | 物理意义 | 典型应用 |
|---------|---------|---------|
| -10 到 -3 | 极陡衰减 | GRB后期喷流能量耗尽 |
| -3 到 -1.5 | 正常衰减 | 典型GRB余辉 |
| -1.5 到 -0.5 | 缓慢衰减 | 能量注入或浅衰减阶段 |
| -0.5 到 0.5 | 平台期 | 磁星供能平台 |
| 0.5 到 3 | 上升 | 峰值前上升段 |

### 各模型的默认初值和边界

#### 1. Powerlaw
```python
# 参数: norm, index, t0
# 初值: [v_max, -1.5 (下降) 或 0.5 (上升), 1.0]
# 边界:
#   norm: [0.01*v_min, 100*v_max]
#   index: [-10, 3]
#   t0: [1e-10, inf]
```

#### 2. Broken Powerlaw
```python
# 参数: norm, index1, index2, t_break
# 初值: [v_max, -0.5, -2.0, t_mid]
# 边界:
#   norm: [0.01*v_min, 100*v_max]
#   index1, index2: [-10, 3]
#   t_break: [t_min, t_max]
```

#### 3. Double Broken Powerlaw
```python
# 参数: norm, index1, index2, index3, t_break1, t_break2
# 初值: [v_max, 0.0, -2.0, -1.0, t_25%, t_75%]
# 边界:
#   norm: [0.01*v_min, 100*v_max]
#   index1, index2, index3: [-10, 3]
#   t_break1: [t_min, t_mid]
#   t_break2: [t_mid, t_max]  # 确保 t_break2 > t_break1
```

#### 4. Smoothly Broken Powerlaw
```python
# 参数: norm, index1, index2, t_break, smoothness
# 初值: [v_max, -0.5, -2.0, t_mid, 0.5]
# 边界:
#   norm: [0.01*v_min, 100*v_max]
#   index1, index2: [-10, 3]
#   t_break: [t_min, t_max]
#   smoothness: [0.01, 5.0]
```

#### 5. Smoothly Double Broken Powerlaw
```python
# 参数: norm, index1, index2, index3, t_break1, t_break2, smoothness1, smoothness2
# 初值: [v_max, 0.0, -2.0, -1.0, t_25%, t_75%, 0.5, 0.5]
# 边界:
#   norm: [0.01*v_min, 100*v_max]
#   index1, index2, index3: [-10, 3]
#   t_break1: [t_min, t_mid]
#   t_break2: [t_mid, t_max]
#   smoothness1, smoothness2: [0.01, 5.0]
```

**注**: v_max、v_min、t_min、t_max、t_mid 等统计量自动从数据中提取

## 模型选择建议

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 简单模型 (≤3参数) | scipy | 快速，足够准确 |
| 中等复杂度 (4-6参数) | lmfit | 平衡速度与准确性 |
| 复杂模型 (>6参数) | lmfit + Bayesian | lmfit快速探索，Bayesian确认 |
| 参数高度相关 | Bayesian | 处理参数退化 |
| 需要误差传播 | Bayesian | 完整后验分布 |
| 生产环境自动化 | lmfit | 稳定性好 |

## 参数设置指南

### lmfit_method 选项

- `'least_squares'` (推荐) - Levenberg-Marquardt 算法
- `'leastsq'` - 传统最小二乘
- `'nelder'` - Nelder-Mead 单纯形法（无需梯度）
- `'powell'` - Powell 方向加速法
- `'differential_evolution'` - 全局优化（慢但稳健）

### MCMC 参数设置

- **nwalkers**: 建议 ≥ 2 × 参数数量，典型值 32-64
- **nsteps**: 测试用 1000-2000，生产用 5000-10000
- **burn**: 约为 nsteps 的 20-30%
- **thin**: 通常为 1，如果自相关长度大可增加到 2-5

## 性能对比

测试环境：8参数平滑双折断幂律，100个数据点

| 方法 | 时间 | 准确度 | 适用场景 |
|------|------|--------|----------|
| scipy curve_fit | ~0.1s | 好 | 快速探索 |
| lmfit least_squares | ~0.3s | 优秀 | 日常使用 |
| lmfit differential_evolution | ~5s | 极好 | 困难拟合 |
| MCMC (2000步) | ~30s | 最佳 | 论文发表 |

## 故障排查

### 拟合失败

1. 检查初值 `p0` 是否合理
2. 确认边界 `bounds` 包含真实值
3. 尝试不同的 `lmfit_method`
4. 使用 `differential_evolution` 全局搜索

### MCMC 不收敛

1. 增加 `mcmc_nsteps` 和 `mcmc_burn`
2. 检查 walker 初始位置是否合理
3. 查看 `mcmc_logprob` 是否稳定
4. 先用 lmfit 找到好的起点

### ImportError

```bash
# 确保安装了所需包
pip install lmfit emcee corner
```

## 最佳实践

1. **多阶段策略**：
   - 阶段1: scipy 快速探索
   - 阶段2: lmfit 精确拟合
   - 阶段3: MCMC 最终确认（用于发表）

2. **参数约束**：
   - 始终设置物理上合理的边界
   - 对 GRB 数据，确保折断时间在数据范围内

3. **误差处理**：
   - 用 10% 相对误差作为最小值
   - 排除 flux ≤ 0 的点

4. **模型比较**：
   - 使用 AIC/BIC 选择模型复杂度
   - 检查残差分布

## 参考文献

- lmfit 文档: https://lmfit.github.io/lmfit-py/
- emcee 文档: https://emcee.readthedocs.io/
- GRB 拟合实践: Evans et al. (2009), MNRAS 397, 1177
