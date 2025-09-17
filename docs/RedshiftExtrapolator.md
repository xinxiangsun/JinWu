# RedshiftExtrapolator 使用与原理说明

本文档详细解释 `autohea.core.utils.RedshiftExtrapolator` 的物理推导、计算流程与使用方法，重点阐述“在给定信噪比阈值下可探测的最大红移”的计算原理。

- 源码位置：`autohea/src/autohea/core/utils.py`
- 相关函数：`compute()`、`find_redshift_for_snr()`、`snr_li_ma()`

---

## 1. 问题背景与目标

目标：在给定的XSPEC物理模型、仪器响应（ARF/RMF）、背景与曝光时间条件下，求在某个SNR阈值（如 7σ）时的最大可探测红移 z_max。

核心变量：
- 模型 M(E; params)：XSPEC光子数谱（photons cm⁻² keV⁻¹ s⁻¹）
- 仪器响应：ARF（有效面积）与 RMF（能量重分配）
- 背景计数与时长：`bkgnum`, `duration`
- SNR阈值：`snr_target`（默认7）

---

## 2. 物理论证与关键公式

记观测到的光子数谱为 N_obs(E_obs)。类中遵循以下“严格的XSPEC红移外推公式”：

1) 几何衰减（真实物理距离）：使用共动距离 r_c(z)

- 光子数密度按真实距离平方反比衰减：∝ 1/r_c²
- 因此归一化的几何因子：

  r_c 因子 = [r_c(z₀) / r_c(z)]²

2) K-correction（幂律谱示例）

- 若 N_rest(E) ∝ E^(-α)，则红移导致能量变换 E_rest = E_obs × (1+z)
- XSPEC的单位与红移下时间膨胀、能量间隔效应相互抵消，最终只需要能谱幂律的额外修正：

  K 因子 = ((1+z₀) / (1+z))^α

3) 归一化缩放总因子

- 设 `norm_original` 为 z₀ 时模型的归一化，z 时应使用：

  norm_new = norm_original × ((1+z₀)/(1+z))^α × [r_c²(z₀)/r_c²(z)]

这正是代码中 `find_redshift_for_snr()` 计算 `total_factor` 的来源：

- `geometric_factor = (r_c_z0 / r_c_grid) ** 2`
- `k_correction_factor = ((1 + z0) / (1 + z_grid)) ** alpha`
- `total_factor = k_correction_factor * geometric_factor`

注：光度距离 D_L = (1+z) r_c 是定义量，适合用于能量通量 F 与光度 L 的关系；但对光子数密度与XSPEC模型的处理，采用真实距离 r_c 更能直接体现几何衰减，因此本类选择 r_c。

---

## 3. 计算流程（逐步说明）

以 `compute(snr_target=7)` → `find_redshift_for_snr()` 的实际实现为主线：

### 3.1 模型初始化与参数注入

- `init_model()` 内部依次执行：
  - 初始化HEASoft环境，创建 `xspec.Model(self._model)` 实例；
  - `_set_par()` 将传入的 `par` 按模型参数顺序写入XSPEC，并冻结参数；
  - 自动识别红移参数（若存在 `...Redshift`）与归一化参数（通常为最后一组件的 `norm`）。

### 3.2 红移网格与缩放因子

- 初始红移区间 [zmin, zmax]（默认 [z0, z0+1]），生成均匀 `z_grid`；
- 依次计算：
  - `r_c_z0 = comoving_distance(z0)`；
  - `r_c_grid = comoving_distance(z_grid)`；
  - `geometric_factor = (r_c_z0 / r_c_grid) ** 2`；
  - `alpha = _get_spectral_index()` 自动识别幂指数；
  - `k_correction_factor = ((1 + z0) / (1 + z_grid)) ** alpha`；
  - `total_factor = k_correction_factor * geometric_factor`。

随后对每个 z：
- 如果模型含有红移参数：`self._par_z.values = z`；
- 始终按公式修正归一化：`self._par_norm.values = original_norm * total_factor[i]`。

### 3.3 谱→计数率（卷积）

- 使用 SOXS 完成谱与响应的卷积：
  1) `spec = soxs.Spectrum.from_pyxspec_model(self._m1)`
  2) 取能段 0.5–4.0 keV：`newspec = spec.new_spec_from_band(0.5, 4.0)`
  3) 指定响应与曝光：设置 `rmf`, `arf`, `bkg`, `exposure`, `backExposure`
  4) 用 ARF 卷积得到计数谱：`cspec = newspec * soxs.AuxiliaryResponseFile(arf)`
  5) 总计数率：`src_rate = cspec.rate.sum().value`
  6) 叠加背景贡献（按您的数据口径）：`total_rate = src_rate + (bkgnum/duration) * area_ratio`

- 最终计数：`total_counts = total_rate * duration`

### 3.4 SNR 计算（Li & Ma公式）

使用 `snr_li_ma(n_src, n_bkg, alpha_area_time)`：

- `n_src = total_counts`
- `n_bkg = bkgnum`
- `alpha_area_time = area_ratio`

返回：

SNR = sqrt{ 2 [ N_on ln((1+α)/α × N_on/(N_on+N_off)) + N_off ln((1+α) × N_off/(N_on+N_off)) ] }

该公式对高能天文的 on/off 计数统计更稳健。

### 3.5 自适应搜索 z_max

- 在 `z_grid` 上找到 SNR 首次低于 `snr_target` 的位置 [z1, z2]；
- 在 [z1, z2] 区间递归细分，线性插值求解 SNR(z)=目标阈值；
- 若整个区间 SNR 都高于阈值，逐步扩大上界（`max_expand` 次）。

---

## 4. 推导要点回顾（为何这样做）

1) 使用共动距离 r_c：
- 光子数密度是与真实几何距离相关的量，选 r_c 体现 1/r² 衰减；
- 时间膨胀与能量间隔效应在 XSPEC 光子数谱单位下相互抵消。

2) K-correction 的来源：
- 对幂律谱 N ∝ E^-α，能量红移带来额外的 (1+z)^-α 因子；
- 最终只需考虑该能谱倾斜产生的修正。

3) 统一修正归一化：
- 无论是否存在 ".Redshift" 参数，归一化都乘以总因子 `total_factor`，保证谱形在观测能段内的正确缩放。

---

## 5. 使用示例

```python
from autohea.core.utils import RedshiftExtrapolator

ex = RedshiftExtrapolator(
    z0=1.0,
    model="TBabs*zTBabs*powerlaw",
    par=[1e21, 1e21, 1.0, 2.0, 1e-3],  # 示例参数
    arfpath="response.arf",
    rmfpath="response.rmf",
    bkgpath="background.pha",
    srcnum=100,
    bkgnum=1200,
    duration=155,
    area_ratio=1/12,
)

z_max = ex.compute(snr_target=7, show_model_info=True)
print("z_max:", z_max)
```

---

## 6. 常见问题与注意事项

- XSPEC 与 HEASoft：需正确初始化 HEASoft 环境（类内部已调用 `HeasoftEnvManager`）。
- SOXS 版本差异：部分属性在不同版本可能只读，但不影响卷积计算。
- 模型参数顺序：`par` 必须与 XSPEC 模型参数顺序完全一致。
- 红移参数缺失：若模型不含显式红移参数，依旧按 `total_factor` 修正归一化。
- 能段选择：示例固定为 0.5–4.0 keV，如需更改可拓展接口。
- 背景口径：`total_rate` 中背景项与 `area_ratio` 的口径请与数据处理保持一致。

---

## 7. 验证方法

使用 `verify_redshift_extrapolation(z_test)` 可输出详细的几何因子、K 因子与总因子，帮助核对推导：

```python
ex.verify_redshift_extrapolation(z_test=1.5)
```

- 打印 r_c、K-correction、总因子与文本解释，便于交叉验证。

---

## 8. 后续改进建议

- 支持更多谱型的 K-correction 自动识别（如 cutoffpl, grbm 的形状依赖）。
- 将能段、背景计数处理与响应组合抽象为策略参数，便于复用。
- 提供误差传播与不确定度评估（如蒙特卡洛抽样）。

---

如需我们将此文档嵌入到API文档或在Notebook中生成可视化示例，请告知。
