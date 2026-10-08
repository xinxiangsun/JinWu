> 历史使用指南：本次仅迁移，API 与叙述尚未重新验证。
> Historical usage guide: relocated only; APIs and claims have not been revalidated.

# LightcurveData 重构完整指南

**版本**: 2.0
**日期**: 2025-12-15
**标准**: OGIP-93-003 (Lightcurve Extensions), HEASoft/Stingray 兼容

---

## 概述

本文档描述了 `jinwu.core.file.LightcurveData` 的重大重构，旨在与 Stingray、HEASoft 等工业标准库对齐，提供完整的 OGIP-93-003 兼容性，包括 GTI 支持、精确的时间箱坐标系统和统一的计数/速率数据管理。

### 设计目标

1. **OGIP 标准完全兼容**: 支持 RATE/COUNTS 表、GTI 扩展、FRACEXP、QUALITY 等所有可选列
2. **Stingray 级别功能**: 实现 `apply_gti()`, `split_by_gti()`, `apply_mask()` 等高级操作
3. **精确时间坐标**: 明确区分 `bin_lo` (左缘) 和 `bin_hi` (右缘)，符合 HEASoft 约定
4. **向后兼容**: 保留旧字段作为 `@property`，发出 DeprecationWarning
5. **Pythonic 设计**: 使用 dataclass、slots、类型注解，性能与可读性并重

---

## 核心变更

### 1. 新增字段 (20+)

#### 时间坐标系统
```python
bin_lo: np.ndarray          # bin 左缘时刻（原 FITS TIME 列）
bin_hi: np.ndarray          # bin 右缘时刻（计算: TIME + TIMEDEL）
dt: float | np.ndarray      # bin 宽度（秒），可变宽度支持
tstart: Optional[float]     # 观测起始时刻（= bin_lo[0]）
tseg: Optional[float]       # 观测总时长（秒）
```

**设计说明**:
- 遵循 HEASoft `lcurve` 工具约定：`TIME` 列为 bin 左缘
- 支持可变时间箱（`dt` 可为数组）
- `bin_hi = bin_lo + dt`，完全确定时间箱边界

#### 计数/速率数据（统一设计）
```python
counts: Optional[np.ndarray]      # 原始计数（优先存储）
rate: Optional[np.ndarray]        # 计数速率 (counts/sec)
counts_err: Optional[np.ndarray]  # 计数不确定度（分离存储）
rate_err: Optional[np.ndarray]    # 速率不确定度（分离存储）
err_dist: Optional[Literal['poisson', 'gauss']]  # 误差分布类型
```

**设计说明**:
- 同时存储 `counts` 和 `rate`（若可推导，则自动计算缺失项）
- 分离 `counts_err` 和 `rate_err`，避免混淆
- `err_dist` 标记误差统计假设，便于后续分析

#### GTI 与质量控制
```python
gti_start: Optional[np.ndarray]   # GTI 起始时间数组
gti_stop: Optional[np.ndarray]    # GTI 结束时间数组
quality: Optional[np.ndarray]     # 每 bin 质量标志（FITS QUALITY 列）
fracexp: Optional[np.ndarray]     # 分数曝光 (0~1)（FITS FRACEXP 列）
backscal: Optional[float | np.ndarray]  # 背景刻度因子
areascal: Optional[float | np.ndarray]  # 面积刻度因子
```

**设计说明**:
- GTI 存储为两个独立数组，与 Stingray 兼容
- `gti` 属性自动生成 `[(start0, stop0), ...]` 元组列表
- 完整支持 OGIP-93-003 所有可选列

#### 时间系统（从 meta 提升）
```python
mjdref: Optional[float]      # MJD 参考日期（MJDREFI + MJDREFF）
timesys: Optional[str]       # 时间系统标记 ('TT', 'UTC', 等)
```

**设计说明**:
- 从 `meta` 字典提升为顶层字段，便于访问
- `mjdref` 自动合成 MJDREFI 和 MJDREFF
- 用于绝对时间转换和多数据源合并

#### 其他字段
```python
exposure: Optional[float]         # 总曝光时间（秒）
bin_exposure: Optional[np.ndarray]  # 单 bin 有效曝光 (FRACEXP × EXPOSURE)
region: Optional[RegionArea]      # 区域信息（如 WXT 的 REG00101）
columns: Tuple[str, ...]          # 原始 FITS 表列名
```

---

### 2. 新增方法（10+）

#### GTI 操作

**`apply_gti(inplace=False) -> LightcurveData`**
```python
# 功能：按 GTI 过滤 bin，移除 GTI 外的数据点
# 参数：inplace - 是否原地修改
# 返回：过滤后的光变曲线

lc_filtered = lc.apply_gti(inplace=False)
# GTI: [[0.0, 3.0], [5.0, 8.0]]
# 原始 bins: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-8
# 过滤后: 0-1, 1-2, 2-3, 5-6, 6-7, 7-8（移除了 3-4, 4-5）
```

**`split_by_gti(min_points=1) -> list[LightcurveData]`**
```python
# 功能：将光变曲线按 GTI 分割为独立的子段
# 参数：min_points - 每段最少 bin 数，小于此数的段被丢弃
# 返回：子光变曲线列表

segments = lc.split_by_gti(min_points=3)
# GTI: [[0.0, 3.0], [5.0, 8.0]]
# 返回 2 个独立的 LightcurveData 对象
# segment[0]: bins 0-1, 1-2, 2-3
# segment[1]: bins 5-6, 6-7, 7-8
```

#### 掩码操作

**`apply_mask(mask, inplace=False, filtered_attrs=None) -> LightcurveData`**
```python
# 功能：按布尔掩码过滤 bin
# 参数：
#   - mask: 布尔数组，长度等于 bin 数
#   - inplace: 是否原地修改
#   - filtered_attrs: 要应用掩码的数组属性列表（默认全部）
# 返回：过滤后的光变曲线

mask = lc.bin_lo >= 2.0  # 仅保留 t >= 2.0 的 bin
lc_masked = lc.apply_mask(mask, inplace=False)
```

**设计说明**:
- 与 Stingray 的 `apply_mask()` 签名兼容
- 自动处理所有数组字段（counts, rate, errors, gti等）
- inplace=True 时返回 self，便于链式调用

#### 数据操作

**`join(other) -> LightcurveData`**
```python
# 功能：合并两条光变曲线
# 参数：other - 另一个 LightcurveData 对象
# 返回：合并后的新光变曲线
# 注意：
#   - 若 mjdref 不同，会警告并转换 other 到 self 的 mjdref
#   - 若时间范围重叠，会警告但仍合并（重叠区域计数相加）
#   - GTI 自动合并（调用 stingray.gti.join_gtis）

lc_joined = lc1.join(lc2)
```

**`truncate(tmin=None, tmax=None) -> LightcurveData`**
```python
# 功能：裁剪到指定时间范围
# 参数：
#   - tmin: 最小时刻（None = 不限）
#   - tmax: 最大时刻（None = 不限）
# 返回：裁剪后的光变曲线
# 注意：GTI 也会相应裁剪

lc_truncated = lc.truncate(tmin=100.0, tmax=200.0)
```

**`sort(inplace=False) -> LightcurveData`**
```python
# 功能：按 bin_lo 排序光变曲线
# 参数：inplace - 是否原地修改
# 返回：排序后的光变曲线

lc_sorted = lc.sort(inplace=False)
```

#### 增强的原有方法

**`slice(tmin=None, tmax=None) -> LightcurveData`**
- 现在支持 GTI 感知（过滤后重新计算 GTI）

**`rebin(factor=None, target_dt=None) -> LightcurveData`**
- 现在支持 GTI 感知（每个 GTI 段独立 rebin）

---

### 3. 向后兼容性

#### 废弃字段（v2.0 后移除）

```python
@property
def time(self) -> np.ndarray:
    """废弃：使用 bin_lo 或 (bin_lo + bin_hi) / 2 代替"""
    warnings.warn("LightcurveData.time is deprecated; use bin_lo instead.",
                  DeprecationWarning, stacklevel=2)
    return self.bin_lo

@property
def value(self) -> Optional[np.ndarray]:
    """废弃：使用 counts 或 rate 代替"""
    warnings.warn("LightcurveData.value is deprecated; use counts/rate instead.",
                  DeprecationWarning, stacklevel=2)
    return self.rate if self.rate is not None else self.counts

@property
def error(self) -> Optional[np.ndarray]:
    """废弃：使用 counts_err 或 rate_err 代替"""
    warnings.warn("LightcurveData.error is deprecated; use counts_err/rate_err instead.",
                  DeprecationWarning, stacklevel=2)
    return self.rate_err if self.rate_err is not None else self.counts_err

@property
def is_rate(self) -> bool:
    """废弃：通过 rate is not None 判断"""
    warnings.warn("LightcurveData.is_rate is deprecated; check 'rate is not None' instead.",
                  DeprecationWarning, stacklevel=2)
    return self.rate is not None
```

**迁移指南**:
```python
# 旧代码
if lc.is_rate:
    y = lc.value
    yerr = lc.error
    t = lc.time

# 新代码
if lc.rate is not None:
    y = lc.rate
    yerr = lc.rate_err
    t = lc.bin_lo  # 或 (lc.bin_lo + lc.bin_hi) / 2
```

---

### 4. OgipLightcurveReader 增强

#### 14步完整读取算法

```python
def read(self) -> LightcurveData:
    """
    步骤 1: 定位 TIME 列所在 HDU
    步骤 2: 读取 TIME 数组，推导 dt（从 TIMEDEL 或 median(diff)）
    步骤 3: 计算 bin_lo/bin_hi（TIME 为左缘）
    步骤 4: 读取 RATE/COUNTS 列
       - 优先级：RATE > COUNTS
       - 若只有一个，根据 dt 推导另一个
    步骤 5: 分离 rate_err/counts_err 从 ERROR 列
    步骤 6: 读取可选列
       - FRACEXP（分数曝光）
       - QUALITY（质量标志）
       - BACKSCAL/AREASCAL（刻度因子）
    步骤 7: 提取 GTI 扩展 (_extract_gti() 辅助方法)
       - 搜索名为 'GTI' 的扩展
       - 读取 START/STOP 列
       - 返回 gti_start/gti_stop 数组
    步骤 8: 计算 tstart, tseg
    步骤 9: 构建 OgipMeta 元数据
    步骤 10: 计算 bin_exposure（FRACEXP × EXPOSURE）
    步骤 11: 推断 err_dist
       - counts 优先: 'poisson'
       - rate 优先: 'gauss'（假设泊松计数已转换）
    步骤 12: 解析区域信息（REG0**** 列）
    步骤 13: 构建 LightcurveData 对象
    步骤 14: 填充废弃字段（向后兼容）
       - value = rate or counts
       - error = rate_err or counts_err
       - is_rate = (rate is not None)
    """
```

#### GTI 提取辅助方法

```python
def _extract_gti(self, hdul: fits.HDUList) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 FITS 文件中提取 GTI 扩展

    返回：(gti_start, gti_stop) 元组，若无 GTI 则返回 (None, None)

    搜索策略：
    1. 查找 EXTNAME='GTI' 的扩展（大小写不敏感）
    2. 读取 START/STOP 列（或 TSTART/TSTOP）
    3. 返回 numpy 数组
    """
```

---

## 使用示例

### 基础用法

```python
from jinwu.core.file import read_lc

# 读取 OGIP 光变曲线 FITS 文件
lc = read_lc('swift_lc.fits')

# 访问新字段
print(f"Bins: {len(lc.bin_lo)}")
print(f"Time range: [{lc.bin_lo[0]:.2f}, {lc.bin_hi[-1]:.2f}] s")
print(f"MJD reference: {lc.mjdref}")
print(f"GTI segments: {len(lc.gti) if lc.gti else 0}")

# 数据内容
if lc.rate is not None:
    print(f"Mean rate: {np.mean(lc.rate):.3f} counts/s")
if lc.counts is not None:
    print(f"Total counts: {np.sum(lc.counts):.0f}")
```

### GTI 操作

```python
# 过滤 GTI 外的数据
lc_clean = lc.apply_gti(inplace=False)
print(f"Original: {len(lc.bin_lo)} bins")
print(f"After GTI: {len(lc_clean.bin_lo)} bins")

# 按 GTI 分割
segments = lc.split_by_gti(min_points=5)
for i, seg in enumerate(segments):
    print(f"Segment {i}: {seg.tstart:.2f} - {seg.tstart + seg.tseg:.2f} s, {len(seg.bin_lo)} bins")
```

### 时间范围操作

```python
# 裁剪到 T90 窗口
lc_t90 = lc.truncate(tmin=t_start, tmax=t_start + 90.0)

# 时间排序（若数据乱序）
lc_sorted = lc.sort(inplace=False)
```

### 合并多个观测

```python
from jinwu.core.file import read_lc

lc1 = read_lc('obs1_lc.fits')
lc2 = read_lc('obs2_lc.fits')

# 合并（自动处理 mjdref 差异）
lc_combined = lc1.join(lc2)

# 若 mjdref 不同，会看到警告：
# UserWarning: MJDref mismatch: self=55000.0, other=55001.0. Converting...
```

### 重新分箱

```python
# 固定因子重 bin（每 4 个 bin 合并为 1 个）
lc_rebinned = lc.rebin(factor=4)

# 目标时间分辨率 rebin（每个 bin 10 秒）
lc_rebinned = lc.rebin(target_dt=10.0)

# 注意：rebin 现在支持 GTI 感知，每个 GTI 段独立处理
```

### 掩码过滤

```python
# 过滤低质量数据
if lc.quality is not None:
    good_mask = lc.quality == 0  # 质量标志为 0 表示良好
    lc_good = lc.apply_mask(good_mask, inplace=False)

# 能量/光度过滤
high_rate_mask = lc.rate > 10.0  # 仅保留高计数率时段
lc_burst = lc.apply_mask(high_rate_mask, inplace=False)
```

---

## 技术细节

### 时间箱坐标系统

#### HEASoft 约定
```
FITS TIME 列: bin 左缘 (t_left)
TIMEDEL 关键字或列: bin 宽度 (Δt)
bin 右缘: t_right = t_left + Δt
bin 中心: t_center = t_left + Δt/2
```

#### 示例
```python
# FITS 文件内容
TIME     = [0.0, 1.0, 2.0]      # 原始 TIME 列
TIMEDEL  = 1.0                  # 头关键字

# LightcurveData 解析结果
bin_lo   = [0.0, 1.0, 2.0]      # 直接使用 TIME
bin_hi   = [1.0, 2.0, 3.0]      # 计算: TIME + TIMEDEL
dt       = 1.0                  # 标量或数组

# 兼容属性（废弃）
time     = [0.0, 1.0, 2.0]      # 返回 bin_lo（发出警告）
```

### GTI 处理

#### 数据结构
```python
# 存储形式
gti_start = np.array([0.0, 10.0, 20.0])     # GTI 起始数组
gti_stop  = np.array([5.0, 15.0, 25.0])     # GTI 结束数组

# 便利属性
gti = [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)]  # 元组列表（自动生成）
```

#### apply_gti 算法
```python
def apply_gti(self, inplace=False):
    # 1. 调用 stingray.gti.create_gti_mask 生成掩码
    mask = create_gti_mask(self.bin_lo, np.array(self.gti), dt=self.dt)

    # 2. 应用掩码到所有数组字段
    return self.apply_mask(mask, inplace=inplace)
```

#### split_by_gti 算法
```python
def split_by_gti(self, min_points=1):
    # 1. 对每个 GTI 段 (start, stop)
    for gti_start, gti_stop in self.gti:
        # 2. 生成该 GTI 的掩码
        mask = (self.bin_lo >= gti_start) & (self.bin_hi <= gti_stop)

        # 3. 应用掩码创建子光变曲线
        lc_segment = self.apply_mask(mask, inplace=False)

        # 4. 若 bin 数 >= min_points，加入结果列表
        if len(lc_segment.bin_lo) >= min_points:
            result.append(lc_segment)

    return result
```

### 计数/速率转换

#### 自动推导逻辑
```python
# 若 FITS 仅有 COUNTS 列
counts = FITS['COUNTS']
rate = counts / dt          # 自动计算速率

# 若 FITS 仅有 RATE 列
rate = FITS['RATE']
counts = rate * dt          # 自动计算计数

# 若两者都有
counts = FITS['COUNTS']     # 直接使用
rate = FITS['RATE']         # 直接使用
```

#### 误差传播
```python
# 泊松统计（counts 优先）
counts_err = np.sqrt(counts)
rate_err = counts_err / dt

# 高斯统计（rate 优先，来自 FITS ERROR 列）
rate_err = FITS['ERROR']
counts_err = rate_err * dt
```

---

## 验证与测试

### 与 Stingray 对比

| 功能 | Stingray Lightcurve | jinwu LightcurveData | 状态 |
|------|---------------------|---------------------|------|
| bin_lo / bin_hi | ✅ | ✅ | 兼容 |
| counts / rate 分离 | ✅ | ✅ | 兼容 |
| GTI 数组 | ✅ (gti 属性) | ✅ (gti_start/stop + gti) | 兼容 |
| apply_gti() | ✅ | ✅ | 兼容 |
| split_by_gti() | ✅ | ✅ | 兼容 |
| apply_mask() | ✅ | ✅ | 兼容 |
| join() | ✅ | ✅ | 兼容 |
| GTI 感知 rebin | ✅ | ✅ | 兼容 |
| FRACEXP 支持 | ✅ | ✅ | 兼容 |
| QUALITY 支持 | ✅ | ✅ | 兼容 |

### 与 HEASoft 对比

| 约定 | HEASoft lcurve | jinwu 实现 | 状态 |
|------|---------------|-----------|------|
| TIME = bin 左缘 | ✅ | ✅ | 兼容 |
| TIMEDEL | ✅ | ✅ | 兼容 |
| MJDREFI + MJDREFF | ✅ | ✅ (自动合成 mjdref) | 兼容 |
| GTI 扩展 | ✅ | ✅ | 兼容 |
| RATE/COUNTS 表 | ✅ | ✅ (自动识别) | 兼容 |

### 已知限制

1. **join() 重叠处理**: 目前简单拼接数组，重叠区域不做特殊处理（Stingray 同样如此）
2. **rebin GTI 边界**: GTI 段边界的 bin 可能被截断或保留，行为与 Stingray 一致
3. **QUALITY 语义**: OGIP 标准未严格定义 QUALITY 值，当前假设 0=好，非0=差
4. **时间系统转换**: 不同 `timesys` 之间的转换需外部工具（如 astropy.time）

---

## FAQ

### Q: 为什么要区分 bin_lo 和 bin_hi？
**A**: 精确的时间箱坐标对于：
- 正确的 GTI 过滤（需判断 bin 是否完全在 GTI 内）
- 准确的 rebin 操作（避免边界模糊）
- 与其他工具（Stingray, XSPEC）的互操作

### Q: counts 和 rate 可以都为 None 吗？
**A**: 理论上不应该，但代码允许。读取器会尝试从 FITS 推导至少一个。如果都为 None，后续分析会失败。

### Q: GTI 是必需的吗？
**A**: 不是。若 FITS 文件无 GTI 扩展，`gti_start`/`gti_stop` 为 None，GTI 相关操作会优雅地处理（无操作或返回原数据）。

### Q: 如何处理旧代码中的 lc.time？
**A**:
1. 短期：忽略 DeprecationWarning，代码仍可运行
2. 中期：替换为 `lc.bin_lo`（若需 bin 中心，使用 `(lc.bin_lo + lc.bin_hi) / 2`）
3. 长期：v2.0 后 `time` 属性将被移除

### Q: rebin 后 GTI 会变化吗？
**A**: 不会。`rebin()` 保持原 GTI 不变，每个 GTI 段独立 rebin。若需调整 GTI，应在 rebin 后调用 `apply_gti()`。

---

## 参考文献

1. **OGIP-93-003**: "The OGIP Standard for Lightcurve Extensions" (1993)
2. **OGIP-94-003**: "The OGIP Standard for Event List Extensions" (1994)
3. **Stingray Documentation**: https://docs.stingray.science/
4. **HEASoft lcurve**: https://heasarc.gsfc.nasa.gov/ftools/caldb/help/lcurve.html
5. **jinwu 设计文档**: LIGHTCURVEDATA_REFACTOR_STATUS.md

---

## 更新日志

### v2.0 (2025-12-15)
- ✅ 重构完成，20+ 新字段
- ✅ 实现 10+ 新方法（GTI/mask/data操作）
- ✅ OgipLightcurveReader 完全重写（14步算法）
- ✅ 向后兼容机制（@property + DeprecationWarning）
- ✅ 类型注解修复（np.ndarray, Optional 检查）
- ✅ 与 Stingray/HEASoft 完全对齐

### v1.x (历史)
- 基础 LightcurveData 实现
- 简单的 OGIP 读取器
- 无 GTI 支持
- 模糊的 time/value/is_rate 字段

---

## 许可证

本文档遵循 jinwu 项目许可证 (见 LICENSE 文件)

---

**文档维护**: jinwu 开发团队
**最后更新**: 2025-12-15
