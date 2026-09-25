# 元素吸收截面、光深与占比

入口：`from jinwu.physics.absorption import absorption_budget`。
所有科学调用在 HEASoft 可用的 `hea` Python 中进行。主函数启动独立进程并初始化
HEASoft，不需要在 Notebook 中清除已有谱或改变丰度。没有安装 PyXspec 不影响
导入模块，但计算需要 HEADAS 或当前 Python 环境下的 `heasoft/headas-init.sh`。

## 指定能量点

```python
import astropy.units as u
from jinwu.physics.absorption import absorption_budget

r = absorption_budget(
    [0.3, 1.0, 6.4, 6.7, 6.97, 10.0] * u.keV,
    1e22 / u.cm**2,                 # 示例输入，不是某次观测的测量值
    backend="atomic",              # 或 "tbabs"
    abundance_table="wilm",
    metallicity=0.5,
    element_factors={"Fe": 2.0},   # 在 metallicity 之后相乘
)
print(r.totals)
print(r.elements[r.elements["element"] == "Fe"])
print(r.to_markdown())
r.to_json("opacity_points.json")
r.to_csv("opacity_points")         # 两张 CSV + 带单位/溯源的 JSON
```

也可以传入标量，例如 `1*u.keV`；输出为一行总量表和逐元素表。乱序和重复能量保留。
`energy_rest` 与 `nh` 必须带单位，NH 为非负标量。

绝对数丰度用 `number_abundances={"H": 1, "He": 0.1, "O": 5e-4}`；未列元素为零，
H 固定为 1。此模式不能和元素缩放或非默认金属度混用。输入相同的默认值
`metallicity=1.0` 与省略该参数等价。非零的未支持元素请求会报错。

## 曲线与图件

```python
import numpy as np
curve = absorption_budget(
    np.geomspace(0.03, 30, 400) * u.keV,
    1e22 / u.cm**2,
    backend="atomic",
    plot=True,
)
curve.savefig("opacity_budget")    # PNG (300 dpi) + PDF
# 限定显示的元素，未显示的支持元素合并为 Other；数据表不删元素。
fig, axes = curve.plot(elements=["H", "He", "O", "Si", "S", "Fe"])
curve.savefig("opacity_selected")
fig, ax = curve.plot(elements=["O", "Fe"], unweighted=True)
curve.savefig("opacity_unweighted")
```

四联图分别展示总/各元素加权截面、截面占比、总/各元素光深、光深占比。
所有横轴都是吸收体静止系能量。`redshift=` 仅向数据表增加观测系能量，
不会再次改变截面。绘图使用独立的 Agg Figure，不切换调用方已有的绘图后端。

## 字段和物理约定

- `sigma_element`：单元素系数，单位 cm²。atomic 为中性原子光电截面；
  tbabs 是孤立成分探针得到的模型有效系数，不能称为孤立原子截面。
- `sigma_weighted = number_abundance * sigma_element`：按数丰度加权、每 H 核的截面。
- `sigma_total`：直接计算完整吸收模型得到的总截面，每 H 核的 cm²。
- `tau = nh * sigma_weighted`，`tau_total = nh * sigma_total`，透射率为 `exp(-tau_total)`。
- `sigma_fraction_pct` 与 `tau_fraction_pct`：百分比。有效分解且 NH>0 时两者相同。
  NH=0 的光深为零、光深占比未定义；未定义值在表中为 NaN，在 JSON 中为 null。
- 数值容差是算法检查，不是物理模型不确定度或统计误差。

支持元素：H、He、C、N、O、Ne、Na、Mg、Al、Si、S、Cl、Ar、Ca、Cr、Fe、Co、Ni。
完整 H–Zn 表中其他元素保留 `unsupported_by_backend` 标记；其截面为未定义，
不能把缺失模型支持解释为原子没有光电吸收。内置丰度表中这些元素的原始数值仍记录。

## TBabs 的已验证限制（不可忽略）

本地 XSPEC 12.15.1 的原生 `TBabs` 混合物截面不能由这里的孤立元素差分严格重建。
例如初始测试在静止系 1 keV、wilm 下，差分求和相对直接总截面偏低约 0.29%；
这超过 1e-6 的闭合门槛。因此 `tbabs` 默认混合物结果可能为
`decomposition_not_closed`：总截面与总光深保留，元素加权截面、光深和占比留空，
图中明确说明未取得有效分解。没有强制归一化或把缺失贡献分摊给元素。
原始孤立成分系数保留供诊断，不能当作已验证的混合物贡献。

同一运行环境中 `zTBabs(z=0)` 与 `TBabs` 的总截面并不完全一致。历史 EP260119a R07
使用 zTBabs，其闭合结果不能替代 TBabs 的验证。本函数不会为了使闭合通过而
悄悄切换模型。纯 H、H+He 或其他可闭合场景仍正常给出分解。

`atomic` 使用 `vphabs` + `vern`，不包含尘埃、分子、Thomson/Compton 散射或电离气体。
`tbabs` 保留其原生分子和尘埃设置；H 项包括该模型的氢相关贡献。
用户若要解释某个拟合，必须核对实际使用的是哪个原生模型包装器。

## 数值验证与溯源

每次调用保存模型名、XSPEC 版本、原生日志、丰度原始值/有效值、丰度表与模型注册表
SHA256、代码 SHA256、参考柱密度、能量小区间、收敛与闭合诊断。
参考柱密度自适应，使提取用的光深落在 0.2–2；并以一半参考柱密度独立核对线性。
能量小区间逐步收窄，不跨边做插值。最终区间两侧出现明显跃变时标记吸收边括区，
不把稳定的跨边平均值当作可靠点值。

当前 XSPEC 的 vphabs 会缓存丰度表名称。实现更换每张自定义表前，先实际计算一次
内置 wilm 模型来刷新缓存；仅设置丰度名称不足以保证刷新。

真实模型回归测试：

```bash
conda run -n hea bash -c 'export HEADAS="$CONDA_PREFIX/heasoft"; source "$HEADAS/headas-init.sh"; export JINWU_TEST_XSPEC=1; pytest test/test_absorption_budget.py -q'
```

未设置 `JINWU_TEST_XSPEC=1` 时，只执行输入/丰度解析单元测试，真实 XSPEC 测试显式跳过。

参考：
- [XSPEC TBabs 系列](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelTbabs.html)
- [XSPEC vphabs](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelPhabs.html)
- [XSPEC 丰度表定义](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSabund.html)

### Accessible plots and Notebook display

Plots reuse `jinwu.core.plotstyle` colors and export helpers. Four colors,
fixed element dash patterns and sparse markers provide redundant encodings.
The legend is shared outside the panels. No backend is forced by the library.

```python
# In a Notebook, enable %matplotlib inline once.
result.plot(elements=['H', 'He', 'O', 'Fe'], fraction_scale='linear')
result.savefig('opacity_linear')
result.show()
result.plot(elements=['H', 'He', 'O', 'Fe'], fraction_scale='log')
result.savefig('opacity_log_percent')
result.show()

# Replot saved numerical results without running XSPEC:
from jinwu.physics.absorption import AbsorptionBudget
result = AbsorptionBudget.from_json('atomic_curve.json')
```

`fraction_scale` accepts `linear` or `log` (default). Logarithmic percentages
omit nonpositive/undefined values without flooring or renormalizing them.
The underlying tables and decomposition-validity masks are unchanged.
`show()` calls matplotlib's `show()` using the caller's selected backend;
PNG/PDF exports remain available for headless sessions.

### TBabs 系列与线性能量轴 / TBabs family and linear energy axes

`backend="ztbabs"` 使用原生 zTBabs；输入已是静止系能量，模型红移固定为零。
Native zTBabs is evaluated at zero model redshift because input energies are
already in the absorber rest frame. It retains its own grain convention;
TBabs and zTBabs are not silently interchanged.

`wilm` 为默认基准；元素因子修改相对数丰度，不切换吸收模型。
The default abundance baseline is wilm; element factors scale number abundances
within the selected model, without changing the absorption backend.

```python
# 静止系能量与显式柱密度 / Rest-frame energy and explicit column density.
result = absorption_budget(energy_rest, nh, backend="ztbabs",
    abundance_table="wilm", element_factors={"O": 0.5, "Fe": 2.0})
# 线性横轴、对数纵轴 / Linear energy axis and logarithmic vertical axes.
result.plot(energy_scale="linear", fraction_scale="log")
# 在 Notebook 中显示 / Display in the Notebook.
result.show()
```

默认横轴线性、百分比纵轴对数；截面与光深纵轴始终对数。
The defaults are energy_scale="linear" and fraction_scale="log"; cross-section
and optical-depth axes are logarithmic. The tau=1 line and sampled total-tau
crossing brackets are marked. 灰色区间是采样括区，不是误差或精确求根。
Gray bands are sampling brackets, not confidence intervals or exact roots.

TBabs 分解不闭合时保留原生总量，并屏蔽无效元素贡献。
If TBabs decomposition fails closure, native totals remain valid and element
contributions stay masked. The code above is the maintained documentation example.
