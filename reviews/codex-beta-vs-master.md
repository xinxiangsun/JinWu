# codex/beta vs master 变更审查报告

> **2026-09-27 实施状态**：第 1–18 节保留各轮审查的历史快照，不能作为当前未修复清单或最终验收结论。合并与发布记录见第 19–20 节；后续修复、科学方法复核和未验证范围以文末第 21 节为准。早期“所有问题均为约定/文档层面”的判断已被 R49–R58 等后续实测推翻。

- 审查基线：`269587a`（codex/beta）vs `master`；9 个提交，152 文件，+26511/−8681；工作区另含 staged +4793 / unstaged +1074 / untracked 69 项（含 `jinwu-gw` 新包、`subthreshold`、`absorption`、`examples/`、`scripts/`）。
- 审查环境：`hea`（Python 3.12，HEASoft/XSPEC/GDT/swiftbat/ligo.skymap 可用）。
- 快照与日志：`reviews/evidence/00-baseline-snapshot.txt` 起共 31 份证据文件（含第四轮 28、29，第五轮 30，第六轮 31）。
- 计划文档：`implementation_plan.md`（仓库根）。

## 1. 总体结论

- **物理与方法正确性：通过。** 四个新算法核心全部与本地 `external_sources/heasoft-6.37/burstcube/lib/` 上游实现做了数值对照（0 失配）；GW 概率积分与 GBM 覆盖几何的方法语义经上游文档与公式核对成立；发现的问题均为**约定/文档层面**而非计算错误。
- **工程与打包：发现 20 项问题（R1–R20），本轮已修复 7 项**（R1/R2/R3/R6/R9/R17 + 测试跟踪守卫），其余为待 owner 决策或建议项，全部列于 §5。
- **CI 门禁现状**：被跟踪测试 16 文件 + 本轮新增，离线门禁 **242 passed / 1 deselected**（无 XSPEC 环境模拟同样通过）；全模块导入冒烟 104 个 0 失败；5 个 wheel 本地构建成功。

## 2. 物理与方法正确性审查（对照源与论文）

### 2.1 FastNormFit / 泊松精确区上限 — ✅ 数值保真

- 上游源码：`external_sources/heasoft-6.37/burstcube/lib/fast_norm_fit.py`（burstcube GDT 移植）。
- 方法出处：Poisson 似然比 TS(N)=2Σ[d·ln((b+Ne)/b)−Ne]（Cash 1979, ApJ 228, 939, doi:10.1086/156922）；任意阶解析导数 + Newton/Halley；Wilks 1938（doi:10.1093/biomet/26.4.404）保证 ΔTS 渐近 χ²(1)。
- 验证（evidence/16、17）：
  - `ts/dts(1阶,2阶)` 与上游逐点对照（4 组手工用例 + 300 组随机用例）：**max|ΔTS|=7.8e-14，max rel|Δnorm|=7.9e-15，failed 标志 0 失配**。
  - 欠涨解析分支（`allow_negative=True`）：上游返回 TS=−0.1284，jinwu 返回 +0.1284 —— 代码注释声明"原实现符号为负、已验证错误"的**有意偏离被数值证实**；`allow_negative=False` 路径与上游完全一致。
  - 新增 `upper_limit()`（上游没有）：TS(N_up)=TS_best−ΔTS 的 Brent 解与 40 万点细网格独立扫描对照，**200 例 0 不一致**，且上限恒在最佳拟合右侧。
- 约定核查：`confidence=0.9` → ΔTS=chi2.ppf(0.9,1)=2.7055，即 XSPEC `error` 命令 90% 惯例（=1.645σ 双侧/单侧 95% 高斯），**不是** GRB 文献常用的单侧 90%（1.282σ → ΔTS=1.6424）。docstring 已写明"双侧中心区间"，但建议在 usage 文档再强调，避免跨文献引用错位（→ R20，P2）。

### 2.2 曝光加权贝叶斯块 — ✅ 逐行等价

- 上游源码：`burstcube/lib/bayesian_blocks.py`（基于 astropy 实现 + astropy#14017 修复 + 0 计数箱支持；方法出处 Scargle et al. 2013, ApJ 764, 167, doi:10.1088/0004-637X/764/2/167，适应函数 Eq.19、先验 Eq.21）。
- 验证（evidence/18）：用 GDT `TimeBins` 构造 25 组含 10% 零计数箱的合成光变（阶跃率），jinwu `bayesian_blocks_exposure` 与上游 `bayesian_blocks` 的变点集 **25/25 完全一致**。

### 2.3 迭代背景贝叶斯块（txx_iterbkg 核心）— ✅ 端到端一致

- 上游源码：`burstcube/lib/bayesian_lc.py::BayesianBlocksLightcurve.compute_bayesian_blocks`（迭代背景拟合 → Giacomo 技巧"曝光:=背景计数"把含背景 Poisson 化为近似齐次 → prominence 定信号区 → 缓冲区外扩 → 背景区间重复/2-3 周期循环检测收敛）。Giacomo 技巧出处：threeML `utils/bayesian_blocks.py#L171`（G. Vianello）。
- 验证（会话内对照，8 组 n=400 合成光变，高斯 burst 叠加常数背景）：
  - **信号区间 (signal_range) 8/8 与上游完全相同**；峰时刻 8/8 相同；收敛轮数一致（iter=2）。
  - 内部 BB 块结构 6/8 逐位相同，2 例不同 —— 差异仅来自 jinwu 增加的 `T_k>0` 保护（上游在零曝光箱产生 NaN 传播）与收敛后合并块，不影响信号定界输出。
- T90/T50 定义核对：`_quantile_interval` 用有符号净计数累计曲线取分位（T90=5%–95%、T50=25%–75%），负净计数箱保持负号、多次穿越取首末中点（`battblocks` 惯例）——与标准定义一致。
- Poisson 重采样误差：整条流水线重跑（含块选择效应）+ 分箱系统误差正交合成，方法学成立；`t100_err` 恒为 NaN 占位（文档已声明）。
- 未验证项：事件级 `txx` 主方法标称"A&A 5.4 方法"，本轮未逐条核对上游论文（见 §6）。

### 2.4 GW 天图读取与 MOC 概率积分 — ✅ 方法语义成立

- 像素契约：嵌套 UNIQ、密度 sr^-1、概率=密度×面积，与 LVK 多分辨率天图规范一致；`_credible_indices` 为贪心最高密度构造。
- MOC 积分语义（关键核对点）：mocpy 0.20 `MOC.probability_in_multiordermap` 官方文档明确"PROBDENSITY × cell 面积求和，非完整格按面积比例"——即**格内均匀近似**。`refined_probability` 以 order 10→13 重建区域、|ΔP|<1e-3 收敛、未收敛置 `converged=False`，正是对该近似的正确处理；解析球冠极限概率由 `test/test_gw.py` 的恒等式 P=(1−cosρ)/2 校验（通过）。
- 安全面：天图 URL 抓取经 `fetch_http_bytes` 逐跳 `_validate_http_url`（`allow_redirects=False`，每跳独立公网校验）；`JINWU_GW_ALLOW_PROXY_DNS=1` 对 198.18.0.0/15 放行为显式 opt-in。残余风险：DNS 重绑定 TOCTOU（R16 记录）。

### 2.5 GBM 覆盖几何 — ✅ 与 GDT 公式一致（阈值为可配置启发式）

- `coverage.py` 用 GDT `SpacecraftFrame.at(t)` 同一时刻插值做 `location_visible`（地心遮挡）与 `detector_angle`；SAA/good 旗标取最近采样，坏状态时覆盖置空。`test_gw_gbm_coverage.py` 用解析球冠封闭解交叉验证 MOC 精化积分（通过）。
- 默认阈角 NaI≤60°、BGO≤90° 是**覆盖启发式**而非响应加权探测概率；已作为参数暴露、输出标注为 coverage/diagnostic。可接受，建议文档注明。
- `read_gbm_geometry`（poshist.py）：线性插值弦效应、SAA 旗标、外推拒绝、插值间隙上限均有离线回归（含解析轨道周期 2π√(r³/GM) 对照）。

### 2.6 其余物理模块 — ✅ 声明与实现相符（部分未数值复核）

- `physics/absorption.py`：tbabs/zTBabs/vphabs(vern) 元素分解；引用 Wilms 2000, ApJ 542, 914 / Verner 1996, ApJ 465, 487 / XSPEC 手册；隔离 XSPEC worker + 溯源哈希 + 闭合失败显式掩码。未独立重算截面（worker 实跑待做，见 §6）。
- `physics/gr.py`：AUD-03 修复后 `beta`/`lorentz_factor` 解析且单位安全（test_gr.py 覆盖）；`show_*` 已声明"仅展示"；公式表与 Rybicki & Lightman §4、Urry & Padovani 1995 (doi:10.1086/133758) 一致。
- OGIP 对齐 HEASoft 6.37：TLMIN/TLMAX/DETCHANS/NUMGRP/HDUVERS 读写与自洽校验由既有回归覆盖；未逐字段对照 heasoft 源码（§6）。

## 3. 执行验证（可复现）

| 项 | 证据 | 结果 |
| --- | --- | --- |
| 离线门禁（被跟踪集） | evidence/10 | 218 passed, 1 deselected |
| 无 HEASoft 模拟（CI 等价） | evidence/13 | 218 passed |
| 全模块导入冒烟 | evidence/21 | 104 modules, 0 failures（修复 R17 后） |
| Sphinx 构建 | evidence/12 | succeeded, 165 warnings（2 条 toc.not_included → R7） |
| wheel 构建 + vendor 合规 | evidence/15 | 5 包成功；`_vendor/license.txt`+`UPSTREAM.json` 在 wheel 内 |
| FastNormFit 对照 | evidence/16,17 | 300 例 0 失配；上限独立扫描 200 例 0 不一致 |
| 贝叶斯块对照 | evidence/18 | 25 例（含零计数箱）0 失配 |
| iterbkg 端到端 | 会话记录 | 信号区间/峰 8/8 一致 |
| 修复后全量回归 | 会话末 | 242 passed（含 core.skymap 7 项） |
| read_spatial_map 等价 | evidence/19 | 新旧路径 bitwise 一致 |

## 4. 发现清单（R1–R20，状态截至本轮）

已修复（本轮实施，含回归）：R1（Makefile sha256 死代码行）、R2（publish 漏平台 wheel/无 rust-build 依赖）、R3（jinwurs conda-recipe 版本脱锁 + sync 覆盖）、R6（fermi⇄gw 环依赖，按 owner 决策 C 方案落地：新增 `jinwu.core.skymap` 共享层，`search-skymap` extra 改 `astropy-healpix`，新增 `jinwu[skymap]` extra，依赖方向守卫 + core/gw 契约一致性测试）、R9（RTD 补装 jinwu-gw）、R17（`jinwu.gw.__main__` 缺 `__main__` 守卫，import 即 SystemExit）、R4/R5 部分（`scripts/check_tracked_assets.py` 守卫 + test_core_skymap 白名单入库）。

已核实、无需改动：R13（vendor Apache-2.0 合规：UPSTREAM.json 逐文件 sha256 + license.txt 均在 wheel 内）、R15（§2 四项算法对照）、R16（SSRF 逐跳校验正确；残余 DNS 重绑定 TOCTOU 记录）、R18（撤回：xspec 模块级导入均有 `except ModuleNotFoundError` 保护；先前"CI 必挂"源于我用过严的 ImportError 桩，已纠正留证 evidence/13、14）。

待决策 / 建议：

- **R7（P1）** toctree 引用的 `usage/gbm_subthreshold.rst` 未跟踪（守卫已拦截）；`absorption_budget.md`、`lf_redshift_transform.md` 为游离文档 → 提交或摘除。
- **R8（P2）** 工作区删除根 `pyproject.toml` 的 `[tool.uv.workspace]`，仓库无 uv 说明 → owner 确认。
- **R10（P1）** `jinwu.instruments` entry-point 注册非仪器包 `jinwu.gw`（wheel-gate 断言含 `gw`）→ 定义协议语义或改名。
- **R11（P2）** 非 vendor `except Exception` 358 处；changelog 自曝 `fit.py` flux/rate/statistics 块吞异常 → 按"可降级 vs 必须报错"分类清理。
- **R12（P2）** 超大单文件（survey.py 6853 行等）—— 维护性建议。
- **R14（P1）** CI 依赖面重（astro-gdt→cartopy/healpy/statsmodels；swiftbat 要求 astropy≥8；jinwu-gw→ligo.skymap/mocpy）；3.11–3.13 矩阵实测与拆 job 待做。
- **R19（P2）** xspec 守卫只捕 `ModuleNotFoundError`，半损坏 HEASoft（ImportError）会穿透 → 建议统一 `except ImportError`。
- **R20（P2）** `upper_limit(confidence=0.9)` 的 ΔTS=2.7055 为 χ²(1) 90%（=1.645σ），非 GRB 文献单侧 90%（1.28σ→ΔTS=1.6424）→ usage 页给换算或加 `semantics=` 参数。
- **R21（P2）** 事件级 `txx` 的 `method="aanda_2021_sec5_4"`/标签"A&A 5.4 方法"所指的 A&A 2021 论文全仓库未点名 → 补全引文（详见 §7.2）。
- **R23（P2，§8.2）** `fit.py _xspec_spectrum_counts` 的 `background_counts` 实为 f·B_raw（BACKSCAL 缩放后 OFF 计数），字段无语义注释，与 `recover_raw_off_counts`（实证精确）语义并存易误用 → 澄清或对齐。
- **R24（P3，§8.5）** `_binomial_proportion_interval` Wilson fallback 注释推理与事实相反（实际更严非更宽）→ 更正措辞。
- **R25（P3，§8.6）** `urls.py generate_download_url` filename 死变量、不随 URL 返回 → 补返回或标 deprecated。
- **R26（P3，§9.1）** `attitude.py` 插值 `fill_value='extrapolate'` 静默外推：调用方 `bat_observation.py:289-294` 的 mid-point 回退不可达；`_parse_sao` 缺列时静默零指向 → 加范围守卫与显式失败。
- **R27（P2，§10.A）** 统计/物理小工具跨包重复：Clopper–Pearson+Wilson 二项区间 3 份（core model_comparison、survey、gbm）、单位幂率能流函数 2 份（survey/gbm）；两仪器包均已依赖 core → 收敛到共享层。
- **R28（P2，§10.B）** `lf/legacy_redshift.py:585 _snr_li_ma_counts` 为**零调用死代码**且公式偏离规范 Li&Ma（无符号化：负超出截断为 0；ε 混入 log）→ 删除或对齐 signed 语义。
- **R29（P3，§10.C）** 10 个零引用死定义（含 core/utils 里的 IPython 演示类 `HydroDynamics` 放错层）→ 删除或接线。

## 5. 本轮变更文件

新增：`packages/jinwu/src/jinwu/core/skymap.py`、`test/test_core_skymap.py`、`scripts/check_tracked_assets.py`、`reviews/`（本报告 + 21 份证据）、`implementation_plan.md`。
修改：`subthreshold/search.py`、`packages/jinwu-fermi/pyproject.toml`、`packages/jinwu/pyproject.toml`、`jinwu-gw/{__main__,skymap}.py`、`Makefile`、`packages/jinwurs/conda-recipe/meta.yaml`、`recipe/meta.yaml`、`.readthedocs.yaml`、`.gitignore`、`docs/changelog.rst`、`REUSABLE_FUNCTIONS.md`。

## 6. 未验证范围（第四轮深审后更新）

~~以下 5 项为第一轮遗留，第二轮已全部完成~~（见 §7）；更新后的未验证范围：

1. CI 3.11/3.13 矩阵与 RTD 实机构建仍未实测（本地已模拟无 HEASoft 场景）。
2. `swift/bat/survey.py`（6853 行）：物理核心（signed_snr/TOTSNR、background_scale、经验灵敏度、Gaussian GLS 上限、幂律能流换算）已逐式审查并数值验证（§8.5）；`BatAnalysisSurveyBackend` 执行编排与 stage 管线部分未逐行深读，由 1416 项本地契约测试兜底。`attitude.py` 与 `gbm/pipeline.py` 物理核已审（§9）；简洁性维度已扫描（§10）。
3. `core/fit.py` / `bxa_fit.py` 的 XSPEC 拟合统计路径：模型选择度量（AIC/AICc/BIC/Bayes）、ON/OFF 似然原语、GLS 上限与 XSPEC 会话计数语义已审查（§8）；flux（calcFlux）取值路径为 XSPEC 内部计算，契约问题已在 R22 记录。

## 7. 第二轮深审（2026-09-20 补充）

### 7.1 全量本地套件复现 — 超越提交声明

`pytest test/ packages/jinwu-swift/tests -m 'not network'`（hea，含真实数据与 HEASoft）：
**1416 passed, 54 skipped, 3 deselected, 15 subtests passed（27.1s）** —— 提交 92987b0 声明的 "939 passed/42 skipped" 已被后续开发覆盖并保持全绿（evidence/22）。同时再次凸显 R5：**1416 项中仅 ~242 项对 CI 可见**。

### 7.2 事件级 `txx`（原"A&A 5.4 方法"）— ✅ 方法学核实，引用标签需改

- 贝叶斯块：`astropy.stats.bayesian_blocks(fitness='events')`，适应度 N_k·ln(N_k/T_k)−ncp_prior = Scargle 2013 Eq.19（astropy 同式实现）——正确。
- 块显著性：`li_ma_snr` 与独立实现的 **Li & Ma 1983 (ApJ 272, 317, doi:10.1086/161095) Eq.17 逐位一致**（9 组用例 max diff 3.6e-15），且满足零假设性质 N_on=α·N_off ⇒ S=0（α∈{0.1…5} 全部为 0）；负超出取负号、退化回退 net/√(S+B) 合理。第一轮我的"参照实现"把 Eq.17 第二项多写了 α 因子导致误判 NaN，已纠正（正确式中第二项无 α）。
- 时长约定：T90=累计净计数 5%–95%、T50=25%–75%，多次穿越取首末中点（battblocks 惯例）。引用核实：**Koshut, Paciesas, Kouveliotou et al. 1996, ApJ 463, 570（doi:10.1086/177272）确为《Systematic Effects on Duration Measurements of Gamma-Ray Bursts》**（T90 测量系统学）；5%–95% 定义的原始出处为 **Kouveliotou et al. 1993, ApJ 413, L101**（"T90: the time during which the integral counting rate goes from 5% to 95%"）——代码引用均名副其实。
- **R21（新，P2）**：事件级方法的用户可见标签 "A&A 5.4 方法"（`timescale.py:507` warning、docstring、`data.py:1516`）与 `method="aanda_2021_sec5_4"`（`timescale.py:1002`，`test_ops_b.py:192` 断言）指向"**某篇 A&A 2021 论文第 5.4 节**"，但该论文**在全仓库中从未被点名**——用户无法从 `method` 标识解析出文献。底层算法组件（Scargle 2013 贝叶斯块、Koshut 1996/Kouveliotou 1993 分位约定）引用齐全且核实无误，但"A&A 2021 §5.4"这一顶层出处必须补全引文（可能是作者自引或团队方法论文）。注意 `method` 标识是公开契约（测试断言），不宜单方面改名；应补引文而非改标识。另 `timescale.py:752` 作者顺序笔误（应为 Koshut, Paciesas, Kouveliotou）。

### 7.3 `physics/absorption.py` XSPEC worker 实跑 — ✅ 通过

- 实跑（evidence/23）：tbabs 与 vphabs(vern) 两后端在 0.5–10 keV、NH=1e22 cm⁻² 下成功出表；内部自洽性 σ=−ln(T)/NH 偏差 **4.5e-39**。
- 闭合语义：vphabs 元素分解闭合 ~1e-7（status=ok）；tbabs 闭合 0.3% 并**显式标记 `decomposition_not_closed`**（H2/尘埃化学非线性，非线性求和）——"闭合失败显式掩码"按文档工作。
- 数值方法审查：`callModelFunction` 直取模型；自适应柱深保持 τ∈[0.2,2] 规避单精度透射下溢；逐元素探针有限差分（tbabs 探针用真实丰度比、vphabs 用单位探针——正确处理线性/非线性）；vphabs `Xset.abund` 缓存陷阱已显式处理。
- 文献一致性：T(1 keV, N_H=1e22)=0.176 ⇒ N_H=1e21 时 T≈0.84，与文献常用值一致（XSPEC wilm 丰度 tbabs）。

### 7.4 OGIP / ftverify 对齐 — ✅（77 项专项测试通过）

`test_ogip.py + test_writefits_roundtrip.py + test_validate_ftverify_alignment.py + test_xselect_external.py`：**77 passed**（含 ftverify 对齐检查）。补齐第一轮 §6.4。

### 7.5 `jinwurs`（Rust）— ✅ 逐位一致 + 守恒精确 + 67× 提速

- 源码审查（lib.rs，127 行）：重叠分数分配（线性重分箱约定）、高斯方差传播 `(e·frac)²`（注意：跨新箱共享父箱计数会产生相关误差——与 XSPEC 分数重分箱同等约定）、二分定位边界正确、零暴露 NaN 语义明确。
- 数值验证（evidence/24、25；wheel 以 `maturin build --manifest-path packages/jinwurs/Cargo.toml` 构建，与 Makefile/CI 一致）：
  - `rebin_lightcurve_rs` vs Python `rebin_lightcurve`：3 种 binsize × 2 方法 **max|Δvalue|=0、max|Δerror|=0（逐位一致）**；
  - 守恒性：300 组无缝连续箱全覆-new-grid，max|Σnew/Σorig−1|=**4.4e-16**；部分覆盖与解析重叠求和差 **6.4e-16**；
  - 性能：200k 箱 10× 重分箱 **0.007s vs 0.491s（67×）**。
- 运维发现：`jinwurs` 未装入 hea 主环境——`rebin_rs.py` 的 Python 回退干净（`_HAS_RUST` 标志 + 显式 ImportError 提示），全套件在纯 Python 路径下通过；Rust 路径由 `testrs/test_rebin_rs.py`（skipif 守卫）与本轮独立验证覆盖。另：从仓库根直接跑 `maturin build` 会命中根 `pyproject.toml`（无 build-system）报错，Makefile/CI 均用 `--manifest-path`/cd，正确。

### 7.6 subthreshold 经验校准/FAR — ✅ 方法学正确

- `estimate_candidate_far`：Garwood 精确泊松区间（χ²(2k)/2T、χ²(2(k+1))/2T，k=0 时单侧上限 −ln α/T）✓；FAP=1−exp(−FAR·T_search) 用 `expm1` 保精度 ✓；"平稳泊松 clustered-events 假设独立于 search trials" 诚实声明 ✓。
- `calibration_from_searches`：配置+定义域指纹去重、**部分重叠离源域拒绝**（防重复计数）、区间合并后计 livetime ✓。
- `calibrate_targeted_search`：离源时刻与源数据上下文（含 margin）重叠检查、离源互不重叠检查、离源运行不带目标校准文件、不自动挑选"科学代表性样本"——边界处理与科研诚实性均正确。验证锚点 GW170817（arXiv:1710.05834）/GRB140606A（arXiv:1806.02378）由 `scripts/validate_gbm_subthreshold.py` 提供（数据在 `.runtime/gbm-search-validation/`）。

### 7.7 Swift BAT survey 物理约定抽样 — ✅ 与官方文档逐式一致

`signed_snr`/TOTSNR：`TOTAL(RATE)/SQRT(SUM(BKG_VAR**2))` 与本地 `external_sources/heasoft-6.37/swift/bat/tasks/batsurvey/batsurvey.html` 官方定义及 `external_sources/BatAnalysis-main` 的 `snr_allband` 逐式一致（survey.py:378-403 注释内引用可查证）；pcode（部分编码比）逐点携带并支持 strict/lenient 过滤；BAT 能段 14–195 keV 引 Barthelmy 2005 (doi:10.1007/s11214-005-5096-3)。AUD-01~06 全部确认修复（06：conf.py 版本取发行元数据 + RTD 按 packages/ 安装，本轮再补 jinwu-gw）。

### 7.8 `fit.py` `_generate_xspec_result` 静默吞异常 — ✅ 方法学成立但契约缺口已明确（R22）

- `_generate_xspec_result`（`fit.py:2205`）是 XSPEC 拟合结果字典化函数。
- **已知问题一（键名破坏兼容）**：`result['conversion']` 中 `counts` 键在 master 曾名为 `counts`，beta 已改为 `total_counts`；下游脚本直接取 `result['conversion']['counts']` 會 KeyError，用 `.get` 則靜默得 None。**FIX 注释明确标注**（`fit.py:2227–2230`），属**发布兼容性与记录准确性问题**，非计算错误。
- **已知问题二（静默吞异常）**：flux（`AllModels.calcFlux`）、rate、conversion factor、statistics 四段均为裸 `except Exception:` 后置 `None`/`{}`，且未写 `warnings_list`（該列為「组件级异常应追加、非静默吞掉」的约定）。
  - 我的审计（`evidence/27-generate-xspec-result-audit.txt`）列出了 `fit.py` 内所有裸 `except Exception:`（共约 17 块）：多数位於绘图/模型拼接/背景读取等脈絡，可接受為「不绘不报」；但 `_generate_xspec_result` 中 4 块是**科学输出缺失无告警**（flux_abs=None / rate=None / statistics={}）。
  - 方法学本身無错：flux/rate/statistics 取自 XSPEC 会话对象，数值计算均位于 XSPEC 内部。
- 结论：**属于契约/鲁棒性缺口**，修复方案 FIX 注释已给出（捕获后追加 warnings_list 再置 None，并标明异常来源）；**建议在发布前修复或至少在文档中显式列为「拟合结果字典的部分字段在 XSPEC 会话异常时为 None 且不警告」**。

## 8. 第四轮深审（2026-09-20 补充：统计原语 + XSPEC 语义实证 + survey/GBM）

### 8.1 `model_comparison.py` ON/OFF 边缘似然 — ✅ 推导与数值全部核实

- **解析推导（手核）**：`n~Pois(s+αb)`、`m~Pois(b)`、`b~Gamma(θ,r)` 下，对 b 边缘化精确等于二项式展开级数（每通道 logsumexp，k=0..n）。代码公式逐项与手推一致（`lnC(n,k)` 中的 `lnΓ(n+1)` 与外层 `−lnΓ(n+1)` 相消，数学等价）。
- **数值验证（evidence/28）**：
  - 40 例随机参数 vs mpmath 任意精度积分（峰值分段 bracket）：max|diff|=**8.4e-4**（mpmath 自身残差）；
  - s=0 闭式（负二项恒等式）：25 例 max|diff|=**2.3e-13**；
  - `PreparedOnOffMarginalLikelihood`（分块缓存+回退）vs 参考实现：**1.1e-11**；
  - 诚实记录：前两次参照（`math.factorial` 溢出、naive `[0,∞)` mpmath quad 漏掉 b~275 处尖峰）给出 51.9/103 的**假失配**，均为参照积分缺陷，已纠正留证——非代码缺陷。
- **`onoff_log_profile_likelihood`**：一阶条件为二次方程 `α(α+1)b²+[(α+1)s−α(n+m)]b−ms=0`（手核求导正确）；数值稳定根选择（bb≥0 走 `2ms/(bb+√disc)`）为标准防相消公式；50 例 vs 有界优化 max|diff|=**9.1e-13**。
- **W-stat 桥接声明核实**：按 XSPEC 手册 W-stat 公式实现参照，同一数据 6 个不同模型下 `−profile−norm−W/2` 的 spread = **0.0 精确**——docstring "差值仅为 data-only 常数" 成立。
- `empirical_tail_probability`（(k+1)/(B+1)、Clopper–Pearson beta 区间、z 排序）与 `summarize_bayes_factor`（误差正交合成、exp 溢出保护）均为标准实现。

### 8.2 PyXspec `background.values` 语义实证（真实数据）— R23 发现

- 数据：EP WXT 源/背谱对（`test/ep11916655873wxtCMOS21_jinwu/spectra/t90/`，t_src=t_bkg=1589.265 s，α=BACKSCAL 比=0.17675）。
- 实证（evidence/28 §1）：**`Spectrum.background.values` 是已缩放到源区的速率**：逐通道 `b.values×t_src/α == B_raw`（max dev **4.4e-16**）；而 `b.values×t_bkg = α·B_raw ≠ B_raw`。
- 结论：`core/model_comparison.py::recover_raw_off_counts` 的 docstring 公式 `m = round(r_bkg,src·t_on/α)` **被实证精确成立**。
- **R23（新，P2）**：`fit.py:1970 _xspec_spectrum_counts` 的 `background_counts = Σ b.values×b.exposure = f·B_raw`（BACKSCAL 缩放后的 OFF 计数）：等曝光时等于源区期望背景计数，但**既非原始背景区计数，在曝光不等时也不是源区期望**；`XspecChainResult.background_counts` 字段无语义注释，易被下游当作原始 OFF 计数消费。建议：docstring 澄清语义，或改用 `b.values×t_src/α`（与 `recover_raw_off_counts` 对齐）。同类用法在 `bxa_fit.py:596`。

### 8.3 `fit.py` 模型选择度量 — ✅ 定义正确

- `calculate_model_fit_metrics`：`n = dof + k` 对 χ² 与 C-stat（deviance = −2lnL+常数）均等于拟合 bin 数（XSPEC dof = 使用通道数 − 自由参数）；AICc 修正项、BIC 的 `k·ln(n)` 与 Burnham & Anderson 惯例一致；`n ≤ k+1` 时 AICc 显式置 None。
- `calculate_bayesian_model_metrics`：logz 排序、logzerr 误差棒、delta 语义与 BXA 体系一致（前轮已核）。

### 8.4 `core/upperlimit.py::profile_gaussian_upper_bound`（BAT survey GLS 上限核）— ✅ 闭式对照精确

- 二次型 GLS 精确解手核：`Â = tᵀC⁻¹r/(tᵀC⁻¹t)`，σ_Â=1/√curvature；**Â≥0 时 A_up=Â+√(Δ·σ_Â²)；Â<0 时锚定边界 A_up=Â+√(Â²+Δ/curv)** —— 两种 regime 均与代码的 brentq 求根一致。
- 数值验证（evidence/29）：60 例**非对角**协方差（Cholesky 路径）max|A_up−解析|=**6.9e-12**；20 例多观测 block-diag 联合 vs 联合解析 = **6.3e-12**；flux 换算路径恒等。
- `_core_survey_gaussian_profile`（survey.py:2533）：SYS_ERR 按 OGIP 分数约定**只加一次**（|rate|·fraction）、covariance=diag(stat²+sys²)、模板=XSPEC fakeit(applyStats=False) 折叠计数/曝光、`_powerlaw_energy_flux_per_norm` 解析 ∫E^{1−γ}dE（含 γ=2 极限）与 keV→erg=1.602176634e-9 换算均正确；置信约定 `Φ(√Δχ²)` 单侧高斯等价标注一致（与 R20 呼应，`delta_stat=9` ⇒ Φ(3)）。链清除（AllChains.clear）防污染处理正确。

### 8.5 `estimate_bat_survey_sensitivity` — ✅ 方法成立（经验校准框架）

- 固定位置 TOTSNR 代理统计量在无源控制样本上的经验分位阈（`method="higher"`）+ 确定性注入-回收二分求 target_power；控制样本不足/零变化/超出行数时**全部显式 unavailable/needs_review**，不用高斯尾外推——科研诚实性良好。注入模型（信号线性加在 total、噪声不重随机化）已声明为代理。引用 Cowan et al. 2011 (arXiv:1007.1727) 恰当。
- 二分收敛性：power 对 amplitude 单调（每控制点 stat 单调增）✓；初始 bracket 取 `threshold·median(noise)/Σtemplate` 合理。
- **R24（新，P3）**：`_binomial_proportion_interval` 的 Wilson fallback 注释称"deliberately widened"——实际 Wilson 区间通常**窄于** Clopper–Pearson；效果是无 SciPy 时门控**更严**（保守安全、方向无害），但注释推理与事实相反，建议更正措辞。

### 8.6 GBM `response.py` / `urls.py` / poshist 下载路径 — ✅ 与官方接口一致（两处小疣）

- `build_gbm_response_command`：非 shell argv、`-C<cspec|ctime>`/`-d<N>`（0–13：n0–n9/na/nb/b0/b1）/`-R/-D/-S/-E`/目录最后 —— 与官方 SA_GBM_RSP_Gen.pl 文档一致；RA/Dec/MET 校验完备；探测器名归一（`nai_N→n(N-1)`、`bgo_N→b(N-1)`）正确。
- **R25（新，P3）**：`urls.py::generate_download_url` 计算 `filename` 局部变量后**不返回**（死变量），函数只返回目录 URL，调用方无法取得文件名；实际下载路径已由 `pipeline.py` 的 GDT `ContinuousFinder` 承担，此函数仅为遗留垫片——建议补 filename 返回或标注 deprecated。
- response.py 小疣（不立编号）：workdir 已存在同名 `.rsp` 时，glob 差集为空会误报 "created no new file"（生成器成功覆盖写时）；`nai_0` 风格入参会以 ValueError 拒绝（约定为 `nai_1↔n0`），失败模式安全。

## 9. 第五轮深审（2026-09-21 补充：姿态、GBM 管线物理核、EP WXT 改动）

### 9.1 `swift/bat/attitude.py` — ✅ 与 BatAnalysis 上游逐行一致（R26）

- `POINTING[:,0/1/2]` 提取与 `FLAGS` 列位序 [10角分稳定, 稳定, SAA, safehold] 与上游 `batanalysis/attitude.py`（58–73 行）逐行一致；四元数仅存档（scalar-last 约定）不做换算，声明与上游同口径。
- jinwu 新增的 360° 解缠绕线性插值（RA/roll 圆周量）方法正确（numpy.unwrap + 回卷 [0,360)），上游无插值器。
- **R26（新，P3）**：`pointing_at`/`roll_at` 用 `fill_value='extrapolate'` 在姿态文件时间范围外静默外推，与 `gbm/poshist.py` 的"外推拒绝"策略不一致；由此 `bat_observation.py:289-294` 的 mid-point 回退分支不可达（pointing_at 永不抛错）；`_parse_sao` 缺列时静默以零指向替代（RA=0/Dec=0 危险）→ 建议加范围守卫（越界抛 ValueError）并将 `_parse_sao` 回退改为显式失败。

### 9.2 `fermi/gbm/pipeline.py` 物理核 — ✅ 全部通过

- **`integrate_background_interval`**：用 `fitter.interpolate_bins` 在源区间上精确积分（规避 GDT `to_bak` 整边界箱纳入）；曝光取 `get_exposure(..., scale=True)`（死时间感知），缺 API 时**拒绝以几何时长替代**并显式报错；2D 情形曝光加权平均率与 quadrature 不确定度传播正确。
- **背景阶数选择**（`_background_holdout_statistics` + `_select_polynomial_background`）：80/20 holdout 上的 Poisson deviance（McCullagh & Nelder 1989 §2.3 定义，y=0 处理正确）、要求 order+2 训练箱；选择 = 最小 holdout 分 + 1-SE 资格线 + 残差门控，**资格阶在全窗重拟合后才产 BAK**（避免不同模型/曝光契约混评）；AICc 回退按每能道一套系数计 k（`k=(order+1)×n_channels`）。方法学全部成立。
- **`_ensure_gdt_polynomial_basis`**：对 GDT `Polynomial` 的进程内归一化补丁——绝对 MET ~8e8 下原始 `t^k` 基法矩阵条件数 ~1e19/1e28，归一化到 [−1,1] 后为**同一多项式空间的等价重参数化**（箱内 s^i 平均 = `(s_hi^(i+1)−s_lo^(i+1))/((i+1)ds)`），最小二乘解在数值上等价；span 首次评估时缓存并被所有后续评估路径（含 interpolate/to_bak）复用，系数语义不变；API 兼容性先行探查。数学上成立。
- **`single_response`/RSP2 选阵**：RSP2 中点插值为默认，但提供 GTI 曝光加权选项，且**仅在折合单位幂率率差 ≤ 容差时保留中点近似**（物理感知门控，metadata 持久化）；加权实现为 DRM 边界分解 + 重叠曝光加权平均（`_rsp2_time_coordinate` 绝对/触发相对 MET 歧义消解、1 ms 边界容差、不相交区间拒绝而非静默取首末 DRM）；0/1-based 通道重编号 TLMIN/TLMAX 同步平移；输入响应只读、输出隔离。

### 9.3 `ep/wxt/pipeline.py` 工作区改动 — ✅（仅注释 + 一处正确修复）

- +24/−1：两段方法注释（曝光图比值法 α 标定；贝叶斯块并段 + Li & Ma Eq.17——公式转写与第二轮已验证实现逐字一致）+ `plot_dpi→plot_density` 关键字修复（回退 beta 对 master 参数名的静默破坏性改名，`fit.py:2984-2986` 佐证）。

### 9.4 针对性回归 — ✅

`test_gbm_pipeline.py + test_gbm_pipeline_stages.py + test_gbm_poshist_predict.py + test_swift_grb_pipeline.py`：**50 passed / 1 skipped**（evidence/30）。

## 10. 第六轮深审（2026-09-21 补充：代码简洁性 / 重复度 / 死代码）

方法：AST 扫描 5 个发行包全部模块级 def/class 的零引用候选（逐个人工 grep 复核排除字符串/getattr 误报），加统计/物理小工具跨包重复扫描（evidence/31）。_vendor 内零引用函数排除（上游保真）。

### 10.A 跨包重复（R27，P2）

- **Clopper–Pearson(+Wilson fallback) 二项区间 ×3**：`core/model_comparison.py:106`（empirical_tail_probability 内联）、`survey.py:920 _binomial_proportion_interval`、`gbm/pipeline.py:1055 _binomial_interval` —— survey/gbm 两份结构完全同构（同一 beta.ppf 参数化、同一 Wilson 公式）。两仪器包均已依赖 `jinwu` core → 应收敛为 core 公共函数（如 `model_comparison.binomial_interval`）。
- **单位归一化幂率能流 ×2**：`survey.py:2725`（内联 1.602176634e-9）与 `gbm/pipeline.py:1786`（KEV_TO_ERG 常数）—— 同一解析积分，仅 γ=2 判阈差 1e-12/1e-9。建议下沉共享层（如 `jinwu.physics` 或 core utils）。
- 简洁性正面确认：Li & Ma 全仓库单一规范实现（`core.utils.li_ma_snr`），ops/timescale/ep/swift/lf 全部仅导入；`lightcurve/duration.py` 为兼容 re-export。

### 10.B 偏离规范的死代码（R28，P2）

`lf/legacy_redshift.py:585 _snr_li_ma_counts`（@staticmethod，全仓库零调用）：无符号化（负超出截断为 0 而非负号）+ ε 混入 log 内，偏离第二轮已验证的规范 signed `li_ma_snr`。数值演示（evidence/31）：`(5,200,0.3)` 规范 −8.5203 vs 死代码 +8.5203；`(0,100,0.2)` −6.0386 vs +6.0386；正超出情形逐位一致。风险：未来一旦被启用会静默引入与全仓库不同的显著性语义 → 删除（或需向量化时给 core 实现加 numpy 路径并对齐 signed 语义）。

### 10.C 死定义清单（R29，P3）

10 个零引用模块级定义：`ep/wxt/pipeline.py:634 _combined_region_mask`、`gbm/pipeline.py:1125 _background_holdout_score`（docstring 自认 historical wrapper）、`core/time.py:174 _leap_seconds_from_elapsed`、`core/utils.py:330 HydroDynamics`（IPython 演示类，放错共享层）、`core/xselect.py:1399 trim_events_to_gti`、`ftools/ftrbnrmf.py:145 map_bins_by_edges`、`ftools/region.py:221 _points_in_polygon_mpl`、`ftools/teldef_helpers.py:77/114`（rotate_vector_by_quat、rotmatrix_to_xform2d）、`lf/legacy_redshift.py:585`（=R28）。

### 10.D 本轮方法学抽查 — ✅

- `gti_intervals_for_paths`：GTI 交集 + 1e-9 合并，不把墙钟时长折算成曝光 ✓。
- `background_windows`：guard 锚定侧窗、远端截断、短窗旗标 + 有 FIX(M15) 记录的 1/4 下限 ✓。




## 11. 第七轮深审（2026-09-21 CodeReview 子代理 + 实测复核）

### 11.1 R30（Critical）：TimeFermi 改 TT 尺度后，`.datetime` UTC 分桶全线偏移 69.184 s

- 本轮工作区把 `core/time.py` TimeFermi 的 `epoch_val`/`epoch_scale` 改为 `2001-01-01 00:01:04.184`/`tt`（闰秒修复本身正确）。但 astropy `Time.datetime` 按**当前 scale** 渲染墙钟：实测 `Time(725040959.5, format='fermi')` scale='tt'，`.datetime` 比 `.utc` 快 **69.184 s**。
- `gbm/pipeline.py::_as_scalar_time`（L79-93）对传入 Time 不做尺度规范化，下游 6 处 `.datetime.date()/strftime()` UTC 日/小时分桶（`_poshist_paths_for_day` L249、`_download_poshist_for_day` L258、`find_gbm_poshist` L406、`fetch_gbm_products_for_interval` L737/753、`_utc_days` L1558、`_utc_hours` L1568、`_poshist_day_stamps` L2985）在 UTC 日界/小时界前 69.184 s 窗口内取错日期 → poshist/TTE/CSPEC 找错目录。`jinwu-gw/models.py::scalar_time` 同病。master 上 epoch_scale='utc' 掩盖了该假设，属**本变更集内部相互作用引入的新回归**。
- 修复：两个入口对 Time 实例统一 `value.utc` 规范化 + 日界回归测试；`time.py:362` docstring（"seconds since 2001-01-01 00:00:00 UTC"）需同步更新。

### 11.2 R31（Warning）：ftgrouppha `group_min_counts` QUALITY 折叠用 max()，与 HEASoft 官方语义不一致

- `ftgrouppha.py:69` `q = qual_arr[idx].max()`（注释"任一成员坏则整组坏"）；HEASoft 6.37 `heacore/heasp/pha.cxx:2475-2486` 官方规则为 **last-non-zero-wins**（与同批变更 `grppha.py::fold_group_quality` 一致）。反例：组内 quality `[2,1]` → max()=2，官方=1。同一输入经两个入口产出不同 QUALITY。

### 11.3 R32（Warning）：TimeGECAM/TimeHXMT epoch 疑似整体偏 69.184/66.184 s，仅"标注待复核"未修即入 0.2.0

- `core/time.py:411-459` 源内注释自曝若官方定义为"UTC 零点起算、TT(SI) 秒计数"则 GECAM 应 +9.184 s、HXMT 应 +6.184 s——跨任务联合触发关联的**静默** ~1 分钟系统差。

### 11.4 Suggestions（R33-R36）

- R33：`find_gbm_poshist` 的 `except KeyError` 兜底分支绕过时间范围守卫（生产中损坏 FITS 可被标 observed），兜底应校验文件名日期戳。
- R34：`fold_group_quality` O(组数×通道数) 嵌套扫描，大谱可向量化。
- R35：grppha 两处 min_counts 贪心逻辑双份实现，易漂移。
- R36：core 弃用垫片在未装 jinwu-fermi 时抛裸 ModuleNotFoundError，应附迁移指引。

### 11.5 验证记录

- 回归：`pytest test/test_base_time.py test/test_ogip.py test/test_utils_txx_units.py test/test_gr.py` → 215 passed（子代理实测）。
- C1 运行时实验由本轮复核复现（scale='tt'、差 69.184 s）；W1 对照 HEASoft pha.cxx 确认。
- 子代理抽样深读 upperlimit/timescale/utils/gbm pipeline/survey/subthreshold 及测试，确认除上述外无新的未记录正确性问题。

## 12. 第八轮深审（2026-09-21 CodeReview 子代理，工作区未提交变更）

### 12.1 R37（Critical）：`normalize_superevent_id` 损坏 MS/TS 超事件 ID 的合法小写后缀

- `jinwu-gw/gracedb.py:38`：`text.upper() if text[:2].upper() in {"MS","TS"}` 把日期后的小写字母后缀一起大写化。实测：`'MS230615az'→'MS230615AZ'`、`'TS230615abc'→'TS230615ABC'`（`'S230615az'` 分支正确）。GraceDB 官方 ID 后缀恒为小写，损坏 ID 令 metadata/files/notices 全部 API 404——中子星并合（MS 事件）恰是 GBM 覆盖分析核心场景。`_SUPEREVENT_PATTERN` 带 IGNORECASE 接受合法输入再损坏之；test_gw_alert_security/test_gw 均无 MS/TS 用例（测试盲区）。
- 修复：`return text[:2].upper() + text[2:].lower()` + 参数化回归用例（MS181101ba / ts230615abc / s230615az）。

### 12.2 R38（Warning）：三处 `GbmPosHist.open()` 从不关闭，`find_gbm_poshist` 历史日循环放大泄漏

- `poshist.py:188`、`pipeline.py:310`、`pipeline.py:543`（本轮复核补充第三处）。GDT `FitsFileContextManager.open` 持有打开的 `HDUList`（原生支持 with），jinwu 侧从不 close；`find_gbm_poshist` 对 offset 1..2 历史日逐文件调用 `_poshist_time_range`，单次选档泄漏 1–3 个 POSHIST（数十 MB/个，memmap 依赖 GC 终结）。修复：三处均改 `with GbmPosHist.open(path) as history:`。

### 12.3 R39（Warning）：`response_folding="rmf_arf"` 预设与 `response_type` 的隐式耦合缺发行警示

- `config.py:280-288/875-917`：FXT/WXT 预设保留 `response_folding="rmf_arf"`，隐式要求 `response_type="rmf"`（`__post_init__` 自动维护，机制经实验确认正确）；但 changelog 与字段文档未告知第三方 preset 子类覆盖 `response_type` 时会在 upperlimit 契约检查处以难懂错误失败。修复：docstring + changelog 补一句耦合说明。

### 12.4 Suggestions（R40-R42）

- R40：`grppha.py:254-258` 非 rebin 路径 quality 折叠缺长度防御（同 diff 的 ftrbnpha/ftgrouppha 均有降级处理），对齐 `q_in.size != cnt.size` 守卫。
- R41：`ftrbnpha.rebin_pha` factor>1 时静默丢弃 EBOUNDS（`ebounds=None`），与 `core/ops.py::rebin_pha`（完整聚合）行为不一致且未写入 changelog；至少补 docstring/changelog 或对齐聚合。
- R42：`TimeFermi` docstring 仍写 "seconds since 2001-01-01 00:00:00 UTC"，未注明 TT 计数口径（同文件 MAXI 类 L536-538 有示范），易诱发 R30 回退；补 "MET counts SI seconds on the TT clock; MET=0 ↔ 2001-01-01T00:00:00 UTC; 墙钟消费用 .utc"。

### 12.5 已审无发现（子代理逐项声明）

完整性：jinwu.response 删除零残留引用；plot_density 无残留调用；fit_prepared 新签名三处调用点显式传参；jinwu-gw 发行配套（CI/publish/RTD/Makefile/docs）全覆盖；config 预设词表合法。正确性：gw/cli 互斥组（更正此前快读误判）、skymap 边界、gw/pipeline 字段继承、alert.py IPv4-mapped 拒绝（实验证实）、ftrbnrmf REDIST 与 HEASoft rmf.cxx:1361-1387 逐行一致、PreparedOnOff 系数逐项核对、新测试均为强断言。影响面：lightcurve 无耦合、swift/grb 消费链不受影响。

### 12.6 验证记录

- R37 由本轮独立复现（normalize 实验四例）；R38 三处 open 由 grep 证实（较子代理多一处 pipeline.py:543）。
- R30 的 .datetime 偏移在第八轮实验中再次复现（Time(1e9,'fermi')：utc 01:46:35 vs datetime 01:47:44.184），未修复状态确认。

## 13. 第九轮深审（2026-09-22 CodeReview 子代理：编排/管线层）

### 13.1 R43（Warning）：BATSurveyPipeline CALDB 全树 sha256 无 memoization，与同文件注释自相矛盾

- `survey.py:4803-4832` `_stage_input_fingerprint` override 对 survey/mosaic/spectra/fit 4 个 stage 每次调用全树 `rglob`+逐文件 sha256、无缓存；而 `survey.py:5002-5006` 注释声称 "the directory location itself is already part of the run fingerprint and **avoids recursively hashing a large calibration tree on every resume**"——两份声明直接冲突，实际行为恰是注释声称已避免的。代价：resume 全命中 = 4 次全树哈希；实验外推 CALDB 20 GiB 热缓存 ~9 s/次（resume 恒付 ~36 s），冷盘最坏数分钟。正确性不受影响（指纹内容正确，无 false hit）。
- 修复：CALDB 内容指纹提取为 `@lru_cache(maxsize=1)` 的进程内 memoize 函数（或持久化 `(size, mtime_ns)→sha256` 映射），并同步改写 L5002-5006 注释消除矛盾。

### 13.2 R44（Suggestion）：`_staged_result_cache` 在 `output_dir=None` 时 mkdtemp 目录永久残留

- `survey.py:3549-3553`：直接调用适配器时每次 `load_observation` 都 mkdtemp + copytree 整个 result cache（可达 GB 级），对象销毁后 /tmp 残留无人清理。管线内路径（output_dir 非 None）不受影响。修复：`weakref.finalize(observation, shutil.rmtree, workspace, ignore_errors=True)` 或 docstring 注明调用者负责。

### 13.3 R45（Suggestion）：BB 段谱缺少 t90/t100 谱同级的 count 交叉校验

- `wxt/pipeline.py:2288-2343`：`merge_bayesian_blocks_for_spectra` 的 n_on/n_off 基于 full band（PI 50-400）过滤事件，而 bb 段 PHA 以 `filter_energy_band=False` 写全通道谱；`_stage_ogip_finalize` 对 t90/t100 谱有 event/PHA 计数交叉验证（不一致即 raise），bb 段无对应兜底——口径差异可为合理设计，但缺校验使未来事件裁剪规则变化的计数漂移不可发现。修复：补 PI 50-400 窗口内 PHA vs 事件计数断言，或 docstring 显式声明口径。

### 13.4 已审无发现（子代理逐项声明，关键实验留证）

- **BatAnalysisSurveyBackend**（survey.py L3485-4406）：staged tree symlink/占位拒绝、mosaic monkey-patch finally 恢复、无 EXPOSURE 列保守排除、download 重试/保守停止、4 组临时环境 helper 全部 finally 恢复——隔离与异常路径完整。
- **BATSurveyPipeline stage 机制**：`safe_extract_archive` 路径穿越防护完整（zip resolve+is_relative_to+拒 symlink；tar 逐成员预检+filter="data"+3.11 回退无缺口）；`_input_fingerprint` 注入对象 `compare=False` 且 Time 显式 `.utc`（规避 R30 同族）；stage_config 按 stage 拆分对称；部分失败降级为显式 warning 而非静默吞掉。
- **core/pipeline.py 通用基础设施**：五重指纹 + outputs 存在性 + output_fingerprints 内容校验 + 非 COMPLETED 不复用，无 false hit 路径；`jsonable` dataclass 对称性实验验证；schema_version=2 老 manifest 有意判 stale（保守正确）；AUD-01 缺失依赖显式 RuntimeError；原子写 NamedTemporaryFile+replace。
- **wxt/pipeline.py 物理语义**：α 计算显式防护（L1890-1892）；`_finalize_pha_pair` 写回后重读重建 α 闭环自验证；**`_pipeline_t0` 经 astropy 实验验证非 R30 同族**（TT MET 真值差 0.0，provenance 往返 −1.5e-08 s）；BB 合并 searchsorted 边界无重复无遗漏；fit method 指纹纳入 stage_config（Major B，有回归测试）。
- **编排层测试**：`test_stage_code_deps.py + test_pipeline.py` 13 passed（本轮运行）；test_wxt_pipeline 关键契约断言（α 产品选择/回退、backscal 往返、t0 对齐 MJDREF、BB 余段、全流程 fake backend）与实现一致，无契约漂移。

### 13.5 回归记录

`pytest test/test_stage_code_deps.py test/test_pipeline.py` → 13 passed（子代理实测，含 AUD-01 与 Major B 回归）。R30/R37 仍开放未修复，本报告不重复。

## 14. 第十轮深审（2026-09-22 CodeReview 子代理：upperlimit/ops/GW 绘图层/backprior 兜底）

### 14.1 结论：无新增 Critical/Warning

5 个 Warning 级候选全部被实验或源码对照排除，逐项留证（`.tmp_review_r10/`）：
1. `refined_probability` 疑似低估 40% → **排除**（上轮实验自身 UNIQ 构造 bug；修正 `uniq=4·4⁹+arange(12·4⁹)` 后偏差随 cap 像素数收敛 +0.03%~+13.5%，与格内均匀近似的量化行为一致，方向为正）。
2. `rebin_pha` 尾组输出语义 → **排除**（实验 + `grouping.cxx:313-317` 逐行对照：HEASoft 不完整尾组即每通道自成组+QUALITY=2；组内输入 q=1 经 last-non-zero-wins 保留、q=5 不被 tail_q=2 降级——注释准确）。
3. survey.py L6382 `plot_density` → **排除**（R27 同族回退修复，非 typo）。
4. `_saturated_loglike` 疑缺 gaussian 背景饱和项 → **排除**（手工推导：μᵢ=dᵢ 时 nuisance profile 恰取 b=b_meas，gaussian 项为零，lnL(d;d) 即联合饱和值，fit_statistic 标度无偏）。
5. attitude unwrap → 无新问题（R26 已记录）。
另：R30/R37 经 diff grep 确认仍开放未修复。

### 14.2 R46（Suggestion）：`layers.py` `tolerance = max(1e-6, 1e-4)` 是恒等于 1e-4 的死表达式

- `jinwu-gw/layers.py:81`：误导读者以为存在随尺度变化的容差设计。实验：1° 球冠 sky_fraction 相对解析偏差 +8.3%（绝对 6.3e-6 < 1e-4 走快速路径）、30° 2.6e-4 才走 adaptive——小 cap 相对误差可达百分之几但绝对占比误差 ≤1e-4，行为无实质危害（下游 refined_probability 链收敛于解析值 ±0.03%）。修复：改为 `tolerance = 1e-4` 并修正注释；若原意是小 cap 更严判据则分离相对/绝对条件。

### 14.3 R47（Suggestion）：`backprior.update_with_off_spectrum`/`update_with_on_bg_spectrum` 用 `np.isin` 对齐通道，不匹配时静默少选

- `background/backprior.py:448/452/492/496`：prior 有而 PHA 无的通道（PHA 已分组/裁能段）被静默丢弃，`a_total` 系统性低估、Gamma-Poisson 后验 rate 偏低且无告警。已核实全仓库无生产调用方（API 冻结层），故为 Suggestion；但 0.2.0 后一旦接线即是静默统计错误。修复：`missing = np.setdiff1d(self.channels, ch)` 非空即 raise ValueError（四处 sel 计算后各一）。

### 14.4 R48（跟踪项）：`utils.generate_xspec_result` 包装层诊断缺口 = R22 同族

- 本轮 diff 新增的 ⚠️ docstring 标注指出包装层未传 `warnings_list`，flux/rate/statistics 段静默异常在此入口不可见——核心问题即 R22（§7.8），发布前按 R22 方案一并收口即可，不单独立项。

### 14.5 已审无发现（关键依据）

- upperlimit.py 主体：`_onoff_profile_background` 二次方程与 gaussian 背景闭式 T²−(b+s−σ²)T−dσ²=0 手工推导核对一致；`_prepare_observation` 形状/alpha/系统差方差合成校验完备；`rmf_arf` 量纲链 ph·cm⁻²·s⁻¹·keV⁻¹×keV×cm²×s=counts 成立；R20 的 legacy 中心覆盖 vs 单侧分位语义区分清晰。
- GBM↔core.upperlimit 契约闭环：BAK/PHA 通道数与能量网格强制一致 → counts 压缩 + mask 形状校验 → `_prepare_observation` 显式 raise——多对一响应映射无静默路径。
- swift/grb/pipeline.py：grep 证实无 upperlimit 调用（职责为 catalog/duration/DAT 解析），BAT 上限消费链在 bat/survey.py；`response_type="rmf"` 预设与新契约无冲突。
- gw/plot.py：mollweide 11 ticks 与标签对齐（实验）；log 颜色 floor 取正密度分位 + masked_where 合理。
- backprior/cluster 本轮 diff 均为方法/参考注释，与既有 Gamma-Poisson/K-Means 实现逐条核对一致。

## 15. 第十一轮复核（2026-09-24：开放问题存在性再确认 + 双独立验证）

方法：工作区自第十轮（2026-09-22 17:29）以来**无代码变更**（git status 与文件 mtime 证实），故本轮对全部 10 项开放问题做存在性复核，并按审查流程派 2 个独立子代理并行交叉验证，另做主代理运行时实验留证。

### 15.1 运行时实验复现（主代理）

- **R30 复现**：`Time(725040959.5, format='fermi')` → scale='tt'，`.datetime` = 2023-12-23 16:17:03.684，`.utc.datetime` = 2023-12-23 16:15:54.500，**差 69.184 s**——UTC 日/小时分桶在边界窗口内取错日期确认可发生。
- **R37 复现**：`normalize_superevent_id('MS230615az')` → `'MS230615AZ'`、`'TS230615abc'` → `'TS230615ABC'`、`'ms230615ba'` → `'MS230615BA'`（S 分支 `'S230615az'` 正确保留小写）——MS/TS 后缀大写化确认。

### 15.2 双验证者共识（10/10 代码层面可复现）

| Issue | 验证者1 | 验证者2 | 共识结论 |
|---|---|---|---|
| R30 | major | critical | ✅ 存在；`_as_scalar_time`（gbm/pipeline.py:79-93）与 `scalar_time`（gw/models.py:21）均无 scale 归一化；下游 L249/258/406/737/753/1558/1568/2985 均消费 `.datetime` 做 UTC 分桶。严重度分歧：触发需调用方传入 TT-scale Time（jinwu 内部入口均构造 UTC Time），但属本变更集内部相互作用引入的新回归。 |
| R37 | major | critical | ✅ 存在；后缀大写化已实测。404 后果取决于 GraceDB 路由大小写敏感性（离线不可证），但官方 ID 后缀恒小写为事实。 |
| R31 | major | major | ✅ 存在；ftgrouppha.py:69 `max()` vs grppha.py:100 `fold_group_quality` last-non-zero-wins，成员 [2,1] → 2 vs 1，两入口产出不同 QUALITY。 |
| R32 | major | minor | ✅ 存在；time.py:418-427/443-452 ⚠️ 待复核注释仍在，epoch 未修。 |
| R38 | major | minor | ✅ 存在且**面更宽**：poshist.py:188、pipeline.py:310、pipeline.py:543 未关闭确认；**新增确认 gbm_observation.py:174、254 同样未关闭**。原报告把 subthreshold/data.py:210 列入系**误报**（L217 有 `history.close()`），予以更正。 |
| R46 | minor | minor | ✅ 存在；layers.py:81 `max(1e-6, 1e-4)` 恒等于 1e-4 死表达式。 |
| R47 | minor | major | ✅ 存在；backprior.py:448/452/492/496 `np.isin` 交集无缺失通道检查，观测项被静默丢弃。 |
| R43 | major | major | ✅ 存在；survey.py:4803-4832 每次调用全树 rglob+sha256 无缓存，与 L5001-5006 注释"avoids recursively hashing a large calibration tree on every resume"直接矛盾。 |
| R42 | minor | minor | ✅ 存在；time.py TimeFermi docstring "seconds since 2001-01-01 00:00:00 UTC" 与 epoch_scale='tt' 表述不完整（物理时刻等价，属文档缺口而非数值错误）。 |
| R26 | minor | minor | ✅ 存在；attitude.py L109-114 interp1d 均 `fill_value='extrapolate'`，pointing_at/roll_at 静默外推。 |

### 15.3 本轮更正

- R38 证据更正：`subthreshold/data.py:210` 已有 `history.close()`（L217），从泄漏清单移除；同时确认 `gbm_observation.py:174/254` 两处未关闭，R38 受影响位置更新为 5 处（poshist.py:188、pipeline.py:310、pipeline.py:543、gbm_observation.py:174、gbm_observation.py:254）。
- 无新增问题；无已修复问题（工作区未变）。

## 16. 第十二轮深审（2026-09-24：core 剩余模块——io/ops/xselect/products/plot）

方法：3 个独立子代理并行深读 `core/io.py`（1593 行）、`core/ops.py`（2262 行）、`core/xselect.py`（2075）+ `core/products.py`（1432）+ `core/plot*.py`（2449），主代理对全部 Critical 做运行时实验复核。共发现 **12 项新问题**（4 Critical / 6 Major / 2 Minor），编号 R49–R60。

### 16.1 R49（Critical）：ops 贝叶斯块聚合跨界 bin 被相邻两块重复计入，计数不守恒

- [ops.py:1180](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/ops.py#L1180-L1181) `mask = (left < b) & (right > a)`：块边界落在 bin 内部时，跨界 bin 同时满足左右两块。实测：edges=[0,1.5,4]、4 个等宽 bin（counts=[10,10,20,20]，总 60）→ 块 [0,1.5] 得 bin0+1（20），块 [1.5,4] 得 bin1+2+3（50），**聚合总 70 > 输入 60**。
- 影响：`fit()` 与 `fit_src_bkg`（L1421-1422 同模式）输出的块计数系统性膨胀；T90/通量等下游量被高估。
- 修复：改用 bin 中心归属判定 `(centers >= a) & (centers < b)`，或按交叠比例分摊 counts/var。

### 16.2 R50（Critical）：slice_pha 不裁剪 ebounds，链式 rebin_pha 崩溃

- [ops.py:588](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/ops.py#L588)：`ebounds=pha.ebounds` 原样透传。实测：切 ch3-4 后输出 2 通道配 6 行 ebounds；再 `rebin_pha(factor=2)` 在 L695 `ch_all, e_lo, e_hi = pha.ebounds` 处 `ValueError: too many values to unpack`（ndarray 情形直接崩；三元组情形下按新索引取旧表 → 能量边界错位）。
- 修复：切片时同步 `ebounds = (ch[sel], e_lo[sel], e_hi[sel])`（三元组）或 `pha.ebounds[sel_idx]`（ndarray 情形）。

### 16.3 R51（Critical）：EventWriter TIMEZERO 回写不一致，绝对时间 round-trip 漂移

- [io.py:1370-1405](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/io.py#L1370-L1405)：读入时 `time = TIME_raw − time_offset`、`timezero = TIMEZERO_raw + time_offset`；写出时回写 `TIME=time`（已减 offset）但 header 透传的 TIMEZERO 未经同步更新（L1384-1394 透传循环不覆盖已有键，TIMEZERO 保留原始值）。round-trip 后 `TIME + TIMEZERO ≠ 原始绝对时间`。
- 修复：写出时显式 `hdu_evt.header['TIMEZERO'] = ev.timezero`（内部统一值），确保 round-trip 后绝对时间一致。

### 16.4 R52（Critical）：RmfWriter 硬编码 TLMIN4 与条件列 N_GRP 冲突

- [io.py:1281-1282](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/io.py#L1281-L1282)：当 `n_grp is None` 时不写 N_GRP 列，列序变为 ENERG_LO(1)/ENERG_HI(2)/F_CHAN(3)/N_CHAN(4)/MATRIX(5)，但仍硬写 `TLMIN4`——此时 TLMIN4 描述的是 N_CHAN 而非 F_CHAN。HEASoft 按 TLMIN4 解析 F_CHAN 会得到 N_CHAN 的值。
- 修复：根据实际写出列序动态计算 F_CHAN 的列索引（无 N_GRP 时为 3，有 N_GRP 时为 4），写入对应 `TLMINn`。

### 16.5 R53（Major）：guess_ogip_kind 误判非标准后缀的 SPECRESP MATRIX RMF 为 PHA

- [io.py:1418-1425](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/io.py#L1418-L1425)：`.fits` 后缀进入内容探测分支，但 `extnames` 含 `'SPECRESP MATRIX'` 时不等于 `'SPECRESP'`、也不等于 `'MATRIX'`（扩展名含空格）→ 漏判 RMF；无 SPECTRUM/EVENTS/TIME → 默认返回 `'pha'`，下游用 OgipPhaReader 打开即 KeyError。
- 修复：增加 `any(name.upper() == 'SPECRESP MATRIX' for name in extnames)` 显式检查。

### 16.6 R54（Major）：LightcurveReader 对合法无 TELESCOP 文件强制 raise

- [io.py:748-749](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/io.py#L748-L749)：`telescop=None` → `_mission_timezero_object` 返回 None → 直接 `raise ValueError`。OGIP 光变曲线可以没有 TELESCOP/MJDREF（仅做相对时间分析），此检查过严。EventReader 对同情形更宽容，两入口策略不一致。
- 修复：降级为 warning 并允许 `timezero_obj=None`，与 EventReader 容错策略一致。

### 16.7 R55（Major）：PhaWriter 对 rate-only + 无效 exposure 输入破坏 COUNTS 语义

- [io.py:1045-1048](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/io.py#L1045-L1048)：读入时 exposure=nan/0 走 counts=rate 分支（不乘曝光）；写出时无条件写 COUNTS 列（=rate 值）+ EXPOSURE header（=nan/0）。下游读回会把 rate 误当 counts 用。
- 修复：写出前检查：rate-only 且 exposure 非正/非有限时优先保留 RATE 列（不写 COUNTS），或至少不写 EXPOSURE 关键字。

### 16.8 R56（Major）：XSelectSession.extract_image 从原始文件重读，丢弃所有过滤状态

- [xselect.py:1897-1913](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/xselect.py#L1897-L1913)：`extract_image` 无论是否传入 tmin，均直接 `read_evt(ev.path)` 并读原始 X/Y 列做 histogram2d，完全忽略 `self.current` 已应用的时间/能量/region/expr 过滤。实测：apply_region 后 current 剩 15 事件，extract_image 仍返回全量 2000 counts。
- 修复：优先使用传入 EventData 的 x/y 属性（若存在且与 time 长度一致），或把 `session.current` 的全部过滤显式应用到读出的列上。

### 16.9 R57（Major）：extract_curve/extract_image 的 tmin 分支在内存 EventData 上 TypeError

- [xselect.py:1767-1768](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/xselect.py#L1767-L1768)（extract_image L1912-1913 同）：`select_events(ev.path if ... else ev.path, ...)` 两个分支均传 `ev.path`；内存 EventData 的 path 为 None → `select_events(None, ...)` TypeError。
- 修复：tmin 分支直接传 `ev`（`select_events` 已支持 EventData 输入）。

### 16.10 R58（Major）：load_net_lightcurve FITS 路径因 astropy 大写 meta 键崩溃

- [products.py:437-438](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/products.py#L437-L438)：`table.meta["alpha"]` / `table.meta["timezero"]` 用小写键；astropy FITS 写出会把 meta key 大写化为 ALPHA/TIMEZERO → FITS 路径 KeyError，ECSV 路径正常（保留小写）。
- 修复：大小写不敏感读取 `float(table.meta.get("alpha") or table.meta.get("ALPHA"))`，timezero 同理。

### 16.11 R59（Minor）：rebin_pha factor=1 静默落入 grouping 分支；factor/min_counts "互斥"未校验

- [ops.py:616-640](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/ops.py#L616-L640)：带 grouping 的 PHA 传 factor=1 仍被聚合（与 docstring "既未提供 factor…才用 grouping" 矛盾）；factor≤0 静默忽略；factor 与 min_counts 同传时 min_counts 静默优先。
- 修复：factor is not None 时显式短路（factor≤1 返回 pha），同传两者则 raise。

### 16.12 R60（Minor）：bayesian_blocks_exposure n=1 返回 [0]，违反"首0尾n"契约并致 fit(use_exposure=True) 输出空 LC

- [ops.py:1006-1018](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/ops.py#L1006-L1018)：实测 `bayesian_blocks_exposure([5],[1])` → `[0]`；fit 中 edges=[left[0]] 单边 → nb=0 → 返回 0 长度 LC。
- 修复：循环后若 `change_points[-1] != n` 则追加 n。

### 16.13 其余已审无发现（子代理声明，主代理抽核）

- ops：`rebin_pha` factor 分组 off-by-one（n=6/f=3、n=10/f=3 边界正确）；OGIP ±1 grouping 转换；`rebin_events_to_lightcurve` GTI 左闭右开与 histogram 范围一致性；`_resolve_alpha_for_src_bkg` BACKSCAL 方向正确；`slice_events` 不重定基时间故 GTI 透传自洽。
- xselect：filter 字符串拼接经 `_normalize_region_paths` 校验存在性并拒绝空格；命令经 subprocess.run 直接喂给 xselect 而非 shell，无注入通道。
- plot/plotpanel/plotstyle：无坐标变换、log 域负值、颜色映射溢出、共享 axes 状态污染等正确性问题。
- products：ECSV/CSV 路径序列化对称。
- 待 owner 决策项（不立编号）：ops.py:2022 `_candidate_headas_dirs` 硬编码 `/home/xinxiang` 回退路径；ops.py:2049-2052 `/tmp/headas_pfiles` 固定目录多进程共享。

### 16.14 验证记录

- R49（O1）：主代理运行时实验复现计数不守恒（输入 60 → 聚合 70）。
- R50（O2）：主代理运行时实验复现 slice_pha ebounds 不裁剪 + rebin ValueError。
- R51/R52/R53/R54/R55：代码精读确认（io.py 各段引用见上）。
- R56/R57：代码精读确认（xselect.py L1913 两分支相同、L1767-1768 传 path）。
- R58：代码精读确认（products.py L437-438 小写键）。
- R59/R60：子代理运行时实验复现，主代理抽核代码路径确认。

## 17. 第十三轮深审（2026-09-24：core/config.py 配置系统 + spectrum_prep/base）

方法：2 个独立子代理并行深读 `core/config.py`（1484 行）与 `core/spectrum_prep.py`（333）+ `core/base.py`（256），主代理对全部 Major 及以上做运行时实验复核。共发现 **14 项新问题**（0 Critical / 8 Major / 6 Minor），编号 R61–R74。本轮无 Critical，但 **R63（BASE-01）影响面最广**：所有含 ndarray 字段的核心数据类 `__eq__` 必抛 ValueError、`__hash__` 为 None，贯穿管线的比较/去重/pytest 断言/dict 键使用全部中招。

### 17.1 R61（Major）：UpperLimitConfig sigma 别名对账使 dataclasses.replace 静默失效或误抛错

- [config.py:336-349](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/config.py#L336-L349)：`__post_init__` 把 `upper_confidence_sigma` 无条件规范化为与 `default_sigma` 相同（L362），`replace()` 时该残留值被当作"用户显式提供的第二个别名"。实测：`replace(UpperLimitConfig(default_sigma=2.5), default_sigma=3.0)` → **静默保持 2.5**；`replace(..., default_sigma=5.0)` → **误抛 ValueError**。代码注释（L337-340）明确声称支持 replace 工作流，但实际只对 baseline=3.0 的 preset 成立。upperlimit.py:1563 直接以 default_sigma 生成 OneSidedLevel，静默错误 sigma 会直接进入科学产物。
- 修复：对账逻辑应区分"用户显式传入"与"上次规范化残留"（如 `_sigma_explicit` 标志），或在 docstring 禁止对 sigma 字段用 replace 并提供 `with_sigma()`。

### 17.2 R62（Major）：calibration 与 calibration_mode 双轨字段无对账，且被不同代码路径分别消费

- [config.py:290-303](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/config.py#L290-L303)：docstring 称二者是同一概念的别名，但 sigma 别名有对账逻辑而 calibration 没有。两字段词表不同（bootstrap/asymptotic/empirical_if_available vs conditional_model/empirical_*），任意矛盾组合均通过校验。下游：upperlimit.py:1645 用 calibration_mode 判定经验校准；upperlimit.py:2191 用 calibration=='bootstrap' 决定是否 bootstrap。实测：`calibration='bootstrap' + calibration_mode='empirical_global_search'` 构造成功；运行时（无 adapter）走 1645 分支返回 unavailable，用户要求的 bootstrap 被静默跳过。
- 修复：参照 sigma 别名做一致性/优先级对账，或弃用其一并在 upperlimit.py 统一入口。

### 17.3 R63（Major）：BATSurvey 同时传 survey= 与 selection= 时 selection 被静默覆盖，违反自身注释承诺

- [config.py:1028-1127](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/config.py#L1028-L1127)：L1034-1036 注释明确写"Passing both groups remains an intentional way to keep them distinct"；但 selection 在 L1028 已从 kwargs pop，L1126 `if "selection" not in kwargs` 恒为 True，L1127 无条件 `defaults['selection']=defaults['survey']`。实测：`BATSurvey(survey=SwiftBATSurveyConfig(detthresh=8000), selection=SwiftBATSurveyConfig(detthresh=9000, min_pcode=0.5))` → **selection.detthresh==8000、survey is selection**，用户传入的 selection 组被静默丢弃。
- 修复：L1126 条件改为基于 supplied_selection 是否为 None。

### 17.4 R64（Major）：GECAM 的强制构造参数不是 dataclass 字段，dataclasses.replace 必然崩溃

- [config.py:1318-1335](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/config.py#L1318-L1335)：`GECAM.__init__` 要求 keyword-only detector 与 energy_range_keV，但 detector 不是 InstrumentConfig 的 dataclass 字段。实测：`replace(GECAM(detector='GRD01', ...), group_min_counts=5)` → ValueError。同仓库 grb/pipeline.py:2021 对 SwiftGRB 用 replace，说明 replace 是本项目公认的配置修改路径；GECAM 作为 __all__ 导出的公共类无法满足该契约。
- 修复：detector 提升为 InstrumentConfig 的可选字段（默认 None，GECAM 强制非 None），或为 GECAM 实现 `__replace__`；至少在 docstring 声明不支持 replace。

### 17.5 R65（Major）：GBM.replace() 静默把 detector 重置为默认值 NAI_1，产生 name/energy/detector 自相矛盾

- [config.py:1204-1236](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/config.py#L1204-L1236)：`GBM.__init__(detector='NAI_1', **kwargs)` 的 detector 是构造参数而非 dataclass 字段，replace 只回传字段。实测：`g=GBM(detector='BGO_1'); replace(g, group_min_counts=10)` → **name='GBM_BGO_1'、energy_range_keV=(200,40000) 保留，但 g.detector=='NAI_1'**，无任何警告。
- 修复：detector 字段化，或 __init__ 中从 kwargs 的 name/energy 反推一致性并校验。

### 17.6 R66（Major）：InstrumentConfig 非 frozen：直接属性赋值破坏 response_requires_arf 不变量；energy_range_keV 无任何校验

- [config.py:750-842](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/config.py#L750-L842)：所有子配置均 frozen=True，唯独 InstrumentConfig 是 `@dataclass(slots=True)`（非 frozen），而 L780-782 注释称 response_requires_arf"由 __post_init__ 自动维护"。实测：`w=instrument('WXT'); w.response_type='rsp'` 后 `w.response_requires_arf` 仍为 True。另外 `energy_range_keV`（L756）无校验：长度≠2、emin≥emax、非有限值均可通过。
- 修复：docstring 明确"不得直接给 response_type 赋值，须用 replace"，或将 InstrumentConfig 改 frozen；energy_range_keV 增加有限性/递增性校验。

### 17.7 R67（Major）：base.py 全部含 ndarray 字段的核心数据类 __eq__ 必抛 ValueError、__hash__ 被置 None

- [base.py:99-240](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/base.py#L99-L240)：dataclass `eq=True` 与 ndarray 字段组合使 `__eq__` 必抛 `ValueError: The truth value of an array with more than one element is ambiguous`；`__hash__` 被置 None。实测：`PhaBase(...)==PhaBase(...)` 抛 ValueError；`hash(PhaBase(...))` 与 `hash(ChannelBand(1,10))` 均 TypeError。ArfBase/RmfBase/PhaBase/LightcurveDataBase/EventDataBase 全部中招。作为贯穿管线的核心容器，任何去重/`in`/pytest 断言/dict 键使用都会以难懂的深层报错炸开。
- 修复：含 ndarray 的类改 `@dataclass(slots=True, eq=False)`（身份语义），或实现数组感知 `__eq__`（np.array_equal）；纯标量类若需可哈希加 `frozen=True`。

### 17.8 R68（Major）：group_min 回退读错配置层：取顶层 InstrumentConfig.group_min_counts 而非 SpectrumConfig.group_min_counts

- [spectrum_prep.py:193-194](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/spectrum_prep.py#L193-L194)：`resolved_group_min = int(group_min if group_min is not None else cfg.group_min_counts or 1)`。L228 注释声称"WXT=1/FXT=3"，但实测 `instrument("WXT").group_min_counts`=3（顶层），WXT=1 只在 `spectrum.group_min_counts`。wxt/pipeline.py:2412、swift/grb/pipeline.py:1095/2015、bat/survey.py:6290 全部读 `config.spectrum.group_min_counts`；而 fit()（fit.py:4029 不传 group_min）与 fit_bxa()（bxa_fit.py:1114 默认 None）走此回退 → **同一 WXT 数据经 fit() 入口按 min=3 分组、经 WXT 管线按 min=1 分组**，分箱不一致且与代码内文档相反。
- 修复：回退链改为 `cfg.spectrum.group_min_counts or cfg.group_min_counts or 1`，或修正注释与 fit.py:4025 的错误信息，统一契约到顶层字段。

### 17.9 R69（Major）：PreparedSpectrum.energy_range_keV 取 cfg.energy_range_keV，绕过 spectrum.fit_energy_range_keV 的全库既定回退模式

- [spectrum_prep.py:211/262](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/spectrum_prep.py#L211-L262)：两处 `energy_range_keV=cfg.energy_range_keV`。该值被 fit.py:2711-2717 与 bxa_fit.py:469/506-507 用作 XSPEC notice 回退。SwiftGRB 配置顶层为 (0.3,150.0) 而 fit 带为 (0.3,10.0) → 用户不显式传 emin/emax 时 XRT 谱会 notice 到 150 keV，10–150 keV 纯噪声道进入拟合。既定模式见 wxt/pipeline.py:1663-1664/2872、swift/grb/pipeline.py:1831-1832、bat/survey.py:6273。
- 修复：赋值改为 `cfg.spectrum.fit_energy_range_keV or cfg.energy_range_keV`。

### 17.10 R70（Minor）：staged ancillary 静默复用不校验指向，陈旧链接可致错误背景/响应进入拟合

- [spectrum_prep.py:129-131](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/spectrum_prep.py#L129-L131)：`if staged.exists() or staged.is_symlink(): if not overwrite: return staged` — 不验证 `staged.resolve() == source.resolve()`。触发链：上次运行 grppha 失败（staged 已建、grouped 缺失）或换 group_min 后，以 overwrite=False 重跑且输入路径已变 → 旧链接被静默复用，fit 以 status='ready' 消费错误 RMF/背景。
- 修复：复用前校验链接/文件指向当前 source。

### 17.11 R71（Minor）：_rewrite_grouped_header_links 盲写 hdus[1] 且 except Exception 吞错后 status 仍为 ready

- [spectrum_prep.py:171-179](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/spectrum_prep.py#L171-L179)：grppha 已通过 chkey 把同名裸文件名写入 SPECTRUM 扩展，本函数正常路径下纯属冗余；当 SPECTRUM 不在 HDU 1（如前置 GTI 扩展）时会把 RESPFILE/ANCRFILE/BACKFILE 写进错误扩展头；L244-247 任何失败仅追加 diagnostics，status 仍置 'ready'。
- 修复：按 EXTNAME='SPECTRUM' 定位 HDU（或删除该冗余回写）；失败时降级 status 或至少升级为 warning。

### 17.12 R72（Minor）：单个 bundle 的 FileExistsError 中止整个 prepare_spectra，与 ancillary 静默复用语义不一致

- [spectrum_prep.py:218-219](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/spectrum_prep.py#L218-L219)：`if grouped_pha.exists() and not overwrite: raise FileExistsError`；L302-313 循环无 try/except，首个已存在产物使后续全部 bundle 无产出。同函数内 ancillary 在 overwrite=False 时却静默复用（L130-131），复用策略前后矛盾。
- 修复：按 bundle 捕获异常记入 diagnostics/status，或提供显式 reuse 模式。

### 17.13 R73（Minor）：标记 ready 前未做 PHA↔RMF/ARF 兼容性检查，check_response_compatibility 在生产链路零调用

- [spectrum_prep.py:230-248](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/spectrum_prep.py#L230-L248)：全库 grep：`check_response_compatibility`（ogip.py:308-331）仅被 test 使用；instruments.py 扫描端 L666/L741 的 'validated' 措辞并无实际 validate() 调用。通道/响应不匹配要到 XSPEC 加载阶段才失败。
- 修复：grppha 成功后对 grouped 谱与 staged RMF 调一次 check_response_compatibility，失败记入 diagnostics 并降级 status。

### 17.14 R74（Minor）：GBM 探测器表对同一物理探测器族给出两种不一致的能段（8–1000 vs 8–900 keV）

- [config.py:1185-1202](file:///home/xinxiang/research/jinwu/packages/jinwu/src/jinwu/core/config.py#L1185-L1202)：detectors 字典：NAI_1..NAI_6 → (8.0, 1000.0)，而原生命名 N0..N9/NA/NB → (8.0, 900.0)。若 NAI_i 是 Ni 的历史别名，同一物理探测器经不同名字构造会得到不同 energy_range_keV。无注释解释 1000 与 900 的差异来源。
- 修复：确认映射关系并统一能段（或禁止混用），并在注释中给出标定依据。

### 17.15 已审无发现（子代理声明，主代理抽核）

- config.py：`fit_settings` 上下文管理器的 finally 恢复与嵌套语义正确；`set_fit_settings` 参数互斥校验完备；BATSurvey/GBMAnalysis/SwiftBATSurvey{Download,Mosaic}Config 的数值边界校验完备；`instrument()` 注册表键规范与冲突无异常；UpperLimitConfig 的 strategy×likelihood×response×combine 组合矩阵与 unsupported 态校验自洽；FXT/WXT/BAT/GBM/UVOT 预设通过全部现有校验。
- spectrum_prep：能量裁剪不在本层（下游 notice 完成）；symlink 断链双判、`_joint_spectra` 显式 FXTA/FXTB 索引、diagnostics 跨层传递与去重、`instrument()` 未知名称显式 ValueError，均正确。
- base.py：可变默认值全部经 `default_factory`，`slots=True`+`ClassVar kind` 无字段泄漏。

### 17.16 验证记录

- R61（CFG-01）：主代实测 `replace(UpperLimitConfig(default_sigma=2.5), default_sigma=3.0)` → 静默保持 2.5；`replace(..., default_sigma=5.0)` → ValueError。
- R63（CFG-03）：实测 `BATSurvey(survey=s1, selection=s2)` → selection.detthresh==8000、survey is selection。
- R64（CFG-04）：实测 `replace(GECAM(...), group_min_counts=5)` → ValueError。
- R65（CFG-05）：实测 `replace(GBM(detector='BGO_1'), group_min_counts=10)` → detector='NAI_1'。
- R67（BASE-01）：实测 `PhaBase(...)==PhaBase(...)` → ValueError；`hash(PhaBase(...))` → TypeError。
- R68（SP-01）：实测 `instrument("WXT").group_min_counts`=3、`spectrum.group_min_counts`=1，两入口不一致确认。

## 18. 第十四轮深审（2026-09-24：subthreshold 包整体）

方法：1 个子代理深读 `subthreshold/` 全部 7 个 py 文件（1230 行，untracked 不在 git 历史），主代理对 Major 发现做运行时实测复核。共发现 **4 项新问题**（0 Critical / 1 Major / 3 Minor），编号 R75–R78。默认配置下数值精确，唯一 Major 仅在非默认 min_duration/num_steps 组合触发。

### 18.1 R75（Major）：有效时间用最短窗中心±min_step/2 拼接，非默认配置下出现空洞，FAR 被静默高估

- [pipeline.py:170-176](file:///home/xinxiang/research/jinwu/packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/pipeline.py#L170-L176)：最短窗起始间隔 `step_bins = max(1, round(dmin/num_steps/base))`；只有间隔==min_step 时 `centers±min_step/2` 才无缝拼接。主代理实测（search_interval ±30s, gti 全覆盖, min_step=0.064）：
  - dmin=0.064（默认）：eff=30.016 s，**正确**（ratio 1.00x）；
  - dmin=1.024, num_steps=8：eff=15.040 s，实际最短窗并集 30.976 s，**低估 2.06 倍**；
  - dmin=8.192, num_steps=8：eff=1.920 s，实际并集 37.888 s，**低估 19.73 倍**。
- 该 intervals 经 `_stage_report` 作为 `estimate_candidate_far` 的 search_time、并经 calibration.py 作为 livetime_s。FAP≈FAR·T 两边同因子相消，但报告的 `far_hz` 本身被按 1/覆盖率 高估；空洞还削弱 pipeline.py:220 的 on/off-source 重叠防护与 calibration.py 的离源区间重叠检查。配置层（models.py:71-78）允许该组合，CLI 直接暴露 `--min-duration/--num-steps`。
- 修复：改用最短窗实际覆盖区间 `[tstart, tstart+duration]` 的并集（间隔 dmin/num_steps ≤ dmin 恒重叠，天然无缝），或直接用全部有效窗的并集；补非默认 min_duration 回归断言。

### 18.2 R76（Minor）：position 先验沉积到全网格最近点，地球边缘处最近点被掩蔽时整窗先验静默丢失

- [search.py:76-81](file:///home/xinxiang/research/jinwu/packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/search.py#L76-L81)：`index=argmin` 在全部 grid.size 个点上取（含 occulted），`weights[index]=1` 后 `mass=weights[visible].sum()`；若最近网格点恰落在地球掩蔽侧而 position 本身仍 visible（5° 网格 + 临边几何，差可达数度），mass=0 → `prior_loglr=NaN` → 该窗在 prior 排名中被记为 `no_prior_support`。`run_search_grid` L234 的 `location_visible` 检查在此之后，无法补救。地图分支（cKDTree bincount）无此问题因其保留可见质量再归一。
- 修复：在 visible 网格点内取最近点（masked argmin），与 L234 的可见性守卫语义一致；或在质量为 0 时记录专门原因。

### 18.3 R77（Minor）：make_search_windows 在 max_duration<min_duration 时静默返回空窗列表

- [search.py:28](file:///home/xinxiang/research/jinwu/packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/search.py#L28)：`round(np.log2(dmax/dmin))+1 ≤ 0` 时 `np.arange(≤0)` 为空 → 时长网格为空 → 返回 shape (0,2)。实测 dmax=0.064<dmin=0.128 时输出空数组而非报错。当前 models.py `__post_init__`（L71 hi<lo 拒绝）挡住了公开入口，属防御性缺口；一旦未来有绕过 config 校验的调用方，搜索结果为空且无失败记录。
- 修复：函数入口断言 `dmax>=dmin>0`，非法时 raise ValueError。

### 18.4 R78（Minor）：无源控制块按块中心排除 search±width/2，边缘块的背景拟合窗可探入搜索区间约 2 s

- [data.py:191](file:///home/xinxiang/research/jinwu/packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/data.py#L191)：use 条件为 `t < search_lo - width/2`（t 为块中心）。块半宽约 block*step/2≈2.048 s；被保留的最内侧块其最右 bin 的滑窗右沿 = center+width/2 可达 search_lo+~2.03 s。若搜索区间内确有源，这些控制块的 NaivePoisson 背景被源光子略抬高，source_free 诊断在边界块上被削弱（125 s 窗内 ≤2 s，约 1.6% 量级，仅影响诊断灵敏度而非搜索结果）。
- 修复：排除条件改用块整体：要求 `(t+块半宽)+width/2 ≤ search_lo` 且对称处理右端，即 t 阈值再外移一个块半宽。

### 18.5 已审无发现（子代理声明，主代理抽核）

- GDT `GbmTte.open` 对 times 和 GTI 同步减 TRIGTIME，data.py 同加 offset 一致；
- `prethresh=-inf` 时 `_normalize_likelihood` 必被调用，`_max_idx`/`marginal_llr` 不会为 None（私有属性契约在 pin 住的 _vendor 内成立）；
- `load_response(t,t)` 无除零；
- 框架在 NEEDS_REVIEW 即停，background 不会在数据缺失时崩溃；
- 死时间修正是 per-detector 活时，counts 与 background 同乘 exposure，比值无偏；
- 中点 bin 背景对 ≤8.192 s 窗 + 125 s 拟合窗是合理近似；
- dyadic 窗口边界 first/last、空 GTI、时长超过 GTI、cumsum 索引、先验归一化、PE veto 门控、poshist 采样间隔校验、模板 hash/shape 校验、JSON 契约 round-trip 均正确。

### 18.6 验证记录

- R75（S1）：主代理运行时实测——dmin=1.024/num_steps=8 时 eff 低估 2.06 倍；dmin=8.192/num_steps=8 时低估 19.73 倍。
- R76/R77/R78：子代理代码精读 + 实测，主代理抽核确认。

## 19. 合并前修复与复核（2026-09-25）

本节覆盖上述历史审查结论，按实际修复后的源码和本次验证记录更新。`examples/` 留在本机供教学，不进入提交；正式文档已去除对根目录未跟踪示例的依赖。

### 19.1 已修复的审查项

- **时间与标识**：R30 的 Fermi/GW 日历分桶先转 UTC；R32 根据 [GECAM 官方数据页](https://gecam.ihep.ac.cn/dailydatadownload.jhtml)和 [Insight-HXMT 时间标定论文](https://doi.org/10.3847/1538-4365/ac4250)把 MET=0 对齐相应 UTC 纪元，并补转换回归；R37 保留 GraceDB MS/TS/S ID 的小写后缀。
- **谱、响应与事件**：R31 使末尾未完成组的 QUALITY 折叠遵循最后非零标志；R47 对先验所需而观测 PHA 缺失的能道显式报错；R49 贝叶斯块按箱中心唯一归属以守恒计数；R50 裁剪 PHA 时同步裁剪 EBOUNDS；R51 回写 Event TIMEZERO；R52 按 F_CHAN 实际列号写 TLMINn；R53–55 修正 RMF 识别、相对时间光变读取及 RATE-only PHA 写出。R59 同族的两通道 factor=2 分组歧义也经回归发现并修正。
- **筛选、配置与诊断**：R56–58 让图像、光变筛选使用内存事件和大小写无关的 FITS 元数据；R61–69 修复上限配置别名与校准冲突、BATSurvey selection、GBM/GECAM detector 替换、配置校验、数组容器相等语义、谱分组与拟合能段；R22 的 XSPEC 结果保留 `conversion['counts']` 兼容别名并记录可恢复异常。R75 用真实最短搜索窗区间的并集计算有效时间。
- **真实流程额外发现**：WXT fit 阶段曾对结果对象执行字典成员检查导致 TypeError；现按对象类型判断。修复后同一工作区恢复执行并完成报告。

### 19.2 本次验证和界限

- `hea` Python 3.12 离线全量套件在 NUNIQ 修复后：1432 passed / 51 skipped / 6 deselected（新增通道、时间、Bayesian Blocks、OGIP、配置、GBM 有效时间及 GW 天图回归在内）；针对性天图和合并回归 23 passed。Sphinx HTML 构建成功（164 条现存 API/docstring 警告）；五个 Python wheel 构建成功。核心 wheel 首次构建发现旧 `build/lib` 残留已删除源码，清理构建缓存后重建，不再包含 `jinwu.response`。
- **WXT 真实观测**：本地 EP260809a / `06800001692_32`，`hea`+HEASoft/XSPEC；入口为本机 `examples/wxt/pointing_pipeline.py`，独立工作区 `/tmp/jinwu-wxt-demo-validation/wxt_20260925T150838425533Z`。先停在 `needs_review`，检查任务图像、源区/背景区及 ARM；出于软件流程验证目的记录批准，然后恢复至 `completed`。源区曝光覆盖 1.0，背景有效覆盖 0.83894，`alpha=0.188032`；存在背景曝光覆盖警告，故这些拟合结果仅证明流程可运行，不能直接用于科学结论。T90 源 PHA 的 BACKFILE/RESPFILE/ANCRFILE 均指向存在的独立工作区文件；时间轴的 FITS 与任务时钟差约 `2.6e-7 s`，报告和拟合图已生成。XSPEC 参数输出仍含 `FFFFFFFFF` 状态，教学时必须展示诊断，不据此宣称参数区间可靠。
- **GBM 真实数据局部复验**：2025-06-05 15z 连续观测，NaI n4 TTE + 当天 measured POSHIST，触发时刻 `2025-06-05T15:09:55 UTC`，独立日志 `/tmp/jinwu-gbm-real-validation.log`。背景准备覆盖 6408 个 64 ms 箱、367252 事件、409.02 s 有效曝光；35 个无源控制块的背景诊断 `passed=true`，实测姿态覆盖该窗口。缺少本地官方响应模板，未运行完整 subthreshold 候选搜索和 FAR 校准；R75 用非默认 1.024 s 窗的离线回归验证。未找到本地 GECAM/HXMT 事件产品，R32 仅由官方定义及 MET 往返回归支持。
- **未完成/后续**：主分支前两轮 CI 失败原因、修复及最终通过结果见 §19.3；R26–29、R38、R59–60、R70–74、R76–78 等其余性能、维护或边界建议仍作为后续项。历史证据中的“通过”仅适用于记录时的特定版本和范围。

### 19.3 主分支 CI 反馈与修复

首次推送 `a36c7e9` 后，[CI run 36157046191](https://github.com/xinxiangsun/JinWu/actions/runs/36157046191) 在 Python 3.11/3.12/3.13 均出现相同四个失败。原因是干净的 CI 环境不会由可选 BatAnalysis 注册 Astropy `swift` 格式，Swift/BAT survey 代码还在使用该外部别名，并在失败时静默退回 Unix 秒；这也让 PHA 时间窗筛选错位。现改用本库已注册的 `swiftmet`，并保留 `extract_time_interval(time_format="swift")` 的兼容入口。补了在刻意移除 `swift` 注册项时的回归检查。

第二次推送 `54b5dad` 后，[CI run 36215029793](https://github.com/xinxiangsun/JinWu/actions/runs/36215029793) 的 Swift 时间问题已全部消失，但 3.11/3.12/3.13 仍各有一个 NUNIQ 失败。首次修复只把合成 FITS 的 UNIQ 列改为 `int64`，生产 `load_skymap` 读入时又转成 `uint64`，在 CI 安装的 astropy-healpix 版本里其位扫描 ufunc 不支持无符号类型。现改为以 FITS `K` 的有符号 64 位数解码（先拒绝 `<4`），再把有效 UNIQ 存入现有无符号结果字段；新增负索引回归。此问题影响真实多分辨率 GW 天图读取，因此是生产修复，不只是测试兼容性。

推送 `fc56d70` 后，[CI run 36215364872](https://github.com/xinxiangsun/JinWu/actions/runs/36215364872) 的 Python 3.11、3.12、3.13 三个作业均完成且为 success；远端 `master` 提交号与本地一致。此次 CI 只覆盖仓库配置的自动检查，真实 WXT 和 GBM 验证范围仍以上述记录为准。

## 20. 0.2.1 版本与标签（2026-09-26）

原有 `v0.2.0` 标签指向 `2a91dd5`，且 `jinwu 0.2.0` 已发布到 PyPI，因此合并后的版本锁步更新为 `0.2.1`，六个包、GW 运行时版本、Rust 清单与 conda 配方一致。`v0.2.1` 是带注释的标签，解引用后指向 `fda5994`；[master CI 36221285873](https://github.com/xinxiangsun/JinWu/actions/runs/36221285873) 的 Python 3.11/3.12/3.13 均通过。本地 1432 项离线测试通过，五个 Python 包 wheel/sdist 构建、隔离 wheel 门禁 25 项测试及 Linux Rust wheel 导入通过。

[标签发布工作流 36221428343](https://github.com/xinxiangsun/JinWu/actions/runs/36221428343) 的 Python/Rust 构建与 wheel 门禁均通过，但 PyPI OIDC 身份只可上传既有 `jinwu` 项目；首次创建 `jinwu-ep` 时 PyPI 返回 400，故 `publish` 作业失败并跳过 GitHub Release。随后使用本机现有 PyPI 用户凭据上传**该次 CI 构建的**五个 Python 包 wheel/sdist 与两个 Rust 平台 wheel，并人工创建 [GitHub Release v0.2.1](https://github.com/xinxiangsun/JinWu/releases/tag/v0.2.1)（12 个 CI 产物）；PyPI 各文件 SHA256 与 CI 下载产物逐一一致。核心 conda 配方的 SHA256 取自 PyPI 已发布 sdist。`jinwurs` 尚未发布 sdist，其 conda 配方保持显式占位，未验收。下一次自动发布前须为每个 PyPI 项目配置 GitHub Trusted Publisher；不能把本次手工补传视为发布工作流已修复。


## 21. 2026-09-27 后续科学复核（上游一致性、边界修复与证据范围）

本节在 `08fd6a6` 的已合并和已发布基线上复核。第 1–18 节的“与上游逐行等价”“方法通过”等语句是当时的历史判断；以下以当前源码、参考资料和重新运行的证据为准。此次不更改 Scargle/HEASoft 的块适应度与先验、不改变 HEASoft `grppha` 分组方法，也不从局部软件测试推出科学探测或参数区间。

### 21.1 方法与实现的区分

- **R60（Bayesian Blocks）**：保留 [Scargle et al. 2013](https://arxiv.org/abs/1207.5578) Eq. 19 的泊松块适应度 `N ln(N/T)`、Eq. 21 的经验先验，以及本地 HEASoft 6.37 `burstcube/lib/bayesian_blocks.py` 使用逐箱有效曝光的变体。当前 [Astropy 实现](https://github.com/astropy/astropy/blob/main/astropy/stats/bayesian_blocks.py) 也使用同型动态规划，并明确警告事件模型的 `p0` 未必等于实际假警率。上游与旧 JinWu 的长度为 `n` 的回溯数组会在 `n=1` 时返回 `[0]`，并在部分多箱输入中丢失最优分界；这属于**最优解解码错误**，不是适应度/先验选择。现以 `[n]` 开始回溯，返回首 `0`、尾 `n` 的边界。对 240 组长度 1–8 的随机计数/曝光/先验，与穷举全部分段得到的最优目标值一致；跟踪回归包含单箱、零计数/零曝光和无效输入。历史 §2.2 的“25/25 与上游完全一致”仅说明当时有限样本的实现一致，不能证明原回溯正确。另将上游“Scargle 每 cell 至少 1 计数”的备注限定为逐事件表示，不再当作所有预分箱贝叶斯块实现的普遍限制。
- **R59（PHA 聚合）**：显式 `factor=1` 表示不聚合；`factor` 与 `min_counts` 同时传入及非正整数因子现在报错。未显式指定两者时，仍按原契约采用已有 OGIP GROUPING。这里修的是调用契约，不更改最小计数分组方法；[HEASoft grppha 帮助](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/grppha.html)说明 `group min` 写 GROUPING 标志而不改原始逐道计数。
- **R76–R78（GBM）**：点定位只有在实测航天器帧中可见时才投到最近的可见响应网格点，保留可见网格角距；被地球遮挡时可见先验质量为零。`max_duration<min_duration` 显式报错。控制块按整块边界及名义半窗宽排除搜索区，而不是只看块中心。此处保留 GDT `NaivePoisson(fast=True)`；[GDT 官方说明](https://astro-gdt.readthedocs.io/en/latest/core/background/unbinned.html)指出快速算法固定窗内事件数、实际窗宽会随计数率变化。因此这些仅是**名义无源控制块**，并非严格无源、独立留出样本，更不能用其残差诊断替代搜索试验数/FAR 校准。这个限制已写入代码和诊断标签，后续若需严格无源验收，应按实际背景窗支持或独立数据重新设计。
- **R38（GBM POSHIST）**：路径输入在所有新增读取位置均在 `with` 范围内完成所需计算，调用者传入的已打开对象仍由调用者管理。[GDT POSHIST 文档](https://fermi.gsfc.nasa.gov/ssc/data/analysis/gbm/gbm_data_tools/gdt-docs/notebooks/PositionHistory.html)把日姿态采样描述为约 1 秒；本地日文件实测中位间隔也是 1.0 秒。旧代码“50 ms 采样”的注释已纠正。此修复主要是资源管理，不改变覆盖几何公式。
- **R26（BAT 姿态）**：比较本地 `external_sources/BatAnalysis-main/batanalysis/attitude.py` 和 GDT `BatSao` 接口后，保留 RA/roll 的圆周解缠插值，但拒绝姿态覆盖外外推、以中点/端点替代源时刻及缺失 SAA 状态时默认好时段。轻量 SAO 行表后备读取仅接受显式 RA/Dec/roll；标准 GDT SAO 的 POSITION/QUATERNION 路径仍交给 `BatSao`。历史 §9.1 的“与 BatAnalysis 上游逐行一致”表述过强；上游方法可作接口比较，不能证明覆盖外替代或 70° 指向角启发式等价于实际地球遮挡/GTI。此次只把这些判断的范围说清楚，尚无真实 BAT 姿态产品完成端到端验收。[Swift Attitude and Alignment Guide](https://swift.gsfc.nasa.gov/analysis/suppl_uguide/att_align_20040422.pdf) 和 [Swift Archive Data Files](https://swift.gsfc.nasa.gov/archive/archiveguide1/node5.html) 是格式与观测文件的官方参照。
- **R43（BAT CALDB 缓存）**：本地目录递归读取或哈希失败时显式失败，避免以不完整文件列表生成可复用缓存指纹；URL/非目录值仍走通用依赖指纹。目录每阶段全树重哈希的性能问题仍在，未以一次结果缓存来交换潜在的校准文件变更漏检。

### 21.2 真实数据与软件验收

- `hea` / Python 3.12：在下述 R71/R73 谱准备补丁落盘**之前**，本地测试 `1456 passed, 51 skipped, 6 deselected`；GBM 定向回归 `18 passed`。这些测试涵盖代码契约和合成边界，不单独证明天体物理结果；不能将这些数字视为最新谱准备补丁的验收。
- EP/WXT `06800001692_32` 的原始光变来自 §19.2 的独立工作区。FITS `RATE × TIMEDEL` 在所查区间可恢复整数源计数；末 188 个 0.5 s 箱总计数 51，新回溯输出箱界 `[0,115,188]`、块计数 `[0,51]`，计数守恒。这验证真实输入上的软件行为；没有据此给出瞬变显著性或 T90 误差。
- GBM 2025-06-05 NaI n4 TTE `glg_tte_n4_250605_15z_v00.fit.gz` 与 measured POSHIST `glg_poshist_all_250605_v00.fit`，沿用 §19.2 的触发时刻与 `hea` 环境，在独立临时输出中重新运行背景准备：6408 个箱、367252 个事件，名义无源控制块 22 个（旧中心判据为 35），残差诊断 `passed=true`；同一 POSHIST 连续读取三次后文件描述符数保持 4→4。由于缺少相应的官方响应模板，本轮仍未运行完整候选搜索或 FAR 标定。`passed=true` 只属于背景残差检查。

### 21.3 仍需单独处理或验证

- R27–29 是复用和维护建议，本轮未改；R43 全树哈希性能、R74 GBM 8–900/8–1000 keV 双重约定仍需结合仪器响应和实际调用场景决定。R72 的已有输出 `FileExistsError` 是显式停止条件，保留以免覆盖既有谱；R71/R73 已有未验收的工作区补丁，状态见 §21.4。
- BAT 的插值和缺失标志处理已有合成回归（BAT attitude/survey 定向测试 59 passed）；项目、常见缓存和下载目录未找到真实 `.sat/.mkf/.sao` 产品，尚缺 BAT 完整分析流程，因此不能报告真实 BAT 可见性/GTI 的科学验收。GBM 点先验在地球边缘的重新投影是有限分辨率近似，尚未由完整响应搜索与注入恢复校准。


### 21.4 暂停交接记录（2026-09-27 01:20，新加坡时间）

用户要求在此停止，留给以后继续。三个子代理的写入已停止。本轮**没有提交、合并、推送、打标签或发布**；当前分支 `codex/new`，`HEAD`、本地 `master`、`origin/master` 均为 `08fd6a614ce6976e9001aa8c24bab37800e54f25`。本节以上改动仍在工作区；`git diff --check` 在暂停时无输出。不要把 §21.2 早于谱补丁的通过数当作本工作区最终门禁。

- 已落盘待复核：`packages/jinwu/src/jinwu/core/ops.py` 的 R59/R60；`jinwu-fermi` 的 R38/R76–R78 与名义无源限定；`jinwu-swift` 的 R26/R43；对应已跟踪回归及新建的 `packages/jinwu-swift/tests/test_bat_attitude.py`；本报告 §21。BAT 定向 59 passed，GBM 定向 18 passed，均在谱补丁前完成。BB 穷举、真实 WXT 光变局部检查、真实 GBM 背景/POSHIST 局部检查见 §21.1–21.2，证据范围不要扩大。
- **谱准备正在施工的最后状态**：`packages/jinwu/src/jinwu/core/spectrum_prep.py` 已把 R70 的旧附件复用改为同目标/同 inode 时保留、不同目标时替换；R71 从固定 HDU1 的静默回写改为只读核验 `SPECTRUM` 扩展里三个 HEASoft `CHKEY` 链接，失败置 `partial`；R73 接入既有 `check_response_compatibility(read_pha(...), read_rmf(...))`，通道不兼容置 `failed`，缺字段、警告或检查异常置 `partial`。这些新分支**尚未新增有意义的跟踪回归，也未用新生成的真实 WXT grouped PHA 或 PyXspec 复验**。既有 `test/test_spectrum_prep.py` 在顶层忽略目录下，其 fake grppha 只写文本；新检查可能改变这些旧测试预期。不能视 R70/R71/R73 已完成。
- 谱准备的科学边界：官方 [grppha 帮助](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/help/grppha.html)说明 `CHKEY` 和 `WRITE` 的语义；[OGIP PHA 规范](https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node6.html)定义 `BACKFILE/RESPFILE/ANCRFILE`。新 R73 checker 只检查能道范围/部分 `DETCHANS`，**不检查 ARF 与 RMF 的能量网格，也不证明 XSPEC 完整可拟合**。旧产品 `/tmp/jinwu-wxt-demo-validation/wxt_20260925T150838425533Z/fit/t90/prepared/unknown_obsid/WXT/t90/grouped_g1.pha` 曾在切到其所在目录后被 PyXspec 读入；这是旧产物证据，不能替代新补丁验收。
- 当前未跟踪的 `.commandcode/`、`.tmp_review_r10/`、`examples/` 是原有本地工作物，**不要 stage/清理/覆盖**；新建的 `packages/jinwu-swift/tests/test_bat_attitude.py` 属于本轮 BAT 回归，后续完成验收时应按确切路径单独纳入。顶层 `test/` 的忽略规则须保持不变；已跟踪的 `test/test_merge_regressions.py` 与 `test/test_gbm_subthreshold.py` 可以继续编辑。

继续时先检查 `git status --short` 和完整 diff，重点审查 `spectrum_prep.py` 的状态传播、同路径覆盖、旧假输出测试与实际 OGIP 头，再为 R70/R71/R73 补跟踪回归。用独立输出目录对 EP/WXT `06800001692_32` 的源/背景 PHA、RMF、ARF **重新**运行准备流程，核对 grouped PHA 的 `SPECTRUM` 链接、PHA↔RMF 能道和 RMF↔ARF 能量网格，并在 grouped PHA 目录用已初始化的 `hea`/PyXspec 实际加载；不覆盖 §19 旧产物。然后重跑受影响定向测试及本地 CI 等价套件，记录跳过项和诊断。BAT 真实姿态文件仍未找到；GBM 完整搜索/FAR 仍缺官方响应模板。只在这些检查完成且科学限制写清后再考虑提交与合并。

> **续审注记（2026-09-29）：** 本段是 2026-09-27 的交接待办清单，只读续审的结论见 §21.6（本次未修改任何代码）。本节历史文本保留，不删改。

### 21.5 AI 工作署名（记录时间 2026-09-27T01:47:16+08:00）

以下按 AI 和工作项分别登记。记录时间是本次整理时间，不代表历史代码落盘时刻。
会话记录可确认三名子代理的身份与工作窗口，但未提供运行模型字段；历史个人
修改或 review 的准确时刻无法追溯。先前派工要求 `gpt-6-luna/max`，这是
请求参数，不作为实际运行模型的证明。每条“未披露”均指模型标识和推理档位。

- **Codex 主代理｜科学复核与 GBM 修订**：harness＝Codex desktop app；
  模型/推理档位＝未披露；记录时间＝2026-09-27T01:47:16+08:00；历史工作时刻＝
  无法追溯到每项修改，主会话窗口见会话记录。范围＝§21 方法边界复核、
  `jinwu-fermi` R38/R76–R78 修订与局部验证、子代理结果复核。
  结果/证据＝§21.1–21.3 及 §21.2 的定向测试和真实 GBM 背景/POSHIST
  局部检查；未完成完整响应搜索或 FAR 标定。
- **/root/bat_attitude｜BAT 姿态与 CALDB 独立修改/review**：
  harness＝Codex 协作子代理（local）；模型/推理档位＝未披露；
  记录时间＝2026-09-27T01:47:16+08:00；历史个人修改时刻＝无法追溯。
  范围＝R26/R43、`jinwu-swift` 姿态/观测/survey 代码及回归。
  结果/证据＝子代理只读复核、§21.1–21.3、BAT 定向 59 passed；
  尚无真实 BAT 姿态产品端到端验收。
- **/root/bayesian_blocks｜贝叶斯块与 PHA 分组修改/review**：
  harness＝Codex 协作子代理（local）；模型/推理档位＝未披露；
  记录时间＝2026-09-27T01:47:16+08:00；历史个人修改时刻＝无法追溯。
  范围＝R59/R60、`jinwu/core/ops.py` 和对应回归。
  结果/证据＝§21.1 的 Scargle/HEASoft/Astropy 方法对照、240 组穷举
  最优值比对及 §21.2 的 WXT 真实光变局部计数守恒检查；不据此声称
  瞬变显著性或 T90 误差。
- **/root/spectral_prep｜谱准备补丁**：harness＝Codex 协作子代理
  （local）；模型/推理档位＝未披露；记录时间＝2026-09-27T01:47:16+08:00；
  历史个人修改时刻＝无法追溯。范围＝R70/R71/R73、
  `jinwu/core/spectrum_prep.py`。结果/证据＝§21.4 的工作区补丁；
  新分支没有跟踪回归及新 grouped PHA/PyXspec 实测，仍待验收。
- **Codex 主代理｜进度交接与署名规范整理**：harness＝Codex desktop
  app；模型/推理档位＝未披露；记录时间＝2026-09-27T01:47:16+08:00；范围＝本节、
  `AGENTS.md`、全局 `AGENTS.md` 与 Obsidian `Progress/`；
  结果/证据＝当前 `git status`、§21.4 与会话记录交叉核对；
  此项是文档整理，不构成谱补丁科学验收。

### 21.6 2026-09-29 codex/new 只读续审：R70/R71/R73 与其余改动复核

本轮按“只审查、不修改”执行：**未新增或修改任何生产代码与测试**，未提交、未
合并、未推送。工作区在 `codex/new`（`HEAD`、`master`、`origin/master` 均为
`08fd6a614ce6976e9001aa8c24bab37800e54f25`），15 个已跟踪文件保持原有未提交
修改；`.commandcode/`、`.tmp_review_r10/`、`examples/`、
`packages/jinwu-swift/tests/test_bat_attitude.py` 保持未跟踪、未改动。审查使用
只读命令、`/tmp/jinwu-spectrum-prep-review-20260929/` 独立输出与
`reviews/evidence/` 原始证据。

**as-found 基线门禁**（证据 32，审查结束复跑见证据 35，两者一致）：

- 命令：`python3 -m pytest test/ packages/jinwu-swift/tests -p no:cacheprovider -m 'not network and not heasoft and not real_data' -ra --strict-markers -q`
- 结果：`2 failed, 1454 passed, 51 skipped, 6 deselected`。
- 两个失败都在本地 ignored 文件 `test/test_spectrum_prep.py`：
  `test_prepare_spectra_groups_fxt_and_builds_joint`、
  `test_fit_catalogs_prepares_joint_and_uses_config_grouping`。原因：该文件的
  fake grppha 只写文本占位，而补丁新增的 R71 头链接核验要解析 FITS、R73 校验
  要经 `read_pha`/`read_rmf` 读回，`PreparedSpectrum.status` 从 `ready` 变为
  `partial/failed`，`prepared.ready` 断言失败。该文件不在 CI checkout
  （`.gitignore:70` 的 `test/*`），所以这是本地测试预期未随补丁更新的问题，
  不是 CI 失败。

**R70（附件 staging）审查**：

- 行为核对（`_stage_ancillary_file`）：目标已存在时，`resolve()` 相同或
  `samefile(source)` 则保留；否则 `unlink` 后重建 symlink；损坏/成环旧链接
  命中异常分支被替换。满足“陈旧链接必须被替换、同目标不重复创建”的意图。
- 发现：`overwrite` 形参在新实现中已不再影响行为——staging 始终刷新不匹配
  目标，参数实际是死参数。属代码整洁性问题，不是验收阻塞；本轮未修改，建议
  后续补注释或移除形参并同步调用方。

**R71（SPECTRUM 头链接核验）审查**：

- 行为核对（`_validate_grouped_header_links`）：只读打开 grouped PHA，取
  `SPECTRUM` 扩展头，逐个核验 `BACKFILE/RESPFILE/ANCRFILE` 与 staged basename
  一致，并检查 staged 目标 `is_file()`；缺扩展、名字不符、目标缺失分别抛
  `ValueError`/`ValueError`/`FileNotFoundError`，由调用方置 `partial` 并写入
  诊断。相比旧的 HDU1 盲写 + `except Exception` 吞错，语义正确、无静默放行
  路径（`fits.open` 失败也会进入 except）。
- 发现：改动没有随行跟踪回归（顶层 `test/` 忽略规则下新用例未落已跟踪文件），
  CI 无法防回归；见“发现与建议”第 3 条。

**R73（响应兼容性）审查**：

- 行为核对：分组后调用 `check_response_compatibility(read_pha(grouped),
  read_rmf(staged RESPFILE))`。PHA 通道超出 `[TLMIN, TLMIN+DETCHANS-1]`
  置 ERROR→`failed`；`DETCHANS` 不一致置 WARN、缺 `TLMIN/DETCHANS` 置
  `COMPAT_NOT_CHECKED`→`partial`；检查异常也置 `partial`。状态优先级（failed
  覆盖 partial）正确。
- **覆盖缺口（本轮主要发现）**：该检查器只比较 PHA 通道范围与 `DETCHANS`，
  **不读 ARF、不比较 RMF↔ARF 能量网格**。R73 的原表述是
  “PHA↔RMF/ARF 兼容性检查”，因此当前补丁只完成 PHA↔RMF 部分；`jinwu-ep`
  `_validate_ogip_bundle`（pipeline.py:1090）另查 PHA 头链接、PHA⊆RMF 通道、
  ARF 覆盖拟合能段，但同样不查 RMF↔ARF 网格。
- 真实数据人工核查（外部脚本，不属补丁行为）：EP/WXT `06800001692_32` 的
  RMF 与 ARF 各 1980 个能量 bin，`ENERG_LO/ENERG_HI` 逐位一致（max diff 0 keV，
  `rtol=1e-6, atol=1e-6` 通过），因此这份数据不受该缺口影响；其他仪器/产品
  无代码保证。本轮按要求未补代码，列为验收缺口。

**真实 EP/WXT 审查证据（as-found 代码）**：

- `prepare_spectra`：独立输出 `/tmp/jinwu-spectrum-prep-review-20260929/out`，
  输入为 `test/EP260809adata/EP260809a/06800001692_32/` 的 s1/bkg PHA、ARF、
  RMF；返回 `catalog status: ready`、`spectrum status=ready`、`group_min=1`，
  无诊断（证据 33）。
- grouped PHA 三个链接为 staged basename 且 symlink 解析到输入文件；PHA 通道
  `0..1023`、`DETCHANS=1024`；RMF `tlmin=0/det_chans=1024`，通道在范围内；
  `check_response_compatibility` 通过。
- PyXspec 在 grouped 目录加载成功：1 spectrum、exposure 2880 s、RMF/ARF/背景
  均解析，noticed channels `1-115`（证据 34）。仅证明加载与 OGIP 头语义。
- 完整 WXT pointing pipeline（独立输出，`auto_approve_regions=True`）
  `FINAL STATUS: completed`；fit 的 bb000/pipeline/t100/t90 四个标签的
  prepared grouped PHA 链接全部有效（证据 36）。不给出瞬变显著性、T90 误差
  或参数区间等科学结论。

**其余改动复核（只读）**：

- **R59/R60**：独立随机种子 20260929，对 n=1–8 共 400 组计数/曝光/先验与
  穷举分段最优值逐一比较，400/400 一致；负计数、NaN、Inf、正计数零曝光、
  非一维输入全部报错；`rebin_pha` 的 `factor=1` 返回原对象、`bool`/非正整数
  拒绝、`factor` 与 `min_counts` 互斥；回溯索引语义（`last[R]` 对应前 `R+1`
  箱）逐行核对无缺陷（证据 37）。用 `git show HEAD` 的旧实现作对照：新增
  穷举用例中 4/6 在旧回溯上次优、新实现全部达到最优；不同随机种子（7）的
  400 组里旧实现 34/400 次优、新实现 0/400，证明跟踪回归能判别旧缺陷
  （证据 40）。`factor=1` 只出现在测试中，无生产调用方。
- **R38/R76–R78（GBM）**：调用方 `run_search_grid` 确实把实测
  `location_visible` 结果传入 `position_visible`；控制块按整块边界排除，诊断
  改为 `nominal_source_free_control_blocks/independent_holdout=False`。真实
  2025-06-05 n4 TTE + POSHIST：`read_gbm_geometry` 路径连续读取 3 次
  fd 4→4，调用者持有的对象在调用后仍可用（86520 states）；背景准备
  `passed=True`。默认配置 16564 bins / 954725 events / 1057.25 s；与 §21.2
  历史数字差异来自窗口/context 配置（事件率同为约 0.9 kHz），不是回归
  （证据 38）。GBM 定向 53 passed + 1 skipped。补充资源所有权验证
  （证据 39）：`GBMObservation` 的 `with` 退出后 frame/states/
  `location_visible` 仍可用；`check_gbm_coverage` 对象调用后 fd 回到基线且
  对象仍可用，路径/对象状态一致（均 full、cadence 1.0 s）；`MeasuredHistory`
  构造后 fd 4→4（86520 samples）；`read_gbm_geometry` 路径/对象结果一致。
- **R26/R43（Swift BAT）**：覆盖外拒绝、端点插值、SAO 显式 RA/Dec/roll、
  缺失 SAA 状态 fail-closed、欠覆盖时不允许端点帧替代、CALDB 树遍历/哈希失败
  报 `RuntimeError` 等与代码一致；`packages/jinwu-swift/tests` 70 passed +
  1 skipped（含未跟踪 `test_bat_attitude.py`，审查认为覆盖合格、可纳入提交）。
  本地未找到真实 `.sao/.sat/.mkf`；本 harness shell 需
  `source $CONDA_PREFIX/bin/heainit.sh` 才有 HEASoft/PyXspec，且无 `heasoftpy`，
  故真实 HEASoft/姿态验收测试未运行。另核对已安装 GDT 源码：
  `BatSao.get_bat_pointing()` 返回主头固定的 `RA_PNT/DEC_PNT`，可见性用
  `srctime` 处的 `_one_frame`；属委托给 GDT 的既有分工，不是本次回归。
- **文档**：`AGENTS.md` 的进度/署名规则与全局规则一致。

**发现与建议（按优先级，本轮均未实施修复）**：

1. **R73 未覆盖 ARF 与 RMF↔ARF 能量网格**：建议后续补自动检查与最小跟踪
   回归；真实 WXT 数据网格一致，不影响本数据集。
2. **本地 `test/test_spectrum_prep.py` 与补丁新行为不一致**：虽不进 CI，但会
   让本地门禁保持 2 failed；建议同步 fixture 或把两处断言改为期望 `partial`。
3. **改动缺少 CI 可见的跟踪回归**：顶层 `test/` 忽略规则下，R71/R73 的新分支
   没有落在已跟踪测试文件里；建议在 `test/test_merge_regressions.py` 补少量
   聚焦用例（按用户要求本轮未加）。
4. **R70 的 `overwrite` 死参数**：建议注释或移除。
5. BAT 真实姿态全流程、GBM 完整搜索/FAR 仍受数据/模板条件限制，见 §21.3。

**证据**：`reviews/evidence/32-baseline-codex-new-postpatch.txt`（as-found 基线）、
`33-spectrum-prep-realdata.txt`（真实 WXT 准备与网格人工核查）、
`34-xspec-load-spectrum-prep.txt`（PyXspec 加载）、
`35-review-only-state.txt`（审查结束时工作区与门禁一致性）、
`36-wxt-pipeline-e2e.txt`（完整 WXT pipeline）、
`37-ops-r59r60-review.txt`（R59/R60 穷举复核）、
`38-gbm-realdata-review.txt`（GBM 真实数据复核）、
`39-gbm-resource-ownership.txt`（GBM 资源所有权）、
`40-regression-discrimination-review.txt`（旧/新实现判别性对照）。

**AI 署名**：

- **pi 主代理｜codex/new 只读续审（R70/R71/R73 等）**：harness＝pi coding
  agent harness；模型/推理档位＝运行环境披露 `PI_MODEL=deepseek-flash`、
  `PI_REASONING_LEVEL=max`；记录时间＝2026-09-29T19:32:08+08:00；范围＝两轮只读
  审查：基线冻结、R70/R71/R73 行为与覆盖缺口、真实 WXT/PyXspec/完整 pipeline
  验证、R59/R60 穷举与旧实现判别性对照、GBM 资源所有权与真实数据、BAT 抽查；
  结果＝本节“发现与建议”；**未修改任何代码或测试**。

### 21.7 2026-09-29T19:41:01+08:00 物理与方法正确性复核（只读）

本节对 §21.6 所列改动做物理与方法层面的核对，不修改代码。对象包括 Scargle
贝叶斯块的适应度/先验/解码、GBM 子阈值搜索的曝光/背景/响应/窗口/边际化口径、
Swift BAT 姿态插值与可见性判据、谱准备的分组与响应配对。原始数值见
`reviews/evidence/40–43-*.txt`。

**1. 曝光加权贝叶斯块（R60/R59）**

- 适应度 `N ln(N/T)`（T=Σ逐箱曝光）与 HEASoft 6.37 `burstcube/lib/bayesian_blocks.py`
  逐行同式；JinWu 追加 `T_k>0` 守卫：正计数零曝光箱显式报错，上游在该情况下会
  得到无穷适应度，因此这是更安全的行为而非方法改变。
- 先验 `ncp_prior = 4 − ln(73.53 p0 N^−0.478)` 与 Scargle (2013) Eq. 21、上游及
  Astropy 一致；`p0` 对事件模型并不等于实际假警率，文档已按 Astropy 官方说明
  改写。
- 解码修复的必要性获独立证实：Astropy 8.0.1 在 2 箱强制分段（`ncp_prior=0`）时
  返回 1 箱（丢失首个内部边界）；旧 HEAD 实现在 400 组随机样本中 34 组次优；
  新实现 0/400 次优，等曝光 200/200 随机样本达到穷举最优（证据 40、41）。
- 变曝光物理动机：模拟 GTI 间隙（活时间 1 s/0.2 s 混合、真实变点在第 120 箱）时，
  曝光加权恢复率 258/300，等权（忽略活时间）239/300，支持逐箱曝光加权的改进
  （证据 41）。
- R59：`factor=1` 显式“不聚合”、与既有 OGIP GROUPING 的契约清晰；`factor=1`
  只出现在测试中，无生产调用方受影响。

**2. GBM 子阈值搜索（R38/R76–R78）**

- 曝光与死时间：逐箱 `GTI 几何重叠 − N_nonovf·edt − N_ovf·odt` 与 GDT 官方
  `EventList.get_exposure` 在真实 TTE 的 4 个区间（含 GTI 边缘）逐位一致
  （rel = 0），两种写法代数等价（证据 43）。死时间物理处理正确。
- 背景：`NaivePoisson(fast=True)` 的 rate 是“计数/实际 elapsed 窗宽”；代码以
  `live_fraction = 窗内活时间/请求窗宽` 换算为“计数/活秒”，期望背景计数 =
  rate_live × 活曝光，响应模板乘 `exp/duration` 的活时间占比——量纲自洽。
  `fast=True` 实际窗宽随计数率变化是上游已知近似，因此控制块正确定名为“名义
  无源”、`independent_holdout=False`。
- 窗口网格：上游 pinned 默认（min_dur 0.064 / max_dur 8.192 / min_step 0.064 /
  num_steps 8，step=max(min_step, dur/num_steps)）下，JinWu 的 min_step 对齐步长
  与之等价，且要求整窗落在 GTI∩搜索区间内（更保守、无双计）。
- 模板与空间先验：上游 `Likelihood.marginal_llr` 内部均匀模板先验（−ln nspec）、
  外部 sky prior 归一化；JinWu `logsumexp(llr + ln w) − ln(N_TEMPLATES)` 与之一致
  （`response.shape[0] = 3 = len(TEMPLATES)`）。点先验在实测帧不可见时质量为零、
  可见时投到最近可见响应格，是对网格化先验的显式近似（文档已写）。
- 控制块诊断：标准化残差 z、均值 95% CI、精确二项覆盖率区间（对 68%/95% 名义
  覆盖）、趋势斜率启发式；均为诊断门，不冒充 FAR。有效时间取最短窗（且
  atmospheric_response 通过）的并集，FAR 用 Poisson 精确区间，
  `FAP=1−exp(−FAR·T)` 与“聚簇事件平稳 Poisson”假设显式标注，
  `science_status=uncalibrated_candidates` 诚实。

**3. Swift BAT（R26/R43）**

- 姿态插值：RA/roll 先按 360° 解缠绕再线性插值、Dec 线性、端点含入、覆盖外拒绝。
  对 Swift ~1 s 采样、秒级指向变化很小的实际工况，线性 RA/Dec 插值与球面插值的
  差异可忽略；这也是 BatAnalysis 的同口径做法（标准 SAO 的 SLERP 由 GDT 负责）。
- 可见性：`Attitude` 回退路径的 70° 指向偏置筛不是地球遮挡几何，只可用于粗筛；
  代码与 docstring 已如实标注，`BatSao` 路径使用 GDT 帧几何。该回退是残余方法
  限制而非本分支引入。
- `check_gti` 在 SAA 缺失/异常时 fail-closed（不把“未知”当“好时段”），保守正确。
- 委托 GDT 的 SAO 路径：`get_bat_pointing()` 返回主头固定 `RA_PNT/DEC_PNT`，而
  可见性用 `srctime` 处的帧——分工明确，但“指向”属性不是时间插值结果。

**4. 谱准备与响应配对**

- 分组语义：WXT g=1、FXT g=3 只改变分箱数、不改变计数，适配 cstat/wstat；
  拟合能段 WXT 0.5–4.0、FXT 0.3–10.0 keV，ARF 覆盖拟合能段由 pipeline
  `_validate_ogip_bundle` 检查。
- R73 的 PHA↔RMF 通道检查与 heasp 通道语义一致，但 RMF↔ARF 网格仍无检查。
  **定量修正**：真实 FXT 产品（`06800001696_FXTA/B_01` 等 5 对，1024 bin、
  0.1–12 keV）的 RMF/ARF 网格存在相对 ≤5.4e-6（绝对 ≤5.5e-5 keV）的差异；
  PyXspec 实测接受这些响应对并成功构建响应。因此若未来补检采用严格
  `rtol/atol=1e-6`，会把 XSPEC 接受的真实产品判为失败；判据应使用与产品生成
  精度相容的容差（例如 rtol≈1e-5）或“长度/覆盖范围 + 明确容差”的组合，并以
  XSPEC 实际接受性为准（证据 42）。当前检查器对这些产品静默，操作上不与 XSPEC
  冲突，但仍属未验证边界。

**5. 结论**

本分支改动引用的物理公式与官方/上游实现一致或更严格，未发现会系统性改变科学
结论的物理错误。主要方法缺口仍是 R73 的 ARF/RMF 网格检查；其判据必须按真实
产品公差设计。BAT 的 70° 可见性回退与 GBM `fast=True` 名义无源控制块是既有
近似，报告口径已正确限定。

**6. 补充核查（第三轮，证据 44–45）**

- GBM 姿态/响应链：`MeasuredHistory`/`GBMObservation` 的 `SpacecraftFrame.at`
  用 GDT `Slerp`（四元数球面插值）+ 位置/速度线性插值；`valid_interval` 要求
  区间内 good、非 SAA 且采样间隔 ≤5 s。响应共享的角位移阈值 `delta=0.1°`；
  大气分量仅在 |geo_zen−130°|≤5° 时启用，域外退化为 direct-only 并置
  `in_rock=0`，经 `quality_passed` 传播为 needs_review。JinWu 的
  `load_atmospheric_response` 重写为循环方位角插值（`% 2π`、取最近两侧、
  权重和为 1），修正上游按非环绕角 `argsort` 在 0/2π 处可能产生负权重的实现；
  direct 模板负值截断为 0。
- 定位与稳定性：统计定位对模板做均匀边际化（`logsumexp − ln nspec`）、只在
  可见格归一化；`response_stable` 比较候选位置在窗首/窗尾与参考响应向量的
  分数变化，阈值 `response_tolerance=1%`，失败进入 needs_review，方法保守。
- 通道分组：`CHANNEL_EDGES` NaI 8 组（0,8,20,33,51,85,106,127,128）、BGO
  （0,8,21,40,65,90,112,124,128）；搜索道 NaI 组 1–6（去最低组与 overflow）、
  BGO 全 8 组，与 vendored `channel_mask` 语义一致。官方 GTS 的 YAML 配置不在
  本地（PyPI `gbm` 0.0.1 为占位包、USRA-STI 仓库无 YAML），未逐字节核对。
- 分组守恒（真实数据）：WXT 116→116 counts、FXT 32→32 counts，EXPOSURE/
  BACKSCAL/AREASCAL 不变（FXT 曝光因 grppha 关键字精度差 2.3e-4 s，相对
  3.8e-8）；FXT 在 RMF/ARF 网格相对差 5.4e-6 下 `prepare_spectra` 为 ready
  且 PyXspec 成功加载（证据 44）。
- `BayesianBlocksBinner(use_exposure=True)`：逐箱 `bin_exposure` 优先，块内
  曝光/计数/方差求和；rate→counts 用有效曝光。次要注意：
  `_effective_exposure_from_lc` 把零曝光箱回退为箱宽，但 `use_exposure` 路径对
  BB 用原始 `bin_exposure`，仅 rate 输入重建 counts 时受影响（零曝光且非零
  rate 本身不自洽）（证据 45）。

**AI 署名**：

- **pi 主代理｜codex/new 物理与方法正确性复核（只读）**：harness＝pi coding
  agent harness；模型/推理档位＝运行环境披露 `PI_MODEL=deepseek-flash`、
  `PI_REASONING_LEVEL=max`；记录时间＝2026-09-29T19:41:01+08:00；范围＝Scargle BB 适应度/先验/
  解码、GBM 曝光与背景/窗口/边际化、Swift 姿态插值与可见性、谱分组与 RMF/ARF
  网格；结果/证据＝本节与 `reviews/evidence/40–45-*.txt`；**未修改任何代码或
  测试**。

### 21.8 2026-09-29T20:35+08:00 独立复验续审（只读，覆盖完整性审计）

本节由另一名独立审查者执行，目标是"审查所有尚未被审查的变更"的收尾：先做
逐 hunk 覆盖审计，再对 §21.6–21.7 的关键结论做独立复验。**未修改任何生产代码
或测试**；工作区在 `codex/new`，无新提交。审查前确认自 §21.6 复核
（2026-09-29 19:59+08:00）以来无 `.py` 文件变更，工作区与 §21.6 审查时
as-found 状态一致。

**覆盖完整性审计（逐 hunk 对照 §21.6/21.7 与证据 32–45）**：15 个已跟踪修改
文件的全部 diff hunk 均可映射到已记录的审查项，无未覆盖变更——

- `spectrum_prep.py` 全部 hunk → R70/R71/R73（证据 33–35）；
- `ops.py` 全部 hunk → R59/R60 与文档措辞（证据 37、40、41）；
- `gbm_{observation,pipeline,poshist}.py`、`subthreshold/{data,search}.py` 全部
  hunk → R38/R76–R78（证据 38、39、43）；
- `bat/{attitude,bat_observation,survey}.py`、`test_bat_survey.py` 全部 hunk →
  R26/R43（§21.6 Swift 抽查、§21.7 第 3 节）；
- `test_gbm_subthreshold.py`、`test_merge_regressions.py` 全部新增用例 →
  R76–R78、R59/R60、R70 回归（证据 37、40 含判别性对照）；
- `AGENTS.md`、`reviews/codex-beta-vs-master.md` 为文档；未跟踪
  `test_bat_attitude.py` 本轮通读（见下）；`.commandcode/`、`.tmp_review_r10/`、
  `examples/`、`.hermes/` 为本地工作物/harness 状态，不属审查对象（`.hermes/`
  仅含本续审的计划草稿）。

**R70/R71/R73 状态传播矩阵（本轮补全 §21.6 未逐条列出的分支）**：通读
`_prepared_from_bundle`（约 194–302 行）与 `prepare_spectra` 聚合（L360-362），
8 种组合与意图全部一致：bundle 未就绪/缺件 → `partial` 早退；grppha 失败 →
`failed`；成功+链接核验通过+兼容 ok → `ready`；链接核验失败 → `partial`；
兼容 ERROR（`INCOMPATIBLE_CHANNELS`）→ `failed`；兼容 WARN
（`DETCHANS_MISMATCH`/`COMPAT_CHECK_FAILED`）或 INFO（`COMPAT_NOT_CHECKED`）→
`partial`；兼容检查外层异常 → `partial`（未 `failed` 时）；链接核验
`partial` 后兼容 ERROR 仍升级为 `failed`（failed 优先）。诊断码与
`ogip.py::check_response_compatibility` 实际 code 一一对应。R78 的
`block_start/block_stop` 索引核对为安全：`n = len(centers) // block` 对尾块
截断一致，`edges[n*block]` 恒有效，无越界。

**门禁与测试独立复验（`hea`，Python 3.12.14）**：

- 跟踪回归：`pytest test/test_merge_regressions.py test/test_gbm_subthreshold.py`
  → **38 passed**；`pytest packages/jinwu-swift/tests` → **70 passed,
  1 skipped**（含未跟踪 `test_bat_attitude.py`）。
- 全量 CI 等价门禁（证据 46，`reviews/evidence/46-offline-gate-reverification.txt`）：
  `2 failed, 1454 passed, 51 skipped, 6 deselected`，与 §21.6 as-found 基线
  （证据 32/35）**完全一致**；两个失败仍为本地 ignored
  `test/test_spectrum_prep.py` 的 fake grppha 用例（§21.6 发现 2，未修复，
  按只读纪律保留）。
- 真实 EP/WXT `06800001692_32` 复验（独立目录
  `/tmp/jinwu-review-reverify-20260929/out`，证据 47）：
  - 未 source `heainit.sh` 时 grppha 不在 PATH，`prepare_spectra` 正确返回
    `status=failed` 且诊断 `grppha executable not found`——意外地独立演示了
    grppha 失败分支的状态传播正确性；
  - HEASoft 初始化后 `catalog status: ready`、`spectrum: ready`、无诊断；
  - grouped PHA `SPECTRUM` 头三链接为 staged basename 且解析到输入文件，
    PHA 通道 0–1023、`DETCHANS=1024`；RMF/ARF 各 1980 bin，
    `ENERG_LO/ENERG_HI` 逐位一致（max diff 0.0）；
  - PyXspec 在 grouped 目录加载成功：1 spectrum、exposure 2880 s、
    背景/RMF/ARF 全部解析、noticed channels 1–115（与证据 33/34 一致）。
- `test_bat_attitude.py` 通读结论：断言覆盖 `pointing_at/roll_at` 覆盖外拒绝
  （含参数化边界）、端点含入插值、`_parse_sao` 三种列路径与两处显式报错、
  Quantity 单位换算与不可换算拒绝、标志查询的 `None` 保留与覆盖外拒绝、
  `BATObservation` 不再用 mid-point/端点替代（异常传播、`pointing` 属性
  不被设置）、SAO 端点替代拒绝、SAA 未知/异常 fail-closed。与 R26 修改意图
  一一对应，覆盖合格，支持纳入提交（维持 §21.6 结论）。

**新发现**：无新增生产代码缺陷。§21.6 的四项发现（R73 不查 RMF↔ARF 网格、
R71/R73 缺 CI 可见跟踪回归、本地 `test_spectrum_prep.py` fixture 过期、
R70 `overwrite` 死参数）经独立复验全部确认仍然开放，状态不变；是否按
计划文档（`.hermes/plans/2026-09-29_185626-codex-new-review-continuation.md`
阶段 1，Option A/B）修复仍待用户决定。

**证据**：`reviews/evidence/46-offline-gate-reverification.txt`、
`reviews/evidence/47-wxt-spectrum-prep-reverify.txt`。

**AI 署名**：

- **Qoder 主代理｜codex/new 覆盖完整性审计与独立复验（只读）**：harness＝
  Qoder agent harness；模型/推理档位＝未披露（运行环境未报告模型标识与
  档位）；记录时间＝2026-09-29T20:35:00+08:00；范围＝15 个已跟踪文件 +
  未跟踪 `test_bat_attitude.py` 的逐 hunk 覆盖审计、R70/R71/R73 状态传播
  矩阵与 R78 索引安全核对、跟踪测试/Swift 套件/全量门禁/真实 WXT
  `prepare_spectra`/PyXspec 加载独立复验；结果＝无未覆盖变更、无新增缺陷、
  §21.6 四项发现确认开放；证据＝本节及 `reviews/evidence/46–47-*.txt`；
  **未修改任何代码或测试，未提交/合并/推送**。

### 21.9 2026-09-29T21:20+08:00 codex/beta 变更集未审部分深审（只读，R79–R102）

用户指出 codex/beta 最后一次提交（原基线 `0e15c32..269587a`，9 提交、152 文件、
+26511/−8681）当时仍有文件从未深读。本轮按只读纪律补齐：5 个独立子代理并行
深读，主代理对全部 Critical 与代表性 Warning 做独立复核（8/8 证实，含 1 项
运行时复现，见证据 48）。**未修改任何生产代码或测试**。历轮已审段落不重复审查。

**确认的覆盖缺口（本轮首次全读）**：`swift/grb/pipeline.py`（2079 行，旧轮仅
grep 级）、`core/bxa_fit.py`（1175 行，仅两处行级提及）、`gbm/pipeline.py`
其余约 2500 行、`bat/survey.py` 其余约 2000 行、`timescale.py`/`ogip.py`
非 txx/非 compatibility 部分。

#### 发现清单

**Critical**

- **R79** `grb/pipeline.py:1578`（`_stage_bblocks`）：alpha 回退分支
  `t0_met = _event_trigger_met(bkg)` 覆盖外层时间原点（L1541 已用
  `trigger_met or _event_trigger_met(src)` 建立并守卫）。后果：(a) bkg 头缺
  `TRIGTIME` 时 t0_met=None，后续 `float(start - t0_met)` 抛未捕获
  TypeError 使整轮崩溃；(b) 头 TRIGTIME 与目录值不一致时同 mode 其余段
  静默换原点，且下游 spectra 阶段（L1676/L1682）按目录原点换算——同一
  数据两套原点，时间窗静默错位。修复：回退分支使用局部变量，不覆写
  `t0_met`；原点未知时显式失败。
- **R80** `bxa_fit.py:585–594`：`fit_statistic`/`fit_dof` 在
  `create_flux_chain` 之后读取；BXA 4.5 的 flux chain 逐后验样本
  `set_parameters` 且不恢复，PyXspec `Fit.statistic` 惰性按当前参数重算——
  默认 `calculate_flux_chain=True` 时报告的统计量是任意后验样本的值而非
  L568 `set_best_fit` 的 ML 点，同一污染状态流入 `_xspec_show_text`（L624）、
  绘图（L636）与 xcm 导出（L654）。`best_fit`（L576）取值正确。修复：在
  flux chain 之前读取统计量，或读取前重调 `solver.set_best_fit()`。

**Warning**

- **R81** `grb/pipeline.py:464`：`Time(record.trigger_utc, format="iso")`
  不接受 `T` 分隔 ISO 串（运行时复现：`Time('2005-09-04T01:42:12.00',
  format='iso')` → ValueError）；目录字段 `BAT_Trig_time_UTC`/
  `SDC_TriggerTime` 格式不受本仓库控制，异常在 `_stage_prompt` 无处理，
  整轮崩溃而非进入 NEEDS_REVIEW。修复：归一化分隔符或 isot 回退。
- **R82** `grb/pipeline.py:1231–1243`：`request.complete` 属性缺失时
  `AttributeError` 被宽 except 吞为"XRT request failed or is pending"，
  掩盖 swifttools 契约漂移。
- **R83** `grb/pipeline.py:1206–1212, 1188`：提交抛出明确异常时状态仍写为
  `"submitting"`，被 L1188 的防重复提交永久锁死，须手工改状态文件恢复；
  "结果不明确"理由不适用于确定性失败。
- **R84** `gbm/pipeline.py:829, 1827, 1854, 3267, 3467`（GbmTte）与
  `:1990`（GbmRsp2）：6 处 `.open()` 无 with/close——R38 同族新位置；
  `_tte_covering`（L3464-3467）还会逐个整载小时级 TTE 直至命中覆盖文件。
- **R85** `gbm/pipeline.py:1645–1659, 1705–1709`：中断下载保留的截断文件，
  只要 `_is_valid_fits`（仅查 ≥2880 字节 + 主头）通过即被永久复用，不再
  重下；截断的数据 HDU 会以不透明 OSError 或（惰性读取时）静默缺事件
  的方式进入下游。
- **R86** `gbm/pipeline.py:3028–3045`：`_select_covering_poshist` 对首个
  `status != "data_missing"` 的候选即返回（含 `none`/`partial`），跨 UTC
  日界的区间不会评估次日文件——docstring 声称"实际覆盖"，行为与文档相反，
  第二天可见段被静默丢弃。
- **R87** `gbm/pipeline.py:3134`：下载阶段完备性仅查 `tte[det]/cspec[det]`
  非空，不校验 TTE 并集 GTI 是否覆盖区间；与 R85 叠加时，缺一小时 TTE 的
  缓存照样通过并产出部分曝光的谱，无 gate 无旗标。
- **R88** `gbm/pipeline.py:2966–2972`：preflight perl 探测加 `timeout=30`
  但未捕获 `subprocess.TimeoutExpired`，挂起的 loader 会使整个运行 FAILED
  而非记 `perl_module=None` 降级。
- **R89** `survey.py:1681–1719`：`_mosaic_source_measurements` 的
  `if not points: continue` 使源不在 `sources_tot.cat` 时回落到树内下一个
  `.cat`（成员指向或 tmp 目录），`detection_basis` 仍标
  `mosaic_source_catalog_snr`——与其 docstring "缺目录行显式 not_detected、
  不借用成员指向 SNR" 直接矛盾，mosaic 检测结论可被错误标注。
- **R90** `survey.py:1767–1793`：`_timeunit_scale` 对未知 token 静默取 1.0；
  OGIP 合法的 `h/hr/ks` 被当秒，MET 起点/TIMEZERO 偏 3600×/1000×，窗口
  交叠、GTI 门控、PHA-率匹配全部错位且无诊断。
- **R91** `survey.py:1881–1889, 5787`：GTI 键为文件父目录名（`point_<id>`），
  查找用 CAT 行原始 `pointing_id/obsid`；拼写差异（`123` vs `point_123`，
  正是 L2515-2523 `_scope_identifier_matches` 显式归一化的别名情形）时查找
  落空、`gti_overlap<=0` 静默剔除该指向——光变静默丢曝光。
- **R92** `timescale.py:714–718`：`txx` 固定模式分段边
  `np.arange(t100_start, t100_stop + binsize, binsize)` 末边可超 t100_stop
  且只补不剪（自适应模式 L703 有 `np.clip`），`(t100_stop, 末边]` 内源事件
  泄入分位累计，抬高净计数总数并移动 95% 穿越。
- **R93** `timescale.py:486–510`：ignored-parameter 告警清单不含
  `src_dist/bkg_dist/window_mode/density_quantile/weak_peak_bins/
  weak_peak_weight/small_bin_threshold`——传 `src_dist='gaussian'` 等旧参数
  被静默忽略。
- **R94** `timescale.py:533–538, 567`：`_evt_bounds` 把 GTI 归约为
  `(min(START), max(STOP))`，数据间隙内的活时间被计入 T90（时长偏大）；
  另 `_txx54_read_event_object` 的畸形 GTI 被 bare except（L322-324）静默
  替换为事件跨度伪 GTI。
- **R95** `ogip.py:255–257`：PHA 校验对完全缺失 `HDUCLAS1` 的文件零消息
  通过（response 基类 L293-295 对缺失会告警，两处不一致；OGIP-92-007
  要求 `HDUCLAS1='SPECTRUM'`）。
- **R96** `ogip.py:100–214` 多处 bare except + L134-137/L230-232 必填关键字
  降级为 WARN：损坏的 GTI 数组/头静默跳过校验，`report.ok` 因此高估
  OGIP 合规性。

**Suggestion（汇总）**

- **R97** `grb/pipeline.py`：L763 `except TypeError` 静默重试掩盖真实
  TypeError；L1562-1567 每 chunk 独立 p0 无多重检验校正（多观测 PC 源
  伪段率上升）；L1674 vs L1570 分段边界 +1e-2 s 闭端使边界事件双计入相邻
  谱（应文档化或去 eps）；L1698 复用 60 s 请求超时作 xselect 子进程超时；
  L1138 "假设 z=0" 告警与 fit 阶段拒拟矛盾；L1800 `nh=None` 时静默落到
  XSPEC 默认 1e22；L1897/1902 报告行键不匹配时静默弃行。
- **R98** `bxa_fit.py`：L587-589 joint 拟合 flux chain 只取 spectrum 1；
  L904/993/1017/1041 `warning_messages` 永不追加（对比报告 Warnings 恒空）；
  L546 `speed` 未校验（float→str→SliceSampler TypeError，bool→nsteps=1）；
  L197-199 `resolve_priors` 的 `xspec` 参数未用。
- **R99** `gbm/pipeline.py`：L3450-3454 丢弃 `generate_gbm_response` 返回的
  路径/探测器名并重新 glob（`glg_rsp_*` 命名即失败，虽是显式失败）；L1623/
  764-777 混合批次把已有文件标 `downloaded` 且失败传输无汇总告警；
  L3783-3790 检测报告取各背景变体中最差上限的 MLE/显著性（保守但混用
  估计量）。
- **R100** `survey.py`：L5248-5252 目录名含 "survey" 即提升为 OBSID 键；
  L6370-6374 fit 门控用任一匹配行 SNR（可用别指向的亮行把非探测 PHA 送入
  全拟合分支）；L6334 vs L6379-6380 分组能段与拟合能段可不一致；L5941-5948
  product_paths 把日志/pickle 全列为产物（审计噪声）；L5339-5380 在线查询
  无空间预裁。
- **R101** `timescale.py`：L732/775/859 分段净计数截断 ≥0 与
  `_quantile_interval` 的 signed 约定并存且未在 docstring 声明（两法 T90
  不同基准）；L644 fallback SNR 方差用 `s+α·B`（正确为 `s+α²·B`，注释
  "S+al^2*B_raw = S+B" 代数错误）；L483 vs L817-866 `nmc` docstring 称
  未用但实际驱动 MC 误差，1–19 静默产出 NaN 误差。
- **R102** `ogip.py`：L189-215 `extract_gti` 不识别 `STDGTI/GTI00NN` 且
  `data is None` 提前返回而非继续扫后续 HDU（Swift/NICER/NuSTAR 事件文件
  回落 TSTART/TSTOP）；无 TLMIN/CHANTYPE 交叉校验（配对时才暴露）。

#### 子代理声明已核干净的段落

- grb/pipeline.py：全 2079 行；Li-Ma 用法、DAT 列映射、tar 路径包含检查、
  全部 FITS/np.load 上下文管理、xselect scc 时间一致性均正确。
- bxa_fit.py：全 1175 行；BXA 4.5 kwargs 传递、cstat 族门控、先验软边界
  统一（`pval,,b,b,t,t` 逗号串）、自由参数清点与 resolve_priors 逐项一致、
  会话进出清理（除 R98 同族 AllChains）——注意未证伪项已并入上文编号。
- gbm/pipeline.py：其余段落（select_gbm_detectors、read_ogip_products 的
  通道掩码/曝光比/协方差处理、single_response EBOUNDS 重编号、fit/report
  的失败传播）未见捏造成功路径。
- survey.py：其余段落（validate_survey_pha 矩阵/EBOUNDS 校验、
  profile_survey_upper_limit 的链清理与单侧边界、下载/发现/preflight
  阶段）无吞科学失败的宽 except、无未关 FITS。
- timescale.py：`_crossing_midpoint` 与 HEASoft burstdur.c 中点约定一致；
  单事件边界处理正确；MC 背景缩放方差正确。
- ogip.py：ValidationReport ok 重算纪律、get_keyword_ci 大小写不敏感
  （运行时证实）、GTI STOP<START/乱序检查、response HDUCLAS 语义正确。

#### 与历史结论的关系

- 本轮发现的 R84/R85/R87 与第 13/14 轮对 backend/stage 机制"隔离与异常
  路径完整"的结论不矛盾——那些结论限定在已读段落；本轮扩展到了此前
  未读的下载/缓存/覆盖选择路径。
- §6 未验证范围第 2、3 条所列"未逐行深读"部分至此基本消除；剩余未系统
  深读的只有 `ep/wxt/pipeline.py` 与 `fit.py` 的绘图/会话管理段落（多轮
  局部审查 + 两轮真实数据端到端覆盖，风险较低，如实记录）。

**证据**：`reviews/evidence/48-unreviewed-remainder-review.txt`。

**AI 署名**：

- **Qoder 主代理｜codex/beta 变更集未审部分深审（只读）**：harness＝Qoder
  agent harness；模型/推理档位＝未披露（运行环境未报告模型标识与档位）；
  记录时间＝2026-09-29T21:20:00+08:00；范围＝基线 `0e15c32..269587a` 覆盖
  缺口确认、5 个并行只读子代理分工深读（swift/grb/pipeline.py 全文、
  core/bxa_fit.py 全文、gbm/pipeline.py 余部、bat/survey.py 余部、
  timescale.py+ogip.py 余部）、Critical/Warning 主代理独立复核（8/8 证实、
  含 1 项运行时复现）、发现 R79–R102 记录；结果＝2 Critical、14 Warning、
  6 组 Suggestion；证据＝本节与 `reviews/evidence/48-*.txt`；**未修改任何
  代码或测试，未提交/合并/推送**。

### 21.10 2026-09-30T20:14+08:00 codex/new 只读续审：最后两个未深读文件（R103–R131）

本轮补齐 §21.9 声明的最后覆盖缺口：`ep/wxt/pipeline.py`（3045 行，首次全文
逐段深读）与 `core/fit.py`（4047 行，重点为历轮未覆盖的绘图/会话/资源管理/
导出编排段落，全文通读复核）。两个只读子代理先后执行（按并发限制串行），
主代理对全部 Critical 与 Warning 独立复核证实，其中 1 项由主代理独立运行时
复现。**未修改任何生产代码或测试，未提交/合并/推送。**

**as-found 核对**：15 个已跟踪修改文件与 §21.8/§21.9 审计时一致；自上轮以来
无 `.py` 内容变更（`gbm/mvt/{models,__init__.py}` 仅 mtime 被触碰，`git diff
HEAD` 为空）；HEAD、master、origin/master 仍为 `08fd6a6`。变更集文件清单
（`269587a..08fd6a6`，144 文件）逐一对照审查记录复核：GW 包（alert/skymap/
layers/plot/coverage）与 cluster/backprior/redshift/model_comparison/
absorption/time 等此前均有实质审查记录，本轮无其他遗漏文件。

#### 发现清单（R103–R131：2 Critical、8 Warning、19 Suggestion）

**Critical**

- **R116** `core/fit.py:2427`（`_prepared_error_parameters`）：delta 前缀
  `f"{float(delta_stat):g}"` 对整数值浮点（3.0/9.0…，含舍入为整数的
  1.0000001）产出无小数点纯数字串；HEASoft 6.37 `xsError.cxx`（先整数
  分支按参数号、`isReal && !parProcessed` 才按 delta）与
  `XSutility.cxx:154-166`（全数字即整数）证实其被 XSPEC 解析为**参数号**：
  `error_delta_stat=3.0` 时整条 error 命令失败（全部误差 uncomputed）或
  无关参数混入并按残留 delta 计算——置信区间级别静默错误；
  `error_parameters` 记账（L3199-3203）与实际解析不一致。默认 1.0/2.706
  不触发，经 `FitConfig.error_delta_stat` 可达（PyXspec 实测与源码核验见
  证据 50）。作者对 1.0 的 `"1."` 特判反证 delta 语义意图。
- **R117** `core/fit.py:777-789`（`LightcurveFitter.fit`）：sigma 非法值
  回退 `0.1*max(|value|, eps)` 对零计数 bin（value=0 → Poisson
  sqrt(0)=0 常态）得 sigma≈2.2e-17、权重 ~2e33，该点支配拟合且无警告。
  主代理独立复现：60 点衰减幂律 clean index=-1.2023±0.0028
  （red_chi2=1.46），植入 5 个 value=0/error=0 bin 后 index 冻结于初值
  ±0.000000、red_chi2=979.5、success=True——静默数据错误。

**Warning**

- **R103** `ep/wxt/pipeline.py:1487-1498,1935-1956`：区域审批仅绑定
  regions.json 哈希，不区分人审与 `auto_approve_regions` 机器批准；同
  workspace 改为要求人工复核重跑时旧机器批准仍有效，needs_review 不触发；
  批准文件无身份/时间戳。
- **R104** `ep/wxt/pipeline.py:1599-1602`：xselect 子进程超时取
  `extraction.timeout_s or execution.command_timeout_s`，两者默认均 None
  （config.py:78,450）；超时管线本身已支持（xselect.py:589/710/725），默认
  配置下挂起即整轮永久阻塞。
- **R105** `ep/wxt/pipeline.py:2740-2750` + `core/products.py:620,623`：
  `build_quicklook_flux_curve` 在 T90 无重叠或 T90 加权净率 ≤0 时抛
  ValueError 且无捕获——弱源在最末科学阶段硬失败，拖垮已完成 fit 的整轮；
  与同阶段 disabled/fit_unavailable 降级路径不一致。
- **R106** `ep/wxt/pipeline.py:2122-2126,2717,2721` 等：target_id 含 "." 时
  `Path.with_suffix` 截断产品名（运行时复现 `GRB.250418_wxt_txx→GRB.png`）；
  核实 full/soft 分属不同子目录，无静默数据覆盖，属命名契约破坏。
- **R107** `ep/wxt/pipeline.py:2557-2569,2605-2637`：全部时段拟合失败时
  fit 阶段仍返回默认 `PipelineStatus.COMPLETED`（core/pipeline.py:93），
  明细仅在 fit_summary.json 与报告文本。
- **R118** `core/fit.py:3023-3024`：`fit_prepared` 清场不清 chain
  （全文件无 `AllChains` 清理；HEASoft `clearChains` 仅 chain-delete/退出
  调用）；同进程 chain 采样后 MLE 拟合时旧 chain 累积。
- **R119** `core/fit.py:1205-1210`：`plot_fit` 传入 `ax` 时 `ax2=None`
  无条件，`show_residuals=True` 被静默忽略（运行时证实）。
- **R120** `core/fit.py:1145`：`plot_fit` 默认 `ylabel="Flux (erg/cm2/s)"`
  而数据通常是 counts/rate——默认图轴单位系统性错标。

**Suggestion（详见证据 49/50）**

- wxt（R108–R115）：.img 推断失败不回退 expcorr（L949-956）；DS9 行内
  第二个 shape 不变换（L657-672）；soft/hard 沿用 full band α（L1985/2024）；
  BB 尾段并入后不复检阈值（L1223-1240）；stage_code_dependencies 覆盖不全
  （L1429-1461）；`_validate_ogip_bundle` ARF 边界全 NaN 静默通过
  （L1124-1141）；`detector.replace("CMOS","CMOS")` 无操作（L1557）与
  不可达分支（L1707-1721）；性能/磁盘卫生（每谱目录复制 RMF/ARF、事件
  全量重读 4 次）。
- fit.py（R121–R131）：astropy 7.0 弃用 LMLSQFitter+bounds（L841-849，
  主代理复现时同现）；`Xset.parallel`/`Fit.query` 进程级残留与 chain
  复跑 FileExistsError（L2042/3027/2007/3366-3370）；`galactic_nh=None`
  时 freeze 静默忽略、joint 不链各组 TBabs.nH（L2572/2771-2801）；候选级
  warnings 不进比较 txt（L3761/3866-3870/3944，R98 同族）；`to_dict`
  遗漏 logz/logzerr（L266-284）；`statistics["value"]` 裸取键
  （L3798-3803）；`fit()` 返回值含不可序列化 Catalog 且落盘缺溯源
  （L4043-4046）；os.chdir 加载非线程安全（L2840-2849）；恒真/不可达
  分支与 docstring 过时（L783/789/828-830/3-16）；chain 模型副本静默丢弃
  （L1888-1893）；智能标签逐候选全画布重绘性能（L1425-1438/1466-1483）。

#### 主代理独立复核记录

- **R116**：fit.py 源码 + `%g` 运行时语义 + HEASoft xsError.cxx/
  XSutility.cxx 权威源码三方证实；XSPEC 真实谱实测（`"3 1"` 失败/
  `"3. 1"` 成功）由子代理完成并引用输出。
- **R117**：主代理独立合成数据复现（与子代理不同随机种子，结论一致，
  数据见证据 50）。
- **R103–R107**（wxt 全部 Warning）：主代理逐一读码证实，R106 运行时
  复现；R109/R111/R114 抽查证实。
- **R118/R119/R120/R121**：主代理读码/运行时证实；R122–R131 按
  file:line 记录，机制经子代理源码核对。

#### 覆盖结论

至此，codex/beta 变更集（基线 `269587a`，现工作区 `08fd6a6` + 15 个未提交
修改文件）内已无未系统深读的生产文件：§21.9 声明的两处剩余缺口（wxt
pipeline 全文、fit.py 绘图/会话段落）本轮补齐，其余文件均有历轮实质审查
记录。历史"通过"结论仍按 §21 前言限定为当时版本与范围；本轮 29 项新发现
的修复决定仍待用户（修复时须在已跟踪测试文件补回归，见 §21.4 忽略规则）。

**证据**：`reviews/evidence/49-wxt-pipeline-deepread-review.txt`、
`reviews/evidence/50-fit-deepread-review.txt`。

**AI 署名**：

- **GLM 主代理｜codex/new 只读续审（第六轮：wxt pipeline + fit.py 补齐）**：
  harness＝ZCode CLI；模型/推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；
  记录时间＝2026-09-30T20:14:08+08:00；范围＝as-found 核对、变更集覆盖
  完整性再确认、两个只读子代理串行派发与结果复核（Critical 全部、
  Warning 全部、Suggestion 抽查）、R117 独立运行时复现、本节与证据
  49/50 及 Obsidian 进度记录撰写；结果＝R103–R131（2 Critical、8 Warning、
  19 Suggestion），覆盖缺口清零；**未修改任何生产代码或测试，未提交/
  合并/推送**。
- **/zcode/wxt-pipeline-deepread｜ep/wxt/pipeline.py 全文深读（只读）**：
  harness＝ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝
  运行环境披露模型标识 `GLM-5.3-Flash`（account:bigmodel-start-plan/
  GLM-5.3-Flash），推理档位未披露；记录时间＝2026-09-30T18:47+07:00
  （子代理自报，本机时区 +07:00）；范围＝`ep/wxt/pipeline.py` 1–3045 行
  全文、真实 WXT L2 产品（ep11916655873wxtCMOS21l23v1）与合成用例
  运行时验证（/tmp）、HEASoft GTI.cxx 口径对照；结果＝R103–R107（5
  Warning）+ R108–R115（8 Suggestion）与全文覆盖清单；证据＝证据 49。
- **/zcode/fit-deepread｜core/fit.py 未覆盖段落深读（只读）**：harness＝
  ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝未披露
  （子代理运行环境未报告）；记录时间＝2026-09-30T19:03:56+07:00（子代理
  自报）；范围＝`core/fit.py` 1–4047 行全文（重点绘图/会话/导出）、真实
  WXT 谱 PyXspec error 命令实测、合成光变复现、HEASoft xsError.cxx/
  XSutility.cxx/ChainManager.cxx 等源码核验；结果＝R116–R131（2 Critical、
  3 Warning、11 Suggestion）与全文覆盖清单、R22 澄清；证据＝证据 50。

#### 21.10 补记（2026-09-30T20:35+08:00）：审查期间工作区出现并行工作流变更

本轮 as-found 核对（18:19–18:20 +07）时工作区为 15 个已跟踪修改文件，上述
结论对该状态有效。审查结束核对时（20:30–20:35 +08）发现**另一并行工作流
（GBM MVT 迁移，实施者自述 harness=Codex desktop，见其报告
`reviews/gbm-mvt-migration.md`）在本轮会话期间（18:24–19:15 +07）向同一
工作区落盘**：

- 已跟踪文件新修改：`gbm/subthreshold/data.py`（18:24，108 行变更——
  `latest_products/merge_tte_events/interval_exposure/read_detector_events`
  等 TTE 公共职责抽取至新 `gbm/tte.py`，旧导入声称保留）、
  `gbm/subthreshold/pipeline.py`（18:41，1 行——`stage_code_dependencies`
  增补 `tte.py`）、`AGENTS.md`（18:55，+26 行 MVT 验收标准）。
- 新增未跟踪：`gbm/mvt/` 包（engine/data/pipeline/models/plots/
  resampling/_vendor）、`gbm/tte.py`、`docs/usage/gbm_mvt.rst`、
  `test/test_gbm_mvt.py`、`scripts/validate_gbm_mvt.py`、
  `reviews/gbm-mvt-migration.md`、`reviews/evidence/gbm-mvt/`，另有
  `reviews/t90-aanda-20260930.md` 及其证据目录。
- 该工作流自述"实现、原样例和弱源完整等价验收通过；亮源细分辨率验收
  进行中"，且明示为**实施者验收，不冒充独立 AI review**。

影响与处理：本轮深读的 `ep/wxt/pipeline.py` 与 `core/fit.py` mtime 仍为
2026-09-25，未受漂移影响，R103–R131 结论有效；原 15 个已跟踪修改文件中
除上述 3 个外其余未变。`subthreshold/data.py` 的 TTE 抽取与 `tte.py` 新
模块**未经任何独立审查**（超出本轮范围，属 MVT 迁移任务的在途工作，且
其亮源验收尚未完成）；§21.6–21.9 对 GBM subthreshold 文件的审查结论
仅适用于其记录时点的状态。MVT 迁移在其实施者宣告完成并稳定后应作为
独立的未审查变更集接受独立审查。

### 21.11 2026-09-30T21:18+08:00 GBM MVT 迁移独立审查（只读，R132–R139）

对 §21.10 补记所列并行工作流（GBM MVT 迁移，实施者自述 harness=Codex
desktop）当日落盘的变更做独立代码审查：`gbm/mvt/` 新包约 900 行、
`gbm/tte.py` 抽取、`subthreshold/data.py` 删减与再导出、test/scripts/
docs/打包，共约 2100 行。1 个只读子代理深读 + 主代理独立核验（哈希、
AST、运行时），发现清单 **R132–R139：无 Critical，1 Warning（R132，主
代理与子代理独立发现收敛），1 项归属澄清（R133），6 Suggestion**。
**未修改任何生产代码或测试，未提交/合并/推送。**

**实施者声明核验（8/8 成立）**：(1) `_vendor/` 三个 Haar 函数相对上游
fork 57e7a0f 仅 53 行适配差异（docstring/去全局 backend 与 warning
过滤/相对导入/pylab 延迟导入/只读诊断钩子），子代理独立整模块 AST 剥离
对照三文件全部一致，且新合成光变（非测试 seed）下上游原函数与 vendored
`compute_mvt` 的七项返回 + 8 类中间数组逐元素 exact match；UPSTREAM.json
9 项 SHA256 与本地上游副本逐一吻合，校准 npz 逐字节一致（主代理独立
复核）。(2) `compute_mvt` 包装语义忠实，多处守卫较上游更严且已声明。
(3) Poisson 重采样显式随机流 worker 数不变（workers 1 vs 3 实测一致）。
(4) 汇总分位数步骤与上游 TTE_SIM_v2.py 逐步一致。(5) 适配层 GTI 覆盖
守卫、每探测器能道边界（实测三探测器能段互异）、死时间口径不重不漏。
(6) tte.py 三个函数 AST 逐字纯移动，旧导入路径运行时验证可用。
(7) 状态链诚实（blank 控制实测 needs_review/unavailable，状态计数完整）。
(8) 许可与资源文件进 wheel 实测；nrbutler/mvt 许可链未决如实记载。

**发现**：

- **R132（Warning）** `gbm/tte.py:65`：抽取时 `seconds(interval)` 改为
  裸 `interval.to_value(u.s)`，丢失 `models.seconds()` 的 Quantity 强制
  与非有限值拒绝两道校验（运行时证实 NaN 区间静默通过并产出空 GTI/
  空事件而非报错；裸 tuple 报 AttributeError）；与模块 docstring
  "without numerical changes" 矛盾。建议恢复 `seconds()` 复用或修正
  声明。默认调用方传 Quantity 配置值，属入口契约回归而非现行数据错误。
- **R133（归属澄清，非缺陷）**：子代理对照 HEAD 疑似 `data.py` 的
  `nominal_source_free_control_blocks` 整块判据为 MVT 批次未申报改动；
  主代理以证据 38（09-29，早于 MVT 落盘）核实其为已审查的 **R78 修复**。
  MVT 对该文件的实际改动仅为抽取删减；MeasuredHistory 的 with 改写为
  资源等价。
- **R134–R139（Suggestion）**：tte.py 错误路径 GbmTte 句柄未关（L72-73/
  L77-78）；`docs/usage/gbm_mvt.rst` 标题下划线短 2 字符（docutils
  ERROR，`-W` 构建失败）；engine 每样本异常捕获列表窄于上游（有意权衡，
  建议文档注明）；validate 脚本 AST 对照未覆盖模块级语句（本轮已用更强
  整模块对照补齐，PASS）及其 sys.modules 裸名注册卫生问题；vendored
  诊断行冗余双条件与死导入（留待上游同步清理）。

**测试**：`test_gbm_mvt.py` 15 passed（两次独立运行）；subthreshold +
merge + swift 回归 108 passed, 1 deselected，无失败。

**边界**：实施者的亮源细分辨率 300 次生产对照仍在执行（20:03 报告更新：
粗档对照通过、细档单次一致、5/300 参考已取得）——本轮审查覆盖当前代码
状态，该验收完成后的实施者结论仍属实施者验收，不替代本独立审查对最终
数据的复核。审查期间 3 个文档/报告文件被并行进程更新（生产代码无变化，
as-found 快照对照确认）；通用物理正确性、论文逐值复现与 nrbutler/mvt
上游授权仍未证明（与实施者披露一致）。

**证据**：`reviews/evidence/51-mvt-migration-review.txt`；实施者报告
`reviews/gbm-mvt-migration.md`、`reviews/gbm-mvt-pg-snr.md` 及
`reviews/evidence/gbm-mvt/` 原件。

**AI 署名**：

- **GLM 主代理｜GBM MVT 迁移独立审查（第七轮）**：harness＝ZCode CLI；
  模型/推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；
  记录时间＝2026-09-30T21:18:38+08:00；范围＝实施者两报告阅读、as-found
  快照（23 文件 sha256）、UPSTREAM.json 清单与上游副本哈希独立核验、
  tte.py 抽取 AST 纯移动比对与 R132 独立发现、旧导入兼容运行时验证、
  test_gbm_mvt 与相关回归运行、AGENTS.md/docs/打包检查、R133 归属澄清
  （证据 38）、子代理发现复核、本节与证据 51 及 Obsidian 记录撰写；
  结果＝R132–R139（无 Critical，1 Warning，1 归属澄清，6 Suggestion），
  实施者 8 项声明全部核验成立；**未修改任何生产代码或测试，未提交/
  合并/推送**。
- **/zcode/mvt-migration-deepread｜GBM MVT 迁移代码深读（只读）**：
  harness＝ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝
  运行环境披露模型标识 `GLM-5.3-Flash`
  （account:bigmodel-start-plan/GLM-5.3-Flash），推理档位未披露；
  记录时间＝2026-09-30（子代理报告，本机时区 +07:00 换算）；范围＝
  mvt 包/tte.py/vendor 全文、上游逐 hunk 差异分类与独立整模块 AST 剥离
  对照、新合成光变数值等价对照（exact match）、随机流 worker 不变性
  实测、.runtime 报告只读抽查、wheel 解包与 .gitignore 验证；结果＝
  R132–R139 与全文覆盖清单；证据＝证据 51。

### 21.12 2026-10-06 未审增量全面审查（只读，R140–R152）

对第七轮检查点（09-30）之后落盘的全部工作区变更做全面独立审查。基线 HEAD
`08fd6a6` 不变。两个只读子代理串行深读 + 主代理独立核验（AST、哈希、运行时
对照）。**本轮无 Critical 新缺陷**；三个工作流的核心主张全部独立核验成立。
**未修改任何生产代码或测试，未提交/合并/推送。**

#### 覆盖盘点

mtime 分层确认：原 15 个已跟踪修改文件与 MVT 迁移文件自第七轮后**无任何
变化**（其审查结论持续有效）。新审增量全部归属三个工作流：
(1) **core 双语注释**（14 模块）：主代理独立 AST 核验（剥离 docstring 后
与 HEAD 可执行 AST 比对）——11 模块纯注释证实；spectrum_prep 逻辑差异恰为
已审 R70/71/73；utils/io 差异归属 SNR 迁移。(2) **duration/T90 修正**
（timescale.py +559/−643 + 新回归 18 passed + 配套）。(3) **snr/
gv_significance 迁移**（新模块 463 行 + 测试/脚本/文档/导出）。

#### duration 修正（R140–R147）

t90-aanda 审查（09-30）的 A01–A08 **逐项核实为真实修复**：A01 EXTraS 背景
时间变换（OFF 率积分变换 + 单调节点映回 + 零率拒绝；经验 CDF 按用户决定
暂缓并文档化）；A02 名义/MC 统一估计量 + Koshut 主误差 + MC 降为重选窗
bootstrap 诊断（裁零删除）；A03 MC 逐样本重选窗（失败原因/有效率/触边
保存）；A04 GTI 严格契约（旧代码 OFF 带洞静默 T90=18 s → 现显式拒绝）；
A05 固定分箱严格止于 T100（旧 fixed 1000 s → T90=900 越界 → 新有界；
T50≤T90≤T100 结构性成立）；A06 TIMEZERO 路径/对象统一（旧 40 vs 1040 s）；
A07 有符号累计（负计数反例与解析值精确一致）；A08 MAD 改标诊断、
`err_sys` 恒 NaN、`confidence_status='not_calibrated_68_percent'`。
Koshut σ²/交点跨度/Eq.16 与误差阈值-名义归一化恒等性逐式核实；当前
timescale.py 与 v4 验收快照逐位一致；真实 EP260119a 复现 v4 报告精确值；
新测试对 HEAD 全部判别失败（回归有判别力）。

- **R140（Warning）** `config.py:114` `nmc=3000` 默认值未随"每样本完整
  重拟合"新语义重新标定（~52 s/次外推），`ep/wxt/pipeline.py:2075` 直接
  传递——管线用户静默承担成本；建议下调默认或文档给出耗时量级。
- **R141（Warning）** `reviews/duration-20261002.md` 把扩面回归触发的
  `bayesian_blocks_exposure` 正曝光校验表述为"已有"，实为本次工作区新增
  （HEAD 0 处、diff 新增 2 处）；fail-loud 合理，建议报告补更正注记。
- **R142–R147（Suggestion）**：MC 生成模型 histogram 末箱右闭与 SNR
  searchsorted 口径差 1 光子（L859 vs L774）；`_duration_koshut` 掩码/
  分箱长度契约；`unresolved_koshut_crossings` 合并两种内部原因；11 个
  `_txx54_*` 旧助手零调用未标 deprecated；`boundary_hits` 精确相等口径；
  `meta.timezero` 真值回退（R142/R147 主代理抽查证实）。

#### snr/gv_significance 迁移（R148–R152）

五分支公式与冻结上游（`gv_significance@946d761a`，本地副本 PROVENANCE
7/7 哈希吻合、LICENSE 逐字节相同）逐式对照一致：known/pp/pg 解析分支
**逐位一致**、pp 高斯优化分支 1e-7 内；子代理独立 337 组数值对照全部通过，
总体最大差异 6.602092305751534e-13 与实施者声明值逐位相同；真实 WXT 三种
PP 值逐位复现。三处声明的有意偏离（零计数解析极限、不复制 squeeze、
零计数/未收敛/非有限拒绝）全部核实且方向保守——上游双零 PP 高斯返回
√2000、未收敛返回 −9.99 伪显著性，新接口均拒绝。Li–Ma 计算体与 HEAD
逐行一致（仅 docstring 双语化，主代理独立核实）。

- **R148（Warning）** 验收证据哈希链过期：acceptance/validation.json 钉定
  的 significance.py 哈希与当前文件失配（10-02 20:51 双语 docstring 修改
  晚于 12:13–13:33 验收）；计算体未变但复现链断裂，建议重跑 validate_snr
  或重签哈希。
- **R149（Warning）** `test_exports_and_legacy_body` 失败仅由 docstring
  引起（剥离后与基线/HEAD AST 完全一致）；全量门禁因此多一项红；建议
  重铸基线或改 docstring 不敏感比较。
- **R150（Warning）** PP 高斯拒绝面大于声明证据网格：大偏差亏损端
  （如 n=1, b=1046, α=1, σ=0.1）上游未收敛返回伪值、新接口正确拒绝，但
  该输入类不在 353 案例内、文档未提及；建议补案例或写明。
- **R151/R152（Suggestion）** known 分支 p→1 溢出消息不专属；
  validate_snr 中间量对照分支顺序假设当前案例成立。

#### 门禁与交叉工作流问题

全量离线门禁（hea）：`4 failed, 1517 passed, 50 skipped, 6 deselected,
15 subtests passed`。4 项全部定位：2 项 spectrum_prep（R70/71/73 既有本地
预期）；1 项 `test_alpha_effect`（新 EXTraS 零背景率守卫触发旧测试场景，
SNR 报告已列为预期，旧测试预期未更新）；1 项 R149（双语 docstring 破坏
SNR 冻结基线测试）。**R148/R149 是本轮发现的交叉工作流问题**：双语注释
任务修改了其 14 模块清单之外的 significance.py，且未同步实施者的冻结哈希
证据——需实施者侧重跑验收或重签哈希，属流程修复而非计算缺陷。

#### 结论

第七轮检查点后的未审变更至此**全部审毕**，无未覆盖变更。累计开放清单
R79–R152：Critical 4 项（R79/R80/R116/R117）不变；Warning 新增 R140/
R141/R148/R149/R150；Suggestion 新增 R142–R147/R151/R152。R73 网格检查、
R79–R139 各项等既有开放项状态不变。

**证据**：`reviews/evidence/52-unreviewed-increment-20261006.txt`；实施者
报告 `reviews/core-bilingual-comments.md`、`reviews/t90-aanda-20260930.md`、
`reviews/duration-*.md`（11 份）、`reviews/snr-gv-significance-migration.md`
及其 evidence 目录为本轮审查输入原件。

**AI 署名**：

- **GLM 主代理｜未审增量全面审查（第八轮）**：harness＝ZCode CLI；模型/
  推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-06T19:09:02+08:00；范围＝mtime 分层盘点、三工作流报告阅读、
  14 模块双语注释独立 AST 核验、Li–Ma 迁移等价性与 PROVENANCE 7/7 哈希
  核验、全量门禁与失败定位（R149 独立发现）、两子代理串行派发与发现复核
  （R140/R141/R142/R147 证实）、证据 52 与本节及 Obsidian 撰写；结果＝
  R140–R152（无 Critical，5 Warning，8 Suggestion），三工作流核心主张
  8/8+8/8 项核验成立；**未修改任何生产代码或测试，未提交/合并/推送**。
- **/zcode/duration-corrections-deepread｜timescale duration 修正深读
  （只读）**：harness＝ZCode CLI 子代理（general-purpose，串行）；模型/
  推理档位＝运行环境披露模型标识 `GLM-5.3-Flash`，推理档位未披露；记录
  时间＝2026-10-06T18:37+08:00；范围＝timescale.py 全文 diff 逐 hunk、
  11 份 duration 报告、HEAD 旁载对照（A02/A04/A05/A06/A07 旧行为复现）、
  真实 EP260119a 复算、18 项新回归运行；结果＝A01–A08 全修复结论 +
  R140–R147 与覆盖清单。
- **/zcode/significance-deepread｜significance 迁移深读（只读）**：harness＝
  ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝运行环境披露
  模型标识 `GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-06T18:02+07:00；范围＝significance.py 五分支逐式对照上游、
  独立 337 组数值对照（max diff 6.6e-13 与声明逐位同）、真实 WXT 复算、
  测试/脚本/文档/许可证审查；结果＝R148–R152 与覆盖清单。

### 21.13 2026-10-06T19:36+08:00 审查收尾：MVT 定稿独立验证 + SNR 证据复现（只读）

第八轮（§21.12）遗留的三项弱验证/在途边界本轮闭合；漂移检查确认第八轮
检查点后工作区无任何变更。**未发现新缺陷，无新增 R 编号**；累计开放清单
仍为 R79–R152（Critical 4 项：R79/R80/R116/R117）。**未修改任何生产代码
或测试，未提交/合并/推送。**

1. **MVT 定稿验收独立验证**：实施者 09-30T21:30+08 定稿报告（第八轮审查
   在定稿前，"在途"记录正确）声明细档 300 次生产完成、单次观测与上游
   exact match、5/300 抽样原版核对，且如实声明未做全量 300 逐一重算。
   主代理独立复算亮源最细档（2.55M 箱，dt=1e-5 s）：**上游原函数与迁移版
   七项返回逐元素相等**（bw0/bw1 两档均证实）；保存产物可复现——power/
   noise/power_error/significant 与 weights 逐位一致，tau 阶梯最大差
   8.5e-10 s（浮点尾差）对应 measure raw_values 最大差 1.973e-12，无科学
   影响（初判"不一致"为 NaN 比较伪差）。
2. **SNR 证据独立复现**：`scripts/validate_snr.py` 全量重跑（exit 0）与
   实施者证据完全一致——353 等价案例、10 上游故障守卫、最大 Z 差异
   6.602092305751534e-13（逐位同）；四组真实数据（WXT + GBM 三组）逐叶
   比对 0 处不匹配。第八轮仅审脚本与证据的 GBM"背景重拟合→协方差→PG"
   链路由此由独立执行复现。
3. **双语 docstring 抽查**：8 个函数机械抽查，raise/拒绝声明与函数体一致，
   无事实性错误；236 条逐条审读的边界保留（验收属性仍为"可执行 AST 无
   变更"）。

至此，截至 2026-10-06 的工作区全部变更（含三个并行工作流及其定稿报告）
审查完毕，无未审变更。待用户决定事项不变：R79–R152 的修复优先级
（Critical 4 项优先）；MVT 细档 300 次逐一重算、论文完整复现与
nrbutler/mvt 授权核实为实施者已声明的遗留边界。

**证据**：`reviews/evidence/53-review-closure-mvt-snr-verification.txt`。

**AI 署名**：

- **GLM 主代理｜审查收尾（第九轮）**：harness＝ZCode CLI；模型/推理档位＝
  运行环境披露模型标识 `account:bigmodel-start-plan/GLM-5.3-Flash`，
  推理档位未披露；记录时间＝2026-10-06T19:36:08+08:00；范围＝漂移检查、
  MVT 定稿报告与新证据审读、亮源最细档上游 vs 迁移独立复算（bw0/bw1 两档
  exact + 产物可复现性 ≤2e-12 定量）、validate_snr 全量重跑与四组真实数据
  逐叶比对、双语 docstring 抽查、证据 53 与本节及 Obsidian 撰写；结果＝
  无新缺陷，MVT 定稿与 SNR 证据独立验证成立；**未修改任何生产代码或测试，
  未提交/合并/推送**。

### 21.14 2026-10-07T03:58+08:00 基线代码首次全面审查（只读，R153–R213）

用户指示"看看还有别的没审查的也审查一下，不止是这次变更"。盘点确认：九轮
审查的覆盖范围始终是"codex/beta 变更集 + 工作区未提交变更 + 新增未跟踪
文件"，**从未被变更触碰、也从未被任何轮次提及的基线代码约 9650 行/44 文件**
（甄别后真未审约 7500 行）从未进入审查范围。本轮以三个只读子代理串行深读
+ 主代理自审完成首次覆盖。**新发现 R153–R213：5 Critical、28 Warning、
28 Suggestion；全部 Critical 经主代理独立证实。未修改任何生产代码或测试，
未提交/合并/推送。**

#### Critical（5 项，全部独立证实）

- **R153** `core/datasets.py:309–353`：**netdata**（公开 API、`src - bkg`
  落点）背景时间对齐契约无验证——shape 相同 + `np.allclose` 默认容差即按
  索引配对相减；rebin 分支 `align_ref=src.timezero or None` 锚点错位且
  rebin 后不回读校验；rebin 的 XRONOS max-bin 规则还会静默抬升 binsize。
  主代理复现：完全不重叠时段零警告静默相减；0.9 s 偏移 @1e5 s 被放行
  （R154 同机理）。用不同时段离源背景这一标准做法会静默出错。
- **R171** `ftools/rmf_mapping.py:147–153`：sparse 'expected' 路径把
  n_channels 尺寸布尔掩码套在 len(unique) 尺寸数组上——通道子集（真实
  事件文件常态）IndexError；调用方 `data.py:871` `except: pass` 静默回退
  EBOUNDS 中点 → **RMF 后验事件能量映射对真实数据长期失效无报错**。
  主代理复现（子集与重复通道均崩，全道顺序查询属偶然正确）。
- **R172** `ftools/ftselect.py:79–115`：`&&`→`&` 未保优先级——多事件
  `PI>100 && PI<1000`（docstring 自举示例句式）RuntimeError，单事件
  静默错判；`xselect.py:1155` 活跃路径。主代理复现。
- **R173（潜伏）** `ftools/teldef.py:423–426/:455`：with_pointing 两方向
  同用 `align.T` 不互逆（往返误差 −16.8/−4.5 px；对照 coordfits
  teldef_xform.c:250 应一处用 `align`）。当前无调用方。
- **R190（潜伏）** `lf/detectability.py:147–196`：模拟 ON 模板背景双计
  （`corrected_counts_src` 按全仓约定为 ON 总计数，又被当源信号缩放并
  叠加 `background_rate`，Li&Ma 只扣新增背景）→ SNR 系统性偏高、z_max
  反保守高估（演示 1.30×）。主代理代码链证实（redshift.py:536 +
  lcfake docstring 分支 2 + utils docstring 三处约定互证）。

#### Warning（28 项，摘关键）

- datasets：`__add__` labels 崩溃（R155）；gti `adjust_gti_to_frame`
  TIMEPIXR 语义混用（潜伏，R158）。
- host.py（宿主星系交叉匹配）：模块级全局屏蔽所有 warnings（R160，复现）；
  **声称 GLADE+ 实为 GLADE v2.3**（VizieR VII/281 vs VII/289，R161）；
  四目录查询 fail-open 且 run() 恒报成功（R162）；Gaia 'Gal?' 分支一字
  之差整体失效（R163）。
- ftools：radecroll↔quat 约定与 HEASoft align.c 不一致且无 ROLLOFF/
  ROLLSIGN（R174）；teldef 畸变 origin 漏 crpix·cdelt + DELTAX/DELTAY
  HDU 约定不兼容（R175）；region 排除形状/sexagesimal 坐标静默丢弃
  （R176/R177）、天球半径量级猜测与 DS9 arcmin 缺省不符（R178）、
  4 参数 ellipse 退化为原点单位圆（R179）；**grppha_hsp 对 UPDPHA FATAL
  仍报 success**（真实 HEASoft 实证，R180）+ heasoftpy 无超时（R181）；
  ftrbnrmf 负 channel_map 静默并道（R182）；fextract 空道崩溃 + GTI
  曝光不裁剪（R183）；teldef det2sky 缓存复用不校验指向（R184）；
  datasets/gti 其余见 R154/R156–R159；heasoft 初始化无超时（R168）。
- lf：area_ratio docstring 语义冲突（R191）；docstring 默认值矛盾族
  （R192）；legacy `_snr_at` 吞错返回 0.0（R193）与无谱指数回退两路
  不一致（R194）；tqdm 全局禁用不在 finally（R195）；z=0 除零（R196）；
  lcfake 随机流不可复现（R197）；cluster silhouette 退化崩溃（R198）。

#### Suggestion（28 项）

R156/R157/R159/R164/R166/R167/R169/R170（datasets/gti/host/plotpanel/
plotstyle/heasoft/galactic——含 plotstyle `with_suffix` 截断=R106 同族、
galactic 缓存并发写）；R185–R189（rmf_mapping numba 死导入与跨后端随机
流不可复现、ftselect eval 沙箱、mdb 大小写、region 边界语义三后端不一、
分组三路径非法 min_counts 行为矛盾）；R199–R212（redshift rate=0 无守卫、
死分支、二分统计契约、T100 静默回退、npz 句柄、单点除零、target_dt 丢尾
仓、specfake KConfig.background 对 K 无效、Type-II BACKSCAL 广播、
detectability 恒真守卫/单点 nan/双缺省保底、legacy 链接失败仅 print、
compute_grid 文档不符）；**R213**（主代理）jinwurs Rust 核逻辑正确但
`ow[i]==0` 无守卫、未知 method 静默走 rate 分支、且为**休眠 API**（全仓
零调用、零回归）。

#### 已核实无误（要点）

grppha.py loadMin ≡ HEASoft grouping.cxx（3000 组随机 0 不匹配）；质量
折叠与重分箱 ≡ pha.cxx/rmf.cxx；四元数基元正交性 2.2e-16；CRVAL/CRPIX
gnomonic 往返 1e-13 px；**K-correction 公式与直接能段积分一致 ≤2.2e-16**；
**legacy 与新版红移缩放公式代数等价（ratio=1.0，无静默科学错误）**；
XSPEC 宇宙学常量 ≡ astropy Planck18（逐位）；13 个兼容 shim 全部正确；
netdata 同网格正路径、galactic 缓存、heasoft rc 解析、mdb 继承查询、
region 像素几何等逐项运行时验证通过。

#### 测试覆盖缺口

ftools 13 文件中 10 个、lf/cluster/model 全部、core-P1 七模块（仅 1 条
正路径回归）**没有任何直接测试**——基线代码的回归覆盖率与生产风险不成
比例，修复时应连带补跟踪回归。

#### 结论与遗留

本轮后**基线代码中已无从未审查的生产模块**，仓库全部生产 Python/Rust
源码至此均至少经过一轮审查。累计开放清单 **R79–R213（Critical 9 项：
R79/R80/R116/R117/R153/R171/R172/R173/R190）**。遗留边界：subthreshold
_vendor（USRA-STI 冻结上游）哈希无本地副本可核、本地 ignored test/ 55 文
件与 examples/ 维持既往边界、docs 散文未审。修复优先级建议：R153/R171/
R172（活跃路径科学正确性）→ R190/R173（潜伏但核心量错误）→ R79/R80/
R116/R117（前轮 Critical）→ 各 Warning。

**证据**：`reviews/evidence/54-baseline-first-review.txt`。

**AI 署名**：

- **GLM 主代理｜基线代码首次全面审查（第十轮）**：harness＝ZCode CLI；
  模型/推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T03:58:07+08:00；范围＝44 零提及文件盘点甄别、三个子代理串行
  派发与全部 Critical 独立复核（R153/R154、R171、R172 运行时复现，R190
  代码链证实，R173 证据审读）、jinwurs/rebin_rs/validate 脚本/_vendor
  溯源自审、证据 54 与本节及 Obsidian 撰写；结果＝R153–R213（5 Critical、
  28 Warning、28 Suggestion），基线生产代码审查盲区清零；**未修改任何
  生产代码或测试，未提交/合并/推送**。

- **/zcode/core-baseline-deepread｜core-P1 基线深读（只读）**：harness＝
  ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝运行环境披露
  模型标识 `GLM-5.3-Flash`，推理档位未披露；记录时间＝2026-10-06（+08:00）；
  范围＝host/datasets/plotpanel/gti/galactic/plotstyle/heasoft 全文
  （HEAD 快照）+ 20 项边界断言复现；结果＝R153–R170 与覆盖清单。
- **/zcode/ftools-baseline-deepread｜ftools 家族基线深读（只读）**：
  harness＝ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝
  运行环境披露模型标识 `GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T02:00+08:00；范围＝ftools 13 文件全文 + HEASoft 6.37 源码
  对照（grouping/pha/rmf/coordfits/align/region）+ grppha_hsp 真实
  HEASoft 端到端实证；结果＝R171–R189 与覆盖清单。
- **/zcode/lf-baseline-deepread｜lf/cluster/model 基线深读（只读）**：
  harness＝ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝
  运行环境披露模型标识 `GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T03:54+08:00；范围＝lf 5 文件 + cluster + modelbase + 13 shim
  全文 + PyXspec/astropy 数值对照；结果＝R190–R212 与覆盖清单。

### 21.15 2026-10-07T19:08+08:00 审查收尾：_vendor 上游核验 + docs/CI/打包扫描（只读，R214）

闭合第十轮遗留边界；漂移检查确认第十轮后无变更。**唯一新发现 R214
（Warning）**；累计开放清单 **R79–R214，Critical 9 项不变**。未修改任何
生产代码或测试，未提交/合并/推送。

1. **subthreshold/_vendor 上游核验（边界闭合）**：克隆
   USRA-STI/gamma-ray-targeted-search@1bc1e913（远程 HEAD 与清单钉住
   commit 一致）——original_sha256 **9/9 逐位吻合**；vendored 9 文件相对
   上游仅差 1 行来源注释 + 包内相对导入（逐文件 diff 核对），零数值/
   逻辑改动；license.txt 逐字节一致。该 vendor 升级到与 MVT _vendor 同等
   验证标准。上游克隆留存 /tmp/gts-upstream 供修复者核对。
2. **docs 扫描（R214）**：机械核验 51 条文档 import 语句——
   `docs/usage/bxa_fitting.rst` 教 `from jinwu.core import WXT, FitConfig`
   与 `WXT(fitting=FitConfig(...))`，均不存在（惰性 `__getattr__` 未导出；
   实际写法 `from jinwu.core.config import instrument, FitConfig` +
   `dataclasses.replace(instrument("WXT"), ...)`，主代理运行时验证）。
   其余 50 条全部解析。docs 散文逐句审读维持为边界。
3. **CI/Makefile/打包**：ci.yml（3.11–3.13 矩阵、离线门禁）、publish.yml
   （5 Python 包 + jinwurs abi3 矩阵）、Makefile、conda 配方（jinwu 主
   配方实 sha256；jinwurs 占位 sha256 显式标注 submit-ready 前）——与
   §19–§20 记录一致，无新发现。
4. **本地 ignored test/ 55 文件**：测试代码非生产代码，失败状态已历轮
   定位，逐文件质量审计与生产风险不成比例，维持不审边界；R149/R153 等
   修复时连带更新。

**结论**：至此仓库内所有生产源码（Python/Rust）、vendor 冻结件、CI/打包
配置、文档 API 契约均已完成至少一轮审查——**无未审代码残留**。开放清单
R79–R214 的修复决定仍待用户；修复须按 §21.4 忽略规则在已跟踪测试文件
补回归。

**证据**：`reviews/evidence/55-vendor-docs-ci-closure.txt`。

**AI 署名**：

- **GLM 主代理｜审查收尾（第十一轮）**：harness＝ZCode CLI；模型/推理
  档位＝运行环境披露模型标识 `account:bigmodel-start-plan/GLM-5.3-Flash`，
  推理档位未披露；记录时间＝2026-10-07T19:08:04+08:00；范围＝漂移检查、
  上游克隆与 _vendor 9/9 哈希核验及逐文件 diff 分类、docs 51 条导入机械
  核验（R214 发现与实际 API 运行时验证）、CI/Makefile/publish/conda 配方
  全读、本地 test/ 边界声明、证据 55 与本节及 Obsidian 撰写；结果＝
  R214 一项（Warning），vendor 溯源边界闭合，全仓审查覆盖完成；**未修改
  任何生产代码或测试，未提交/合并/推送**。

### 21.16 2026-10-07T20:28+08:00 弱覆盖基线模块深读（只读，R214–R248）

按"报告提及次数 ≤2"再甄别，发现六个基线模块的逻辑仅经 diff/局部/交叉
核对、从未系统审查（约 4.6k 行）。两个只读子代理串行深读 + 主代理独立
复核（R215 代码级、R227 运行时复现）。**新发现 R214–R248：11 Warning、
24 Suggestion，无新 Critical**；累计开放清单 **R79–R248，Critical 9 项
不变**。未修改任何生产代码或测试，未提交/合并/推送。

- **R214（W）** `docs/usage/bxa_fitting.rst`：文档教
  `from jinwu.core import WXT, FitConfig` 与 `WXT(fitting=...)`，均不存在
  （实际写法 `jinwu.core.config.instrument` + `dataclasses.replace`，主
  代理运行时验证）。51 条文档 import 中唯一断裂。
- **R215（W）** `core/instruments.py:625–652`：`_choose_related` 在头部
  显式引用的背景/ARF/RMF 找不到时静默返回不相关的唯一候选（仅记
  diagnostics，manifest 状态只看 warnings）→ bundle 以 ready=True 配错
  文件，下游 fit()/prepare_spectra 不读 diagnostics。
- **R216–R218（W）** instruments：多源 FXT 平铺目录零 bundle（与 WXT 按
  source_id 成束不对称）；单个不可读子目录使整次 scan() PermissionError；
  WXT manifest 可 "ready" 但零 bundle（ProductReport 分支无诊断）。
- **R227（W，主代理复现）** `gbm/response.py:40–43`：`nai_XX` 别名 1 基
  减一——`nai_05→n4`、`nai_00` ValueError；按 FITS/GDT 命名传参静默取错
  探测器响应（官方 NAI_00–09 为 0 基，对照 gbm_drm_gen）。
- **R228–R232（W）** bat_observation gdt-swift 缺失时 NameError（回退
  永不出现）；earthmap 真实 SAO 在 gdt-fermi 内崩溃且回退未 return；
  response 新产物判定文件名集合差（重跑误报）；SAO HTTPS 下载无超时；
  backprior 后验无符号/正性校验（负计数到采样才爆）。
- **R219–R226、R233–R248（S 24 项）**：含 units STmag 被当 Vega（错若干
  量级）、LightcurveSNREvaluator 无类型校验与先验敏感、nhtot 空单元格
  全废、滑窗失配假亏损、alert SSRF 多播/DNS rebinding 残余、backprior
  死参数与 EBOUNDS 按位置掩码、calibration 空 scores、bat_observation
  FOV 接缝/figure 泄漏/静默跳过初始化等。
- **已核实**：Gamma-Poisson 共轭 vs scipy.stats.nbinom（40 万样本一致）；
  Garwood FAR 区间 vs chi2（逐位）；alert URL 钉扎/重定向/哈希链安全
  语义（19 组用例全对）；units 换算 vs astropy（2e-16）；真实 WXT/BAT
  数据契约吻合。测试缺口：LightcurveSNREvaluator、ftselect、xselect_mdb、
  teldef 家族等仍无直接测试。

**覆盖结论**：提及次数 ≤2 的全部源文件处置完毕，**全仓 132 个 src
Python 文件 + Rust + vendor 无一处于"从未审查"状态**。本地 ignored
test/ 55 文件与 examples/ 维持既往边界。遗留为 R79–R248 的修复决定
（Critical 9 项优先）。

**证据**：`reviews/evidence/56-weak-coverage-modules-review.txt`。

**AI 署名**：

- **GLM 主代理｜弱覆盖基线模块深读（第十二轮）**：harness＝ZCode CLI；
  模型/推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T20:28:40+08:00；范围＝≤2 提及文件再甄别、两个子代理串行
  派发与复核（R215 代码级、R227 运行时复现）、docs 调用签名静态检查、
  examples 边界确认、证据 56 与本节及 Obsidian 撰写；结果＝R214–R248
  （11 Warning、24 Suggestion，无新 Critical），弱覆盖甄别处置完毕；
  **未修改任何生产代码或测试，未提交/合并/推送**。
- **/zcode/core-trio-deepread｜core instruments/units/utils 深读（只读）**：
  harness＝ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝
  运行环境披露模型标识 `GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T18:53+07:00；范围＝三模块全文 + 真实 WXT 数据契约 + 12 个
  最小可复现用例；结果＝R215–R226 与覆盖清单。
- **/zcode/periphery-baseline-deepread｜外围五文件基线深读（只读）**：
  harness＝ZCode CLI 子代理（general-purpose，串行）；模型/推理档位＝
  运行环境披露模型标识 `GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-06（+08:00）；范围＝backprior/response/calibration/alert/
  bat_observation 全文 + Gamma-Poisson/Garwood/alert 安全链运行时对照
  + 真实 SAO 全流程；结果＝R227–R248 与覆盖清单。

### 21.17 2026-10-07T21:38+08:00 测试套件审计与 .gitignore 漂移（只读，R249–R256）

生产代码面穷尽后，本轮收掉最后的内容边界：本地 `test/` 55 个测试文件
（27,398 行、1,370 个测试函数）的质量与覆盖映射审计（一个只读子代理），
examples/ 脚本与 docs conf/changelog 主代理过目。**新发现 R249–R256：
3 Warning、5 Suggestion，其中 R256 为新的跨工作流漂移**。累计开放清单
**R79–R256，Critical 9 项不变**。未修改任何生产代码或测试，未提交/
合并/推送。

- **R256（Warning，主代理发现）**：`.gitignore` 于 2026-10-06 17:18+07
  被并行进程重写——引入 `.*/` 通配并**删除 MVT/SNR/duration 工作流所加的
  三个 `!test/...` 白名单条目**。`git check-ignore -v` 证实
  `test/test_gbm_mvt.py`、`test/test_significance.py`、
  `test/test_txx_duration_corrections.py` 现被静默忽略（不在 git status、
  不在 git ls-files）——`git add .` 式提交将整体丢弃三个新测试套件。该
  重写本身是未经审查的工作区变更；§21.15 所记"三个测试白名单"自此不成立
  （更正）。提交前须恢复 `!` 条目或显式跟踪。
- **R249（Warning，子代理发现，主代理抽读证实）**：18 个测试
  （test_data_a.py 11 处、test_data_b.py 7 处）把含真实断言的整个测试体
  包进 `try: … except Exception: pytest.skip`（名义 matplotlib 守卫）——
  slice/rebin/group/grppha 的 ValueError 正是回归典型信号，却被吞成
  skip、套件保持绿色。修复 R79–R248 时这批测试会最先失效为假绿。
- **R250（W）** test_grppha 对分组委托零设防（仅 is not None）；
  **R251–R255（S）**：7 个 XSPEC 后端测试只断言 isinstance、恒真
  hasattr 断言、闰秒测试 try/except pass 无断言、15 处 is-not-None 单
  断言群、宽容差备查。不可能失败扫描 61 原始命中 → 3 真命中（58 误报
  系漏计辅助断言，已复核）。
- **开放发现覆盖映射**：10 项抽样中 **9 项缺陷路径完全无回归**（R171/
  R172/R190/R215/R227/R79/R80/R116/R117；R153 有一般对齐测试但缺陷路径
  缺失）——修复时每项都需新补回归；fake grppha 根因确证（文本占位 vs
  现在 FITS 校验，诊断 "No SIMPLE card found"）。
- **套件卫生**：CI 覆盖仅 21%（16/55 文件被跟踪，其余被 `/test/` 整目录
  忽略）；skip 分布正常；test_base_time roundtrip/test_ops_a 数值断言
  质量良好。
- examples/ 4 脚本（约 575 行）自带诚实状态标注、独立输出目录——维持
  本机教学材料边界；docs/conf.py（AUD-06 在位）与 changelog 无问题。

**结论**：测试套件审计后，仓库内容的审查边界全部收口——生产源码、
vendor、CI/打包、文档契约、测试套件、示例均已覆盖。R249/R250/R256 与
R149 同属"提交前必须处理的流程项"；修复 R79–R248 时逐项补跟踪回归
（覆盖映射表为直接输入）。

**证据**：`reviews/evidence/57-test-suite-audit-gitignore-drift.txt`。

**AI 署名**：

- **GLM 主代理｜测试套件审计与 .gitignore 漂移（第十三轮）**：harness＝
  ZCode CLI；模型/推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T21:38:49+08:00；范围＝漂移检查、测试套件审计子代理派发与
  复核（R249 计数与机制抽读、R256 gitignore 漂移独立发现与
  check-ignore/ls-files 证实）、examples/docs conf 快速过目、证据 57 与
  本节及 Obsidian 撰写；结果＝R249–R256（3 Warning、5 Suggestion），
  仓库内容审查边界全部收口；**未修改任何生产代码或测试，未提交/合并/
  推送**。
- **/zcode/test-suite-audit｜本地测试套件审计（只读）**：harness＝ZCode
  CLI 子代理（general-purpose，串行）；模型/推理档位＝运行环境披露模型
  标识 `GLM-5.3-Flash`，推理档位未披露；记录时间＝2026-10-06（+08:00）；
  范围＝55 文件 27,398 行 AST 扫描 + 16 核心文件抽查 + 10 项开放发现
  覆盖映射 + fixture 抽查；结果＝R249–R255 与覆盖映射表、套件卫生记录。

### 21.18 2026-10-07T22:22+08:00 Critical 全量亲验闭环 + 修复优先级矩阵（只读）

1. **九个 Critical 全部经主代理个人独立验证闭环**：R116/R117/R153/R171/
   R172/R190（八/十轮复现或代码链）与本轮补齐的 **R79**（当前代码复核：
   L1541 带守卫外层 `t0_met` 在 L1576–1580 alpha 回退分支被
   `_event_trigger_met(bkg)` 覆写，内联 None 守卫仅保护当次、覆写延续
   同 mode 后续段，与 L1676/L1682 目录原点换算并存）与 **R80**（当前代码
   + 安装版 BXA 4.5 solver.py：`create_flux_chain` 逐后验样本
   `set_parameters` 从不恢复，`fit_statistic` 在其后读取 → 默认
   `calculate_flux_chain=True` 时每次 BXA 拟合报告末个后验样本的统计量）
   ——R80 为活跃可达（每次 BXA 拟合默认触发），优先级应列 T1。
2. 覆盖疑点闭合：`gw/pipeline.py` 已有第九轮编排层实质审查（L312）。
3. **全量离线门禁复跑**：`4 failed, 1517 passed, 50 skipped, 6 deselected`
   ——与八/十二轮逐项一致，4 项失败归因不变。
4. **修复优先级矩阵**（证据 58，供修复决策）：T0 提交前流程项（R256
   白名单恢复、R149 基线重铸/重签、R249/R250 try-skip 清理与设防、fake
   grppha fixture、R214 文档、R73 跟踪回归）；T1 活跃可达 Critical
   （R153/R171/R172/R80/R117）；T2 潜伏 Critical（R116/R79/R190/R173）；
   T3 Warning 按模块影响面；T4 Suggestion 约 90 项随模块顺带。

**审查工作至此全部完成**：十三轮内容覆盖 + 本轮亲验闭环与决策矩阵。
待用户决定是否进入修复阶段。未修改任何生产代码或测试，未提交/合并/
推送。

**证据**：`reviews/evidence/58-critical-reverify-priority-matrix.txt`。

**AI 署名**：

- **GLM 主代理｜Critical 亲验闭环与修复矩阵（第十四轮）**：harness＝
  ZCode CLI；模型/推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T22:22:30+08:00；范围＝R79/R80 当前代码亲验（R80 含安装版
  BXA solver 源码核验）、gw/pipeline.py 覆盖疑点闭合、全量门禁复跑、
  R79–R256 修复优先级矩阵（证据 58）与本节及 Obsidian 撰写；结果＝
  9 Critical 全部个人独立验证闭环、矩阵交付；**未修改任何生产代码或
  测试，未提交/合并/推送**。

### 21.19 2026-10-07T23:39+08:00 XSPEC 测试层首跑 + docs 环境核验（只读，R257）

十四轮以来所有门禁均排除 heasoft/real_data 标记。本轮 `source heainit.sh`
后**首次运行 XSPEC 依赖测试层**（`-m 'not network'`）：**6 failed, 1,550
passed, 18 skipped**——此前从未运行的 test_bxa_fit heasoft 标记测试、
test_fit_b 的 7 个 XSPEC 后端 fit_prepared 测试、test_gbm_pipeline_stages
的 PyXspec profile fit **全部通过**；离线门禁的 4 项失败照旧。

2 项"新增"失败逐一诊断：(1) test_txx_iterbkg_validation real_data
（EP260809a）——`ops.py:1003` 新增的"正计数箱必须具有正曝光"fail-loud
守卫触发，即 duration 报告（R141 处）声明**"用户决定暂缓"的 R3 iterbkg
稀疏 OFF** 问题，实施者披露准确，旧实现静默错误数值 vs 新守卫显式报错；
(2) test_headas_environment_is_stage_local——清洁环境 passed、全局
heainit 后失败，属测试-环境交互；但暴露 **R257（Suggestion）**：
`_headas_environment` 钉扎 HEADAS/PFILES/HOME/PATH 却不覆盖继承的
LHEASOFT。另记录：运行尾部成批 XSPEC/pgplot "YMIN=YMAX" 退化窗口输出
（测试通过，待定位，不设编号）。

**docs/usage/bxa_fitting.rst 声明核验（最复杂 435 页全文）**：环境声明
全部属实（jiasui 环境存在；hea 的 init 路径可用；构建日志证实 hea=
**HEASoft 6.36**——**更正**第十/十二轮子代理把 hea 环境写成 6.37 的归属
错误，6.37 为 external_sources 对照副本）；API 声明全部核验通过
（fit_prepared_bxa 全参数、BXAFitResult 属性、UpperLimit.from_chain、
BXAPriorSpec 四 kind、占位类存在）；**R214 修复路径精确化**：WXT/
FitConfig 实际在 `jinwu.core.config`，`WXT(fitting=FitConfig(...))`
写法可用，仅 import 来源错。

**证据**：`reviews/evidence/59-xspec-layer-first-run.txt`。

**AI 署名**：

- **GLM 主代理｜XSPEC 测试层首跑与 docs 核验（第十五轮）**：harness＝
  ZCode CLI；模型/推理档位＝运行环境披露模型标识
  `account:bigmodel-start-plan/GLM-5.3-Flash`，推理档位未披露；记录时间＝
  2026-10-07T23:39:13+08:00；范围＝heainit 后 XSPEC 层首跑（1,550 通过）
  与 2 项失败诊断（R3 暂缓项确认 + R257 发现）、bxa_fitting.rst 全文
  声明核验（环境/初始化/版本/API）、HEASoft 版本归属更正（hea=6.36）、
  R214 修复路径精确化、证据 59 与本节及 Obsidian 撰写；**未修改任何
  生产代码或测试，未提交/合并/推送**。

### 21.20 2026-10-08T00:16:18+08:00 仅补审没有详细记录的内容（Codex）

用户将本轮限定为“只需要审查没有审查记录的那些”，并明确“测试需要被gitignore”。既有算法与 16 份 review Markdown 已有记录，本轮只核对覆盖映射及文件哈希，不重复审查其行为。补齐证据 53 的函数说明准确性边界与证据 57 的示例详细内容边界：16 模块/246 个函数说明逐条核对，未发现新的事实性错误；当前 12 个脚本和 13 个 Notebook 代码审读与语法检查，10 组历史迁移共享代码 AST 相同。额外 duration_recalculation Notebook 的源码只比已登记 exploration 多一个 FITS 导入单元，没有新增算法；历史执行状态另有差异，不视为重算验收。

真实 EP250615a 光变只读验证具体确认旧示例恢复前的两项限制：E01，当前 `netdata` 不接收 `label`、返回对象没有 `.data/.area_ratio`、Swift 时间格式为 `swiftmet`；E02，BAT 示例从读取器相对时间直接减绝对 MET，遗漏保存的原点。示例原先已标为 `legacy_analysis`，本轮未将这些历史限制升格为迁移引入的生产回归，没有新增 R 编号。完整范围、位置、数据哈希、实际输出和未验证边界见 [本轮正式补审报告](codex-reviews-vs-origin-master-20261007.md)，证据位于 `reviews/evidence/worktree-review-20261007/`。

**当前政策更正：R256 按用户要求关闭。**保持 `/test/`、`/tests/`、`/testrs/` 的忽略规则，不恢复三个测试白名单、不强制跟踪新测试；§21.17/§21.18 关于必须恢复白名单的建议不再适用于当前工作。历史段落保留，不继续将此项列为缺陷或提交门禁。已有其他开放发现的状态不由本轮改变。

独立 review 署名：Codex 主代理 `/root`；harness＝Codex desktop；具体模型标识＝未披露；推理档位＝未披露；记录时间＝2026-10-08T00:16:18+08:00；范围＝覆盖缺口补审与测试忽略政策更正；结果＝无新增生产缺陷，旧示例限制 E01/E02 有真实输入证据，R256 关闭；证据＝以上报告及诊断文件。**未修改生产代码、示例、测试或 `.gitignore`，未提交/合并/推送。**

### 21.21 2026-10-08T00:45:41+08:00 剩余教程与包内测试覆盖缺口补审（Codex，R258–R264）

继续检查证据 55/57 的具体边界：文档主要做过导入检查；根目录 `test/` 的质量审计不含包内 Swift 测试。只补齐此前没有详细记录的教程调用/单位、独立图脚本、2 个包内示例与 4 个包内测试；未重新通审已有记录的算法。§21.20 的“无新增生产缺陷”限于那一批说明/历史示例，不是当前所有教程与测试契约的结论。

| 新发现 | 优先级 | 已复现的具体行为 |
| --- | --- | --- |
| R258 | P1 | `core/fit.py:897–898` 无条件成功：教程拟合模型与 χ² 均 NaN 仍 success；独立 TRF 求值耗尽、后端 success=False 时公共结果仍 success=True |
| R259 | P2 | `swift/grb/pipeline.py:302–307` 将 14 列表的 flux error 取为 ECF；包内 fixture 也用错列；GRB 140614A 真实 WT/PC 表和同仓库另一读取器证实 rate/error 换算错误 |
| R260 | P2 | `usage/lightcurve.rst:50–52` 对 `(Axes, Axes)` 错误嵌套解包，无法保存图；示例的 bounds 也需同时核对 |
| R261 | P2 | `redshift_extrapolator_flowchart.py:62/74/104` 未限定普通幂律的 norm 缩放；zpowerlw 合成反例重复计入红移，谱为正确值的 0.25 |
| R262 | P3 | 同一图脚本 `:108–110` 写不存在的旧 autohea 项目路径；仅输出重定向的诊断渲染完成 |
| R263 | P2 | `usage/nhtot.rst:47–48` 在“用于 XSPEC”例子漏掉 cm⁻² → 10²² cm⁻² 转换；本次实际公开查询为 5.26e19 cm⁻²，应对应 0.00526 |
| R264 | P2 | `quickstart.rst:65–67` 假定 PHA 自带 EBOUNDS；真实 EP260119a FXTB PHA 没此扩展，教程调用 TypeError；配套 RMF 有有效边界 |

上述文件当前与 `origin/master` 一致，属于现存问题，不是 `codex/reviews` 已提交回归。R258/R259 是从未审使用路径触发的直接生产落点，不重复 R117/R121 或重新验收整模块。详见 [正式报告](unreviewed-code-science-20261008.md)：逐项位置、方法与单位、原始资料、反例、真实输入 SHA256、适用边界和建议均已记录。

验证：13 RST 页/33 Python 块语法通过，92 个可静态解析调用可绑定、64 个动态调用未静态验收；Markdown 6 块解析，4 个当前吸收预算块在 hea/XSPEC 12.15.1 下运行并核查独立 JSON/CSV/图件，2 个明确历史块不执行。真实 WXT 读取/净光变/拟合/绘图调用链完成，保留负净值；该单幂律不是物理上合适的突发拟合。Swift/FXT 完成字段与故障路径验证，没有伪称完整管线验收。门禁仍有已记录失败，没有普遍“无 bug”结论。

证据：`reviews/evidence/worktree-review-20261007/{doc-code-contracts,doc-science-diagnostics,additional-doc-diagnostics,swift-grb-column-diagnostics,markdown-doc-diagnostics,gap-coverage-manifest}.{py,json}`。测试按用户要求继续忽略；R256 关闭状态不变。早期 131 文件快照的非 review 文件无变化。

独立 review 署名：**Codex 主代理 `/root`**；harness＝**Codex desktop**；具体模型标识＝**未披露**；推理档位＝**未披露**；记录时间＝**2026-10-08T00:45:41+08:00**；范围＝未记录的教程/包内测试/示例与图脚本；结果＝R258–R264 七项未修复发现；证据＝正式报告及上述诊断。无子代理参与，**未修改生产代码、示例、测试或 `.gitignore`，未提交/合并/推送**。


### 21.22 2026-10-08T11:45:22+08:00 未提交批次审查、科学修复与 0.2.2 合并验收

用户指定重点审查文档，并明确授权包含全部可见未提交变更；审查后自主修复明显影响算法/物理正确性的内容，再更新版本、合并 master 并 push。起点为 `codex/duration` / `08fd6a614ce6976e9001aa8c24bab37800e54f25`；最终范围包括原有文档、双语说明、T90/SNR/MVT、仪器及谱准备改动和本次修复。忽略的本地数据、测试、产物没有强行加入 Git。

**审查结论：没有剩余需要阻止本批次合并的文档问题。** 下表是当前源码重新确认、按用户额外授权修复的现存生产缺陷；不能将它们误称为原文档批次新引入的问题。历史审查原文保留，本节覆盖这些编号的当前状态。

| 已修复编号 | 落点与原场景 | 修复及验证 |
|---|---|---|
| R117 / R258 | `core/fit.py` 光变误差为零/NaN，或 Astropy 求值耗尽/非有限模型 | 拒绝无效误差，核验后端 success 和有限统计量；失败误差为 NaN。合成边界与真实 WXT 有符号光变拟合通过 |
| R116 | `_prepared_error_parameters` 中 delta=3 被格式化为整数，XSPEC 误读为参数编号 | 总是使用实数字面量；真实 FXT profile 命令 `3.0 2 3`、误差计算和图件成功。语法依据 [XSPEC error](https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSerror.html) |
| R80 | `core/bxa_fit.py` 通量链访问样本后导出随机样本状态的统计量/会话 | 成功或失败均恢复 best fit；恢复失败直接传播。真实 FXT 5230 个通量样本后参数、统计量与导出一致；失败分支回归通过 |
| R259 | `swift/grb/pipeline.py` 14 列表把 flux 的 ECF 误差列当成 ECF | 第 12 列换算 rate，非正/非有限 ECF 为不可用；GRB140614A 的 WT/PC 实表与 flux/ECF 逐项一致。依据 [UKSSDC schema](https://www.swift.ac.uk/API/ukssdc/data/GRB.md) |
| R79 | Swift BB 面积回退覆盖源触发时间，背景没有 TRIGTIME 时崩溃 | 背景时间使用独立变量；回归证实源触发参考下区间 1–4 s 保持正确 |
| R153 | `datasets.netdata` 大 MET 默认相对容差放行错位背景，或 rebin 后直接相减 | 严格检查 TIMEZERO、中心/边缘；只允许完整覆盖且不跨源边缘的聚合，之后再次核验。真实 WXT 同网格通过、177 个负净 bin 保留；错位/宽度/时间零点回归通过 |
| R171 | `ftools/rmf_mapping.py` 稀疏分支用全通道掩码索引通道子集 | 按唯一通道索引回填；重复/越界/空响应回归及真实 FXT RMF 的 sparse/dense 对照通过 |
| R172 | `ftools/ftselect.py` 比较和逻辑优先级在替换运算符时丢失 | 安全 AST 逐元素求值，支持括号、not、链式比较和科学计数法；真实 EP 事件的 PI 50–400 与直接掩码一致 |
| R173 | TELDEF with_pointing 两方向都用转置，倾斜平面亦缺投影归一化 | 求逆旋转并与反向配对，投影到固定焦平面；合成倾斜矩阵及真实 XRT 标定矩阵的方向往返通过。实际 Swift 姿态/探测器视场仍需完整 HEASoft 转换，不把此测试作为 aspect solution 验收 |
| R190 | `lf/lcfake.py` / detectability 将 ON 总计数当信号缩放后再次加背景 | 优先显式净计数，否则扣 OFF 背景或显式输入背景；目标背景只添加一次；真实 WXT 源模板保留负计数、背景专用 ON/OFF 回归通过。Poisson 期望阶段才截断负值；不构成背景联合推断 |

文档同步描述上述异常与迁移要求；历史红移算法、模型条件性、Koshut 未解决交点等边界保留。新增辅助函数已在 `REUSABLE_FUNCTIONS.md` 登记。文档资产检查器复用已跟踪的 API 生成器，检查生成页面的源码输入，避免要求提交构建产物。

**验证（本次重新执行）：**

- 五个 Python 包均构建 sdist/wheel 为 0.2.2；五个 wheel 的 Python 源与最终工作区逐字一致，SNR 许可证/来源和 MVT NPZ 资源齐全。可选 Rust 包及 conda 配方版本同步；核心配方 SHA256 匹配本地 sdist。没有上传 PyPI，没有推发布 tag。
- 独立 Python3.12 venv 安装 wheel，`pip check` 无依赖缺口；发布/CI 的受控离线范围 **380 passed, 1 skipped, 1 deselected**。跳过项为未安装可选 gwpy 的时间测试；network/heasoft/real_data 不在该离线范围。hea 的额外 BXA 本地测试 **153 passed, 1 skipped**，最终修复目标检查 36 passed（后续新增 Swift 触发回归包含在 380 中）。首次发现的本地旧谱准备夹具写入非 FITS 文本未纳入上述验收；真实谱分组/拟合替代该夹具路径的验收。
- 独立环境 Sphinx8.2.3 严格全量构建零警告；212 HTML 页，25006 内部链接/4064 资源引用均有效，84 模块/571 公共定义锚点齐全。源码编译与 `git diff --check` 通过。
- 真实 WXT11900273154、EP260119a 11900560514、FXT06800001128、GRB140614A 输入哈希保留。FXT 经过真实 grppha 分组→RMF/辅助文件核查→MLE/profile→图件/报告/会话→BXA 通量链/状态导出，18 个本次受保护输入未改变。拟合仅用于回归诊断，不发布新科学结论。
- EP260119a duration 完整事件路径四种输入/累计设置、每种30次 MC：T50≤T90≤T100 及有符号累计/窗内交点检查通过；未解决 Koshut 交点仍明确标记，不冒称有效科学区间。
- SNR 全量参考对照：353 等价案例、10 故障守卫、四组真实数据；最大 Z 差 6.602092305751534e-13。GBM170817 在1 ms和0.1 ms两档的观测结果/中间量及每档2个重采样与冻结上游精确一致；不是亮源最细档全部300次重算，也不是通用测量可靠性证明。

本地原件：`reviews/evidence/release-0.2.2-20261008/`，含源码快照、wheel/HTML 检查、`run-initialized/real-fixes.json`、真实谱产物、SNR/MVT 与 duration 结果；终端日志位于 `/tmp/jinwu-*-gate*.log` 等。环境差异：hea 激活后当前 sandbox 会话未初始化 HEASoft，正常初始化被用户目录写权限阻止；允许正常初始化后跑通，临时 PFILES 保存在 `/tmp/jinwu-review-pfiles`，未替换用户 HOME。

适用规则核验：[AGENTS.md:59–69](../AGENTS.md#real-data-validation) 要求真实流程及明确未验证范围；[AGENTS.md:75–83](../AGENTS.md#code-migration-acceptance) 要求迁移来源、相同输入与差异说明。本次算法修复独立于未改变的 MVT/SNR 移植核心记录，不将局部回归推广为普遍科学正确。

- **原批次独立审查署名**：Codex 主代理 `/root`；harness＝Codex desktop；运行环境实际模型标识＝未披露；推理档位＝未披露；记录时间＝2026-10-08T11:45:22+08:00；范围＝全部可见未提交变更，重点文档与代码契约；结果＝已修复事项见本节，未证实新阻断文档问题。
- **修改及验收署名**：Codex 主代理 `/root`；harness＝Codex desktop；实际模型/推理档位＝未披露；记录时间＝2026-10-08T11:45:22+08:00；范围＝上述科学修复、回归、文档/版本/打包及进度记录；结果＝本节证据。由同一代理完成修复验收，不称为第二名独立 reviewer；无子代理参与。

补充卫生核验：暂存新增文件后发现生成 SVG、迁移脚本/旧指南与 vendor 的历史尾空白；仅清理空白，可执行 Python AST 逐文件一致，Notebook 未改。来源 hash 保留，迁移清单补充最终 destination/script hash，MVT integration_changes 明记空白清理；随后重新构建受影响 wheel。
