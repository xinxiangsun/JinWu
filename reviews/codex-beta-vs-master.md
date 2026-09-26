# codex/beta vs master 变更审查报告

> **2026-09-25 实施状态**：下文第 1–18 节保留各轮审查的历史快照，不能作为当前未修复清单或最终验收结论。合并前的修复、真实数据验证和未验证范围以文末第 19 节为准；早期“所有问题均为约定/文档层面”的判断已被 R49–R58 等后续实测推翻。

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
