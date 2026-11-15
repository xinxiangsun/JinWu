# GECAMTools 与共享源分析验证

当前状态（2026-11-15）：GECAM-B 官方样例已覆盖产品提取、成品转换、PG-stat 联合谱
拟合、光变与恢复；共享层已验证逐谱统计量、参数链接和跨历元编排。以下各节分别记录
验证范围及证据，流程完成不代表科学接受。实际 GECAM-A/C 与 EP–GECAM 同源同时窗
科学验收尚未完成，样例响应有效面积及谱形限制仍保留。代码未提交、推送或发布。

## 历史记录：初始接口验证（2026-11-02）

范围：新增仪器包与上游接口的等价性、阶段恢复、安装和文档；不作科学接受声明。

## 来源与环境

- 官方源码入口：<https://github.com/zhangpeng-sci/GECAMTools-Public>。
  使用其链接的 [20240514 发布包](https://ihepbox.ihep.ac.cn/ihepbox/index.php/s/x6HkDidQ8eCvrrG)，
  外层 ZIP SHA256：`f7a93898556f42ed8fd2b884f337fc8ea6d67961414444fddc23a2ee1964e932`。
  原样安装，未修改上游算法或依赖源码。
- 官方手册样例：[GRD EVT 下载](https://ihepbox.ihep.ac.cn/ihepbox/index.php/s/iEE9AZQyNuJljr0)，
  文件 `gbg_evt_tn210511_112749_fb_v00.fits`，触发标识 `tn210511_112749_fb`。
  SHA256：`69c5e301c8ce834c79bf8f8e2c8ef3336bca63010be11e9a2806fdefe22b7b59`。
- Python 3.12.15；NumPy 2.4.6、SciPy 1.17.1、Astropy 8.0.1、lmfit 1.3.4。
  从本机 `hea` 的 Python 建立 `/tmp/jinwu-gecam-env`，共享其科学依赖，
  仅在这个临时环境安装 GECAMTools 和 editable 插件；没有修改 `hea` 安装。

## 已完成

- 两个高增益探测器 GRD17/18 完整运行 `inspect/lightcurve/spectra/report`。
  MET 窗口 `[74431550.6, 74431660.6] s`，光变步长 `0.5 s`，请求能量边界
  `[40, 100, 300] keV`；实际边界保存在 NPZ。
- 源区间 `[74431599.6, 74431609.6] s`；本底区间
  `[74431560.6, 74431580.6]` 与 `[74431625.6, 74431660.6] s`。
  谱提取步长 `1 s`、通道分组 `4`、背景阶数 `1`，未绑定响应。
- 独立直接调用官方 API，与适配层进行 **114 项数组逐元素完全相等**比较，
  包含原始/修正计数及误差、时间边界、谱文件全部表列（曝光、质量、分组、
  背景/响应关联等）。谱计数和误差有限、曝光为正。无随机步骤，容差为 0。
- GRD17 低增益另做 6 项数组完全相等比较，同时检查 `500 ms → 0.5 s`
  与 `[0.4, 1, 3] MeV → [400, 1000, 3000] keV` 的单位转换。
- 真实输入恢复时不调用光变/谱提取，阶段 manifest 修改时间不变。
  离线回归检查部分运行状态、改变输入/配置后的缓存失效及旧产物保留。
- 新插件与共享 pipeline/代码依赖测试：22 passed。
- 全库现有离线门禁：1706 passed、50 skipped、6 deselected；包括新插件测试。
- wheel 与 sdist 构建成功；独立临时安装环境在未安装 GECAMTools 时可导入插件、
  发现 `gecam` entry point，插件的 9 项离线测试通过。
- Sphinx 严格全库构建通过。此机已安装的 ligo/reproject/xarray 链会与 Sphinx
  的 pandas mock 冲突，验证时先导入 pandas、xarray、ligo.skymap.plot，再启动
  Sphinx；没有修改 Swift 实现或全局环境。

本地可复现证据位于仓库忽略目录 `local_private/gecam_validation_20261102/`：
`inputs/`、官方发布 ZIP、`run_validation.py`、`compare.py`、`comparison.json`、
`low_gain.json`、`run/` 以及 `direct_upstream/`。在具有上述依赖的环境中运行
`python local_private/gecam_validation_20261102/compare.py` 可重做高增益完整对照。
输入与旧运行产品均保留。

## 初始接口阶段未验证范围（2026-11-02 历史）

以下是 11 月 2 日截止状态；后续响应生成、谱拟合与共享分析验证见后续各节。

GECAM-A/C 实际观测、CPD/并道数据、CALDB 响应生成、响应校准/谱拟合、T90，
以及上游背景算法和死时间处理在一般情形下的科学准确性均未验收。
流程输出继续标记 `science_status=unvalidated`；接口等价不代表独立科学验证。
本次未提交、推送或发布。

署名：Codex /root；harness：Codex desktop；实际模型标识/推理档位：未披露。
记录时间：2026-11-02T17:44:52+08:00。范围：接口实现与上述自检，非独立代理审查。

## 成品分析与响应对照（2026-11-10）

沿用上面的公开 GECAM-B EVT 和 GRD17/18 高增益提取设置。响应由 GECAMTools
20240514、官方 `CALDB_v1.0_Sep2024` 和配套姿态文件
`gb_posatt_tn210511_112749_v00.fits` 生成。使用触发 MET `74431600.6 s`、事件定位
RA `317.99°`、Dec `59.53°` 和对应卫星姿态。生成命令与警告见
`local_private/gecam_analysis_20261104/generate_response.py` 和 `response.log`。
复验使用 `hea` 的 Python 3.12.15、PyXspec 2.1.5、XSPEC 12.15.1，以及从 20240514
发布包解压的原样源码。

- EVT 与独立 GECAMTools 成品两条路径均完成，输出四条光变、一个共同时间窗的联合谱和
  一个通量时间箱。两条路径的参数、统计量、通量和能段一致。恢复时各阶段清单和产物路径
  未改动。缺 GRD18 响应时返回 `needs_review` 并写出四条光变，补入响应后恢复并完成拟合。
- Type-I 拟合副本逐项对照原始 type-II 第 1 行的通道、计数、误差、分组、质量、曝光，
  确认 `POISSERR=True/False` 分别赋给 full 谱和模型本底。原始文件保持不变。
  原生谱 `TIME` 的左边界语义见 GECAMTools 20240514 `gecam/data/spec.py` 的
  `_ensemble_spec_data`，同一文件的 GTI 亦写入区间起止。两条路径的 UTC 区间在 `1 µs`
  内一致，约为 `2021-05-11T11:26:39.6` 至 `11:26:49.6 UTC`。EVT 读取 FITS TT 时钟，
  成品记录采用 GECAM-B 2019 UTC 零点的假设。
- 四条总计数率光变均为 220 箱。GRD17/18 的实际能段均为约 `39.118–98.484` 和
  `98.484–301.731 keV`。计数率和误差分别与原生计数及误差除以实际箱宽逐项相等。
- `cflux*powerlaw` 最大似然拟合链接两探测器的 `lg10Flux` 和 `PhoIndex`，固定幂律归一化为
  `1`。各使用 31 个分组后的通道，`pgstat` 约为 `73.90 + 76.66 = 150.56`，自由度 `60`。
  请求的观测者系通量能段 `40–300 keV` 超出实际采用的约 `41.244–286.040 keV`，产物将
  通量标为模型外推。残差偏大，且 CALDB 在该 MET 无两探测器的经验有效面积修正记录，
  因此本例仅用于流程验证，科学状态为 `requires_review`。

### 已核对的样例响应编码

两份官方 RSP 的矩阵均为 `300 × 498`，每个入射能量箱只有一个完整通道组。
16 位求和溢出使 `NUMELT` 写成 `18328`，正确总数为 `149400`。原文件的 `F_CHAN` 为 1，
缺失 `TLMIN4`，而 EBOUNDS 和 PHA 通道号为 0–497。GRD17/18 各有 50 个原生 `QUALITY=1`
的溢出通道，PHA 将其能量限填零，其余 448 个可用通道的能量边界与响应一致。
JinWu 的准备阶段会拒绝此原始索引冲突。以下为这一确切布局的显式规范化示例，输出独立副本。
字段定义见 [HEASARC OGIP CAL/GEN/92-002](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.pdf)。

```python
from astropy.io import fits
import numpy as np

with fits.open("gbg_17H_trigger.rsp") as hdus:
    matrix = hdus["SPECRESP MATRIX"]
    channels = hdus["EBOUNDS"].data["CHANNEL"]
    rows = matrix.data
    if not (
        np.array_equal(channels, np.arange(498))
        and rows["MATRIX"].shape == (300, 498)
        and np.all(rows["N_GRP"] == 1)
        and np.all(rows["N_CHAN"] == 498)
        and np.all(rows["F_CHAN"] == 1)
        and matrix.header.get("NUMELT") == 18328
        and "TLMIN4" not in matrix.header
        and matrix.columns.names[3] == "F_CHAN"
    ):
        raise ValueError("The RSP does not have the documented GECAM-B sample layout")
    matrix.header["NUMELT"] = int(rows["N_CHAN"].sum(dtype=np.int64))
    rows["F_CHAN"][:] = 0
    matrix.header["TLMIN4"] = 0
    hdus.writeto("gbg_17H_trigger_ogip.rsp", checksum=True)
```

真实流程使用上述规范响应。另在 XSPEC 中逐个换入原始、仅改 `NUMELT`、以及同时对齐通道
索引的响应做对照。三组重拟合的 PG-stat `150.55594386463682`、`60` 自由度、两个自由参数
及两探测器各 31 个分组箱的折叠预测均完全相等，容差为 0。此等价性限于样例、指定响应和
XSPEC 12.15.1。修改响应编码不涉及经验有效面积校正。

复验入口为 `python local_private/gecam_analysis_20261110/run_analysis.py`，随后运行同目录
`compare_analysis.py`。`inputs.sha256` 记录输入指纹，`comparison.json` 记录数值对照，
`raw_pipeline_rejected.log` 记录准备阶段对原生索引冲突的诊断。图与表的路径可从
`evt_fit/gecam_report.json` 和 `finished_fit/gecam_report.json` 的 `analysis` 阶段定位。
插件定向测试 13 项通过，见 `target-tests.log`。实际 GECAM-A/C 及跨仪器科学验收仍待完成。

署名为 Codex /root/gecam_finalize，harness 为 Codex desktop，实际模型标识和推理档位未披露。
记录时间为 2026-11-10T09:12:00+08:00。范围包括时钟及响应关联检查、使用文档、插件定向测试和 GECAM-B 流程复验。

## 时钟契约收尾自检（2026-11-15）

- 确认 full/background 都有 FITS 时钟时比较的是换算后的 MET 零点，容差 `1 µs`，
  不是仅比较 `TIME` 数组。调用方显式指定零点也不能掩盖两份原生时钟的冲突。
  本底没有时钟时仍可使用，但每条准备谱的诊断明确记录假设与 full 共用零点。
- `hea` 中以当前源码运行 `test_gecam.py`、`test_gecam_analysis.py`：13 passed。
  扩充覆盖相同 `TIME` 但零点相差 2 s 的拒绝、显式覆盖不能绕过冲突、等价的缺省
  `TIMEZERO=0` 可接受，以及本底缺时钟的假设记录。运行时通过 `PYTHONPATH` 指向
  `packages/jinwu-gecam/src:packages/jinwu/src`，未修改环境安装。
- 重新核对 11 月 10 日 `inputs.sha256` 的九项输入，均相符；读取原阶段指纹及数值对照。
  将同一官方 GRD17/18 成品及规范响应在临时目录重新准备，与 `finished_fit` 既有产品逐列
  比较：PHA、本底、响应表数组完全相等，曝光/Poisson 标志/缩放不变，UTC 边界差小于
  `1 µs`，统计仍为 `pgstat`。临时产物已清理，原始数据和既有结果未改动。
  本次只补诊断与回归覆盖，未重新拟合；上节真实拟合和科学限制仍适用。

署名：Codex `/root/gecam_acceptance`；harness：Codex desktop；实际模型标识/推理档位：未披露。
记录时间：2026-11-15T00:50:48+08:00。范围：原生时钟契约自检、诊断/测试/文档收尾，非独立科学验收。


## 共享谱核心、真实 EP 与最终门禁（2026-11-15）

谱核心在激活的 `hea` / XSPEC 12.15.1 中完成 5 项定向回归，覆盖 mixed W/PG、
交叉校准、模型网格、XCM 回放及 OGIP correction 文件的显式拒绝。
真实 GRD17/18 各有 31 个 noticed grouped bins，独立 PG 计算与 XSPEC 差小于
`5e-9`。每份本底有两个 `STAT_ERR=0` 的箱，此处只验证 sigma→0 已知背景极限
的数值行为，不据此证明实际背景已知，未改动原误差。2000→4000 个对数模型箱的
Δstat 为 `-1.2977532e-4`、ΔlgFlux 为 `1.2225073e-6`、ΔGamma 为 `-4.0989285e-6`。
XCM 文本参数舍入使回放为近似等价；最大参数绝对差约 `4.72e-6`、折叠预测相对差约
`1.40e-5`、统计量绝对差约 `1.32e-4`，原会话与回放分别再优化均保持各自结果。
容差、完整快照与控制结果见
`local_private/gecam_analysis_20261104/spectral_core_validation_20261110/run-4corx7x4/validation.json`，
回归产品保留在同级 `pytest_products_20261115/`，日志为 `validation_20261115.log`。
科学状态继续为 `requires_review`。

共享源分析使用 EP260119a WXT `11900560514` 提示期及 FXT `06800001128` 后随期，
两者分别拟合；FXTA/B 同 epoch，采用 10 ms 容差，实际起点相差 6.385 ms。
`source_validation/run-9n1dt43w/validation.json` 记录与直接调用完全相同的参数、统计量、
通量、`1e-7 s` UTC 对齐和恢复一致性。本次重算其 19 个输入 SHA256，全部相符。
WXT 三箱 Gaussian linear 拟合与直接 `LightcurveFitter` 的参数、误差、协方差和
chi-square 相等，但 dof=1，仅用于接口集成验证；FXTA 含零误差，保留图表而不运行该
Gaussian 拟合。此 EP 验证不包含 GECAM，也不构成 EP–GECAM 同时窗科学验收。

最终门禁证据位于 `local_private/gecam_analysis_20261104/gates_20261115/`：

- `offline.log`：**1718 passed、51 skipped、6 deselected**。选择与 CI 相同的
  `not network and not heasoft and not real_data`，含根测试、既有 Swift 和 GECAM 测试。
  使用共享 `hea` 科学依赖的独立 Python 3.12 环境；只将此前持久保存的官方
  `swclockcor20041120v175.fits` 放回 `/tmp/swiftclock` 原搜索路径，未改 Swift 代码或测试。
- `docs.log`：严格 Sphinx `-W --keep-going` 构建成功。预加载 ligo/reproject 时有第三方
  Astropy 弃用警告；未将整个日志描述为零警告。
- `wheels.log`、`wheels/`：六个 Python 包的 0.2.4 wheel 构建成功并记录 SHA256；
  `wheel-test.log`：独立环境安装、`pip check`、五个仪器 entry points、导入来自安装目录、
  不安装 GECAMTools 的惰性导入全部通过；定向回归 **46 passed、1 skipped**。
  唯一 skip 是此隔离环境没有 PyXspec，真实引擎验收见上文。未构建或验收 Rust wheel。
- `check_gate.sh`、`build_docs.py`、`smoke_wheels.py` 是复现入口；
  `offline-env-freeze.txt` 与 `wheel-env-freeze.txt` 保留依赖差异；
  `source_hashes.json` 确认最终门禁期间相关生产源码与测试未变化，
  `source_input_hashes.json` 保留 19 个 EP 输入的本次核验。

以上为限定工程与数值等价验收；实际 A/C、跨仪器同源同时窗、背景方法的一般科学准确性、
CALDB 样例缺经验有效面积记录及残差偏大等限制仍保留。原始数据与既有结果未覆盖。
本任务未提交、推送或发布。

署名：Codex `/root/final_gates`；harness：Codex desktop；实际模型标识/推理档位：未披露。
记录时间：2026-11-15T17:50:08+08:00。范围：共享源文档与复用索引、输入指纹、最终工程门禁、证据汇总；非独立科学接受。

审查署名：Codex `/root`；harness：Codex desktop；实际模型标识/推理档位：未披露。
记录时间：2026-11-15T17:50:08+08:00；11/04、11/10 历史实际工作时间不详。范围：显式 epoch/UTC、PG/W 统计、
参数链接、共同能段外推和输入保护审查，11/10 联合谱双色图与光变目视，以及最终门禁核查。
