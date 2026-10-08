# REUSABLE_FUNCTIONS — jinwu 公共 API 索引

> jinwu 是跨项目**可复用库**（monorepo：`packages/jinwu` 为核心，`packages/jinwu-ep`、
> `packages/jinwu-fermi`、`packages/jinwu-swift` 为按仪器拆分的发行包；editable 安装，
> 改源码即时生效）。自 0.2.0 起顶层只有命名空间，无便捷导出——**一律从 `jinwu.core.*`
> 导入**（如 `from jinwu.core import Time, read_pha`）。
> 本索引列出高频公共入口，供所有 AI 代理（Codex / Claude / Hermes）在写任何分析
> 代码前快速定位："库里已有 X，别重复造"。
>
> 规则：
> 1. 先查本表 + 模块 docstring + 测试，再写新代码。
> 2. 新增公共函数 → 在文末「登记区」追加；已有函数不改签名时无需条目。
> 3. 提升新函数：遵守 `ARCHITECTURE_ROADMAP.md` 分层（`core` 不导入任务包/HEASoft/XSPEC 等
>   外部运行时；任务相关放 `packages/jinwu-<mission>`，经 `jinwu.instruments` entry point
>   被 pipeline 注册表发现）。参考现有 `jinwu.core.fit`（XSPEC 调用封装为函数）与
>   `jinwu.core.heasoft` 的边界方式。

## core 层高频入口

| 模块 | 入口 | 用途 |
|---|---|---|
| `jinwu.core.utils` | `nhtot(ra, dec, equinox=2000, timeout)` | 银河 NH / E(B-V) 查询（EP 的 `resolve_galactic_nh` 包装自它） |
| | `li_ma_snr(n_on, n_off, alpha, signed)` | Li-Ma (1983) 显著性 |
| | `generate_xspec_result(model, spectrum)` | XSPEC 拟合结果结构化 |
| | `get_asym_err` / `flux_err_from_log10` | 不对称误差提取/换算 |
| | `generate_download_url` / `extract_all_gz_recursive` / `gunzip` | 数据获取与解压 |
| `jinwu.core.time` | `TimeEP` / `TimeSwift` / `TimeFermi` / `TimeGECAM` / `TimeHXMT` / `TimeMAXI` / `TimeLIGO` 等 | 任务时间系统（EP 秒、MET、卫星时间转换），禁止手工 MJDREF |
| | `extract_time_interval` / `check_time_overlap` / `get_overlap_duration` / `compare_time_intervals` | 时间区间运算 |
| `jinwu.core.io` | `readfits(path, kind)` / `read_arf/rmf/pha/lc/evt` | OGIP 读取（自动识别类型） |
| | `writefits` / `write_pha/arf/rmf/lc/evt` | OGIP 写入 |
| | `guess_ogip_kind` / `band_from_arf_bins` / `channel_mask_from_ebounds` | 类型判定、能带工具 |
| `jinwu.core.xselect` | `extract_products_with_xselect` / `extract_spectrum_with_xselect` / `XSelectSession` | xselect 产品提取 |
| | `select_events` / `filter_time` / `filter_energy` / `filter_region` / `merge_gti` / `trim_events_to_gti` | 事件筛选与 GTI |
| | `extract_spectrum` / `extract_curve` / `extract_image` / `accumulate_spectrum_from_events` | 从事件提取产品 |
| | `write_curve` / `write_image` / `write_pha` | 产品写入 |
| `jinwu.core.spectrum_prep` | `prepare_spectra` / `PreparedSpectrum` / `PreparedJointSpectrum` / `PreparedCatalog` | 谱准备、链接、联合 |
| `jinwu.core.fit` | `fit_prepared` / `fit_xray_models` / `fit` / `run_xspec_chain` | XSPEC 拟合入口（`fit_prepared` 单谱、`fit_xray_models` 多候选比较；旧 `fit_spectrum`/`fit_spectrum_from_files` 已于 2026-08 移除） |
| | `PowerLawModel` / `BrokenPowerLawModel` / `SmoothlyBrokenPowerLawModel` / `DoubleBrokenPowerLawModel` / `ExponentialModel` / `GaussianModel` / `ConstantModel` | 模型类 |
| | `ModelRegistry` / `LightcurveFitter` / `resolve_xray_model_specs` / `calculate_model_fit_metrics` | 模型注册、拟合器、指标 |
| | `FitResult` / `XspecChainResult` / `XRayModelSpec` | 结果与规格类型 |
| `jinwu.core.plot` | `plot_spectrum` / `plot_lightcurve` / `plot_event_txx` / `plotfit` | X 射线绘图（统一样式 `jinwu.core.plotstyle`；旧 `plot_ogip`/`plot_xspec_origin` 已移除） |
| `jinwu.core.galactic` | `resolve_galactic_absorption` / `GalacticAbsorptionResult` | 银河吸收 |
| `jinwu.core.datasets` | `netdata` / `LightcurveDataset` / `SpectrumDataset` / `JointDataset` | 数据集契约 |
| `jinwu.core.config` | `instrument(name, **kwargs)` / `register_instrument` / `FXT` / `WXT` / `BAT` / `GBM` / `GECAM` / `UVOT` + 各 `Config` | 仪器与配置 |
| `jinwu.core.units` | `magnitude_to_flux` / `Magnitude` / `FilterInfo` / `InstrumentFilterLibrary` | 光学 mag/mJy 转换。⚠️ 2026-08-22 实测有缺陷待重写：默认 Vega（AB 需显式传 system）、强制 filter_info（纯 AB→fν 传 None 报错）、`u.ABmag` 不自动切系统。重写前勿推荐复用，详见 EP `REUSABLE_FUNCTIONS.md` 已知问题节 |
| `jinwu.core.base` | `EnergyBand` / `RegionArea` / `OgipMeta` / `PhaBase` / `RmfBase` / `ArfBase` / `EventDataBase` / `LightcurveDataBase` | 基类与数据原语 |
| `jinwu.core.ogip` | `ValidationReport` / `OgipFitsBase` | OGIP 校验 |
| `jinwu.core.pipeline` | `register_pipeline` / `pipeline` / `InstrumentPipeline` / `PipelineStatus` | 流水线 |
| `jinwu.core.timescale` | `txx` / `txx_iterbkg` / `iterative_bayesian_blocks` | 时标（T100/T90/T50）计算：`txx` 事件级贝叶斯块 + A&A 5.4 分位；`txx_iterbkg` 迭代背景自洽分箱贝叶斯块（burstcube 移植，含全流水线重采样误差）。`ops.txx` 为兼容别名；推荐入口 `jinwu.core.data.timescale` 分析器（`method='aanda'/'iterbkg'`） |
| `jinwu.core.ops` | `bin_bblocks` / `autobin` / `BayesianBlocksBinner(use_exposure=True)` / `bayesian_blocks_exposure` | 贝叶斯块分箱（`use_exposure` 启用逐箱曝光加权变点，适合 EP/WXT） |
| `jinwu.core.host` | `HostGalaxyFinder` | 宿主星系查找/分类 |
| `jinwu.core.heasoft` | `HeasoftEnvManager` | HEASoft 环境管理（生产标准路径） |
| `jinwu.core.rebin_rs` | `rebin_lightcurve_rs` | 光变重分箱 |
| `jinwu.core.products` | `FitProductSet` / `ExternalRunArtifacts` / `write_json` / `sha256_file` / `safe_filename_token` | 产物集与统一的 JSON 落盘/哈希/文件名助手 |

## 使用示例（规范写法）

```python
from jinwu.core.utils import nhtot, li_ma_snr
from jinwu.core.time import TimeEP, extract_time_interval
from jinwu.core.io import readfits
from jinwu.core.xselect import extract_spectrum_with_xselect
from jinwu.core.spectrum_prep import prepare_spectra
from jinwu.core.fit import fit_prepared, fit_xray_models
```

## 登记区（新增/提升公共函数时，在此追加）

格式：`| 函数或类 | 模块 | 一句话用途 | 日期 |`

| 函数或类 | 模块 | 用途 | 日期 |
|---|---|---|---|
| `GBMFlareInterval` / `check_gbm_coverage` / `select_gbm_detectors` | `jinwu.fermi.gbm.pipeline` | 连续 GBM 覆盖、遮挡/SAA/GTI 判定与几何选探测器 | 2026-08-27 |
| `fetch_gbm_continuous_products` / `extract_gbm_spectral_products` | `jinwu.fermi.gbm.pipeline` | 可恢复连续数据下载及 TTE 局部多项式 PHA/BAK 提取 | 2026-08-27 |
| `build_gbm_response_command` / `generate_gbm_response` | `jinwu.fermi.gbm.response` | 官方 GBM 响应生成器的无 shell 安全封装 | 2026-08-27 |
| `profile_source_amplitude` | `jinwu.core.upperlimit` | 与 response-aware 上限同契约的非负幅度 profile 显著性 | 2026-08-27 |
| `FastNormFit` | `jinwu.core.upperlimit` | Poisson 似然比（TS）归一化快速拟合（解析任意阶导数 + Newton/Halley；`upper_limit()` 给泊松精确区上限），移植自 HEASoft burstcube | 2026-08-30 |
| `bayesian_blocks_exposure` | `jinwu.core.ops` | 曝光加权分箱贝叶斯块（支持 0 计数箱，返回变点箱索引），移植自 HEASoft burstcube | 2026-08-30 |
| `iterative_bayesian_blocks` / `txx_iterbkg` | `jinwu.core.timescale` | 迭代背景自洽贝叶斯块（Giacomo 技巧 + prominence 定界 + 循环检测）与其 Txx 封装 | 2026-08-30 |
| `txx`（拆分） | `jinwu.core.timescale` | 时标计算从 `ops.py` 拆入独立模块；`jinwu.core.ops.txx` 及 `_txx54_*` 助手保持兼容重导出 | 2026-08-30 |
| `PhaWriter`/`RmfWriter`/`ArfWriter`（关键字对齐） | `jinwu.core.io` | 写出对齐 HEASoft 6.37 heasp 约定：PHA 补 `TLMIN1/TLMAX1/DETCHANS`；RMF 补 `DETCHANS/NUMGRP/NUMELT` 与实际 `F_CHAN` 列对应的 `TLMINn` + header 透传；ARF 补 `HDUVERS` | 2026-08-30 |
| 读端结构化解析 + 通道校验 | `jinwu.core.io` / `jinwu.core.data` | `PhaData.tlmin/tlmax/det_chans`、`RmfData.tlmin/det_chans`、`ArfData.hduvers` 读入即解析（缺失时回退推断）；`RmfData.validate()` 新增 `INCONSISTENT_CHANNELS` 校验（F_CHAN+N_CHAN vs TLMIN+DETCHANS，同 6.37 heasp） | 2026-08-30 |
| `check_response_compatibility` + validate 对齐 ftverify/heasp | `jinwu.core.ogip` | 谱↔响应通道兼容性检查；`validate()` 全面对齐 HEASoft 6.37 校验：HDUCLAS1/HDUCLAS2/HDUVERS、PHA 通道三件套自洽、RMF DETCHANS↔EBOUNDS、GTI 自洽（BAD_GTI/UNSORTED_GTI）；均从 `jinwu.core` 懒加载导出 | 2026-08-30 |
| `SwiftGRB` / `SwiftGRBDataConfig` / `SwiftGRBSegmentationConfig` | `jinwu.core.config` | 单 GRB Swift BAT+XRT 的公共产品、分段、拟合和执行预设（插件保持惰性导入） | 2026-08-31 |
| `safe_extract_tar` | `jinwu.swift.grb.pipeline` | UKSSDC 外部产品归档的路径遍历与链接防护解包 | 2026-08-31 |
| `InstrumentPipeline.stage_input_dependencies` | `jinwu.core.pipeline` | 声明外部科学输入并纳入阶段缓存哈希，输入变化自动失效相关阶段 | 2026-08-31 |
| `BATSurvey` / `BATSurveyInput` / `BATSurveyPipeline` | `jinwu.swift.bat.survey` | 通用 Swift/BAT survey 单目标观测发现、survey、光变、谱/上限、可选 mosaic 与报告流程 | 2026-08-31 |
| `read_bat_survey_rates` / `select_overlapping_pointings` | `jinwu.swift.bat.survey` | 保留 BAT survey 有符号净率、BKG_VAR 误差语义并按完整指向与 GTI 重叠选择 | 2026-08-31 |
| `parse_area_table` / `calculate_background_scale` | `jinwu.swift.bat.survey` | 解析压缩面积表并用带单位的源/背景面积计算 alpha，缺失信息明确失败 | 2026-08-31 |
| `read_gti_intervals` / `gti_overlap_duration` | `jinwu.swift.bat.survey` | 读取 Swift GTI 的 START/STOP 并按请求窗口计算真实 GTI 重叠曝光（不缩放完整指向谱） | 2026-08-31 |
| `validate_survey_pha` / `profile_survey_upper_limit` | `jinwu.swift.bat.survey` | 校验 survey PHA/响应并以 Gaussian chi 固定光子指数 profile 上限 | 2026-08-31 |
| `ensure_headas_env` | `jinwu.core.ops` | 为 HEASoft/Perl 任务构造可写、阶段隔离的运行环境变量（含 Conda `heainit.sh` 运行时路径） | 2026-08-31 |
| `EmpiricalTailResult` / `empirical_tail_probability` / `p_to_sigma` / `sigma_to_p` | `jinwu.core.model_comparison` | 非规则嵌套模型的经验尾概率、Clopper--Pearson 区间及单/双侧 Gaussian-equivalent sigma 换算 | 2026-09-01 |
| `GaussianNetRateObservation` / `profile_gaussian_upper_bound` | `jinwu.core.upperlimit` | 带单位的有符号 Gaussian 净率、完整协方差和 `A>=0` profile 上限；保留 signed MLE 与物理边界状态 | 2026-09-01 |
| `BATSurveySensitivityAdapter` / `estimate_bat_survey_sensitivity` | `jinwu.swift.bat.survey` | 使用 BAT 原生八能道 TOTSNR、空白控制和固定位置注入估计独立 detection sensitivity | 2026-09-01 |
| `integrate_background_interval` / `validate_background_residuals` | `jinwu.fermi.gbm.pipeline` | GBM 背景精确区间积分与连续留出块残差/预测覆盖诊断 | 2026-09-01 |
| `gti_intervals_for_paths` / `single_response` | `jinwu.fermi.gbm.pipeline` | 用真实 TTE GTI 选择 RSP2 加权区间并安全写出单矩阵响应，保留输入文件只读 | 2026-09-01 |
| `BayesFactorResult` / `summarize_bayes_factor` | `jinwu.core.model_comparison` | Numerically stable log-evidence and Bayes-factor intervals | 2026-09-02 |
| `model_posterior_probability` / `model_averaged_direction_probabilities` | `jinwu.core.model_comparison` | Convert Bayes factors and prior odds to posterior model and directional probabilities | 2026-09-02 |
| `onoff_log_marginal_likelihood` | `jinwu.core.model_comparison` | Analytic Poisson ON/OFF background marginal likelihood with a proper Gamma prior | 2026-09-02 |
| `model_averaged_direction_probability_interval` / `onoff_log_profile_likelihood` | `jinwu.core.model_comparison` | Propagate evidence/q numerical intervals and provide a normalized ON/OFF profile diagnostic for W-stat cross-checks | 2026-09-02 |
| `recover_raw_off_counts` | `jinwu.core.model_comparison` | Recover raw OFF PHA counts from PyXspec source-scaled background rates with fail-closed integer validation | 2026-09-04 |
| `GBMPosHistSelection` / `find_gbm_poshist` / `estimate_gbm_orbit_period` | `jinwu.fermi.gbm.pipeline` | 按目标时刻选择真实或 RapidGBM 风格 30 轨历史 POSHIST，并记录预测来源 | 2026-09-06 |
| `GBMGeometryState` / `fetch_poshist_for_time` / `read_gbm_geometry` | `jinwu.fermi.gbm.poshist` | 按时间取得 POSHIST 并读取单时刻地心位置、地球遮挡、SAA 和 GBM 指向 | 2026-09-06 |
| `SkyMap` / `load_skymap` / `credible_region` / `probability_in_footprint` / `refined_probability` | `jinwu.gw.skymap` | 统一读取 LVK 多分辨率/普通 HEALPix 天图，计算可信区、MOC 覆盖概率和 10--13 阶收敛 | 2026-09-06 |
| `SkyMapData` / `load_skymap` / `sky_map_pixel_vectors` | `jinwu.core.skymap` | 纯数据 HEALPix 天图读取（本地文件、嵌套 UNIQ、sr^-1 密度），供仪器包使用；jinwu-fermi 的 subthreshold 天图先验由此提供，不再依赖 jinwu-gw | 2026-09-20 |
| `GWPipeline` / `run_gw_pipeline` | `jinwu.gw.pipeline` | 单 GW 警报天区、GBM 与 EP/BAT 图层交叉及静态报告 | 2026-09-06 |
| `GraceDBClient` / `normalize_superevent_id` | `jinwu.gw.gracedb` | 匿名只读检索 GraceDB 超事件、公告版本与文件，支持事件页 URL | 2026-09-06 |
| `spherical_cap_moc` | `jinwu.gw.layers` | 使用 MOCPy 锥体并以自适应 HEALPix 后备构造带来源方法的球冠 MOC | 2026-09-06 |
| `fetch_gbm_products_for_interval` | `jinwu.fermi.gbm.pipeline` | 带单位上下文的跨小时/跨日连续产品下载；已泛化到 jinwu：是 | 2026-09-12 |
| `GBMTargetedSearchInput` / `GBMTargetedSearchConfig` / `GBMTargetedSearchPipeline` / `run_targeted_search` | `jinwu.fermi.gbm.subthreshold` | 外部触发亚临界搜索、位置/天图先验、候选和统计定位产品；已泛化到 jinwu：是 | 2026-09-12 |
| `calibrate_targeted_search` / `calibration_from_searches` / `estimate_candidate_far` | `jinwu.fermi.gbm.subthreshold.calibration` | 同配置离源搜索及有限曝光经验 FAR，零尾返回上限；已泛化到 jinwu：是 | 2026-09-12 |
| `merge_tte_events` / `interval_exposure` / `read_detector_events` / `prepare_search_data` / `MeasuredHistory` | `jinwu.fermi.gbm.subthreshold.data` | 多重事件去重、GTI 活时间、背景和真实姿态；已泛化到 jinwu：是 | 2026-09-12 |
| `make_search_windows` / `spatial_prior_weights` / `evaluate_search_likelihood` / `select_search_candidates` / `run_search_grid` | `jinwu.fermi.gbm.subthreshold.search` | 多尺度窗口、先验加权、GTS 核及候选筛选；已泛化到 jinwu：是 | 2026-09-12 |
| `validate_search_templates` / `make_search_plots` / `write_candidate_localizations` | `jinwu.fermi.gbm.subthreshold.pipeline` / `plots` | 模板来源验证、光变/瀑布图、统计 HEALPix 与响应检查；已泛化到 jinwu：是 | 2026-09-12 |
| `compute_mvt` / `classify_mvt` / `load_validation_curve` / `summarize_resamples` | `jinwu.fermi.gbm.mvt.engine` | 固定上游 Haar MVT 核心、原始返回与中间量、显式上限/失败、经验曲线及条件抽样汇总；已泛化到 jinwu：是 | 2026-09-30 |
| `GBMMVTInput` / `GBMMVTConfig` / `GBMMVTPipeline` / `run_gbm_mvt` | `jinwu.fermi.gbm.mvt` | 独立 GBM TTE MVT 流程、探测器迭代、Poisson 重采样、稳定性与诊断；已泛化到 jinwu：是 | 2026-09-30 |
| `latest_products` / `merge_tte_events` / `interval_exposure` / `read_detector_events` | `jinwu.fermi.gbm.tte` | 从 subthreshold 提取公共 TTE 版本选择、保留多重性的去重、GTI 与计数读取；旧导入保留；已泛化到 jinwu：是 | 2026-09-30 |
| `bin_events` / `prepare_detector` / `peak_snr` / `select_detectors` / `resample_mvt` | `jinwu.fermi.gbm.mvt.data` / `resampling` | 显式时窗/能段 TTE 适配、原版总计数 SNR、累积前缀选探测器与可复现随机流；已泛化到 jinwu：是 | 2026-09-30 |
| `load_pg_reference` / `compare_report` | `scripts/compare_gbm_mvt_snr.py` | 执行原 gv_significance PG 函数体、按同一 MVT bin 和拟合协方差核对统计量；分析验收脚本，未泛化到 jinwu：否 | 2026-09-30 |

| `PreparedOnOffMarginalLikelihood` | `jinwu.core.model_comparison` | Exact normalized Gamma-Poisson ON/OFF likelihood with cached data terms | 是 |

| `absorption_budget`, `AbsorptionBudget` | `packages/jinwu/src/jinwu/physics/absorption.py` | 单位化中性吸收截面/光深/占比，独立 XSPEC 进程，tbabs/atomic 后端，查询、绘图及导出；闭合失败显式掩码 | 是 |

| `AbsorptionBudget.plot(fraction_scale=...)`, `.show()`, `.from_json()` | `jinwu.physics.absorption` | Accessible opacity plotting, log percentages, Notebook display and archived table reload | 是 |

| `absorption_budget(backend="ztbabs")`, `AbsorptionBudget.plot(energy_scale=...)` | `jinwu.physics.absorption` | 原生 zTBabs/wilm 丰度缩放、线性能量轴与 tau=1 标注 / Native zTBabs/wilm scaling and tau=1 plots | 是 |

| `absorption_budget` 教程 / tutorial | `examples/absorption/absorption_budget.ipynb`; `test/test_absorption_plot.py` | 自包含双语教程与自动化回归 / Self-contained bilingual tutorial and regression tests | 是 |

| 吸收示例 / absorption examples | `examples/absorption/absorption_budget.ipynb`, `examples/absorption/absorption_budget.py` | 独立双语教程目录 / Dedicated bilingual example directory | 是 |

| 其他示例教程 / Other tutorials | `examples/README.md`, `examples/migration_manifest.json` | 11 组双语迁移说明与 Notebook/Python 示例；状态分级 / 11 classified Notebook/script pairs | 是 |

| `snr` / `li_ma_snr` | `jinwu.core.significance`（`core` / `core.utils` 导出，旧 Li–Ma 路径兼容） | 明确单位与背景统计的五种计数显著性；冻结 gv_significance 源码、PP 系统误差与 PG 背景拟合误差；已泛化到 jinwu：是 |
| `load_gv_reference` / `compare_snr_case` / `validate_wxt` / `validate_gbm` | `scripts/validate_snr.py` | 固定来源与中间量对照、真实 PHA/TTE→背景拟合/协方差→SNR 验收；分析验证脚本，未泛化到 jinwu：否 |
| `_duration_fixed_edges` / `_duration_event_blocks` / `_duration_plateau` / `_duration_koshut` | `packages/jinwu/src/jinwu/core/timescale.py` | 严格定界分箱、已知观测区间事件 BB、累计平台散布、Koshut Eq8–16 与 ON/OFF 方差扩展（内部秒/计数接口） | 是 |
| `load_baseline` / `filtered_copy` / `main` | `scripts/compare_ep260119a_duration.py` | hash 固定旧版与修正版同输入 EP260119a 时标对照，独立 PI 筛选、真实累计诊断和原件完整性验收 | 否 |
| `load_rejected_v2` / `check_window_consistency` | `scripts/compare_ep260119a_duration.py` | 固定撤回版本重现 T90/T100 不一致；逐名义结果及模拟样本验收同窗口计数守恒、区间包含关系 | 否 |
| `_duration_reference_boundary_options` / `_duration_reference_edges` | `packages/jinwu/src/jinwu/core/timescale.py` | 显式秒参数复用参考WXT移边、孤立事件补边和最终snap规则；不读隐式配置文件，供名义及每次MC定窗使用 | 是 |
| `run_case`（参考自动时标执行） | `reviews/evidence/duration-reference-run-20261003/run_reference.py` | 显式ON/OFF事件、alpha、配置及输出布局调用原get_lc→get_Td90_normal；固定随机种子保存自动窗口/计数/原码误差/溯源，供原事件和PI独立输入比较 | 否 |
| `clipped_event_copy`（PI时窗副本） | `reviews/evidence/duration-pi-window-20261004/run_restricted_pi.py` | 单GTI PI50–400分析用显式EP秒上下限复用参考事件裁剪；保留事件列，更新副本GTI/时间头并逐列核验，原件不变；未泛化到jinwu | 否 |

| `generate_api_pages` | `docs/_ext/jinwu_docs.py` | 从五包源码 AST 生成规范模块 API、导航与排除清单；供本地和 RTD 构建复用 | 否（文档构建工具） |
| `normalize_docstrings` / `separate_docstring_blocks` | `docs/_ext/jinwu_docs.py` | 构建时统一中英文 NumPy/RST 文档格式、保留代码块并修复块边界；不改分析源码 | 否（Sphinx 文档工具） |

| `_evaluate_expression` | `packages/jinwu/src/jinwu/ftools/ftselect.py` | 按语法树保留优先级并逐元素计算事件筛选表达式 | 是 |
| `_load_signal_counts_npz` | `packages/jinwu/src/jinwu/lf/lcfake.py` | 从显式净计数或 ON/OFF 模板提取带符号源计数，避免模拟背景双计 | 是 |
| `_create_flux_chain_at_best_fit` | `packages/jinwu/src/jinwu/core/bxa_fit.py` | 计算后验通量后恢复 XSPEC 最佳拟合；可选通量失败与状态恢复失败分开处理 | 是 |
| `BackgroundCountsPosterior.sample_rate_off` | `packages/jinwu/src/jinwu/background/backprior.py` | 从 Gamma 后验直接抽取 OFF 潜在背景率（cts/s），与后验预测 Poisson 计数分开；共用于 ON/OFF 采样 | 是 |
| `BackgroundCountsPosterior._observation` / `_exposure` / `_validate` | `packages/jinwu/src/jinwu/background/backprior.py` | 在采样、期望值和更新边界校验背景计数、秒曝光和后验参数 | 是 |
| `_grppha_output_error` | `packages/jinwu/src/jinwu/ftools/grppha_hsp.py` | 共用于两种 HEASoft 后端，检查任务状态、FATAL 信息与分组 PHA 结构 | 是 |
| `_parse_ds9_region_shape` / `parse_ds9_region_line` | `packages/jinwu/src/jinwu/ftools/region.py` | 解析 DS9 区域、保留排除符号及坐标系统；赤道六十进制坐标和显式角秒/角分尺寸 | 是 |
