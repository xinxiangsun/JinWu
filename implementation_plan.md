# Implementation Plan — codex/beta 相对 master 的变更审查与分级修复

## [Overview]

对 `codex/beta` 相对 `master` 的全部改动（9 个提交的 152 文件，+26511/−8681 行；以及工作区 staged 34 文件 +4793、unstaged 46 文件 +1074/−160、untracked 69 项，含 `jinwu-gw` 新发行包、`jinwu-fermi/gbm/subthreshold`+`_vendor`、`physics/absorption.py`、`examples/`、`scripts/`），做一次可复现的分级审查，产出问题清单（P0/P1/P2，每条带 file:line 与复现命令）与对应修复提交。

范围与上下文：

- 基线事实：`master` 是 `codex/beta` 的祖先（非分叉）；`origin/beta` 落后 1 个提交（`269587a` 未推送）；`codex/beta` 与工作区合计约 3.3 万行新增。
- 结构变化：仓库已重构为 `packages/{jinwu,jinwu-ep,jinwu-swift,jinwu-fermi,jinwu-gw,jinwurs}` 六发行包 + PEP 420 命名空间（顶层 `jinwu/__init__.py` 已删除，根 `pyproject.toml` 不再可安装）；仪器插件经 `jinwu.instruments` entry points 发现。
- 本轮改动同时带来：新增离线 CI（3.11/3.12/3.13 矩阵）与 publish wheel-gate、burstcube 算法移植、OGIP IO/校验对齐 HEASoft 6.37、AUD-01~06 缺陷修复、四/五条可恢复单目标管线 + CLI、BXA 拟合与模型比较、GW 覆盖（`jinwu-gw`）、若干文档新页。

审查维度（六块，逐块产出发现）：

1. 架构与包边界：命名空间包/可安装性、entry-point 契约、插件间依赖方向、核心 vs 仪器的分层。
2. 打包与发布：版本锁步、conda recipe、Makefile 发布链、CI/publish 工作流、wheel 内容与 package-data。
3. 科学算法与数值正确性：burstcube 移植（FastNormFit/upper_limit/bayesian_blocks_exposure/txx_iterbkg）、OGIP 读写与校验、GR 解析解、GW 概率积分与 MOC 精化。
4. 管线与数据契约：可恢复性/阶段缓存指纹、网络阶段 opt-in、provenance/sha256、输出目录与再运行语义。
5. 安全与健壮性：SSRF 门（`jinwu.gw.alert`）、外网抓取的超时/重试/重定向、子进程（XSPEC/BXA/ftools）边界、异常吞错面。
6. 文档与可复现性：RTD toctree/交叉引用、changelog 完整性、测试门禁覆盖面与证据可复现性。

证据纪律（对齐仓库 AGENTS.md“真实数据验证”）：每条发现必须给出 file:line 与可执行复现；报告分列“流程运行证据 / 结果验证证据 / 科学解释证据”；无法完成的部分显式标注未验证范围。本轮为只读审查 + 复现，不改变任何既有产物。

## [Types]

审查阶段不引入新的类型定义；本节的目的是把**必须逐条核对的公共契约**固定下来（这些是 P0/P1 级破坏性变更面）：

- 命名空间契约：顶层 `jinwu/__init__.py` 删除（提交 ef49ea1）→ `import jinwu` 不再提供 `jinwu.<submodule>` 属性；`from jinwu import Time/TimeDelta` 等便捷导出失效，改由 `jinwu.core` 提供。需核对 `docs/changelog.rst`、`README.md`、`docs/quickstart.rst`、`docs/index.rst` 是否给出迁移路径。
- 包移除契约：`jinwu.response` 被移除（工作区 staged 删除 `packages/jinwu/src/jinwu/response/{__init__.py,basic.py}`）。已核对引用面仅 `docs/changelog.rst`、`docs/api.rst` 提到，代码侧 GBM 响应改由 `jinwu.fermi.gbm.response` 提供；需在报告中确认无第三方迁移说明缺失。
- entry-point 契约：`jinwu.instruments` 名称集合 `{ep, swift, fermi, gw}`（publish.yml wheel-gate 断言）。语义问题：`jinwu.gw` 不是仪器却注册为 instruments 插件，需决定是改名协议（如 `jinwu.pipelines`）还是保留并在文档中定义“非仪器管线”也走该协议。
- 阶段/结果契约：`InstrumentPipeline.stage_code_dependencies` / `_stage_code_fingerprint`（AUD-01 语义：声明依赖必须存在、内容变化必须改变指纹）、`StageResult`、`GaussianNetRateObservation`/`profile_gaussian_upper_bound`、`BXAFitResult`、`CoverageResult`/`GWEvent`/`SkyMap`。
- 依赖方向契约：禁止 `jinwu.fermi → jinwu.gw`（当前存在，见 [Dependencies]）。修复后若下沉 skymap 读取器到 `jinwu.core`，属于新增公共 API，需给出命名与向后兼容别名。

目标态（仅列出已识别需要新增/调整的契约，具体取决于审查结论）：

- 新增 `jinwu.core.skymap`（或等价共享层）承载 flat/multi-order HEALPix 概率图读取与可信区统计，`jinwu-gw.skymap` 保留同名别名做兼容转发。
- `make sync` 覆盖 `packages/jinwurs/conda-recipe/meta.yaml`（版本 + 占位 sha256 显式失败而非静默保留占位）。

## [Files]

审查产物（新建）：

- `reviews/codex-beta-vs-master.md`：审查报告正文（范围、方法、发现表、证据、未验证范围、复现命令）。
- `reviews/evidence/`：复现日志（离线门禁、import 冒烟、wheel-gate、Sphinx、真实数据验收）——只放日志与小体积摘要，不复制原始数据与既有结果。
- `implementation_plan.md`（仓库根，本文件）：计划与执行进度。

拟修改文件（按发现，P0→P2 顺序；编号见“候选发现清单”）：

- `Makefile`：R1 删除 `sha256` 目标中失效的重复赋值行（第 64 行）；R2 `publish` 依赖链与上传清单（平台 wheel、`rust-build`）。
- `packages/jinwurs/conda-recipe/meta.yaml`：R3 版本 0.1.0→0.2.0、`sha256: placeholder` 处理、纳入 `make sync` 覆盖范围。
- `.gitignore`：R4 `test/*` 白名单策略收敛（避免新增测试默认被忽略）。
- `.github/workflows/ci.yml`：R5 显式声明收集面（包含 `packages/jinwu-swift/tests`、GW 契约测试文件）；markers 审计后修正跳过矩阵。
- `packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/search.py`、`packages/jinwu-fermi/pyproject.toml`、`packages/jinwu-gw/pyproject.toml`：R6 打破 `fermi → gw` 反向依赖。
- `docs/index.rst`、`docs/usage/gbm_subthreshold.rst`、`docs/usage/absorption_budget.md`、`docs/usage/gw_coverage.rst`、`docs/api.rst`：R7 toctree/引用完整性（`gbm_subthreshold.rst`、`absorption_budget.md` 目前未提交）。
- `pyproject.toml`（仓库根）：R8 决策并落地 `[tool.uv.workspace]` 去留（工作区已删 3 行，HEAD 中存在）。
- `.readthedocs.yaml`：R9 若文档/API 页引用 `jinwu.gw`，需与其它四包一致安装 `packages/jinwu-gw`。
- `docs/changelog.rst`、`ARCHITECTURE_ROADMAP.md`、`REUSABLE_FUNCTIONS.md`：修复完成后登记（仓库既有约定）。

待删除/移动：仅 R6 涉及的 skymap 读取器下沉（若结论支持）；其余无删除。

配置更新：`.github/workflows/{ci,publish}.yml`、`recipe/meta.yaml`（若触发 sha256 重算）、`packages/*/pyproject.toml`（仅版本/依赖声明）。

## [Functions]

审查重点函数（对照验证清单，全部只读比对 + 数值复核）：

- `packages/jinwu/src/jinwu/core/upperlimit.py`：`FastNormFit`（解析导数 + Newton/Halley + 欠涨分支）、`FastNormFit.upper_limit()`（ΔTS=2.7055 对应单侧 90% 的泊松精确区约定）、`profile_gaussian_upper_bound()`、`GaussianNetRateObservation`。对照 HEASoft 6.37 `burstcube/lib` 原始实现与解析公式。
- `packages/jinwu/src/jinwu/core/ops.py::bayesian_blocks_exposure()`：曝光加权分箱（0 计数箱处理）；对照 HEASoft 实现与 `BayesianBlocksBinner(use_exposure=True)` 集成路径。
- `packages/jinwu/src/jinwu/core/timescale.py::txx_iterbkg()`：迭代背景自洽（Giacomo 技巧 + prominence 定界 + 循环检测 + Poisson 重采样误差）；复核提交声称修复的三处隐蔽 bug（`upper_limit` 区间方向、重分箱相位丢尾箱、`maximum.accumulate` 误用）。
- `packages/jinwu/src/jinwu/core/pipeline.py`：`InstrumentPipeline._stage_code_fingerprint()`（缺失依赖必须显式报错）、`_discover` 经 entry points 的导入与错误提示。
- `packages/jinwu-gw/src/jinwu/gw/alert.py`：`_validate_http_url()`、`fetch_http_bytes()`（逐跳校验重定向、`allow_redirects=False`、重试）、`skymap_from_notice()` provenance（sha256 + `_versioned_path` 版本保留）。
- `packages/jinwu/src/jinwu/physics/gr.py`：`GeneralRelativity.v` setter/beta/lorentz_factor（AUD-03 回归面）。
- `packages/jinwu/src/jinwu/core/io.py` / `ogip.py`：PHA/RMF/ARF 结构字段（TLMIN/TLMAX/DETCHANS/NUMGRP/HDUVERS）读写与 `check_response_compatibility`。
- `packages/jinwu-fermi/src/jinwu/fermi/gbm/poshist.py`：`estimate_gbm_orbit_period`、`fetch_poshist_for_time`、`find_gbm_poshist`、`read_gbm_geometry`（SAA/地心遮挡/插值间隙）。
- `Makefile`：`sha256`、`publish`、`sync`、`release` 目标（发现 R1/R2/R3 的落点）。

修复阶段拟修改函数：

- `Makefile::sha256`：删除失效行，保留 `dist/jinwu-[0-9]*.tar.gz` 选择器，并在未匹配时报错。
- `Makefile::publish`：改为上传全部发行包产物（含 `packages/jinwurs` 平台 wheel），或在缺少 Rust 轮子时显式告警。
- `packages/jinwu/src/jinwu/core/timescale.py::txx_iterbkg` / `upperlimit.py`：仅在审查确证数值偏差时修改（先补对照测试再改）。
- `packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/search.py`：将 `from jinwu.gw.skymap import load_skymap`（第 104 行）改为依赖下沉后的共享读取器接口。

无删除函数（除非 R6 迁移后保留转发别名）。

## [Classes]

审查对象类（逐类核对公开方法与行为契约）：

- `jinwu.core.pipeline.InstrumentPipeline` / `PipelineStage`：阶段指纹、可恢复工作区、网络阶段 opt-in。
- `jinwu.ep.wxt.pipeline.WXTPointingPipeline`、`jinwu.swift.grb.pipeline.SwiftGRBPipeline`、`jinwu.swift.bat.survey.BATSurveyPipeline`、`jinwu.gw.pipeline.GWPipeline`：`stages` 与 `stage_code_dependencies()` 一致性（已由 `test/test_stage_code_deps.py` 覆盖 4 条管线）。
- `jinwu.gw.gracedb.GraceDBClient` / `GraceDBNotice`：只读语义、超时/重试、`normalize_superevent_id` 校验。
- `jinwu.gw.skymap.SkyMap` 及其 MOC 统计；`jinwu.gw.coverage.*` 的 GBM 覆盖积分。
- `jinwu.core.ogip.PhaData/RmfData/ArfData`：与 HEASoft 6.37 `heasp` 的结构字段对齐。
- `jinwu.physics.GeneralRelativity`、`jinwu.physics.absorption.AbsorptionBudget`（工作区新增）。
- `jinwu.fermi.gbm.pipeline` 的 GBM 管线类（4443 行单文件，含网络下载阶段）。

修复阶段拟修改类：

- `GWPipeline`：若 skymap 读取器下沉，更新 `stage_code_dependencies()` 返回的模块清单（注意：这会改变阶段缓存指纹，需在迁移说明中提示既有缓存失效与重算范围）。
- `GraceDBClient`：仅在安全/超时审查确证问题时调整。

无类删除。

## [Dependencies]

已核实的 PyPI 现状（本轮通过 PyPI JSON 元数据核实，作为 CI 可安装性证据）：

- `astro-gdt` 2.2.3（requires-python ≥3.11，`py3-none-any`；拉 `numpy>=2.2.3`、`scipy>=1.16.3`、`astropy>=7.0`、`matplotlib~=3.10`、`healpy~=1.18`、`cartopy`、`statsmodels`、`pyproj`、`rich`）→ CI 安装面显著变重。
- `astro-gdt-fermi` 2.2.2（≥3.11，`py3-none-any`，含 8 MB 数据）。
- `swiftbat` 0.1.7（≥3.11，**要求 `astropy>=8`**，另拉 `swifttools`、`skyfield`、`truststore`、`astroquery`）→ 与 core 的 `astropy>=7.1` 兼容但在 CI 中会装 astropy 8，需核对 core/仪器的 astropy 8 兼容性。
- `ligo.skymap`（classifiers 覆盖 3.11–3.14）、`mocpy` 0.20.0（≥3.9）、`ligo-gracedb` 2.15.7（≥3.6，`py3-none-any`）→ `jinwu-gw` 依赖面在 3.11–3.13 可安装。
- `batanalysis` 2.1.0 仅作为 `jinwu-swift[survey]` extra（CI 不安装）。

问题项（拟在修复阶段处理）：

1. **环形依赖**：`jinwu-gw` 硬依赖 `jinwu-fermi`，而 `jinwu-fermi[search-skymap]` 依赖 `jinwu-gw`，且 `subthreshold/search.py:104` 直接 `from jinwu.gw.skymap import load_skymap`。修复方向：把通用 HEALPix 概率图读取下沉到 `jinwu.core`，`jinwu-gw` 与 `jinwu-fermi` 均依赖 core；保留 `jinwu.gw.skymap.load_skymap` 作为转发别名。
2. **CI 安装矩阵成本**：ci.yml 以 editable 安装 5 个包（含 cartopy/healpy/statsmodels/ligo.skymap/mocpy）于 3 个 Python 版本；建议按“core+EP+契约测试”与“GW/Fermi 重依赖”拆 job，并评估缓存或 `--no-deps` 分层安装。
3. **conda 侧欠覆盖**：`recipe/meta.yaml` 只覆盖核心包且 `sha256` 为硬编码值；`packages/jinwurs/conda-recipe/meta.yaml` 版本仍是 0.1.0、`sha256: placeholder`，且不在 `make sync` 覆盖面内。

版本锁步现状：`jinwu` / `jinwu-ep` / `jinwu-swift` / `jinwu-fermi` / `jinwu-gw` / `jinwurs` 均为 0.2.0（与 `recipe/meta.yaml` 一致）；需在修复阶段用 `make sha256` + 实际 `dist/` 产物验证 `recipe/meta.yaml` 的 sha256 与锁定版本 sdist 一致（本机未发布该版本，需在打 tag 前复算）。

## [Testing]

本轮审查的可复现验证清单（全部在 `hea` 环境执行；命令与输出写入 `reviews/evidence/`）：

1. 离线门禁复现（与 .github/workflows/ci.yml 一致）：
   `python -m pytest test/ packages/jinwu-swift/tests -p no:cacheprovider -m 'not network and not heasoft and not real_data' -ra --strict-markers`
2. 导入冒烟：对 6 个发行包全部模块执行 `pkgutil.walk_packages` + `importlib.import_module`，统计失败数（对照提交信息中的“79 模块导入冒烟 0 失败”）。
3. wheel-gate 本地复现：逐包 `python -m build`（输出到独立目录）→ 干净 venv 安装全部 wheel → 断言 `jinwu.instruments` entry points 含 `{ep,swift,fermi,gw}`、`inspect.getsourcefile(jinwu.core.timescale)` 为真实文件 → 运行 gate 测试 `test/test_quickstart_examples.py test/test_stage_code_deps.py test/test_gr.py`。
4. 文档构建：`sphinx-build -b html docs <tmp-out>`（RTD 配置 `fail_on_warning: false`，因此必须人工检查 toctree/交叉引用告警），重点核对 `usage/gbm_subthreshold`、`usage/gw_coverage`、`usage/absorption_budget` 的引用完整性。
5. 真实数据验收（沿用既有产物，不覆盖原结果，输出到独立目录）：
   - OGIP：EP250111a 相关读写与 `ftverify` 对照（hea 具 HEASoft）。
   - GBM 定向搜索：`conda run -n hea python scripts/validate_gbm_subthreshold.py --root .runtime/gbm-search-validation/data --templates .runtime/gbm-search-validation/templates --output <new-dir>`，核对 `validation_summary.json` 的 status/science_status 与既有 `results/` 是否一致。
6. 算法对照验证：burstcube（HEASoft 6.37 `burstcube/lib` 源码，逐函数比对 FastNormFit 的 ΔTS/自由度/欠涨分支与 `bayesian_blocks_exposure` 的曝光加权语义）、`txx_iterbkg` 三处修复的定向数值实验（构造已知真值的合成光源，验证区间方向与尾箱保留）。
7. 安全回归：`test/test_gw_alert_security.py`（离线，monkeypatch DNS）+ 逐跳重定向校验；记录残余风险（DNS 重绑定 TOCTOU、`JINWU_GW_ALLOW_PROXY_DNS=1` 对 `198.18.0.0/15` 的显式放行）。
8. 修复阶段新增测试：
   - Makefile 目标级测试（sha256 选择器在 `dist/` 多包场景下选中核心 sdist；`publish` 清单覆盖平台 wheel）。
   - CI 收集面测试：断言 CI 实际收集的测试文件集合与“需要他机可复现”的清单一致（当前 `test/` 仅 12 个文件被跟踪，而本地存在约 45 个）。
   - 依赖方向测试：静态检查 `jinwu/fermi/**` 不得导入 `jinwu.gw`（修复后应转为 core）。
   - 文档 toctree 测试：`docs/index.rst` 引用的每个 `usage/*` 文件必须存在于被跟踪文件集合中。

已有测试可直接复用：`test/test_stage_code_deps.py`（AUD-01）、`test/test_gr.py`（AUD-03）、`test/test_quickstart_examples.py`（AUD-05）、`test/test_upperlimit_regression.py`、`test/test_txx_iterbkg_validation.py`、`test/test_gw*.py`、`packages/jinwu-swift/tests/test_bat_survey.py`（1155 行契约测试）。

## [Implementation Order]

1. 冻结审查基线：记录 `git rev-parse HEAD`、`git status --porcelain` 摘要与 `git diff --stat master...codex/beta`，写入报告开头的“审查对象快照”；审查期间不改动工作区。
2. 建立 `reviews/codex-beta-vs-master.md` 骨架（范围、方法、六维清单、证据目录约定、未验证范围声明）。
3. 执行可执行复现（[Testing] 1–4），日志落 `reviews/evidence/`，将失败项登记为发现。
4. 执行依赖与打包面核查（[Dependencies] + Makefile/CI/publish/conda recipe/版本锁步）。
5. 执行算法与数据契约核查（burstcube/OGIP/timescale/poshist/GW 覆盖），先补对照测试再下结论。
6. 汇总分级：P0（发布/CI 阻断、科学正确性、安全）、P1（交付质量与可复现性）、P2（卫生与可维护性）；每条含 file:line、复现命令、影响、建议修复。
7. 按 P0→P2 实施修复，每项一个原子提交并附回归测试；修复顺序建议：R1/R2/R3（Makefile+conda 锁步）→ R4/R5（测试跟踪与 CI 收集面）→ R6（环形依赖下沉）→ R7/R9（文档与 RTD）→ R8（uv workspace 决策）。
8. 回归：重跑 [Testing] 1–4，并按需重跑真实数据流程（第 5 项）以确认修复未改变科学结果；比较修复前后同一输入的产物哈希/关键数值。
9. 登记与交付：更新 `docs/changelog.rst`（Unreleased）、`ARCHITECTURE_ROADMAP.md`（状态标记）、`REUSABLE_FUNCTIONS.md`（新增/变更条目）；提交审查报告与证据目录；向用户汇报“流程运行 / 结果验证 / 科学解释”三类证据的完成度与未验证范围。

## 候选发现清单（审查执行时逐条确认/否决并填证据）

状态含义：**已核实** = 本轮已通过只读检查确认；**待核实** = 已定位嫌疑点，需在实施阶段用命令/测试确认。

| 编号 | 严重度 | 发现 | 证据（file:line / 命令） | 状态 | 建议修复 |
| --- | --- | --- | --- | --- | --- |
| R1 | P2 | `Makefile` `sha256` 目标存在失效的重复赋值行：第 64 行 `grep -v jinwu-` 会过滤掉全部候选（死代码），实际生效的是第 65 行 | `Makefile:64-65` | 已核实 | 删除第 64 行；未匹配核心 sdist 时报错 |
| R2 | P1 | `make publish` 上传清单 `dist/*.tar.gz dist/*-py3-none-any.whl` 不含平台 wheel（`jinwurs` 的 manylinux/macos 轮），且 `publish` 无 `rust-build` 依赖 → 本地一键发布漏发 Rust 扩展 | `Makefile:81-83` | 已核实 | 上传全部产物或显式提示只发 Python 包并改由 CI 负责 Rust 轮 |
| R3 | P1 | `packages/jinwurs/conda-recipe/meta.yaml` 版本仍为 0.1.0（实际 0.2.0）、`sha256: placeholder`；`make sync` 覆盖面不含该文件 | `packages/jinwurs/conda-recipe/meta.yaml:2,10`、`Makefile:97-109` | 已核实 | 纳入 sync 覆盖；占位 sha256 改为显式失败或注记 |
| R4 | P1 | `.gitignore` 对 `test/*` 全忽略 + 逐条反向白名单，新增测试默认不被跟踪（必须 `git add -f`）；当前 12/45 个测试文件被跟踪 | `.gitignore`（`test/*`、`!test/test_gw*.py` 等） | 已核实 | 改为目录级显式规则或列出全部受跟踪测试 |
| R5 | P1 | CI 门禁的实际收集面只有 12（`test/`）+4（`packages/jinwu-swift/tests/`）个被跟踪文件；提交信息中的“939 passed/42 skipped”来自本地未跟踪测试集，他机不可复现 | `.github/workflows/ci.yml`（`pytest test/ packages/jinwu-swift/tests`）、`git ls-files test/` | 已核实 | 决定哪些测试进入仓库并作为门禁；报告需区分“本机验证”与“仓库可复现验证” |
| R6 | P1 | 发行包环形依赖：`jinwu-gw` → `jinwu-fermi`（硬依赖），`jinwu-fermi[search-skymap]` → `jinwu-gw`，且 `subthreshold/search.py:104` 直接导入 `jinwu.gw.skymap` | `packages/jinwu-gw/pyproject.toml`、`packages/jinwu-fermi/pyproject.toml`（`search-skymap` extra）、`packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/search.py:104` | 已核实 | 通用 HEALPix 概率图读取下沉 `jinwu.core`，两侧保留转发别名 |
| R7 | P1 | `docs/index.rst` toctree 引用 `usage/gbm_subthreshold`（当前未跟踪）与 `usage/gw_coverage`（staged）；`docs/usage/absorption_budget.md` 未提交也未入目录 → RTD 构建/发布存在缺页风险 | `docs/index.rst:20-31`、`git status` | 已核实 | 提交缺页文档或从 toctree 摘除；新增 toctree 完整性检查 |
| R8 | P2 | 工作区删除了 HEAD 中存在的 `[tool.uv.workspace]`（根 `pyproject.toml`，unstaged −3 行），仓库未见 uv 使用说明 → 需确认是刻意移除还是误删 | `pyproject.toml`（工作区 vs `HEAD:pyproject.toml`） | 已核实 | 明确决策并记录（若保留 uv 工作流则恢复） |
| R9 | P2 | `.readthedocs.yaml` 只安装 4 个 Python 发行包，未安装 `packages/jinwu-gw`；若 `docs/api.rst`/新页引用 `jinwu.gw`，RTD 侧将缺失或 mock 不一致 | `.readthedocs.yaml`、`docs/api.rst` | 待核实 | 按文档引用面补充安装 `packages/jinwu-gw` |
| R10 | P1 | `jinwu.instruments` entry-point 协议把非仪器包 `jinwu.gw` 注册为仪器插件，`publish.yml` wheel-gate 断言 `{ep,swift,fermi,gw}`；协议语义（“仪器” vs “管线”）未在文档中定义 | `packages/jinwu-gw/pyproject.toml`（`[project.entry-points."jinwu.instruments"] gw`）、`.github/workflows/publish.yml` | 已核实 | 明确协议语义并写入文档，或改名/新增管线协议 |
| R11 | P2 | 关键路径异常吞错面大：非 vendor 代码中 `except Exception` 共 358 处、`except ...: pass` 形态 98 处，需确认科学路径无静默降级（对照 AUD 类问题模式） | `grep -rn 'except Exception' packages/*/src`（非 `_vendor`） | 待核实 | 对算法/管线关键路径做定向审计，区分“可降级”与“必须报错” |
| R12 | P2 | 超大单文件：`swift/bat/survey.py` 6853 行、`fermi/gbm/pipeline.py` 4443 行、`core/fit.py` 4049 行、`core/upperlimit.py` 3259 行；可维护性与审查成本高 | `wc -l` 结果 | 已核实 | 仅作建议性记录（不阻塞发布），必要时按 AGENTS.md 的模块拆分原则提出方案 |
| R13 | P2 | 上游 vendor 合规：`subthreshold/_vendor` 为 Apache-2.0 上游（`UPSTREAM.json` 记录 repository/commit/原始 sha256 + `license.txt`），已在 pyproject `package-data` 声明；需核对 wheel 内实际包含且 GPL-3.0 主包的分发说明完整 | `packages/jinwu-fermi/src/jinwu/fermi/gbm/subthreshold/_vendor/{UPSTREAM.json,license.txt}`、`packages/jinwu-fermi/pyproject.toml` | 待核实 | 构建 wheel 后核对内容清单与 LICENSE/NOTICE 说明 |
| R14 | P1 | CI 依赖面与工具链：`astro-gdt` 拉 cartopy/healpy/statsmodels，`swiftbat` 要求 `astropy>=8`，`jinwu-gw` 拉 `ligo.skymap`/`mocpy`/`ligo-gracedb`；3 个 Python 版本的安装时长与失败面需实测（含 astropy 8 与 core `astropy>=7.1` 声明的兼容性核对） | PyPI 元数据核实 + `.github/workflows/ci.yml` | 待核实 | 拆分 job / 缓存 / 收紧版本约束并复测矩阵 |
| R15 | P0 | 科学算法正确性需对照验证（burstcube 移植、OGIP 6.37 对齐、txx_iterbkg 三处修复、GW 概率积分）：现有测试是否足以定值断言，尚未逐项确认；若存在静默偏差属发布阻断项 | `core/{upperlimit,ops,timescale,ogip}.py`、HEASoft 6.37 源码、`test/test_upperlimit_regression.py`、`test/test_txx_iterbkg_validation.py` | 待核实 | 先做对照与数值实验，再决定是否修改实现 |
| R16 | P0 | 安全：SSRF 防护实现为逐跳校验（`allow_redirects=False`）已实现；残余风险点为 DNS 重绑定 TOCTOU 与 `JINWU_GW_ALLOW_PROXY_DNS=1` 显式放行 `198.18.0.0/15` 的文档化/默认关闭验证 | `packages/jinwu-gw/src/jinwu/gw/alert.py:44-66,140-180`、`test/test_gw_alert_security.py` | 待核实 | 补文档说明与“默认拒绝对保留段解析”的回归用例 |
