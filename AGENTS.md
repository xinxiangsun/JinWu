# JinWu agent guidance

## 任务规模与简洁性

- 用户偏好：避免小题大作。少量数据绘图、排版和小范围修改，优先短脚本或
  直接修改现有文件，尽快交付结果，说明保持简短。
- 代码包开发同样要求短代码、简洁明了、避免过度工程化。新增功能和修复
  优先在现有模块直接实现；仅为实际需求增加抽象、配置或扩展机制，
  不为假想的扩展性增加类层级、适配层或通用框架。
- 代码保持简洁可读；不为一次性需求新增通用框架、配置层、过多辅助函数、
  测试套件、报告或多份元数据。不以机械压行数牺牲正确性和可读性。
- 新增抽象必须在改动说明中写明当前用途、实际调用者或独立职责，以及相对
  直接实现的收益；仅有未来扩展设想不足以支持新增抽象。
- 完成修改后检查能否用更直接的结构表达，合并无意义的转发和层级；保留
  科学语义、必要校验和公共接口，不以减少行数代替提高可读性。
- 先做一次有针对性的复用查找，避免无关仓库探索和扩展任务范围。仅有明确
  复用价值的函数需要登记；一次性任务入口无需为登记而抽象。
- 验证与影响相匹配。只用给定数据绘图时，核对数值、误差、上限方向并检查
  最终版式即可；分析算法或数据处理改变时，仍需适当真实数据验证。
- 简单且已完成的一次性任务默认不新增进度笔记或正式报告。跨会话续做、
  受阻或实质科学/工程改动需要留档时，在同一篇笔记简短更新并署名。
- 下文复用、登记、验证和留档约定按上述任务规模适用；结果真实性、原始
  数据保护和必要验证始终必须遵守。

## Progress and AI attribution

- 需要跨会话续做或有实质科学/工程改动的任务，在暂停、受阻或重要里程碑时，
  更新现有 Obsidian 笔记库的
  `Progress/` 文件夹：同一任务维护一篇可独立续做的摘要，更新顶部状态、追加
  带日期的记录，并同步 `00-进度索引.md`。仓库内保留代码、正式审查报告和科学
  证据原件；Obsidian 笔记链接到原件，不逐条记录日常对话。
- 需要留档时，每名参与修改的 AI 按每项修改任务、独立 review 的 AI 按每项 review，
  分别在进度笔记留下署名；正式 review 报告也应分别署名。不把模型信息散写进
  生产代码。
- 每条署名写明 AI 名称或子代理标识、harness、运行环境实际报告的模型标识及
  推理档位、带时区的记录时间（本机使用 `+08:00`）、修改或审查范围、结果与
  证据。历史实际工作时间若无法还原，写“历史时间不详”；模型或档位未由
  运行环境披露时写“未披露”，不能根据派工偏好或文件修改时间猜测。

## Python environment

日常开发暂以 Python 3.12 为基准。这是当前工作环境约定，不是永久架构限制。
包的 Python 支持范围以当前各包 `pyproject.toml` 和兼容性验证配置为准；
引入新版本专属要求前，检查兼容性并说明迁移理由。
天文分析默认使用 `hea`；隔离安装、兼容性验证和新后端实验可采用其他合适环境。
本机 `hea` 中已安装 HEASoft、XSPEC 和 PyXspec；执行 `conda activate hea` 后
即可使用 HEASoft 命令、`xspec` 和 Python 的 `import xspec`，无需默认追加手工初始化。
先激活环境再运行或检查；未激活时找不到命令不代表未安装，不应据此重新安装。
若激活后仍报错，依据实际报错排查环境。

## Scientific models and architecture

Prefer existing, validated spectral models from astromodels, threeML, XSPEC,
or other established scientific packages when they satisfy the scientific and
computational requirements. Search the current repository before adding logic.
Do not duplicate an implementation merely for stylistic reasons.

Reimplementation or an alternative implementation is appropriate when there is
a concrete scientific or technical advantage, including independent validation,
new physical assumptions, automatic differentiation or JAX compatibility,
vectorization or accelerator support, performance, free-threading or parallel
execution, numerical stability, XSPEC/HEASoft compatibility workarounds, or
limitations of the existing implementation. Explain the reason and, when
practical, compare against the established implementation.

New functionality should usually integrate through adapters, workflows, plugins,
visualization utilities, or other suitable existing extension points. This is
an architectural default, not a restriction to glue code. Add or revise core
algorithms and abstractions when scientific or technical requirements justify it;
explain the tradeoff and affected interfaces.

Instrument plugins must not contain source-model definitions unless intrinsically
instrument-specific, such as instrument-specific forward or calibration models.
Keep generally reusable source models in the appropriate shared layer.

Preserve public API compatibility unless explicitly approved or part of a
major-version change. For breaking changes, identify affected callers, explain
the migration path, and update relevant examples and documentation.

## Real-data validation

涉及科学分析或数据处理行为的改动，主动寻找适用的真实数据：先查本地数据、
已有示例和分析产物；不足时查找并获取公开观测数据。跑通受影响的完整流程，
核查关键中间产物、最终结果和诊断信息，并与可用参考结果进行比较。

记录来源、观测标识、环境、参数、运行入口和输出位置；使用独立输出目录，
保留原始数据与既有结果。分别报告流程运行、结果验证和科学解释的证据。
无法完成时说明已尝试的途径、阻碍和未验证范围，不把部分验收报告为完成。

不强制使用 pytest 或 ruff；按需选择回归测试、模拟、合成数据或静态检查，
补充边界条件和已知故障的覆盖。现有明确的 CI/发布门禁仍须满足，
但离线检查通过不能替代真实数据流程验收。纯文档改动无需重跑科学流程。

## Exploration and independent technical judgment

### Code migration acceptance

For code migrations, preserve the original algorithm core, scientific
computation procedure, intermediate quantities, and final results. Pin and
record the source version, inputs, settings, environment, and randomness;
compare the original and migrated implementations on identical inputs,
including representative real data. State numerical tolerances and explain
every difference. Interface, packaging, and execution changes are acceptable
when these comparisons establish equivalence. Algorithm corrections, new
scientific assumptions, and upstream version updates must be separate changes
with their own validation, rather than silently included in a migration.

For exploratory, scientific, architectural, or open-ended tasks, apply the
global independent-judgment principles: treat historical conventions as defaults,
check stale assumptions, and consider materially different approaches for important
decisions. Prefer scientific correctness, reproducibility, and technical merit
over preserving historical choices; surface better architectures when justified.
