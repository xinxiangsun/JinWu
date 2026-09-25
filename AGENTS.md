# JinWu agent guidance

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

For exploratory, scientific, architectural, or open-ended tasks, apply the
global independent-judgment principles: treat historical conventions as defaults,
check stale assumptions, and consider materially different approaches for important
decisions. Prefer scientific correctness, reproducibility, and technical merit
over preserving historical choices; surface better architectures when justified.
