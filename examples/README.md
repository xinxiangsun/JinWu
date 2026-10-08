# 示例教程 / Example tutorials

[吸收截面与光深 / Absorption cross sections and optical depth](absorption/README.md)

每个示例子目录同时提供 Notebook 和可运行 Python 脚本，中英双语注释。
Each example directory includes a Notebook and runnable Python script with bilingual comments.
自动化测试单独保留在 test；不改变其 Git 忽略规则。
Automated tests remain in test with their existing Git ignore rules.

## 迁移的其他教程 / Other migrated tutorials

| 主题 / Topic | 入口 / Entry |
| --- | --- |
| FITS 读取与仪器扫描 / FITS I/O and discovery | [io](io/README.md) |
| 坐标校准 / Teldef coordinates | [coordinates](coordinates/README.md) |
| 光变 / Light curves | [lightcurves](lightcurves/README.md) |
| 任务时间 / Mission time | [time](time/README.md) |
| 持续时间探索 / Duration exploration | [durations](durations/README.md) |
| WXT 流水线 / WXT pipeline | [wxt](wxt/README.md) |
| 事件筛选 / Event selection | [events](events/README.md) |
| 背景谱准备 / Background spectrum preparation | [spectra](spectra/README.md) |
| 拟合指南 / Fitting guide | [fitting](fitting/README.md) |
| 未完成草稿 / Incomplete drafts | [drafts](drafts/README.md) |

11 组 Notebook/Python 示例和两份旧指南已从 test 迁入。
Eleven Notebook/script pairs and two historical guides were moved from test.
请先查看各目录状态；data_required、legacy_* 和 draft 不表示已验证可运行。
Read the status labels: data_required, legacy_* and draft do not mean execution-validated.
只删除了没有单元内容的 teldef_porting_notebook.ipynb；原始数据与测试仍在 test。
Only the empty teldef porting notebook was discarded; data and tests remain in test.
迁移清单包含来源哈希 / Source hashes: [migration_manifest.json](migration_manifest.json).
