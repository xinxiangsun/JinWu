# coordinates 示例 / Examples

每个 Notebook 配有同名 Python 脚本；新增配置与说明使用中英双语。
Each Notebook has a paired script; new configuration and guidance are bilingual.

| 示例 / Example | 状态 / Status | 说明 / Notes |
| --- | --- | --- |
| [teldef.ipynb](teldef.ipynb) | needs_review | 需要 test/data 中的 teldef 校准文件 / Requires the teldef calibration fixture |

选择 hea 内核。输入根目录可通过 JINWU_EXAMPLE_RESEARCH_ROOT、JINWU_EXAMPLE_TEST_ROOT、JINWU_EXAMPLE_DOWNLOAD_ROOT 覆盖。
Use the hea kernel. Override input roots with the JINWU_EXAMPLE_* environment variables documented in each setup cell.

外部数据依赖与已知草稿状态没有因迁移自动解决。历史输出不代表本次执行成功。
External data requirements and draft limitations remain; historical outputs do not establish a successful new run.

试运行状态 / Smoke status: 脚本完成；坐标往返结果不一致，不能视作校准验证通过。
Script completed, but coordinate round-trip values disagree; this is not a calibration pass.
输入 / Input: (180.001, 45.0) deg; 返回 / Returned: (234.6492374636717, 29.90758495085327) deg.
迁移只更新模块路径，未改动坐标算法。Only the import path was updated, not the coordinate algorithm.
