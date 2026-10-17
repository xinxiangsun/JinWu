# 光学观测表绘图

当前版本：v2。原始输入为用户粘贴的13行 `observations.csv`，9个测量、4个上限。源名称、AB/Vega系统、VT上限置信度未提供。

运行：`/home/xinxiang/miniconda3/bin/conda run -n hea python artifacts/optical_table_20261008/plot_optical.py`

脚本70行，复用 JinWu `save_figure`。导出 `optical_lightcurve_v2.png`（600 dpi）、`.pdf`、`.svg`、`_preview.png`。原版 `optical_lightcurve.*` 保留。

触发后使用对数时间主图，触发前两条上限放在线性时间插图。直接标注望远镜和滤波器，Liverpool简写LT；双色VT上限保持同一坐标。星等不校正银河系消光、不转换系统、不加显示偏移、不拟合模型。

原输入校验和、13行逐项验收见 `optical_lightcurve_v2.json` 和 `validation.json`。此次复核CSV校验和未变；新版预览检查标签、点、误差棒和上限箭头，没有重叠或截断。原图误差值不变，不独立复核测光。
