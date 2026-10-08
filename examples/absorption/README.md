# 吸收截面与光深示例 / Absorption examples

- **`absorption_budget.ipynb`**：含已执行图件的双语交互教程。
  Bilingual interactive tutorial with executed figures.
- **`absorption_budget.py`**：同一流程的独立脚本。
  Standalone script demonstrating the same workflow.

模型为原生 zTBabs/TBabs 与 wilm 丰度。能量横轴线性、纵轴对数；标记 tau=1。
Native zTBabs/TBabs models with wilm abundances; linear energy and logarithmic
vertical axes, including tau=1. 示例 NH 不是观测测量 / Example NH is not measured.

```bash
conda run -n hea python examples/absorption/absorption_budget.py --output /tmp/my_opacity_example
# 快速检查 / Quick check
conda run -n hea python examples/absorption/absorption_budget.py --points 30 --output /tmp/my_opacity_smoke
```

选择 hea 内核运行 Notebook；函数在独立子进程初始化 HEASoft。
Select the hea kernel for the Notebook; the API initializes HEASoft in a child process.
脚本和 Notebook 均调用 show()；无界面运行时可设置 MPLBACKEND=Agg。
Both examples call show(); use MPLBACKEND=Agg for headless script execution.
PNG/PDF、JSON、CSV、Markdown 保存到每次新建的输出目录，不覆盖旧结果。
PNG/PDF, JSON, CSV and Markdown go into a fresh output directory without overwriting results.

自动化测试仍位于 `test/test_absorption_budget.py`、`test/test_absorption_plot.py`。
Automated tests remain in `test/`, separate from examples; Git ignore rules for test are unchanged.
