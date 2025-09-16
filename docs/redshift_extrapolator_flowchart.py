import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import font_manager as _fm
import glob
import os

# 尝试自动配置中文字体，避免 DejaVu Sans 缺字警告
def _setup_chinese_font():
    # 常见中文字体候选（按优先级）
    candidates = [
        'Noto Sans CJK SC', 'Noto Sans CJK SC Regular', 'Noto Sans CJK',
        'Noto Sans SC', 'Noto Serif SC', 'Noto Sans',
        'Source Han Sans SC', 'Source Han Serif SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC',
    ]
    try:
        # 允许从项目内置字体目录加载字体（无需系统安装）
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        if os.path.isdir(fonts_dir):
            for fp in glob.glob(os.path.join(fonts_dir, '*.[o,t]tf')) + glob.glob(os.path.join(fonts_dir, '*.ttc')):
                try:
                    _fm.fontManager.addfont(fp)
                except Exception:
                    pass

        # 收集系统字体列表
        font_names = {f.name for f in _fm.fontManager.ttflist}
        chosen = None
        for name in candidates:
            if name in font_names:
                chosen = name
                break
        if chosen:
            plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            return True
        else:
            # 未找到中文字体，尽量不报错，同时提示用户可安装字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            return False
    except Exception:
        # 出现异常也保证脚本能继续执行
        return False

_HAS_ZH_FONT = _setup_chinese_font()


fig, ax = plt.subplots(figsize=(10, 16))
ax.axis('off')

if _HAS_ZH_FONT:
    # 中文标签（加入换行以优化竖版排版）
    blocks = [
        ("输入参数", "模型、参数、响应文件、\n计数、曝光、面积比、z0"),
        ("模型初始化", "XSPEC模型 + 参数注入\n冻结部分参数"),
        ("红移网格生成", "z_grid = linspace(z0, z0+Δz)"),
        ("归一化修正", "norm_new = norm0 × ((1+z0)/(1+z))^α\n× (r_c²(z0)/r_c²(z))"),
        ("谱卷积", "SOXS: 模型→能谱→响应卷积→计数率"),
        ("SNR计算", "Li & Ma 公式：SNR = sqrt{...}"),
        ("自适应搜索", "细分 z 区间；插值 SNR→z_max"),
        ("输出结果", "最大可探测红移 z_max")
    ]
else:
    # 英文降级标签（避免中文字体缺失告警）
    blocks = [
        ("Inputs", "Model, params, RSP,\ncounts, exposure, alpha, z0"),
        ("Init Model", "XSPEC model + set params\nFreeze constants"),
        ("Redshift Grid", "z_grid = linspace(z0, z0+Δz)"),
        ("Norm Scaling", "norm_new = norm0 × ((1+z0)/(1+z))^α\n× (r_c²(z0)/r_c²(z))"),
        ("Convolution", "SOXS: model→spectrum→RSP conv→count rate"),
        ("SNR Compute", "Li & Ma: SNR = sqrt{...}"),
        ("Adaptive Search", "Refine z-interval; interpolate SNR→z_max"),
        ("Outputs", "Max detectable redshift z_max")
    ]

# 竖向布局参数
x0, y0 = 0.08, 0.90
w, h, dy = 0.84, 0.09, 0.11

# 标题
title_text = "红移外推流程" if _HAS_ZH_FONT else "Redshift Extrapolation Flow"
ax.text(0.5, 0.96, title_text, fontsize=20, fontweight='bold', color="#004d40", ha='center', va='center')

for i, (title, desc) in enumerate(blocks):
    y = y0 - i * dy
    # 块
    ax.add_patch(mpatches.FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.02", fc="#e0f7fa", ec="#00838f", lw=2))
    # 标题
    ax.text(x0+0.015, y+h-0.02, title, fontsize=16, fontweight='bold', color="#006064", va='top', ha='left')
    # 描述
    ax.text(x0+0.015, y+0.02, desc, fontsize=13, color="#333333", va='bottom', ha='left')
    # 向下箭头
    if i < len(blocks)-1:
        y_next = y0 - (i+1) * dy
        ax.annotate('', xy=(x0+w/2, y_next + h), xytext=(x0+w/2, y), arrowprops=dict(arrowstyle="->", lw=2, color="#00838f"))

# 公式区（底部）
ax.text(0.08, 0.08, "物理推导核心公式：" if _HAS_ZH_FONT else "Key formulas:", fontsize=15, fontweight='bold', color="#006064", ha='left')
ax.text(0.08, 0.055, r"$\mathbf{norm}_{\mathrm{new}} = \mathbf{norm}_0 \times \left(\frac{1+z_0}{1+z}\right)^\alpha \times \left(\frac{r_c^2(z_0)}{r_c^2(z)}\right)$", fontsize=17, color="#d84315", ha='left')
ax.text(0.08, 0.023, r"$\mathbf{SNR} = \sqrt{2\left[N_{on}\ln\frac{(1+\alpha)N_{on}}{\alpha(N_{on}+N_{off})} + N_{off}\ln\frac{(1+\alpha)N_{off}}{N_{on}+N_{off}}\right]}$", fontsize=16, color="#6a1b9a", ha='left')

plt.tight_layout()
out_png = "/home/xinxiang/research/autohea/docs/redshift_extrapolator_flowchart.png"
out_svg = "/home/xinxiang/research/autohea/docs/redshift_extrapolator_flowchart.svg"
plt.savefig(out_png, dpi=200, bbox_inches='tight')
try:
    plt.savefig(out_svg, bbox_inches='tight')
except Exception:
    pass
