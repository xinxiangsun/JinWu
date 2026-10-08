"""Render the documented redshift-validation path; no analysis is executed."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from jinwu.core.plotstyle import apply_style

if __name__ == '__main__':
    apply_style()
    # Put an installed CJK font first for text; mathtext keeps DejaVu.
    cjk_fonts = sorted({f.name for f in font_manager.fontManager.ttflist if 'CJK' in f.name})
    if cjk_fonts:
        plt.rcParams['font.sans-serif'] = [cjk_fonts[0], 'DejaVu Sans']
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.5, .965, '红移模拟与验证', ha='center', va='center', fontsize=18, weight='bold')
    blocks = [
        ('输入与源假设', '模型、参数、响应、曝光、背景、cosmology 与随机种子'),
        ('时间和光子谱变换', r'$r=(1+z_2)/(1+z_1),\quad d=[D_L(z_1)/D_L(z_2)]^2$'+'\n'+r'$N_2(E,t)=d\,r^2 N_1(rE,t/r)$'),
        ('模型结构与吸收', '区分 powerlaw / zpowerlw，更新红移与前景/源吸收'),
        ('响应折叠与随机实现', '保留源/背景期望；按有效曝光生成 ON/OFF 数据'),
        ('检测规则与校准', '背景实验、时间窗口、阈值、试验次数与效率'),
        ('诊断与结果', '独立比较、模型敏感性、测量/上限/未验证状态'),
    ]
    for i, (title, detail) in enumerate(blocks):
        y = .80 - i * .145
        ax.add_patch(FancyBboxPatch((.045, y), .91, .112, boxstyle='round,pad=0.01', facecolor='#eaf4fa', edgecolor='#30759a'))
        ax.text(.075, y+.09, title, fontsize=13, weight='bold', va='top')
        ax.text(.075, y+.043, detail, fontsize=10, va='center')
        if i < len(blocks)-1:
            ax.annotate('', xy=(.5, y-.026), xytext=(.5,y-.011), arrowprops={'arrowstyle':'->','color':'#30759a'})
    fig.subplots_adjust(left=.02, right=.98, top=.99, bottom=.01)
    target = Path(__file__).resolve().parent / '_static'
    target.mkdir(exist_ok=True)
    fig.savefig(target / 'redshift_transform.png', dpi=160)
    fig.savefig(target / 'redshift_transform.svg')
    plt.close(fig)
