"""TBabs/wilm 单点查询、丰度调整与绘图示例。

TBabs/wilm point queries, abundance adjustments and plotting examples.
输入能量为静止系，NH 为演示输入；不读取观测项目或仪器响应。
Energies are rest-frame and NH is illustrative; no project data or responses.
"""
import argparse
from pathlib import Path


def main():
    """运行示例并保存到新目录 / Run examples into a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='新输出目录 / New output directory')
    parser.add_argument('--points', type=int, default=600, help='曲线采样数 / Curve sample count')
    args = parser.parse_args()
    if args.points < 2:
        parser.error('--points must be at least 2 / 采样数至少为 2')
    from datetime import datetime, timezone
    import numpy as np
    import astropy.units as u
    from jinwu.physics.absorption import absorption_budget, AbsorptionBudget

    # 每次运行创建独立输出目录，保留已有结果。
    # Create a unique output directory on every run, preserving previous results.
    output = args.output or (Path.cwd() / 'absorption_tutorial_outputs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    nh = 1e22 / u.cm**2
    shown = ['H', 'He', 'O', 'Ne', 'Si', 'Fe']

    # 不默认采用任何观测的 NH；显式指定模型及丰度。
    # Specify NH, model and abundance explicitly, independently of observations.
    points = absorption_budget([1, 6.4, 6.7]*u.keV, nh,
                               backend='ztbabs', abundance_table='wilm')
    print(points.totals)
    print(points.elements[np.isin(points.elements['element'], shown)])

    # 静止系能量；避免重复施加红移。
    # Rest-frame energies; do not apply a second redshift.
    energy = np.linspace(.3, 10, args.points)*u.keV
    curve = absorption_budget(energy, nh, backend='ztbabs', abundance_table='wilm')
    curve.plot(elements=shown, energy_scale='linear', fraction_scale='log')
    curve.savefig(output/'ztbabs_wilm')
    curve.show()  # 显示图件 / Display the figure.

    # 元素因子修改数丰度，不切换到原子吸收后端。
    # Element factors alter number abundances without switching absorption backends.
    modified = absorption_budget(energy, nh, backend='ztbabs', abundance_table='wilm',
                                 element_factors={'O': .5, 'Fe': 2.})
    modified.plot(elements=shown, energy_scale='linear', fraction_scale='log')
    modified.savefig(output/'ztbabs_modified')
    modified.show()  # 显示图件 / Display the figure.

    # 保留原生 TBabs；不能通过更换模型隐藏闭合失败。
    # Retain native TBabs rather than hide failed closure by substituting models.
    tb = absorption_budget(energy, nh, backend='tbabs', abundance_table='wilm')
    tb.plot(elements=shown, energy_scale='linear', fraction_scale='log')
    tb.savefig(output/'tbabs_wilm')
    tb.show()  # 显示图件 / Display the figure.
    print('状态 / Status:', set(tb.totals['status']))

    # 单点表格和完整曲线分别导出。
    # Export point tables and the complete curve separately.
    points.to_csv(output/'points')
    points.to_markdown(output/'points.md')
    curve.to_json(output/'curve.json')
    restored = AbsorptionBudget.from_json(output/'curve.json')
    np.testing.assert_allclose(restored.totals['tau_total'], curve.totals['tau_total'])
    print('输出目录 / Output directory:', output)


if __name__ == '__main__':
    main()
