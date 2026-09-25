"""在隔离的 XSPEC worker 中计算按元素分解的中性 X 射线不透明度。

Element-resolved neutral X-ray opacity, evaluated in an isolated XSPEC worker.

``tbabs`` is an ISM effective-opacity model (including its H2/grain assumptions);
``atomic`` uses vphabs with Verner photoelectric cross sections, without scattering.
The decomposition generalizes EP260119a R07's H-baseline subtraction and closure
check. No target, redshift, column density, response or fit state is implicit.
References: Wilms et al. 2000, ApJ 542, 914; Verner et al. 1996, ApJ 465, 487;
https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelTbabs.html
https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelPhabs.html
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import astropy.units as u
from astropy.table import QTable
import numpy as np

__all__ = ['absorption_budget', 'AbsorptionBudget']
ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn').split()
SUPPORTED = ('H He C N O Ne Na Mg Al Si S Cl Ar Ca Cr Fe Co Ni').split()


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _table_payload(table):
    return {
        'units': {name: str(getattr(table[name], 'unit', '') or '') for name in table.colnames},
        'rows': [{name: _json_safe(getattr(table[name], 'value', table[name])[i])
                  for name in table.colnames} for i in range(len(table))],
    }


@dataclass
class AbsorptionBudget:
    """保存带单位的总量表、元素表、溯源信息以及可选的 matplotlib 图像。

    Unit-bearing total/element tables, provenance, and optional matplotlib figure.

    sigma_total and sigma_weighted are cm² per H nucleus; sigma_element is cm²
    per element nucleus, but is a model-effective coefficient for TBabs, not an
    isolated-atom cross section. Undefined fractions are NaN (JSON null).
    """
    totals: QTable
    elements: QTable
    metadata: dict
    figure: object = field(default=None, repr=False)
    axes: object = field(default=None, repr=False)

    def to_json(self, path):
        """将带单位的表格写入 JSON；未定义值写为 JSON null。

        Write tables with units; undefined values become JSON null.
        """
        Path(path).write_text(json.dumps(_json_safe(dict(
            metadata=self.metadata, totals=_table_payload(self.totals),
            elements=_table_payload(self.elements))), indent=2, allow_nan=False) + '\n')

    def to_csv(self, prefix):
        """写出 PREFIX_totals.csv、PREFIX_elements.csv 以及单位和溯源 JSON 文件。

        Write PREFIX_totals.csv, PREFIX_elements.csv and units/provenance JSON.
        """
        prefix = Path(prefix)
        for name, table in [('totals', self.totals), ('elements', self.elements)]:
            table.write(str(prefix) + '_' + name + '.csv', format='ascii.csv', overwrite=False)
        self.to_json(str(prefix) + '_metadata.json')

    def to_markdown(self, path=None):
        """返回单点查询报告；如果指定路径则同时写入文件。

        Return a point-query report; optionally write it to a file.
        """
        lines = [f"# Absorption budget: {self.metadata['backend']}", '', self.metadata['interpretation'], '',
                 'Cross sections are in cm²; weighted/total cross sections are per H nucleus. Energy is rest-frame keV. Fractions are percentages. Undefined values are —.', '']
        for title, table in [('Totals', self.totals), ('Elements', self.elements)]:
            lines += [f'## {title}', '', '| ' + ' | '.join(table.colnames) + ' |',
                      '| ' + ' | '.join(['---'] * len(table.colnames)) + ' |']
            for row in _table_payload(table)['rows']:
                vals = ['—' if v is None else (f'{v:.7g}' if isinstance(v, float) else str(v)) for v in row.values()]
                lines.append('| ' + ' | '.join(vals) + ' |')
            lines.append('')
        result = '\n'.join(lines)
        if path is not None:
            Path(path).write_text(result)
        return result

    def plot(self, elements=None, *, unweighted=False, fraction_scale="log", energy_scale="linear"):
        """绘制四个不透明度面板，或绘制单个元素的未加权截面。

        Plot four opacity panels, or individual unweighted cross sections.

        energy_scale defaults to linear; fraction_scale defaults to log.
        The tau=1 reference and sampled total-tau crossing brackets are marked.
        fraction_scale selects linear or logarithmic percentage axes. Zero and
        undefined values are omitted on logarithmic axes, never floored.
        Hidden supported elements sum into Other in the four-panel plot. All
        data remain in the tables. This method never reevaluates XSPEC.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from jinwu.core.plotstyle import PALETTE, apply_style
        if fraction_scale not in ("linear", "log"):
            raise ValueError("fraction_scale must be linear or log")
        if energy_scale not in ('linear', 'log'):
            raise ValueError('energy_scale must be linear or log')
        apply_style()
        selected = list(SUPPORTED if elements is None else elements)
        if len(set(selected)) != len(selected) or any(e not in SUPPORTED for e in selected):
            raise ValueError('Plot elements must be distinct supported element symbols')
        order = np.argsort(self.totals['energy_rest'].to_value(u.keV), kind='stable')
        energy = self.totals['energy_rest'].to_value(u.keV)[order]
        style = {'font.family': 'serif', 'mathtext.fontset': 'cm',
                 'xtick.direction': 'in', 'ytick.direction': 'in'}
        with matplotlib.rc_context(style):
            fig, axes = plt.subplots(1 if unweighted else 2, 1 if unweighted else 2,
                                     figsize=(10, 6) if unweighted else (13, 9), sharex=True)
            panels = [axes] if unweighted else list(axes.flat)
            palette = [PALETTE[k] for k in ('data', 'model', 'net', 'secondary')]
            dashes = ['-', (0, (6, 2)), (0, (2, 2)), (0, (6, 2, 1, 2)), (0, (1, 2))]
            styles = {el: dict(color=palette[i % 4], linestyle=dashes[i // 4],
                              marker=['o', 's', '^', 'D'][i % 4],
                              markevery=(i % 11, max(1, len(energy) // 12)),
                              markersize=3, markerfacecolor='white', markeredgewidth=.7)
                      for i, el in enumerate(SUPPORTED)}
            quantities = ['sigma_element'] if unweighted else ['sigma_weighted', 'sigma_fraction_pct', 'tau', 'tau_fraction_pct']
            rows = {el: self.elements[self.elements['element'] == el] for el in SUPPORTED}
            def values(el, key):
                column = rows[el][key]
                return np.asarray(getattr(column, 'value', column), float)[order]
            for panel, key in zip(panels, quantities):
                logarithmic = ('fraction' not in key or fraction_scale == 'log')
                def draw(y, label, **kwargs):
                    valid = np.isfinite(y) & ((y > 0) if logarithmic else True)
                    if np.any(valid):
                        panel.plot(energy, np.where(valid, y, np.nan), label=label,
                                   **kwargs)
                for el in selected:
                    draw(values(el, key), el, linewidth=1.5, **styles[el])
                hidden = [el for el in SUPPORTED if el not in selected]
                if hidden and not unweighted:
                    draw(np.sum([values(el, key) for el in hidden], axis=0), 'Other', color=PALETTE['background'], linestyle=(0, (3, 1, 1, 1)), linewidth=1.4)
                if key in ['sigma_weighted', 'tau']:
                    column = self.totals['sigma_total' if key == 'sigma_weighted' else 'tau_total']
                    draw(np.asarray(getattr(column, 'value', column))[order], 'Total', color=PALETTE['text'], linewidth=2.4, zorder=5)
                panel.set_xscale(energy_scale)
                if energy_scale == 'linear' and len(energy) > 1:
                    panel.set_xlim(energy.min(), energy.max())
                if logarithmic:
                    panel.set_yscale('log')
                    if 'fraction' in key and panel.lines:
                        positive = np.concatenate([line.get_ydata() for line in panel.lines])
                        positive = positive[np.isfinite(positive) & (positive > 0)]
                        if positive.size:
                            panel.set_ylim(10 ** np.floor(np.log10(positive.min())), 120)
                else:
                    panel.set_ylim(0, 100)
                if key == 'tau':
                    panel.axhline(1., color=PALETTE['reference'], linestyle=(0, (8, 3)),
                                  linewidth=1.8, zorder=10)
                    panel.text(.98, 1., r'$\tau=1$', transform=panel.get_yaxis_transform(),
                               ha='right', va='bottom', fontsize=11,
                               bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
                    total_tau = np.asarray(self.totals['tau_total'])[order]
                    for i in range(len(energy) - 1):
                        if (np.isfinite(total_tau[i:i+2]).all() and
                            (total_tau[i] - 1) * (total_tau[i+1] - 1) < 0):
                            # 这里只标出相邻采样点构成的区间，不跨吸收边插值求根。
                            # Mark only the bracket formed by adjacent samples; do not interpolate a root across an edge.
                            panel.axvspan(energy[i], energy[i+1], color=PALETTE['reference'], alpha=.16)
                panel.set_xlabel('Rest-frame energy (keV)')
                panel.tick_params(which='both', top=True, right=True)
                panel.grid(True, which='major', alpha=.18, linewidth=.6)
                panel.grid(False, which='minor')
                if not panel.get_legend_handles_labels()[0] and not unweighted:
                    panel.text(.5, .5, 'No validated element decomposition', ha='center', va='center', transform=panel.transAxes, color='darkred')
            labels = {'sigma_element': 'Element cross section (cm²)' if self.metadata['backend'] == 'atomic' else 'Isolated-composition effective coefficient (cm²)', 'sigma_weighted': 'Cross section per H nucleus (cm²)',
                      'sigma_fraction_pct': 'Cross-section contribution (%)', 'tau': 'Optical depth', 'tau_fraction_pct': 'Optical-depth contribution (%)'}
            for panel, key in zip(panels, quantities):
                panel.set_ylabel(labels[key])
            invalid = int(np.sum(self.totals['status'] != 'ok'))
            if invalid:
                fig.text(.5, .005, f'{invalid} point(s): unvalidated decomposition/edge; contributions masked. See status and diagnostics.', ha='center', fontsize=9, color='darkred')
            handles = {}
            for panel in panels:
                for handle, label in zip(*panel.get_legend_handles_labels()):
                    handles.setdefault(label, handle)
            if handles:
                labels_order = (['Total'] if 'Total' in handles else []) + [x for x in handles if x != 'Total']
                fig.legend([handles[x] for x in labels_order], labels_order,
                           loc='upper center', bbox_to_anchor=(.5, .95), ncol=6,
                           handlelength=3.8, columnspacing=1.8, fontsize=10)
            title = ('Neutral atomic opacity · vphabs / vern' if self.metadata['backend'] == 'atomic'
                     else ('Effective ISM opacity · zTBabs (rest frame)' if self.metadata['backend'] == 'ztbabs' else 'Effective ISM opacity · TBabs'))
            fig.suptitle(title + (' — log percentages' if fraction_scale == 'log' and not unweighted else ''),
                         fontsize=14, fontweight='bold', y=.995)
            fig.tight_layout(rect=(0, .035, 1, .86))
        self.figure, self.axes = fig, axes
        return fig, axes

    def savefig(self, prefix):
        """将当前或默认图像保存为 PREFIX.png（300 dpi）和 PREFIX.pdf。

        Save the current/default figure as PREFIX.png (300 dpi) and PREFIX.pdf.
        """
        if self.figure is None:
            self.plot()
        from jinwu.core.plotstyle import save_figure
        return save_figure(self.figure, prefix, formats=('png', 'pdf'), dpi=300)

    def show(self):
        """使用调用方的 Notebook 或 GUI 后端显示当前图像。

        Display the current figure using the caller's Notebook/GUI backend.
        """
        import matplotlib.pyplot as plt
        if self.figure is None:
            self.plot()
        plt.show()

    @classmethod
    def from_json(cls, path):
        """在不启动 XSPEC 的情况下，重新载入带单位且包含未定义值的导出表格。

        Reload exported tables with units and undefined values, without XSPEC.
        """
        payload = json.loads(Path(path).read_text())
        def restore(part):
            table = QTable()
            for name, unit in part['units'].items():
                values = [row[name] if row[name] is not None else np.nan for row in part['rows']]
                table[name] = np.asarray(values) * u.Unit(unit) if unit else values
            return table
        return cls(restore(payload['totals']), restore(payload['elements']), payload['metadata'])


def _validate(energy_rest, nh, backend, metallicity, element_factors, number_abundances, redshift):
    if not isinstance(energy_rest, u.Quantity) or not isinstance(nh, u.Quantity):
        raise TypeError('energy_rest and nh must be astropy Quantity objects')
    energy = np.atleast_1d(energy_rest.to_value(u.keV))
    column = np.asarray(nh.to_value(u.cm**-2))
    if energy.ndim != 1 or not energy.size or not np.all(np.isfinite(energy)) or np.any(energy < .03):
        raise ValueError('Use a nonempty scalar/1D energy array >=0.03 keV within the XSPEC X-ray model domain')
    if column.ndim != 0 or not np.isfinite(column) or column < 0:
        raise ValueError('nh must be a finite nonnegative scalar column density')
    if backend not in ['tbabs', 'ztbabs', 'atomic']:
        raise ValueError('backend must be tbabs, ztbabs or atomic')
    if not np.isfinite(metallicity) or metallicity < 0:
        raise ValueError('metallicity must be finite and nonnegative')
    if number_abundances is not None and (metallicity != 1. or element_factors is not None):
        raise ValueError('Absolute abundances cannot be combined with metallicity/element_factors')
    for mapping in [element_factors, number_abundances]:
        for el, value in (mapping or {}).items():
            if el not in ELEMENTS or not np.isfinite(value) or value < 0:
                raise ValueError('Unknown element or invalid abundance: ' + str(el))
            if el == 'H' and value != 1:
                raise ValueError('H is the fixed number-abundance normalization, 1')
            if el not in SUPPORTED and value != 0:
                raise ValueError(f'{el} is not supported by {backend}; nonzero opacity cannot be supplied')
    if redshift is not None and (not np.isfinite(redshift) or redshift < 0):
        raise ValueError('redshift must be finite and nonnegative')
    return energy, float(column)


def absorption_budget(energy_rest, nh, *, backend='tbabs', abundance_table='wilm',
                      metallicity=1., element_factors=None, number_abundances=None,
                      redshift=None, plot=False):
    """计算标量或数组形式静止系能量处的中性元素不透明度。

    Compute neutral element opacity at scalar/array rest-frame energies.

    Parameters
    ----------
    energy_rest : astropy.units.Quantity
        Scalar or 1D energy, convertible to keV; input order/duplicates preserved.
    nh : astropy.units.Quantity
        Explicit nonnegative scalar H-nucleus column density in cm^-2.
    backend : {'tbabs', 'ztbabs', 'atomic'}
        TBabs or zTBabs effective opacity, or vphabs/vern atomic opacity.
        zTBabs is evaluated at z=0 because input energies are already rest-frame;
        it retains its native grain convention, distinct from TBabs.
    abundance_table : str
        Installed XSPEC abundance table name, default wilm (number ratios to H).
    metallicity : float
        Multiplier for Z>=3; element_factors multiply after this scaling.
    element_factors : dict[str, float], optional
        Relative multipliers; H must remain 1. He may vary independently.
    number_abundances : dict[str, float], optional
        Absolute n_i/n_H, omitted elements zero and H fixed at 1. Mutually
        exclusive with non-default metallicity and any element_factors.
    redshift : float, optional
        Only adds observed energies E_rest/(1+z); no change in absorption.
    plot : bool
        Attach a four-panel matplotlib figure; never writes files implicitly.

    Returns
    -------
    AbsorptionBudget
        Unit-bearing tables and provenance. Unsupported model elements are
        explicitly flagged. Invalid decompositions are masked, not normalized.
        NH=0 has zero tau and undefined tau fractions. No statistical errors are
        assigned to atomic/model differences.

    Notes
    -----
    Run from an HEASoft-capable Python environment. A fresh Python subprocess
    initializes HEASoft, with private PFILES, leaving the caller's XSPEC state
    untouched. Energy-bin convergence tolerance is 1e-4, closure tolerance 1e-6.
    """
    energy, column = _validate(energy_rest, nh, backend, metallicity, element_factors, number_abundances, redshift)
    request = dict(energy=energy.tolist(), nh=column, backend=backend, abundance_table=abundance_table,
                   metallicity=metallicity, element_factors=element_factors,
                   number_abundances=number_abundances, redshift=redshift)
    root = Path(os.environ.get('HEADAS', Path(sys.prefix) / 'heasoft'))
    if not (root / 'headas-init.sh').is_file():
        raise RuntimeError('HEASoft initialization not found; initialize HEADAS in your hea environment')
    with tempfile.TemporaryDirectory(prefix='jinwu-opacity-') as temporary:
        directory = Path(temporary)
        request_path, output_path = directory/'request.json', directory/'result.json'
        request_path.write_text(json.dumps(request))
        env = dict(os.environ, HEADAS=str(root), JINWU_ABSORPTION_PFILES=str(directory))
        for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
            env[key] = '1'
        command = ['bash', '-c', 'source "$HEADAS/headas-init.sh" || exit $?; export PFILES="$JINWU_ABSORPTION_PFILES;$HEADAS/syspfiles"; exec "$@"',
                   'jinwu-opacity', sys.executable, str(Path(__file__).resolve()), str(request_path), str(output_path)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
        if completed.returncode or not output_path.exists():
            raise RuntimeError('Isolated XSPEC opacity calculation failed:\n' + completed.stderr[-6000:] + completed.stdout[-2000:])
        raw = json.loads(output_path.read_text())
        raw['metadata']['native_runtime_log'] = completed.stdout
    result = _result_tables(request, raw)
    if plot:
        result.plot()
    return result


def _result_tables(request, raw):
    energy = np.asarray(request['energy'])
    count = len(energy)
    nh = request['nh']
    total = np.asarray(raw['sigma_total'])
    coefficient = np.asarray(raw['sigma_element'], dtype=float)
    abundance = np.asarray(raw['abundance'])
    weighted = coefficient * abundance[None, :]
    valid = np.asarray(raw['closure_ok']) & np.asarray(raw['point_converged'])
    # 分解无效时保留 NaN，避免把不可靠的归因伪装成看似合理的归一化结果。
    # Preserve NaN for invalid decompositions instead of presenting an unreliable attribution as normalized.
    weighted[~valid, :] = np.nan
    fraction = np.divide(weighted, total[:, None], out=np.full_like(weighted, np.nan), where=total[:, None] > 0) * 100
    tau = weighted * nh
    if nh == 0:
        tau[:] = 0.
    tau_fraction = fraction.copy() if nh > 0 else np.full_like(fraction, np.nan)
    totals = QTable()
    totals['point_index'] = np.arange(count)
    totals['energy_rest'] = energy * u.keV
    if request['redshift'] is not None:
        totals['energy_observed'] = energy / (1 + request['redshift']) * u.keV
    totals['sigma_total'] = total * u.cm**2
    totals['tau_total'] = total * nh
    totals['transmission'] = np.exp(-total * nh)
    totals['sample_lower'] = np.asarray(raw['sample_lower']) * u.keV
    totals['sample_upper'] = np.asarray(raw['sample_upper']) * u.keV
    totals['closure_relative_error'] = raw['closure_relative_error']
    totals['point_relative_change'] = raw['point_relative_change']
    totals['status'] = ['ok' if v else ('point_not_converged' if not c else 'decomposition_not_closed') for v, c in zip(valid, raw['point_converged'])]
    elements = QTable()
    elements['point_index'] = np.repeat(np.arange(count), len(ELEMENTS))
    elements['energy_rest'] = np.repeat(energy, len(ELEMENTS)) * u.keV
    elements['element'] = np.tile(ELEMENTS, count)
    elements['number_abundance'] = np.tile(abundance, count)
    elements['sigma_element'] = coefficient.reshape(-1) * u.cm**2
    elements['sigma_weighted'] = weighted.reshape(-1) * u.cm**2
    elements['sigma_fraction_pct'] = fraction.reshape(-1)
    elements['tau'] = tau.reshape(-1)
    elements['tau_fraction_pct'] = tau_fraction.reshape(-1)
    elements['status'] = [
        'unsupported_by_backend' if el not in SUPPORTED else ('ok' if valid[i] else 'decomposition_invalid')
        for i in range(count) for el in ELEMENTS]
    elements['coefficient_kind'] = [
        'unsupported' if el not in SUPPORTED else
        ('isolated_neutral_atom' if request['backend'] == 'atomic' else ('effective_H_including_model_H2' if el == 'H' else 'effective_ISM_element'))
        for _ in range(count) for el in ELEMENTS]
    metadata = dict(raw['metadata'], nh_cm2=nh, redshift=request['redshift'],
                    query=request, tolerance=dict(closure=1e-6, point=1e-4),
                    fraction_convention='Weighted cross-section/total; tau fractions identical for NH>0, undefined at NH=0',
                    statistical_uncertainty='Not supplied: deterministic model coefficients, not fitted measurements',
                    unsupported_elements=[e for e in ELEMENTS if e not in SUPPORTED])
    return AbsorptionBudget(totals, elements, metadata)


def _abundances(request, table_path):
    name = request['abundance_table']
    lines = table_path.read_text().splitlines()
    matching = [line for line in lines if line.startswith(name + ':') and len(line.split(':', 1)[1].split()) == 30]
    if len(matching) != 1:
        raise ValueError(f'Unknown installed abundance table: {name}')
    base = np.array([float(v) for v in matching[0].split(':', 1)[1].split()])
    if base.size != 30 or not np.isfinite(base).all():
        raise ValueError('Invalid 30-element abundance table')
    if request['number_abundances'] is not None:
        values = np.array([request['number_abundances'].get(e, 0.) for e in ELEMENTS])
    else:
        values = base.copy()
        # 先对 Li 及更重元素施加整体金属丰度缩放；H、He 不受 metallicity 影响。
        # First apply the global metallicity scaling to Li and heavier elements; H and He are unaffected.
        values[2:] *= request['metallicity']
        # 再施加逐元素相对因子；最终值为基础丰度乘以适用的缩放因子。
        # Then apply per-element relative factors; the final value is the base abundance times the applicable factors.
        for el, factor in (request['element_factors'] or {}).items():
            values[ELEMENTS.index(el)] *= factor
    values[0] = 1.
    if not np.isfinite(values).all():
        raise ValueError('Abundance scaling overflowed; use finite number ratios')
    return values, base


class _NativeOpacity:
    """仅供 worker 使用的原生模型计算器；调用方不会实例化它。

    Worker-only native model evaluator; never instantiated in the caller.
    """
    def __init__(self, xspec, backend, directory):
        self.x = xspec
        self.backend = backend
        self.name = {'tbabs': 'TBabs', 'ztbabs': 'zTBabs', 'atomic': 'vphabs'}[backend]
        self.directory = directory
        self.counter = 0

    def _call(self, edges, column):
        params = [float(column / 1e22)] + ([1.] * 17 if self.backend == 'atomic' else ([0.] if self.backend == 'ztbabs' else []))
        result = []
        self.x.callModelFunction(self.name, np.asarray(edges).tolist(), params, result)
        return np.asarray(result)

    def composition(self, abundances):
        # vphabs 缓存的是丰度文件的名称，而不是连续自定义文件的内容。
        # vphabs caches the abundance file name, not the contents of successive custom files.
        # 因此每次切换自定义文件前，必须先实际计算一次内置丰度表；
        # An actual built-in-table evaluation is required before each switch;
        # 仅重新赋值 Xset.abund 字符串并不能刷新缓存。
        # merely assigning a new Xset.abund string does not refresh the cache.
        self.x.Xset.abund = 'wilm'
        self._call([.99999, 1.00001], 1e20)
        self.counter += 1
        path = self.directory / f'abundance_{self.counter}.dat'
        path.write_text('\n'.join(f'{v:.17g}' for v in abundances) + '\n')
        self.x.Xset.abund = 'file ' + str(path)

    def transmission(self, energy, halfwidth, column):
        edges = np.column_stack((energy - halfwidth, energy + halfwidth)).reshape(-1)
        return self._call(edges, column)[::2]

    def sigma(self, energy, halfwidth):
        column = np.full(len(energy), 1e20)
        result = np.full(len(energy), np.nan)
        pending = np.ones(len(energy), bool)
        for _ in range(40):
            if not pending.any():
                break
            for value in np.unique(column[pending]):
                select = np.flatnonzero(pending & (column == value))
                transmission = self.transmission(energy[select], halfwidth[select], value)
                with np.errstate(divide='ignore', invalid='ignore'):
                    depth = -np.log(transmission)
                good = np.isfinite(depth) & (depth >= .2) & (depth <= 2.)
                result[select[good]] = depth[good] / value
                pending[select[good]] = False
                low = np.isfinite(depth) & (depth < .2)
                column[select[~good & low]] *= 4
                column[select[~good & ~low]] /= 4
        if pending.any():
            raise RuntimeError('Unable to extract stable optical depth at energies ' + repr(energy[pending].tolist()))
        self.reference_columns = column.copy()
        return result


def _budget_at_width(engine, energy, halfwidth, abundances):
    h = np.zeros(30)
    h[0] = 1
    engine.composition(h)
    baseline = engine.sigma(energy, halfwidth)
    coefficients = np.full((len(energy), 30), np.nan)
    coefficients[:, 0] = baseline
    for el in SUPPORTED[1:]:
        i = ELEMENTS.index(el)
        probe = h.copy()
        probe_amount = abundances[i] if engine.backend in ('tbabs', 'ztbabs') and abundances[i] > 0 else 1.
        probe[i] = probe_amount
        engine.composition(probe)
        contribution = engine.sigma(energy, halfwidth) - baseline
        # 原生 vphabs 返回单精度透射率。明显为负的贡献应被拒绝，
        # Native vphabs returns single-precision transmissions; reject materially negative contributions,
        # 但数值精度导致的、接近零的抵消可以接受并置零。
        # but accept cancellation near zero caused by numerical precision and set it to zero.
        tiny = (contribution < 0) & (np.abs(contribution) <= 1e-6 * baseline)
        contribution[tiny] = 0
        coefficients[:, i] = contribution / probe_amount
    engine.composition(abundances)
    total = engine.sigma(energy, halfwidth)
    supported = [ELEMENTS.index(e) for e in SUPPORTED]
    summed = np.sum(coefficients[:, supported] * abundances[None, supported], axis=1)
    closure = np.abs(summed - total) / np.maximum(total, 1e-300)
    valid = (closure <= 1e-6) & np.all(coefficients[:, supported] >= 0, axis=1)
    return total, coefficients, closure, valid


def _worker(request_path, output_path):
    import xspec
    request = json.loads(Path(request_path).read_text())
    xspec.Xset.chatter = 0
    xspec.Xset.xsect = 'vern'
    root = Path(os.environ['HEADAS'])
    table_path = root / 'spectral/manager/abundances.dat'
    abundances, base = _abundances(request, table_path)
    energy, inverse = np.unique(request['energy'], return_inverse=True)
    engine = _NativeOpacity(xspec, request['backend'], Path(output_path).parent)
    gap = np.full(len(energy), np.inf)
    if len(energy) > 1:
        gap[:-1] = np.minimum(gap[:-1], np.diff(energy) / 4)
        gap[1:] = np.minimum(gap[1:], np.diff(energy) / 4)
    width = np.minimum(energy * 1e-4, gap)
    old_total = old_coeff = None
    converged = np.zeros(len(energy), bool)
    changes = np.full(len(energy), np.inf)
    for iteration in range(7):
        total, coefficients, closure, closed = _budget_at_width(engine, energy, width, abundances)
        if old_total is not None:
            supported = [ELEMENTS.index(e) for e in SUPPORTED]
            a, b = coefficients[:, supported], old_coeff[:, supported]
            # 只有相对于 H 基准小到 1e-10 的变化才视为可忽略；
            # Treat changes below 1e-10 relative to the H baseline as negligible;
            # 这个数值地板不随用户输入的元素丰度改变。
            # this numerical floor is independent of the supplied element abundances.
            denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), total[:, None] * 1e-10)
            changes = np.maximum(np.abs(total - old_total) / np.maximum(total, 1e-300),
                                 np.max(np.abs(a - b) / denom, axis=1))
            converged = changes <= 1e-4
            if converged.all():
                break
        if iteration == 6:
            break
        old_total, old_coeff = total, coefficients
        width /= 10
    reference_columns = engine.reference_columns.copy()
    linearity = np.zeros(len(energy))
    for value in np.unique(reference_columns):
        select = np.flatnonzero(reference_columns == value)
        half = engine.transmission(energy[select], width[select], value / 2)
        linearity[select] = np.abs(-np.log(half) / (value / 2) / total[select] - 1)
    native_transmission = engine.transmission(energy, width, request['nh'])
    # 检查最终窄区间两侧，以发现吸收边区间；即使对称平均值已经收敛，
    # Check both sides of the final narrow interval to detect an absorption-edge bracket;
    # 吸收边仍可能被这个检查识别出来。
    # an edge can still be identified even when the symmetric average appears converged.
    left = engine.sigma(energy - width / 2, width / 5)
    right = engine.sigma(energy + width / 2, width / 5)
    edge_jump = np.abs(right - left) / np.maximum(total, 1e-300)
    edge_bracket = edge_jump > 1e-3
    converged &= ~edge_bracket
    closed &= linearity <= 1e-6
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [table_path, root/'spectral/manager/model.dat', Path(__file__)]}
    metadata = dict(backend=request['backend'], model=engine.name, cross_section_setting='vern',
                    xspec_version=list(xspec.Xset.version), abundance_table=request['abundance_table'],
                    reference_abundances=dict(zip(ELEMENTS, base.tolist())),
                    effective_abundances=dict(zip(ELEMENTS, abundances.tolist())),
                    sha256=hashes, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    interpretation=('ISM effective opacity; H includes model H2, metals retain TBabs grain assumptions' if request['backend'] == 'tbabs' else ('zTBabs effective opacity evaluated at z=0 on rest energies; retains H2 and native zTBabs grain convention' if request['backend'] == 'ztbabs' else 'Neutral-atom photoelectric opacity; excludes grains, molecules and scattering')),
                    abundance_cache_refresh='Built-in wilm model evaluation before each custom abundance file',
                    numerical_method='Adaptive reference column with 0.2<=tau<=2; narrowing symmetric bins, no interpolation',
                    reference_column_cm2_start=1e20, reference_columns_cm2=reference_columns[inverse].tolist(),
                    nh_linearity_relative_error=linearity[inverse].tolist(),
                    native_transmission_at_requested_nh=native_transmission[inverse].tolist(),
                    edge_bracket=edge_bracket[inverse].tolist(),
                    edge_relative_jump=edge_jump[inverse].tolist(), narrowing_iterations=iteration+1,
                    hydrogen_baseline='H=1, other elements zero; TBabs ignores the abundance-file H entry',
                    tbabs_decomposition_caveat='Isolated-element differences need not add to the mixed TBabs opacity. Closure failures are masked; zTBabs is not silently substituted.',
                    model_version_provenance='Runtime XSPEC version plus model registry SHA256; TBabs internal version is emitted in native runtime log')
    result = dict(sigma_total=total[inverse], sigma_element=coefficients[inverse], abundance=abundances,
                  sample_lower=(energy-width)[inverse], sample_upper=(energy+width)[inverse],
                  point_converged=converged[inverse], point_relative_change=changes[inverse],
                  closure_relative_error=closure[inverse], closure_ok=closed[inverse], metadata=metadata)
    Path(output_path).write_text(json.dumps(_json_safe(result), allow_nan=False))


if __name__ == '__main__':
    _worker(sys.argv[1], sys.argv[2])
