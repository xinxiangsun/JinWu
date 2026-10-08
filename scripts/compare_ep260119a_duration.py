#!/usr/bin/env python
"""Recompute EP260119a durations without editing its original analysis/products.

Run from the JinWu checkout in hea. Every comparison uses identical ON/OFF
photons, GTIs and parameters in the pinned old module and the corrected module.
PI 50--400 is a separate remeasurement, not a display-only filter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys

import astropy
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from jinwu.core.io import read_evt
from jinwu.core.data import timescale as TimescaleAnalyzer
from jinwu.core import timescale as corrected
from jinwu.core.plot import plot_event_txx

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / 'reviews/evidence/duration-20261002'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_baseline():
    path = EVIDENCE / 'timescale_baseline.py'
    provenance = json.loads((EVIDENCE / 'baseline_provenance.json').read_text())
    if sha256(path) != provenance['baseline_sha256']:
        raise RuntimeError('Pinned baseline hash differs')
    name = 'jinwu.core._duration_baseline_20261002'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, provenance


def load_rejected_v2():
    """Pin the withdrawn implementation to reproduce its T90/T100 violation."""
    root = EVIDENCE / 'window_consistent_v3'
    path = root / 'timescale_rejected_v2.py'
    provenance = json.loads((root / 'rejected_v2_provenance.json').read_text())
    if sha256(path) != provenance['source_sha256']:
        raise RuntimeError('Rejected v2 snapshot hash differs')
    name = 'jinwu.core._duration_rejected_v2_20261002'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, provenance


def check_window_consistency(result):
    """Verify nominal and each bootstrap interval against their own T100."""
    ordered = np.array([result[k] for k in ('t100_tstart', 't90_tstart',
                       't50_tstart', 't50_tstop', 't90_tstop', 't100_tstop')])
    if np.any(~np.isfinite(ordered)) or np.any(np.diff(ordered) < -1e-7):
        raise AssertionError(f'T100/T90/T50 endpoint inclusion failed: {ordered}')
    if not (0 < result['t50'] <= result['t90'] <= result['t100']+1e-7):
        raise AssertionError('T50 <= T90 <= T100 failed')
    signal = result['cumulative_signal_signed_counts']
    np.testing.assert_allclose(signal[0], 0., rtol=0, atol=1e-12)
    np.testing.assert_allclose(signal[-1], np.sum(result['signal_on_counts']
                               -result['alpha']*result['signal_off_counts']), rtol=0, atol=1e-10)
    np.testing.assert_allclose(result['cumulative_search_range'],
                               [result['t100_tstart'], result['t100_tstop']], rtol=0, atol=1e-7)
    mc = result['bootstrap']
    if mc.get('valid'):
        windows, pairs = np.asarray(mc['window_samples']), np.asarray(mc['endpoint_samples'])
        good = np.isfinite(mc['duration_samples'])
        assert np.all((pairs[:, :, 0] >= windows[:, 0, None]-1e-7)[good])
        assert np.all((pairs[:, :, 1] <= windows[:, 1, None]+1e-7)[good])
        assert np.all((np.asarray(mc['duration_samples']) <= np.diff(windows)+1e-7)[good])
        assert np.all((pairs[:, 0, 0] >= pairs[:, -1, 0]-1e-7) | ~good[:, 0] | ~good[:, -1])
        assert np.all((pairs[:, 0, 1] <= pairs[:, -1, 1]+1e-7) | ~good[:, 0] | ~good[:, -1])
    return dict(nominal_nested=True, bootstrap_within_own_t100=True,
                signal_count_conservation=True, bootstrap_valid=mc['valid'])


def filtered_copy(source: Path, output: Path):
    # Keep the backing heap open while writing variable-length REGION columns.
    # This read-only handle is changed in memory; only output is written.
    with fits.open(source,memmap=False) as hdus:
        pi = hdus['EVENTS'].data['PI']
        hdus['EVENTS'].data = hdus['EVENTS'].data[(pi >= 50) & (pi <= 400)]
        hdus.writeto(output, overwrite=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/home/xinxiang/research/EP260119a'))
    parser.add_argument('--output-dir', type=Path, default=EVIDENCE / 'window_consistent_v3/ep260119a')
    parser.add_argument('--nmc', type=int, default=1000)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    project = args.project_root.resolve()
    if out == project or project in out.parents:
        raise ValueError('Output directory must be outside the original EP260119a project')
    out.mkdir(parents=True, exist_ok=True)
    baseline, provenance = load_baseline()
    rejected_v2, rejected_provenance = load_rejected_v2()
    protected = provenance['protected_files']
    before = {p: sha256(Path(p)) for p in protected}
    root = project / 'ep11900560514wxtCMOS48l23v2/wxtt90'
    src, off = root / 'EP260119a_src.evt', root / 'EP260119a_bkg.evt'
    selected_src, selected_off = out / 'src_pi50_400.evt', out / 'off_pi50_400.evt'
    filtered_copy(src, selected_src)
    filtered_copy(off, selected_off)
    inputs = {'original_extracted_PI': (src, off), 'PI_50_400': (selected_src, selected_off)}
    results = {}
    params = dict(alpha=1/12, p0=.05, block_snr_threshold=3., seed=0, evt_binsize=8.)
    for label, (s, b) in inputs.items():
        modes = ['adaptive', 'fixed']
        results[label] = {}
        for mode in modes:
            print(f'Comparing {label}, {mode}', flush=True)
            kwargs = dict(params, cumulative_mode=mode, nmc=args.nmc)
            old = baseline.txx(s, b, **kwargs)
            new = corrected.txx(s, b, **kwargs)
            rejected = rejected_v2.txx(s, b, **dict(kwargs, nmc=0))
            results[label][mode] = dict(parameters=kwargs, old=old, new=new,
                                        rejected_v2_nominal=rejected,
                                        validation=check_window_consistency(new),
                                        delta_seconds={k: float(new[k]-old[k]) for k in ('t50','t90','t100')})
            fig, axes = plot_event_txx(s, new, background=b, alpha=1/12,
                                     srcname='EP260119a', title=f'{label}; {mode}; corrected duration',
                                     timezero=float(new['bb_edges_time'][0]),
                                     out=out / f'{label}_{mode}_diagnostic.png')
            plt.close(fig)
        if label == 'original_extracted_PI':
            legacy_window = results[label]['adaptive']['old']
            results[label]['legacy_BB_window'] = corrected.txx(s,b,**params,nmc=0,
                burst_tstart=legacy_window['t100_tstart'], burst_tstop=legacy_window['t100_tstop'])
            # Deterministic A05 regression: cumulative bin edges must be bounded.
            results[label]['coarse_bins'] = {}
            for width in (1.,8.,100.,1000.):
                kw = dict(params,evt_binsize=width,cumulative_mode='fixed',nmc=0)
                old = baseline.txx(s,b,**kw)
                new = corrected.txx(s,b,**kw)
                assert new['cumulative_edges_time'][-1] == new['t100_tstop']
                assert new['cumulative_edges_time'][0] == new['t100_tstart']
                check_window_consistency(new)
                results[label]['coarse_bins'][str(width)] = dict(old=old,new=new)
    # Exercise the public analyzer on the same real data and check path/object
    # parity; the heavy bootstrap was already run above.
    analyzer = TimescaleAnalyzer(read_evt(src), background=read_evt(off), alpha=1/12)
    object_result = analyzer.compute(method='aanda',**{k:v for k,v in params.items() if k != 'alpha'},nmc=0)
    path_result = results['original_extracted_PI']['adaptive']['new']
    for key in ('t50','t90','t100','t90_tstart','t90_tstop'):
        np.testing.assert_allclose(object_result[key],path_result[key],rtol=0,atol=1e-7)
    check_window_consistency(object_result)
    check_window_consistency(results['original_extracted_PI']['legacy_BB_window'])
    # Full cumulative/plateau diagnostics and T100-anchored nominal thresholds.
    fig, axs = plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax, (label, mode) in zip(axs.flat, [(l,m) for l in inputs for m in ('adaptive','fixed')]):
        r = results[label][mode]['new']; old = results[label][mode]['old']
        t0 = float(r['bb_edges_time'][0])
        ax.plot(r['cumulative_curve_time']-t0,r['cumulative_signed_counts'],label='Signed net cumulative')
        ax.axvspan(r['t100_tstart']-t0,r['t100_tstop']-t0,color='grey',alpha=.15,label='T100')
        for plateau,level in zip(r['plateau_intervals'],r['plateau_levels']):
            ax.hlines(level,plateau[0]-t0,plateau[1]-t0,color='green',linewidth=2)
        lz,lt = r['cumulative_levels']
        for q in (.05,.95):
            ax.axhline((1-q)*lz+q*lt,color='orange',linewidth=.8)
        ax.axvspan(old['t90_tstart']-t0,old['t90_tstop']-t0,color='blue',alpha=.15,label='Old T90')
        ax.axvline(r['t90_tstart']-t0,color='red',linestyle='--',label='New T90 endpoints')
        ax.axvline(r['t90_tstop']-t0,color='red',linestyle='--')
        rejected = results[label][mode]['rejected_v2_nominal']
        for j, key in enumerate(('t90_tstart', 't90_tstop')):
            ax.axvline(rejected[key]-t0,color='purple',linestyle=':',
                       label='Withdrawn v2 endpoints' if j == 0 else '_nolegend_')
        ax.set(title=f'{label}, {mode}',xlabel='Seconds since GTI start (observer frame)',ylabel='Signed net counts')
        ax.legend(fontsize=8)
    fig.savefig(out/'cumulative_comparison.png',dpi=170)
    fig.savefig(out/'cumulative_comparison.pdf')
    plt.close(fig)
    changed = [p for p in protected if sha256(Path(p)) != before[p]]
    if changed:
        raise RuntimeError(f'Original files changed during analysis: {changed}')
    payload = dict(observation='EP260119a / WXT 11900560514 / CMOS48',
                   time_unit='s',frame='observer',baseline_provenance=provenance,
                   rejected_v2_provenance=rejected_provenance,
                   environment=dict(python=platform.python_version(),numpy=np.__version__,astropy=astropy.__version__,
                                    python_executable=sys.executable,imported_timescale=inspect.getfile(corrected)),
                   current_timescale_sha256=sha256(Path(inspect.getfile(corrected))),
                   inputs={label:{'source':str(s),'off':str(b),'source_sha256':sha256(s),'off_sha256':sha256(b),
                                  'source_events':read_evt(s).n,'off_events':read_evt(b).n} for label,(s,b) in inputs.items()},
                   original_files_unchanged=True,unchanged_file_count=len(before),results=results)
    (out/'comparison.json').write_text(json.dumps(json_value(payload),indent=2,allow_nan=False)+'\n')
    rows = ['# EP260119a 时标重算与对照','',
            '单位：观测者系秒。旧算法来自保存的源码快照；新旧每行使用相同事件、GTI、alpha 和设置。',
            '原提取事件的 PI 范围比 PI 50–400 宽；这两种输入是分别测量。', '',
            '| 输入 | 累计 | 旧 T50 | 新 T50 | 旧 T90 | 撤回 v2 T90 | 新 T90 | ΔT90 对旧版 | T100 | 新 T90 误差状态 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for label in inputs:
        for mode in ('adaptive','fixed'):
            r=results[label][mode];a,b=r['old'],r['new']
            rejected = r['rejected_v2_nominal']
            rows.append(f"| {label} | {mode} | {a['t50']:.6f} | {b['t50']:.6f} | {a['t90']:.6f} | {rejected['t90']:.6f} | {b['t90']:.6f} | {b['t90']-a['t90']:+.6f} | {b['t100']:.6f} | {b['t90_error_status']} |")
    rows += ['', '## 解释与验收边界','',
             '- 撤回 v2 的点估计及模拟结果：它混用了 BB T100 与全观测平台总计数和交点，违反同一个爆发区间的定义；原报告将 T100 改称活动窗不能解决该错误。',
             '- T100 是首尾显著 BB 的边界区间；T50/T90 的总净计数、累计起点及名义交点均来自同一 T100。全部名义值和每次有效模拟均通过 T50 区间包含于 T90、T90 包含于各自 T100 的验证。',
             '- 平台均值只作诊断。误差参考 Koshut，明确适配为 T100 边界累计水平、窗内 ON+alpha² OFF 计数方差和窗外平台散布；不是原论文的平台均值名义估计量。',
             '- 默认平台必须检查是否无源辐射；sigma 阈值没有交点时返回 null/NaN。误差交点搜索可用完整观测，其范围与名义交点搜索分别保存。',
             '- nmc 现在对完整观测模拟并重新拟合背景、定窗和计算同一个累计估计量；原始分位、失败率和边界触达保存在 JSON，属于拟合模型下的敏感性诊断。',
             '- 分箱比较没有加入统计误差；没有声称本例误差为校准后的 68% 区间。',
             f'- 本次执行前后 {len(before)} 个原项目文件的 SHA256 相同。独立产物均位于本目录。','',
             '[累计对比图](cumulative_comparison.png) · [完整数值与模拟](comparison.json)', '',
             '方法出处：[De Luca et al. 2021 §5.4](https://doi.org/10.1051/0004-6361/202039783)、'
             '[Koshut et al. 1996 §2.2–2.3](https://ntrs.nasa.gov/api/citations/19970025588/downloads/19970025588.pdf)。']
    (out/'comparison.md').write_text('\n'.join(rows)+'\n')
    print('\n'.join(rows[:14]),flush=True)
    print(f'Output: {out}; original files unchanged: {len(before)}')


if __name__ == '__main__':
    main()
