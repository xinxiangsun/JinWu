"""迁移示例 / Migrated example: events/xselect.
状态 / Status: data_required. 需要事件、源和背景文件以及 XSELECT / Requires event/source/background files and XSELECT
历史探索不保证可完整运行 / Historical exploration is not guaranteed runnable.
见同名 Notebook 与目录 README / See the paired Notebook and README.
"""


# 保持单元共享全局变量语义；仅直接运行脚本时执行。
# Preserve shared Notebook globals; execute only when run as a script.
if __name__ == '__main__':
    # 原 Notebook 单元 / Notebook cell 1

    # 从仓库定位输入，允许通过环境变量覆盖；不移动原始数据。
    # Locate repository inputs with environment overrides; original data stay in place.
    from pathlib import Path
    from datetime import datetime, timezone
    import os
    _anchor = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    REPO_ROOT = next((p for p in (_anchor, *_anchor.parents)
                      if (p/'packages/jinwu').is_dir()), None)
    if REPO_ROOT is None:
        raise RuntimeError('请从 JinWu 仓库内运行 / Run from within the JinWu checkout')
    RESEARCH_ROOT = Path(os.environ.get('JINWU_EXAMPLE_RESEARCH_ROOT', REPO_ROOT.parent))
    TEST_DATA_ROOT = Path(os.environ.get('JINWU_EXAMPLE_TEST_ROOT', REPO_ROOT/'test'))
    DOWNLOAD_ROOT = Path(os.environ.get('JINWU_EXAMPLE_DOWNLOAD_ROOT', Path.home()/'下载'))
    # 图件写入独立运行目录，不覆盖研究目录中的历史图。
    # Write figures into a unique run directory, preserving historical research plots.
    OUTPUT_DIR = Path(os.environ.get('JINWU_EXAMPLE_OUTPUT_ROOT', REPO_ROOT/'examples/_outputs')) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)


    # 原 Notebook 单元 / Notebook cell 2

    from pathlib import Path
    evtpath = Path(str(RESEARCH_ROOT / 'EP250315a/ep13600005120wxtCMOS2l23v3_20251016_225305/ep13600005120wxt2po_cl.evt'))

    # 原 Notebook 单元 / Notebook cell 3

    from jinwu.core import readfits
    evt = readfits(str(RESEARCH_ROOT / 'EP250315a/ep13600005120wxtCMOS2l23v3_20251016_225305/ep13600005120wxt2po_cl.evt'))

    # 原 Notebook 单元 / Notebook cell 4

    lc= evt.extract_curve(binsize=1)

    # 原 Notebook 单元 / Notebook cell 5

    lc.time

    # 原 Notebook 单元 / Notebook cell 6

    lc.plot()

    # 原 Notebook 单元 / Notebook cell 7

    lc.value

    # 原 Notebook 单元 / Notebook cell 8

    lc.timezero

    # 原 Notebook 单元 / Notebook cell 9

    lc.telescop

    # 原 Notebook 单元 / Notebook cell 10

    lc.timezero_obj

    # 原 Notebook 单元 / Notebook cell 11

    lc.plot()

    # 原 Notebook 单元 / Notebook cell 12

    evt.clear_all()

    # 原 Notebook 单元 / Notebook cell 13

    lc = evt.extract_curve(binsize=1)

    # 原 Notebook 单元 / Notebook cell 14

    lc.time

    # 原 Notebook 单元 / Notebook cell 15

    lc.plot()

    # 原 Notebook 单元 / Notebook cell 16

    from pathlib import Path
    out_lc = Path(str(OUTPUT_DIR / 'xselect_lc.fits'))
    evt.filter_time(tmin=0, tmax=1000)  # 再次加一个时间过滤
    lc = evt.extract_curve(binsize=1.0)
    saved_path = evt.save(out_lc, kind="lc", binsize=1.0, overwrite=True)
    saved_path

    # 原 Notebook 单元 / Notebook cell 17

    bkglc = readfits(str(RESEARCH_ROOT / 'EP250315a/ep13600005120wxtCMOS2l23v3_20251016_225305/ep13600005120wxt2s1bk.lc'))
    srclc = readfits(str(RESEARCH_ROOT / 'EP250315a/ep13600005120wxtCMOS2l23v3_20251016_225305/ep13600005120wxt2s1.lc'))

    # 原 Notebook 单元 / Notebook cell 18

    srclc.value

    # 原 Notebook 单元 / Notebook cell 19

    bkglc.value

    # 原 Notebook 单元 / Notebook cell 20

    lc = srclc - bkglc

    # 原 Notebook 单元 / Notebook cell 21

    lc.timezero

    # 原 Notebook 单元 / Notebook cell 22

    lc.timezero_obj.utc.isot

    # 原 Notebook 单元 / Notebook cell 23

    lc.slice(5000, 10000).rebin(binsize=20).plot(srcname = 'EP250315a')

    # 原 Notebook 单元 / Notebook cell 24

    lcnew = lc.slice(5000, 10000)

    # 原 Notebook 单元 / Notebook cell 25

    lcnew.timezero_obj

    # 原 Notebook 单元 / Notebook cell 26

    lc.slice(5000,10000).rebin(binsize=20).plot(srcname = 'EP250315a')

    # 原 Notebook 单元 / Notebook cell 27

    lc= readfits(str(RESEARCH_ROOT / 'EP250315a/ep13600005120wxtCMOS2l23v3_20251016_225305/ep13600005120wxt2s1.lc'))


    # 原 Notebook 单元 / Notebook cell 28

    lc.timezero

    # 原 Notebook 单元 / Notebook cell 29

    lc.tstart

    # 原 Notebook 单元 / Notebook cell 30

    lc.dt

    # 原 Notebook 单元 / Notebook cell 31

    from jinwu.core.time import TimeDelta

    # 原 Notebook 单元 / Notebook cell 32

    TimeDelta(1, format='sec')
