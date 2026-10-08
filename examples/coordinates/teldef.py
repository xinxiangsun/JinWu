"""迁移示例 / Migrated example: coordinates/teldef.
状态 / Status: needs_review. 需要 test/data 中的 teldef 校准文件 / Requires the teldef calibration fixture
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
    from jinwu.ftools import teldef
    import numpy as np
    p = Path(str(TEST_DATA_ROOT / 'data/swx20230701v001.teldef')).resolve()
    print('teldef path:', p)
    T = teldef.Teldef.from_file(str(p))
    print('has align:', hasattr(T,'align'))
    if hasattr(T,'align'):
        print('ALIGNM:')
        print(T.align)
        print('focal_length:', getattr(T,'focal_length',None))
        print('det_xscl, det_yscl:', getattr(T,'det_xscl',None), getattr(T,'det_yscl',None))
        print('optaxis:', getattr(T,'optaxis',None))
    # example pointing (RA_PNT, DEC_PNT) -- replace with real pointing when available
    ra_pnt = 180.0
    dec_pnt = 45.0
    # sample sky coordinate near boresight
    ra = 180.001
    dec = 45.0
    if hasattr(T,'align'):
        xpix, ypix = T.sky_to_det_with_pointing(ra, dec, ra_pnt, dec_pnt)
        print('sky->det (with pointing) =>', xpix, ypix)
        ra2, dec2 = T.det_to_sky_with_pointing(xpix, ypix, ra_pnt, dec_pnt)
        print('det->sky =>', ra2, dec2)
    else:
        print('teldef has no ALIGNM, use simple sky_to_det/det_to_sky')
        if T.crval is not None and T.crpix is not None:
            print(T.sky_to_det(ra,dec))
            print(T.det_to_sky(*T.sky_to_det(ra,dec)))
