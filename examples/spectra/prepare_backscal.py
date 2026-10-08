"""迁移示例 / Migrated example: spectra/prepare_backscal.
状态 / Status: legacy_external_dependency. 依赖旧 autohea API；会写入指定谱目录，未重跑 / Uses legacy autohea and writes spectra; not rerun
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

    import pandas as pd
    df = pd.read_csv(str(RESEARCH_ROOT / 'results_all.csv'))


    # 原 Notebook 单元 / Notebook cell 3

    name = df['Unnamed: 0']

    # 原 Notebook 单元 / Notebook cell 4

    import os

    # 原 Notebook 单元 / Notebook cell 5

    from autohea.spectrum.specfake import prepare_background_for_fakeit

    # 原 Notebook 单元 / Notebook cell 6

    # 批量遍历 DataFrame，生成用于绘图的三个数组：L_arr, zmax7_arr, mvt_arr
    from glob import glob
    from tqdm import tqdm


    for i in tqdm(range(len(name)), desc='processing'):
        nm = str(df['Unnamed: 0'][i])


        spec_dir = f"/home/xinxiang/research/for_LF_study/spec_new/{nm}/all/"
        if not os.path.isdir(spec_dir):
            print(f"skip {nm}: spec dir not found: {spec_dir}")
            continue
        os.chdir(spec_dir)
        bkgfile = glob('back_all.pha')
        rmffile = glob('*.rmf')
        arffile = glob('*.arf')
        srcfile = glob('src_all.pha')

        prepare_background_for_fakeit(src_pha=srcfile[0], bkg_pha=bkgfile[0], out_path=spec_dir)


    # 原 Notebook 单元 / Notebook cell 7

    from astropy.io import fits
    with fits.open(str(RESEARCH_ROOT / 'for_LF_study/spec_new/EP240315a/all/back_all_for_fakeit.pha')) as hdu :
        spectrum = hdu[1]
        header = spectrum.header
        print(header['BACKSCAL'])
