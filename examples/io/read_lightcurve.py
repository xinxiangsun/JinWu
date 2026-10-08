"""迁移示例 / Migrated example: io/read_lightcurve.
状态 / Status: data_required. 需要 EP250111a 光变文件 / Requires an EP250111a light-curve file
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

    from jinwu.core.io import readfits
    from pathlib import Path

    # 原 Notebook 单元 / Notebook cell 3

    lcpath = Path(str(RESEARCH_ROOT / 'EP250111a/ep/ep11916646654wxt12s1.lc'))

    # 原 Notebook 单元 / Notebook cell 4

    lc = readfits(lcpath)
