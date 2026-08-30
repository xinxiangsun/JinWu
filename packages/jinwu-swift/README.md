# jinwu-swift

Swift/BAT instrument support for the
[jinwu](https://pypi.org/project/jinwu/) analysis toolkit: BAT observation
handling, attitude reconstruction and survey/posted data products.

```bash
pip install jinwu-swift
```

Import as `jinwu.swift`, e.g. `from jinwu.swift.bat import BATObservation`.

## Optional extras

```bash
pip install "jinwu-swift[gdt]"   # gdt-swift: SAO/poshist 指向文件读取与天图
```

Without the `gdt` extra the core BAT workflows work; SAO/poshist-based
features degrade gracefully (`jinwu.swift.bat.bat_observation.HAS_GDT_SWIFT`
is `False`).
