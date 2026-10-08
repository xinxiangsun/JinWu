"""Generate canonical module reference pages without executing scientific code."""
from __future__ import annotations
import ast
import json
import re
import unicodedata
from pathlib import Path

CATEGORIES = [("core", "数据、时间与分析核心"), ("ftools", "FITS 与 FTOOLS"),
              ("model", "模型与背景"), ("physics", "吸收与物理工具"),
              ("lf", "模拟与红移"), ("host", "星表与聚类"), ("ep", "Einstein Probe"),
              ("swift", "Swift"), ("fermi", "Fermi / GBM"), ("gw", "GW 定位与覆盖")]

def normalize_docstrings(app, what, name, obj, options, lines):
    """Normalize source markup for autodoc without modifying scientific source.

    Chinese NumPy section headings become their Napoleon equivalents. Inline
    code and unpaired markup characters are rendered literally; content is
    retained. Sphinx supplies and consumes the mutable lines list.
    """
    headings = {'参数': 'Parameters', '返回': 'Returns', '属性': 'Attributes',
                '字段': 'Attributes', '注记': 'Notes', '注意': 'Notes',
                '示例': 'Examples', '用法': 'Examples', '使用示例': 'Examples'}
    literal_indent = None
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        if literal_indent is not None:
            if not line.strip() or indent > literal_indent:
                continue
            literal_indent = None
        if line.rstrip().endswith('::') or re.match(r'\s*\.\. (?:code|code-block)::', line):
            literal_indent = indent
            continue
        if line.lstrip().startswith(('>>>', '...')):
            continue
        if i + 1 < len(lines) and re.fullmatch(r'[-=~]{3,}', lines[i + 1]):
            lines[i] = headings.get(line.strip(), line)
            width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in lines[i])
            lines[i + 1] = lines[i + 1][0] * max(width, 12)
        # Markdown inline code occurs in some historical docstrings.
        line = re.sub(r'(?<![:`])`([^`\n]+)`(?!`)', r'``\1``', lines[i])
        line = line.replace('(n_chan,)', '(``n_chan``,)')
        # RST requires boundaries around literals, including CJK punctuation.
        line = re.sub(r'``[^`]+``', lambda m:
                      (' ' if m.start() and not line[m.start() - 1].isspace() else '')
                      + m.group()
                      + (' ' if m.end() < len(line) and not line[m.end()].isspace() else ''), line)
        parts = re.split(r'(``[^`]+``|\*\*[^*]+\*\*|:\w+:`[^`]+`)', line)
        for j in range(0, len(parts), 2):
            parts[j] = re.sub(r'(?<!\\)\*', r'\\*', parts[j])
            parts[j] = re.sub(r'(?<!\\)\|', r'\\|', parts[j])
            parts[j] = re.sub(r'(?<=\w)_(?!\w)', r'\\_', parts[j])
        lines[i] = re.sub(r'^(\s*)\\\* ', r'\1* ', ''.join(parts))
    # This historical return sentence sits inside an unterminated NumPy
    # Parameters section; make it a note instead of a fictitious parameter.
    if name == 'jinwu.core.timescale.iterative_bayesian_blocks':
        for i, line in enumerate(lines):
            if line.startswith('返回 :class:'):
                lines[i:i] = ['', 'Notes', '-----', '']
                break

def separate_docstring_blocks(app, what, name, obj, options, lines):
    """Add required RST block boundaries after Napoleon's field conversion."""
    result = []
    bullet_indent = None
    bullet_start = None
    literal_indent = None
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if literal_indent is not None:
            if not line.strip() or indent > literal_indent:
                result.append(line)
                continue
            literal_indent = None
        if line.rstrip().endswith('::') or re.match(r'\s*\.\. (?:code|code-block)::', line):
            literal_indent = indent
            bullet_indent = None
            result.append(line)
            continue
        line = re.sub(r'^(\s*(?:[-*+]|\d+[.)]))\s+', r'\1 ', line)
        bullet = re.match(r'^(\s*)(?:[-*+] |\d+[.)] )', line)
        if bullet:
            bullet_indent = bullet.end()
            bullet_start = len(bullet.group(1))
            if result and result[-1].strip() and (len(bullet.group(1)) > len(result[-1]) - len(result[-1].lstrip()) or not re.match(r'\s*(?:[-*+] |\d+[.)] )', result[-1])):
                result.append('')
        elif line.strip() and bullet_indent is not None:
            indent = len(line) - len(line.lstrip())
            if indent > bullet_start:
                line = ' ' * bullet_indent + line.lstrip()
            else:
                bullet_indent = None
        else:
            bullet_indent = None
        if result and line.strip() and result[-1].strip():
            prev = result[-1]
            indent = len(line) - len(line.lstrip())
            prev_indent = len(prev) - len(prev.lstrip())
            # Retain field, directive and list continuations; otherwise an
            # indented example/paragraph needs an explicit block boundary.
            if indent > prev_indent and not (prev.rstrip().endswith('::') or re.match(r'\s*(?:[-*+] |\d+[.)] |:\w+|\.\. )', prev)):
                result.append('')
            elif indent < prev_indent:
                result.append('')
            elif re.match(r'\s*[-+] ', prev) and not re.match(r'\s*[-+] ', line) and indent == prev_indent:
                result.append('')
        result.append(line)
    lines[:] = result

def generate_api_pages(app):
    """Generate canonical module RST and an exclusion inventory for Sphinx.

    Input: a Sphinx app with srcdir pointing to this checkout's docs.
    Output: deterministic files under docs/api, which is gitignored.
    AST reading does not execute mission workflows or import optional runtimes.
    """
    docs = Path(app.srcdir)
    root = docs.parent
    target = docs / "api"
    target.mkdir(exist_ok=True)
    modules, excluded = [], []
    for path in sorted((root / "packages").glob("*/src/jinwu/**/*.py")):
        namespace = path.parts[path.parts.index("src") + 1:]
        module = ".".join(namespace)[:-3]
        reason = None
        if "_vendor" in namespace:
            reason = "第三方冻结实现；公开包装见 MVT / targeted search"
        elif path.name in {"__init__.py", "__main__.py"}:
            reason = "包导出或 CLI；对象在定义模块记录，CLI 见流程指南"
        elif module.startswith(("jinwu.lightcurve.", "jinwu.spectrum.")) or module in {
            "jinwu.core.lf", "jinwu.core.redshift", "jinwu.fermi.gbm.GBMObservation", "jinwu.swift.bat.BATObservation"
        }:
            reason = "兼容转发；见版本兼容说明和规范模块"
        elif module == "jinwu.physics.radiation":
            reason = "仅导入 naima 第三方模型，没有 JinWu 自定义接口"
        tree = ast.parse(path.read_text())
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
        if reason or not names:
            excluded.append({"module": module, "source": str(path.relative_to(root)), "reason": reason or "没有本模块定义的公共函数或类"})
            continue
        if module.startswith(("jinwu.core.host", "jinwu.cluster")):
            category = "host"
        elif module.startswith(("jinwu.model", "jinwu.background")):
            category = "model"
        else:
            category = module.split(".")[1]
        content = f"{module}\n{'=' * len(module)}\n\n"
        content += "参数、返回值和单位来自当前源码。构建不执行科学计算；适用条件见使用指南。\n\n"
        content += f".. automodule:: {module}\n   :members: {', '.join(names)}\n   :ignore-module-all:\n   :undoc-members:\n   :show-inheritance:\n"
        (target / f"module-{module}.rst").write_text(content)
        modules.append({"module": module, "source": str(path.relative_to(root)), "category": category, "public_definitions": names})
    content = "按模块浏览 API\n============================\n\n公共对象在定义模块中记录；顶层包导出是便捷导入入口。\n\n"
    for key, title in CATEGORIES:
        rows = [m for m in modules if m["category"] == key]
        if rows:
            content += f"{title}\n{'-' * len(title) * 2}\n\n.. toctree::\n   :maxdepth: 1\n\n"
            content += "".join(f"   module-{m['module']}\n" for m in rows) + "\n"
    (target / "modules.rst").write_text(content)
    (target / "inventory.json").write_text(json.dumps({"modules": modules, "excluded": excluded}, ensure_ascii=False, indent=2))

def setup(app):
    app.connect("builder-inited", generate_api_pages)
    app.connect("autodoc-process-docstring", normalize_docstrings, priority=400)
    app.connect("autodoc-process-docstring", separate_docstring_blocks, priority=800)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
