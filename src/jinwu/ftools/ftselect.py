"""轻量级 ftselect 风格表达式解析器（受限但实用）。

功能：将类似于 ftselect 的表达式（例如 ``PHA>100 && PHA<1000``）转换为
对 EventData 的布尔掩码（numpy array）。

实现说明：
- 支持比较运算：== != > >= < <=
- 支持逻辑运算符：&& (and), || (or), ! (not) 以及括号
- 支持字符串常量（单引号或双引号）和数值常量
- 对列名（标识符）使用 EventData 内的列（如 PHA, PI, ENERGY, TIME, X, Y）

此实现并非完整的 ftselect 语法解析器，但覆盖常见用例；如果需要 100% 兼容
应考虑直接调用系统的 ftselect 或完整移植其解析器。
"""
from __future__ import annotations

import re
from typing import Optional
import numpy as np


def _normalize_expr(expr: str) -> str:
    # map ftselect style operators to Python/numpy equivalents
    s = expr
    s = s.replace('&&', ' and ')
    s = s.replace('||', ' or ')
    # treat single ! as not (but avoid !=)
    s = re.sub(r'(?<![=!])!(?!=)', ' not ', s)
    return s


_tok_re = re.compile(r"('[^']*'|\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_]*|[0-9]+\.?[0-9]*|==|!=|>=|<=|[<>]|\(|\)|and|or|not|\&|\||\^|\+|\-|\*|/)")


def expression_to_mask(ev, expr: str) -> np.ndarray:
    """把表达式转换为 EventData 上的布尔掩码。

    参数:
      ev: EventData
      expr: 表达式字符串（ftselect 风格）

    返回:
      numpy 布尔数组，长度等于事件数。
    """
    if expr is None or expr.strip() == '':
        return np.ones(len(ev.time), dtype=bool)

    s = _normalize_expr(expr)

    # tokenization
    toks = _tok_re.findall(s)
    if not toks:
        raise ValueError('Cannot parse expression')

    # build mapping from identifier to numpy array expression
    # safe names: columns in ev
    colmap = {}
    # try common names
    for name in ('PHA', 'PI', 'CHANNEL', 'TIME', 'X', 'Y'):
        lower = name.lower()
        val = None
        if hasattr(ev, lower):
            val = getattr(ev, lower)
        else:
            # try to access via column arrays if attributes absent
            try:
                from ..core.xselect import _read_column_from_evt
                v = _read_column_from_evt(ev.path, name)
                if v is not None:
                    val = v
            except Exception:
                val = None
        if val is not None:
            colmap[name] = np.asarray(val)
            colmap[lower] = np.asarray(val)

    # function to render token stream into a python/numpy expression
    out_tokens = []
    for tok in toks:
        tt = tok.strip()
        if tt == 'and' or tt == 'or' or tt == 'not' or tt in ('(', ')'):
            # keep
            if tt == 'and':
                out_tokens.append('&')
            elif tt == 'or':
                out_tokens.append('|')
            elif tt == 'not':
                out_tokens.append('~')
            else:
                out_tokens.append(tt)
        elif re.match(r"^'[^']*'$|^\"[^\"]*\"$", tt):
            # string literal
            out_tokens.append(tt)
        elif re.match(r'^[0-9]+\.?[0-9]*$', tt):
            out_tokens.append(tt)
        elif tt in ('==', '!=', '>=', '<=', '>', '<'):
            # map to numpy-friendly ops
            if tt == '==':
                out_tokens.append('==')
            elif tt == '!=':
                out_tokens.append('!=')
            else:
                out_tokens.append(tt)
        else:
            # identifier: column name or unknown
            key = tt
            if key in colmap:
                # we will substitute a temporary name like __col_PHA
                name = f'__col_{key}'
                out_tokens.append(name)
            else:
                # unknown identifier: keep as-is (might be function or constant)
                out_tokens.append(tt)

    pyexpr = ' '.join(out_tokens)

    # build eval environment
    env = {}
    # put numpy into env for numeric ops
    env['np'] = np
    # place column arrays
    for k, v in colmap.items():
        env[f'__col_{k}'] = np.asarray(v)

    # Evaluate expression safely
    try:
        # result should be a numpy boolean array
        res = eval(pyexpr, {'__builtins__': {}}, env)
    except Exception as e:
        raise RuntimeError(f'Failed to evaluate expression: {e}\npython expr: {pyexpr}')

    # ensure boolean mask
    res = np.asarray(res)
    if res.dtype != bool:
        # try nonzero
        try:
            res = res != 0
        except Exception:
            raise RuntimeError('Expression did not produce boolean mask')
    return res
