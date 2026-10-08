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

import ast
import operator
import re
from typing import Optional
import numpy as np


def _normalize_expr(expr: str) -> str:
    # Normalize operators only outside quoted string literals.
    pattern = r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|&&|\|\||!(?!=))"
    replacements = {'&&': ' and ', '||': ' or ', '!': ' not '}
    return re.sub(pattern, lambda match: replacements.get(match[0], match[0]), expr).strip()


def _evaluate_expression(node: ast.AST, columns: dict):
    """Evaluate the supported scalar/array expression AST without Python eval.

    Comparisons and boolean operators keep Python precedence and operate
    elementwise on explicitly supplied event columns. Unsupported syntax raises
    ValueError; numeric constants, strings and arithmetic are supported.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in columns:
            raise ValueError(f'Unknown event column: {node.id}')
        return columns[node.id]
    if isinstance(node, ast.BoolOp):
        operation = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        values = [_evaluate_expression(value, columns) for value in node.values]
        result = values[0]
        for value in values[1:]:
            result = operation(result, value)
        return result
    if isinstance(node, ast.UnaryOp):
        operations = {ast.Not: np.logical_not, ast.Invert: operator.invert,
                      ast.USub: operator.neg, ast.UAdd: operator.pos}
        operation = operations.get(type(node.op))
        if operation is not None:
            return operation(_evaluate_expression(node.operand, columns))
    if isinstance(node, ast.BinOp):
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
                      ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
        operation = operations.get(type(node.op))
        if operation is not None:
            return operation(_evaluate_expression(node.left, columns),
                             _evaluate_expression(node.right, columns))
    if isinstance(node, ast.Compare):
        operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Gt: operator.gt,
                      ast.GtE: operator.ge, ast.Lt: operator.lt, ast.LtE: operator.le}
        left = _evaluate_expression(node.left, columns)
        result = True
        for comparison, operand in zip(node.ops, node.comparators):
            operation = operations.get(type(comparison))
            if operation is None:
                raise ValueError('Unsupported comparison operator')
            right = _evaluate_expression(operand, columns)
            result = np.logical_and(result, operation(left, right))
            left = right
        return result
    raise ValueError(f'Unsupported event expression syntax: {type(node).__name__}')


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

    try:
        tree = ast.parse(s, mode='eval')
        result = _evaluate_expression(tree.body, colmap)
        mask = np.broadcast_to(np.asarray(result, dtype=bool), np.asarray(ev.time).shape)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f'Invalid event selection expression: {exc}') from exc
    return np.array(mask, dtype=bool, copy=True)
