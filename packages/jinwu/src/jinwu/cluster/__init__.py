"""Cluster analysis and spatial grouping utilities.

This module provides:
- ClusterAnalyzer: Spatial clustering and grouping algorithms

Example:
    >>> from jinwu.cluster import ClusterAnalyzer
"""

from __future__ import annotations

from .cluster import ClusterAnalyzer

# 历史名称：早期文档中写作 ``Cluster``，实际实现早已改名 ``ClusterAnalyzer``。
Cluster = ClusterAnalyzer

__all__ = [
    'ClusterAnalyzer',
    'Cluster',
]
