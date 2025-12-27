"""
Baseline analysis module for PromptAD.

This module provides comprehensive evaluation and analysis tools for the 
PromptAD baseline (single positive + single negative anchor).
"""

from .margin_analysis import (
    calculate_margin_statistics,
    analyze_split_side_risks,
    compute_overlap_ratio
)

from .anchor_geometry import (
    analyze_anchor_geometry,
    analyze_anchor_decomposition
)

__all__ = [
    'calculate_margin_statistics',
    'analyze_split_side_risks', 
    'compute_overlap_ratio',
    'analyze_anchor_geometry',
    'analyze_anchor_decomposition'
]
