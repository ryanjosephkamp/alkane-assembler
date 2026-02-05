#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The Alkane Assembler - Source Package
================================================================================

Project:        Week 4 Project 1: The Alkane Assembler
Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        February 5, 2026
License:        MIT License
================================================================================

This package provides modules for algorithmic enumeration of alkane structural
isomers using graph-theoretic methods.

Modules:
--------
- alkane_generator: Core tree generation and isomer enumeration
- tree_utils: Graph isomorphism checking and tree operations
- visualization: 2D rendering of molecular structures
"""

from .alkane_generator import (
    AlkaneGenerator,
    AlkaneIsomer,
    generate_alkane_isomers,
    count_alkane_isomers,
    get_isomer_statistics
)

from .tree_utils import (
    TreeGraph,
    are_trees_isomorphic,
    canonical_form,
    get_tree_center,
    tree_to_adjacency_list
)

from .visualization import (
    draw_carbon_skeleton,
    draw_lewis_structure,
    plot_isomer_grid,
    plot_isomer_count_curve,
    create_interactive_structure,
    ATOM_COLORS
)

__version__ = "1.0.0"
__author__ = "Ryan Kamp"
__all__ = [
    # Generator
    "AlkaneGenerator",
    "AlkaneIsomer",
    "generate_alkane_isomers",
    "count_alkane_isomers",
    "get_isomer_statistics",
    # Tree utilities
    "TreeGraph",
    "are_trees_isomorphic",
    "canonical_form",
    "get_tree_center",
    "tree_to_adjacency_list",
    # Visualization
    "draw_carbon_skeleton",
    "draw_lewis_structure",
    "plot_isomer_grid",
    "plot_isomer_count_curve",
    "create_interactive_structure",
    "ATOM_COLORS",
]
