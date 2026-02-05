#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Visualization Module for Alkane Isomers
================================================================================

Project:        Week 4 Project 1: The Alkane Assembler
Module:         visualization.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        February 5, 2026
Last Updated:   February 5, 2026

License:        MIT License
================================================================================

This module provides visualization functions for alkane molecular structures,
including:
- 2D carbon skeleton drawings
- Lewis structure representations with hydrogens
- Grid layouts for multiple isomers
- Isomer count explosion plots

The visualizations are designed to help understand molecular structure and
demonstrate the combinatorial explosion of chemical space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from typing import Dict, List, Tuple, Optional, Any
from collections import deque


# ============================================================================
# Color Schemes
# ============================================================================

ATOM_COLORS = {
    'C': '#404040',       # Dark gray for carbon
    'H': '#FFFFFF',       # White for hydrogen
    'O': '#FF0000',       # Red for oxygen
    'N': '#0000FF',       # Blue for nitrogen
    'S': '#FFFF00',       # Yellow for sulfur
    'P': '#FFA500',       # Orange for phosphorus
}

BOND_COLORS = {
    'single': '#333333',
    'double': '#333333',
    'triple': '#333333',
}

# Color schemes for different visualization styles
STYLE_COLORS = {
    'classic': {
        'carbon': '#404040',
        'hydrogen': '#808080',
        'bond': '#333333',
        'background': '#FFFFFF',
    },
    'colorful': {
        'carbon': '#2C3E50',
        'hydrogen': '#3498DB',
        'bond': '#34495E',
        'background': '#ECF0F1',
    },
    'neon': {
        'carbon': '#00FF00',
        'hydrogen': '#FF00FF',
        'bond': '#00FFFF',
        'background': '#1A1A2E',
    },
    'organic': {
        'carbon': '#2ECC71',
        'hydrogen': '#95A5A6',
        'bond': '#27AE60',
        'background': '#FDFEFE',
    }
}


# ============================================================================
# Layout Algorithms
# ============================================================================

def compute_tree_layout(adjacency: Dict[int, List[int]], 
                        root: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """
    Compute 2D positions for tree nodes using hierarchical layout.
    
    Args:
        adjacency: Adjacency list representation
        root: Optional root node (auto-detected if not provided)
        
    Returns:
        Dict mapping node -> (x, y) position
    """
    n = len(adjacency)
    
    if n == 0:
        return {}
    if n == 1:
        return {0: (0.0, 0.0)}
    
    # Find root (center of tree) if not provided
    if root is None:
        root = _find_tree_center(adjacency)[0]
    
    positions = {}
    
    # Use Reingold-Tilford-like algorithm for nice tree layout
    def compute_subtree_width(node: int, parent: int) -> float:
        """Compute width needed for subtree."""
        children = [n for n in adjacency[node] if n != parent]
        
        if not children:
            return 1.0
        
        return sum(compute_subtree_width(c, node) for c in children)
    
    def layout_subtree(node: int, parent: int, x: float, y: float, width: float):
        """Recursively position nodes."""
        positions[node] = (x, y)
        
        children = [n for n in adjacency[node] if n != parent]
        
        if not children:
            return
        
        # Compute width for each child
        child_widths = [compute_subtree_width(c, node) for c in children]
        total_width = sum(child_widths)
        
        # Position children
        current_x = x - width / 2
        for child, child_width in zip(children, child_widths):
            child_x = current_x + child_width / 2
            layout_subtree(child, node, child_x, y - 1.5, child_width)
            current_x += child_width
    
    # Start layout from root
    total_width = compute_subtree_width(root, -1)
    layout_subtree(root, -1, 0, 0, total_width)
    
    return positions


def compute_radial_layout(adjacency: Dict[int, List[int]],
                          root: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
    """
    Compute radial tree layout.
    
    Nodes are arranged in concentric circles with root at center.
    """
    n = len(adjacency)
    
    if n == 0:
        return {}
    if n == 1:
        return {0: (0.0, 0.0)}
    
    if root is None:
        root = _find_tree_center(adjacency)[0]
    
    positions = {}
    
    def layout_radial(node: int, parent: int, angle_start: float, 
                      angle_span: float, depth: int):
        radius = depth * 1.2
        angle = angle_start + angle_span / 2
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[node] = (x, y)
        
        children = [n for n in adjacency[node] if n != parent]
        
        if not children:
            return
        
        # Divide angle span among children
        child_span = angle_span / len(children)
        current_angle = angle_start
        
        for child in children:
            layout_radial(child, node, current_angle, child_span, depth + 1)
            current_angle += child_span
    
    positions[root] = (0.0, 0.0)
    
    children = adjacency[root]
    if children:
        angle_span = 2 * np.pi / len(children)
        for i, child in enumerate(children):
            layout_radial(child, root, i * angle_span, angle_span, 1)
    
    return positions


def compute_spring_layout(adjacency: Dict[int, List[int]], 
                          iterations: int = 50,
                          k: float = 1.0) -> Dict[int, Tuple[float, float]]:
    """
    Compute force-directed spring layout.
    
    Nodes repel each other, edges act as springs.
    """
    n = len(adjacency)
    
    if n == 0:
        return {}
    if n == 1:
        return {0: (0.0, 0.0)}
    
    # Initialize random positions
    np.random.seed(42)
    pos = {i: (np.random.uniform(-1, 1), np.random.uniform(-1, 1)) 
           for i in range(n)}
    
    # Get edge list
    edges = set()
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            edges.add((min(node, neighbor), max(node, neighbor)))
    
    # Force-directed iterations
    for _ in range(iterations):
        forces = {i: [0.0, 0.0] for i in range(n)}
        
        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx**2 + dy**2) + 0.01
                
                # Repulsion proportional to 1/dist^2
                force = k * k / dist
                
                fx = force * dx / dist
                fy = force * dy / dist
                
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy
        
        # Attractive forces along edges
        for u, v in edges:
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            dist = np.sqrt(dx**2 + dy**2) + 0.01
            
            # Attraction proportional to dist
            force = dist / k
            
            fx = force * dx / dist
            fy = force * dy / dist
            
            forces[u][0] -= fx
            forces[u][1] -= fy
            forces[v][0] += fx
            forces[v][1] += fy
        
        # Update positions with damping
        damping = 0.85
        for i in range(n):
            pos[i] = (
                pos[i][0] + damping * forces[i][0] * 0.1,
                pos[i][1] + damping * forces[i][1] * 0.1
            )
    
    return pos


def _find_tree_center(adjacency: Dict[int, List[int]]) -> List[int]:
    """Find center(s) of tree by iterative leaf removal."""
    n = len(adjacency)
    if n <= 2:
        return list(range(n))
    
    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    remaining = set(adjacency.keys())
    
    while len(remaining) > 2:
        leaves = [node for node in remaining if degrees[node] <= 1]
        if not leaves:
            break
        
        for leaf in leaves:
            remaining.discard(leaf)
            for neighbor in adjacency[leaf]:
                if neighbor in remaining:
                    degrees[neighbor] -= 1
    
    return sorted(remaining)


# ============================================================================
# Carbon Skeleton Drawing
# ============================================================================

def draw_carbon_skeleton(adjacency: Dict[int, List[int]],
                         ax: Optional[plt.Axes] = None,
                         layout: str = 'tree',
                         show_labels: bool = True,
                         node_size: float = 0.3,
                         style: str = 'classic',
                         title: Optional[str] = None,
                         font_scale: float = 1.0) -> plt.Axes:
    """
    Draw carbon skeleton of molecule (without hydrogens).
    
    Args:
        adjacency: Adjacency list of carbon backbone
        ax: Matplotlib axes (created if not provided)
        layout: Layout algorithm ('tree', 'radial', 'spring')
        show_labels: Whether to show carbon numbers
        node_size: Size of carbon circles
        style: Color style ('classic', 'colorful', 'neon', 'organic')
        title: Optional title for the plot
        font_scale: Scale factor for font sizes (default 1.0)
        
    Returns:
        Matplotlib axes with drawing
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = STYLE_COLORS.get(style, STYLE_COLORS['classic'])
    ax.set_facecolor(colors['background'])
    
    n = len(adjacency)
    
    if n == 0:
        ax.text(0.5, 0.5, 'Empty molecule', ha='center', va='center',
                transform=ax.transAxes)
        return ax
    
    # Compute layout
    if layout == 'tree':
        positions = compute_tree_layout(adjacency)
    elif layout == 'radial':
        positions = compute_radial_layout(adjacency)
    elif layout == 'spring':
        positions = compute_spring_layout(adjacency)
    else:
        positions = compute_tree_layout(adjacency)
    
    # Draw bonds
    drawn_edges = set()
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = tuple(sorted([node, neighbor]))
            if edge not in drawn_edges:
                drawn_edges.add(edge)
                
                x1, y1 = positions[node]
                x2, y2 = positions[neighbor]
                
                ax.plot([x1, x2], [y1, y2], 
                        color=colors['bond'], 
                        linewidth=3,
                        zorder=1,
                        solid_capstyle='round')
    
    # Draw carbon atoms
    for node, (x, y) in positions.items():
        circle = Circle((x, y), node_size, 
                        facecolor=colors['carbon'],
                        edgecolor='black',
                        linewidth=1.5,
                        zorder=2)
        ax.add_patch(circle)
        
        if show_labels:
            ax.text(x, y, 'C', ha='center', va='center',
                    fontsize=12 * font_scale, fontweight='bold', color='white',
                    zorder=3)
    
    # Set axis properties
    ax.set_aspect('equal')
    ax.autoscale()
    
    # Add padding
    x_coords = [p[0] for p in positions.values()]
    y_coords = [p[1] for p in positions.values()]
    padding = 1.0
    ax.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
    ax.set_ylim(min(y_coords) - padding, max(y_coords) + padding)
    
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14 * font_scale, fontweight='bold')
    
    return ax


def draw_lewis_structure(adjacency: Dict[int, List[int]],
                         ax: Optional[plt.Axes] = None,
                         layout: str = 'tree',
                         show_hydrogens: bool = True,
                         style: str = 'classic',
                         title: Optional[str] = None,
                         font_scale: float = 1.0) -> plt.Axes:
    """
    Draw Lewis structure with hydrogens.
    
    Each carbon needs 4 bonds total. Unfilled valences get hydrogens.
    
    Args:
        adjacency: Adjacency list of carbon backbone
        ax: Matplotlib axes
        layout: Layout algorithm
        show_hydrogens: Whether to explicitly show H atoms
        style: Color style
        title: Plot title
        font_scale: Scale factor for font sizes (default 1.0)
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = STYLE_COLORS.get(style, STYLE_COLORS['classic'])
    ax.set_facecolor(colors['background'])
    
    n = len(adjacency)
    
    if n == 0:
        return ax
    
    # Compute carbon positions
    if layout == 'tree':
        c_positions = compute_tree_layout(adjacency)
    elif layout == 'radial':
        c_positions = compute_radial_layout(adjacency)
    else:
        c_positions = compute_spring_layout(adjacency)
    
    # Scale up positions for hydrogen spacing
    scale = 1.5
    c_positions = {k: (v[0] * scale, v[1] * scale) for k, v in c_positions.items()}
    
    # Draw C-C bonds
    drawn_edges = set()
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = tuple(sorted([node, neighbor]))
            if edge not in drawn_edges:
                drawn_edges.add(edge)
                x1, y1 = c_positions[node]
                x2, y2 = c_positions[neighbor]
                ax.plot([x1, x2], [y1, y2],
                        color=colors['bond'], linewidth=2, zorder=1)
    
    # Compute hydrogen positions and draw
    h_positions = []
    h_radius = 0.15
    c_radius = 0.25
    h_distance = 0.8
    
    for carbon, (cx, cy) in c_positions.items():
        n_bonds = len(adjacency[carbon])
        n_hydrogens = 4 - n_bonds  # Carbon valency = 4
        
        # Get directions to neighboring carbons
        neighbor_angles = []
        for neighbor in adjacency[carbon]:
            nx, ny = c_positions[neighbor]
            angle = np.arctan2(ny - cy, nx - cx)
            neighbor_angles.append(angle)
        
        # Place hydrogens in remaining directions
        if n_hydrogens > 0:
            if not neighbor_angles:
                # Methane - place H's tetrahedrally projected to 2D
                h_angles = [0, np.pi/2, np.pi, 3*np.pi/2][:n_hydrogens]
            else:
                # Find gaps between neighbors
                neighbor_angles.sort()
                gaps = []
                
                for i in range(len(neighbor_angles)):
                    a1 = neighbor_angles[i]
                    a2 = neighbor_angles[(i + 1) % len(neighbor_angles)]
                    
                    gap = a2 - a1
                    if gap <= 0:
                        gap += 2 * np.pi
                    
                    gaps.append((gap, (a1 + a2) / 2 if gap < np.pi else (a1 + a2) / 2 + np.pi))
                
                # Sort by gap size, place H's in largest gaps
                gaps.sort(reverse=True)
                h_angles = []
                
                for i in range(min(n_hydrogens, len(gaps))):
                    base_angle = gaps[i % len(gaps)][1]
                    
                    # Spread multiple H's in same gap
                    if n_hydrogens > len(gaps):
                        offset = (i // len(gaps)) * 0.5 - 0.25
                        h_angles.append(base_angle + offset)
                    else:
                        h_angles.append(base_angle)
            
            # Place and draw hydrogens
            for angle in h_angles[:n_hydrogens]:
                hx = cx + h_distance * np.cos(angle)
                hy = cy + h_distance * np.sin(angle)
                h_positions.append((hx, hy))
                
                if show_hydrogens:
                    # Draw C-H bond
                    ax.plot([cx, hx], [cy, hy],
                            color=colors['bond'], linewidth=1.5, zorder=1)
                    
                    # Draw H atom
                    circle = Circle((hx, hy), h_radius,
                                    facecolor='white',
                                    edgecolor=colors['hydrogen'],
                                    linewidth=1,
                                    zorder=2)
                    ax.add_patch(circle)
                    ax.text(hx, hy, 'H', ha='center', va='center',
                            fontsize=8 * font_scale, color=colors['hydrogen'], zorder=3)
    
    # Draw carbon atoms on top
    for carbon, (cx, cy) in c_positions.items():
        circle = Circle((cx, cy), c_radius,
                        facecolor=colors['carbon'],
                        edgecolor='black',
                        linewidth=1.5,
                        zorder=4)
        ax.add_patch(circle)
        ax.text(cx, cy, 'C', ha='center', va='center',
                fontsize=10 * font_scale, fontweight='bold', color='white', zorder=5)
    
    ax.set_aspect('equal')
    ax.autoscale()
    
    # Padding
    all_x = [p[0] for p in c_positions.values()] + [p[0] for p in h_positions]
    all_y = [p[1] for p in c_positions.values()] + [p[1] for p in h_positions]
    
    if all_x and all_y:
        padding = 1.0
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14 * font_scale, fontweight='bold')
    
    return ax


# ============================================================================
# Grid Visualizations
# ============================================================================

def plot_isomer_grid(isomers: List[Any],
                     n_cols: int = 4,
                     layout: str = 'tree',
                     style: str = 'classic',
                     show_lewis: bool = False,
                     figsize: Optional[Tuple[float, float]] = None,
                     title: Optional[str] = None,
                     font_scale: float = 1.0) -> plt.Figure:
    """
    Plot grid of multiple isomers.
    
    Args:
        isomers: List of AlkaneIsomer objects
        n_cols: Number of columns in grid
        layout: Layout algorithm for each structure
        style: Color style
        show_lewis: If True, show Lewis structures with H's
        figsize: Optional figure size
        title: Overall figure title
        font_scale: Scale factor for font sizes in individual isomer plots
        
    Returns:
        Matplotlib figure
    """
    n = len(isomers)
    
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No isomers to display', ha='center', va='center')
        return fig
    
    n_rows = (n + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, isomer in enumerate(isomers):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        isomer_title = isomer.name if hasattr(isomer, 'name') and isomer.name else f"Isomer {i+1}"
        
        if show_lewis:
            draw_lewis_structure(isomer.adjacency, ax=ax, layout=layout,
                               style=style, title=isomer_title, font_scale=font_scale)
        else:
            draw_carbon_skeleton(isomer.adjacency, ax=ax, layout=layout,
                               style=style, title=isomer_title, font_scale=font_scale)
    
    # Hide empty subplots
    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def plot_isomer_count_curve(max_n: int = 20,
                           log_scale: bool = True,
                           show_known: bool = True,
                           figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot the "explosion" curve showing isomer count vs carbon number.
    
    This is "The Explosion Plot" from the project specification.
    
    Args:
        max_n: Maximum carbon number to plot
        log_scale: Use log scale for y-axis
        show_known: Show known values from OEIS
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Known isomer counts (OEIS A000602)
    known_counts = {
        1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75,
        11: 159, 12: 355, 13: 802, 14: 1858, 15: 4347, 16: 10359,
        17: 24894, 18: 60523, 19: 148284, 20: 366319, 21: 910726,
        22: 2278658, 23: 5731580, 24: 14490245, 25: 36797588,
        26: 93839412, 27: 240215803, 28: 617105614, 29: 1590507121,
        30: 4111846763
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get counts up to max_n
    n_values = list(range(1, min(max_n + 1, 31)))
    counts = [known_counts.get(n, 0) for n in n_values]
    
    # Main plot
    ax.plot(n_values, counts, 'o-', color='#2C3E50', linewidth=2, 
            markersize=8, label='Alkane Isomers')
    
    # Fill area under curve
    ax.fill_between(n_values, counts, alpha=0.3, color='#3498DB')
    
    # Annotations for key points
    annotations = [
        (5, "C₅: 3 isomers\n(pentanes)"),
        (10, "C₁₀: 75 isomers"),
        (15, "C₁₅: 4,347 isomers"),
        (20, "C₂₀: 366,319 isomers"),
        (25, "C₂₅: 36,797,588 isomers"),
        (30, "C₃₀: 4,111,846,763 isomers"),
    ]
    
    # Adjust annotation positions to avoid overlap
    annotation_offsets = {
        5: (10, 30),
        10: (10, 30),
        15: (10, 30),
        20: (-80, 40),
        25: (-100, 30),
        30: (-120, 20),
    }
    
    for n, text in annotations:
        if n <= max_n:
            count = known_counts.get(n, 0)
            offset = annotation_offsets.get(n, (10, 30))
            ax.annotate(text, xy=(n, count), xytext=offset,
                       textcoords='offset points',
                       fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='gray'),
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('Number of Carbons (n)', fontsize=12)
    ax.set_ylabel('Number of Structural Isomers', fontsize=12)
    ax.set_title('The Combinatorial Explosion of Chemical Space\n'
                 'Structural Isomers of Alkanes (CₙH₂ₙ₊₂)',
                 fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Add formula annotation
    ax.text(0.95, 0.05, 
            'Growth rate ≈ 2.5ⁿ\n(asymptotic)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def create_interactive_structure(adjacency: Dict[int, List[int]],
                                 positions: Optional[Dict[int, Tuple[float, float]]] = None
                                 ) -> Dict[str, Any]:
    """
    Create data structure for interactive visualization (e.g., in Streamlit).
    
    Returns dict with:
        - positions: Node positions
        - edges: List of edges
        - node_info: Info about each node
    """
    if positions is None:
        positions = compute_tree_layout(adjacency)
    
    edges = []
    for node, neighbors in adjacency.items():
        for neighbor in neighbors:
            if node < neighbor:
                edges.append({
                    'from': node,
                    'to': neighbor,
                    'from_pos': positions[node],
                    'to_pos': positions[neighbor]
                })
    
    node_info = {}
    for node in adjacency:
        degree = len(adjacency[node])
        node_info[node] = {
            'position': positions[node],
            'degree': degree,
            'n_hydrogens': 4 - degree,
            'type': 'CH3' if degree == 1 else ('CH2' if degree == 2 else 
                    ('CH' if degree == 3 else 'C'))
        }
    
    return {
        'positions': positions,
        'edges': edges,
        'node_info': node_info,
        'n_carbons': len(adjacency),
        'n_hydrogens': sum(4 - len(neighbors) for neighbors in adjacency.values())
    }


# ============================================================================
# Comparison and Analysis Plots
# ============================================================================

def plot_branching_analysis(isomers: List[Any],
                           figsize: Tuple[float, float] = (12, 5)) -> plt.Figure:
    """
    Analyze and plot branching characteristics of isomers.
    
    Args:
        isomers: List of AlkaneIsomer objects
        figsize: Figure size
        
    Returns:
        Matplotlib figure with branching analysis
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Extract properties
    chain_lengths = [iso.max_chain_length for iso in isomers]
    branch_points = [iso.n_branch_points for iso in isomers]
    methyl_groups = [iso.n_methyl_groups for iso in isomers]
    
    # Longest chain histogram
    ax = axes[0]
    ax.hist(chain_lengths, bins=range(min(chain_lengths), max(chain_lengths) + 2),
            color='#3498DB', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Longest Chain Length')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Longest Chains')
    
    # Branch points histogram
    ax = axes[1]
    ax.hist(branch_points, bins=range(min(branch_points), max(branch_points) + 2),
            color='#E74C3C', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Branch Points')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Branch Points')
    
    # Scatter plot
    ax = axes[2]
    ax.scatter(chain_lengths, branch_points, c=methyl_groups, 
               cmap='viridis', s=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Longest Chain')
    ax.set_ylabel('Branch Points')
    ax.set_title('Chain Length vs Branching')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Methyl Groups')
    
    plt.tight_layout()
    return fig


def plot_degree_distribution(isomers: List[Any],
                            figsize: Tuple[float, float] = (8, 5)) -> plt.Figure:
    """
    Plot distribution of vertex degrees across all isomers.
    
    Args:
        isomers: List of AlkaneIsomer objects
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all degrees
    all_degrees = []
    for iso in isomers:
        degrees = [len(neighbors) for neighbors in iso.adjacency.values()]
        all_degrees.extend(degrees)
    
    # Count by degree
    degree_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for d in all_degrees:
        if d in degree_counts:
            degree_counts[d] += 1
    
    labels = ['Terminal\n(CH₃)', 'Linear\n(CH₂)', 'Branch\n(CH)', 'Quaternary\n(C)']
    colors = ['#2ECC71', '#3498DB', '#E74C3C', '#9B59B6']
    
    bars = ax.bar(range(1, 5), [degree_counts[d] for d in range(1, 5)],
                  color=colors, edgecolor='black')
    
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Count')
    ax.set_title('Carbon Atom Types in All Isomers')
    
    # Add value labels
    for bar, count in zip(bars, [degree_counts[d] for d in range(1, 5)]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Demo visualization
    print("Visualization Module Demo")
    print("=" * 50)
    
    # Create sample adjacency (n-hexane)
    hexane = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4]
    }
    
    # Create sample branched isomer (2-methylpentane)
    isopentane = {
        0: [1],
        1: [0, 2, 4],
        2: [1, 3],
        3: [2],
        4: [1]
    }
    
    # Plot carbon skeleton
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    draw_carbon_skeleton(hexane, ax=axes[0, 0], layout='tree', 
                        title='n-Hexane (tree layout)')
    draw_carbon_skeleton(hexane, ax=axes[0, 1], layout='spring',
                        title='n-Hexane (spring layout)')
    draw_lewis_structure(isopentane, ax=axes[1, 0], layout='tree',
                        title='Isopentane (Lewis structure)')
    draw_lewis_structure(hexane, ax=axes[1, 1], layout='tree',
                        title='n-Hexane (Lewis structure)')
    
    plt.tight_layout()
    plt.savefig('visualization_demo.png', dpi=150)
    print("Saved visualization_demo.png")
    
    # Plot explosion curve
    fig = plot_isomer_count_curve(max_n=20)
    plt.savefig('explosion_plot.png', dpi=150)
    print("Saved explosion_plot.png")
    
    plt.show()
