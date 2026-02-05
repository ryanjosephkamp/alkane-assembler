#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Tree Utilities Module for Alkane Isomer Generation
================================================================================

Project:        Week 4 Project 1: The Alkane Assembler
Module:         tree_utils.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        February 5, 2026
Last Updated:   February 5, 2026

License:        MIT License
================================================================================

This module provides utility functions for tree graph operations, including:
- Tree isomorphism checking
- Canonical form computation
- Tree center finding
- Various tree traversal algorithms

These utilities support the alkane isomer generation by providing efficient
methods to detect duplicate molecular structures.

Mathematical Background:
-----------------------
A tree is an undirected, connected, acyclic graph. Two trees are isomorphic
if there exists a bijection between their vertex sets that preserves adjacency.

The isomorphism problem for trees is solvable in linear time O(n) using
canonical forms based on the tree center. This is much faster than general
graph isomorphism (unknown complexity) because trees have special structure.

Canonical Form Algorithm:
------------------------
1. Find the center(s) of the tree (1 or 2 nodes)
2. Root the tree at the center
3. Assign "signatures" to subtrees recursively
4. Sort children by their signatures
5. The resulting labeled tree is the canonical form
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, FrozenSet, Iterator
from collections import deque
import hashlib


@dataclass
class TreeGraph:
    """
    A tree graph represented by adjacency list.
    
    Attributes:
        n_nodes: Number of nodes in the tree
        adjacency: Dict mapping node -> list of neighbors
        root: Optional designated root node
        node_labels: Optional labels for nodes (e.g., atom types)
    """
    n_nodes: int
    adjacency: Dict[int, List[int]]
    root: Optional[int] = None
    node_labels: Optional[Dict[int, str]] = None
    
    def __post_init__(self):
        """Validate tree structure."""
        if not self._is_valid_tree():
            raise ValueError("Invalid tree structure: not connected or contains cycles")
    
    def _is_valid_tree(self) -> bool:
        """Check if graph is a valid tree (connected, acyclic)."""
        if self.n_nodes == 0:
            return True
        if self.n_nodes == 1:
            return len(self.adjacency.get(0, [])) == 0
        
        # Check edge count (tree has n-1 edges)
        edge_count = sum(len(neighbors) for neighbors in self.adjacency.values())
        if edge_count != 2 * (self.n_nodes - 1):
            return False
        
        # Check connectivity via BFS
        visited = set()
        queue = [0]
        visited.add(0)
        
        while queue:
            node = queue.pop(0)
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == self.n_nodes
    
    @classmethod
    def from_edges(cls, edges: List[Tuple[int, int]], n_nodes: Optional[int] = None) -> 'TreeGraph':
        """
        Create tree from edge list.
        
        Args:
            edges: List of (u, v) tuples representing edges
            n_nodes: Number of nodes (inferred if not provided)
        """
        if n_nodes is None:
            n_nodes = max(max(e) for e in edges) + 1 if edges else 1
        
        adjacency = {i: [] for i in range(n_nodes)}
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)
        
        # Sort adjacency lists for consistency
        for node in adjacency:
            adjacency[node].sort()
        
        return cls(n_nodes=n_nodes, adjacency=adjacency)
    
    @classmethod
    def linear_tree(cls, n: int) -> 'TreeGraph':
        """Create a linear chain tree (path graph) with n nodes."""
        if n < 1:
            raise ValueError("Need at least 1 node")
        
        edges = [(i, i + 1) for i in range(n - 1)]
        return cls.from_edges(edges, n_nodes=n)
    
    @classmethod
    def star_tree(cls, n: int) -> 'TreeGraph':
        """Create a star tree with n nodes (center + n-1 leaves)."""
        if n < 1:
            raise ValueError("Need at least 1 node")
        
        edges = [(0, i) for i in range(1, n)]
        return cls.from_edges(edges, n_nodes=n)
    
    def get_degree(self, node: int) -> int:
        """Get degree (number of neighbors) of a node."""
        return len(self.adjacency.get(node, []))
    
    def get_leaves(self) -> List[int]:
        """Get all leaf nodes (degree 1)."""
        return [node for node in self.adjacency if self.get_degree(node) == 1]
    
    def get_degree_sequence(self) -> Tuple[int, ...]:
        """Get sorted degree sequence of the tree."""
        degrees = [self.get_degree(node) for node in range(self.n_nodes)]
        return tuple(sorted(degrees))
    
    def get_max_degree(self) -> int:
        """Get maximum degree in the tree."""
        return max(self.get_degree(node) for node in range(self.n_nodes))
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """Get list of edges (each edge once, with smaller node first)."""
        edges = set()
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                edges.add((min(node, neighbor), max(node, neighbor)))
        return sorted(edges)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to adjacency matrix."""
        matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                matrix[node, neighbor] = 1
        return matrix


def get_tree_center(tree: TreeGraph) -> List[int]:
    """
    Find the center(s) of a tree.
    
    The center is the set of vertices that minimize the maximum distance
    to any other vertex (the eccentricity). A tree has either 1 center
    (central vertex) or 2 centers (central edge).
    
    Algorithm: Iteratively remove all leaves until 1 or 2 nodes remain.
    
    Args:
        tree: A TreeGraph object
        
    Returns:
        List of center node(s) (1 or 2 elements)
    """
    if tree.n_nodes <= 2:
        return list(range(tree.n_nodes))
    
    # Work with mutable copies
    degrees = {node: tree.get_degree(node) for node in range(tree.n_nodes)}
    remaining = set(range(tree.n_nodes))
    
    while len(remaining) > 2:
        # Find current leaves
        leaves = [node for node in remaining if degrees[node] <= 1]
        
        if not leaves:
            break
        
        # Remove leaves
        for leaf in leaves:
            remaining.discard(leaf)
            for neighbor in tree.adjacency[leaf]:
                if neighbor in remaining:
                    degrees[neighbor] -= 1
    
    return sorted(remaining)


def compute_subtree_signature(tree: TreeGraph, node: int, parent: int) -> str:
    """
    Compute canonical signature of subtree rooted at node.
    
    The signature uniquely identifies the tree structure (up to isomorphism)
    and is computed recursively as parenthesized concatenation of child signatures.
    
    Args:
        tree: The tree graph
        node: Root of subtree
        parent: Parent of node (to avoid backtracking)
        
    Returns:
        String signature like "()(())" representing subtree structure
    """
    children = [n for n in tree.adjacency[node] if n != parent]
    
    if not children:
        return "()"
    
    child_sigs = []
    for child in children:
        child_sigs.append(compute_subtree_signature(tree, child, node))
    
    child_sigs.sort()
    return "(" + "".join(child_sigs) + ")"


def canonical_form(tree: TreeGraph) -> Tuple[TreeGraph, str]:
    """
    Compute canonical form of tree.
    
    The canonical form is a relabeling of the tree such that isomorphic
    trees get identical canonical forms. This allows O(n log n) isomorphism
    checking by simple comparison.
    
    Args:
        tree: Input tree graph
        
    Returns:
        Tuple of (canonical_tree, canonical_hash)
    """
    if tree.n_nodes <= 1:
        return tree, hashlib.md5(b"single").hexdigest()[:16]
    
    centers = get_tree_center(tree)
    
    if len(centers) == 1:
        root = centers[0]
        canonical_adj = _canonical_from_root(tree, root)
    else:
        # Bicentral tree - try both centers, pick lexicographically smaller
        adj1 = _canonical_from_root(tree, centers[0])
        adj2 = _canonical_from_root(tree, centers[1])
        
        str1 = _adjacency_to_string(adj1)
        str2 = _adjacency_to_string(adj2)
        
        canonical_adj = adj1 if str1 <= str2 else adj2
    
    hash_val = hashlib.md5(_adjacency_to_string(canonical_adj).encode()).hexdigest()[:16]
    
    return TreeGraph(n_nodes=tree.n_nodes, adjacency=canonical_adj), hash_val


def _canonical_from_root(tree: TreeGraph, root: int) -> Dict[int, List[int]]:
    """Create canonical adjacency from rooted tree."""
    n = tree.n_nodes
    
    # Assign new labels based on DFS with signature ordering
    new_label = {}
    current_label = [0]  # Use list for mutable in nested function
    
    def assign_labels(node: int, parent: int):
        new_label[node] = current_label[0]
        current_label[0] += 1
        
        children = [n for n in tree.adjacency[node] if n != parent]
        
        # Sort children by subtree signature for consistent ordering
        child_sigs = [(compute_subtree_signature(tree, c, node), c) for c in children]
        child_sigs.sort()
        
        for _, child in child_sigs:
            assign_labels(child, node)
    
    assign_labels(root, -1)
    
    # Build canonical adjacency
    canonical = {i: [] for i in range(n)}
    for node, neighbors in tree.adjacency.items():
        new_node = new_label[node]
        for neighbor in neighbors:
            canonical[new_node].append(new_label[neighbor])
        canonical[new_node].sort()
    
    return canonical


def _adjacency_to_string(adj: Dict[int, List[int]]) -> str:
    """Convert adjacency dict to deterministic string for hashing."""
    items = sorted(adj.items())
    return str([(k, sorted(v)) for k, v in items])


def are_trees_isomorphic(tree1: TreeGraph, tree2: TreeGraph) -> bool:
    """
    Check if two trees are isomorphic.
    
    Uses canonical form comparison for O(n log n) complexity.
    
    Args:
        tree1, tree2: Trees to compare
        
    Returns:
        True if trees are isomorphic
    """
    if tree1.n_nodes != tree2.n_nodes:
        return False
    
    if tree1.get_degree_sequence() != tree2.get_degree_sequence():
        return False
    
    _, hash1 = canonical_form(tree1)
    _, hash2 = canonical_form(tree2)
    
    return hash1 == hash2


def tree_to_adjacency_list(tree: TreeGraph) -> Dict[int, List[int]]:
    """Convert tree to adjacency list representation."""
    return {k: list(v) for k, v in tree.adjacency.items()}


# ============================================================================
# Tree Metrics and Analysis
# ============================================================================

def compute_diameter(tree: TreeGraph) -> int:
    """
    Compute diameter (longest path length) of tree.
    
    Uses two-BFS algorithm: BFS from any node to find farthest,
    then BFS from that node to find actual diameter.
    """
    if tree.n_nodes <= 1:
        return 0
    
    def bfs_farthest(start: int) -> Tuple[int, int]:
        visited = {start: 0}
        queue = deque([start])
        farthest = start
        max_dist = 0
        
        while queue:
            node = queue.popleft()
            dist = visited[node]
            
            if dist > max_dist:
                max_dist = dist
                farthest = node
            
            for neighbor in tree.adjacency[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append(neighbor)
        
        return farthest, max_dist
    
    # First BFS from arbitrary node
    end1, _ = bfs_farthest(0)
    # Second BFS from found endpoint
    _, diameter = bfs_farthest(end1)
    
    return diameter


def compute_wiener_index(tree: TreeGraph) -> int:
    """
    Compute Wiener index (sum of all pairwise distances) of tree.
    
    The Wiener index is a topological descriptor used in chemistry
    to characterize molecular branching.
    """
    if tree.n_nodes <= 1:
        return 0
    
    total = 0
    
    for start in range(tree.n_nodes):
        # BFS to find all distances from start
        visited = {start: 0}
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            dist = visited[node]
            
            for neighbor in tree.adjacency[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append(neighbor)
                    total += dist + 1  # Add distance to this neighbor
    
    # Each pair counted twice
    return total // 2


def compute_balaban_index(tree: TreeGraph) -> float:
    """
    Compute Balaban J index (average distance sum connectivity).
    
    Another topological index for molecular characterization.
    """
    if tree.n_nodes <= 2:
        return 0.0
    
    n = tree.n_nodes
    m = n - 1  # Number of edges in tree
    
    # Compute distance sums for each node
    dist_sums = []
    
    for start in range(n):
        visited = {start: 0}
        queue = deque([start])
        dist_sum = 0
        
        while queue:
            node = queue.popleft()
            dist = visited[node]
            
            for neighbor in tree.adjacency[node]:
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append(neighbor)
                    dist_sum += dist + 1
        
        dist_sums.append(dist_sum)
    
    # Compute J index
    j_sum = 0.0
    for u, v in tree.get_edges():
        j_sum += 1.0 / np.sqrt(dist_sums[u] * dist_sums[v])
    
    cyclomatic = m - n + 2  # For tree, this is 1
    
    return m / (cyclomatic + 1) * j_sum


def find_all_paths(tree: TreeGraph, start: int, end: int) -> List[List[int]]:
    """
    Find all paths between two nodes in tree.
    
    Since it's a tree, there's exactly one path.
    """
    if start == end:
        return [[start]]
    
    def dfs(node: int, target: int, path: List[int], visited: Set[int]) -> Optional[List[int]]:
        if node == target:
            return path + [node]
        
        visited.add(node)
        
        for neighbor in tree.adjacency[node]:
            if neighbor not in visited:
                result = dfs(neighbor, target, path + [node], visited)
                if result:
                    return result
        
        return None
    
    path = dfs(start, end, [], set())
    return [path] if path else []


def get_subtree_sizes(tree: TreeGraph, root: int) -> Dict[int, int]:
    """
    Compute size of subtree rooted at each node.
    
    Args:
        tree: The tree graph
        root: Root node for the tree
        
    Returns:
        Dict mapping node -> subtree size (including node itself)
    """
    sizes = {}
    
    def compute_size(node: int, parent: int) -> int:
        size = 1
        for child in tree.adjacency[node]:
            if child != parent:
                size += compute_size(child, node)
        sizes[node] = size
        return size
    
    compute_size(root, -1)
    return sizes


# ============================================================================
# Tree Enumeration Helpers
# ============================================================================

def enumerate_rooted_trees(n: int, max_degree: int = 4) -> List[TreeGraph]:
    """
    Enumerate all non-isomorphic rooted trees with n nodes.
    
    Args:
        n: Number of nodes
        max_degree: Maximum allowed degree (4 for carbon)
        
    Returns:
        List of non-isomorphic trees
    """
    if n == 1:
        return [TreeGraph(n_nodes=1, adjacency={0: []})]
    
    seen_hashes = set()
    trees = []
    
    # Recursively build from smaller trees
    for smaller in enumerate_rooted_trees(n - 1, max_degree):
        for attach_point in range(n - 1):
            if smaller.get_degree(attach_point) >= max_degree:
                continue
            
            # Add new node
            new_adj = {k: list(v) for k, v in smaller.adjacency.items()}
            new_adj[n - 1] = [attach_point]
            new_adj[attach_point].append(n - 1)
            
            new_tree = TreeGraph(n_nodes=n, adjacency=new_adj)
            _, tree_hash = canonical_form(new_tree)
            
            if tree_hash not in seen_hashes:
                seen_hashes.add(tree_hash)
                trees.append(new_tree)
    
    return trees


if __name__ == "__main__":
    # Quick tests
    print("Tree Utilities Module Tests")
    print("=" * 50)
    
    # Test linear tree
    linear = TreeGraph.linear_tree(5)
    print(f"Linear tree (5 nodes): {linear.get_edges()}")
    print(f"  Degree sequence: {linear.get_degree_sequence()}")
    print(f"  Center: {get_tree_center(linear)}")
    print(f"  Diameter: {compute_diameter(linear)}")
    
    # Test star tree
    star = TreeGraph.star_tree(5)
    print(f"\nStar tree (5 nodes): {star.get_edges()}")
    print(f"  Degree sequence: {star.get_degree_sequence()}")
    print(f"  Center: {get_tree_center(star)}")
    print(f"  Diameter: {compute_diameter(star)}")
    
    # Test isomorphism
    print(f"\nLinear ≅ Star: {are_trees_isomorphic(linear, star)}")
    
    # Test canonical form
    _, hash1 = canonical_form(linear)
    _, hash2 = canonical_form(linear)
    print(f"Same tree gives same hash: {hash1 == hash2}")
    
    # Enumerate small trees
    for n in range(1, 8):
        trees = enumerate_rooted_trees(n, max_degree=4)
        print(f"\nTrees with {n} nodes (max degree 4): {len(trees)}")
