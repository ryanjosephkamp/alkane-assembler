#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Alkane Isomer Generator - Core Module
================================================================================

Project:        Week 4 Project 1: The Alkane Assembler
Module:         alkane_generator.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        February 5, 2026
Last Updated:   February 5, 2026

License:        MIT License
================================================================================

This module provides the core algorithms for generating all structural isomers
of alkanes (CnH2n+2) using graph-theoretic tree generation.

Chemical Background:
-------------------
Alkanes are saturated hydrocarbons consisting only of carbon and hydrogen atoms
with single bonds. Each carbon has a maximum valency of 4 (degree ≤ 4 in the
graph representation). The carbon skeleton of an alkane forms a tree structure
(acyclic connected graph).

The number of structural isomers grows rapidly with carbon count:
- C1 (methane): 1 isomer
- C4 (butane): 2 isomers  
- C5 (pentane): 3 isomers
- C10 (decane): 75 isomers
- C15 (pentadecane): 4,347 isomers
- C20 (icosane): 366,319 isomers

This combinatorial explosion is known as "Chemical Space" and motivates
computational approaches to molecular enumeration.

Algorithm:
---------
We use recursive tree generation with canonical isomorphism checking:
1. Generate all possible trees with n nodes
2. Filter to trees where all nodes have degree ≤ 4 (carbon valency)
3. Use canonical form to eliminate isomorphic duplicates
4. Store unique isomers with their structural properties

Key insight: Two trees are isomorphic if one can be transformed into the
other by relabeling nodes. We use the "canonical form" (based on tree center
and level-order traversal) to efficiently detect duplicates.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, FrozenSet, Iterator
from collections import defaultdict
import itertools
from functools import lru_cache
import hashlib


@dataclass
class AlkaneIsomer:
    """
    Represents a unique alkane structural isomer.
    
    Attributes:
        n_carbons: Number of carbon atoms
        adjacency: Adjacency list representation {node: [neighbors]}
        name: IUPAC name if known (for small alkanes)
        canonical_hash: Unique hash identifying this structure
        degree_sequence: Sorted list of node degrees
        max_chain_length: Length of longest carbon chain (for nomenclature)
        n_methyl_groups: Count of CH3 terminal groups
        n_branch_points: Count of carbons with degree > 2
    """
    n_carbons: int
    adjacency: Dict[int, List[int]]
    name: str = ""
    canonical_hash: str = ""
    degree_sequence: Tuple[int, ...] = field(default_factory=tuple)
    max_chain_length: int = 0
    n_methyl_groups: int = 0
    n_branch_points: int = 0
    
    def __post_init__(self):
        """Compute derived properties."""
        if not self.degree_sequence:
            degrees = sorted([len(neighbors) for neighbors in self.adjacency.values()])
            self.degree_sequence = tuple(degrees)
        
        if self.n_methyl_groups == 0:
            # Methyl groups: terminal carbons (degree 1) or isolated carbons (degree 0 for methane)
            self.n_methyl_groups = sum(1 for neighbors in self.adjacency.values() 
                                        if len(neighbors) <= 1)
        
        if self.n_branch_points == 0:
            self.n_branch_points = sum(1 for neighbors in self.adjacency.values() 
                                        if len(neighbors) > 2)
        
        if self.max_chain_length == 0:
            self.max_chain_length = self._compute_longest_chain()
    
    def _compute_longest_chain(self) -> int:
        """Find length of longest path in the tree (diameter)."""
        if self.n_carbons <= 1:
            return self.n_carbons
        
        # BFS from arbitrary node to find farthest node
        def bfs_farthest(start: int) -> Tuple[int, int]:
            visited = {start}
            queue = [(start, 0)]
            farthest_node = start
            farthest_dist = 0
            
            while queue:
                node, dist = queue.pop(0)
                if dist > farthest_dist:
                    farthest_dist = dist
                    farthest_node = node
                
                for neighbor in self.adjacency.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            return farthest_node, farthest_dist
        
        # Two BFS passes to find diameter
        end1, _ = bfs_farthest(0)
        _, diameter = bfs_farthest(end1)
        
        return diameter + 1  # +1 to convert edge count to node count
    
    def get_edge_list(self) -> List[Tuple[int, int]]:
        """Return list of edges (each edge appears once)."""
        edges = set()
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))
                edges.add(edge)
        return list(edges)
    
    def to_networkx(self):
        """Convert to NetworkX graph (if available)."""
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(self.n_carbons))
            G.add_edges_from(self.get_edge_list())
            return G
        except ImportError:
            raise ImportError("NetworkX is required for this operation")
    
    def __hash__(self):
        return hash(self.canonical_hash)
    
    def __eq__(self, other):
        if not isinstance(other, AlkaneIsomer):
            return False
        return self.canonical_hash == other.canonical_hash


class AlkaneGenerator:
    """
    Generator for all structural isomers of alkanes with n carbons.
    
    Uses recursive tree generation with isomorphism pruning based on
    canonical form comparison. This is more efficient than generating
    all labeled trees and then checking isomorphism pairwise.
    
    The algorithm:
    1. Trees are built by adding one node at a time
    2. Each addition is canonicalized immediately  
    3. Duplicates are detected by hash lookup
    4. Only valid additions (degree ≤ 4) are considered
    
    Example usage:
        generator = AlkaneGenerator()
        isomers = generator.generate(n=6)  # All hexane isomers
        print(f"Found {len(isomers)} hexane isomers")
    """
    
    # Known isomer counts for validation (OEIS A000602)
    KNOWN_COUNTS = {
        1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75,
        11: 159, 12: 355, 13: 802, 14: 1858, 15: 4347, 16: 10359,
        17: 24894, 18: 60523, 19: 148284, 20: 366319
    }
    
    # IUPAC names for small alkanes
    IUPAC_NAMES = {
        1: ["methane"],
        2: ["ethane"],
        3: ["propane"],
        4: ["n-butane", "isobutane"],
        5: ["n-pentane", "isopentane", "neopentane"],
        6: ["n-hexane", "2-methylpentane", "3-methylpentane", 
            "2,2-dimethylbutane", "2,3-dimethylbutane"],
        7: ["n-heptane", "2-methylhexane", "3-methylhexane",
            "2,2-dimethylpentane", "2,3-dimethylpentane", 
            "2,4-dimethylpentane", "3,3-dimethylpentane",
            "3-ethylpentane", "2,2,3-trimethylbutane"]
    }
    
    MAX_CARBON_DEGREE = 4  # Tetravalent carbon
    
    def __init__(self, use_cache: bool = True, verbose: bool = False):
        """
        Initialize the alkane generator.
        
        Args:
            use_cache: Cache generated isomers for reuse
            verbose: Print progress information
        """
        self.use_cache = use_cache
        self.verbose = verbose
        self._cache: Dict[int, List[AlkaneIsomer]] = {}
    
    def generate(self, n: int) -> List[AlkaneIsomer]:
        """
        Generate all structural isomers of alkane with n carbons.
        
        Args:
            n: Number of carbon atoms (must be >= 1)
            
        Returns:
            List of unique AlkaneIsomer objects
        """
        if n < 1:
            raise ValueError("Number of carbons must be at least 1")
        
        # Check cache
        if self.use_cache and n in self._cache:
            return self._cache[n]
        
        if self.verbose:
            print(f"Generating alkane isomers for C{n}...")
        
        # Base cases
        if n == 1:
            isomers = [self._create_methane()]
        elif n == 2:
            isomers = [self._create_ethane()]
        else:
            isomers = self._generate_recursive(n)
        
        # Assign IUPAC names if known
        self._assign_names(isomers, n)
        
        # Cache result
        if self.use_cache:
            self._cache[n] = isomers
        
        if self.verbose:
            print(f"  Found {len(isomers)} isomers")
        
        return isomers
    
    def _create_methane(self) -> AlkaneIsomer:
        """Create methane (single carbon, no bonds)."""
        adj = {0: []}
        return AlkaneIsomer(
            n_carbons=1,
            adjacency=adj,
            name="methane",
            canonical_hash=self._compute_hash(adj),
            max_chain_length=1,
            n_methyl_groups=1,
            n_branch_points=0
        )
    
    def _create_ethane(self) -> AlkaneIsomer:
        """Create ethane (two carbons, one bond)."""
        adj = {0: [1], 1: [0]}
        return AlkaneIsomer(
            n_carbons=2,
            adjacency=adj,
            name="ethane",
            canonical_hash=self._compute_hash(adj),
            max_chain_length=2,
            n_methyl_groups=2,
            n_branch_points=0
        )
    
    def _generate_recursive(self, n: int) -> List[AlkaneIsomer]:
        """
        Generate all trees with n nodes where max degree ≤ 4.
        
        Uses the algorithm from Otter's theorem and practical tree enumeration:
        1. Start with smaller trees (n-1 nodes)
        2. Add a new node by connecting to an existing node
        3. Check if the result is canonical (to avoid duplicates)
        4. Validate that all degrees remain ≤ 4
        """
        # Get all (n-1) node trees
        smaller_trees = self.generate(n - 1)
        
        seen_hashes: Set[str] = set()
        isomers: List[AlkaneIsomer] = []
        
        for tree in smaller_trees:
            # Try adding a new node to each existing node
            for attach_point in range(n - 1):
                # Check if attachment point has room (degree < 4)
                current_degree = len(tree.adjacency[attach_point])
                if current_degree >= self.MAX_CARBON_DEGREE:
                    continue
                
                # Create new tree with added node
                new_adj = self._add_node_to_tree(tree.adjacency, attach_point, n - 1)
                
                # Compute canonical hash
                canonical_adj = self._canonicalize(new_adj)
                hash_val = self._compute_hash(canonical_adj)
                
                # Check for duplicate
                if hash_val not in seen_hashes:
                    seen_hashes.add(hash_val)
                    
                    isomer = AlkaneIsomer(
                        n_carbons=n,
                        adjacency=canonical_adj,
                        canonical_hash=hash_val
                    )
                    isomers.append(isomer)
        
        return isomers
    
    def _add_node_to_tree(self, adj: Dict[int, List[int]], 
                          attach_to: int, new_node: int) -> Dict[int, List[int]]:
        """Create new adjacency dict with added node."""
        new_adj = {k: list(v) for k, v in adj.items()}
        new_adj[new_node] = [attach_to]
        new_adj[attach_to].append(new_node)
        return new_adj
    
    def _canonicalize(self, adj: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Compute canonical form of tree.
        
        Uses tree centering and consistent labeling based on subtree signatures.
        The canonical form is unique for each isomorphism class.
        """
        n = len(adj)
        if n <= 2:
            return {i: list(adj[i]) for i in range(n)}
        
        # Find tree center(s) - nodes that minimize max distance to any leaf
        centers = self._find_centers(adj)
        
        # Root at center (or centroid edge for bicentral trees)
        if len(centers) == 1:
            root = centers[0]
        else:
            # Bicentral tree - create virtual root
            # Use the center that gives lexicographically smaller canonical form
            canonical1 = self._canonical_from_root(adj, centers[0])
            canonical2 = self._canonical_from_root(adj, centers[1])
            
            hash1 = self._compute_hash(canonical1)
            hash2 = self._compute_hash(canonical2)
            
            return canonical1 if hash1 <= hash2 else canonical2
        
        return self._canonical_from_root(adj, root)
    
    def _find_centers(self, adj: Dict[int, List[int]]) -> List[int]:
        """
        Find center(s) of tree using iterative leaf removal.
        
        A tree has either 1 center (odd diameter) or 2 centers (even diameter).
        """
        n = len(adj)
        if n <= 2:
            return list(range(n))
        
        # Work with copies
        degrees = {node: len(neighbors) for node, neighbors in adj.items()}
        remaining = set(adj.keys())
        
        # Iteratively remove leaves
        while len(remaining) > 2:
            leaves = [node for node in remaining if degrees[node] <= 1]
            
            if not leaves:
                break
            
            for leaf in leaves:
                remaining.discard(leaf)
                for neighbor in adj[leaf]:
                    if neighbor in remaining:
                        degrees[neighbor] -= 1
        
        return list(remaining)
    
    def _canonical_from_root(self, adj: Dict[int, List[int]], root: int) -> Dict[int, List[int]]:
        """
        Create canonical labeling of tree rooted at given node.
        
        Uses DFS with subtree signature-based ordering of children.
        """
        n = len(adj)
        
        # Compute subtree signatures for consistent ordering
        signatures = {}
        
        def compute_signature(node: int, parent: int) -> str:
            """Compute canonical signature of subtree."""
            children = [n for n in adj[node] if n != parent]
            
            if not children:
                return "()"
            
            child_sigs = []
            for child in children:
                child_sigs.append(compute_signature(child, node))
            
            child_sigs.sort()
            return "(" + "".join(child_sigs) + ")"
        
        # Get signature for whole tree
        root_sig = compute_signature(root, -1)
        
        # Now relabel nodes in canonical order
        new_label = {}
        current_label = 0
        
        def assign_labels(node: int, parent: int):
            nonlocal current_label
            new_label[node] = current_label
            current_label += 1
            
            children = [n for n in adj[node] if n != parent]
            # Sort children by their subtree signatures
            child_sigs = [(compute_signature(c, node), c) for c in children]
            child_sigs.sort()
            
            for _, child in child_sigs:
                assign_labels(child, node)
        
        assign_labels(root, -1)
        
        # Build canonical adjacency
        canonical = {i: [] for i in range(n)}
        for node, neighbors in adj.items():
            new_node = new_label[node]
            for neighbor in neighbors:
                canonical[new_node].append(new_label[neighbor])
            canonical[new_node].sort()
        
        return canonical
    
    def _compute_hash(self, adj: Dict[int, List[int]]) -> str:
        """Compute unique hash for adjacency structure."""
        # Create deterministic string representation
        items = sorted(adj.items())
        repr_str = str([(k, sorted(v)) for k, v in items])
        return hashlib.md5(repr_str.encode()).hexdigest()[:16]
    
    def _assign_names(self, isomers: List[AlkaneIsomer], n: int):
        """Assign IUPAC names to isomers if known."""
        if n not in self.IUPAC_NAMES:
            # Generate systematic names for larger alkanes
            for i, isomer in enumerate(isomers):
                isomer.name = f"C{n}H{2*n+2} isomer {i+1}"
            return
        
        names = self.IUPAC_NAMES[n]
        
        # Sort isomers by longest chain (descending) then branch points (ascending)
        # This matches typical IUPAC ordering
        sorted_isomers = sorted(
            isomers,
            key=lambda x: (-x.max_chain_length, x.n_branch_points, x.degree_sequence)
        )
        
        for isomer, name in zip(sorted_isomers, names):
            isomer.name = name
    
    def count(self, n: int) -> int:
        """Return count of isomers without generating all structures."""
        return len(self.generate(n))
    
    def validate(self, max_n: int = 10) -> bool:
        """
        Validate generator against known isomer counts.
        
        Returns:
            True if all counts match known values
        """
        all_valid = True
        
        for n in range(1, min(max_n + 1, 21)):
            generated = self.count(n)
            expected = self.KNOWN_COUNTS.get(n, -1)
            
            if expected != -1 and generated != expected:
                print(f"MISMATCH at C{n}: generated {generated}, expected {expected}")
                all_valid = False
            elif self.verbose:
                print(f"C{n}: {generated} isomers ✓")
        
        return all_valid


# ============================================================================
# Convenience Functions
# ============================================================================

def generate_alkane_isomers(n: int, verbose: bool = False) -> List[AlkaneIsomer]:
    """
    Generate all structural isomers of alkane with n carbons.
    
    Args:
        n: Number of carbon atoms
        verbose: Print progress information
        
    Returns:
        List of AlkaneIsomer objects
    
    Example:
        >>> isomers = generate_alkane_isomers(5)
        >>> for iso in isomers:
        ...     print(iso.name)
        n-pentane
        isopentane
        neopentane
    """
    generator = AlkaneGenerator(verbose=verbose)
    return generator.generate(n)


def count_alkane_isomers(n: int) -> int:
    """
    Count structural isomers of alkane with n carbons.
    
    Args:
        n: Number of carbon atoms
        
    Returns:
        Number of unique structural isomers
    """
    generator = AlkaneGenerator()
    return generator.count(n)


def get_isomer_statistics(max_n: int = 15) -> Dict[str, List]:
    """
    Compute statistics about alkane isomers for n = 1 to max_n.
    
    Returns dictionary with:
        - 'n': list of carbon counts
        - 'count': list of isomer counts
        - 'log_count': log10 of counts
        - 'ratio': ratio of count[n] / count[n-1]
    """
    generator = AlkaneGenerator()
    
    stats = {
        'n': [],
        'count': [],
        'log_count': [],
        'ratio': []
    }
    
    prev_count = 1
    for n in range(1, max_n + 1):
        count = generator.count(n)
        
        stats['n'].append(n)
        stats['count'].append(count)
        stats['log_count'].append(np.log10(count) if count > 0 else 0)
        stats['ratio'].append(count / prev_count if prev_count > 0 else 0)
        
        prev_count = count
    
    return stats


# ============================================================================
# Alternative Counting Methods (for verification)
# ============================================================================

@lru_cache(maxsize=1000)
def polya_count_alkyl_groups(n: int) -> int:
    """
    Count rooted trees (alkyl groups) using Pólya enumeration.
    
    An alkyl group is an alkane with one hydrogen removed (like a radical).
    This is equivalent to counting rooted trees where the root has degree ≤ 3.
    
    Uses the recurrence from Pólya's theorem specialized to trees.
    """
    if n == 0:
        return 1
    if n == 1:
        return 1
    
    # Memoized partition function for trees
    # This is a simplified version - full Pólya requires generating functions
    count = 0
    
    # Enumerate partitions of n-1 into at most 3 parts
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                if i + j + k == n - 1:
                    # Count trees with subtrees of sizes i, j, k
                    ti = polya_count_alkyl_groups(i)
                    tj = polya_count_alkyl_groups(j)
                    tk = polya_count_alkyl_groups(k)
                    
                    if i == j == k:
                        # All same size - symmetry factor
                        count += (ti * (ti + 1) * (ti + 2)) // 6
                    elif i == j or j == k:
                        # Two same size
                        if i == j:
                            count += ((ti * (ti + 1)) // 2) * tk
                        else:
                            count += ti * ((tj * (tj + 1)) // 2)
                    else:
                        # All different
                        count += ti * tj * tk
    
    return count


if __name__ == "__main__":
    # Quick demonstration
    print("=" * 60)
    print("ALKANE ISOMER GENERATOR")
    print("=" * 60)
    
    # Validate against known counts
    generator = AlkaneGenerator(verbose=True)
    print("\nValidating against known isomer counts...")
    generator.validate(max_n=10)
    
    # Show examples
    print("\n" + "=" * 60)
    print("PENTANE ISOMERS (C5H12)")
    print("=" * 60)
    
    pentanes = generator.generate(5)
    for isomer in pentanes:
        print(f"  {isomer.name}:")
        print(f"    Longest chain: {isomer.max_chain_length}")
        print(f"    Branch points: {isomer.n_branch_points}")
        print(f"    Methyl groups: {isomer.n_methyl_groups}")
        print(f"    Degree sequence: {isomer.degree_sequence}")
        print()
