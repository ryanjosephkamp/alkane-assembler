#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Unit Tests for The Alkane Assembler
================================================================================

Project:        Week 4 Project 1: The Alkane Assembler
Module:         test_alkane_generator.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        February 5, 2026
Last Updated:   February 5, 2026

License:        MIT License
================================================================================

Comprehensive test suite for alkane isomer generation, tree utilities, and
visualization functions.

Run tests with: pytest tests/test_alkane_generator.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alkane_generator import (
    AlkaneGenerator,
    AlkaneIsomer,
    generate_alkane_isomers,
    count_alkane_isomers,
    get_isomer_statistics
)
from src.tree_utils import (
    TreeGraph,
    are_trees_isomorphic,
    canonical_form,
    get_tree_center,
    compute_diameter,
    compute_wiener_index,
    compute_balaban_index,
    enumerate_rooted_trees
)
from src.visualization import (
    compute_tree_layout,
    compute_spring_layout,
    compute_radial_layout,
    create_interactive_structure
)


# ============================================================================
# AlkaneIsomer Tests
# ============================================================================

class TestAlkaneIsomer:
    """Tests for AlkaneIsomer dataclass."""
    
    def test_methane_creation(self):
        """Methane should have 1 carbon, no bonds."""
        adj = {0: []}
        isomer = AlkaneIsomer(n_carbons=1, adjacency=adj, name="methane")
        
        assert isomer.n_carbons == 1
        assert isomer.name == "methane"
        assert isomer.max_chain_length == 1
        assert isomer.n_branch_points == 0
        assert isomer.n_methyl_groups == 1
    
    def test_ethane_creation(self):
        """Ethane should have 2 carbons, 1 bond."""
        adj = {0: [1], 1: [0]}
        isomer = AlkaneIsomer(n_carbons=2, adjacency=adj, name="ethane")
        
        assert isomer.n_carbons == 2
        assert isomer.max_chain_length == 2
        assert isomer.n_branch_points == 0
        assert isomer.n_methyl_groups == 2
    
    def test_linear_pentane(self):
        """n-Pentane: linear chain of 5 carbons."""
        adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
        isomer = AlkaneIsomer(n_carbons=5, adjacency=adj, name="n-pentane")
        
        assert isomer.n_carbons == 5
        assert isomer.max_chain_length == 5
        assert isomer.n_branch_points == 0
        assert isomer.n_methyl_groups == 2
    
    def test_neopentane(self):
        """Neopentane: central carbon with 4 CH3 groups."""
        adj = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}
        isomer = AlkaneIsomer(n_carbons=5, adjacency=adj, name="neopentane")
        
        assert isomer.n_carbons == 5
        assert isomer.max_chain_length == 3  # Longest path is 3
        assert isomer.n_branch_points == 1   # Central carbon
        assert isomer.n_methyl_groups == 4
    
    def test_edge_list(self):
        """Test edge list extraction."""
        adj = {0: [1], 1: [0, 2], 2: [1]}
        isomer = AlkaneIsomer(n_carbons=3, adjacency=adj)
        
        edges = isomer.get_edge_list()
        assert len(edges) == 2
        assert (0, 1) in edges
        assert (1, 2) in edges
    
    def test_degree_sequence(self):
        """Test degree sequence computation."""
        # Linear propane: degrees [1, 2, 1]
        adj = {0: [1], 1: [0, 2], 2: [1]}
        isomer = AlkaneIsomer(n_carbons=3, adjacency=adj)
        
        assert isomer.degree_sequence == (1, 1, 2)
        
        # Isobutane: degrees [1, 1, 1, 3]
        adj = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}
        isomer = AlkaneIsomer(n_carbons=4, adjacency=adj)
        
        assert isomer.degree_sequence == (1, 1, 1, 3)


# ============================================================================
# AlkaneGenerator Tests
# ============================================================================

class TestAlkaneGenerator:
    """Tests for AlkaneGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a generator instance for testing."""
        return AlkaneGenerator(use_cache=True)
    
    def test_generate_methane(self, generator):
        """C1 should produce exactly 1 isomer (methane)."""
        isomers = generator.generate(1)
        assert len(isomers) == 1
        assert isomers[0].name == "methane"
    
    def test_generate_ethane(self, generator):
        """C2 should produce exactly 1 isomer (ethane)."""
        isomers = generator.generate(2)
        assert len(isomers) == 1
        assert isomers[0].name == "ethane"
    
    def test_generate_propane(self, generator):
        """C3 should produce exactly 1 isomer (propane)."""
        isomers = generator.generate(3)
        assert len(isomers) == 1
    
    def test_generate_butane(self, generator):
        """C4 should produce exactly 2 isomers."""
        isomers = generator.generate(4)
        assert len(isomers) == 2
    
    def test_generate_pentane(self, generator):
        """C5 should produce exactly 3 isomers."""
        isomers = generator.generate(5)
        assert len(isomers) == 3
    
    def test_generate_hexane(self, generator):
        """C6 should produce exactly 5 isomers."""
        isomers = generator.generate(6)
        assert len(isomers) == 5
    
    def test_generate_heptane(self, generator):
        """C7 should produce exactly 9 isomers."""
        isomers = generator.generate(7)
        assert len(isomers) == 9
    
    def test_generate_octane(self, generator):
        """C8 should produce exactly 18 isomers."""
        isomers = generator.generate(8)
        assert len(isomers) == 18
    
    def test_generate_decane(self, generator):
        """C10 should produce exactly 75 isomers."""
        isomers = generator.generate(10)
        assert len(isomers) == 75
    
    def test_validate_known_counts(self, generator):
        """Validate against all known counts up to C12."""
        expected = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75, 11: 159, 12: 355}
        
        for n, count in expected.items():
            generated = generator.count(n)
            assert generated == count, f"C{n}: expected {count}, got {generated}"
    
    def test_all_isomers_have_valid_degree(self, generator):
        """All generated isomers should have max degree ≤ 4 (carbon valency)."""
        for n in range(1, 10):
            isomers = generator.generate(n)
            for isomer in isomers:
                for node, neighbors in isomer.adjacency.items():
                    assert len(neighbors) <= 4, f"Node {node} has degree {len(neighbors)} > 4"
    
    def test_all_isomers_are_trees(self, generator):
        """All generated structures should be valid trees."""
        for n in range(1, 10):
            isomers = generator.generate(n)
            for isomer in isomers:
                # Edge count should be n-1 for a tree
                edge_count = sum(len(neighbors) for neighbors in isomer.adjacency.values()) // 2
                assert edge_count == n - 1, f"Invalid tree: {n} nodes, {edge_count} edges"
    
    def test_no_duplicate_hashes(self, generator):
        """All generated isomers should have unique canonical hashes."""
        for n in range(1, 10):
            isomers = generator.generate(n)
            hashes = [iso.canonical_hash for iso in isomers]
            assert len(hashes) == len(set(hashes)), f"Duplicate hashes found for C{n}"
    
    def test_cache_functionality(self, generator):
        """Cached results should be identical to fresh generation."""
        # First call populates cache
        isomers1 = generator.generate(6)
        # Second call should use cache
        isomers2 = generator.generate(6)
        
        assert len(isomers1) == len(isomers2)
        for i1, i2 in zip(isomers1, isomers2):
            assert i1.canonical_hash == i2.canonical_hash
    
    def test_invalid_n_raises_error(self, generator):
        """Generating with n < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            generator.generate(0)
        
        with pytest.raises(ValueError):
            generator.generate(-5)


# ============================================================================
# TreeGraph Tests
# ============================================================================

class TestTreeGraph:
    """Tests for TreeGraph class."""
    
    def test_linear_tree_creation(self):
        """Test creation of linear chain tree."""
        tree = TreeGraph.linear_tree(5)
        
        assert tree.n_nodes == 5
        assert len(tree.get_edges()) == 4
        assert tree.get_degree_sequence() == (1, 1, 2, 2, 2)
    
    def test_star_tree_creation(self):
        """Test creation of star tree."""
        tree = TreeGraph.star_tree(5)
        
        assert tree.n_nodes == 5
        assert len(tree.get_edges()) == 4
        assert tree.get_degree_sequence() == (1, 1, 1, 1, 4)
    
    def test_from_edges(self):
        """Test tree creation from edge list."""
        edges = [(0, 1), (1, 2), (2, 3)]
        tree = TreeGraph.from_edges(edges)
        
        assert tree.n_nodes == 4
        assert len(tree.get_edges()) == 3
    
    def test_get_leaves(self):
        """Test leaf node identification."""
        tree = TreeGraph.linear_tree(5)
        leaves = tree.get_leaves()
        
        assert len(leaves) == 2
        assert 0 in leaves
        assert 4 in leaves
    
    def test_adjacency_matrix(self):
        """Test adjacency matrix conversion."""
        tree = TreeGraph.linear_tree(3)
        matrix = tree.to_matrix()
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 1
        assert matrix[1, 0] == 1
        assert matrix[0, 2] == 0


# ============================================================================
# Tree Center Tests
# ============================================================================

class TestTreeCenter:
    """Tests for tree center finding."""
    
    def test_linear_tree_center(self):
        """Linear tree with odd nodes has 1 center."""
        tree = TreeGraph.linear_tree(5)
        centers = get_tree_center(tree)
        
        assert len(centers) == 1
        assert centers[0] == 2  # Middle node
    
    def test_linear_tree_bicentral(self):
        """Linear tree with even nodes has 2 centers."""
        tree = TreeGraph.linear_tree(4)
        centers = get_tree_center(tree)
        
        assert len(centers) == 2
    
    def test_star_tree_center(self):
        """Star tree has center at hub."""
        tree = TreeGraph.star_tree(5)
        centers = get_tree_center(tree)
        
        assert len(centers) == 1
        assert centers[0] == 0  # Hub node


# ============================================================================
# Tree Isomorphism Tests
# ============================================================================

class TestTreeIsomorphism:
    """Tests for tree isomorphism detection."""
    
    def test_identical_trees_isomorphic(self):
        """Same tree should be isomorphic to itself."""
        tree1 = TreeGraph.linear_tree(5)
        tree2 = TreeGraph.linear_tree(5)
        
        assert are_trees_isomorphic(tree1, tree2)
    
    def test_different_trees_not_isomorphic(self):
        """Linear and star trees should not be isomorphic."""
        linear = TreeGraph.linear_tree(5)
        star = TreeGraph.star_tree(5)
        
        assert not are_trees_isomorphic(linear, star)
    
    def test_relabeled_tree_isomorphic(self):
        """Relabeled tree should be isomorphic to original."""
        # Create path 0-1-2-3
        edges1 = [(0, 1), (1, 2), (2, 3)]
        tree1 = TreeGraph.from_edges(edges1)
        
        # Create same structure with different labels: 3-0-2-1
        adj2 = {3: [0], 0: [3, 2], 2: [0, 1], 1: [2]}
        tree2 = TreeGraph(n_nodes=4, adjacency=adj2)
        
        assert are_trees_isomorphic(tree1, tree2)
    
    def test_different_size_not_isomorphic(self):
        """Trees of different sizes cannot be isomorphic."""
        tree1 = TreeGraph.linear_tree(5)
        tree2 = TreeGraph.linear_tree(6)
        
        assert not are_trees_isomorphic(tree1, tree2)


# ============================================================================
# Canonical Form Tests
# ============================================================================

class TestCanonicalForm:
    """Tests for canonical form computation."""
    
    def test_canonical_form_consistent(self):
        """Same tree should always give same canonical hash."""
        tree = TreeGraph.linear_tree(5)
        
        _, hash1 = canonical_form(tree)
        _, hash2 = canonical_form(tree)
        
        assert hash1 == hash2
    
    def test_isomorphic_trees_same_hash(self):
        """Isomorphic trees should have same canonical hash."""
        # Two different labelings of the same structure
        adj1 = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
        tree1 = TreeGraph(n_nodes=4, adjacency=adj1)
        
        adj2 = {0: [3], 3: [0, 1], 1: [3, 2], 2: [1]}
        tree2 = TreeGraph(n_nodes=4, adjacency=adj2)
        
        _, hash1 = canonical_form(tree1)
        _, hash2 = canonical_form(tree2)
        
        assert hash1 == hash2


# ============================================================================
# Tree Metric Tests
# ============================================================================

class TestTreeMetrics:
    """Tests for tree metrics (diameter, Wiener index, etc.)."""
    
    def test_linear_tree_diameter(self):
        """Linear tree diameter = n-1."""
        for n in range(2, 8):
            tree = TreeGraph.linear_tree(n)
            assert compute_diameter(tree) == n - 1
    
    def test_star_tree_diameter(self):
        """Star tree diameter = 2."""
        for n in range(3, 8):
            tree = TreeGraph.star_tree(n)
            assert compute_diameter(tree) == 2
    
    def test_linear_tree_wiener(self):
        """Wiener index of linear tree = n(n-1)(n+1)/6."""
        for n in range(2, 8):
            tree = TreeGraph.linear_tree(n)
            expected = n * (n - 1) * (n + 1) // 6
            assert compute_wiener_index(tree) == expected
    
    def test_star_tree_wiener(self):
        """Wiener index of star tree = (n-1)² + (n-1)."""
        for n in range(2, 8):
            tree = TreeGraph.star_tree(n)
            # Each leaf-leaf distance = 2, each leaf-center distance = 1
            # Sum = (n-1 choose 2)*2 + (n-1)*1 = (n-1)(n-2) + (n-1) = (n-1)²
            expected = (n - 1) * (n - 1)
            assert compute_wiener_index(tree) == expected


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_generate_alkane_isomers(self):
        """Test generate_alkane_isomers function."""
        isomers = generate_alkane_isomers(5)
        assert len(isomers) == 3
    
    def test_count_alkane_isomers(self):
        """Test count_alkane_isomers function."""
        assert count_alkane_isomers(1) == 1
        assert count_alkane_isomers(5) == 3
        assert count_alkane_isomers(10) == 75
    
    def test_get_isomer_statistics(self):
        """Test get_isomer_statistics function."""
        stats = get_isomer_statistics(max_n=5)
        
        assert 'n' in stats
        assert 'count' in stats
        assert 'log_count' in stats
        assert 'ratio' in stats
        
        assert len(stats['n']) == 5
        assert stats['count'] == [1, 1, 1, 2, 3]


# ============================================================================
# Layout Tests
# ============================================================================

class TestLayouts:
    """Tests for visualization layout algorithms."""
    
    def test_tree_layout_positions(self):
        """Tree layout should produce valid positions."""
        adj = {0: [1, 2], 1: [0], 2: [0]}
        positions = compute_tree_layout(adj)
        
        assert len(positions) == 3
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions.values())
    
    def test_spring_layout_positions(self):
        """Spring layout should produce valid positions."""
        adj = {0: [1, 2], 1: [0], 2: [0]}
        positions = compute_spring_layout(adj)
        
        assert len(positions) == 3
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions.values())
    
    def test_radial_layout_positions(self):
        """Radial layout should produce valid positions."""
        adj = {0: [1, 2], 1: [0], 2: [0]}
        positions = compute_radial_layout(adj)
        
        assert len(positions) == 3
    
    def test_interactive_structure_data(self):
        """Interactive structure should contain required data."""
        adj = {0: [1, 2], 1: [0], 2: [0]}
        data = create_interactive_structure(adj)
        
        assert 'positions' in data
        assert 'edges' in data
        assert 'node_info' in data
        assert data['n_carbons'] == 3


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_single_carbon_isomer(self):
        """Single carbon should work correctly."""
        isomers = generate_alkane_isomers(1)
        assert len(isomers) == 1
        assert isomers[0].adjacency == {0: []}
    
    def test_large_alkane_generation(self):
        """Generation should work for larger alkanes (C12)."""
        isomers = generate_alkane_isomers(12)
        assert len(isomers) == 355
    
    def test_generator_cache_independence(self):
        """Different generator instances should work independently."""
        gen1 = AlkaneGenerator(use_cache=True)
        gen2 = AlkaneGenerator(use_cache=True)
        
        isomers1 = gen1.generate(5)
        isomers2 = gen2.generate(5)
        
        assert len(isomers1) == len(isomers2)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple modules."""
    
    def test_full_generation_and_visualization_pipeline(self):
        """Test complete pipeline from generation to visualization data."""
        # Generate isomers
        isomers = generate_alkane_isomers(5)
        
        # Create visualization data for each
        for isomer in isomers:
            data = create_interactive_structure(isomer.adjacency)
            
            assert data['n_carbons'] == 5
            assert data['n_hydrogens'] == 12  # C5H12
            assert len(data['edges']) == 4  # Tree with 5 nodes has 4 edges
    
    def test_isomer_uniqueness_by_properties(self):
        """Generated isomers should have distinguishing properties."""
        isomers = generate_alkane_isomers(5)
        
        # Pentane isomers: n-pentane, isopentane, neopentane
        # They should have different degree sequences or chain lengths
        properties = set()
        for iso in isomers:
            props = (iso.degree_sequence, iso.max_chain_length, iso.n_branch_points)
            properties.add(props)
        
        # At least some properties should differ
        assert len(properties) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
