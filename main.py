#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The Alkane Assembler - Command-Line Demonstration
================================================================================

Project:        Week 4 Project 1: The Alkane Assembler
Module:         main.py

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        February 5, 2026
Last Updated:   February 5, 2026

License:        MIT License
================================================================================

This script provides command-line demonstrations of alkane isomer generation
and visualization. It showcases:

1. Isomer enumeration for any carbon count
2. The "Explosion Plot" showing combinatorial growth
3. Structural comparison of isomers
4. Validation against known counts (OEIS A000602)
5. Interactive exploration of chemical space

Usage:
    python main.py                   # Run all demonstrations
    python main.py --generate 6      # Generate hexane isomers
    python main.py --explosion       # Show explosion plot
    python main.py --validate        # Validate against known counts
    python main.py --compare 5       # Compare all pentane isomers
    python main.py --save            # Save figures to output/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

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
    compute_wiener_index
)
from src.visualization import (
    draw_carbon_skeleton,
    draw_lewis_structure,
    plot_isomer_grid,
    plot_isomer_count_curve,
    plot_branching_analysis,
    plot_degree_distribution,
    STYLE_COLORS
)


def print_header():
    """Print welcome banner."""
    print("=" * 70)
    print(" " * 15 + "⚗️  THE ALKANE ASSEMBLER  ⚗️")
    print(" " * 10 + "Algorithmic Enumeration of Structural Isomers")
    print("=" * 70)
    print()
    print("Author: Ryan Kamp")
    print("Affiliation: University of Cincinnati, Department of Computer Science")
    print("Project: Week 4 Project 1 - Biophysics Portfolio")
    print()


def demo_generate(n: int, save_figures: bool = False, output_dir: Path = None):
    """
    Demonstrate isomer generation for n carbons.
    
    This shows all unique structural isomers of alkane CnH(2n+2).
    """
    print("\n" + "=" * 60)
    print(f"📊  GENERATING C{n}H{2*n+2} ISOMERS  📊")
    print("=" * 60)
    print()
    
    generator = AlkaneGenerator(verbose=True)
    
    start_time = time.time()
    isomers = generator.generate(n)
    elapsed = time.time() - start_time
    
    print(f"\n✓ Generated {len(isomers)} structural isomers in {elapsed:.3f}s")
    print()
    
    # Display isomer information
    print("Isomer Details:")
    print("-" * 60)
    
    for i, iso in enumerate(isomers, 1):
        print(f"\n{i}. {iso.name}")
        print(f"   Formula: C{n}H{2*n+2}")
        print(f"   Longest chain: {iso.max_chain_length} carbons")
        print(f"   Branch points: {iso.n_branch_points}")
        print(f"   Terminal CH₃ groups: {iso.n_methyl_groups}")
        print(f"   Degree sequence: {iso.degree_sequence}")
    
    # Create visualization
    if len(isomers) <= 20:  # Only plot if reasonable number
        print(f"\n📈 Creating structure visualization...")
        
        fig = plot_isomer_grid(
            isomers, 
            n_cols=min(4, len(isomers)),
            layout='tree',
            style='classic',
            show_lewis=False,
            title=f"All Structural Isomers of C{n}H{2*n+2}"
        )
        
        if save_figures and output_dir:
            filepath = output_dir / f"c{n}_isomers_skeleton.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        # Also create Lewis structure view
        fig_lewis = plot_isomer_grid(
            isomers,
            n_cols=min(3, len(isomers)),
            layout='tree',
            style='classic',
            show_lewis=True,
            title=f"Lewis Structures: C{n}H{2*n+2} Isomers"
        )
        
        if save_figures and output_dir:
            filepath = output_dir / f"c{n}_isomers_lewis.png"
            fig_lewis.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        if not save_figures:
            plt.show()
    else:
        print(f"\n⚠ Too many isomers ({len(isomers)}) to display individually.")
        print("  Use --explosion to see the growth curve.")
    
    return isomers


def demo_explosion(max_n: int = 20, save_figures: bool = False, output_dir: Path = None):
    """
    Demonstrate the combinatorial explosion of chemical space.
    
    This is "The Explosion Plot" - showing how isomer count grows with carbon number.
    """
    print("\n" + "=" * 60)
    print("💥  THE EXPLOSION PLOT  💥")
    print("=" * 60)
    print()
    
    print("Counting structural isomers for n = 1 to", max_n, "...")
    print()
    
    # Known counts for display
    known_counts = {
        1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75,
        11: 159, 12: 355, 13: 802, 14: 1858, 15: 4347, 16: 10359,
        17: 24894, 18: 60523, 19: 148284, 20: 366319
    }
    
    print("┌────────┬─────────────────┬────────────────┐")
    print("│   n    │    Isomers      │    Formula     │")
    print("├────────┼─────────────────┼────────────────┤")
    
    for n in range(1, min(max_n + 1, 21)):
        count = known_counts.get(n, "?")
        formula = f"C{n}H{2*n+2}"
        
        # Add commas for readability
        if isinstance(count, int):
            count_str = f"{count:,}"
        else:
            count_str = str(count)
        
        print(f"│  {n:>3}   │  {count_str:>13}  │  {formula:^12}  │")
    
    print("└────────┴─────────────────┴────────────────┘")
    
    # Key insights
    print("\n📊 Key Observations:")
    print("   • C5 (pentane): 3 isomers")
    print("   • C10 (decane): 75 isomers")
    print("   • C15: 4,347 isomers")
    print("   • C20: 366,319 isomers")
    print("   • C30: ~4.1 billion isomers!")
    print()
    print("   The growth rate is approximately 2.5^n (exponential)")
    print("   This is why computational methods are essential for")
    print("   exploring 'Chemical Space' in drug discovery.")
    
    # Create explosion plot
    print(f"\n📈 Creating explosion plot...")
    
    fig = plot_isomer_count_curve(max_n=max_n, log_scale=True)
    
    if save_figures and output_dir:
        filepath = output_dir / "explosion_plot.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"   Saved: {filepath}")
    
    if not save_figures:
        plt.show()
    
    return fig


def demo_validate(max_n: int = 12, save_figures: bool = False, output_dir: Path = None):
    """
    Validate generator against known isomer counts.
    
    The known counts come from OEIS sequence A000602.
    """
    print("\n" + "=" * 60)
    print("✓  VALIDATION AGAINST OEIS A000602  ✓")
    print("=" * 60)
    print()
    
    print("The Online Encyclopedia of Integer Sequences (OEIS)")
    print("sequence A000602 lists the number of alkane isomers.")
    print()
    print("Validating our generator...")
    print()
    
    generator = AlkaneGenerator()
    
    # Known counts
    known = AlkaneGenerator.KNOWN_COUNTS
    
    all_correct = True
    results = []
    
    print("┌────────┬───────────┬──────────┬─────────┐")
    print("│   n    │  Expected │ Generated│ Status  │")
    print("├────────┼───────────┼──────────┼─────────┤")
    
    for n in range(1, min(max_n + 1, 13)):
        start_time = time.time()
        generated = generator.count(n)
        elapsed = time.time() - start_time
        
        expected = known.get(n, -1)
        
        if generated == expected:
            status = "  ✓  "
        else:
            status = "  ✗  "
            all_correct = False
        
        results.append({
            'n': n,
            'expected': expected,
            'generated': generated,
            'correct': generated == expected,
            'time': elapsed
        })
        
        print(f"│  {n:>3}   │   {expected:>5}   │   {generated:>5}  │ {status} │")
    
    print("└────────┴───────────┴──────────┴─────────┘")
    
    if all_correct:
        print("\n✓ All counts match! Generator is validated.")
    else:
        print("\n✗ Some counts don't match. Check the algorithm.")
    
    # Timing analysis
    print("\n⏱ Performance Analysis:")
    for r in results:
        if r['time'] > 0.001:
            print(f"   C{r['n']}: {r['time']*1000:.1f} ms")
    
    return all_correct


def demo_compare(n: int, save_figures: bool = False, output_dir: Path = None):
    """
    Compare structural properties of all isomers with n carbons.
    """
    print("\n" + "=" * 60)
    print(f"🔬  STRUCTURAL COMPARISON: C{n}H{2*n+2} ISOMERS  🔬")
    print("=" * 60)
    print()
    
    isomers = generate_alkane_isomers(n)
    
    print(f"Analyzing {len(isomers)} isomers...")
    print()
    
    # Table header
    print("┌─────────────────────────┬───────┬────────┬───────┬─────────┐")
    print("│        Name             │ Chain │ Branch │  CH₃  │ Wiener  │")
    print("├─────────────────────────┼───────┼────────┼───────┼─────────┤")
    
    for iso in isomers:
        # Compute Wiener index
        tree = TreeGraph(n_nodes=n, adjacency=iso.adjacency)
        wiener = compute_wiener_index(tree)
        
        name = iso.name[:23] if len(iso.name) > 23 else iso.name
        print(f"│ {name:<23} │  {iso.max_chain_length:>3}  │   {iso.n_branch_points:>2}   │  {iso.n_methyl_groups:>3}  │  {wiener:>5}  │")
    
    print("└─────────────────────────┴───────┴────────┴───────┴─────────┘")
    
    print("\nLegend:")
    print("  Chain  = Longest carbon chain (main chain for naming)")
    print("  Branch = Number of branch points (carbons with 3+ bonds to C)")
    print("  CH₃    = Number of terminal methyl groups")
    print("  Wiener = Wiener index (sum of all pairwise distances)")
    
    # Create analysis plots
    if len(isomers) >= 3:
        print(f"\n📈 Creating structural analysis...")
        
        fig = plot_branching_analysis(isomers)
        
        if save_figures and output_dir:
            filepath = output_dir / f"c{n}_branching_analysis.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"   Saved: {filepath}")
        
        if not save_figures:
            plt.show()
    
    return isomers


def demo_interactive_exploration():
    """
    Interactive exploration of isomers at the command line.
    """
    print("\n" + "=" * 60)
    print("🎮  INTERACTIVE ISOMER EXPLORER  🎮")
    print("=" * 60)
    print()
    print("Enter a carbon count (1-15) to see its isomers.")
    print("Type 'q' to quit.")
    print()
    
    generator = AlkaneGenerator()
    
    while True:
        try:
            user_input = input("Carbon count > ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            
            n = int(user_input)
            
            if n < 1 or n > 15:
                print("Please enter a number between 1 and 15.")
                continue
            
            print()
            isomers = generator.generate(n)
            
            print(f"C{n}H{2*n+2} has {len(isomers)} structural isomer(s):")
            
            for i, iso in enumerate(isomers, 1):
                print(f"  {i}. {iso.name}")
            
            print()
            
        except ValueError:
            print("Invalid input. Enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def demo_all(save_figures: bool = False):
    """Run all demonstrations."""
    output_dir = None
    if save_figures:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        print(f"Saving figures to: {output_dir}")
    
    # 1. Generate examples
    demo_generate(5, save_figures, output_dir)
    demo_generate(6, save_figures, output_dir)
    demo_generate(7, save_figures, output_dir)
    
    # 2. Explosion plot
    demo_explosion(20, save_figures, output_dir)
    
    # 3. Validation
    demo_validate(12, save_figures, output_dir)
    
    # 4. Structural comparison
    demo_compare(6, save_figures, output_dir)
    
    print("\n" + "=" * 60)
    print("✓ All demonstrations complete!")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="The Alkane Assembler - Structural Isomer Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                   Run all demonstrations
    python main.py --generate 6      Generate hexane isomers
    python main.py --explosion       Show explosion plot
    python main.py --validate        Validate against known counts
    python main.py --compare 5       Compare pentane isomers
    python main.py --interactive     Interactive mode
    python main.py --save            Save all figures
        """
    )
    
    parser.add_argument('--generate', '-g', type=int, metavar='N',
                        help='Generate and display all isomers for CN')
    parser.add_argument('--explosion', '-e', action='store_true',
                        help='Show the explosion plot')
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Validate against OEIS sequence')
    parser.add_argument('--compare', '-c', type=int, metavar='N',
                        help='Compare structural properties of CN isomers')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive exploration mode')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save figures to output directory')
    parser.add_argument('--max-n', type=int, default=20,
                        help='Maximum carbon count for explosion plot')
    
    args = parser.parse_args()
    
    print_header()
    
    # Set up output directory
    output_dir = None
    if args.save:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        print(f"Saving figures to: {output_dir}")
    
    # Run specific demos based on arguments
    if args.generate:
        demo_generate(args.generate, args.save, output_dir)
    elif args.explosion:
        demo_explosion(args.max_n, args.save, output_dir)
    elif args.validate:
        demo_validate(12, args.save, output_dir)
    elif args.compare:
        demo_compare(args.compare, args.save, output_dir)
    elif args.interactive:
        demo_interactive_exploration()
    else:
        # Run all demos
        demo_all(args.save)


if __name__ == "__main__":
    main()
