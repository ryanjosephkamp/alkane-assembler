#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The Alkane Assembler - Interactive Streamlit Application
================================================================================

Project:        Week 4 Project 1: The Alkane Assembler
Module:         app.py (Streamlit Web Application)

Author:         Ryan Kamp
Affiliation:    University of Cincinnati Department of Computer Science
Email:          kamprj@mail.uc.edu
GitHub:         https://github.com/ryanjosephkamp

Created:        February 5, 2026
Last Updated:   February 5, 2026

License:        MIT License
================================================================================

An interactive Streamlit application demonstrating the algorithmic enumeration
of alkane structural isomers using graph-theoretic tree generation.

Features:
---------
- Interactive carbon count selection
- Real-time isomer generation and visualization
- "The Explosion Plot" showing combinatorial growth of chemical space
- Click-to-unfold Lewis structure visualization
- Structural comparison and analysis tools
- Educational theory section with Cayley's theorem and Burnside's lemma

The key educational insight is demonstrating CHEMICAL SPACE complexity:
- Why C5H12 has 3 isomers but C20H42 has 366,319 isomers
- Why computational approaches are essential for drug discovery
- How graph theory connects to organic chemistry
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import time
from typing import Optional, Tuple, List, Dict
import sys
from pathlib import Path

# Add src to path
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
    compute_diameter,
    compute_wiener_index,
    compute_balaban_index
)
from src.visualization import (
    draw_carbon_skeleton,
    draw_lewis_structure,
    plot_isomer_grid,
    plot_isomer_count_curve,
    plot_branching_analysis,
    plot_degree_distribution,
    compute_tree_layout,
    compute_spring_layout,
    STYLE_COLORS
)
from src.properties import (
    get_alkane_properties,
    format_state_emoji,
    format_temperature,
    AlkaneProperties
)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="The Alkane Assembler",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .isomer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .explosion-text {
        font-size: 3rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .formula-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 1rem 0;
        color: #2c3e50;
    }
    .warning-box {
        background: #fdf2e9;
        border-left: 4px solid #e67e22;
        padding: 1rem;
        margin: 1rem 0;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

if 'generator' not in st.session_state:
    st.session_state.generator = AlkaneGenerator(use_cache=True)

if 'selected_isomer' not in st.session_state:
    st.session_state.selected_isomer = None

if 'current_isomers' not in st.session_state:
    st.session_state.current_isomers = []


# ============================================================================
# Helper Functions
# ============================================================================

def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


def get_alkane_name(n: int) -> str:
    """Get IUPAC prefix for alkane with n carbons."""
    prefixes = {
        1: "meth", 2: "eth", 3: "prop", 4: "but",
        5: "pent", 6: "hex", 7: "hept", 8: "oct",
        9: "non", 10: "dec", 11: "undec", 12: "dodec",
        13: "tridec", 14: "tetradec", 15: "pentadec",
        16: "hexadec", 17: "heptadec", 18: "octadec",
        19: "nonadec", 20: "icos"
    }
    return prefixes.get(n, f"C{n}") + "ane" if n in prefixes else f"C{n}H{2*n+2}"


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("# ⚗️ The Alkane Assembler")
    st.markdown("---")
    
    st.markdown("### Navigation")
    page = st.radio(
        "Select a page:",
        ["🔢 Isomer Generator", "💥 The Explosion Plot", "🔬 Structure Explorer", 
         "📊 Analysis Dashboard", "📚 Theory & Background"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool demonstrates the **algorithmic enumeration** of alkane 
    structural isomers using graph-theoretic methods.
    """)
    
    st.markdown("**Key Concepts:**")
    
    with st.expander("🌳 Alkanes as Trees"):
        st.markdown("""
        **Alkanes** are saturated hydrocarbons with formula C_n H_{2n+2}.
        
        In graph theory, alkane carbon skeletons are **trees**:
        - **Connected**: All carbons are linked
        - **Acyclic**: No rings or cycles
        - **Max degree 4**: Carbon has 4 bonds (valency)
        
        This allows us to enumerate alkane isomers by generating
        all unique trees with n nodes and maximum degree 4.
        """)
    
    with st.expander("🔄 Graph Isomorphism"):
        st.markdown("""
        Two graphs are **isomorphic** if one can be transformed into
        the other by relabeling vertices.
        
        For alkanes, this means two structures are the *same molecule*
        if their carbon connectivity patterns are identical, regardless
        of how we number the atoms.
        
        **Example:** These are the SAME molecule:
        ```
        C-C-C-C-C    vs    C-C-C-C-C
        1-2-3-4-5          5-4-3-2-1
        ```
        """)
    
    with st.expander("📐 Canonical Forms"):
        st.markdown("""
        A **canonical form** is a unique, standardized representation
        of a graph that allows efficient isomorphism testing.
        
        **How it works:**
        1. Find the tree's center (1-2 central nodes)
        2. Root the tree at the center
        3. Recursively compute subtree signatures
        4. Sort and hash to create a unique fingerprint
        
        Two trees are isomorphic **if and only if** they have the
        same canonical form. This enables O(n log n) comparison.
        """)
    
    with st.expander("💥 Combinatorial Explosion"):
        st.markdown("""
        The number of alkane isomers grows **exponentially** with
        carbon count:
        
        | Carbons | Isomers |
        |---------|----------|
        | 5 | 3 |
        | 10 | 75 |
        | 15 | 4,347 |
        | 20 | 366,319 |
        
        This "explosion" demonstrates why computational methods
        are essential in chemistry—we cannot enumerate by hand!
        """)
    
    with st.expander("📊 Structural Properties"):
        st.markdown("""
        Key metrics for comparing isomers:
        
        - **Longest Chain**: Maximum path length in the tree
        - **Branch Points**: Carbons with 3+ neighbors
        - **Methyl Groups**: Terminal carbons (degree ≤ 1)
        - **Wiener Index**: Sum of all pairwise distances
        - **Diameter**: Maximum distance between any two nodes
        """)
    
    st.markdown("---")
    st.markdown("### Author")
    st.markdown("""
    **Ryan Kamp**  
    University of Cincinnati  
    Department of Computer Science  
    [kamprj@mail.uc.edu](mailto:kamprj@mail.uc.edu)
    """)


# ============================================================================
# Page: Isomer Generator
# ============================================================================

if page == "🔢 Isomer Generator":
    st.markdown('<h1 class="main-header">🔢 Alkane Isomer Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate all structural isomers for any alkane</p>', unsafe_allow_html=True)
    
    # Known isomer counts for estimation (OEIS A000602)
    known_counts = {
        1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75,
        11: 159, 12: 355, 13: 802, 14: 1858, 15: 4347, 16: 10359,
        17: 24894, 18: 60523, 19: 148284, 20: 366319, 21: 910726,
        22: 2278658, 23: 5731580, 24: 14490245, 25: 36797588,
        26: 93839412, 27: 240215803, 28: 617105614, 29: 1590507121,
        30: 4111846763
    }
    
    # Estimated generation times (rough approximations based on isomer counts)
    def estimate_generation_time(n: int) -> str:
        """Estimate generation time based on carbon count."""
        if n <= 10:
            return "< 1 second"
        elif n <= 12:
            return "~1-2 seconds"
        elif n <= 14:
            return "~5-15 seconds"
        elif n <= 16:
            return "~30 seconds - 1 minute"
        elif n <= 18:
            return "~2-5 minutes"
        elif n <= 20:
            return "~10-30 minutes"
        elif n <= 22:
            return "~1-3 hours"
        elif n <= 24:
            return "~6-12 hours"
        elif n <= 26:
            return "~1-3 days"
        else:
            return "Several days or more"
    
    # Initialize session state for this page
    if 'generator_n' not in st.session_state:
        st.session_state.generator_n = 5
    if 'generated_isomers' not in st.session_state:
        st.session_state.generated_isomers = None
    if 'last_generated_n' not in st.session_state:
        st.session_state.last_generated_n = None
    
    # Carbon count selector
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        n_carbons = st.slider(
            "Number of Carbons",
            min_value=1,
            max_value=30,
            value=st.session_state.generator_n,
            key="n_carbon_slider",
            help="Select the number of carbon atoms (1-30)"
        )
        st.session_state.generator_n = n_carbons
        
        alkane_name = get_alkane_name(n_carbons)
        formula = f"C{n_carbons}H{2*n_carbons+2}"
        
        st.markdown(f"### {alkane_name.capitalize()} ({formula})")
        
        # Show estimated info
        expected_isomers = known_counts.get(n_carbons, "Unknown")
        estimated_time = estimate_generation_time(n_carbons)
        
        st.markdown(f"""
        **Expected isomers:** {expected_isomers:,} | **Estimated time:** {estimated_time}
        """)
        
        # Warning for large n
        if n_carbons >= 18:
            st.warning(f"⚠️ **Warning:** Generating isomers for C{n_carbons} may take {estimated_time}. Consider starting with a smaller value.")
        elif n_carbons >= 14:
            st.info(f"ℹ️ Generation for C{n_carbons} may take {estimated_time}.")
        
        # Generate button
        generate_clicked = st.button("🧪 Generate Isomers", type="primary", use_container_width=True)
    
    # Generate isomers only when button is clicked or if we have cached results
    if generate_clicked or (st.session_state.last_generated_n == n_carbons and st.session_state.generated_isomers is not None):
        if generate_clicked or st.session_state.last_generated_n != n_carbons:
            # Generate new isomers
            with st.spinner(f"Generating isomers for {formula}... This may take {estimated_time}."):
                start_time = time.time()
                isomers = st.session_state.generator.generate(n_carbons)
                elapsed = time.time() - start_time
            
            st.session_state.generated_isomers = isomers
            st.session_state.last_generated_n = n_carbons
            st.session_state.generation_time = elapsed
        else:
            # Use cached isomers
            isomers = st.session_state.generated_isomers
            elapsed = st.session_state.generation_time
        
        st.session_state.current_isomers = isomers
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(isomers):,}</div>
                <div class="metric-label">Structural Isomers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_carbons}</div>
                <div class="metric-label">Carbon Atoms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{2*n_carbons + 2}</div>
                <div class="metric-label">Hydrogen Atoms</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if elapsed < 1:
                time_str = f"{elapsed*1000:.1f}ms"
            elif elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                time_str = f"{elapsed/60:.1f}min"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{time_str}</div>
                <div class="metric-label">Generation Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization options
        col1, col2 = st.columns(2)
        
        with col1:
            view_type = st.radio(
                "View Type",
                ["Carbon Skeleton", "Lewis Structure"],
                horizontal=True
            )
        
        with col2:
            layout_type = st.selectbox(
                "Layout Algorithm",
                ["tree", "spring", "radial"]
            )
        
        # Display isomers with adaptive sizing and font scaling
        # Determine display parameters based on isomer count and carbon count
        num_isomers = len(isomers)
        
        if num_isomers <= 12:
            display_count = num_isomers
            n_cols = min(4, num_isomers)
            cell_size = 3.0
            font_scale = 1.0
        elif num_isomers <= 20:
            display_count = num_isomers
            n_cols = 5
            cell_size = 2.5
            font_scale = 0.85
        elif num_isomers <= 50:
            display_count = 20
            n_cols = 5
            cell_size = 2.5
            font_scale = 0.8
        elif num_isomers <= 200:
            display_count = 16
            n_cols = 4
            cell_size = 2.5
            font_scale = 0.75
        elif num_isomers <= 1000:
            display_count = 12
            n_cols = 4
            cell_size = 2.0
            font_scale = 0.65
        elif num_isomers <= 10000:
            display_count = 12
            n_cols = 4
            cell_size = 1.8
            font_scale = 0.55
        else:
            display_count = 12
            n_cols = 4
            cell_size = 1.6
            font_scale = 0.45
        
        n_rows = (display_count + n_cols - 1) // n_cols
        figsize = (cell_size * n_cols, cell_size * n_rows)
        
        if num_isomers <= display_count:
            st.markdown("### All Structural Isomers")
            
            fig = plot_isomer_grid(
                isomers,
                n_cols=n_cols,
                layout=layout_type,
                show_lewis=(view_type == "Lewis Structure"),
                figsize=figsize,
                title=None,
                font_scale=font_scale
            )
            st.pyplot(fig)
            plt.close(fig)
            
        else:
            st.warning(f"⚠️ {num_isomers:,} isomers is too many to display all at once. Showing first {display_count}.")
            
            fig = plot_isomer_grid(
                isomers[:display_count],
                n_cols=n_cols,
                layout=layout_type,
                show_lewis=(view_type == "Lewis Structure"),
                figsize=figsize,
                title=f"First {display_count} of {num_isomers:,} {formula} Isomers",
                font_scale=font_scale
            )
            st.pyplot(fig)
            plt.close(fig)
        
        # Isomer details table
        st.markdown("### Isomer Details")
        
        # Limit table to reasonable size
        table_limit = min(len(isomers), 100)
        isomer_data = []
        for i, iso in enumerate(isomers[:table_limit]):
            tree = TreeGraph(n_nodes=n_carbons, adjacency=iso.adjacency)
            isomer_data.append({
                "Name": iso.name,
                "Longest Chain": iso.max_chain_length,
                "Branch Points": iso.n_branch_points,
                "CH₃ Groups": iso.n_methyl_groups,
                "Wiener Index": compute_wiener_index(tree)
            })
        
        if len(isomers) > table_limit:
            st.info(f"Showing first {table_limit} of {len(isomers):,} isomers in table.")
        
        st.dataframe(isomer_data, width='stretch')
        
        # Informational dropdowns
        with st.expander("What does this visualization show? (Click to expand)"):
            st.markdown("""
            The **All Structural Isomers** visualization displays all unique structural 
            arrangements of carbon atoms for the selected alkane formula. Each diagram 
            represents a distinct molecule with the same molecular formula but different 
            atomic connectivity.
            
            **Key Terms:**
            
            - **Carbon Skeleton:** The arrangement of carbon-carbon bonds in a molecule, 
              shown without hydrogen atoms. This simplified representation highlights the 
              fundamental structure that distinguishes one isomer from another. Carbon 
              atoms are shown as nodes (often labeled "C"), and C-C bonds as connecting lines.
            
            - **Lewis Structure:** A more complete representation showing all atoms and bonds, 
              including hydrogen atoms. Each carbon forms 4 bonds total (to other carbons 
              and/or hydrogens), satisfying the octet rule. This view helps visualize the 
              actual molecular composition.
            
            - **Layout Algorithms:**
              - *Tree Layout:* Arranges the carbon skeleton in a hierarchical tree structure, 
                with branches extending downward. Best for visualizing the branching pattern 
                and overall structure of the molecule.
              - *Spring Layout:* Uses a physics-based simulation where atoms repel each other 
                like charged particles while bonds act as springs. Creates a balanced, 
                aesthetically pleasing arrangement.
              - *Radial Layout:* Places one carbon at the center with others arranged in 
                concentric circles based on their distance from the center. Useful for 
                highlighting symmetry and connectivity patterns.
            """)
        
        with st.expander("What does this table show? (Click to expand)"):
            st.markdown("""
            The **Isomer Details** table provides quantitative structural properties for each 
            isomer, allowing systematic comparison of different molecular arrangements.
            
            **Column Definitions:**
            
            - **Name:** The IUPAC (International Union of Pure and Applied Chemistry) systematic 
              name or common name for the isomer. Names reflect the structure: "n-" prefix 
              indicates a straight chain, "iso-" indicates a specific branching pattern, and 
              "neo-" indicates maximum branching at one carbon.
            
            - **Longest Chain:** The maximum number of carbon atoms in any continuous path 
              through the molecule. This determines the base name in IUPAC nomenclature. 
              A longer chain generally indicates a more linear (less branched) structure.
            
            - **Branch Points:** The number of carbon atoms bonded to 3 or more other carbons. 
              These are the "junction" carbons where the molecule branches. More branch points 
              typically mean a more compact, spherical molecular shape.
            
            - **CH₃ Groups:** The count of methyl groups (terminal carbons bonded to only one 
              other carbon). These are the "ends" of carbon chains. A molecule with more CH₃ 
              groups has more chain terminations and typically more branching.
            
            - **Wiener Index:** A topological index calculated as the sum of all shortest-path 
              distances between pairs of carbon atoms. Lower values indicate more compact 
              structures (highly branched), while higher values indicate more extended 
              structures (linear chains). This index correlates with physical properties 
              like boiling point and molecular surface area.
            
            **Interpreting the Data:**
            
            Compare isomers by their properties: linear alkanes (like n-pentane) have the 
            longest chain, zero branch points, and highest Wiener index. Highly branched 
            isomers (like neopentane) have shorter longest chains, more branch points, 
            and lower Wiener indices.
            """)
    
    else:
        # No isomers generated yet - show instructions
        st.markdown("---")
        st.markdown("""
        <div class="info-box">
        <strong>👆 Select a carbon count above and click "Generate Isomers" to begin.</strong><br><br>
        The generator will enumerate all structurally distinct alkane isomers for your chosen 
        formula using graph-theoretic tree generation with isomorphism detection.
        </div>
        """, unsafe_allow_html=True)
        
        # Show a preview table of what to expect
        st.markdown("### Preview: Expected Isomer Counts")
        preview_data = []
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
            preview_data.append({
                "Carbons": n,
                "Formula": f"C{n}H{2*n+2}",
                "Isomers": f"{known_counts[n]:,}",
                "Est. Time": estimate_generation_time(n)
            })
        st.dataframe(preview_data, width='stretch')
        
        # Dropdown explaining time estimation
        with st.expander("How are generation times estimated? (Click to expand)"):
            st.markdown("""
            The estimated generation times are **rough approximations** based on:
            
            1. **Exponential Growth**: Isomer counts grow as approximately $2.5^n$, so 
               generation time increases exponentially with carbon count.
            
            2. **Algorithm Complexity**: Each isomer requires tree generation and 
               canonical form computation for isomorphism checking.
            
            3. **Calibration**: Estimates are calibrated against known isomer counts 
               (e.g., C₁₀: 75 isomers → fast; C₂₀: 366,319 → minutes).
            
            | Carbon Range | Est. Time | Isomers |
            |--------------|-----------|----------|
            | 1-10 | < 1 second | ≤75 |
            | 11-12 | ~1-2 seconds | ≤355 |
            | 13-14 | ~5-15 seconds | ≤1,858 |
            | 15-16 | ~30s - 1 min | ≤10,359 |
            | 17-18 | ~2-5 minutes | ≤60,523 |
            | 19-20 | ~10-30 minutes | ≤366,319 |
            
            **Note:** Actual times vary based on your CPU speed, memory, and whether 
            results are cached from previous runs.
            
            See the **Theory & Background** page for more details on the algorithm.
            """)


# ============================================================================
# Page: The Explosion Plot
# ============================================================================

elif page == "💥 The Explosion Plot":
    st.markdown('<h1 class="main-header">💥 The Explosion Plot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Witness the combinatorial explosion of chemical space</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Why this matters:</strong> The number of possible molecules grows exponentially 
    with size. This is why <strong>computational methods</strong> are essential for drug discovery - 
    we can't synthesize and test all possibilities!
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive explosion demonstration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Enter a Carbon Count")
        
        user_n = st.number_input(
            "How many carbons?",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="Try different values to see how the isomer count explodes!"
        )
        
        # Known counts
        known_counts = {
            1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75,
            11: 159, 12: 355, 13: 802, 14: 1858, 15: 4347, 16: 10359,
            17: 24894, 18: 60523, 19: 148284, 20: 366319, 21: 910726,
            22: 2278658, 23: 5731580, 24: 14490245, 25: 36797588,
            26: 93839412, 27: 240215803, 28: 617105614, 29: 1590507121,
            30: 4111846763
        }
        
        count = known_counts.get(user_n, "?")
        
        if isinstance(count, int):
            st.markdown(f'<p class="explosion-text">{format_number(count)}</p>', unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>structural isomers for C{user_n}H{2*user_n+2}</p>", unsafe_allow_html=True)
            
            # Fun facts - only show when there's at least one comparison to make
            if count > 1000000:
                st.markdown("""
                <div class="warning-box">
                ⚠️ <strong>That's more isomers than...</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"- The population of a small city")
                if count > 100000000:
                    st.markdown(f"- The number of stars visible from Earth")
                if count > 1000000000:
                    st.markdown(f"- The population of China!")
    
    with col2:
        st.markdown("### The Explosion Curve")
        
        max_n = st.slider("Plot up to n =", min_value=10, max_value=30, value=20)
        
        fig = plot_isomer_count_curve(max_n=max_n, log_scale=True)
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("---")
    
    # Comparison table
    st.markdown("### How Fast Does It Grow?")
    
    comparison_data = []
    prev_count = 1
    for n in range(1, 21):
        count = known_counts[n]
        ratio = count / prev_count if prev_count > 0 else 0
        comparison_data.append({
            "n": n,
            "Formula": f"C{n}H{2*n+2}",
            "Isomers": format_number(count),
            "Growth Factor": f"{ratio:.2f}x" if n > 1 else "-"
        })
        prev_count = count
    
    st.dataframe(comparison_data, width='stretch')
    
    st.markdown("""
    **Key Insight:** The growth factor approaches ~2.5x for each additional carbon!
    This means the number of isomers roughly follows $2.5^n$ (exponential growth).
    """)
    
    # Informational dropdowns
    with st.expander("What does this visualization show? (Click to expand)"):
        st.markdown("""
        The **Explosion Curve** visualization demonstrates the **combinatorial explosion** 
        of alkane structural isomers as the number of carbon atoms increases.
        
        **Key Features:**
        
        - **X-axis (Number of Carbons):** The carbon count from 1 to your selected maximum (up to 30).
        
        - **Y-axis (Number of Isomers):** The count of structurally distinct alkane isomers 
          for each carbon number. Note the **logarithmic scale** — each major gridline represents 
          a 10× increase!
        
        - **Blue Curve:** Shows how isomer count grows with carbon number. The steep upward 
          curve illustrates exponential growth.
        
        - **Annotation Boxes:** Key milestone counts are labeled (C₅, C₁₀, C₁₅, C₂₀, C₂₅, C₃₀) 
          to help visualize the magnitude of growth.
        
        - **Growth Rate Formula:** The asymptotic growth rate of ~2.5ⁿ means each additional 
          carbon roughly multiplies the isomer count by 2.5.
        
        **Why "Explosion"?**
        
        The term "combinatorial explosion" describes how quickly possibilities multiply:
        - C₁₀: 75 isomers (manageable)
        - C₂₀: 366,319 isomers (a small city's population)
        - C₃₀: 4.1 billion isomers (more than Earth's population!)
        
        This demonstrates why **computational methods** are essential in chemistry — 
        we cannot possibly synthesize and test all molecular possibilities by hand.
        """)
    
    with st.expander("What does this table show? (Click to expand)"):
        st.markdown("""
        The **How Fast Does It Grow?** table provides a numerical breakdown of isomer 
        counts and their growth rate for alkanes from C₁ to C₂₀.
        
        **Column Definitions:**
        
        - **n:** The number of carbon atoms in the alkane.
        
        - **Formula:** The molecular formula CₙH₂ₙ₊₂ for the alkane. All isomers of 
          a given alkane share the same formula but have different structures.
        
        - **Isomers:** The total count of structurally distinct isomers for that carbon 
          number. These values come from OEIS sequence A000602, which has been mathematically 
          proven and computationally verified.
        
        - **Growth Factor:** How many times larger the isomer count is compared to the 
          previous carbon number (isomersₙ / isomersₙ₋₁). This factor:
          - Varies for small n (due to limited structural possibilities)
          - Stabilizes around **2.4-2.5×** for larger n
          - Reflects the asymptotic growth rate α ≈ 2.4833
        
        **Interpreting the Growth:**
        
        The stabilizing growth factor (~2.5×) means:
        - Adding 1 carbon: ~2.5× more isomers
        - Adding 5 carbons: ~100× more isomers (2.5⁵ ≈ 98)
        - Adding 10 carbons: ~10,000× more isomers (2.5¹⁰ ≈ 9,536)
        
        This exponential relationship is why molecular enumeration becomes computationally 
        intractable for large molecules, driving the need for sampling methods and 
        machine learning in modern drug discovery.
        """)


# ============================================================================
# Page: Structure Explorer
# ============================================================================

elif page == "🔬 Structure Explorer":
    st.markdown('<h1 class="main-header">🔬 Structure Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Click on isomers to explore their Lewis structures</p>', unsafe_allow_html=True)
    
    # Known isomer counts for display
    known_counts = {
        1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75,
        11: 159, 12: 355, 13: 802, 14: 1858, 15: 4347, 16: 10359,
        17: 24894, 18: 60523, 19: 148284, 20: 366319, 21: 910726,
        22: 2278658, 23: 5731580, 24: 14490245, 25: 36797588,
        26: 93839412, 27: 240215803, 28: 617105614, 29: 1590507121,
        30: 4111846763
    }
    
    # Estimated generation times (same as page 1)
    def estimate_generation_time(n: int) -> str:
        """Estimate generation time based on carbon count."""
        if n <= 10:
            return "< 1 second"
        elif n <= 12:
            return "~1-2 seconds"
        elif n <= 14:
            return "~5-15 seconds"
        elif n <= 16:
            return "~30 seconds - 1 minute"
        elif n <= 18:
            return "~2-5 minutes"
        elif n <= 20:
            return "~10-30 minutes"
        elif n <= 22:
            return "~1-3 hours"
        elif n <= 24:
            return "~6-12 hours"
        elif n <= 26:
            return "~1-3 days"
        else:
            return "Several days or more"
    
    # Initialize session state for explorer page
    if 'explorer_n' not in st.session_state:
        st.session_state.explorer_n = 6
    if 'explorer_isomers' not in st.session_state:
        st.session_state.explorer_isomers = None
    if 'explorer_last_n' not in st.session_state:
        st.session_state.explorer_last_n = None
    
    # Carbon selector with info
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        n_carbons = st.slider(
            "Number of Carbons",
            min_value=1,
            max_value=30,
            value=st.session_state.explorer_n,
            key="explorer_n_slider",
            help="Select the number of carbon atoms (1-30)"
        )
        st.session_state.explorer_n = n_carbons
        
        alkane_name = get_alkane_name(n_carbons)
        formula = f"C{n_carbons}H{2*n_carbons+2}"
        
        st.markdown(f"### {alkane_name.capitalize()} ({formula})")
        
        # Show expected isomers and estimated time
        expected_isomers = known_counts.get(n_carbons, "Unknown")
        estimated_time = estimate_generation_time(n_carbons)
        
        st.markdown(f"""
        **Total structural isomers:** {expected_isomers:,} | **Estimated time:** {estimated_time}
        """)
        
        # Warning for large n
        if n_carbons >= 18:
            st.warning(f"⚠️ **Warning:** Generating isomers for C{n_carbons} may take {estimated_time}. Consider starting with a smaller value.")
        elif n_carbons >= 14:
            st.info(f"ℹ️ Generation for C{n_carbons} may take {estimated_time}.")
        
        # Generate button
        generate_clicked = st.button("🧪 Generate Isomers", type="primary", use_container_width=True, key="explorer_generate")
    
    # Generate isomers only when button is clicked or if we have cached results
    if generate_clicked or (st.session_state.explorer_last_n == n_carbons and st.session_state.explorer_isomers is not None):
        if generate_clicked or st.session_state.explorer_last_n != n_carbons:
            # Generate new isomers
            with st.spinner(f"Generating isomers for {formula}... This may take {estimated_time}."):
                start_time = time.time()
                isomers = st.session_state.generator.generate(n_carbons)
                elapsed = time.time() - start_time
            
            st.session_state.explorer_isomers = isomers
            st.session_state.explorer_last_n = n_carbons
            st.session_state.explorer_time = elapsed
        else:
            # Use cached isomers
            isomers = st.session_state.explorer_isomers
            elapsed = st.session_state.explorer_time
        
        st.markdown(f"### Select an isomer of {formula}")
        
        # Handle large isomer lists - limit dropdown to reasonable size
        MAX_DROPDOWN_ITEMS = 500
        isomer_names = [iso.name for iso in isomers]
        num_isomers = len(isomers)
        
        if num_isomers > MAX_DROPDOWN_ITEMS:
            st.warning(f"⚠️ **{num_isomers:,} isomers available.** Showing first {MAX_DROPDOWN_ITEMS} in dropdown. For large isomer sets, consider using the Isomer Generator page to view a grid of structures.")
            display_isomers = isomers[:MAX_DROPDOWN_ITEMS]
            display_names = isomer_names[:MAX_DROPDOWN_ITEMS]
        else:
            display_isomers = isomers
            display_names = isomer_names
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_name = st.selectbox(
                "Choose isomer:",
                display_names,
                help=f"Select an isomer to see its detailed structure{' (showing first ' + str(MAX_DROPDOWN_ITEMS) + ')' if num_isomers > MAX_DROPDOWN_ITEMS else ''}"
            )
            
            selected_idx = display_names.index(selected_name)
            selected = display_isomers[selected_idx]
            
            # Get chemical properties for this isomer
            props = get_alkane_properties(
                name=selected.name,
                n_carbons=n_carbons,
                n_branch_points=selected.n_branch_points,
                max_chain_length=selected.max_chain_length
            )
            
            # Layout selector (above properties)
            st.markdown("### Visualization Options")
            layout_type = st.selectbox(
                "Layout Algorithm",
                ["tree", "spring", "radial"],
                key="explorer_layout",
                help="Choose how the molecular structure is arranged"
            )
            
            st.markdown("### Structural Properties")
            st.markdown(f"**Name:** {selected.name}")
            st.markdown(f"**Formula:** {formula}")
            st.markdown(f"**Molar Mass:** {props.molar_mass:.2f} g/mol")
            st.markdown(f"**Longest Chain:** {selected.max_chain_length} carbons")
            st.markdown(f"**Branch Points:** {selected.n_branch_points}")
            st.markdown(f"**Terminal CH₃:** {selected.n_methyl_groups}")
            
            tree = TreeGraph(n_nodes=n_carbons, adjacency=selected.adjacency)
            st.markdown(f"**Wiener Index:** {compute_wiener_index(tree)}")
            st.markdown(f"**Diameter:** {compute_diameter(tree)}")
            
            st.markdown("### Physical Properties (STP)")
            state_emoji = format_state_emoji(props.state_of_matter)
            st.markdown(f"**State at 25°C:** {state_emoji} {props.state_of_matter.capitalize()}")
            st.markdown(f"**Boiling Point:** {format_temperature(props.boiling_point)}")
            st.markdown(f"**Melting Point:** {format_temperature(props.melting_point)}")
            st.markdown(f"**Density:** {props.density:.3f} g/cm³")
            st.markdown(f"**Water Solubility:** {props.water_solubility}")
            
            st.markdown("### Thermodynamic Properties")
            st.markdown(f"**ΔH°f:** {props.heat_of_formation:.1f} kJ/mol")
            st.markdown(f"**ΔH°c:** {props.heat_of_combustion:,.1f} kJ/mol")
            
            # Show generation stats
            if elapsed < 1:
                time_str = f"{elapsed*1000:.1f}ms"
            elif elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                time_str = f"{elapsed/60:.1f}min"
            st.markdown(f"**Generation Time:** {time_str}")
        
        with col2:
            # Show both views
            tab1, tab2 = st.tabs(["🔲 Carbon Skeleton", "⚗️ Lewis Structure"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(8, 6))
                draw_carbon_skeleton(
                    selected.adjacency, ax=ax,
                    layout=layout_type, show_labels=True,
                    title=f"{selected.name} - Carbon Skeleton"
                )
                st.pyplot(fig)
                plt.close(fig)
            
            with tab2:
                fig, ax = plt.subplots(figsize=(10, 8))
                draw_lewis_structure(
                    selected.adjacency, ax=ax,
                    layout=layout_type, show_hydrogens=True,
                    title=f"{selected.name} - Lewis Structure"
                )
                st.pyplot(fig)
                plt.close(fig)
        
        # Explanation - inside the conditional block since it uses 'selected'
        st.markdown("---")
        
        with st.expander("📖 Understanding the Structure (Click to expand)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **Carbon Types:**
                - **Terminal (CH₃):** Degree 1
                - **Linear (CH₂):** Degree 2
                - **Branch (CH):** Degree 3
                - **Quaternary (C):** Degree 4
                """)
            
            with col2:
                degree_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                for neighbors in selected.adjacency.values():
                    degree_counts[len(neighbors)] += 1
                
                st.markdown("**In this molecule:**")
                st.markdown(f"- CH₃ groups: {degree_counts[1]}")
                st.markdown(f"- CH₂ groups: {degree_counts[2]}")
                st.markdown(f"- CH groups: {degree_counts[3]}")
                st.markdown(f"- C (quaternary): {degree_counts[4]}")
            
            with col3:
                st.markdown("""
                **Nomenclature:**
                - Find the longest chain
                - Number from end nearest to branch
                - Name branches as prefixes
                - Use multipliers (di-, tri-)
                """)
        
        # Informational dropdown about the tool
        with st.expander("What is the Structure Explorer? (Click to expand)"):
            st.markdown("""
            The **Structure Explorer** allows you to examine individual alkane isomers in detail. 
            Unlike the Isomer Generator which shows all structures at once, this tool focuses on 
            one molecule at a time, providing:
            
            **Features:**
            - **Structural properties:** Chain length, branching, and topological indices
            - **Physical properties:** State of matter, boiling/melting points, density
            - **Thermodynamic data:** Enthalpies of formation and combustion
            - **Carbon Skeleton view:** Simplified C-C bond representation
            - **Lewis Structure view:** Full representation with all hydrogen atoms
            - **Degree analysis:** Distribution of carbon types (CH₃, CH₂, CH, C)
            
            **Tips:**
            - For large isomer sets (>500), only the first 500 are shown in the dropdown
            - Use the Isomer Generator page for grid views of many structures
            - The Wiener Index and Diameter are topological descriptors used in QSPR studies
            """)
        
        # Informational dropdown about property estimation
        with st.expander("About the Property Estimates (Click to expand)"):
            st.markdown("""
            The physical and thermodynamic properties shown are **estimates** based on 
            empirical correlations and reference data for n-alkanes, adjusted for structural 
            features like branching.
            
            **Standard State Conditions:**
            - Temperature: 298.15 K (25°C, 77°F)
            - Pressure: 100 kPa (1 bar, ~1 atm)
            
            **Property Sources & Methods:**
            
            - **State of Matter:** C₁-C₄ are gases, C₅-C₁₇ are liquids, C₁₈+ are waxy solids 
              at room temperature
            
            - **Boiling Point:** Based on NIST data for n-alkanes. Branching typically 
              *lowers* boiling point (reduced surface area → weaker van der Waals forces)
            
            - **Melting Point:** Based on NIST data. Branching generally *lowers* melting 
              point (less efficient crystal packing), except for highly symmetric molecules
            
            - **Density:** From CRC Handbook values. Branching slightly reduces density
            
            - **Heat of Formation (ΔH°f):** Estimated using group additivity methods. 
              Branching *stabilizes* the molecule (more negative ΔH°f) due to hyperconjugation
            
            - **Heat of Combustion (ΔH°c):** Based on the methylene increment method 
              (~659 kJ/mol per CH₂ group). All isomers with the same formula have 
              approximately equal combustion enthalpies
            
            **References:**
            - NIST Chemistry WebBook (https://webbook.nist.gov/)
            - CRC Handbook of Chemistry and Physics
            - Yaws' Handbook of Thermodynamic Properties
            
            *Note: These are estimates for educational purposes. For precise values, 
            consult experimental databases or perform quantum chemical calculations.*
            """)
    else:
        # Show instruction when no isomers generated yet
        st.info("👆 Select a carbon count and click **Generate Isomers** to explore individual structures.")


# ============================================================================
# Page: Analysis Dashboard
# ============================================================================

elif page == "📊 Analysis Dashboard":
    st.markdown('<h1 class="main-header">📊 Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Statistical analysis of alkane isomers</p>', unsafe_allow_html=True)
    
    # Known isomer counts for display
    known_counts = {
        1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 5, 7: 9, 8: 18, 9: 35, 10: 75,
        11: 159, 12: 355, 13: 802, 14: 1858, 15: 4347, 16: 10359,
        17: 24894, 18: 60523, 19: 148284, 20: 366319, 21: 910726,
        22: 2278658, 23: 5731580, 24: 14490245, 25: 36797588,
        26: 93839412, 27: 240215803, 28: 617105614, 29: 1590507121,
        30: 4111846763
    }
    
    # Estimated generation times
    def estimate_generation_time(n: int) -> str:
        """Estimate generation time based on carbon count."""
        if n <= 10:
            return "< 1 second"
        elif n <= 12:
            return "~1-2 seconds"
        elif n <= 14:
            return "~5-15 seconds"
        elif n <= 16:
            return "~30 seconds - 1 minute"
        elif n <= 18:
            return "~2-5 minutes"
        elif n <= 20:
            return "~10-30 minutes"
        elif n <= 22:
            return "~1-3 hours"
        elif n <= 24:
            return "~6-12 hours"
        elif n <= 26:
            return "~1-3 days"
        else:
            return "Several days or more"
    
    # Initialize session state for analysis page
    if 'analysis_n' not in st.session_state:
        st.session_state.analysis_n = 8
    if 'analysis_isomers' not in st.session_state:
        st.session_state.analysis_isomers = None
    if 'analysis_last_n' not in st.session_state:
        st.session_state.analysis_last_n = None
    
    # Carbon count selector
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        n_carbons = st.slider(
            "Number of Carbons to Analyze",
            min_value=4,
            max_value=30,
            value=st.session_state.analysis_n,
            key="analysis_n_slider",
            help="Select the number of carbon atoms (4-30)"
        )
        st.session_state.analysis_n = n_carbons
        
        alkane_name = get_alkane_name(n_carbons)
        formula = f"C{n_carbons}H{2*n_carbons+2}"
        
        st.markdown(f"### {alkane_name.capitalize()} ({formula})")
        
        # Show expected isomers and estimated time
        expected_isomers = known_counts.get(n_carbons, "Unknown")
        estimated_time = estimate_generation_time(n_carbons)
        
        st.markdown(f"""
        **Expected isomers:** {expected_isomers:,} | **Estimated time:** {estimated_time}
        """)
        
        # Warning for large n
        if n_carbons >= 18:
            st.warning(f"⚠️ **Warning:** Generating isomers for C{n_carbons} may take {estimated_time}. Consider starting with a smaller value.")
        elif n_carbons >= 14:
            st.info(f"ℹ️ Generation for C{n_carbons} may take {estimated_time}.")
        
        # Generate button
        generate_clicked = st.button("📊 Generate & Analyze", type="primary", use_container_width=True, key="analysis_generate")
    
    # Generate isomers only when button is clicked or if we have cached results
    if generate_clicked or (st.session_state.analysis_last_n == n_carbons and st.session_state.analysis_isomers is not None):
        if generate_clicked or st.session_state.analysis_last_n != n_carbons:
            # Generate new isomers
            with st.spinner(f"Generating isomers for {formula}... This may take {estimated_time}."):
                start_time = time.time()
                isomers = st.session_state.generator.generate(n_carbons)
                elapsed = time.time() - start_time
            
            st.session_state.analysis_isomers = isomers
            st.session_state.analysis_last_n = n_carbons
            st.session_state.analysis_time = elapsed
        else:
            # Use cached isomers
            isomers = st.session_state.analysis_isomers
            elapsed = st.session_state.analysis_time
        
        # Display generation time
        if elapsed < 1:
            time_str = f"{elapsed*1000:.1f}ms"
        elif elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            time_str = f"{elapsed/60:.1f}min"
        
        st.markdown(f"### Analyzing {len(isomers):,} isomers of {formula}")
        st.caption(f"Generated in {time_str}")
        
        # Branching analysis
        st.markdown("### Branching Characteristics")
        
        fig = plot_branching_analysis(isomers)
        st.pyplot(fig)
        plt.close(fig)
        
        # Degree distribution
        st.markdown("### Carbon Atom Types")
        
        fig = plot_degree_distribution(isomers)
        st.pyplot(fig)
        plt.close(fig)
        
        # Informational dropdown for visualizations
        with st.expander("What do these visualizations show? (Click to expand)"):
            st.markdown("""
            **Branching Characteristics:**
            
            This visualization shows the distribution of structural features across all isomers:
            
            - **Chain Length Distribution:** How the length of the longest carbon chain varies 
              across isomers. Shorter chains indicate more compact, highly branched structures.
            
            - **Branch Point Distribution:** The number of tertiary (degree 3) and quaternary 
              (degree 4) carbon atoms. More branch points = more compact structure.
            
            - **Methyl Group Distribution:** The count of terminal CH₃ groups. Linear alkanes 
              have exactly 2 methyl groups; branched alkanes have more.
            
            **Carbon Atom Types:**
            
            This chart shows the average composition of carbon atom types across all isomers:
            
            - **Terminal (CH₃):** Degree 1 - end-of-chain carbons bonded to 3 hydrogens
            - **Linear (CH₂):** Degree 2 - mid-chain carbons bonded to 2 hydrogens
            - **Branch (CH):** Degree 3 - branch point carbons bonded to 1 hydrogen
            - **Quaternary (C):** Degree 4 - fully substituted carbons with no hydrogens
            
            The degree distribution reveals how "branchy" the typical isomer is for this carbon count.
            """)
        
        # Statistics
        st.markdown("### Summary Statistics")
        
        chain_lengths = [iso.max_chain_length for iso in isomers]
        branch_points = [iso.n_branch_points for iso in isomers]
        methyl_groups = [iso.n_methyl_groups for iso in isomers]
        
        wiener_indices = []
        for iso in isomers:
            tree = TreeGraph(n_nodes=n_carbons, adjacency=iso.adjacency)
            wiener_indices.append(compute_wiener_index(tree))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Chain Length", f"{np.mean(chain_lengths):.2f}")
            st.metric("Max Chain Length", max(chain_lengths))
        
        with col2:
            st.metric("Avg Branch Points", f"{np.mean(branch_points):.2f}")
            st.metric("Max Branch Points", max(branch_points))
        
        with col3:
            st.metric("Avg CH₃ Groups", f"{np.mean(methyl_groups):.2f}")
            st.metric("Max CH₃ Groups", max(methyl_groups))
        
        with col4:
            st.metric("Avg Wiener Index", f"{np.mean(wiener_indices):.1f}")
            st.metric("Wiener Range", f"{min(wiener_indices)}-{max(wiener_indices)}")
        
        # Informational dropdown for statistics
        with st.expander("What do these statistics mean? (Click to expand)"):
            st.markdown(f"""
            **Summary Statistics Explained:**
            
            These metrics summarize the structural diversity of all {len(isomers):,} isomers of {formula}:
            
            ---
            
            **Chain Length Statistics:**
            
            - **Average Chain Length ({np.mean(chain_lengths):.2f}):** The mean length of the longest 
              continuous carbon chain across all isomers. Lower values indicate that most isomers 
              are branched rather than linear.
            
            - **Max Chain Length ({max(chain_lengths)}):** The longest possible chain, which equals n 
              for the linear (unbranched) isomer. This is always n for alkanes with n carbons.
            
            ---
            
            **Branch Point Statistics:**
            
            - **Average Branch Points ({np.mean(branch_points):.2f}):** The mean number of tertiary 
              (CH) and quaternary (C) carbons across all isomers. Higher values indicate more 
              complex branching patterns are common.
            
            - **Max Branch Points ({max(branch_points)}):** The most branch points possible for this 
              carbon count. This occurs in the most compact isomers.
            
            ---
            
            **Methyl Group Statistics:**
            
            - **Average CH₃ Groups ({np.mean(methyl_groups):.2f}):** The mean number of terminal 
              methyl groups. Linear alkanes have exactly 2; highly branched structures have more.
            
            - **Max CH₃ Groups ({max(methyl_groups)}):** The maximum number of terminal methyls, 
              found in the most highly branched isomers (e.g., neopentane-like structures).
            
            ---
            
            **Wiener Index Statistics:**
            
            The **Wiener Index** is a topological descriptor equal to the sum of all shortest-path 
            distances between pairs of atoms. It correlates with physical properties like boiling 
            point and surface area.
            
            - **Average Wiener Index ({np.mean(wiener_indices):.1f}):** The mean topological 
              compactness across all isomers.
            
            - **Wiener Range ({min(wiener_indices)}-{max(wiener_indices)}):** 
              - **Minimum ({min(wiener_indices)}):** The most compact isomer (most branched)
              - **Maximum ({max(wiener_indices)}):** The most extended isomer (linear n-alkane)
            
            *Lower Wiener index → more compact molecule → lower boiling point*
            
            *Higher Wiener index → more extended molecule → higher boiling point*
            """)
    else:
        # Show instruction when no isomers generated yet
        st.info("👆 Select a carbon count and click **Generate & Analyze** to see statistical analysis of all isomers.")


# ============================================================================
# Page: Theory & Background
# ============================================================================

elif page == "📚 Theory & Background":
    st.markdown('<h1 class="main-header">📚 Theory & Background</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">The mathematics of molecular enumeration</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graph Theory
    st.markdown("## 1. Alkanes as Trees")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        An **alkane** is a saturated hydrocarbon with formula **CₙH₂ₙ₊₂**. 
        The carbon backbone of an alkane forms a **tree** (connected, acyclic graph):
        
        - Each carbon atom is a **node**
        - Each C-C bond is an **edge**
        - Carbon has valency 4, so max degree ≤ 4
        - No cycles (saturated = no double bonds)
        
        Two alkanes are **structural isomers** if they have the same molecular 
        formula but different connectivity (different trees).
        """)
    
    with col2:
        # Show example
        adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3]}
        fig, ax = plt.subplots(figsize=(5, 3))
        draw_carbon_skeleton(adj, ax=ax, title="n-Pentane as a Tree")
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("---")
    
    # Isomorphism
    st.markdown("## 2. Tree Isomorphism")
    
    st.markdown("""
    Two trees are **isomorphic** if one can be transformed into the other by 
    relabeling nodes. For molecules, this means the same structure regardless 
    of how we number the atoms.
    
    **The Key Challenge:** Detect duplicates efficiently!
    
    **Solution:** Compute a **canonical form** for each tree:
    1. Find the **center** of the tree (minimizes max distance to any leaf)
    2. Root the tree at the center
    3. Assign labels based on **subtree signatures** (recursive structure)
    4. Isomorphic trees get identical canonical forms
    
    This allows **O(n log n)** isomorphism checking instead of expensive comparison.
    """)
    
    st.latex(r"\text{Signature}(T) = \begin{cases} () & \text{if } T \text{ is a leaf} \\ (\text{sorted children signatures}) & \text{otherwise} \end{cases}")
    
    st.markdown("---")
    
    # Cayley's Theorem
    st.markdown("## 3. Cayley's Tree Formula")
    
    st.markdown("""
    **Cayley's Theorem** (1889) states that the number of labeled trees on n vertices is:
    """)
    
    st.latex(r"T_n = n^{n-2}")
    
    st.markdown("""
    For example:
    - n=3: 3¹ = 3 labeled trees
    - n=4: 4² = 16 labeled trees
    - n=5: 5³ = 125 labeled trees
    
    However, we want **unlabeled** trees (up to isomorphism), which is much harder 
    to count and requires **Pólya enumeration theory**.
    """)
    
    st.markdown("---")
    
    # Pólya Enumeration
    st.markdown("## 4. Pólya Enumeration (Burnside's Lemma)")
    
    st.markdown("""
    To count distinct structures, we use **Burnside's Lemma**:
    """)
    
    st.latex(r"|X/G| = \frac{1}{|G|} \sum_{g \in G} |X^g|")
    
    st.markdown("""
    Where:
    - **X** = set of labeled structures
    - **G** = symmetry group (permutations)
    - **Xᵍ** = structures fixed by permutation g
    
    For alkanes specifically, the generating function approach gives the recurrence 
    (OEIS A000602):
    """)
    
    st.latex(r"a_n \sim \frac{c \cdot \alpha^n}{n^{5/2}} \quad \text{where } \alpha \approx 2.4833")
    
    st.markdown("""
    This explains the approximately **2.5× growth** per carbon!
    """)
    
    st.markdown("---")
    
    # Chemical Space
    st.markdown("## 5. Chemical Space & Drug Discovery")
    
    st.markdown("""
    The combinatorial explosion of isomers demonstrates the vastness of **Chemical Space**:
    
    | Carbon Count | Isomers | Context |
    |--------------|---------|---------|
    | 10 | 75 | Manageable |
    | 15 | 4,347 | Large but tractable |
    | 20 | 366,319 | Computationally intensive |
    | 30 | 4.1 billion | Astronomical |
    
    **Implications for Drug Discovery:**
    - Cannot synthesize all possibilities
    - Need computational screening
    - Machine learning for property prediction
    - Focused libraries based on pharmacophores
    
    This is why **computational chemistry** and **cheminformatics** are essential 
    tools in modern drug discovery pipelines.
    """)
    
    st.markdown("---")
    
    # Generation Time Estimation
    st.markdown("## 6. Generation Time Estimation")
    
    st.markdown("""
    The generation time estimates shown in this app are **rough empirical approximations** 
    rather than precise measurements. Understanding why generation takes longer for larger 
    molecules helps illustrate the computational challenges of chemical enumeration.
    """)
    
    st.markdown("### The Algorithm")
    
    st.markdown("""
    The isomer generator uses a **recursive tree generation** algorithm:
    
    1. **Generate candidate trees** by systematically adding edges
    2. **Compute canonical forms** for each tree (using tree center + subtree signatures)
    3. **Filter duplicates** by comparing canonical hashes
    4. **Store unique isomers** with their structural properties
    
    The complexity is roughly **O(n × isomer_count)**, where each isomer requires 
    canonical form computation in O(n log n) time.
    """)
    
    st.markdown("### Why Time Grows Exponentially")
    
    st.markdown("""
    The number of alkane isomers follows the asymptotic formula:
    """)
    
    st.latex(r"a_n \sim \frac{c \cdot \alpha^n}{n^{5/2}} \quad \text{where } \alpha \approx 2.4833")
    
    st.markdown("""
    This means:
    - Each additional carbon multiplies the isomer count by ~2.5×
    - Going from C₁₀ (75 isomers) to C₂₀ (366,319 isomers) is a ~4,900× increase
    - Generation time scales similarly
    """)
    
    st.markdown("### Estimation Table")
    
    st.markdown("""
    The time estimates used in this app are based on the following calibration:
    
    | Carbon Count | Isomers | Estimated Time | Rationale |
    |--------------|---------|----------------|------------|
    | 1-10 | 1-75 | < 1 second | Very fast, negligible overhead |
    | 11-12 | 159-355 | ~1-2 seconds | Still quick, minimal computation |
    | 13-14 | 802-1,858 | ~5-15 seconds | Noticeable but manageable |
    | 15-16 | 4,347-10,359 | ~30s - 1 minute | Requires patience |
    | 17-18 | 24,894-60,523 | ~2-5 minutes | Significant computation |
    | 19-20 | 148,284-366,319 | ~10-30 minutes | Coffee break time |
    | 21-22 | 910,726-2,278,658 | ~1-3 hours | Background task |
    | 23-24 | 5,731,580-14,490,245 | ~6-12 hours | Overnight run |
    | 25-26 | 36,797,588-93,839,412 | ~1-3 days | Multi-day computation |
    | 27-30 | 240M - 4.1B | Several days+ | Requires dedicated computing |
    """)
    
    st.markdown("### Important Caveats")
    
    st.markdown("""
    These estimates are **approximate** and actual times depend on:
    
    - **CPU Speed**: Faster processors complete generation more quickly
    - **Caching**: The generator caches results, so repeated queries are instant
    - **Memory**: Large isomer sets require substantial RAM
    - **Python Overhead**: A compiled implementation would be significantly faster
    - **Algorithm Optimizations**: More sophisticated pruning could reduce time
    
    For production use with large molecules, consider:
    - Pre-computing and storing results in a database
    - Using parallel/distributed computation
    - Implementing in a compiled language (C++, Rust)
    - Using approximate/sampling methods instead of exhaustive enumeration
    """)
    
    st.markdown("---")
    
    # References
    st.markdown("## References")
    
    st.markdown("""
    1. Cayley, A. (1889). "A theorem on trees." *Quart. J. Math.* 23:376-378.
    2. Pólya, G. (1937). "Kombinatorische Anzahlbestimmungen für Gruppen, Graphen und chemische Verbindungen." *Acta Math.* 68:145-254.
    3. OEIS Foundation. Sequence A000602: Number of n-node unrooted unlabeled trees with maximum degree 4.
    4. Faulon, J.L. & Bender, A. (2010). *Handbook of Chemoinformatics Algorithms*. CRC Press.
    """)


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p><strong>Week 4 Project 1: The Alkane Assembler</strong></p>
    <p>Biophysics Portfolio • Ryan Kamp • University of Cincinnati</p>
    <p>February 05, 2026</p>
</div>
""", unsafe_allow_html=True)
