# ⚗️ The Alkane Assembler

<!-- AI-PORTFOLIO-NOTICE:START -->
> **Portfolio note — Spring 2026 AI Research Prototype Portfolio (S26 AIRP)**
>
> This repository is part of the **Spring 2026 AI Research Prototype Portfolio (S26 AIRP)**, a portfolio exploring **AI-assisted research software prototyping** and LLM-assisted scientific software development. The code, interface, documentation, and any accompanying reports were developed with substantial AI assistance as part of an exploratory learning workflow.
>
> The scientific/domain-specific content is provisional and has not been independently validated by domain experts. This repository should be read as a software-engineering, workflow-design, and AI-methodology artifact, not as validated scientific research.
>
> For full context, see [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).
<!-- AI-PORTFOLIO-NOTICE:END -->

**Week 4 - Project 1: Tree Generation & Isomer Counting**

---

| **Author** | Ryan Kamp |
|------------|-------------------|
| **Affiliation** | University of Cincinnati Department of Computer Science |
| **Email** | kamprj@mail.uc.edu |
| **GitHub** | [github.com/ryanjosephkamp](https://github.com/ryanjosephkamp) |
| **Repository** | [alkane-assembler](https://github.com/ryanjosephkamp/alkane-assembler) |
| **Created** | February 5, 2026 |
| **Last Updated** | February 5, 2026 |
| **License** | MIT |

---

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red.svg)
![Tests](https://img.shields.io/badge/Tests-Comprehensive-brightgreen.svg)

## Overview

This project demonstrates the **algorithmic enumeration of alkane structural isomers** using graph-theoretic tree generation with isomorphism pruning. The key insight is treating organic molecules as mathematical graphs, enabling computational approaches to chemistry problems.

### The Gap It Fills

Mastery of **Organic Nomenclature**, **Structural Isomerism**, and **Combinatorics**. One of the first hurdles in Organic Chemistry is realizing that C₅H₁₂ isn't just one molecule—it's three (n-pentane, isopentane, neopentane). As carbon count rises, isomers explode combinatorially.

### The Concept

- A generator that accepts an integer N (number of Carbons)
- Generates every mathematically possible **tree graph** with N nodes (max degree ≤ 4)
- Filters out duplicates via isomorphism checking (canonical forms)
- Renders the unique carbon backbones as 2D structures

### The Novelty: "The Explosion Plot"

Plot N vs. Number of Isomers on a log scale. The user types "10" and sees 75 isomers. They type "15" and the screen floods with 4,347 structures. It visually demonstrates **Chemical Space** complexity—why we need AI for drug discovery.

## 🎯 Key Features

- **🔢 Complete Enumeration**: Generates all structural isomers for any alkane (validated against OEIS A000602)
- **💥 The Explosion Plot**: Interactive visualization of combinatorial growth (~2.5ⁿ) with milestone annotations
- **🔬 Structure Explorer**: Click on graphs to unfold into Lewis structures with explicit hydrogens, plus estimated physical and thermodynamic properties
- **📊 Analysis Dashboard**: Topological indices (Wiener, Balaban), branching statistics with detailed explanations
- **📚 Theory Section**: Graph theory, Pólya enumeration, chemical space concepts
- **⚡ Smart Generation**: Time estimation and warnings for large carbon counts (n≥14)
- **🧪 Property Estimation**: Standard-state physical properties (boiling/melting points, density, state of matter) and thermodynamic data (ΔH°f, ΔH°c)
- **📖 Educational Dropdowns**: Expandable explanations on every page explaining visualizations, statistics, and concepts
- **✅ Validated**: Matches known isomer counts from C₁ through C₃₀

## 🎓 Learning Objectives

This project teaches you to understand:

1. **Structural Isomerism** - Same formula, different connectivity
2. **Graph-Molecule Mapping** - Alkanes as trees (max degree 4)
3. **Tree Isomorphism** - Canonical forms for duplicate detection
4. **Combinatorial Explosion** - Why chemical space is astronomical
5. **Pólya Enumeration** - Mathematical framework for counting
6. **Topological Indices** - Wiener index, molecular descriptors
7. **Drug Discovery Context** - Why computation is essential

## 🏗️ Project Structure

```
week_4_project_1/
├── app.py                 # Interactive Streamlit web application
├── main.py                # Command-line demonstrations
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── docs/
│   ├── scientific_report.md           # Technical documentation
│   └── w4p1_alkane_assembler_ieee.tex # IEEE-formatted paper
├── src/
│   ├── __init__.py           # Package initialization
│   ├── alkane_generator.py   # Core isomer enumeration
│   ├── tree_utils.py         # Graph isomorphism & tree operations
│   ├── visualization.py      # 2D structure rendering
│   └── properties.py         # Chemical property estimation
├── tests/
│   ├── test_alkane_generator.py  # Generator test suite
│   └── test_properties.py        # Property estimation tests
└── output/
    └── (generated figures)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository (or navigate to the project)
cd week_4_project_1

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Command Line Demo

```bash
# Run all demonstrations
python main.py

# Generate specific carbon count
python main.py --generate 6

# Show the explosion plot
python main.py --explosion

# Validate against known counts
python main.py --validate

# Compare isomer properties
python main.py --compare 5

# Save figures to output/
python main.py --save
```

### Interactive Web Application

```bash
# Launch Streamlit app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Isomer Counts (OEIS A000602)

| Carbons | Isomers | Name |
|---------|---------|------|
| 1 | 1 | Methane |
| 2 | 1 | Ethane |
| 3 | 1 | Propane |
| 4 | 2 | Butane |
| 5 | 3 | Pentane |
| 6 | 5 | Hexane |
| 7 | 9 | Heptane |
| 8 | 18 | Octane |
| 9 | 35 | Nonane |
| 10 | 75 | Decane |
| 15 | 4,347 | Pentadecane |
| 20 | 366,319 | Icosane |
| 30 | 4,111,846,763 | Triacontane |

## 💡 Algorithm Overview

### Tree Generation

```
1. Base case: C₁ = single node (methane)
2. Recursive: For each tree with (n-1) carbons:
   - Try adding a new carbon to each node with degree < 4
   - Compute canonical form of new tree
   - Keep if unique (not seen before)
3. Return all unique n-carbon trees
```

### Canonical Form (Isomorphism Detection)

```
1. Find tree center (minimizes max distance to leaves)
2. Root tree at center
3. Compute subtree signatures recursively
4. Relabel nodes based on sorted signatures
5. Isomorphic trees → identical canonical forms
```

## 🔬 Technical Details

### Core Classes

**AlkaneIsomer**
- Stores adjacency list representation
- Computes: longest chain, branch points, CH₃ count
- Includes canonical hash for duplicate detection

**AlkaneGenerator**
- Recursive tree generation with memoization
- Validates against known OEIS counts
- Supports caching for repeated queries

**TreeGraph**
- General tree operations
- Isomorphism checking via canonical forms
- Topological metrics (Wiener, Balaban indices)

### Visualization

- **Carbon Skeleton**: Just the C-C backbone
- **Lewis Structure**: Full structure with explicit hydrogens
- **Layout Options**: Tree (hierarchical), Spring (force-directed), Radial

## 📈 Example Usage

```python
from src.alkane_generator import generate_alkane_isomers, count_alkane_isomers

# Count isomers
print(f"C10 has {count_alkane_isomers(10)} isomers")  # 75

# Generate all pentane isomers
isomers = generate_alkane_isomers(5)
for iso in isomers:
    print(f"{iso.name}: chain={iso.max_chain_length}, branches={iso.n_branch_points}")
```

Output:
```
n-pentane: chain=5, branches=0
isopentane: chain=4, branches=1
neopentane: chain=3, branches=1
```

## 🌐 Interactive Features

### Streamlit App Pages

1. **🔢 Isomer Generator**
   - Slider to select carbon count (1-30)
   - Generate button with time estimation and warnings
   - Grid view of all isomers (adaptive sizing for large sets)
   - Properties table with topological indices
   - Choice of layout algorithms (tree, spring, radial)
   - Informational dropdowns explaining visualizations

2. **💥 The Explosion Plot**
   - Log-scale visualization of isomer count growth
   - Annotations at key milestones (C₁₀, C₁₅, C₂₀, C₂₅, C₃₀)
   - Interactive carbon count input with fun facts
   - Growth rate comparison table
   - Educational dropdowns on exponential growth

3. **🔬 Structure Explorer**
   - Generate button workflow with time estimation
   - Select individual isomers from dropdown (limited to 500 for large sets)
   - Toggle between skeleton and Lewis structure views
   - Layout algorithm selection (tree, spring, radial)
   - **Physical Properties**: State of matter, boiling/melting points, density, water solubility
   - **Thermodynamic Properties**: Enthalpy of formation (ΔH°f), enthalpy of combustion (ΔH°c)
   - **Structural Properties**: Chain length, branch points, Wiener index, diameter
   - Informational dropdowns about properties and the tool

4. **📊 Analysis Dashboard**
   - Generate button with time estimation
   - Branching distribution histograms
   - Carbon type analysis (CH₃, CH₂, CH, C)
   - Statistical summary with detailed explanations
   - Informational dropdowns explaining all metrics

5. **📚 Theory & Background**
   - Graph theory fundamentals (alkanes as trees)
   - Tree isomorphism and canonical forms
   - Cayley's tree formula
   - Pólya enumeration explanation (Burnside's lemma)
   - Chemical space discussion
   - Generation time estimation methodology

### Sidebar Features

- **Key Concepts Dropdowns**: Expandable definitions for structural isomer, carbon skeleton, Lewis structure, Wiener index, and chemical space
- **Page Navigation**: Easy switching between all five pages

## 📚 References

1. Cayley, A. (1889). "A theorem on trees." *Quart. J. Math.* 23:376-378.
2. Pólya, G. (1937). "Kombinatorische Anzahlbestimmungen für Gruppen, Graphen und chemische Verbindungen." *Acta Math.* 68:145-254.
3. OEIS Foundation Inc. Sequence A000602: Number of n-node unrooted unlabeled trees with maximum degree 4.
4. Wiener, H. (1947). "Structural determination of paraffin boiling points." *J. Am. Chem. Soc.* 69:17-20.
5. Faulon, J.L. & Bender, A. (2010). *Handbook of Chemoinformatics Algorithms*. CRC Press.

## 📄 License

MIT License - See LICENSE file for details.

---

<p align="center">
  <strong>⚗️ The Alkane Assembler</strong><br>
  Week 4 Project 1 | Biophysics Portfolio<br>
  University of Cincinnati | February 2026
</p>
