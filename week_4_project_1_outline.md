# Week 4 - Project 1: "The Alkane Assembler" – Tree Generation & Isomer Counting

## Overview

**Week:** 4 (Feb 10 – Feb 17)  
**Theme:** Graph Theory, Isomerism, and Combinatorics  
**Goal:** Treat organic molecules as data structures to solve structural problems (isomers, connectivity) algorithmically.

---

## Project Details

### The "Gap" It Fills
Mastery of **Organic Nomenclature**, **Structural Isomerism**, and **Combinatorics**.

One of the first hurdles in O-Chem is realizing that C₅H₁₂ isn't just one molecule; it's three (n-pentane, isopentane, neopentane). As carbon count rises, isomers explode combinatorially. This project solves the "counting problem" using Cayley's Tree Formula and graph generation.

### The Concept
- A generator that accepts an integer N (number of Carbons).
- It generates every mathematically possible **tree graph** with N nodes (where max degree ≤ 4, representing Carbon's valency).
- It filters out duplicates (isomorphism check) and renders the unique carbon backbones.

### Novelty/Creative Angle
**"The Explosion Plot":** Plot N vs. Number of Isomers on a log scale. The user types "10" and sees 75 isomers. They type "15" and the screen floods with 4,347 structures.

It visually demonstrates **Chemical Space** complexity—why we need AI for drug discovery.

### Technical Implementation
- **Language:** Python (NetworkX for graph operations).
- **Algorithm:** Burnside's Lemma or recursive tree generation with isomorphism pruning.

### The "Paper" & Interactive Element
- *Interactive:* Click on a generated graph to "unfold" it into a 2D Lewis structure.
- *Paper Focus:* "Algorithmic Enumeration of Alkane Isomers: A Graph-Theoretic Approach to Chemical Space Exploration."

---

## Progress Tracking

- [ ] Initial research and planning
- [ ] Core implementation
- [ ] Testing and validation
- [ ] Documentation and paper draft
- [ ] Interactive demo creation
