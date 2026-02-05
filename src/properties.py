"""
Chemical property estimation for alkane isomers.

This module provides functions to estimate standard-state thermodynamic
and physical properties of alkane structural isomers based on their
molecular formula and structural characteristics.

Standard State Conditions:
- Temperature: 298.15 K (25°C, 77°F)
- Pressure: 100 kPa (1 bar)

References:
- NIST Chemistry WebBook (https://webbook.nist.gov/)
- CRC Handbook of Chemistry and Physics
- Yaws' Handbook of Thermodynamic Properties
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Atomic masses (IUPAC 2021)
ATOMIC_MASS_C = 12.011  # g/mol
ATOMIC_MASS_H = 1.008   # g/mol


@dataclass
class AlkaneProperties:
    """Container for estimated alkane properties at standard state."""
    
    # Basic properties
    name: str
    formula: str
    n_carbons: int
    molar_mass: float  # g/mol
    
    # Physical state at STP
    state_of_matter: str  # "gas", "liquid", or "solid"
    
    # Thermodynamic properties (estimated)
    boiling_point: Optional[float]  # °C
    melting_point: Optional[float]  # °C
    density: Optional[float]  # g/cm³ (liquid at 20°C or solid)
    
    # Combustion properties
    heat_of_combustion: Optional[float]  # kJ/mol (standard enthalpy)
    heat_of_formation: Optional[float]  # kJ/mol (standard enthalpy)
    
    # Solubility
    water_solubility: str  # Qualitative description


def calculate_molar_mass(n_carbons: int) -> float:
    """
    Calculate molar mass of alkane CₙH₂ₙ₊₂.
    
    Args:
        n_carbons: Number of carbon atoms
        
    Returns:
        Molar mass in g/mol
    """
    n_hydrogens = 2 * n_carbons + 2
    return ATOMIC_MASS_C * n_carbons + ATOMIC_MASS_H * n_hydrogens


def estimate_state_of_matter(n_carbons: int) -> str:
    """
    Estimate state of matter at standard conditions (298.15 K, 100 kPa).
    
    At room temperature and atmospheric pressure:
    - C₁-C₄: Gases
    - C₅-C₁₇: Liquids  
    - C₁₈+: Waxy solids
    
    Args:
        n_carbons: Number of carbon atoms
        
    Returns:
        "gas", "liquid", or "solid"
    """
    if n_carbons <= 4:
        return "gas"
    elif n_carbons <= 17:
        return "liquid"
    else:
        return "solid"


# Known boiling points for n-alkanes (°C) - NIST data
N_ALKANE_BOILING_POINTS = {
    1: -161.5,   # methane
    2: -88.6,    # ethane
    3: -42.1,    # propane
    4: -0.5,     # butane
    5: 36.1,     # pentane
    6: 68.7,     # hexane
    7: 98.4,     # heptane
    8: 125.7,    # octane
    9: 150.8,    # nonane
    10: 174.1,   # decane
    11: 195.9,   # undecane
    12: 216.3,   # dodecane
    13: 235.4,   # tridecane
    14: 253.5,   # tetradecane
    15: 270.6,   # pentadecane
    16: 286.8,   # hexadecane
    17: 302.0,   # heptadecane
    18: 316.3,   # octadecane
    19: 329.9,   # nonadecane
    20: 343.0,   # icosane
    21: 356.5,   # henicosane
    22: 368.6,   # docosane
    23: 380.0,   # tricosane
    24: 391.3,   # tetracosane
    25: 401.9,   # pentacosane
    26: 412.2,   # hexacosane
    27: 422.0,   # heptacosane
    28: 431.6,   # octacosane
    29: 440.8,   # nonacosane
    30: 449.7,   # triacontane
}


def estimate_boiling_point(n_carbons: int, n_branch_points: int = 0, 
                           max_chain_length: Optional[int] = None) -> float:
    """
    Estimate boiling point at standard pressure (100 kPa).
    
    Branching reduces boiling point due to decreased surface area
    and weaker van der Waals interactions.
    
    Args:
        n_carbons: Number of carbon atoms
        n_branch_points: Number of branch points in the molecule
        max_chain_length: Length of longest carbon chain
        
    Returns:
        Estimated boiling point in °C
    """
    # Get n-alkane baseline
    if n_carbons in N_ALKANE_BOILING_POINTS:
        base_bp = N_ALKANE_BOILING_POINTS[n_carbons]
    else:
        # Extrapolate using empirical correlation
        # BP ≈ 1042.8 - 1134.7 * exp(-0.036 * n) for large n
        import math
        base_bp = 1042.8 - 1134.7 * math.exp(-0.036 * n_carbons)
    
    # Adjust for branching
    # Each branch point typically lowers BP by ~5-15°C
    # More compact molecules have less surface area
    if n_branch_points > 0 and max_chain_length is not None:
        # Branching factor: how much shorter is the main chain vs n-alkane?
        branch_factor = (n_carbons - max_chain_length) / n_carbons if n_carbons > 0 else 0
        # Typical reduction: 5-10°C per branch point, scaled by compactness
        bp_reduction = n_branch_points * (5 + 10 * branch_factor)
        base_bp -= bp_reduction
    
    return round(base_bp, 1)


# Known melting points for n-alkanes (°C) - NIST data
N_ALKANE_MELTING_POINTS = {
    1: -182.5,   # methane
    2: -182.8,   # ethane
    3: -187.7,   # propane
    4: -138.4,   # butane
    5: -129.7,   # pentane
    6: -95.3,    # hexane
    7: -90.6,    # heptane
    8: -56.8,    # octane
    9: -53.5,    # nonane
    10: -29.7,   # decane
    11: -25.6,   # undecane
    12: -9.6,    # dodecane
    13: -5.4,    # tridecane
    14: 5.9,     # tetradecane
    15: 9.9,     # pentadecane
    16: 18.2,    # hexadecane
    17: 22.0,    # heptadecane
    18: 28.2,    # octadecane
    19: 32.1,    # nonadecane
    20: 36.8,    # icosane
    21: 40.5,    # henicosane
    22: 44.4,    # docosane
    23: 47.6,    # tricosane
    24: 50.9,    # tetracosane
    25: 53.7,    # pentacosane
    26: 56.4,    # hexacosane
    27: 59.0,    # heptacosane
    28: 61.4,    # octacosane
    29: 63.7,    # nonacosane
    30: 65.8,    # triacontane
}


def estimate_melting_point(n_carbons: int, n_branch_points: int = 0) -> float:
    """
    Estimate melting point.
    
    Branching affects melting point in complex ways - symmetric branching
    can increase it, while asymmetric branching decreases it.
    
    Args:
        n_carbons: Number of carbon atoms
        n_branch_points: Number of branch points
        
    Returns:
        Estimated melting point in °C
    """
    if n_carbons in N_ALKANE_MELTING_POINTS:
        base_mp = N_ALKANE_MELTING_POINTS[n_carbons]
    else:
        # Extrapolate: MP increases roughly linearly for large alkanes
        base_mp = 2.5 * n_carbons - 15
    
    # Branching generally lowers melting point (less efficient packing)
    # unless the molecule is highly symmetric
    if n_branch_points > 0:
        mp_reduction = n_branch_points * 3  # Approximate
        base_mp -= mp_reduction
    
    return round(base_mp, 1)


# Known densities for n-alkanes (g/cm³ at 20°C) - liquid or solid
N_ALKANE_DENSITIES = {
    1: 0.424,    # methane (liquid at BP)
    2: 0.546,    # ethane (liquid at BP)
    3: 0.493,    # propane (liquid at BP)
    4: 0.579,    # butane (liquid at BP)
    5: 0.626,    # pentane
    6: 0.659,    # hexane
    7: 0.684,    # heptane
    8: 0.703,    # octane
    9: 0.718,    # nonane
    10: 0.730,   # decane
    11: 0.740,   # undecane
    12: 0.749,   # dodecane
    13: 0.756,   # tridecane
    14: 0.763,   # tetradecane
    15: 0.769,   # pentadecane
    16: 0.773,   # hexadecane
    17: 0.778,   # heptadecane
    18: 0.777,   # octadecane
    19: 0.786,   # nonadecane
    20: 0.789,   # icosane
}


def estimate_density(n_carbons: int, n_branch_points: int = 0) -> float:
    """
    Estimate density at 20°C.
    
    For gases (C1-C4), this is the liquid density at the boiling point.
    Branching typically decreases density slightly.
    
    Args:
        n_carbons: Number of carbon atoms
        n_branch_points: Number of branch points
        
    Returns:
        Estimated density in g/cm³
    """
    if n_carbons in N_ALKANE_DENSITIES:
        base_density = N_ALKANE_DENSITIES[n_carbons]
    else:
        # Density approaches ~0.80 g/cm³ for long-chain alkanes
        # Empirical fit: ρ ≈ 0.85 - 0.85 * exp(-0.15 * n)
        import math
        base_density = 0.85 - 0.85 * math.exp(-0.15 * n_carbons)
    
    # Branching slightly reduces density (less efficient packing)
    if n_branch_points > 0:
        density_reduction = n_branch_points * 0.005
        base_density -= density_reduction
    
    return round(base_density, 3)


def estimate_heat_of_combustion(n_carbons: int) -> float:
    """
    Estimate standard enthalpy of combustion (ΔH°c).
    
    Complete combustion: CₙH₂ₙ₊₂ + (3n+1)/2 O₂ → n CO₂ + (n+1) H₂O
    
    The heat of combustion increases linearly with chain length:
    ΔH°c ≈ -659 kJ/mol per CH₂ group (methylene increment)
    
    Args:
        n_carbons: Number of carbon atoms
        
    Returns:
        Standard enthalpy of combustion in kJ/mol (negative, exothermic)
    """
    # Empirical correlation based on NIST data
    # ΔH°c = -890.4 - 658.8*(n-1) for n ≥ 1
    # This gives: CH4: -890.4, C2H6: -1549.2, etc.
    heat_of_combustion = -890.4 - 658.8 * (n_carbons - 1)
    return round(heat_of_combustion, 1)


def estimate_heat_of_formation(n_carbons: int, n_branch_points: int = 0) -> float:
    """
    Estimate standard enthalpy of formation (ΔH°f).
    
    Alkanes have negative (favorable) enthalpies of formation.
    Branching slightly stabilizes the molecule (more negative ΔH°f).
    
    Args:
        n_carbons: Number of carbon atoms
        n_branch_points: Number of branch points
        
    Returns:
        Standard enthalpy of formation in kJ/mol
    """
    # Base correlation for n-alkanes
    # ΔH°f ≈ -84.7 - 20.6*(n-1) for linear alkanes
    # CH4: -74.8 kJ/mol (actual), C2H6: -84.0 kJ/mol, etc.
    
    if n_carbons == 1:
        base_hf = -74.8  # Methane
    else:
        base_hf = -84.0 - 20.6 * (n_carbons - 2)
    
    # Branching stabilizes by ~2-8 kJ/mol per branch point
    # Tertiary carbons more stable than secondary
    stabilization = n_branch_points * 5.0
    
    return round(base_hf - stabilization, 1)


def get_water_solubility_description(n_carbons: int) -> str:
    """
    Get qualitative description of water solubility.
    
    Alkanes are nonpolar and essentially insoluble in water.
    Solubility decreases with increasing chain length.
    
    Args:
        n_carbons: Number of carbon atoms
        
    Returns:
        Qualitative solubility description
    """
    if n_carbons <= 4:
        return "Practically insoluble (~0.02-0.06 g/L)"
    elif n_carbons <= 8:
        return "Insoluble (<0.01 g/L)"
    else:
        return "Insoluble (negligible)"


def get_alkane_properties(name: str, n_carbons: int, 
                          n_branch_points: int = 0,
                          max_chain_length: Optional[int] = None) -> AlkaneProperties:
    """
    Get comprehensive estimated properties for an alkane isomer.
    
    Args:
        name: IUPAC name of the isomer
        n_carbons: Number of carbon atoms
        n_branch_points: Number of branch points
        max_chain_length: Length of longest carbon chain
        
    Returns:
        AlkaneProperties dataclass with estimated values
    """
    if max_chain_length is None:
        max_chain_length = n_carbons  # Assume linear if not specified
    
    formula = f"C{n_carbons}H{2*n_carbons+2}"
    
    return AlkaneProperties(
        name=name,
        formula=formula,
        n_carbons=n_carbons,
        molar_mass=calculate_molar_mass(n_carbons),
        state_of_matter=estimate_state_of_matter(n_carbons),
        boiling_point=estimate_boiling_point(n_carbons, n_branch_points, max_chain_length),
        melting_point=estimate_melting_point(n_carbons, n_branch_points),
        density=estimate_density(n_carbons, n_branch_points),
        heat_of_combustion=estimate_heat_of_combustion(n_carbons),
        heat_of_formation=estimate_heat_of_formation(n_carbons, n_branch_points),
        water_solubility=get_water_solubility_description(n_carbons)
    )


def format_state_emoji(state: str) -> str:
    """Get emoji representation of state of matter."""
    return {
        "gas": "💨",
        "liquid": "💧", 
        "solid": "🧊"
    }.get(state, "❓")


def format_temperature(temp_c: float, include_f: bool = True) -> str:
    """Format temperature with optional Fahrenheit conversion."""
    temp_f = temp_c * 9/5 + 32
    if include_f:
        return f"{temp_c:.1f}°C ({temp_f:.1f}°F)"
    return f"{temp_c:.1f}°C"
