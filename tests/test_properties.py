"""
Tests for alkane property estimation module.
"""

import pytest
from src.properties import (
    calculate_molar_mass,
    estimate_state_of_matter,
    estimate_boiling_point,
    estimate_melting_point,
    estimate_density,
    estimate_heat_of_combustion,
    estimate_heat_of_formation,
    get_water_solubility_description,
    get_alkane_properties,
    format_state_emoji,
    format_temperature,
    ATOMIC_MASS_C,
    ATOMIC_MASS_H
)


class TestMolarMass:
    """Test molar mass calculations."""
    
    def test_methane(self):
        """CH4: 12.011 + 4*1.008 = 16.043"""
        mass = calculate_molar_mass(1)
        assert abs(mass - 16.043) < 0.001
    
    def test_ethane(self):
        """C2H6: 2*12.011 + 6*1.008 = 30.070"""
        mass = calculate_molar_mass(2)
        assert abs(mass - 30.070) < 0.001
    
    def test_decane(self):
        """C10H22: 10*12.011 + 22*1.008 = 142.286"""
        mass = calculate_molar_mass(10)
        assert abs(mass - 142.286) < 0.001
    
    def test_icosane(self):
        """C20H42: 20*12.011 + 42*1.008 = 282.556"""
        mass = calculate_molar_mass(20)
        assert abs(mass - 282.556) < 0.001


class TestStateOfMatter:
    """Test state of matter estimation at STP."""
    
    def test_gases(self):
        """C1-C4 should be gases."""
        for n in range(1, 5):
            assert estimate_state_of_matter(n) == "gas"
    
    def test_liquids(self):
        """C5-C17 should be liquids."""
        for n in range(5, 18):
            assert estimate_state_of_matter(n) == "liquid"
    
    def test_solids(self):
        """C18+ should be solids."""
        for n in range(18, 31):
            assert estimate_state_of_matter(n) == "solid"


class TestBoilingPoint:
    """Test boiling point estimation."""
    
    def test_methane_bp(self):
        """Methane BP should be around -161.5°C."""
        bp = estimate_boiling_point(1)
        assert abs(bp - (-161.5)) < 1.0
    
    def test_pentane_bp(self):
        """n-Pentane BP should be around 36°C."""
        bp = estimate_boiling_point(5)
        assert abs(bp - 36.1) < 1.0
    
    def test_decane_bp(self):
        """n-Decane BP should be around 174°C."""
        bp = estimate_boiling_point(10)
        assert abs(bp - 174.1) < 1.0
    
    def test_branching_lowers_bp(self):
        """Branched isomers should have lower BP than n-alkane."""
        n_alkane_bp = estimate_boiling_point(8, n_branch_points=0, max_chain_length=8)
        branched_bp = estimate_boiling_point(8, n_branch_points=2, max_chain_length=6)
        assert branched_bp < n_alkane_bp
    
    def test_bp_increases_with_chain_length(self):
        """BP should increase with chain length."""
        for n in range(1, 25):
            bp_small = estimate_boiling_point(n)
            bp_large = estimate_boiling_point(n + 1)
            assert bp_large > bp_small


class TestMeltingPoint:
    """Test melting point estimation."""
    
    def test_methane_mp(self):
        """Methane MP should be around -182.5°C."""
        mp = estimate_melting_point(1)
        assert abs(mp - (-182.5)) < 1.0
    
    def test_decane_mp(self):
        """n-Decane MP should be around -30°C."""
        mp = estimate_melting_point(10)
        assert abs(mp - (-29.7)) < 2.0
    
    def test_icosane_mp(self):
        """n-Icosane MP should be around 37°C (solid at room temp)."""
        mp = estimate_melting_point(20)
        assert mp > 25  # Should be solid at room temperature


class TestDensity:
    """Test density estimation."""
    
    def test_pentane_density(self):
        """n-Pentane density should be around 0.626 g/cm³."""
        d = estimate_density(5)
        assert abs(d - 0.626) < 0.02
    
    def test_decane_density(self):
        """n-Decane density should be around 0.730 g/cm³."""
        d = estimate_density(10)
        assert abs(d - 0.730) < 0.02
    
    def test_density_increases_with_chain_length(self):
        """Density should increase with chain length (up to ~0.8)."""
        for n in range(5, 15):
            d_small = estimate_density(n)
            d_large = estimate_density(n + 1)
            assert d_large > d_small
    
    def test_branching_decreases_density(self):
        """Branched isomers should have slightly lower density."""
        d_linear = estimate_density(8, n_branch_points=0)
        d_branched = estimate_density(8, n_branch_points=2)
        assert d_branched < d_linear


class TestHeatOfCombustion:
    """Test heat of combustion estimation."""
    
    def test_methane_combustion(self):
        """Methane combustion: ΔH°c ≈ -890 kJ/mol."""
        hc = estimate_heat_of_combustion(1)
        assert abs(hc - (-890.4)) < 5.0
    
    def test_combustion_is_exothermic(self):
        """All combustion enthalpies should be negative."""
        for n in range(1, 31):
            assert estimate_heat_of_combustion(n) < 0
    
    def test_combustion_increases_with_size(self):
        """Magnitude of combustion should increase with carbon count."""
        for n in range(1, 25):
            hc_small = abs(estimate_heat_of_combustion(n))
            hc_large = abs(estimate_heat_of_combustion(n + 1))
            assert hc_large > hc_small


class TestHeatOfFormation:
    """Test heat of formation estimation."""
    
    def test_methane_formation(self):
        """Methane formation: ΔH°f ≈ -74.8 kJ/mol."""
        hf = estimate_heat_of_formation(1)
        assert abs(hf - (-74.8)) < 1.0
    
    def test_branching_stabilizes(self):
        """Branched isomers should have more negative ΔH°f."""
        hf_linear = estimate_heat_of_formation(8, n_branch_points=0)
        hf_branched = estimate_heat_of_formation(8, n_branch_points=2)
        assert hf_branched < hf_linear  # More negative = more stable


class TestWaterSolubility:
    """Test water solubility description."""
    
    def test_small_alkanes(self):
        """Small alkanes should be practically insoluble."""
        desc = get_water_solubility_description(4)
        assert "insoluble" in desc.lower()
    
    def test_large_alkanes(self):
        """Large alkanes should be insoluble."""
        desc = get_water_solubility_description(20)
        assert "insoluble" in desc.lower()


class TestGetAlkaneProperties:
    """Test the main property getter function."""
    
    def test_returns_all_properties(self):
        """Should return all expected properties."""
        props = get_alkane_properties("n-hexane", 6, 0, 6)
        
        assert props.name == "n-hexane"
        assert props.formula == "C6H14"
        assert props.n_carbons == 6
        assert props.molar_mass > 0
        assert props.state_of_matter in ["gas", "liquid", "solid"]
        assert props.boiling_point is not None
        assert props.melting_point is not None
        assert props.density is not None
        assert props.heat_of_combustion is not None
        assert props.heat_of_formation is not None
        assert props.water_solubility is not None
    
    def test_hexane_is_liquid(self):
        """Hexane should be a liquid at STP."""
        props = get_alkane_properties("n-hexane", 6, 0, 6)
        assert props.state_of_matter == "liquid"
    
    def test_methane_is_gas(self):
        """Methane should be a gas at STP."""
        props = get_alkane_properties("methane", 1, 0, 1)
        assert props.state_of_matter == "gas"


class TestFormatFunctions:
    """Test formatting helper functions."""
    
    def test_format_state_emoji(self):
        """Should return correct emoji for each state."""
        assert format_state_emoji("gas") == "💨"
        assert format_state_emoji("liquid") == "💧"
        assert format_state_emoji("solid") == "🧊"
    
    def test_format_temperature(self):
        """Should format temperature correctly."""
        result = format_temperature(25.0)
        assert "25.0°C" in result
        assert "77.0°F" in result
    
    def test_format_temperature_no_f(self):
        """Should format temperature without Fahrenheit."""
        result = format_temperature(100.0, include_f=False)
        assert "100.0°C" in result
        assert "°F" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
