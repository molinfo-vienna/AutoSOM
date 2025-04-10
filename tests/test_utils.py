"""Tests for utility functions."""

import pandas as pd
import pytest
from rdkit import Chem

from autosom.utils import (
    check_and_collapse_substrate_id,
    concat_lists,
    count_elements,
    get_bond_order,
    get_neighbor_atomic_nums,
    is_carbon_count_unchanged,
    is_halogen_count_decreased,
    is_oxygen_count_increased,
    mol_to_graph,
    symmetrize_soms,
)


@pytest.fixture
def sample_molecule():
    """Create a sample molecule for testing."""
    return Chem.MolFromSmiles("c1ccccc1O")  # Phenol


def test_check_and_collapse_substrate_id():
    """Test substrate ID collapsing functionality."""

    # Test with single ID
    series = pd.Series([1])
    assert check_and_collapse_substrate_id(series) == 1

    # Test with multiple identical IDs
    series = pd.Series([1, 1, 1])
    assert check_and_collapse_substrate_id(series) == 1

    # Test with different IDs
    series = pd.Series([1, 2, 3])
    assert check_and_collapse_substrate_id(series) is None

    # Test with None
    assert check_and_collapse_substrate_id(None) is None


def test_concat_lists():
    """Test list concatenation."""
    test_lists = [[1, 2], [3, 4], [5]]
    expected = [1, 2, 3, 4, 5]
    assert concat_lists(test_lists) == expected


def test_count_elements(sample_molecule):
    """Test element counting in molecules."""
    elements = count_elements(sample_molecule)
    assert elements["C"] == 6  # 6 carbons in phenol
    assert elements["O"] == 1  # 1 oxygen in phenol


def test_get_bond_order(sample_molecule):
    """Test bond order retrieval."""
    # Get the bond between C6 and O
    bond_order = get_bond_order(sample_molecule, 5, 6)
    assert bond_order == 1  # Single bond


def test_get_neighbor_atomic_nums(sample_molecule):
    """Test neighbor atomic number retrieval."""
    # Get neighbors of the oxygen atom (index 6)
    neighbors = get_neighbor_atomic_nums(sample_molecule, 6)
    assert len(neighbors) == 1  # Oxygen has one neighbor (carbon)
    assert neighbors[6] == 1  # Only one neighboring carbon atom ({6: 1})


def test_is_carbon_count_unchanged():
    """Test carbon count comparison."""
    substrate_elements = {"C": 6, "H": 6, "O": 0}
    metabolite_elements = {"C": 6, "H": 6, "O": 1}
    assert is_carbon_count_unchanged(substrate_elements, metabolite_elements) is True


def test_is_halogen_count_decreased():
    """Test halogen count comparison."""
    substrate_elements = {"C": 6, "H": 5, "Cl": 1}
    metabolite_elements = {"C": 6, "H": 6, "Cl": 0}
    assert is_halogen_count_decreased(substrate_elements, metabolite_elements) is True


def test_is_oxygen_count_increased():
    """Test oxygen count comparison."""
    substrate_elements = {"C": 6, "H": 6, "O": 0}
    metabolite_elements = {"C": 6, "H": 6, "O": 1}
    assert is_oxygen_count_increased(substrate_elements, metabolite_elements) is True


def test_mol_to_graph(sample_molecule):
    """Test molecule to graph conversion."""
    graph = mol_to_graph(sample_molecule)
    assert len(graph.nodes) == 7  # 6 carbons + 1 oxygen
    assert len(graph.edges) == 7  # 6 C-C bonds + 1 C-O bond


def test_symmetrize_soms(sample_molecule):
    """Test SOM symmetrization."""
    soms = [0]  # Single SOM at position 0
    symmetrized = symmetrize_soms(sample_molecule, soms)
    # In benzene, position 0 is equivalent to position 4
    assert len(symmetrized) > 1
    assert all(pos in symmetrized for pos in [0, 4])
