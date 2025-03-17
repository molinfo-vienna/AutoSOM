"""Tests for the BaseAnnotator class."""

import pytest
from rdkit import Chem
from rdkit.Chem import Mol

from src.base_annotator import BaseAnnotator

@pytest.fixture
def sample_molecules():
    """Create sample molecules for testing."""
    # Create a simple benzene molecule as substrate
    substrate = Chem.MolFromSmiles("c1ccccc1")
    # Create a phenol molecule as metabolite
    metabolite = Chem.MolFromSmiles("c1ccccc1O")
    return substrate, metabolite


@pytest.fixture
def base_annotator(sample_molecules):
    """Create a BaseAnnotator instance for testing."""
    substrate, metabolite = sample_molecules
    params = ("tests/test.log", 55, True)  # logger_path, filter_size, ester_hydrolysis_flag
    substrate_data = (substrate, 1)
    metabolite_data = (metabolite, 2)
    return BaseAnnotator(params, substrate_data, metabolite_data)


def test_initialization(base_annotator):
    """Test BaseAnnotator initialization."""
    assert base_annotator.logger_path == "tests/test.log"
    assert base_annotator.filter_size == 55
    assert base_annotator.ester_hydrolysis_flag is True
    assert base_annotator.substrate_id == 1
    assert base_annotator.metabolite_id == 2


def test_remove_hydrogens(base_annotator):
    """Test hydrogen removal from molecules."""
    base_annotator.remove_hydrogens()
    assert base_annotator.substrate.GetNumAtoms() == 6  # Benzene has 6 carbons
    assert base_annotator.metabolite.GetNumAtoms() == 7  # Phenol has 6 carbons + 1 oxygen


def test_standardize_molecules(base_annotator):
    """Test molecule standardization."""
    result = base_annotator.standardize_molecules()
    assert result is True
    # Check if molecules are still valid
    assert Chem.SanitizeMol(base_annotator.substrate) == Chem.SANITIZE_NONE
    assert Chem.SanitizeMol(base_annotator.metabolite) == Chem.SANITIZE_NONE


def test_is_too_large_to_process(base_annotator):
    """Test molecule size check."""
    # Benzene and phenol are small molecules
    assert base_annotator.is_too_large_to_process() is False


def test_compute_weight_ratio(base_annotator):
    """Test molecular weight ratio computation."""
    ratio = base_annotator.compute_weight_ratio()
    # Phenol (metabolite) should be heavier than benzene (substrate),
    # thus returning 1
    assert ratio == 1


def test_check_atom_types(base_annotator):
    """Test atom type checking."""
    result = base_annotator.check_atom_types()
    assert result is True  # Benzene and phenol have compatible atom types


def test_check_validity(base_annotator):
    """Test molecule validity check."""
    result = base_annotator.check_validity()
    assert result is True  # Both molecules are valid 