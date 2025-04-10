"""Tests for specific annotator classes."""

import pytest
from rdkit import Chem

from autosom.autosom import annotate_soms


@pytest.fixture
def sample_molecules():
    """Create sample molecules for testing different reaction types."""
    # Addition reaction: benzene -> phenol
    addition_substrate = Chem.MolFromSmiles("c1ccccc1")
    addition_metabolite = Chem.MolFromSmiles("c1ccccc1O")

    # Elimination reaction: phenol -> benzene
    elimination_substrate = Chem.MolFromSmiles("c1ccccc1O")
    elimination_metabolite = Chem.MolFromSmiles("c1ccccc1")

    # Complex reaction: hydrolysis of butyrolactone
    complex_substrate = Chem.MolFromSmiles("C1CC(=O)OC1")
    complex_metabolite = Chem.MolFromSmiles("OCCCC(=O)O")

    # Invalid reaction: benzene -> benzene
    invalid_substrate = None
    invalid_metabolite = Chem.MolFromSmiles("c1ccccc1")

    return {
        "addition": (addition_substrate, addition_metabolite),
        "elimination": (elimination_substrate, elimination_metabolite),
        "complex": (complex_substrate, complex_metabolite),
        "invalid": (invalid_substrate, invalid_metabolite),
    }


@pytest.fixture
def annotator_params():
    """Create parameters for annotators."""
    return (
        "tests/test.log",
        55,
        True,
    )  # logger_path, filter_size, ester_hydrolysis_flag


def test_addition_annotator(sample_molecules, annotator_params):
    """Test addition reaction annotation."""
    substrate, metabolite = sample_molecules["addition"]
    substrate_data = (substrate, 1)
    metabolite_data = (metabolite, 2)

    soms, reaction_type, _ = annotate_soms(
        annotator_params, substrate_data, metabolite_data
    )

    assert len(soms) > 0  # Should identify addition sites
    assert "addition" in reaction_type


def test_elimination_annotator(sample_molecules, annotator_params):
    """Test elimination reaction annotation."""
    substrate, metabolite = sample_molecules["elimination"]
    substrate_data = (substrate, 1)
    metabolite_data = (metabolite, 2)

    soms, reaction_type, _ = annotate_soms(
        annotator_params, substrate_data, metabolite_data
    )

    assert len(soms) > 0  # Should identify elimination sites
    assert "elimination" in reaction_type


def test_complex_annotator(sample_molecules, annotator_params):
    """Test complex reaction annotation."""
    substrate, metabolite = sample_molecules["complex"]
    substrate_data = (substrate, 1)
    metabolite_data = (metabolite, 2)

    soms, reaction_type, _ = annotate_soms(
        annotator_params, substrate_data, metabolite_data
    )

    assert len(soms) > 0  # Should identify complex sites
    assert "complex" in reaction_type


def test_invalid_reactions(sample_molecules, annotator_params):
    """Test that annotators reject invalid reactions."""
    # Test addition annotator with elimination reaction
    substrate, metabolite = sample_molecules["invalid"]
    substrate_data = (substrate, 1)
    metabolite_data = (metabolite, 2)

    soms, reaction_type, _ = annotate_soms(
        annotator_params, substrate_data, metabolite_data
    )

    assert len(soms) == 0  # Should return empty list
    assert "unknown" in reaction_type
