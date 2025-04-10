"""Tests for the main AutoSOM functionality."""

import pytest
from rdkit import Chem

from autosom.autosom import annotate_soms


@pytest.fixture
def sample_reactions():
    """Create sample reactions for testing."""
    # Create a list of substrate-metabolite pairs
    reactions = [
        # Addition reaction: benzene -> phenol
        (Chem.MolFromSmiles("c1ccccc1"), Chem.MolFromSmiles("c1ccccc1O")),
        # Elimination reaction: phenol -> benzene
        (Chem.MolFromSmiles("c1ccccc1O"), Chem.MolFromSmiles("c1ccccc1")),
        # Complex reaction: hydrolysis of butyrolactone
        (Chem.MolFromSmiles("C1CC(=O)OC1"), Chem.MolFromSmiles("OCCCC(=O)O")),
    ]
    return reactions


def test_annotate_soms_initialization():
    """Test annotate_soms function parameters."""
    params = (
        "tests/test.log",
        55,
        True,
    )  # logger_path, filter_size, ester_hydrolysis_flag
    substrate = Chem.MolFromSmiles("c1ccccc1")
    metabolite = Chem.MolFromSmiles("C1CCCCC1")
    substrate_data = (substrate, 1)
    metabolite_data = (metabolite, 2)

    soms, reaction_type, time = annotate_soms(params, substrate_data, metabolite_data)
    assert isinstance(soms, list)
    assert isinstance(reaction_type, str)
    assert isinstance(time, float)


def test_annotate_soms_process_reactions(sample_reactions):
    """Test processing of multiple reactions."""
    params = ("tests/test.log", 55, True)

    results = []
    for substrate, metabolite in sample_reactions:
        substrate_data = (substrate, 1)
        metabolite_data = (metabolite, 2)
        result = annotate_soms(params, substrate_data, metabolite_data)
        results.append(result)

    # Check that we got results for all reactions
    assert len(results) == len(sample_reactions)

    # Check that each result has the expected structure
    for result in results:
        assert isinstance(result, tuple)
        assert len(result) == 3  # soms, reaction_type, time
        soms, reaction_type, time = result
        assert isinstance(soms, list)
        assert isinstance(reaction_type, str)
        assert isinstance(time, float)


def test_annotate_soms_reaction_types(sample_reactions):
    """Test identification of different reaction types."""
    params = ("tests/test.log", 55, True)

    results = []
    for substrate, metabolite in sample_reactions:
        substrate_data = (substrate, 1)
        metabolite_data = (metabolite, 2)
        result = annotate_soms(params, substrate_data, metabolite_data)
        results.append(result)

    # Check that different reaction types are identified
    reaction_types = [result[1] for result in results]
    assert "addition" in reaction_types[0]
    assert "elimination" in reaction_types[1]
    assert "complex" in reaction_types[2]
