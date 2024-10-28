"""This module provides various utility functions for processing and analyzing molecular data using RDKit, pandas, and NetworkX."""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List

import networkx as nx
import pandas as pd
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, Mol, MolToInchi
from rdkit.Chem.MolStandardize import rdMolStandardize

ALLOWED_ATOMS = {
    1,  # H
    5,  # B
    6,  # C
    7,  # N
    8,  # O
    9,  # F
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    35,  # Br
    53,  # I
}


def _find_symmetry_groups(mol: Mol):
    """Identify symmetry groups in a molecule.

    Args:
        mol (RDKit Mol)
    Returns:
        groups: a set of tuples containing the ids of the atoms belonging to one symmetry group
    """
    equivs = defaultdict(set)
    matches = mol.GetSubstructMatches(mol, uniquify=False)
    for match in matches:
        for idx1, idx2 in enumerate(match):
            equivs[idx1].add(idx2)
    groups = set()
    for s in equivs.values():
        groups.add(tuple(s))
    return groups


def _set_allowed_elements_flag(mol: Mol) -> int:
    """Check if a molecule contains only allowed chemical elements (see ALLOWED_ATOMS).

    Args:
        mol (RDKit Mol): Molecule to check.

    Returns:
        int: 1 if the molecule contains only allowed chemical elements, 0 otherwise.
    """
    atom_counts = set()
    for atom in mol.GetAtoms():
        atom_counts.add(atom.GetAtomicNum())
        unallowed_atoms_count = atom_counts - ALLOWED_ATOMS
    if len(unallowed_atoms_count) != 0:
        return False
    return True


def _standardize_row(row: pd.Series) -> pd.Series:
    """Standardize a dataframe row of the containing the substrate and metabolite molecules.

    Args:
        row (pd.Series): Row of the dataframe containing the substrate and metabolite molecules.

    Returns:
        pd.Series: Standardized row of the dataframe containing the substrate and metabolite molecules.
    """
    try:
        row["substrate_mol"] = rdMolStandardize.Cleanup(row["substrate_mol"])
        row["substrate_mol"] = rdMolStandardize.CanonicalTautomer(row["substrate_mol"])

        row["metabolite_mol"] = rdMolStandardize.Cleanup(row["metabolite_mol"])
        row["metabolite_mol"] = rdMolStandardize.CanonicalTautomer(
            row["metabolite_mol"]
        )
    except (ValueError, KeyError, AttributeError) as e:
        print(f"Error: {e} in row {row}")
        row["substrate_mol"] = None
        row["metabolite_mol"] = None
    return row


def concat_lists(lst: List) -> List:
    """Concatenate a list of lists into a single list.

    Args:
        lst (List): List of lists to concatenate.

    Returns:
        List: Concatenated list.
    """
    return list(set(sum(lst, [])))


def count_elements(mol: Mol) -> dict:
    """Count the number of atoms of each element in a molecule.

    Args:
        mol (RDKit Mol): Molecule to count the elements of.

    Returns:
        dict: Dictionary containing the counts of each element in the molecule.
    """
    element_counts = Counter()
    periodic_table = GetPeriodicTable()
    for atom in mol.GetAtoms():
        element = periodic_table.GetElementSymbol(atom.GetAtomicNum())
        element_counts[element] += 1
    return element_counts


def curate_data(data: pd.DataFrame, logger_path: str) -> pd.DataFrame:
    """Curate the data according to a subset of the rules defined in the SoM predictor (AweSOM).

    Rules:
    (1) Compute each compound's InChI. Remove any entries for which an InChI cannot be computed, and entries with identical database-internal molecular identifiers but differing InChI.
    (2) Discard compounds containing any chemical element other than H, B, C, N, O, F, Si, P, S, Cl, Br, I.
    (3) Discard compounds with molecular mass above 1000 Da.
    (4) Discard compounds with fewer than 5 heavy atoms.

    Args:
        data (pd.DataFrame): DataFrame containing the substrate and metabolite molecules.

    Returns:
        pd.DataFrame: Curated DataFrame containing the substrate and metabolite molecules.
    """
    # Filter out reactions with missing InChI
    data_size = len(data)
    data["substrate_inchi"] = data["substrate_mol"].map(MolToInchi)
    data["metabolite_inchi"] = data["metabolite_mol"].map(MolToInchi)
    data = data.dropna(subset=["substrate_inchi", "metabolite_inchi"])
    log(
        logger_path,
        f"Removed {data_size - len(data)} reactions with missing InChI. Data set now contains {len(data)} reactions.",
    )
    data_size = len(data)

    # Filter out reactions with unusual chemical elements
    data["substrate_allowed_elements_flag"] = data["substrate_mol"].apply(
        _set_allowed_elements_flag
    )
    data["metabolite_allowed_elements_flag"] = data["metabolite_mol"].apply(
        _set_allowed_elements_flag
    )
    data = data[
        data.substrate_allowed_elements_flag & data.metabolite_allowed_elements_flag
    ]
    log(
        logger_path,
        f"Chemical element filter removed {data_size - len(data)} reactions. Data set now contains {len(data)} reactions.",
    )
    data_size = len(data)

    # Clean up the DataFrame
    data = data.drop(
        columns=[
            "substrate_inchi",
            "metabolite_inchi",
            "substrate_allowed_elements_flag",
            "metabolite_allowed_elements_flag",
        ]
    )

    # Reset the index
    data = data.reset_index(drop=True, inplace=False)

    return data


def is_carbon_count_unchanged(
    substrate_elements: dict, metabolite_elements: dict
) -> bool:
    """Check if the number of carbons remains the same."""
    return substrate_elements.get("C", 0) == metabolite_elements.get("C", 0)


def is_halogen_count_decreased(
    substrate_elements: dict, metabolite_elements: dict
) -> bool:
    """Check if the number of halogens decreases by 1."""
    for hal in ["F", "Cl", "Br", "I"]:
        if (substrate_elements.get(hal, 0) - 1 == metabolite_elements.get(hal, 0)) or (
            substrate_elements.get(hal, 0) == 1 and metabolite_elements.get(hal, 0) == 0
        ):
            return True
    return False


def is_oxygen_count_increased(
    substrate_elements: dict, metabolite_elements: dict
) -> bool:
    """Check if the number of oxygens increases by 1."""
    return substrate_elements.get("O", 0) + 1 == metabolite_elements.get("O", 0) or (
        substrate_elements.get("O", 0) == 0 and metabolite_elements.get("O", 0) == 1
    )


def get_bond_order(molecule: Mol, atom_idx1: int, atom_idx2: int) -> int:
    """Get the order of the bond between two specified atoms.

    Args:
        molecule: RDKit molecule object.
        atom_idx1: Index of the first atom.
        atom_idx2: Index of the second atom.
    Returns:
        The bond order (1 for single, 2 for double, 3 for triple, 4 for aromatic).
        Returns None if no bond exists between the specified atoms.
    """
    # Get the bond between the specified atoms
    bond = molecule.GetBondBetweenAtoms(atom_idx1, atom_idx2)

    if bond is None:
        return None  # No bond exists between the specified atoms

    # Determine the bond multiplicity based on the bond type
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return 1
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.BondType.AROMATIC:
        return 4

    return None  # In case of an unknown bond type


def log(path: str, message: str) -> None:
    """Log a message to a text file.

    Args:
        path (str): Path to the log file.
        message (str): Message to log.

    Returns:
        None
    """
    with open(path, "a+", encoding="utf-8") as f:
        f.write(f"{datetime.now()} {message}\n")


def mol_to_graph(mol: Mol) -> nx.Graph:
    """Convert an RDKit molecule to a NetworkX graph.

    Args:
        mol (RDKit Mol): Molecule to convert.

    Returns:
        mol_graph (NetworkX Graph): Graph representation of the molecule.
    """
    mol_graph = nx.Graph()
    for atom in mol.GetAtoms():
        mol_graph.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        mol_graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return mol_graph


def standardize_data(data: pd.DataFrame, logger_path: str) -> pd.DataFrame:
    """Standardize the data using the ChEMBL standardizer.

    Args:
        data (pd.DataFrame): DataFrame containing the substrate and metabolite molecules.

    Returns:
        pd.DataFrame: DataFrame containing the standardized substrate and metabolite molecules.
    """
    data_size = len(data)

    data = data.apply(_standardize_row, axis=1)
    data = data.dropna(subset=["substrate_mol", "metabolite_mol"])

    log(
        logger_path,
        f"Standardization removed {data_size - len(data)} reactions. Data set now contains {len(data)} reactions.",
    )

    return data


def symmetrize_soms(mol: Mol, soms: List[int]) -> List[int]:
    """Add all atoms in a symmetry group to the list of SoMs, if any atom in the group is already a SoM.

    Args:
        mol (Mol): RDKit molecule
        soms (List[int]): list of atom indices of the already found SoMs

    Returns:
        List[int]: updated list of SoMs
    """
    symmetry_groups = _find_symmetry_groups(mol)

    soms_symmetrized = set(soms)
    for group in symmetry_groups:
        if len(group) > 1:
            for som in soms:
                if som in group:
                    soms_symmetrized.update(group)

    return sorted(list(soms_symmetrized))
