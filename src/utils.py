# pylint: disable=E1101

"""This module provides various utility functions for processing and \
analyzing molecular data using RDKit, pandas, and NetworkX."""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Optional

import networkx as nx
import pandas as pd
from chembl_structure_pipeline import standardizer
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, Mol, MolToInchiKey
from rdkit.Chem.MolStandardize import rdMolStandardize


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


def _standardize_row(row: pd.Series) -> pd.Series:
    """Standardize a dataframe row of the containing the substrate and metabolite.

    Args:
        row (pd.Series): Row of the dataframe containing the substrate and metabolite.

    Returns:
        pd.Series: Standardized row of the dataframe containing the substrate and metabolite.
    """
    try:
        row["substrate_mol"] = standardizer.get_parent_mol(row["substrate_mol"])[0]
        row["metabolite_mol"] = standardizer.get_parent_mol(row["metabolite_mol"])[0]

        row["substrate_mol"] = rdMolStandardize.CanonicalTautomer(row["substrate_mol"])
        row["metabolite_mol"] = rdMolStandardize.CanonicalTautomer(
            row["metabolite_mol"]
        )

        # Sanitize the molecules (this operation is in place)
        Chem.SanitizeMol(row["substrate_mol"])
        Chem.SanitizeMol(row["metabolite_mol"])

    except (ValueError, KeyError, AttributeError) as e:
        print(f"Error: {e} in row {row}")
        row["substrate_mol"] = None
        row["metabolite_mol"] = None
    return row


def check_and_collapse_substrate_id(substrate_id) -> Optional[int]:
    """Collapse substrate_id to a single id if multiple ids are present."""
    if substrate_id is None:
        return None
    substrate_id_lst = substrate_id.to_list()
    if len(substrate_id_lst) > 1:
        if len(set(substrate_id_lst)) > 1:
            print(f"Warning: Multiple substrate ids found: {substrate_id_lst}")
            return None
    return substrate_id_lst[0]


def concat_lists(lst: List) -> List:
    """Concatenate a list of lists into a single list.

    Args:
        lst (List): List of lists to concatenate.

    Returns:
        List: Concatenated list.
    """
    return list(set(sum(lst, [])))


def count_elements(mol: Mol) -> dict[str, int]:
    """Count the number of atoms of each element in a molecule.

    Args:
        mol (RDKit Mol): Molecule to count the elements of.

    Returns:
        dict: Dictionary containing the counts of each element in the molecule.
    """
    element_counts: dict[str, int] = Counter()
    periodic_table = GetPeriodicTable()
    for atom in mol.GetAtoms():
        element = periodic_table.GetElementSymbol(atom.GetAtomicNum())
        element_counts[element] += 1
    return element_counts


def curate_data(data: pd.DataFrame, logger_path: str) -> pd.DataFrame:
    """Curate the data according to the following rules.

    Rules:
    (1) Remove any entries for which an InChI cannot be computed,
    and entries with identical database-internal molecular identifiers
    but differing InChI.
    (2) Discard compounds containing any chemical element other
    than H, B, C, N, O, F, Si, P, S, Cl, Br, I.
    (3) Remove all hydrogen atoms.

    Args:
        data (pd.DataFrame): DataFrame containing the substrate and metabolite molecules.

    Returns:
        pd.DataFrame: Curated DataFrame containing the substrate and metabolite molecules.
    """
    # Filter out reactions with missing InChI
    data_size = len(data)
    data["substrate_inchikey"] = data["substrate_mol"].map(MolToInchiKey)
    data["substrate_inchikey"] = data["metabolite_mol"].map(MolToInchiKey)
    data = data.dropna(subset=["substrate_inchikey", "substrate_inchikey"])
    log(
        logger_path,
        f"Removed {data_size - len(data)} reactions with missing InChI.",
    )
    data_size = len(data)

    # Clean up the DataFrame
    data = data.drop(
        columns=[
            "substrate_inchikey",
            "substrate_inchikey",
        ]
    )

    # Reset the index
    data = data.reset_index(drop=True, inplace=False)

    # Remove all hydrogen atoms
    data["substrate_mol"] = data["substrate_mol"].apply(
        lambda x: Chem.RemoveAllHs(x) if x is not None else None
    )
    data["metabolite_mol"] = data["metabolite_mol"].apply(
        lambda x: Chem.RemoveAllHs(x) if x is not None else None
    )

    return data


def get_bond_order(molecule: Mol, atom_idx1: int, atom_idx2: int) -> Optional[int]:
    """Get the order of the bond between two specified atoms.

    Args:
        molecule: RDKit molecule object.
        atom_idx1: Index of the first atom.
        atom_idx2: Index of the second atom.
    Returns:
        The bond order (1 for single, 2 for double, 3 for triple, 4 for aromatic).
        Returns None if no bond exists between the specified atoms.
    """
    bond = molecule.GetBondBetweenAtoms(atom_idx1, atom_idx2)

    if bond is None:
        return None

    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return 1
    if bond_type == Chem.BondType.DOUBLE:
        return 2
    if bond_type == Chem.BondType.TRIPLE:
        return 3
    if bond_type == Chem.BondType.AROMATIC:
        return 4

    return None


def get_neighbor_atomic_nums(mol, atom_id) -> set:
    """Return a set of atomic numbers of neighboring atoms."""
    return {
        neighbor.GetAtomicNum()
        for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors()
    }


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
        f"Standardization removed {data_size - len(data)} reactions.",
    )

    return data


def symmetrize_soms(mol: Mol, soms: List[int]) -> List[int]:
    """Add all atoms in a symmetry group to the list of SoMs, \
    if any atom in the group is already a SoM.

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
