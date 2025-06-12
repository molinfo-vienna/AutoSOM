"""Provides utility functions for processing and analyzing molecular data."""

from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Optional

import networkx as nx
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable, Mol


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


def check_and_collapse_substrate_id(substrate_id) -> Optional[int]:
    """Collapse list of substrate ids within a canonical smiles group into a single substrate id. \
    If more than one substrate id is found, print a warning."""
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


def get_neighbor_atomic_nums(mol, atom_id) -> dict[int, int]:
    """Return a dict of atomic numbers and counts of neighboring atoms."""
    neighboring_atoms = {}
    for neighbor in mol.GetAtomWithIdx(atom_id).GetNeighbors():
        atomic_num = neighbor.GetAtomicNum()
        if atomic_num not in neighboring_atoms:
            neighboring_atoms[atomic_num] = 0
        neighboring_atoms[atomic_num] += 1
    return neighboring_atoms


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
