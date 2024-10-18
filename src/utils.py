"""
This module provides utility functions for processing and curating molecular data using RDKit and pandas. The functions include:

- _find_symmetry_groups(mol: Mol): Identifies symmetry groups in a molecule.
- _set_allowed_elements_flag(mol: Mol): Checks if a molecule contains only allowed chemical elements.
- _standardize_row(row: pd.Series): Standardizes a row of a DataFrame containing substrate and metabolite molecules.
- concat_lists(lst: List): Concatenates a list of lists into a single list.
- curate_data(data: pd.DataFrame): Curates molecular data according to specific rules.
- standardize_data(data: pd.DataFrame): Standardizes molecular data using the ChEMBL standardizer.
- symmetrize_soms(mol: Mol, soms: List[int]): Adds all atoms in a symmetry group to the list of Sites of Metabolism (SoMs) if any atom in the group is already a SoM.

The module also defines a set of allowed atoms for chemical elements and imports necessary libraries for molecular processing.
"""


from collections import defaultdict

from typing import List

import pandas as pd
from rdkit.Chem import Mol, MolToInchi, rdMolDescriptors
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
    """
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
    """
    Checks if a molecule contains only allowed chemical elements (see ALLOWED_ATOMS).

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
        return 0
    return 1


def _standardize_row(row: pd.Series) -> pd.Series:
    """
    Standardize a row of the DataFrame containing the substrate and metabolite molecules.

    Args:
        row (pd.Series): Row of the DataFrame containing the substrate and metabolite molecules.

    Returns:
        pd.Series: Standardized row of the DataFrame containing the substrate and metabolite molecules.
    """
    try:
        row["substrate_mol"] = rdMolStandardize.Cleanup(row["substrate_mol"])
        row["substrate_mol"] = rdMolStandardize.CanonicalTautomer(row["substrate_mol"])

        row["metabolite_mol"] = rdMolStandardize.Cleanup(row["metabolite_mol"])
        row["metabolite_mol"] = rdMolStandardize.CanonicalTautomer(
            row["metabolite_mol"]
        )
    except:
        row["substrate_mol"] = None
        row["metabolite_mol"] = None

    return row


def concat_lists(lst: List) -> List:
    """
    Concatenate a list of lists into a single list.

    Args:
        lst (List): List of lists to concatenate.

    Returns:
        List: Concatenated list.
    """
    return list(set(sum(lst, [])))


def curate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Curate the data according to a subset of the rules defined in the SoM predictor (AweSOM):
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
    print(
        f"Removed {data_size - len(data)} reactions with missing InChI. Data set now contains {len(data)} reactions."
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
        (data.substrate_allowed_elements_flag == 1)
        & (data.metabolite_allowed_elements_flag == 1)
    ]
    print(
        f"Chemical element filter removed {data_size - len(data)} reactions. Data set now contains {len(data)} reactions."
    )
    data_size = len(data)

    # Filter out reactions with molecular mass above 1000 Da
    data["substrate_molecular_weight"] = data["substrate_mol"].map(
        rdMolDescriptors.CalcExactMolWt
    )
    data["metabolite_molecular_weight"] = data["metabolite_mol"].map(
        rdMolDescriptors.CalcExactMolWt
    )
    data = data[
        (data.substrate_molecular_weight <= 1000)
        & (data.metabolite_molecular_weight <= 1000)
    ]
    print(
        f"Molecular weight filter removed {data_size - len(data)} reactions. Data set now contains {len(data)} reactions."
    )
    data_size = len(data)

    # Filter out reactions with fewer than 5 heavy atoms
    data["substrate_num_heavy_atoms"] = data.substrate_mol.map(
        lambda x: x.GetNumHeavyAtoms()
    )
    data["metabolite_num_heavy_atoms"] = data.metabolite_mol.map(
        lambda x: x.GetNumHeavyAtoms()
    )
    data = data[
        (data.substrate_num_heavy_atoms >= 5) & (data.metabolite_num_heavy_atoms >= 5)
    ]
    print(
        f"Minimum number of heavy atoms filter removed {data_size - len(data)} reactions. Data set now contains {len(data)} reactions."
    )
    data_size = len(data)

    # Clean up the DataFrame
    data = data.drop(
        columns=[
            "substrate_inchi",
            "metabolite_inchi",
            "substrate_allowed_elements_flag",
            "metabolite_allowed_elements_flag",
            "substrate_molecular_weight",
            "metabolite_molecular_weight",
            "substrate_num_heavy_atoms",
            "metabolite_num_heavy_atoms",
        ]
    )

    # Reset the index
    data = data.reset_index(drop=True, inplace=False)

    return data


# def filter_data(data: pd.DataFrame, n: int) -> pd.DataFrame:
#     """
#     Filters out reactions where the substrate or the metabolite has more than 30 heavy atoms.

#     Args:
#         data (pd.DataFrame): DataFrame containing the substrate and metabolite molecules.
#         n (int): Maximum number of heavy atoms allowed in the substrate and metabolite.

#     Returns:
#         data (pd.DataFrame): Filtered DataFrame containing the substrate and metabolite molecules.
#     """
#     data_size = len(data)
#     data["substrate_num_heavy_atoms"] = data.substrate_mol.map(
#         lambda x: x.GetNumHeavyAtoms()
#     )
#     data["metabolite_num_heavy_atoms"] = data.metabolite_mol.map(
#         lambda x: x.GetNumHeavyAtoms()
#     )
#     data = data[
#         (data.substrate_num_heavy_atoms < n) & (data.metabolite_num_heavy_atoms < n)
#     ]
#     print(
#         f"Maximum number of heavy atoms filter removed {data_size - len(data)} reactions. Data set now contains {len(data)} reactions."
#     )
#     data_size = len(data)
#     data = data.reset_index(drop=True, inplace=False)
#     return data


def standardize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the data using the ChEMBL standardizer.

    Args:
        data (pd.DataFrame): DataFrame containing the substrate and metabolite molecules.

    Returns:
        pd.DataFrame: DataFrame containing the standardized substrate and metabolite molecules.
    """
    data_size = len(data)

    data = data.apply(_standardize_row, axis=1)
    data = data.dropna(subset=["substrate_mol", "metabolite_mol"])

    print(
        f"Standardization removed {data_size - len(data)} reactions. Data set now contains {len(data)} reactions."
    )

    return data


def symmetrize_soms(mol: Mol, soms: List[int]) -> List[int]:
    """Adds all atoms in a symmetry group to the list of SoMs, if any atom in the group is already a SoM.

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
