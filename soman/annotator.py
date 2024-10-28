"""This module provides functionalities for annotating the sites of metabolism (SoMs) given a substrate and a metabolite molecule."""

from typing import Callable, List, Tuple

import numpy as np
from networkx.algorithms import isomorphism
from rdkit.Chem import (
    FragmentOnBonds,
    GetMolFrags,
    Mol,
    MolFromSmarts,
    MolFromSmiles,
    rdFingerprintGenerator,
    rdFMCS,
)
from rdkit.DataStructs import TanimotoSimilarity

from soman.utils import (
    count_elements,
    get_bond_order,
    is_carbon_count_unchanged,
    is_halogen_count_decreased,
    is_oxygen_count_increased,
    log,
    mol_to_graph,
)


class Annotator:
    """
    Annotator annotates Sites of Metabolism (SoMs) in chemical reactions.

    Attributes:
        substrate (Tuple[Mol, int]): The substrate molecule and its ID.
        metabolite (Tuple[Mol, int]): The metabolite molecule and its ID.
        logger_path (str): Path to the log file.
        filter_size (int): Size of the filter used in the analysis.
        ester_hydrolysis (bool): Flag for enabling ester hydrolysis.
        params (rdFMCS.MCSParameters): Parameters for the Maximum Common Substructure (MCS) search.
        mapping (dict): Mapping of atom indices between substrate and metabolite.
        soms (List[int]): List of identified SoMs.
        reaction_type (str): Type of reaction identified.

    Methods:
        _correct_acetal_hydrolysis: Correct SoMs for acetals if applicable.
        _correct_carnitine_addition: Correct SoMs for the addition of carnitine to a carboxylic acid if applicable.
        _correct_cn_redox: Apply corrections if the redox reaction involves a C-N bond.
        _correct_epoxide: Correct the SoMs for halogen to hydroxy oxidation if the reaction produces an epoxide instead of the typical alcohol.
        _correct_ester_hydrolysis: Correct SoMs for ester hydrolysis if applicable.
        _correct_phosphore: Add exception handling for reactions containing phosphorus.
        _correct_piperazine_ring_hydroysis: Correct SoMs for piperazine ring opening if applicable.
        _correct_quinone_like_oxidation: Correct the SoMs for halogen to hydroxy oxidation if the reaction produces a quinone-like metabolite instead of the typical alcohol.
        _correct_sulfur_derivatives_hydrolysis: Correct SoMs for the hydrolysis of sulfur derivatives (sulfamate, sulfonamide, sulfonate, sulfuric diamide etc.) if applicable.
        _has_single_and_double_bonded_oxygen: Check if an atom has both single and double bonded oxygen neighbors.
        _initialize_mcs_params: Initialize the MCS parameters.
        _is_in_epoxide: Check if the atom is in an epoxide.
        _is_in_quinone: Check if the atom is in a quinone-like structure.
        _find_unmapped_halogen: Find the halogen atom in the substrate that is not present in the mapping.
        _find_unmatched_atoms: Find unmatched atoms between the target and the query molecule.
        _general_case_simple_addition: Identify SoMs in the simple addition case based on unmatched atoms.
        _general_case_simple_elimination: Identify SoMs in the simple elimination case based on unmatched atoms.
        _find_sulfur_index: Find the sulfur atom index in the glutathione structure.
        _find_sulfur_neighbor_index: Find the index of the atom neighboring the sulfur atom.
        _get_non_glutathione_fragment: Split the metabolite into fragments and identify the one without the glutathione moiety.
        _map_atoms_glutathione: Map atoms between two molecules using Maximum Common Substructure (MCS).
        _map_atoms: Create mapping between query and target based on MCS.
        _set_mcs_bond_typer_param: Set the MCS bond compare parameter.
        handle_complex_reaction: Handle complex reactions.
        handle_complex_reaction_global_subgraph_isomorphism_matching: Annotate SoMs for complex reactions using subgraph isomorphism matching.
        handle_complex_reaction_largest_common_subgraph_matching: Annotate SoMs for complex reactions using largest common subgraph matching (maximum common substructure).
        handle_glutathione_conjugation: Annotate SoMs for glutathione conjugation.
        handle_halogen_to_hydroxy_oxidation: Annotate SoMs for halogen to hydroxy oxidation.
        handle_simple_addition_reaction: Annotate SoMs for simple addition reactions.
        handle_simple_elimination_addition: Annotate SoMs for simple elimination reactions.
        handle_redox_reaction: Annotate SoMs for redox reactions.
        has_equal_number_halogens: Check if substrate and metabolite have the same number of halogens.
        initialize_atom_notes: Initialize the atom note properties for the substrate and the metabolite.
        is_glutathione_conjugation: Check if the reaction is a glutathione conjugation.
        is_halogen_to_hydroxy_oxidation: Check if the reaction is a halogen to hydroxy oxidation.
        is_too_large_to_process: Check if the reaction is too large to process.
        log_and_return_soms: Return SoMs and annotation rule.
        log_initial_reaction_info: Log the initial reaction information.
    """

    def __init__(
        self,
        substrate: Tuple[Mol, int],
        metabolite: Tuple[Mol, int],
        logger_path: str,
        filter_size: int,
        ester_hydrolysis: bool,
    ):
        """Initialize the SOMFinder class."""
        substrate_mol, substrate_id = substrate
        metabolite_mol, metabolite_id = metabolite

        self.substrate = substrate_mol
        self.metabolite = metabolite_mol
        self.substrate_id = substrate_id
        self.metabolite_id = metabolite_id
        self.logger_path = logger_path
        self.filter_size = filter_size
        self.ester_hydrolysis = ester_hydrolysis
        self.params = self._initialize_mcs_params()
        self.mapping = {}
        self.soms = []
        self.reaction_type = "unknown"

    def _correct_acetal_hydrolysis(self) -> bool:
        """Correct SoMs for acetals if applicable."""
        acetal_pattern = MolFromSmarts("[C;X4](O[*])O[*]")
        exclusion_pattern = MolFromSmarts("[C;X4](OC(=O))O[*]")

        if not any(
            som in self.substrate.GetSubstructMatch(acetal_pattern) for som in self.soms
        ):
            return False
        if any(
            som in self.substrate.GetSubstructMatch(exclusion_pattern)
            for som in self.soms
        ):
            return False

        corrected_soms = [
            atom.GetIdx()
            for atom in self.substrate.GetAtoms()
            if atom.GetIdx()
            in self.substrate.GetSubstructMatch(MolFromSmarts("[C;X4](O)O"))
            and atom.GetAtomicNum() == 6
        ]
        if corrected_soms:
            self.soms = corrected_soms
            self.reaction_type = "simple elimination (acetal)"
            log(self.logger_path, "Acetal elimination detected. Corrected SoMs.")
            return True
        return False

    def _correct_carnitine_addition(self) -> bool:
        """Correct SoMs for the addition of carnitine to a carboxylic acid if applicable."""
        carnitine_pattern = MolFromSmarts("[N+](C)(C)(C)-C-C(O)C-C(=O)[O]")

        if (
            len(self.soms) != 1
            or len(self.metabolite.GetSubstructMatch(carnitine_pattern)) == 0
        ):
            return False

        atom_id_in_substrate = self.mapping[self.soms[0]]
        corrected_soms = [
            self.substrate.GetAtomWithIdx(atom_id_in_substrate)
            .GetNeighbors()[0]
            .GetIdx()
        ]
        if corrected_soms:
            self.soms = corrected_soms
            self.reaction_type = "simple addition (carnitine)"
            log(self.logger_path, "Carnitine addition detected. Corrected SoMs.")
            return True
        return False

    def _correct_cn_redox(self) -> bool:
        """
        Apply corrections if the redox reaction involves a C-N bond.

        Returns:
            bool: True if corrections were applied, False otherwise.
        """
        covered_atom_types = [
            self.substrate.GetAtomWithIdx(atom_id).GetAtomicNum()
            for atom_id in self.soms
        ]

        if 6 in covered_atom_types and 7 in covered_atom_types:
            self.soms = [
                atom_id
                for atom_id in self.soms
                if self.substrate.GetAtomWithIdx(atom_id).GetAtomicNum() == 6
            ]
            self.reaction_type = "redox (C-N bond)"
            return True

        return False

    def _correct_epoxide(self) -> bool:
        """Correct the SoMs for halogen to hydroxy oxidation if the reaction produces an epoxide instead of the typical alcohol."""
        som_atom_in_metabolite = self.metabolite.GetAtomWithIdx(
            self.mapping[self.soms[0]]
        )
        info = [
            (
                neighbor.GetIdx(),
                neighbor.GetSymbol(),
                "O"
                in [
                    superneighbor.GetSymbol()
                    for superneighbor in neighbor.GetNeighbors()
                ],
            )
            for neighbor in som_atom_in_metabolite.GetNeighbors()
        ]
        id_of_additional_som_atom_in_metabolite = [
            id
            for (id, symbol, has_oxygen_neighbor) in info
            if symbol == "C" and has_oxygen_neighbor
        ]
        if len(id_of_additional_som_atom_in_metabolite) == 1:
            # translate that atom id to the substrate
            id_of_additional_som_atom_in_substrate = self.mapping[
                id_of_additional_som_atom_in_metabolite[0]
            ]
            self.soms.append(id_of_additional_som_atom_in_substrate)
            return True
        return False

    def _correct_ester_hydrolysis(self) -> bool:
        """Correct SoMs for ester hydrolysis if applicable."""
        if not self.ester_hydrolysis:
            return False

        ester_pattern = MolFromSmarts("[*][C](=O)[O][*]")
        if not any(
            som in self.substrate.GetSubstructMatch(ester_pattern) for som in self.soms
        ):
            return False

        corrected_soms = [
            atom.GetIdx()
            for atom in self.substrate.GetAtoms()
            if atom.GetIdx() in self.substrate.GetSubstructMatch(ester_pattern)
            and atom.GetAtomicNum() == 6
            and self._has_single_and_double_bonded_oxygen(atom)
        ]
        if corrected_soms:
            self.soms = corrected_soms
            self.reaction_type = "simple elimination (ester hydrolysis)"
            log(self.logger_path, "Ester hydrolysis detected. Corrected SoMs.")
            return True
        return False

    def _correct_phosphate_hydrolysis(self) -> bool:
        """Correct SoMs for phosphate hydrolysis if applicable."""
        phosphate_derivate_pattern = MolFromSmarts("P(=O)")

        if not self.substrate.GetSubstructMatch(phosphate_derivate_pattern):
            return False

        som_atom = self.substrate.GetAtomWithIdx(self.soms[0])
        if (
            som_atom.GetSymbol() == "P"
        ):  # if the som is a phosphore atom, leave it as it is
            self.reaction_type = "simple elimination (phosphate-derivative hydrolysis)"
            log(
                self.logger_path,
                "Phosphate-derivative hydrolysis detected. Corrected SoMs.",
            )
            return True
        for neighbor in som_atom.GetNeighbors():
            if neighbor.GetSymbol() == "P":
                # if one of its neighbors is a phosphore atoms,
                # we have the case where a phosphore hydrolysis took place,
                # and the metabolite does **not** contain the phosphate functional group anymore
                self.soms = [neighbor.GetIdx()]
                self.reaction_type = (
                    "simple elimination (phosphate-derivative hydrolysis)"
                )
                log(
                    self.logger_path,
                    "Phosphate-derivative hydrolysis detected. Corrected SoMs.",
                )
                return True
            # for double_neighbor in neighbor.GetNeighbors():
            #     if double_neighbor.GetSymbol() == "P":
            #         # if one of its neighbors is a phosphore atoms,
            #         # we have the case where a phosphore hydrolysis took place,
            #         # and the metabolite contains the phosphate functional group
            #         self.soms = [double_neighbor.GetIdx()]
            #         self.reaction_type = "simple elimination (phosphate hydrolysis)"
            #         log(self.logger_path, "Phosphate hydrolysis detected. Corrected SoMs.")
            #         return True
        return False

    def _correct_piperazine_ring_hydroysis(self) -> bool:
        """Correct SoMs for piperazine ring opening if applicable."""
        piperazine_pattern = MolFromSmarts("N1CCNCC1")

        if not any(
            som in self.substrate.GetSubstructMatch(piperazine_pattern)
            for som in self.soms
        ):
            return False

        additional_soms = [
            neighbor.GetIdx()
            for som in self.soms
            if som in self.substrate.GetSubstructMatch(piperazine_pattern)
            for neighbor in self.substrate.GetAtomWithIdx(som).GetNeighbors()
            if neighbor.GetSymbol() == "C"
        ]
        if additional_soms:
            self.soms.extend(additional_soms)
            self.reaction_type = "simple elimination (piperazine ring opening)"
            log(self.logger_path, "Piperazine ring opening detected. Corrected SoMs.")
            return True
        return False

    def _correct_quinone_like_oxidation(self) -> bool:
        # TODO: This is very unelegant and prone to errors. We need to find a better way to do this.
        """Correct the SoMs for halogen to hydroxy oxidation if the reaction produces a quinone-like metabolite instead of the typical alcohol."""
        som_atom_in_metabolite = self.metabolite.GetAtomWithIdx(
            self.mapping[self.soms[0]]
        )
        # if the atom had a double bond to an oxygen atom
        bond_to_oxygen_atom_ids = [
            (som_atom_in_metabolite.GetIdx(), neighbor.GetIdx())
            for neighbor in som_atom_in_metabolite.GetNeighbors()
            if neighbor.GetSymbol() == "O"
        ]
        if len(bond_to_oxygen_atom_ids) == 1:
            if (
                get_bond_order(
                    self.metabolite,
                    bond_to_oxygen_atom_ids[0][0],
                    bond_to_oxygen_atom_ids[0][1],
                )
                == 2
            ):
                temp_atom_id = self.metabolite.GetSubstructMatch(
                    MolFromSmarts("C1=CCC=CC1=O")
                )[2]
                temp_atom_id_in_substrate = next(
                    (k for k, v in self.mapping.items() if v == temp_atom_id), None
                )  # we need to find the key of the value in the mapping
                temp_atom_in_substrate = self.substrate.GetAtomWithIdx(
                    temp_atom_id_in_substrate
                )
                temp_neighbor_idx = [
                    neighbor.GetIdx()
                    for neighbor in temp_atom_in_substrate.GetNeighbors()
                ]
                possible_soms = [
                    id
                    for id in temp_neighbor_idx
                    if self.substrate.GetAtomWithIdx(id).GetSymbol() != "C"
                ]
                if len(possible_soms) == 1:
                    self.soms.append(possible_soms[0])
                    return True
        return False

    def _correct_sulfur_derivatives_hydrolysis(self) -> bool:
        """Correct SoMs for the hydrolysis of sulfur derivatives (sulfamate, sulfonamide, sulfonate, sulfuric diamide etc.) if applicable."""
        if len(self.soms) != 1:
            return False

        sulfur_pattern = MolFromSmarts("[*][S](=O)(=O)[*]")
        exclusion_pattern = MolFromSmarts("[*]S(=O)(=O)NO")

        if not any(
            som in self.substrate.GetSubstructMatch(sulfur_pattern) for som in self.soms
        ):
            return False
        if any(
            som in self.substrate.GetSubstructMatch(exclusion_pattern)
            for som in self.soms
        ):
            return False

        self.soms = [
            atom.GetIdx()
            for atom in self.substrate.GetAtoms()
            if atom.GetSymbol() == "S"
        ]
        self.reaction_type = "simple elimination (sulfur-derivative hydrolysis)"
        log(self.logger_path, "Sulfur-derivative hydrolysis detected. Corrected SoMs.")
        return True

    def _has_single_and_double_bonded_oxygen(self, atom) -> bool:
        """Check if an atom has both single and double bonded oxygen neighbors."""
        neighbor_bonds = [
            neighbor.GetSymbol()
            + str(get_bond_order(self.substrate, atom.GetIdx(), neighbor.GetIdx()))
            for neighbor in atom.GetNeighbors()
        ]
        return "O1" in neighbor_bonds and "O2" in neighbor_bonds

    def _initialize_mcs_params(self):
        params = rdFMCS.MCSParameters()
        params.timeout = 10
        params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        params.BondTyper = rdFMCS.BondCompare.CompareOrder
        params.BondCompareParameters.CompleteRingsOnly = True
        params.BondCompareParameters.MatchFusedRings = True
        params.BondCompareParameters.MatchFusedRingsStrict = False
        params.BondCompareParameters.MatchStereo = False
        params.BondCompareParameters.RingMatchesRingOnly = True
        return params

    def _is_in_epoxide(self, som_id_in_metabolite: int) -> bool:
        """Check if the atom is in an epoxide."""
        epoxide_atom_ids = self.metabolite.GetSubstructMatch(MolFromSmarts("c1cO1"))
        if som_id_in_metabolite in epoxide_atom_ids:
            return True
        return False

    def _is_in_quinone(self, som_id_in_metabolite: int) -> bool:
        """Check if the atom is in a quinone-like structure."""
        quinone_like_atom_ids = self.metabolite.GetSubstructMatch(
            MolFromSmarts("C1=CCC=CC1=O")
        )
        if som_id_in_metabolite in quinone_like_atom_ids:
            return True
        return False

    def _find_unmapped_halogen(self) -> int:
        """Find the halogen atom in the substrate that is not present in the mapping."""
        halogen_symbols = ["F", "Cl", "Br", "I"]
        for atom in self.substrate.GetAtoms():
            # self.mapping maps the atom indices in the metabolite to the atom indices in the substrate ({id_s: id_m}):
            if (
                atom.GetSymbol() in halogen_symbols
                and atom.GetIdx() not in self.mapping.values()
            ):
                return atom
        return None

    def _find_unmatched_atoms(self, target: Mol, mcs) -> list:
        """Find unmatched atoms between the target and the query molecule."""
        return [
            atom
            for atom in target.GetAtoms()
            if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
        ]

    def _general_case_simple_addition(self, unmatched_atoms, query, mcs):
        """Identify SoMs in the simple addition case based on unmatched atoms."""
        for atom in unmatched_atoms:  # iterate over unmatched atoms
            for (
                neighbor
            ) in atom.GetNeighbors():  # iterate over neighbors of the unmatched atom
                if (
                    not neighbor.GetIdx() in self.mapping
                ):  # if the neighbor is not in the mapping, meaning it is not in the MCS...
                    continue  # ...skip the neighbor
                mapped_idx = self.mapping[
                    neighbor.GetIdx()
                ]  # get the index of the correct neighbor in the query molecule (substrate)
                if mapped_idx in query.GetSubstructMatch(
                    mcs.queryMol
                ):  # if the correct neighbor is in the query molecule (substrate)...
                    self.soms.append(
                        mapped_idx
                    )  # ...add the correct neighbor to the SoMs
                    self.reaction_type = "simple addition"

    def _general_case_simple_elimination(self, unmatched_atoms, target, mcs):
        """Identify SoMs in the simple elimination case based on unmatched atoms."""
        for atom in unmatched_atoms:  # iterate over unmatched atoms
            for (
                neighbor
            ) in atom.GetNeighbors():  # iterate over neighbors of the unmatched atom
                if neighbor.GetIdx() in target.GetSubstructMatch(
                    mcs.queryMol
                ):  # if the neighbor is in the MCS...
                    if (
                        atom.GetAtomicNum() != 6
                    ):  # ... and the unmatched atom is not a carbon atom...
                        self.soms.append(
                            neighbor.GetIdx()
                        )  # ...add the neighbor to the SoMs
                    else:  # if the unmatched atom is a carbon atom...
                        self.soms.append(
                            atom.GetIdx()
                        )  # ...add the unmatched atom to the SoMs
                    self.reaction_type = "simple elimination (general)"

    def _find_sulfur_index(self, glutathione_indices: list) -> int:
        """
        Find the sulfur atom index in the glutathione structure.

        Args:
            glutathione_indices (list): Indices of atoms in the glutathione moiety.

        Returns:
            int: Index of the sulfur atom, or None if not found.
        """
        return next(
            (
                idx
                for idx in glutathione_indices
                if self.metabolite.GetAtomWithIdx(idx).GetAtomicNum() == 16
            ),
            None,
        )

    def _find_sulfur_neighbor_index(
        self, s_index: int, glutathione_indices: list
    ) -> int:
        """
        Find the index of the atom neighboring the sulfur atom.

        Args:
            s_index (int): Index of the sulfur atom.
            glutathione_indices (list): Indices of atoms in the glutathione moiety.

        Returns:
            int: Index of the atom neighboring the sulfur atom, or None if not found.
        """
        s_neighbors = [
            neighbor.GetIdx()
            for neighbor in self.metabolite.GetAtomWithIdx(s_index).GetNeighbors()
        ]
        return next(
            (idx for idx in s_neighbors if idx not in glutathione_indices), None
        )

    def _get_non_glutathione_fragment(self, s_index: int, som_index: int):
        """
        Split the metabolite into fragments and identify the one without the glutathione moiety.

        Args:
            s_index (int): Index of the sulfur atom.
            som_index (int): Index of the SoM atom.

        Returns:
            Mol: The fragment that does not contain the glutathione moiety, or None if not found.
        """
        bond_id = self.metabolite.GetBondBetweenAtoms(s_index, som_index).GetIdx()
        fragments = GetMolFrags(
            FragmentOnBonds(self.metabolite, [bond_id], addDummies=False), asMols=True
        )
        glutathione_pattern = MolFromSmiles(
            "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N"
        )
        for fragment in fragments:
            if not fragment.HasSubstructMatch(glutathione_pattern):
                return fragment
        return None

    def _map_atoms_glutathione(self, source_mol, target_mol) -> dict:
        """
        Map atoms between two molecules using Maximum Common Substructure (MCS).

        Args:
            source_mol (Mol): The source molecule.
            target_mol (Mol): The target molecule.

        Returns:
            dict: A mapping between the atoms in the source molecule and the atoms in the target molecule.
        """
        self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareAny)
        mcs = rdFMCS.FindMCS([source_mol, target_mol], self.params)
        if not mcs or not mcs.queryMol:
            return False

        highlights_query = source_mol.GetSubstructMatch(mcs.queryMol)
        highlights_target = target_mol.GetSubstructMatch(mcs.queryMol)

        if not highlights_query or not highlights_target:
            return None

        return dict(zip(highlights_query, highlights_target))

    def _map_atoms(self, query, target, mcs):
        """Create mapping between query and target based on MCS."""
        highlights_query = query.GetSubstructMatch(mcs.queryMol)
        highlights_target = target.GetSubstructMatch(mcs.queryMol)
        if not highlights_query or not highlights_target:
            return False
        self.mapping = dict(zip(highlights_target, highlights_query))
        return True

    def _set_mcs_bond_typer_param(self, bond_typer_param):
        """Set the MCS bond compare parameter."""
        self.params.BondTyper = bond_typer_param

    def compute_weight_ratio(self) -> int:
        """Compute whether the substrate is lighter, heavier or equally heavy than the metabolite."""
        if self.substrate.GetNumHeavyAtoms() < self.metabolite.GetNumHeavyAtoms():
            log(self.logger_path, "Substrate lighter than metabolite.")
            return -1
        if self.substrate.GetNumHeavyAtoms() > self.metabolite.GetNumHeavyAtoms():
            log(self.logger_path, "Substrate heavier than the metabolite.")
            return 1
        else:
            return 0

    def handle_complex_reaction(self) -> bool:
        """Annotate SoMs for complex reactions."""
        log(self.logger_path, "Attempting global subgraph isomorphism matching.")
        if self.handle_complex_reaction_global_subgraph_isomorphism_matching():
            return True

        log(self.logger_path, "Attempting maximum common substructure (MCS) matching.")
        if self.handle_complex_reaction_largest_common_subgraph_matching():
            return True

        return False

    def handle_complex_reaction_global_subgraph_isomorphism_matching(
        self,
    ) -> bool:
        """
        Annotate SoMs for complex reactions using subgraph isomorphism matching.

        Returns:
            bool: True if a complex reaction is found, False otherwise.
        """
        mol_graph_substrate = mol_to_graph(self.substrate)
        mol_graph_metabolite = mol_to_graph(self.metabolite)

        # Check if the substrate is a subgraph of the metabolite or vice versa
        graph_matching = isomorphism.GraphMatcher(
            mol_graph_substrate,
            mol_graph_metabolite,
            node_match=isomorphism.categorical_node_match(["atomic_num"], [0]),
        )
        if graph_matching.is_isomorphic():
            log(
                self.logger_path,
                "Global graph matching found! Metabolite is an isomorphic subgraph of the substrate.",
            )
            self.mapping = graph_matching.mapping
            already_matched_metabolite_atom_indices = set(self.mapping.values())
        else:
            graph_matching = isomorphism.GraphMatcher(
                mol_graph_metabolite,
                mol_graph_substrate,
                node_match=isomorphism.categorical_node_match(["atomic_num"], [0]),
            )
            if graph_matching.is_isomorphic():
                log(
                    self.logger_path,
                    "Global graph matching found! Substrate is an isomorphic subgraph of the metabolite.",
                )
                self.mapping = graph_matching.mapping
                already_matched_metabolite_atom_indices = set(self.mapping.keys())
            else:
                log(self.logger_path, "No global graph matching found.")
                return False

        # Check if the mapping is complete
        for atom_s in self.substrate.GetAtoms():
            # Check that every atom in the substrate is matched to an atom in the metabolite
            # If no atom in the metabolite is assigned to the atom in the substrate,
            # assign the first available atom
            if self.mapping.get(atom_s.GetIdx()) is None:
                try:
                    first_of_remaining_unmapped_ids = [
                        i
                        for i in range(self.metabolite.GetNumHeavyAtoms())
                        if i not in already_matched_metabolite_atom_indices
                    ][0]
                    self.mapping[atom_s.GetIdx()] = first_of_remaining_unmapped_ids
                    already_matched_metabolite_atom_indices.add(
                        first_of_remaining_unmapped_ids
                    )
                except IndexError:
                    self.mapping[atom_s.GetIdx()] = -1

        # Compute the SoMs by comparing the degree and the atomic numbers of the atoms in the substrate and the metabolite
        # If 1.) the degree or 2.) the neighbors of the atoms in the substrate and the metabolite are different (in terms of atomic number),
        # or 3.) the number of bonded hydrogens is different, then the atom is a SoM.
        self.soms = [
            atom_id_s
            for atom_id_s, atom_id_m in self.mapping.items()
            if atom_id_m != -1
            and (
                self.substrate.GetAtomWithIdx(atom_id_s).GetDegree()
                != self.metabolite.GetAtomWithIdx(atom_id_m).GetDegree()
                or set(
                    neighbor.GetAtomicNum()
                    for neighbor in self.substrate.GetAtomWithIdx(
                        atom_id_s
                    ).GetNeighbors()
                )
                != set(
                    neighbor.GetAtomicNum()
                    for neighbor in self.metabolite.GetAtomWithIdx(
                        atom_id_m
                    ).GetNeighbors()
                )
                or self.substrate.GetAtomWithIdx(atom_id_s).GetTotalNumHs()
                != self.metabolite.GetAtomWithIdx(atom_id_m).GetTotalNumHs()
            )
        ]
        self.reaction_type = "complex (global graph matching)"
        return True

    def handle_complex_reaction_largest_common_subgraph_matching(
        self,
    ) -> bool:
        """
        Annotate SoMs for complex reactions using largest common subgraph matching (maximum common substructure).

        Returns:
            bool: True if a complex reaction is found, False otherwise.
        """
        params = self.params
        self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareAny)
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], params)

        if mcs.numAtoms > 0:
            highlights_substrate = self.substrate.GetSubstructMatch(mcs.queryMol)
            highlights_metabolite = self.metabolite.GetSubstructMatch(mcs.queryMol)
            self.mapping = dict(zip(highlights_substrate, highlights_metabolite))

            # Map the unmapped atoms in the substrate to the remaining atoms in the metabolite
            # based on the Tanimoto similarity of their Morgan fingerprints
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            atom_based_fp_metabolite = [
                mfpgen.GetFingerprint(self.metabolite, fromAtoms=[atom_m.GetIdx()])
                for atom_m in self.metabolite.GetAtoms()
            ]
            already_matched_atom_indices = set(highlights_metabolite)

            for atom_s in self.substrate.GetAtoms():
                if atom_s.GetIdx() not in highlights_substrate:
                    # Calculate the Morgan fingerprint for the atom in the substrate
                    atom_based_fp_substrate = mfpgen.GetFingerprint(
                        self.substrate,
                        fromAtoms=[atom_s.GetIdx()],
                    )
                    # Calculate the Tanimoto similarity between the atom in the substrate and all atoms in the metabolite
                    similarities = [
                        TanimotoSimilarity(
                            atom_based_fp_substrate,
                            atom_based_fp_metabolite[atom_m.GetIdx()],
                        )
                        for atom_m in self.metabolite.GetAtoms()
                    ]

                    # Assign the atom in the metabolite with the highest similarity to the atom in the substrate
                    for _ in range(self.metabolite.GetNumHeavyAtoms()):
                        if (self.mapping.get(atom_s.GetIdx()) is None) and (
                            np.argmax(similarities) not in already_matched_atom_indices
                        ):
                            self.mapping[atom_s.GetIdx()] = int(np.argmax(similarities))
                            already_matched_atom_indices.add(np.argmax(similarities))
                            break
                        similarities[np.argmax(similarities)] = 0

                    # If no atom in the metabolite is assigned to the atom in the substrate, assign the first available atom
                    if self.mapping.get(atom_s.GetIdx()) is None:
                        try:
                            first_of_remaining_unmapped_ids = [
                                i
                                for i in range(self.metabolite.GetNumHeavyAtoms())
                                if i not in already_matched_atom_indices
                            ][0]
                            self.mapping[
                                atom_s.GetIdx()
                            ] = first_of_remaining_unmapped_ids
                            already_matched_atom_indices.add(
                                first_of_remaining_unmapped_ids
                            )
                        except IndexError:
                            self.mapping[atom_s.GetIdx()] = -1

            # Compute the SoMs by comparing the degree and the atomic numbers of the atoms in the substrate and the metabolite
            # If 1.) the degree or 2.) the neighbors of the atoms in the substrate and the metabolite are different (in terms of atomic number),
            # or 3.) the number of bonded hydrogens is different, then the atom is a SoM.
            self.soms = [
                atom_id_s
                for atom_id_s, atom_id_m in self.mapping.items()
                if atom_id_m != -1
                and (
                    self.substrate.GetAtomWithIdx(atom_id_s).GetDegree()
                    != self.metabolite.GetAtomWithIdx(atom_id_m).GetDegree()
                    or set(
                        neighbor.GetAtomicNum()
                        for neighbor in self.substrate.GetAtomWithIdx(
                            atom_id_s
                        ).GetNeighbors()
                    )
                    != set(
                        neighbor.GetAtomicNum()
                        for neighbor in self.metabolite.GetAtomWithIdx(
                            atom_id_m
                        ).GetNeighbors()
                    )
                    or self.substrate.GetAtomWithIdx(atom_id_s).GetTotalNumHs()
                    != self.metabolite.GetAtomWithIdx(atom_id_m).GetTotalNumHs()
                )
            ]
            self.reaction_type = "complex (partial graph matching)"

            # Add an exception for the reduction of thiourea groups
            smarts_thiourea = "NC(=S)N"
            if self.substrate.HasSubstructMatch(MolFromSmarts(smarts_thiourea)):
                corrected_thiourea_soms = [
                    som
                    for som in self.soms
                    if self.substrate.GetAtomWithIdx(som).GetAtomicNum() == 16
                ]
                self.soms = corrected_thiourea_soms
                self.reaction_type = "complex (thiourea reduction)"
                log(self.logger_path, "Thiourea reduction detected. Corrected SoMs.")

            # Correct SoMs for the hydrolysis of oxacyclopropane rings
            smarts_oxycyclopropane = "C1OC1"
            if (
                len(
                    set(self.soms).intersection(
                        set(
                            self.substrate.GetSubstructMatch(
                                MolFromSmarts(smarts_oxycyclopropane)
                            )
                        )
                    )
                )
                > 0
            ):
                corrected_soms_oxacyclopropane = [
                    atom.GetIdx()
                    for atom in self.substrate.GetAtoms()
                    if (
                        atom.GetIdx()
                        in self.substrate.GetSubstructMatch(
                            MolFromSmarts(smarts_oxycyclopropane)
                        )
                    )
                    & (atom.GetAtomicNum() == 6)
                ]
                self.soms = corrected_soms_oxacyclopropane
                self.reaction_type = "complex (oxacyclopropane hydrolysis)"
                log(
                    self.logger_path,
                    "Oxacyclopropane hydrolysis detected. Corrected SoMs.",
                )
            return True
        return False

    def handle_glutathione_conjugation(self) -> bool:
        """
        Annotate SoMs for glutathione conjugation.

        Returns:
            bool: True if annotation is successful, False otherwise.
        """
        try:
            # Find the glutathione substructure in the metabolite
            glutathione_pattern = MolFromSmiles(
                "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N"
            )
            glutathione_indices = self.metabolite.GetSubstructMatch(glutathione_pattern)

            if glutathione_indices:
                # Identify the sulfur atom
                s_index = self._find_sulfur_index(glutathione_indices)
                if s_index is not None:
                    # Identify the neigbor of the sulfur atom that is not part of the glutathione moiety
                    s_neighbor_index = self._find_sulfur_neighbor_index(
                        s_index, glutathione_indices
                    )
                    if s_neighbor_index is not None:
                        # Get the fragment without the glutathione moiety
                        non_glutathione_fragment = self._get_non_glutathione_fragment(
                            s_index, s_neighbor_index
                        )
                        if non_glutathione_fragment is not None:
                            # Map atoms between the metabolite and the fragment
                            initial_mapping = self._map_atoms_glutathione(
                                self.metabolite, non_glutathione_fragment
                            )
                            if (
                                initial_mapping is not None
                                and s_neighbor_index in initial_mapping
                            ):
                                # Find the index of the SoM atom in the fragment
                                som_index_in_non_glutathione_fragment = initial_mapping[
                                    s_neighbor_index
                                ]
                                # Map the atoms between the fragment and the substrate
                                final_mapping = self._map_atoms_glutathione(
                                    non_glutathione_fragment, self.substrate
                                )
                                if (
                                    final_mapping is not None
                                    and som_index_in_non_glutathione_fragment
                                    in final_mapping
                                ):
                                    # Set the identified SoM and reaction type
                                    self.soms = [
                                        final_mapping[
                                            som_index_in_non_glutathione_fragment
                                        ]
                                    ]
                                    self.reaction_type = "glutathione conjugation"
                                    return True

        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path, f"Glutathione conjugation matching failed. Error: {e}"
            )

        return False

    def handle_halogen_to_hydroxy(self) -> bool:
        """
        Annotate SoMs for halogen to hydroxy oxidation.

        Returns:
            bool: True if annotation is successful, False otherwise.
        """
        try:
            query, target = self.substrate, self.metabolite

            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareAny)
            mcs = rdFMCS.FindMCS([query, target], self.params)

            if not self._map_atoms(query, target, mcs):
                return False

            # Find the halogen atom in the substrate that is not in the metabolite
            halogen_atom = self._find_unmapped_halogen()
            if halogen_atom is None:
                return False

            # The SoM is the neighbor of that halogen atom
            self.soms = [halogen_atom.GetNeighbors()[0].GetIdx()]
            self.reaction_type = "halogen to hydroxy oxidation"

            # If the reaction produces an epoxide (instead of the typical alcohol),
            # find the other atom that is part of the epoxide and add it to the SoMs
            if self._is_in_epoxide(self.mapping[self.soms[0]]):
                if self._correct_epoxide():
                    self.reaction_type = "halogen to epoxide oxidation"
                    return True

            if self._is_in_quinone(self.mapping[self.soms[0]]):
                if self._correct_quinone_like_oxidation():
                    self.reaction_type = "halogen to quinone-like oxidation"
                    return True

            return True
        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path,
                f"Halogen to hydroxy oxidation matching failed. Error: {e}",
            )
            return False

    def handle_simple_addition(self) -> bool:
        """
        Annotate SoMs for simple addition reactions.

        Returns:
            bool: True if a simple addition reaction is found, False otherwise.
        """
        log(self.logger_path, "Attempting simple addition matching.")

        if not self.metabolite.HasSubstructMatch(self.substrate):
            return False

        log(
            self.logger_path,
            "Susbtrate is a substructure of the metabolite.",
        )

        try:
            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.params)
            if not self._map_atoms(self.substrate, self.metabolite, mcs):
                return False
            unmatched_atoms = self._find_unmatched_atoms(self.metabolite, mcs)

            self._general_case_simple_addition(unmatched_atoms, self.substrate, mcs)

            if self._correct_carnitine_addition():
                return True

            return True
        except (ValueError, KeyError, AttributeError) as e:
            log(self.logger_path, f"Simple addition matching failed. Error: {e}")
            return False

    def handle_simple_elimination(self) -> bool:
        """
        Annotate SoMs for simple elimination reactions.

        Returns:
            bool: True if a simple elimination reaction is found, False otherwise.
        """
        log(self.logger_path, "Attempting simple elimination matching.")

        if not self.substrate.HasSubstructMatch(self.metabolite):
            return False

        log(
            self.logger_path,
            "Metabolite is a substructure the substrate.",
        )

        try:
            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            mcs = rdFMCS.FindMCS([self.metabolite, self.substrate], self.params)
            unmatched_atoms = self._find_unmatched_atoms(self.substrate, mcs)

            self._general_case_simple_elimination(unmatched_atoms, self.substrate, mcs)

            if self._correct_ester_hydrolysis():
                return True

            if self._correct_acetal_hydrolysis():
                return True

            if self._correct_phosphate_hydrolysis():
                return True

            if self._correct_sulfur_derivatives_hydrolysis():
                return True

            if self._correct_piperazine_ring_hydroysis():
                return True

            return True
        except (ValueError, KeyError, AttributeError) as e:
            log(self.logger_path, f"Simple elimination matching failed. Error: {e}")
            return False

    def handle_redox_reaction(self) -> bool:
        """Annotate SoMs for redox reactions.

        Returns:
            bool: True if a redox reaction is found, False otherwise.
        """
        try:
            log(self.logger_path, "Attempting redox reaction matching.")
            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.params)

            # Check if the MCS covers all but one heavy atom in the substrate
            if mcs.numAtoms != (self.substrate.GetNumHeavyAtoms() - 1):
                log(self.logger_path, "Not a redox.")
                return False

            unmatched_atoms = self._find_unmatched_atoms(self.substrate, mcs)

            for atom in unmatched_atoms:
                for neighbor in atom.GetNeighbors():
                    # Skip if the neighbor is not in the MCS
                    if not neighbor.GetIdx() in self.metabolite.GetSubstructMatch(
                        mcs.queryMol
                    ):
                        continue

                    # Annotate redox reaction sites
                    self.soms.extend([atom.GetIdx(), neighbor.GetIdx()])
                    self.reaction_type = "redox"

                    # Apply corrections for C-N bond redox reactions
                    if self._correct_cn_redox():
                        log(
                            self.logger_path,
                            "C-N redox reaction detected. Corrected SoMs.",
                        )

            if len(self.soms) != 0:
                return True

            log(self.logger_path, "Not a redox reaction.")
            return False
        except (ValueError, KeyError, AttributeError) as e:
            log(self.logger_path, f"Redox reaction matching failed. Error: {e}")
            return False

    def has_equal_number_halogens(self) -> bool:
        """Check if substrate and metabolite have the same number of halogens."""
        halogen_atomic_nums = {9, 17, 35, 53}  # Atomic numbers for F, Cl, Br, I

        num_halogens_substrate = sum(
            atom.GetAtomicNum() in halogen_atomic_nums
            for atom in self.substrate.GetAtoms()
        )
        num_halogens_metabolite = sum(
            atom.GetAtomicNum() in halogen_atomic_nums
            for atom in self.metabolite.GetAtoms()
        )

        if num_halogens_substrate == num_halogens_metabolite:
            log(
                self.logger_path,
                "Substrate and metabolite have the same number of halogen atoms.",
            )
            return True
        return False

    def initialize_atom_notes(self) -> None:
        """Initialize the atom note properties for the substrate and the metabolite."""
        for atom in self.substrate.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())
        for atom in self.metabolite.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())

    def is_glutathione_conjugation(self) -> bool:
        """Check if the reaction is a glutathione conjugation."""
        glutathione_smiles = "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N"
        if self.metabolite.HasSubstructMatch(
            MolFromSmiles(glutathione_smiles)
        ) and not self.substrate.HasSubstructMatch(MolFromSmiles(glutathione_smiles)):
            log(self.logger_path, "Glutathione conjugation detected.")
            return True
        return False

    def is_halogen_to_hydroxy(self) -> bool:
        """Check if the reaction is an oxidation of a halogen to a hydroxy/epoxide functional group."""
        substrate_elements = count_elements(self.substrate)
        metabolite_elements = count_elements(self.metabolite)

        if (
            is_carbon_count_unchanged(substrate_elements, metabolite_elements)
            and is_halogen_count_decreased(substrate_elements, metabolite_elements)
            and is_oxygen_count_increased(substrate_elements, metabolite_elements)
        ):
            log(self.logger_path, "Halogen to hydroxy oxidation detected.")
            return True
        return False

    def is_too_large_to_process(self) -> bool:
        """Check if the substrate or metabolite is too large for further processing."""
        if (
            self.substrate.GetNumHeavyAtoms() > self.filter_size
            or self.metabolite.GetNumHeavyAtoms() > self.filter_size
        ):
            log(
                self.logger_path,
                "Substrate or metabolite too large for processing.",
            )
            self.reaction_type = "maximum number of heavy atoms filter"
            return True
        return False

    def log_and_return_soms(self) -> Tuple[List[int], str]:
        """Return SoMs and annotation rule."""
        if self.reaction_type == "unknown" or self.reaction_type == "maximum number of heavy atoms filter":
            log(self.logger_path, "SOMAN is unable to annotate SoMs.")
        log(self.logger_path, f"{self.reaction_type.capitalize()} successful.")
        return sorted(self.soms), self.reaction_type

    def log_initial_reaction_info(self) -> None:
        """Log the initial reaction information."""
        log(
            self.logger_path,
            f"Substrate ID: {self.substrate_id}, Metabolite ID: {self.metabolite_id}",
        )


def annotate_soms(
    substrate: Tuple[Mol, int],
    metabolite: Tuple[Mol, int],
    logger_path: str,
    filter_size: int,
    ester_hydrolysis: bool,
) -> Callable[[], Tuple[List[int], str]]:
    """Annotates the SoMs for a given substrate-metabolite pair."""
    som_finder = Annotator(
        substrate,
        metabolite,
        logger_path,
        filter_size=filter_size,
        ester_hydrolysis=ester_hydrolysis,
    )

    som_finder.initialize_atom_notes()
    som_finder.log_initial_reaction_info()

    if som_finder.is_glutathione_conjugation():
        if som_finder.handle_glutathione_conjugation():
            return som_finder.log_and_return_soms()

    if som_finder.is_halogen_to_hydroxy():
        if som_finder.handle_halogen_to_hydroxy():
            return som_finder.log_and_return_soms()

    weight_ratio = som_finder.compute_weight_ratio()

    if weight_ratio == 1:
        if som_finder.handle_simple_addition():
            return som_finder.log_and_return_soms()

    if weight_ratio == -1:
        if som_finder.handle_simple_elimination():
            return som_finder.log_and_return_soms()

    # The next steps rely more heavily on MCS matching,
    # which can take very long for large molecules,
    # so we skip them if the substrate or metabolite is too large.
    if som_finder.is_too_large_to_process():
        return som_finder.log_and_return_soms()

    if weight_ratio == 0 and som_finder.has_equal_number_halogens():
        if som_finder.handle_redox_reaction():
            return som_finder.log_and_return_soms()

    if som_finder.handle_complex_reaction():
        return som_finder.log_and_return_soms()

    return som_finder.log_and_return_soms()
