"""
This module provides functionality for identifying Sites of Metabolism (SoMs) in chemical reactions using RDKit and NetworkX libraries. 

Classes:
    SOMFinder: A class for finding and annotating Sites of Metabolism (SoMs) in chemical reactions.

Methods (SOMFinder):
    _initialize_atom_notes() -> None:
        Initialize the atom note properties for the substrate and the metabolite.
    _handle_glutathione_conjugation() -> bool:
        Annotate SoMs for glutathione conjugation.
    _handle_halogen_to_hydroxy() -> bool:
        Annotate SoMs for halogen to hydroxy oxidations.
    _handle_simple_addition() -> bool:
        Annotate SoMs for simple addition reactions.
    _handle_simple_elimination() -> bool:
        Annotate SoMs for simple elimination reactions.
    _handle_redox_reaction() -> bool:
        Annotate SoMs for redox reactions.
    _handle_complex_non_redox_reaction_global_subgraph_isomorphism_matching() -> bool:
        Annotate SoMs for complex non-redox reactions using subgraph isomorphism matching.
    _handle_complex_non_redox_reaction_largest_common_subgraph_matching() -> bool:
        Annotate SoMs for complex non-redox reactions using largest common subgraph matching.
    get_soms() -> List[int]:
        Returns the Sites of Metabolism (SoMs) associated with a biochemical reaction.
"""


from typing import List

import numpy as np
from networkx.algorithms import isomorphism
from rdkit.Chem import (
    FragmentOnBonds,
    GetMolFrags,
    MolFromSmarts,
    MolFromSmiles,
    # MolToSmiles,
    rdFingerprintGenerator,
    rdFMCS,
)
from rdkit.DataStructs import TanimotoSimilarity

from src.utils import (
    detect_halogen_to_hydroxy,
    equal_number_halogens,
    log,
    mol_to_graph,
)


class SOMFinder:
    """
    SOMFinder is a class designed to identify Sites of Metabolism (SoMs) in chemical reactions.

    Attributes:
        substrate (RDKit Mol): The substrate molecule.
        metabolite (RDKit Mol): The metabolite molecule.
        substrate_id (int): Identifier for the substrate.
        metabolite_id (int): Identifier for the metabolite.
        logger_path (str): Path to the log file.
        filter_size (int): Size of the filter used in the analysis.
        params (rdFMCS.MCSParameters): Parameters for the Maximum Common Substructure (MCS) search.
        soms (List[int]): List of identified Sites of Metabolism (SoMs).
        mapping (dict): Mapping of atom indices between substrate and metabolite.

    Methods:
        _initialize_mcs_params(): Initialize the MCS parameters.
        detect_halogen_to_hydroxy(substrate: Mol, metabolite: Mol) -> bool: Detects reactions consisting in the oxidation of a halogen to a hydroxy group.
        log(message: str) -> None: Log a message to a text file.
        _initialize_atom_notes() -> None: Initialize the atom note properties for the substrate and the metabolite.
        _handle_glutathione_conjugation() -> bool: Annotate SoMs for glutathione conjugation.
        _handle_halogen_to_hydroxy() -> bool: Annotate SoMs for halogen to hydroxy oxidations.
        _handle_simple_addition() -> bool: Annotate SoMs for simple addition reactions.
        _handle_simple_elimination() -> bool: Annotate SoMs for simple elimination reactions.
        _handle_redox_reaction() -> bool: Annotate SoMs for redox reactions.
    """

    def __init__(
        self,
        substrate,
        metabolite,
        substrate_id,
        metabolite_id,
        logger_path,
        filter_size,
    ):
        self.substrate = substrate
        self.metabolite = metabolite
        self.substrate_id = substrate_id
        self.metabolite_id = metabolite_id
        self.logger_path = logger_path
        self.filter_size = filter_size
        self.params = self._initialize_mcs_params()
        self.soms = []
        self.mapping = {}

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

    def _initialize_atom_notes(self) -> None:
        """
        Initialize the atom note properties for the substrate and the metabolite.
        """
        for atom in self.substrate.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())
        for atom in self.metabolite.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())

    def _is_glutathione_conjugation(self) -> bool:
        """Check if the reaction is a glutathione conjugation."""
        glutathione_smiles = "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N"
        return self.metabolite.HasSubstructMatch(
            MolFromSmiles(glutathione_smiles)
        ) and not self.substrate.HasSubstructMatch(MolFromSmiles(glutathione_smiles))

    def _is_too_large_to_process(self) -> bool:
        """Check if the substrate or metabolite is too large for further processing."""
        if (
            self.substrate.GetNumHeavyAtoms() > self.filter_size
            or self.metabolite.GetNumHeavyAtoms() > self.filter_size
        ):
            log(
                self.logger_path,
                "Substrate or metabolite too large for processing. No SoMs found.",
            )
            return True
        return False

    def _handle_glutathione_conjugation(self) -> bool:
        """
        Annotate SoMs for glutathione conjugation.

        Returns:
            bool: True if annotation is successful, False otherwise.
        """
        try:
            glutathione_atom_idx = self.metabolite.GetSubstructMatch(
                MolFromSmiles("C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N")
            )
            if len(glutathione_atom_idx) > 0:
                # Find the index of the sulfur atom of the glutathione moiety in the metabolite
                s_index = [
                    atom_id
                    for atom_id in glutathione_atom_idx
                    if self.metabolite.GetAtomWithIdx(atom_id).GetAtomicNum() == 16
                ][0]
                # Find the indices of the neighbors of the sulfur atom in the metabolite
                s_neighbors_idx = [
                    neighbor.GetIdx()
                    for neighbor in self.metabolite.GetAtomWithIdx(
                        s_index
                    ).GetNeighbors()
                ]
                # Find the index of the neighbor of the sulfur atom that is not in the glutathione moiety
                som_idx_in_metabolite = [
                    neighbor
                    for neighbor in s_neighbors_idx
                    if neighbor not in glutathione_atom_idx
                ][0]
                # Get bond id between the sulfur atom and the atom that was the som in the metabolite
                bond_id = self.metabolite.GetBondBetweenAtoms(
                    s_index, som_idx_in_metabolite
                ).GetIdx()
                # Split the metabolite into two fragements along the bond between the sulfur atom and the atom that was the som
                fragments = GetMolFrags(
                    FragmentOnBonds(self.metabolite, [bond_id], addDummies=False),
                    asMols=True,
                )
                # Find the fragment that does not contain the glutathione moiety
                non_glutathione_fragment = [
                    fragment
                    for fragment in fragments
                    if len(
                        fragment.GetSubstructMatch(
                            MolFromSmiles(
                                "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N"
                            )
                        )
                    )
                    == 0
                ][0]
                # Find a mapping between the atoms in the non-glutathione fragment and the atoms in the metabolite
                mcs = rdFMCS.FindMCS(
                    [self.metabolite, non_glutathione_fragment], self.params
                )
                highlights_query = self.metabolite.GetSubstructMatch(mcs.queryMol)
                highlights_target = non_glutathione_fragment.GetSubstructMatch(
                    mcs.queryMol
                )
                self.mapping = dict(zip(highlights_query, highlights_target))
                # Find the index of the som_idx_in_metabolite in the fragment that does not contain the glutathione moiety
                som_idx_in_fragment = [self.mapping[som_idx_in_metabolite]][0]
                # Find a mapping between the atoms in the non-glutathione fragment and the atoms in the substrate
                self.params.BondTyper = (
                    rdFMCS.BondCompare.CompareAny
                )  # Allow any bond type to be matched
                mcs = rdFMCS.FindMCS(
                    [self.substrate, non_glutathione_fragment], self.params
                )
                self.params.BondTyper = (
                    rdFMCS.BondCompare.CompareOrder
                )  # Reset the bond type comparison to the default
                highlights_query = non_glutathione_fragment.GetSubstructMatch(
                    mcs.queryMol
                )
                highlights_target = self.substrate.GetSubstructMatch(mcs.queryMol)
                self.mapping = dict(zip(highlights_query, highlights_target))
                # Find the index of the som_idx_in_fragment in the substrate
                self.soms = [self.mapping[som_idx_in_fragment]]
                log(self.logger_path, "Glutathione conjugation successful.")
                return True
            return False
        except (ValueError, KeyError, AttributeError) as e:
            log(self.logger_path, f"Glutathione conjugation failed. Error: {str(e)}")
            return False

    def _handle_halogen_to_hydroxy(self) -> bool:
        """
        Annotate SoMs for halogen to hydroxy oxidations.

        Returns:
            bool: True if annotation is successful, False otherwise.
        """
        try:
            # Match the indices of the atoms in the substrate and the metabolite
            # based on their MCS match
            self.params.BondTyper = (
                rdFMCS.BondCompare.CompareAny
            )  # Allow any bond type to be matched
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.params)
            self.params.BondTyper = (
                rdFMCS.BondCompare.CompareOrder
            )  # Reset the bond type comparison to the default
            highlights_substrate = self.substrate.GetSubstructMatch(mcs.queryMol)
            highlights_metabolite = self.metabolite.GetSubstructMatch(mcs.queryMol)
            self.mapping = dict(zip(highlights_substrate, highlights_metabolite))
            # Find the halogen atom in the substrate that is not in the metabolite
            hal_atom_ids = [
                atom.GetIdx()
                for atom in self.substrate.GetAtoms()
                if atom.GetSymbol() in ["F", "Cl", "Br", "I"]
            ]
            for atom_id in hal_atom_ids:
                if atom_id not in self.mapping:
                    halogen_atom_id = atom_id
                    break
            # Find the index of the neighbor of that halogen atom
            halogen_atom_neighbor_id = (
                self.substrate.GetAtomWithIdx(halogen_atom_id)
                .GetNeighbors()[0]
                .GetIdx()
            )
            self.soms = [halogen_atom_neighbor_id]
            log(self.logger_path, "Halogen to hydroxy oxidation successful.")
            return True
        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path,
                f"Halogen to hydroxy oxidation matching failed. Error: {str(e)}",
            )
            return False

    def _handle_simple_addition(self) -> bool:
        """
        Annotate SoMs for simple addition reactions.

        Returns:
            bool: True if a simple addition reaction is found, False otherwise.
        """
        query, target = self.substrate, self.metabolite
        if target.HasSubstructMatch(query):
            log(
                self.logger_path,
                "Query (substrate) is a substructure the target (metabolite). Looking for a match...",
            )
            try:
                mcs = rdFMCS.FindMCS([query, target], self.params)
                highlights_query = query.GetSubstructMatch(mcs.queryMol)
                highlights_target = target.GetSubstructMatch(mcs.queryMol)
                self.mapping = dict(zip(highlights_target, highlights_query))
                unmatched_atoms = [
                    atom
                    for atom in target.GetAtoms()
                    if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
                ]
                for atom in unmatched_atoms:
                    neighbors = atom.GetNeighbors()
                    for neighbor in neighbors:
                        if not neighbor.GetIdx() in self.mapping:
                            continue
                        if self.mapping[neighbor.GetIdx()] in query.GetSubstructMatch(
                            mcs.queryMol
                        ):
                            self.soms.append(self.mapping[neighbor.GetIdx()])
                log(self.logger_path, "Simple addition successful.")

                # Correct SoMs for the addition of carnitine to a carboxylic acid
                if (
                    len(self.soms) == 1
                    and len(
                        self.metabolite.GetSubstructMatch(
                            MolFromSmarts("[N+](C)(C)(C)-C-C(O)C-C(=O)[O]")
                        )
                    )
                    > 0
                ):
                    atom_id_in_substrate = self.mapping[self.soms[0]]
                    corrected_soms_carnitine = [
                        self.substrate.GetAtomWithIdx(atom_id_in_substrate)
                        .GetNeighbors()[0]
                        .GetIdx()
                    ]
                    self.soms = corrected_soms_carnitine
                    log(
                        self.logger_path, "Carnitine addition detected. Corrected SoMs."
                    )

                return True
            except (ValueError, KeyError, AttributeError) as e:
                log(
                    self.logger_path,
                    f"Simple addition matching failed. Error: {str(e)}",
                )
                return False
        return False

    def _handle_simple_elimination(self) -> bool:
        """
        Annotate SoMs for simple elimination reactions.

        Returns:
            bool: True if a simple elimination reaction is found, False otherwise.
        """
        query, target = self.metabolite, self.substrate
        if target.HasSubstructMatch(query):
            log(
                self.logger_path,
                "Query (metabolite) is a substructure the target (substrate). Looking for a match...",
            )
            try:
                # Add exception for reactions containing a phosphore atom
                if (
                    len(self.substrate.GetSubstructMatch(MolFromSmarts("[P](=O)"))) > 0
                    or len(self.substrate.GetSubstructMatch(MolFromSmarts("[P](=S)")))
                    > 0
                ):
                    self.soms = [
                        atom.GetIdx()
                        for atom in self.substrate.GetAtoms()
                        if atom.GetAtomicNum() == 15
                    ]
                    log(
                        self.logger_path,
                        "Phosphore atom detected. SoM is the phosphore atom.",
                    )
                    return True

                mcs = rdFMCS.FindMCS([query, target], self.params)
                unmatched_atoms = [
                    atom
                    for atom in target.GetAtoms()
                    if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
                ]
                for atom in unmatched_atoms:
                    neighbors = atom.GetNeighbors()
                    for neighbor in neighbors:
                        if neighbor.GetIdx() in target.GetSubstructMatch(mcs.queryMol):
                            if atom.GetAtomicNum() != 6:
                                self.soms.append(neighbor.GetIdx())
                            else:
                                self.soms.append(atom.GetIdx())

                # Add exception for acetals
                if (
                    len(
                        set(self.soms).intersection(
                            set(
                                self.substrate.GetSubstructMatch(
                                    MolFromSmarts("[C;X4](O[*])O[*]")
                                )
                            )
                        )
                    )
                    > 0
                ):
                    corrected_soms_acetal = [
                        atom.GetIdx()
                        for atom in self.substrate.GetAtoms()
                        if (
                            atom.GetIdx()
                            in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[C;X4](O)O")
                            )
                        )
                        & (atom.GetAtomicNum() == 6)
                    ]
                    self.soms = corrected_soms_acetal
                    log(
                        self.logger_path, "Acetal elimination detected. Corrected SoMs."
                    )

                # Correct SoMs for ester hydrolysis
                if (
                    len(
                        set(self.soms).intersection(
                            set(
                                self.substrate.GetSubstructMatch(
                                    MolFromSmarts("[*][C](=O)[O][*]")
                                )
                            )
                        )
                    )
                    > 0
                ):
                    corrected_soms_ester = [
                        atom.GetIdx()
                        for atom in self.substrate.GetAtoms()
                        if (
                            atom.GetIdx()
                            in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[*][C](=O)[O][*]")
                            )
                        )
                        & (
                            atom.GetIdx()
                            in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[C](=O)")
                            )
                        )
                        & (atom.GetAtomicNum() == 6)
                    ]
                    self.soms = corrected_soms_ester
                    log(self.logger_path, "Ester hydrolysis detected. Corrected SoMs.")

                # Correct SoMs for the hydrolysis of sulfonamines
                if len(self.soms) == 1:
                    som = self.soms[0]
                    if (
                        som
                        in self.substrate.GetSubstructMatch(
                            MolFromSmarts("[S](=O)(=O)[N]")
                        )
                    ) or (
                        som
                        in self.substrate.GetSubstructMatch(
                            MolFromSmarts("[N][S](=O)(=O)[N]")
                        )
                    ):
                        if (
                            som
                            not in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[N][S](=O)(=O)[O]")
                            )
                        ) and (
                            som
                            not in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[S](=O)(=O)[N][O]")
                            )
                        ):
                            self.soms = [
                                atom.GetIdx()
                                for atom in self.substrate.GetAtoms()
                                if atom.GetSymbol() == "S"
                            ]
                            log(
                                self.logger_path,
                                "Sulfonamine hydrolysis detected. Corrected SoMs.",
                            )

                # Correct SoMs for the hydrolysis of sulfones
                if len(self.soms) == 1:
                    som = self.soms[0]
                    if som in self.substrate.GetSubstructMatch(
                        MolFromSmarts("[*][*][S](=O)(=O)[*]")
                    ):
                        self.soms = [
                            atom.GetIdx()
                            for atom in self.substrate.GetAtoms()
                            if atom.GetSymbol() == "S"
                        ]
                        log(
                            self.logger_path,
                            "Sulfone hydrolysis detected. Corrected SoMs.",
                        )

                # Correct SoMs on piperazine ring opening
                if (
                    len(
                        set(self.soms).intersection(
                            set(
                                self.substrate.GetSubstructMatch(
                                    MolFromSmarts("N1CCNCC1")
                                )
                            )
                        )
                    )
                    != 0
                ):
                    additional_soms = []
                    for som in self.soms:
                        if som in self.substrate.GetSubstructMatch(
                            MolFromSmarts("N1CCNCC1")
                        ):
                            neighbors = self.substrate.GetAtomWithIdx(
                                som
                            ).GetNeighbors()
                            for neighbor in neighbors:
                                if neighbor.GetSymbol() == "C":
                                    additional_soms.append(neighbor.GetIdx())
                    self.soms.extend(additional_soms)
                    log(
                        self.logger_path,
                        "Piperazine ring opening detected. Corrected SoMs.",
                    )

                # Correct SoMs on morpholine ring opening
                if (
                    len(
                        set(self.soms).intersection(
                            set(
                                self.substrate.GetSubstructMatch(
                                    MolFromSmarts("O1CCNCC1")
                                )
                            )
                        )
                    )
                    != 0
                ):
                    additional_soms = []
                    for som in self.soms:
                        if som in self.substrate.GetSubstructMatch(
                            MolFromSmarts("O1CCNCC1")
                        ):
                            neighbors = self.substrate.GetAtomWithIdx(
                                som
                            ).GetNeighbors()
                            for neighbor in neighbors:
                                if neighbor.GetSymbol() == "C":
                                    additional_soms.append(neighbor.GetIdx())
                    self.soms.extend(additional_soms)
                    log(
                        self.logger_path,
                        "Morpholine ring opening detected. Corrected SoMs.",
                    )

                # Correct SoMs for the hydrolysis of 1,3-dioxolane rings
                matches = self.substrate.GetSubstructMatch(MolFromSmarts("O1COcc1"))
                for match in matches:
                    if match in self.soms:
                        corrected_soms = [
                            atom.GetIdx()
                            for atom in self.substrate.GetAtoms()
                            if (
                                atom.GetIdx()
                                in self.substrate.GetSubstructMatch(
                                    MolFromSmarts("O1COcc1")
                                )
                            )
                            & (atom.GetAtomicNum() == 6)
                            & (atom.GetIsAromatic() is False)
                        ]
                        self.soms = corrected_soms
                        log(
                            self.logger_path,
                            "1,3-dioxolane ring opening detected. Corrected SoMs.",
                        )

                log(self.logger_path, "Simple elimination successful.")
                return True
            except (ValueError, KeyError, AttributeError) as e:
                log(
                    self.logger_path,
                    f"Simple elimination matching failed. Error: {str(e)}",
                )
                return False
        return False

    def _handle_redox_reaction(self) -> bool:
        """
        Annotate SoMs for redox reactions.

        Returns:
            bool: True if a redox reaction is found, False otherwise.
        """
        log(self.logger_path, "Checking for MCS matching...")
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.params)
        if mcs.numAtoms > 0:
            if mcs.numAtoms == (self.substrate.GetNumHeavyAtoms() - 1):
                for atom in self.substrate.GetAtoms():
                    if atom.GetIdx() not in self.substrate.GetSubstructMatch(
                        mcs.queryMol
                    ):
                        neighbors = atom.GetNeighbors()
                        for neighbor in neighbors:
                            try:
                                if (
                                    neighbor.GetIdx()
                                    in self.metabolite.GetSubstructMatch(mcs.queryMol)
                                ):
                                    log(self.logger_path, "Redox matching successful.")
                                    self.soms.append(neighbor.GetIdx())
                                    if atom.GetAtomicNum() not in [9, 17, 35, 53]:
                                        log(
                                            self.logger_path,
                                            "Redox reaction of non-halogen bond.",
                                        )
                                        self.soms.append(atom.GetIdx())
                                    else:
                                        log(
                                            self.logger_path,
                                            "Redox reaction of halogen bond.",
                                        )

                                    # Add a correction for C-N bond redox reactions (only the carbon is the som)
                                    covered_atom_types = [
                                        self.substrate.GetAtomWithIdx(
                                            atom_id
                                        ).GetAtomicNum()
                                        for atom_id in self.soms
                                    ]
                                    if (
                                        6 in covered_atom_types
                                        and 7 in covered_atom_types
                                    ):
                                        corrected_soms_cn_redox = [
                                            atom_id
                                            for atom_id in self.soms
                                            if self.substrate.GetAtomWithIdx(
                                                atom_id
                                            ).GetAtomicNum()
                                            == 6
                                        ]
                                        self.soms = corrected_soms_cn_redox
                                        log(
                                            self.logger_path,
                                            "C-N redox reaction detected. Corrected SoMs.",
                                        )

                                    return True
                            except KeyError:
                                log(self.logger_path, "Redox matching failed.")
                                return False
            else:
                log(self.logger_path, "Redox matching failed.")
        return False

    def _handle_complex_non_redox_reaction_global_subgraph_isomorphism_matching(
        self,
    ) -> bool:
        """
        Annotate SoMs for complex non-redox reactions using subgraph isomorphism matching.

        Returns:
            bool: True if a complex non-redox reaction is found, False otherwise.
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
        log(self.logger_path, "Global graph matching successful.")
        return True

    def _handle_complex_non_redox_reaction_largest_common_subgraph_matching(
        self,
    ) -> bool:
        """
        Annotate SoMs for complex non-redox reactions using largest common subgraph matching (maximum common substructure).

        Returns:
            bool: True if a complex non-redox reaction is found, False otherwise.
        """

        log(self.logger_path, "Checking for partial graph matching")

        params = self.params
        params.BondTyper = rdFMCS.BondCompare.CompareAny
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], params)

        if mcs.numAtoms > 0:
            log(self.logger_path, "MCS matching found!")
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

            # Add an exception for the reduction of nitro groups
            if len(self.soms) == 3:
                if set(
                    self.substrate.GetAtomWithIdx(som).GetAtomicNum()
                    for som in self.soms
                ) == {6, 7, 8}:
                    corrected_nitro_soms = [
                        som
                        for som in self.soms
                        if self.substrate.GetAtomWithIdx(som).GetAtomicNum() == 7
                    ]
                    self.soms = corrected_nitro_soms
                    log(self.logger_path, "Nitro reduction detected. Corrected SoMs.")

            # Add an exception for the reduction of thiourea groups
            smarts_thiourea = "NC(=S)N"
            if self.substrate.HasSubstructMatch(MolFromSmarts(smarts_thiourea)):
                corrected_thiourea_soms = [
                    som
                    for som in self.soms
                    if self.substrate.GetAtomWithIdx(som).GetAtomicNum() == 16
                ]
                self.soms = corrected_thiourea_soms
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
                log(
                    self.logger_path,
                    "Oxacyclopropane hydrolysis detected. Corrected SoMs.",
                )
            log(self.logger_path, "MCS matching successful.")
            return True
        log(self.logger_path, "MCS matching failed.")
        return False

    def _log_initial_reaction_info(self) -> None:
        """Logs the initial reaction information."""
        log(
            self.logger_path,
            f"Substrate ID: {self.substrate_id}, Metabolite ID: {self.metabolite_id}",
        )

    def _handle_and_return_soms(self, handler_method, reaction_type: str) -> List[int]:
        """Try to handle a specific reaction type and return SoMs if successful."""
        log(self.logger_path, f"{reaction_type} detected.")
        if handler_method():
            return sorted(self.soms)
        log(self.logger_path, f"{reaction_type} matching failed.")
        return []

    def _handle_failure(self) -> List[int]:
        """Handle the case where no specific reaction type was detected."""
        log(self.logger_path, "SoM matching failed.")
        return sorted(self.soms)

    def _handle_complex_reaction(self) -> List[int]:
        """Handle reactions where the substrate and metabolite have an equal number of halogens."""
        log(self.logger_path, "Complex reaction detected. Possibly a redox reaction.")
        if self._is_too_large_to_process():
            return []

        if self._handle_redox_reaction():
            return sorted(self.soms)

        log(self.logger_path, "No redox reaction detected.")
        return self._handle_complex_non_redox_case()

    def _handle_complex_non_redox_case(self) -> List[int]:
        """Handle complex non-redox reactions."""
        log(self.logger_path, "Attempting global subgraph isomorphism matching.")
        if (
            self._handle_complex_non_redox_reaction_global_subgraph_isomorphism_matching()
        ):
            return sorted(self.soms)

        log(self.logger_path, "Attempting maximum common substructure (MCS) matching.")
        if self._is_too_large_to_process():
            return []

        if self._handle_complex_non_redox_reaction_largest_common_subgraph_matching():
            return sorted(self.soms)

        return self._handle_failure()

    def get_soms(self) -> List[int]:
        """
        Returns the Sites of Metabolism (SoMs) associated with a biochemical reaction.

        Returns:
            soms (List[int]): List of SoMs.
        """
        # self.substrate = MolFromSmiles(
        #     MolToSmiles(self.substrate, isomericSmiles=False)
        # )
        # self.metabolite = MolFromSmiles(
        #     MolToSmiles(self.metabolite, isomericSmiles=False)
        # )

        self._initialize_atom_notes()
        self._log_initial_reaction_info()

        if self._is_glutathione_conjugation():
            return self._handle_and_return_soms(
                self._handle_glutathione_conjugation, "Glutathione conjugation"
            )

        if detect_halogen_to_hydroxy(self.substrate, self.metabolite):
            return self._handle_and_return_soms(
                self._handle_halogen_to_hydroxy, "Halogen to hydroxy"
            )

        if self.substrate.GetNumHeavyAtoms() < self.metabolite.GetNumHeavyAtoms():
            return self._handle_and_return_soms(
                self._handle_simple_addition, "Simple addition"
            )

        if self.substrate.GetNumHeavyAtoms() > self.metabolite.GetNumHeavyAtoms():
            return self._handle_and_return_soms(
                self._handle_simple_elimination, "Simple elimination"
            )

        if equal_number_halogens(self.substrate, self.metabolite):
            return self._handle_complex_reaction()

        return self._handle_failure()
