"""Annotates SOMs for complex reactions."""

from networkx.algorithms import isomorphism
from rdkit.Chem import MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import get_neighbor_atomic_nums, log, mol_to_graph


class ComplexAnnotator(BaseAnnotator):
    """Annotate SoMs for complex reactions."""

    # def _correct_furan_hydrolysis(self) -> bool:
    #     """Correct SoMs for furan hydrolysis reactions."""
    #     smarts_furan = "c1ccoc1"
    #     matched_atoms = set(
    #         self.substrate.GetSubstructMatch(MolFromSmarts(smarts_furan))
    #     )
    #     if not matched_atoms.intersection(self.soms):
    #         return False
    #     self.soms = [atom.GetIdx() for atom in self.substrate.GetAtoms()
    #                  if atom.GetIdx() in matched_atoms
    #                  and atom.GetAtomicNum() == 6
    #                  and (get_neighbor_atomic_nums(self.substrate, atom.GetIdx()).get(8) == 1)]
    #     self.reaction_type = (
    #         "complex (maximum common subgraph mapping - furan hydrolysis)"
    #     )
    #     log(self.logger_path, "Furan hydrolysis detected. Corrected SoMs.")
    #     return bool(self.soms)
        
    def _correct_oxacyclopropane_hydrolysis(self) -> bool:
        """Correct SoMs for oxacyclopropane hydrolysis reactions."""
        smarts_oxacyclopropane = "C1OC1"
        matched_atoms = set(
            self.substrate.GetSubstructMatch(MolFromSmarts(smarts_oxacyclopropane))
        )
        if not matched_atoms.intersection(self.soms):
            return False

        self.soms = [
            atom.GetIdx()
            for atom in self.substrate.GetAtoms()
            if atom.GetIdx() in matched_atoms and atom.GetAtomicNum() == 6
        ]
        self.reaction_type = (
            "complex (maximum common subgraph mapping - oxacyclopropane hydrolysis)"
        )
        log(self.logger_path, "Oxacyclopropane hydrolysis detected. Corrected SoMs.")
        return bool(self.soms)

    def _correct_lactone_hydrolysis(self) -> bool:
        """Correct SoMs for lactone hydrolysis reactions."""
        # Check that the metabolite has exactly one ring less than the substrate
        if not (
            self.metabolite.GetRingInfo().NumRings()
            == self.substrate.GetRingInfo().NumRings() - 1
        ):
            return False

        # Check that the SOM that was already found by the general procedure is just one
        if len(self.soms) != 1:
            return False

        # Check that the SOM is part of a lactone
        general_lactone_smarts_pattern = "[C;R](=O)[O;R][C;R]"
        if not any(
            som
            in self.substrate.GetSubstructMatch(
                MolFromSmarts(general_lactone_smarts_pattern)
            )
            for som in self.soms
        ):
            return False

        # If all the previous conditions are met,
        # change SOM to the carbon atom of the lactone
        atom = self.substrate.GetAtomWithIdx(self.soms[0])
        # Check that the atom that was found in the sp3 oxygen of the lactone
        if atom.GetSymbol() == "O":
            for neighbor in atom.GetNeighbors():
                if (
                    neighbor.GetSymbol() == "C"
                    and str(neighbor.GetHybridization()) == "SP2"
                ):
                    self.soms = [neighbor.GetIdx()]
                    break

            self.reaction_type = (
                "complex (maximum common subgraph mapping - lactone hydrolysis)"
            )
            log(self.logger_path, "Lactone hydrolysis detected. Corrected SoMs.")
            return True
        return False

    def _correct_other_heterocycle_opening(self) -> bool:
        """Correct SoMs for ring-opening reactions."""
        if (
            self.metabolite.GetRingInfo().NumRings()
            != self.substrate.GetRingInfo().NumRings() - 1
        ):
            return False

        if len(self.soms) > 1:
            som_symbols = [
                self.substrate.GetAtomWithIdx(som).GetSymbol() for som in self.soms
            ]
            if som_symbols.count("C") == 1 and any(
                sym in som_symbols for sym in ["O", "N", "S"]
            ):
                self.soms = [
                    som
                    for som in self.soms
                    if self.substrate.GetAtomWithIdx(som).GetSymbol() == "C"
                ]
                self.reaction_type = (
                    "complex (maximum common subgraph mapping - heterocycle opening)"
                )
                log(self.logger_path, "Heterocycle opening detected. Corrected SoMs.")
                return True

        return False

    def _correct_thiourea_reduction(self) -> bool:
        """Correct SoMs for thiourea reduction reactions."""
        if not self.substrate.HasSubstructMatch(MolFromSmarts("NC(=S)N")):
            return False

        self.soms = [
            som
            for som in self.soms
            if self.substrate.GetAtomWithIdx(som).GetAtomicNum() == 16
        ]
        self.reaction_type = (
            "complex (maximum common subgraph mapping - thiourea reduction)"
        )
        log(self.logger_path, "Thiourea reduction detected. Corrected SoMs.")
        return bool(self.soms)

    def handle_complex_reaction(self) -> bool:
        """Annotate SoMs for complex reactions."""
        log(self.logger_path, "Attempting subgraph isomorphism mapping.")
        if self.handle_complex_reaction_subgraph_ismorphism_mapping():
            return True

        log(self.logger_path, "Attempting maximum common subgraph mapping.")
        if self.handle_complex_reaction_maximum_common_subgraph_mapping():
            return True

        return False

    def handle_complex_reaction_subgraph_ismorphism_mapping(
        self,
    ) -> bool:
        """Annotate SoMs for complex reactions using subgraph isomorphism
        mapping."""
        mol_graph_substrate = mol_to_graph(self.substrate)
        mol_graph_metabolite = mol_to_graph(self.metabolite)

        metabolite_in_substrate_flag = False
        substrate_in_metabolite_flag = False

        # Check if the substrate is a subgraph of the metabolite or vice versa
        graph_mapping_metabolite_in_substrate = isomorphism.GraphMatcher(
            mol_graph_substrate,
            mol_graph_metabolite,
            node_match=isomorphism.categorical_node_match(["atomic_num"], [0]),
        )
        graph_mapping_substrate_in_metabolite = isomorphism.GraphMatcher(
            mol_graph_metabolite,
            mol_graph_substrate,
            node_match=isomorphism.categorical_node_match(["atomic_num"], [0]),
        )

        if graph_mapping_metabolite_in_substrate.is_isomorphic():
            metabolite_in_substrate_flag = True
        if graph_mapping_substrate_in_metabolite.is_isomorphic():
            substrate_in_metabolite_flag = True

        # Find the SOMs in the case of complete mapping
        # (metabolite is fully mapped to substrate and vice versa)

        if metabolite_in_substrate_flag and substrate_in_metabolite_flag:
            log(
                self.logger_path,
                "Subgraph isomorphism mapping found!",
            )
            log(
                self.logger_path,
                "Substrate and metabolite have complete mapping.",
            )
            self.mapping = graph_mapping_metabolite_in_substrate.mapping
            self.reaction_type = "complex (subgraph isomorphism mapping)"
            
            # An atom is a SoM if the number of bonded hydrogens is different
            # or the formal charge is different
            self.soms = [
                atom_id_s
                for atom_id_s, atom_id_m in self.mapping.items()
                if (
                    self.substrate.GetAtomWithIdx(atom_id_s).GetTotalNumHs()
                    != self.metabolite.GetAtomWithIdx(atom_id_m).GetTotalNumHs()
                )
                or (
                    self.substrate.GetAtomWithIdx(atom_id_s).GetFormalCharge()
                    != self.metabolite.GetAtomWithIdx(atom_id_m).GetFormalCharge()
                )
            ]
            return True
        log(
            self.logger_path,
            "No subgraph isomorphism mapping found.",
        )
        return False
    

    def handle_complex_reaction_maximum_common_subgraph_mapping(
        self,
    ) -> bool:
        """Annotate SoMs for complex reactions using largest common subgraph
        (maximum common substructure) mapping."""
        self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareAny)
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.mcs_params)

        if mcs.numAtoms == 0:
            return False

        self.mapping = dict(
            zip(
                self.substrate.GetSubstructMatch(mcs.queryMol),
                self.metabolite.GetSubstructMatch(mcs.queryMol),
            )
        )

        # Identify SoMs based on atom environment differences.
        # if 1.) the neighbors (in terms of atomic number and counts),
        # or 2.) the number of bonded hydrogens,
        # are different, then the atom is a SoM.
        self.soms = [
            atom_id_s
            for atom_id_s, atom_id_m in self.mapping.items()
            if (
                get_neighbor_atomic_nums(self.substrate, atom_id_s)
                != get_neighbor_atomic_nums(self.metabolite, atom_id_m)
            )
            or (
                self.substrate.GetAtomWithIdx(atom_id_s).GetTotalNumHs()
                != self.metabolite.GetAtomWithIdx(atom_id_m).GetTotalNumHs()
            )
        ]

        if self._correct_thiourea_reduction():
            return True

        if self._correct_oxacyclopropane_hydrolysis():
            return True

        if self._correct_lactone_hydrolysis():
            return True

        if self._correct_other_heterocycle_opening():
            return True
        
        # if self._correct_furan_hydrolysis():
        #     return True

        if bool(self.soms):
            self.reaction_type = "complex (maximum common subgraph mapping)"
            return True

        return False
