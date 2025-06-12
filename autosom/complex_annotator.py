"""Annotates SOMs for complex reactions."""

from datetime import datetime
from typing import Optional

from networkx.algorithms import isomorphism
from rdkit.Chem import Atom, MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import (
    count_elements,
    get_neighbor_atomic_nums,
    is_carbon_count_unchanged,
    is_halogen_count_decreased,
    is_oxygen_count_increased,
    log,
    mol_to_graph,
)


class ComplexAnnotator(BaseAnnotator):
    """Annotate SoMs for complex reactions."""

    time: datetime

    def _correct_epoxide(self) -> bool:
        """Correct the SoMs for oxidative dehalogenation if the reaction \
        produces a stable epoxide instead of the typical alcohol \
        resulting from the hydrolysis of the intermediate epoxide."""
        # Get the SOM atom in the metabolite
        som_atom_in_metabolite = self.metabolite.GetAtomWithIdx(
            self.mapping[self.soms[0]]
        )

        # Log the atom indices, the symbols, and whether the atom has an oxygen neighbor
        # for each neighbor of the SOM atom in the metabolite
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

        # If one of the neighbors is a carbon atom with an oxygen neighbor,
        # add that atom to the SoMs
        id_of_additional_som_atom_in_metabolite = [
            id
            for (id, symbol, has_oxygen_neighbor) in info
            if symbol == "C" and has_oxygen_neighbor
        ]

        # If exactly one atom was found, add it to the SoMs
        if len(id_of_additional_som_atom_in_metabolite) == 1:
            id_of_additional_som_atom_in_substrate = self.mapping[
                id_of_additional_som_atom_in_metabolite[0]
            ]  # translate that atom id to its atom id in the substrate
            self.soms.append(id_of_additional_som_atom_in_substrate)
            return True
        return False

    def _correct_oxacyclopropane_hydrolysis(self) -> bool:
        """Correct SoMs for oxacyclopropane hydrolysis reactions."""
        smarts_oxacyclopropane = "C1OC1"
        matched_atoms = set(
            self.substrate.GetSubstructMatch(MolFromSmarts(smarts_oxacyclopropane))
        )
        if not matched_atoms.intersection(self.soms):
            return False

        corrected_soms = [
            atom.GetIdx()
            for atom in self.substrate.GetAtoms()
            if atom.GetIdx() in matched_atoms and atom.GetAtomicNum() == 6
        ]
        if len(corrected_soms) == 0:
            return False
        self.soms = corrected_soms
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
                    corrected_soms = [neighbor.GetIdx()]
                    break
            if len(corrected_soms) == 0:
                return False
            self.soms = corrected_soms
            self.reaction_type = (
                "complex (maximum common subgraph mapping - lactone hydrolysis)"
            )
            log(self.logger_path, "Lactone hydrolysis detected. Corrected SoMs.")
            return True
        return False

    def _correct_other_heterocyclic_ring_hydrolysis(self) -> bool:
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
                corrected_soms = [
                    som
                    for som in self.soms
                    if self.substrate.GetAtomWithIdx(som).GetSymbol() == "C"
                ]
                if len(corrected_soms) == 0:
                    return False
                self.soms = corrected_soms
                self.reaction_type = (
                    "complex (maximum common subgraph mapping - heterocycle opening)"
                )
                log(self.logger_path, "Heterocycle opening detected. Corrected SoMs.")
                return True

        return False

    def _find_unmapped_halogen(self) -> Optional[Atom]:
        """Find the halogen atom in the substrate that is not present in the \
        mapping."""
        halogen_symbols = ["F", "Cl", "Br", "I"]
        for atom in self.substrate.GetAtoms():
            # self.mapping maps the atom indices in the metabolite to the
            # atom indices in the substrate ({id_s: id_m}):
            if (
                atom.GetSymbol() in halogen_symbols
                and atom.GetIdx() not in self.mapping.values()
            ):
                return atom
        return None

    def _is_in_epoxide(self, som_id_in_metabolite: int) -> bool:
        """Check if the atom is in an epoxide."""
        epoxide_atom_ids = self.metabolite.GetSubstructMatch(MolFromSmarts("c1cO1"))
        if som_id_in_metabolite in epoxide_atom_ids:
            return True
        return False

    def _is_oxidative_dehalogenation(self) -> bool:
        """Check if the reaction is an oxidative dehalogenation."""
        substrate_elements = count_elements(self.substrate)
        metabolite_elements = count_elements(self.metabolite)

        if (
            is_carbon_count_unchanged(substrate_elements, metabolite_elements)
            and is_halogen_count_decreased(substrate_elements, metabolite_elements)
            and is_oxygen_count_increased(substrate_elements, metabolite_elements)
        ):
            log(self.logger_path, "Oxidative dehalogenation detected.")
            return True
        return False

    def _handle_oxidative_dehalogenation(self) -> bool:
        """Annotate SoMs for oxidative dehalogenation."""
        try:

            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.mcs_params)

            if not self._map_atoms(self.substrate, self.metabolite, mcs):
                return False

            # Find the halogen atom in the substrate that is not in the metabolite
            halogen_atom: Optional[Atom] = self._find_unmapped_halogen()
            if halogen_atom is None:
                return False

            # The SoM is the neighbor of that halogen atom
            self.soms = [halogen_atom.GetNeighbors()[0].GetIdx()]
            self.reaction_type = "complex (oxidative dehalogenation)"

            # If the reaction produces an epoxide (instead of the typical alcohol),
            # find the other atom that is part of the epoxide and add it to the SoMs
            if self._is_in_epoxide(self.mapping[self.soms[0]]):
                if self._correct_epoxide():
                    self.reaction_type = "complex (oxidative dehalogenation epoxide)"
                    return True

            return True
        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path,
                f"Halogen to hydroxy matching failed. Error: {e}",
            )
            return False

    def handle_complex_reaction_subgraph_ismorphism_mapping(
        self,
    ) -> bool:
        """Annotate SoMs for complex reactions using subgraph isomorphism \
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
        """Annotate SoMs for complex reactions using largest common subgraph \
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
        # if the neighbors (in terms of atomic number and counts),
        # are different, then the atom is a SoM.
        self.soms = [
            atom_id_s
            for atom_id_s, atom_id_m in self.mapping.items()
            if (
                get_neighbor_atomic_nums(self.substrate, atom_id_s)
                != get_neighbor_atomic_nums(self.metabolite, atom_id_m)
            )
            # or (
            #     self.substrate.GetAtomWithIdx(atom_id_s).GetTotalNumHs()
            #     != self.metabolite.GetAtomWithIdx(atom_id_m).GetTotalNumHs()
            # )
            # or (
            #     self.substrate.GetAtomWithIdx(atom_id_s).GetFormalCharge()
            #     != self.metabolite.GetAtomWithIdx(atom_id_m).GetFormalCharge()
            # )
        ]

        if self._correct_oxacyclopropane_hydrolysis():
            return True

        if self._correct_lactone_hydrolysis():
            return True

        if self._correct_other_heterocyclic_ring_hydrolysis():
            return True

        if bool(self.soms):
            self.reaction_type = "complex (maximum common subgraph mapping)"
            return True

        return False

    def handle_complex_reaction(self) -> bool:
        """Annotate SoMs for complex reactions."""
        # Handle oxidative dehalogenation
        if self._is_oxidative_dehalogenation():
            if self._handle_oxidative_dehalogenation():
                return True

        # Handle subgraph isomorphism mapping
        log(self.logger_path, "Attempting subgraph isomorphism mapping.")
        if self.handle_complex_reaction_subgraph_ismorphism_mapping():
            return True

        # Handle maximum common subgraph mapping
        log(self.logger_path, "Attempting maximum common subgraph mapping.")
        if self.handle_complex_reaction_maximum_common_subgraph_mapping():
            return True

        return False
