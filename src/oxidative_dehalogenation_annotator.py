"""This module provides functionalities to annotate SOMs for oxidative dehalogenation reactions.

In the context of AutoSOM, these are reactions where
the number of carbon atoms in the substrate is the same as in the metabolite,
the number of halogens in the substrate is decreased by one,
and the number of oxygen atoms in the substrate is increased by one.
This class provides functionalities to annotate SoMs for general oxidative dehalogenation reactions,
as well as for specific cases: ox. dehal. producing an epoxide or a quinone-like metabolite.
"""

from typing import Optional

from rdkit.Chem import Atom, MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import (count_elements, get_bond_order, is_carbon_count_unchanged,
                    is_halogen_count_decreased, is_oxygen_count_increased, log)


class OxidativeDehalogenationAnnotator(BaseAnnotator):
    """Annotate SoMs for oxidative dehalogenation reactions."""

    def __init__(self, params, substrate_data, metabolite_data):
        super().__init__(params, substrate_data, metabolite_data)

    def _correct_epoxide(self) -> bool:
        """Correct the SoMs for oxidative dehalogenation if \
        the reaction produces an epoxide instead of the typical alcohol."""
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

    def _correct_quinone_like_oxidation(self) -> bool:
        """Correct the SoMs foroxidative dehalogenation if \
        the reaction produces a quinone-like metabolite instead of the typical alcohol."""
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
                temp_atom_id_in_substrate: Optional[int] = next(
                    (k for k, v in self.mapping.items() if v == temp_atom_id), None
                )  # we need to find the key of the value in the mapping
                if temp_atom_id_in_substrate:
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

    def _find_unmapped_halogen(self) -> Optional[Atom]:
        """Find the halogen atom in the substrate that is not present in the mapping."""
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

    def _is_in_quinone(self, som_id_in_metabolite: int) -> bool:
        """Check if the atom is in a quinone-like structure."""
        quinone_like_atom_ids = self.metabolite.GetSubstructMatch(
            MolFromSmarts("C1=CCC=CC1=O")
        )
        if som_id_in_metabolite in quinone_like_atom_ids:
            return True
        return False

    def handle_oxidative_dehalogenation(self) -> bool:
        """
        Annotate SoMs for oxidative dehalogenation.

        Returns:
            bool: True if annotation is successful, False otherwise.
        """
        try:

            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareAny)
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.params)

            if not self._map_atoms(self.substrate, self.metabolite, mcs):
                return False

            # Find the halogen atom in the substrate that is not in the metabolite
            halogen_atom: Optional[Atom] = self._find_unmapped_halogen()
            if halogen_atom is None:
                return False

            # The SoM is the neighbor of that halogen atom
            self.soms = [halogen_atom.GetNeighbors()[0].GetIdx()]
            self.reaction_type = "oxidative dehalogenation"

            # If the reaction produces an epoxide (instead of the typical alcohol),
            # find the other atom that is part of the epoxide and add it to the SoMs
            if self._is_in_epoxide(self.mapping[self.soms[0]]):
                if self._correct_epoxide():
                    self.reaction_type = "oxidative dehalogenation (epoxide)"
                    return True

            # If the reaction produces a quinone-like metabolite (instead of the typical alcohol),
            # find the other atom that is part of the quinone-like structure and add it to the SoMs
            if self._is_in_quinone(self.mapping[self.soms[0]]):
                if self._correct_quinone_like_oxidation():
                    self.reaction_type = "oxidative dehalogenation (quinone-like)"
                    return True

            return True
        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path,
                f"Halogen to hydroxy matching failed. Error: {e}",
            )
            return False

    def is_oxidative_dehalogenation(self) -> bool:
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
