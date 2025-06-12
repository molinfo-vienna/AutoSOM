"""Annotates SOMs for redox reactions.

In the context of AutoSOM, these are reactions where:
- the number of heavy atoms in the substrate and metabolite are the same
- the MCS covers all but one heavy atom in the substrate.
Examples of redox reactions are the reduction of a ketone to an alcohol
or the oxidation of an aldehyde to a carboxylic acid.

Prior to annotating general redox reactions, the annotator checks if the
reaction is an oxidative dehalogenation. If the reaction is an oxidative
dehalogenation, the annotator annotates the SoMs for that specific reaction.
If the reaction is not an oxidative dehalogenation, the annotator annotates
the SoMs for a general redox reaction.
The criteria for an oxidative dehalogenation reaction are:
- The number of carbon atoms in the substrate is the same as in the metabolite.
- The number of halogens in the substrate is decreased by one.
- The number of oxygen atoms in the substrate is increased by one.
Oxidative dehalogenation reactions come in two flavors:
- General oxidative dehalogenation.
- Oxidative dehalogenation producing an epoxide.
"""

from datetime import datetime
from typing import Optional

from rdkit.Chem import Atom, Mol, MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import (
    count_elements,
    is_carbon_count_unchanged,
    is_halogen_count_decreased,
    is_oxygen_count_increased,
    log,
)


class RedoxAnnotator(BaseAnnotator):
    """Annotate SoMs for redox reactions."""

    time: datetime

    @classmethod
    def _find_unmatched_atoms(cls, target: Mol, mcs) -> list:
        """Find unmatched atoms between the target and the query molecule."""
        return [
            atom
            for atom in target.GetAtoms()
            if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
        ]

    def _correct_cn_redox(self) -> bool:
        """Apply corrections if the redox reaction involves a C-N bond."""
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

    def _has_equal_number_halogens(self) -> bool:
        """Check if substrate and metabolite have the same number of \
        halogens."""
        halogen_atomic_nums = {9, 17, 35, 53}

        num_halogens_substrate = sum(
            atom.GetAtomicNum() in halogen_atomic_nums
            for atom in self.substrate.GetAtoms()
        )
        num_halogens_metabolite = sum(
            atom.GetAtomicNum() in halogen_atomic_nums
            for atom in self.metabolite.GetAtoms()
        )

        if num_halogens_substrate == num_halogens_metabolite:
            return True
        return False

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
            self.reaction_type = "redox (oxidative dehalogenation)"

            # If the reaction produces an epoxide (instead of the typical alcohol),
            # find the other atom that is part of the epoxide and add it to the SoMs
            if self._is_in_epoxide(self.mapping[self.soms[0]]):
                if self._correct_epoxide():
                    self.reaction_type = "redox (oxidative dehalogenation epoxide)"
                    return True

            return True
        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path,
                f"Halogen to hydroxy matching failed. Error: {e}",
            )
            return False

    def handle_redox_reaction(self) -> bool:
        """Annotate SoMs for redox reactions."""
        try:
            log(self.logger_path, "Attempting redox reaction matching.")

            # Check if the reaction is an oxidative dehalogenation
            if self._is_oxidative_dehalogenation():
                if self._handle_oxidative_dehalogenation():
                    return True

            # Find the MCS between the substrate and metabolite
            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            self._set_mcs_bond_compare_params_to_redox()
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.mcs_params)
            self._reset_mcs_bond_compare_params()

            # Check if the MCS covers all but one heavy atom in the substrate
            if mcs.numAtoms != (self.substrate.GetNumHeavyAtoms() - 1):
                log(self.logger_path, "Not a redox reaction.")
                return False

            # Find unmatched atoms in the substrate
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
                    self.reaction_type = "redox (general)"

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
