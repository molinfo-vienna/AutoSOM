"""Annotates SOMs for elimination reactions.

In the context of AutoSOM, these are reactions where
the number of heavy atoms in the substrate is greater than in the metabolite,
and the mol graph of the substrate is entirely contained in the mol graph of the metabolite.
An example of an elimination reaction would be the demethylation of a methylamine functional group.
This class provides functionalities to annotate SoMs for general elimination reactions,
as well as for specific cases: ester hydrolysis, acetal hydrolysis, phosphate hydrolysis,
sulfur-derivatives hydrolysis, and piperazine ring opening.
"""

from rdkit.Chem import Mol, MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import get_bond_order, log


class EliminationAnnotator(BaseAnnotator):
    """Annotate SoMs for elimination reactions."""

    def _correct_acetal_hydrolysis(self) -> bool:
        """Correct SoMs for acetals."""
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
            self.reaction_type = "elimination (acetal)"
            log(self.logger_path, "Acetal elimination detected. Corrected SoMs.")
            return True
        return False

    def _correct_ester_hydrolysis(self) -> bool:
        """Correct SoMs for ester hydrolysis."""
        if not self.ester_hydrolysis_flag:
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
            self.reaction_type = "elimination (ester hydrolysis)"
            log(self.logger_path, "Ester hydrolysis detected. Corrected SoMs.")
            return True
        return False

    def _correct_phosphate_hydrolysis(self) -> bool:
        """Correct SoMs for phosphate hydrolysis."""
        phosphate_derivate_pattern = MolFromSmarts("P(=O)")

        if not self.substrate.GetSubstructMatch(phosphate_derivate_pattern):
            return False

        som_atom = self.substrate.GetAtomWithIdx(self.soms[0])
        if (
            som_atom.GetSymbol() == "P"
        ):  # if the som is a phosphore atom, leave it as it is
            self.reaction_type = "elimination (phosphate-derivative hydrolysis)"
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
                self.reaction_type = "elimination (phosphate-derivative hydrolysis)"
                log(
                    self.logger_path,
                    "Phosphate-derivative hydrolysis detected. Corrected SoMs.",
                )
                return True
        return False

    def _correct_sulfur_derivatives_hydrolysis(self) -> bool:
        """Correct SoMs for the hydrolysis of sulfur derivatives.
        E.g.: sulfamate, sulfonamide, sulfonate, sulfuric diamide etc."""
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
        self.reaction_type = "elimination (sulfur-derivative hydrolysis)"
        log(self.logger_path, "Sulfur-derivative hydrolysis detected. Corrected SoMs.")
        return True

    def _correct_piperazine_ring_hydroysis(self) -> bool:
        """Correct SoMs for piperazine ring opening."""
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
            self.reaction_type = "elimination (piperazine ring opening)"
            log(self.logger_path, "Piperazine ring opening detected. Corrected SoMs.")
            return True
        return False

    @classmethod
    def _find_unmatched_atoms(cls, target: Mol, mcs) -> list:
        """Find unmatched atoms between the target and the query molecule."""
        return [
            atom
            for atom in target.GetAtoms()
            if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
        ]

    def _general_case_elimination(self, unmatched_atoms, target, mcs):
        """Identify SoMs in the elimination case based on unmatched
        atoms."""
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
                    self.reaction_type = "elimination (general)"

    def _has_single_and_double_bonded_oxygen(self, atom) -> bool:
        """Check if an atom has both single and double bonded oxygen
        neighbors."""
        neighbor_bonds = [
            neighbor.GetSymbol()
            + str(get_bond_order(self.substrate, atom.GetIdx(), neighbor.GetIdx()))
            for neighbor in atom.GetNeighbors()
        ]
        return "O1" in neighbor_bonds and "O2" in neighbor_bonds

    def handle_elimination(self) -> bool:
        """Annotate SoMs for elimination reactions.

        Returns:
            bool: True if a elimination reaction is found, False otherwise.
        """
        log(self.logger_path, "Attempting elimination matching.")

        if not self.substrate.HasSubstructMatch(self.metabolite):
            return False

        log(
            self.logger_path,
            "Metabolite is a substructure the substrate.",
        )

        try:
            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            mcs = rdFMCS.FindMCS([self.metabolite, self.substrate], self.mcs_params)
            unmatched_atoms = self._find_unmatched_atoms(self.substrate, mcs)

            self._general_case_elimination(unmatched_atoms, self.substrate, mcs)

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
            log(self.logger_path, f"Elimination matching failed. Error: {e}")
            return False
