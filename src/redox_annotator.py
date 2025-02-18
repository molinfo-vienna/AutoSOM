"""Annotates SOMs for redox reactions.

In the context of AutoSOM, these are reactions where the number of hevay
atoms in the substrate and metabolite are the same, the number of
halogens in the substrate and metabolite are the same, and the MCS
covers all but one heavy atom in the substrate. An example of a redox
reaction is the conversion of a ketone to an alcohol.
"""

from rdkit.Chem import Mol, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import log


class RedoxAnnotator(BaseAnnotator):
    """Annotate SoMs for redox reactions."""

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

    def _find_unmatched_atoms(self, target: Mol, mcs) -> list:
        """Find unmatched atoms between the target and the query molecule."""
        return [
            atom
            for atom in target.GetAtoms()
            if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
        ]

    def _has_equal_number_halogens(self) -> bool:
        """Check if substrate and metabolite have the same number of
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

    def handle_redox_reaction(self) -> bool:
        """Annotate SoMs for redox reactions."""
        try:
            log(self.logger_path, "Attempting redox reaction matching.")
            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            self._set_mcs_bond_compare_params_to_redox()
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.mcs_params)
            self._reset_mcs_bond_compare_params()

            # Check if the MCS covers all but one heavy atom in the substrate
            if mcs.numAtoms != (self.substrate.GetNumHeavyAtoms() - 1):
                log(self.logger_path, "Not a redox reaction.")
                return False

            if not self._has_equal_number_halogens():
                log(self.logger_path, "Not a redox reaction.")
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
