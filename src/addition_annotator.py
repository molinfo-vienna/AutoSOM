"""Annotates SOMs for addition reactions.

In the context of AutoSOM, these are reactions where
the number of heavy atoms in the substrate is less than in the metabolite,
and the mol graph of the metabolite is entirely contained in the mol graph of the substrate.
An example of an addition reaction would be the "addition" of an hydroxy group to an aromatic ring.
This class provides functionalities to annotate SoMs for general addition reactions,
as well as for a specific case: carnitine addition.
"""

from rdkit.Chem import Mol, MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import log


class AdditionAnnotator(BaseAnnotator):
    """Annotate SoMs for addition reactions."""

    def _correct_carnitine_addition(self) -> bool:
        """Correct SoMs for the addition of carnitine to a carboxylic acid."""
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

    @classmethod
    def _find_unmatched_atoms(cls, target: Mol, mcs) -> list:
        """Find unmatched atoms between the target and the query molecule."""
        return [
            atom
            for atom in target.GetAtoms()
            if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
        ]

    def _general_case_simple_addition(self, unmatched_atoms, query, mcs):
        """Identify SoMs in the simple addition case based on unmatched
        atoms."""
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

    def handle_simple_addition(self) -> bool:
        """Annotate SoMs for simple addition reactions."""
        log(self.logger_path, "Attempting simple addition matching.")

        if not self.metabolite.HasSubstructMatch(self.substrate):
            return False

        log(
            self.logger_path,
            "Susbtrate is a substructure of the metabolite.",
        )

        try:
            self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
            mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.mcs_params)
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
