"""Annotates SOMs for addition reactions.

In the context of AutoSOM, these are reactions where
the number of heavy atoms in the substrate is less than in the metabolite,
and the mol graph of the metabolite is entirely contained in the mol graph of the substrate.
An example of an addition reaction would be the "addition" of an hydroxy group to an aromatic ring.
This class provides functionalities to annotate SoMs for general addition reactions,
as well as for two specific case: carnitine addition and glutathione conjugation.
"""

from datetime import datetime
from typing import Optional

from rdkit.Chem import (
    FragmentOnBonds,
    GetMolFrags,
    Mol,
    MolFromSmarts,
    MolFromSmiles,
    rdFMCS,
)

from .base_annotator import BaseAnnotator
from .utils import log


class AdditionAnnotator(BaseAnnotator):
    """Annotate SoMs for addition reactions."""

    time: datetime

    @classmethod
    def _find_unmatched_atoms(cls, target: Mol, mcs) -> list:
        """Find unmatched atoms between the target and the query molecule."""
        return [
            atom
            for atom in target.GetAtoms()
            if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
        ]

    def _correct_carnitine_addition(self) -> bool:
        """Correct SoMs for the addition of carnitine to a carboxylic acid. \
        The conjugation of carnitine to carboxylic acid is mediated by \
        the coenzyme A. Thus, the SOM here is not the SP3-hybridized oxygen \
        of the carboxylic acid, the carbon atom of the carboxylic acid. \
        Other amino acids undergo conjugation via CoA (glycine, cysteine, taurine etc.), \
        but they are not matched by the general addition scenario because in those cases, \
        the carboxylic acid reacts to an amide. These cases teherfore do not need to be \
        corrected by this pipeline, as they correctly handled later in the complex reaction pipeline."""
        carnitine_pattern = MolFromSmarts("[N+](C)(C)(C)-C-C(O)C-C(=O)[O]")

        if (
            len(self.soms) != 1
            or len(self.metabolite.GetSubstructMatch(carnitine_pattern)) == 0
        ):
            return False

        som = self.soms[0]
        corrected_som = [
            self.substrate.GetAtomWithIdx(som).GetNeighbors()[0].GetIdx()
        ]  # We can take the first neighbor of the SOM atom in the substrate, because it is the only neighbor.
        if corrected_som:
            self.soms = corrected_som
            self.reaction_type = "addition (carnitine)"
            log(self.logger_path, "Carnitine addition detected. Corrected SoMs.")
            return True
        return False

    def _find_sulfur_index(self, glutathione_indices: list) -> Optional[int]:
        """Find the sulfur atom index in the glutathione structure.

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
    ) -> Optional[int]:
        """Find the index of the atom neighboring the sulfur atom.

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
        """Split the metabolite into fragments and identify the one without the \
        glutathione moiety.

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

    def _general_case_addition(self):
        """Annotate SoMs for addition elimination reactions."""
        self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.mcs_params)

        if not self._map_atoms(self.substrate, self.metabolite, mcs):
            return False

        unmatched_atoms = self._find_unmatched_atoms(self.metabolite, mcs)

        for atom in unmatched_atoms:  # iterate over unmatched atoms
            for (
                neighbor
            ) in atom.GetNeighbors():  # iterate over neighbors of the unmatched atom
                if (
                    not neighbor.GetIdx() in self.mapping
                ):  # if the neighbor is not in the mapping, meaning it is not in the MCS...
                    continue  # ...skip the neighbor
                    # We fomulate the condition as a negative affirmation to cover cases
                    # where there is more than one addition site in the substrate.
                    # This should not be the case in clean, single-step reactions,
                    # but it is a possibility in badly curated data.
                self.soms.append(
                    self.mapping[neighbor.GetIdx()]
                )  # get the index of the neighbor in the MCS query molecule (substrate)
                self.reaction_type = "addition (general)"

        if len(self.soms) == 0:
            log(self.logger_path, "General addition matching failed.")
            return False
        return True

    def _is_glutathione_conjugation(self) -> bool:
        """Check if the reaction is a glutathione conjugation."""
        glutathione_smiles = "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N"
        if self.metabolite.HasSubstructMatch(
            MolFromSmiles(glutathione_smiles)
        ) and not self.substrate.HasSubstructMatch(MolFromSmiles(glutathione_smiles)):
            log(self.logger_path, "Glutathione conjugation detected.")
            return True
        return False

    def _map_atoms_glutathione(
        self, source_mol, target_mol
    ) -> Optional[dict[int, int]]:
        """Map atoms between two molecules using MCS.

        Args:
            source_mol (Mol): The source molecule.
            target_mol (Mol): The target molecule.

        Returns:
            dict: A mapping between the atoms in the source
                  molecule and the atoms in the target molecule.
        """
        self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareAny)
        mcs = rdFMCS.FindMCS([source_mol, target_mol], self.mcs_params)
        if not mcs or not mcs.queryMol:
            return None

        highlights_query = source_mol.GetSubstructMatch(mcs.queryMol)
        highlights_target = target_mol.GetSubstructMatch(mcs.queryMol)

        if not highlights_query or not highlights_target:
            return None

        return dict(zip(highlights_query, highlights_target))

    def _handle_glutathione_conjugation(self) -> bool:
        """Annotate SoMs for glutathione conjugation.

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
                    # Identify the neigbor of the sulfur atom
                    # that is not part of the glutathione moiety
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
                                    self.reaction_type = (
                                        "addition (glutathione conjugation)"
                                    )
                                    return True

        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path, f"Glutathione conjugation matching failed. Error: {e}"
            )

        return False

    def handle_addition(self) -> bool:
        """Annotate SoMs for addition reactions.

        Returns:
            bool: True if an addition reaction is found, False otherwise.
        """
        try:

            log(self.logger_path, "Attempting addition matching.")

            if self._is_glutathione_conjugation():
                if self._handle_glutathione_conjugation():
                    return True

            if not self.metabolite.HasSubstructMatch(self.substrate):
                return False

            log(
                self.logger_path,
                "Substrate is a substructure of the metabolite.",
            )

            if not self._general_case_addition():
                return False

            if self._correct_carnitine_addition():
                return True

            return True

        except (ValueError, KeyError, AttributeError) as e:
            log(self.logger_path, f"Addition matching failed. Error: {e}")
            return False
