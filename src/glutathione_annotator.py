"""Annotate SOMs for glutathione conjugation reactions.

The glutathione moiety is a tripeptide composed of glutamic acid,
cysteine, and glycine. AutoSOM finds these reactions by identifying the
glutathione moiety in the metabolite, via SMARTS pattern matching.
"""

from typing import Optional

from rdkit.Chem import FragmentOnBonds, GetMolFrags, MolFromSmiles, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import log


class GlutathioneAnnotator(BaseAnnotator):
    """Annotate SoMs for glutathione conjugation reactions."""

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
        """Split the metabolite into fragments and identify the one without the
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

    def handle_glutathione_conjugation(self) -> bool:
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
                                    self.reaction_type = "glutathione conjugation"
                                    return True

        except (ValueError, KeyError, AttributeError) as e:
            log(
                self.logger_path, f"Glutathione conjugation matching failed. Error: {e}"
            )

        return False

    def is_glutathione_conjugation(self) -> bool:
        """Check if the reaction is a glutathione conjugation."""
        glutathione_smiles = "C(CC(=O)N[C@@H](CS)C(=O)NCC(=O)O)[C@@H](C(=O)O)N"
        if self.metabolite.HasSubstructMatch(
            MolFromSmiles(glutathione_smiles)
        ) and not self.substrate.HasSubstructMatch(MolFromSmiles(glutathione_smiles)):
            log(self.logger_path, "Glutathione conjugation detected.")
            return True
        return False
