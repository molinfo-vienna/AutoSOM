"""Annotates SOMs for complex reactions."""

from networkx.algorithms import isomorphism
from rdkit.Chem import MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import get_neighbor_atomic_nums, log, mol_to_graph


class ComplexAnnotator(BaseAnnotator):
    """Annotate SoMs for complex reactions."""

    def _correct_alkyl_chain_deletion(self) -> bool:
        """Correct SoMs for the deletion of one or more carbon atoms from an
        alkyl chain."""
        # Compare carbon counts
        substrate_carbon_count = sum(
            1 for atom in self.substrate.GetAtoms() if atom.GetSymbol() == "C"
        )
        metabolite_carbon_count = sum(
            1 for atom in self.metabolite.GetAtoms() if atom.GetSymbol() == "C"
        )

        # Compute carbon loss
        carbon_loss = substrate_carbon_count - metabolite_carbon_count
        if carbon_loss < 1:
            return False

        # Find the MCS
        self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareAny)
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.mcs_params)
        mcs_mol = MolFromSmarts(mcs.smartsString)

        if not mcs_mol:
            return False

        # Map atoms from substrate to metabolite using substructure match
        substrate_atoms = set(range(self.substrate.GetNumHeavyAtoms()))
        mcs_match_substrate_atoms = set(self.substrate.GetSubstructMatch(mcs_mol))
        deleted_atoms = substrate_atoms - mcs_match_substrate_atoms

        # Extract deleted carbon fragment
        deleted_carbons = []
        for atom_idx in deleted_atoms:
            atom = self.substrate.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == "C":
                deleted_carbons.append(atom_idx)

        # If no carbon was deleted, return False
        if len(deleted_carbons) < 1:
            return False

        # Ensure the lost carbons formed a connected alkyl chain
        visited = set()

        def dfs(atom_idx):
            """Depth-First Search to explore connected deleted carbons."""
            if atom_idx in visited:
                return
            visited.add(atom_idx)
            for neighbor in self.substrate.GetAtomWithIdx(atom_idx).GetNeighbors():
                if (
                    neighbor.GetIdx() in deleted_carbons
                    and neighbor.GetIdx() not in visited
                ):
                    dfs(neighbor.GetIdx())

        # Start DFS from the first deleted carbon
        dfs(deleted_carbons[0])

        # Check if all deleted carbons are part of one connected fragment
        if visited != set(deleted_carbons):
            return False

        # Check if the deleted fragment is an alkyl chain (only C and H neighbors)
        for atom_idx in deleted_carbons:
            neighbors = [
                n.GetSymbol()
                for n in self.substrate.GetAtomWithIdx(atom_idx).GetNeighbors()
            ]
            for n in neighbors:
                if n not in {"C", "H"}:
                    deleted_carbons.remove(atom_idx)
                    break

        # Update SoMs
        self.soms = list(deleted_carbons)
        self.reaction_type = (
            "complex (maximum common subgraph matching - alkyl chain deletion)"
        )
        log(self.logger_path, "Alkyl chain deletion detected. Corrected SoMs.")
        return bool(self.soms)

    def _correct_oxacyclopropane_hydrolysis(self) -> bool:
        """Corrects SoMs for oxacyclopropane hydrolysis reactions."""
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
            "complex (maximum common subgraph matching - oxacyclopropane hydrolysis)"
        )
        log(self.logger_path, "Oxacyclopropane hydrolysis detected. Corrected SoMs.")
        return bool(self.soms)

    def _correct_ring_opening(self) -> bool:
        """Corrects SoMs for ring-opening reactions."""
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
                    "complex (maximum common subgraph matching - heterocycle opening)"
                )
                log(self.logger_path, "Heterocycle opening detected. Corrected SoMs.")
                return True

        return False

    def _correct_thiourea_reduction(self) -> bool:
        """Corrects SoMs for thiourea reduction reactions."""
        if not self.substrate.HasSubstructMatch(MolFromSmarts("NC(=S)N")):
            return False

        self.soms = [
            som
            for som in self.soms
            if self.substrate.GetAtomWithIdx(som).GetAtomicNum() == 16
        ]
        self.reaction_type = (
            "complex (maximum common subgraph matching - thiourea reduction)"
        )
        log(self.logger_path, "Thiourea reduction detected. Corrected SoMs.")
        return bool(self.soms)

    def handle_complex_reaction(self) -> bool:
        """Annotate SoMs for complex reactions."""
        log(self.logger_path, "Attempting subgraph isomorphism matching.")
        if self.handle_complex_reaction_subgraph_ismorphism_matching():
            return True

        log(self.logger_path, "Attempting maximum common subgraph matching.")
        if self.handle_complex_reaction_maximum_common_subgraph_matching():
            return True

        return False

    def handle_complex_reaction_subgraph_ismorphism_matching(
        self,
    ) -> bool:
        """Annotate SoMs for complex reactions using subgraph isomorphism
        matching."""
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
                "Subgraph isomorphism matching found!",
            )
            log(
                self.logger_path,
                "Metabolite is an isomorphic subgraph of the substrate.",
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
                    "Subgraph isomorphism matching found!",
                )
                log(
                    self.logger_path,
                    "Substrate is an isomorphic subgraph of the metabolite.",
                )
                self.mapping = graph_matching.mapping
                already_matched_metabolite_atom_indices = set(self.mapping.keys())
            else:
                log(self.logger_path, "No subgraph isomorphism matching found.")
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

        # Compute the SoMs by comparing the atoms in the substrate and the metabolite:
        # if 1.) the degree,
        # or 2.) the neighbors (in terms of atomic number),
        # or 3.) the number of bonded hydrogens,
        # or 4.) the charge,
        # are different, then the atom is a SoM.
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
                or self.substrate.GetAtomWithIdx(atom_id_s).GetFormalCharge()
                != self.metabolite.GetAtomWithIdx(atom_id_m).GetFormalCharge()
            )
        ]
        self.reaction_type = "complex (subgraph isomorphism matching)"
        return True

    def handle_complex_reaction_maximum_common_subgraph_matching(
        self,
    ) -> bool:
        """Annotate SoMs for complex reactions using largest common subgraph
        (maximum common substructure) matching."""
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
        # if 1.) the neighbors (in terms of atomic number),
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

        if self._correct_ring_opening():
            return True

        # if self._correct_alkyl_chain_deletion():
        #     return True

        if bool(self.soms):
            self.reaction_type = "complex (maximum common subgraph matching)"
            return True

        return False
