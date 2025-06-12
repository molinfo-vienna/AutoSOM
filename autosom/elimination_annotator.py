"""Annotates SOMs for elimination reactions.

In the context of AutoSOM, these are reactions where
the number of heavy atoms in the substrate is greater than in the metabolite,
and the mol graph of the substrate is entirely contained in the mol graph of the metabolite.
An example of an elimination reaction would be the demethylation of a methylamine functional group.
This class provides functionalities to annotate SoMs for general elimination reactions,
as well as for specific cases: ester hydrolysis, acetal hydrolysis, phosphate hydrolysis,
sulfur-derivatives hydrolysis, and piperazine ring opening.
"""

from datetime import datetime

from rdkit.Chem import Mol, MolFromSmarts, rdFMCS

from .base_annotator import BaseAnnotator
from .utils import get_bond_order, log


class EliminationAnnotator(BaseAnnotator):
    """Annotate SoMs for elimination reactions."""

    time: datetime

    @classmethod
    def _find_unmatched_atoms(cls, target: Mol, mcs) -> list:
        """Find unmatched atoms between the target and the query molecule."""
        return [
            atom
            for atom in target.GetAtoms()
            if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
        ]

    def _correct_ester_hydrolysis(self) -> bool:
        """Correct SoMs for ester hydrolysis."""
        if not self.ester_hydrolysis_flag:
            return False

        ester_pattern = MolFromSmarts("[*][C](=O)[O,S][*]")
        matches = self.substrate.GetSubstructMatches(ester_pattern)
        match_index = -1
        for i, match in enumerate(matches):
            if any(som in match for som in self.soms):
                match_index = i
                break

        if match_index == -1:
            return False

        corrected_soms = [
            atom.GetIdx()
            for atom in self.substrate.GetAtoms()
            if atom.GetIdx() in matches[match_index]
            and atom.GetAtomicNum() == 6
            and self._has_single_and_double_bonded_oxygen_neighbors(atom)
        ]
        if corrected_soms:
            self.soms = corrected_soms
            self.reaction_type = "elimination (ester hydrolysis)"
            log(self.logger_path, "Ester hydrolysis detected. Corrected SoMs.")
            return True
        return False

    def _correct_hydrolysis_of_esters_of_inorganic_acids_phosphor(
        self, smarts, reaction_type
    ) -> bool:
        """Correct SoMs for phosphate or thiophosphate hydrolysis."""
        if not self.substrate.GetSubstructMatch(MolFromSmarts(smarts)):
            return False

        som_atom = self.substrate.GetAtomWithIdx(self.soms[0])
        if (
            som_atom.GetSymbol() == "P"
        ):  # if the som is a phosphore atom, leave it as it is
            self.reaction_type = f"elimination ({reaction_type}-derivative hydrolysis)"
            log(
                self.logger_path,
                "Hydrolysis of an ester of an inorganic (phosphore-based) acid detected. No action needed.",
            )
            return True
        for neighbor in som_atom.GetNeighbors():
            if neighbor.GetSymbol() == "P":
                # if one of its neighbors is a phosphore atoms,
                # we have the case where a phosphore hydrolysis took place,
                # and the metabolite does **not** contain the phosphate functional group anymore
                self.soms = [neighbor.GetIdx()]
                self.reaction_type = (
                    f"elimination ({reaction_type}-derivative hydrolysis)"
                )
                log(
                    self.logger_path,
                    "Hydrolysis of an ester of an inorganic (phosphore-based) acid detected. Corrected SoM.",
                )
                return True
            for neighbor_bis in neighbor.GetNeighbors():
                if neighbor_bis.GetSymbol() == "P":
                    # if one of the neighbors of the neighbors is a phosphore atoms,
                    # we have the case where the hydrolsysis of the ester was picked up
                    # correctly by the algorithm, but the wrong side of the ester was annotated.
                    # We therefore correct the SOM.
                    self.soms = [neighbor_bis.GetIdx()]
                    self.reaction_type = (
                        f"elimination ({reaction_type}-derivative hydrolysis)"
                    )
                    log(
                        self.logger_path,
                        "Hydrolysis of an ester of an inorganic (phosphore-based) acid detected. Corrected SoM.",
                    )
                    return True
        return False

    def _correct_hydrolysis_of_esters_of_inorganic_acids_sulfur(self) -> bool:
        """
        Correct SoMs for the hydrolysis of esters of sulfur-based inorganic.

        E.g.: sulfamate, sulfonamide, sulfonate, sulfuric diamide etc.
        """
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
        log(
            self.logger_path,
            "Hydrolysis of an ester of an inorganic (sulfur-based) acid detected. Corrected SoMs.",
        )
        return True

    def _general_case_elimination(self):
        """Annotate SoMs for general elimination reactions."""
        self._set_mcs_bond_typer_param(rdFMCS.BondCompare.CompareOrder)
        mcs = rdFMCS.FindMCS([self.metabolite, self.substrate], self.mcs_params)

        if not self._map_atoms(self.substrate, self.metabolite, mcs):
            return False

        unmatched_atoms = self._find_unmatched_atoms(self.substrate, mcs)

        for atom in unmatched_atoms:  # iterate over unmatched atoms
            for (
                neighbor
            ) in atom.GetNeighbors():  # iterate over neighbors of the unmatched atom
                if neighbor.GetIdx() in self.substrate.GetSubstructMatch(
                    mcs.queryMol
                ):  # if the neighbor is in the mapping...
                    if (
                        atom.GetAtomicNum() == 6
                    ):  # ...and the unmatched atom is a carbon atom...
                        self.soms.append(
                            atom.GetIdx()
                        )  # ...add the unmatched atom to the SoMs (TYPE 1)
                        self.reaction_type = "elimination (general - type 1)"
                    else:  # ...and the unmatched atom is NOT a carbon atom...
                        self.soms.append(
                            neighbor.GetIdx()
                        )  # ...add the neighbor to the SoMs (TYPE 2)
                        self.reaction_type = "elimination (general - type 2)"

        if len(self.soms) == 0:
            log(self.logger_path, "General elimination matching failed.")
        return True

        # TYPE 1: dealkylation, deacylation
        # TYPE 2: also dealkylation and deacylation, but with the "leaving group"
        #         as recorded metabolite (rare), dehalogenation,
        #         reduction at heteroatom (nitro, sulfoxide etc.),
        #         reduction at SP3 carbon (alcohol to alkane)

    def _has_single_and_double_bonded_oxygen_neighbors(self, atom) -> bool:
        """Check if an atom has both a single and a double bonded oxygen neighbors."""
        neighbor_bonds = [
            neighbor.GetSymbol()
            + str(get_bond_order(self.substrate, atom.GetIdx(), neighbor.GetIdx()))
            for neighbor in atom.GetNeighbors()
        ]
        return "O1" in neighbor_bonds and "O2" in neighbor_bonds

    def handle_elimination(self) -> bool:
        """Annotate SoMs for elimination reactions.

        Returns:
            bool: True if an elimination reaction is found, False otherwise.
        """
        log(self.logger_path, "Attempting elimination matching.")

        if not self.substrate.HasSubstructMatch(self.metabolite):
            return False

        log(
            self.logger_path,
            "Metabolite is a substructure the substrate.",
        )

        try:

            if not self._general_case_elimination():
                return False

            if self._correct_ester_hydrolysis():
                return True

            if self._correct_hydrolysis_of_esters_of_inorganic_acids_phosphor(
                "P(=O)", "phosphate"
            ):
                return True

            if self._correct_hydrolysis_of_esters_of_inorganic_acids_phosphor(
                "P(=S)", "thiophosphate"
            ):
                return True

            if self._correct_hydrolysis_of_esters_of_inorganic_acids_sulfur():
                return True

            return True

        except (ValueError, KeyError, AttributeError) as e:
            log(self.logger_path, f"Elimination matching failed. Error: {e}")
            return False
