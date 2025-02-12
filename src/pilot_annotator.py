# pylint: disable=I1101

"""This module provides functionalities for annotating the sites of metabolism (SoMs) \
given a substrate and a metabolite molecule."""

from typing import List, Tuple

from rdkit.Chem import Mol

from .addition_annotator import AdditionAnnotator
from .complex_annotator import ComplexAnnotator
from .elimination_annotator import EliminationAnnotator
from .glutathione_annotator import GlutathioneAnnotator
from .oxidative_dehalogenation_annotator import OxidativeDehalogenationAnnotator
from .redox_annotator import RedoxAnnotator
from .utils import log


class PilotAnnotator:
    def __init__(
        self,
        logger_path: str,
        filter_size: int,
        ester_hydrolysis: bool,
    ):
        """Initialize the SOMFinder class."""
        self.ester_hydrolysis = ester_hydrolysis
        self.filter_size = filter_size
        self.logger_path = logger_path

    def annotate_soms(
        self,
        substrate: Mol,
        substrate_id: int,
        metabolite: Mol,
        metabolite_id: int,
    ) -> Tuple[List[int], str]:
        """Annotates SoMs for a given substrate-metabolite pair."""

        # Initialize atom notes
        for atom in substrate.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())
        for atom in metabolite.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())

        # Log the substrate and metabolite IDs
        log(
            self.logger_path,
            f"Substrate ID: {substrate_id}, Metabolite ID: {metabolite_id}",
        )

        glutathione_annotator = GlutathioneAnnotator(
            substrate, substrate_id, metabolite, metabolite_id
        )
        oxidative_dehalogation_annotator = OxidativeDehalogenationAnnotator(
            substrate, substrate_id, metabolite, metabolite_id
        )
        addition_annotator = AdditionAnnotator(
            substrate, substrate_id, metabolite, metabolite_id
        )
        elimination_annotator = EliminationAnnotator(
            substrate, substrate_id, metabolite, metabolite_id
        )
        redox_annotator = RedoxAnnotator(
            substrate, substrate_id, metabolite, metabolite_id
        )
        complex_annotator = ComplexAnnotator(
            substrate, substrate_id, metabolite, metabolite_id
        )

        if glutathione_annotator.is_glutathione_conjugation():
            if glutathione_annotator.handle_glutathione_conjugation():
                return glutathione_annotator.log_and_return()

        if oxidative_dehalogation_annotator.is_oxidative_dehalogenation():
            if oxidative_dehalogation_annotator.handle_oxidative_dehalogenation():
                return oxidative_dehalogation_annotator.log_and_return()

        weight_ratio = self._compute_weight_ratio(substrate, metabolite)

        if weight_ratio == 1:
            if addition_annotator.handle_simple_addition():
                return addition_annotator.log_and_return()

        if weight_ratio == -1:
            if elimination_annotator.handle_simple_elimination():
                return elimination_annotator.log_and_return()

        elif weight_ratio == 0:
            # The next steps rely more heavily on MCS matching,
            # which can take very long for large molecules,
            # so we skip them if the substrate or metabolite is
            # too large (filter_size parameter).
            if self._is_too_large_to_process(substrate, metabolite):
                log(
                    self.logger_path,
                    "Substrate or metabolite too large for processing.",
                )
                return [], "too many atoms"

            if redox_annotator.handle_redox_reaction():
                return redox_annotator.log_and_return()

            if complex_annotator.handle_complex_reaction():
                return complex_annotator.log_and_return()

        log(self.logger_path, "No reaction detected.")
        return [], "unknown"

    def _compute_weight_ratio(self, substrate, metabolite) -> int:
        """Compute whether the substrate is lighter, \
        heavier or equally heavy than the metabolite."""
        if substrate.GetNumHeavyAtoms() < metabolite.GetNumHeavyAtoms():
            log(self.logger_path, "Substrate lighter than metabolite.")
            return 1
        if substrate.GetNumHeavyAtoms() > metabolite.GetNumHeavyAtoms():
            log(self.logger_path, "Substrate heavier than the metabolite.")
            return -1
        return 0

    def _is_too_large_to_process(self, substrate, metabolite) -> bool:
        """Check if the substrate or metabolite is too large for further processing."""
        if (
            substrate.GetNumHeavyAtoms() > self.filter_size
            or metabolite.GetNumHeavyAtoms() > self.filter_size
        ):
            log(
                self.logger_path,
                "Substrate or metabolite too large for processing.",
            )
            return True
        return False
