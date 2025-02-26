"""Contains pilot function for SOM annotation"""

from typing import List, Tuple

from rdkit.Chem import Mol

from .addition_annotator import AdditionAnnotator
from .base_annotator import BaseAnnotator
from .complex_annotator import ComplexAnnotator
from .elimination_annotator import EliminationAnnotator
from .redox_annotator import RedoxAnnotator


def annotate_soms(
    params: Tuple[str, int, bool],
    substrate_data: Tuple[Mol, int],
    metabolite_data: Tuple[Mol, int],
) -> Tuple[List[int], str]:
    """Annotates SoMs for a given substrate-metabolite pair."""
    annotator = BaseAnnotator(params, substrate_data, metabolite_data)

    annotator.log_initial_reaction_info()

    if not annotator.check_inchi_validity():
        return annotator.log_and_return()
    if not annotator.check_atom_types():
        return annotator.log_and_return()
    # if not annotator.standardize_molecules():
    #     return annotator.log_and_return()

    annotator.remove_hydrogens()
    annotator.initialize_atom_notes()

    addition_annotator = AdditionAnnotator(params, substrate_data, metabolite_data)
    complex_annotator = ComplexAnnotator(params, substrate_data, metabolite_data)
    elimination_annotator = EliminationAnnotator(
        params, substrate_data, metabolite_data
    )
    redox_annotator = RedoxAnnotator(params, substrate_data, metabolite_data)

    weight_ratio = annotator.compute_weight_ratio()

    if weight_ratio == 1:
        if addition_annotator.handle_addition():
            return addition_annotator.log_and_return()

    if weight_ratio == -1:
        if elimination_annotator.handle_elimination():
            return elimination_annotator.log_and_return()

    # The next steps rely more heavily on MCS matching,
    # which can take very long for large molecules,
    # so we skip them if the substrate or metabolite is
    # too large (filter_size parameter).
    if annotator.is_too_large_to_process():
        return annotator.log_and_return()

    if weight_ratio == 0:
        if redox_annotator.handle_redox_reaction():
            return redox_annotator.log_and_return()

    # This last category is to catch all cases that failed previously.
    # It is less specific and has a higher error rate.
    if complex_annotator.handle_complex_reaction():
        return complex_annotator.log_and_return()

    return annotator.log_and_return()
