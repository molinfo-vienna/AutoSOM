"""Contains pilot function for SOM annotation."""

import threading
from datetime import datetime
from functools import wraps
from typing import List, Tuple

from rdkit.Chem import Mol

from .addition_annotator import AdditionAnnotator
from .base_annotator import BaseAnnotator
from .complex_annotator import ComplexAnnotator
from .elimination_annotator import EliminationAnnotator


def with_timeout(seconds):
    """Decorate functions to handle timeouts."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                return None
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]

        return wrapper

    return decorator


def annotate_soms(
    params: Tuple[str, int, bool],
    substrate_data: Tuple[Mol, int],
    metabolite_data: Tuple[Mol, int],
) -> Tuple[List[int], str, float]:
    """Annotates SoMs for a given substrate-metabolite pair."""
    timeout = params[1]

    annotator = BaseAnnotator(params, substrate_data, metabolite_data)
    annotator.time = datetime.now()

    annotator.log_initial_reaction_info()

    if not annotator.check_validity():
        return annotator.log_and_return()
    if not annotator.check_atom_types():
        return annotator.log_and_return()
    if not annotator.standardize_molecules():
        return annotator.log_and_return()

    annotator.initialize_atom_notes()

    addition_annotator = AdditionAnnotator(params, substrate_data, metabolite_data)
    complex_annotator = ComplexAnnotator(params, substrate_data, metabolite_data)
    elimination_annotator = EliminationAnnotator(
        params, substrate_data, metabolite_data
    )

    addition_annotator.time = datetime.now()
    complex_annotator.time = datetime.now()
    elimination_annotator.time = datetime.now()

    weight_ratio = annotator.compute_weight_ratio()

    if weight_ratio == 1:
        handle_addition_with_timeout = with_timeout(timeout)(
            addition_annotator.handle_addition
        )
        result = handle_addition_with_timeout()
        if result is None:
            return annotator.log_and_timeout()
        if result:
            return addition_annotator.log_and_return()

    if weight_ratio == -1:
        handle_elimination_with_timeout = with_timeout(timeout)(
            elimination_annotator.handle_elimination
        )
        result = handle_elimination_with_timeout()
        if result is None:
            return annotator.log_and_timeout()
        if result:
            return elimination_annotator.log_and_return()

    handle_complex_with_timeout = with_timeout(timeout)(
        complex_annotator.handle_complex_reaction
    )
    result = handle_complex_with_timeout()
    if result is None:
        return annotator.log_and_timeout()
    if result:
        return complex_annotator.log_and_return()

    return annotator.log_and_return()
