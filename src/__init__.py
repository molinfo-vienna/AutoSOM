"""
soman: a Python package for the automated annotation of sites of metabolism (SoMs) \
from substrates and their metabolites.

soman is a Python package that provides a set of tools for
the automated annotation of sites of metabolism (SoMs)
from substrates and their metabolites. The package is designed
to be used in the context of drug metabolism prediction,
where SoMs are the atoms in a substrate molecule that are
expected to undergo a metabolic transformation.
The package uses the RDKit and NetworkX libraries for
molecular representation and substructure matching,
and provides a set of predefined rules for the annotation
of SoMs based on common metabolic reactions.

The package is intended to be used in combination with a
dataset of substrate-metabolite pairs, where each pair
represents a metabolic reaction.

The package is currently in development and is not yet available on PyPI.
To install and use the package, follow the instructions in
the README.md file in the GitHub repository.
"""

__version__ = "0.1.0"
__author__ = "Roxane A. Jacob"

from .addition_annotator import AdditionAnnotator
from .autosom import annotate_soms
from .base_annotator import BaseAnnotator
from .complex_annotator import ComplexAnnotator
from .elimination_annotator import EliminationAnnotator
from .redox_annotator import RedoxAnnotator

__all__ = [
    "annotate_soms",
    "AdditionAnnotator",
    "BaseAnnotator",
    "ComplexAnnotator",
    "EliminationAnnotator",
    "RedoxAnnotator",
]
