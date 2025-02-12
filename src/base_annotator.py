# pylint: disable=I1101

"""This module provides base functionalities for annotating sites of metabolism (SOMs) \
given a substrate and a metabolite molecule."""

from typing import List

from rdkit.Chem import Mol, rdFMCS

from .utils import log


class BaseAnnotator:
    """Annotates Sites of Metabolism (SoMs) from substrate/metabolite pairs.

    Attributes:
        substrate (Mol): The substrate molecule.
        substrate_id (int): The substrate molecule ID.
        metabolite (Mol): The metabolite molecule.
        metabolite_id (int): The metabolite molecule ID.
        mapping (dict[int, int]): Mapping of atom indices between substrate and metabolite.
        params (rdFMCS.MCSParameters): Parameters for the Maximum Common Substructure (MCS) search.
        reaction_type (str): Type of reaction identified.
        soms (List[int]): List of identified SoMs.

    Methods:
    """

    def __init__(
        self,
        substrate_mol: Mol,
        substrate_id: int,
        metabolite_mol: Mol,
        metabolite_id: int,
    ):
        """Initialize the BaseAnnotator class."""

        self.substrate = substrate_mol
        self.substrate_id = substrate_id
        self.metabolite = metabolite_mol
        self.metabolite_id = metabolite_id

        self.mapping: dict[int, int] = {}
        self.params = self._initialize_mcs_params()
        self.reaction_type: str = "unknown"
        self.soms: List[int] = []

    def _initialize_mcs_params(self):
        params = rdFMCS.MCSParameters()
        params.timeout = 10
        params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        params.BondTyper = rdFMCS.BondCompare.CompareOrder
        params.BondCompareParameters.CompleteRingsOnly = False
        params.BondCompareParameters.MatchFusedRings = False
        params.BondCompareParameters.MatchFusedRingsStrict = False
        params.BondCompareParameters.MatchStereo = False
        params.BondCompareParameters.RingMatchesRingOnly = False
        return params

    def _map_atoms(self, query, target, mcs):
        """Create mapping between query and target based on MCS."""
        highlights_query = query.GetSubstructMatch(mcs.queryMol)
        highlights_target = target.GetSubstructMatch(mcs.queryMol)
        if not highlights_query or not highlights_target:
            return False
        self.mapping = dict(zip(highlights_target, highlights_query))
        return True

    def _set_mcs_bond_typer_param(self, bond_typer_param):
        """Set the MCS bond compare parameter."""
        self.params.BondTyper = bond_typer_param

    def _set_mcs_bond_compare_params_to_redox(self):
        """Set the MCS bond compare parameters for redox reactions."""
        self.params.BondCompareParameters.CompleteRingsOnly = True
        self.params.BondCompareParameters.MatchFusedRings = True
        self.params.BondCompareParameters.MatchFusedRingsStrict = True
        self.params.BondCompareParameters.RingMatchesRingOnly = True

    def _reset_mcs_bond_compare_params(self):
        """Reset the MCS bond compare parameters to their default value."""
        self.params.BondCompareParameters.CompleteRingsOnly = False
        self.params.BondCompareParameters.MatchFusedRings = False
        self.params.BondCompareParameters.MatchFusedRingsStrict = False
        self.params.BondCompareParameters.RingMatchesRingOnly = False

    def log_and_return(self) -> tuple[list[int], str]:
        """Log annotation rule and return SOMs and annotation rule."""
        log(self.logger_path, f"{self.reaction_type.capitalize()} successful.")
        return sorted(self.soms), self.reaction_type
