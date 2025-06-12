# pylint: disable=I1101
"""Provides base functionalities for annotating sites of metabolism (SOMs)."""

from datetime import datetime
from typing import List, Tuple

from chembl_structure_pipeline import standardizer
from rdkit.Chem import Mol, MolToInchiKey, SanitizeMol, rdFMCS
from rdkit.Chem.MolStandardize import rdMolStandardize

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
    """

    mapping: dict[int, int]
    params: rdFMCS.MCSParameters
    reaction_type: str
    soms: List[int]
    time: datetime

    def __init__(
        self,
        params: Tuple[str, int, bool],
        substrate_data: Tuple[Mol, int],
        metabolite_data: Tuple[Mol, int],
    ):
        """Initialize the BaseAnnotator class."""
        self.logger_path = params[0]
        self.filter_size = params[1]
        self.ester_hydrolysis_flag = params[2]
        self.substrate = substrate_data[0]
        self.substrate_id = substrate_data[1]
        self.metabolite = metabolite_data[0]
        self.metabolite_id = metabolite_data[1]

        self.mapping = {}
        self.mcs_params = self._initialize_mcs_params()
        self.reaction_type = "unknown"
        self.soms = []
        self.time = datetime.min

    def _initialize_mcs_params(self):
        mcs_params = rdFMCS.MCSParameters()
        mcs_params.timeout = 10
        mcs_params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        mcs_params.BondTyper = rdFMCS.BondCompare.CompareOrder
        mcs_params.BondCompareParameters.CompleteRingsOnly = False
        mcs_params.BondCompareParameters.MatchFusedRings = False
        mcs_params.BondCompareParameters.MatchFusedRingsStrict = False
        mcs_params.BondCompareParameters.MatchStereo = False
        mcs_params.BondCompareParameters.RingMatchesRingOnly = False
        return mcs_params

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
        self.mcs_params.BondTyper = bond_typer_param

    def _set_mcs_bond_compare_params_to_redox(self):
        """Set the MCS bond compare parameters for redox reactions."""
        self.mcs_params.BondCompareParameters.CompleteRingsOnly = True
        self.mcs_params.BondCompareParameters.MatchFusedRings = True
        self.mcs_params.BondCompareParameters.MatchFusedRingsStrict = True

    def _reset_mcs_bond_compare_params(self):
        """Reset the MCS bond compare parameters to their default value."""
        self.mcs_params.BondCompareParameters.CompleteRingsOnly = False
        self.mcs_params.BondCompareParameters.MatchFusedRings = False
        self.mcs_params.BondCompareParameters.MatchFusedRingsStrict = False

    def check_atom_types(self) -> bool:
        """Check if the molecules contain any invalid atoms."""
        allowed_atoms = {
            "H",
            "C",
            "N",
            "O",
            "S",
            "P",
            "F",
            "Cl",
            "Br",
            "I",
            "B",
            "Si",
            "Se",
        }
        substrate_atoms = set(atom.GetSymbol() for atom in self.substrate.GetAtoms())
        metabolite_atoms = set(atom.GetSymbol() for atom in self.metabolite.GetAtoms())
        if not substrate_atoms.issubset(allowed_atoms):
            log(
                self.logger_path,
                f"Invalid atom in the substrate: {substrate_atoms.difference(allowed_atoms)}",
            )
            return False
        if not metabolite_atoms.issubset(allowed_atoms):
            log(
                self.logger_path,
                f"Invalid atom in the metabolite: {metabolite_atoms.difference(allowed_atoms)}",
            )
            return False
        return True

    def check_validity(self) -> bool:
        """Check if the substrate and metabolite are valid molecules \
            (inchikey can be computed)."""
        if self.substrate is None or self.metabolite is None:
            substrate_inchikey = None
            metabolite_inchikey = None
        else:
            try:
                substrate_inchikey = MolToInchiKey(self.substrate)
                metabolite_inchikey = MolToInchiKey(self.metabolite)
            except (ValueError, RuntimeError) as e:
                log(self.logger_path, f"InChIKey conversion failed: {e}")
                substrate_inchikey = None
                metabolite_inchikey = None
        if substrate_inchikey is None:
            log(self.logger_path, "Invalid substrate.")
            return False
        if metabolite_inchikey is None:
            log(self.logger_path, "Invalid metabolite.")
            return False
        if substrate_inchikey == metabolite_inchikey:
            log(self.logger_path, "Identical substrate and metabolite.")
            return False
        return True

    def compute_weight_ratio(self) -> int:
        """Compute whether the substrate is lighter, heavier or equally heavy \
        than the metabolite."""
        if self.substrate.GetNumHeavyAtoms() < self.metabolite.GetNumHeavyAtoms():
            log(self.logger_path, "Substrate lighter than metabolite.")
            return 1
        if self.substrate.GetNumHeavyAtoms() > self.metabolite.GetNumHeavyAtoms():
            log(self.logger_path, "Substrate heavier than the metabolite.")
            return -1
        return 0

    def initialize_atom_notes(self) -> None:
        """Initialize the atom note properties for the substrate and the \
        metabolite."""
        for atom in self.substrate.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())
        for atom in self.metabolite.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())

    def log_and_return(self) -> tuple[list[int], str, float]:
        """Log annotation rule and return SOMs and annotation rule."""
        if self.reaction_type == "unknown":
            log(self.logger_path, "No reaction detected.")
        else:
            log(self.logger_path, f"{self.reaction_type.capitalize()} successful.")
        return (
            sorted(self.soms),
            self.reaction_type,
            (datetime.now() - self.time).total_seconds(),
        )

    def log_and_timeout(self) -> tuple[list[int], str, float]:
        """Log annotation rule and return SOMs and annotation rule in case of timeout."""
        log(self.logger_path, "Timeout occurred.")
        return (
            sorted(self.soms),
            "timeout",
            (datetime.now() - self.time).total_seconds(),
        )

    def log_initial_reaction_info(self) -> None:
        """Log the initial reaction information."""
        log(
            self.logger_path,
            f"Substrate ID: {self.substrate_id}, Metabolite ID: {self.metabolite_id}",
        )

    def standardize_molecules(self) -> bool:
        """Standardize the substrate and metabolite."""
        # Get main fragment (remove counterions, solvents, etc.)
        self.substrate = standardizer.get_parent_mol(self.substrate)[0]
        self.metabolite = standardizer.get_parent_mol(self.metabolite)[0]

        # Standardize the molecules
        self.substrate = standardizer.standardize_mol(self.substrate)
        self.metabolite = standardizer.standardize_mol(self.metabolite)

        # Get canonical tautomers
        self.substrate = rdMolStandardize.CanonicalTautomer(self.substrate)
        self.metabolite = rdMolStandardize.CanonicalTautomer(self.metabolite)

        # Sanitize (this operation is in place)
        SanitizeMol(self.substrate)
        SanitizeMol(self.metabolite)

        if self.substrate is None:
            log(self.logger_path, "Substrate standardization failed.")
            return False
        if self.metabolite is None:
            log(self.logger_path, "Metabolite standardization failed.")
            return False
        return True
