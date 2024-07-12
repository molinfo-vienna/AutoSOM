import networkx as nx
import numpy as np

from datetime import datetime
from networkx.algorithms import isomorphism
from rdkit.Chem import Mol, MolFromSmarts, rdFingerprintGenerator, rdFMCS
from rdkit.DataStructs import TanimotoSimilarity


class SOMFinder:
    def __init__(
        self,
        substrate,
        metabolite,
        substrate_id,
        metabolite_id,
        logger_path,
    ):
        self.substrate = substrate
        self.metabolite = metabolite
        self.substrate_id = substrate_id
        self.metabolite_id = metabolite_id
        self.logger_path = logger_path
        self.params = self._initialize_mcs_params()
        self.soms = []
        self.mapping = {}

    def _initialize_mcs_params(self):
        params = rdFMCS.MCSParameters()
        params.timeout = 10
        params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        params.BondTyper = rdFMCS.BondCompare.CompareOrder
        params.BondCompareParameters.CompleteRingsOnly = True
        params.BondCompareParameters.MatchFusedRings = True
        params.BondCompareParameters.MatchFusedRingsStrict = False
        params.BondCompareParameters.MatchStereo = False
        params.BondCompareParameters.RingMatchesRingOnly = True
        return params

    @staticmethod
    def mol_to_nx(mol: Mol) -> nx.Graph:
        """
        Convert an RDKit molecule to a NetworkX graph.

        Args:
            mol (RDKit Mol): Molecule to convert.

        Returns:
            G (NetworkX Graph): Graph representation of the molecule.
        """
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        return G

    @staticmethod
    def is_substructure(query, target):
        """
        Check if the query is a substructure of the target.

        Args:
            query (RDKit Mol): Query molecule.
            target (RDKit Mol): Target molecule.

        Returns:
            bool: True if the query is a substructure of the target, False otherwise.
        """
        return target.HasSubstructMatch(query)

    @staticmethod
    def equal_number_halogens(mol1: Mol, mol2: Mol) -> bool:
        """
        Check if two molecules have the same number of halogens.

        Args:
            mol1 (RDKit Mol): First molecule.
            mol2 (RDKit Mol): Second molecule.
        
        Returns:
            bool: True if the two molecules have the same number of halogens, False otherwise.
        """
        num_halogens1 = 0
        num_halogens2 = 0
        for atom in mol1.GetAtoms():
            if atom.GetAtomicNum() in [9, 17, 35, 53]:
                num_halogens1 += 1
        for atom in mol2.GetAtoms():
            if atom.GetAtomicNum() in [9, 17, 35, 53]:
                num_halogens2 += 1
        if num_halogens1 == num_halogens2:
            return True
        else:
            return False

    def log(self, message):
        """
        Log a message to a text file.

        Args:
            message (str): Message to log.

        Returns:
            None
        """
        with open(self.logger_path, "a+") as f:
            f.write(f"{datetime.now()} {message}\n")

    def _initialize_atom_notes(self):
        """
        Initialize the atom note properties for the substrate and the metabolite.
        """
        for atom in self.substrate.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())
        for atom in self.metabolite.GetAtoms():
            atom.SetIntProp("atomNote", atom.GetIdx())

    def _handle_simple_addition(self):
        """
        Check for simple addition reactions.

        Returns:
            bool: True if a simple addition reaction is found, False otherwise.
        """
        query, target = self.substrate, self.metabolite
        if self.is_substructure(query, target):
            self.log(
                "Query (substrate) is a substructure the target (metabolite). Looking for a match..."
            )
            try:
                mcs = rdFMCS.FindMCS([query, target], self.params)
                highlights_query = query.GetSubstructMatch(mcs.queryMol)
                highlights_target = target.GetSubstructMatch(mcs.queryMol)
                self.mapping = {
                    i: j for i, j in zip(highlights_target, highlights_query)
                }

                unmatched_atoms = [
                    atom
                    for atom in target.GetAtoms()
                    if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
                ]
                for atom in unmatched_atoms:
                    neighbors = atom.GetNeighbors()
                    for neighbor in neighbors:
                        if not neighbor.GetIdx() in self.mapping:
                            continue
                        if self.mapping[neighbor.GetIdx()] in query.GetSubstructMatch(
                            mcs.queryMol
                        ):
                            self.soms.append(self.mapping[neighbor.GetIdx()])
                self.log("Simple addition successful.")

                # Correct SoMs for the addition of carnitine to a carboxylic acid
                if (
                    len(self.soms) == 1
                    and len(
                        self.metabolite.GetSubstructMatch(
                            MolFromSmarts("[N+](C)(C)(C)-C-C(O)C-C(=O)[O]")
                        )
                    )
                    > 0
                ):
                    atom_id_in_substrate = self.mapping[self.soms[0]]
                    corrected_soms_carnitine = [
                        self.substrate.GetAtomWithIdx(atom_id_in_substrate)
                        .GetNeighbors()[0]
                        .GetIdx()
                    ]
                    self.soms = corrected_soms_carnitine
                    self.log("Carnitine addition detected. Corrected SoMs.")

                return True
            except:
                self.log("Simple addition matching failed.")
        return False

    def _handle_simple_elimination(self):
        """
        Check for simple elimination reactions.

        Returns:
            bool: True if a simple elimination reaction is found, False otherwise.
        """
        query, target = self.metabolite, self.substrate
        if self.is_substructure(query, target):
            self.log(
                "Query (metabolite) is a substructure the target (substrate). Looking for a match..."
            )
            try:

                # Add exception for reactions containing a phosphore atom
                if len(self.substrate.GetSubstructMatch(MolFromSmarts("[P](=O)"))) > 0 \
                    or len(self.substrate.GetSubstructMatch(MolFromSmarts("[P](=S)"))) > 0:
                    self.soms = [atom.GetIdx() for atom in self.substrate.GetAtoms() if atom.GetAtomicNum() == 15]
                    self.log("Phosphore atom detected. SoM is the phosphore atom.")
                    return True

                mcs = rdFMCS.FindMCS([query, target], self.params)
                unmatched_atoms = [
                    atom
                    for atom in target.GetAtoms()
                    if atom.GetIdx() not in target.GetSubstructMatch(mcs.queryMol)
                ]
                for atom in unmatched_atoms:
                    neighbors = atom.GetNeighbors()
                    for neighbor in neighbors:
                        if neighbor.GetIdx() in target.GetSubstructMatch(mcs.queryMol):
                            if atom.GetAtomicNum() != 6:
                                self.soms.append(neighbor.GetIdx())
                            else:
                                self.soms.append(atom.GetIdx())

                # Build exception for acetals
                if (
                    len(
                        set(self.soms).intersection(
                            set(
                                self.substrate.GetSubstructMatch(
                                    MolFromSmarts("[C;X4](O[*])O[*]")
                                )
                            )
                        )
                    )
                    > 0
                ):
                    corrected_soms_acetal = [
                        atom.GetIdx()
                        for atom in self.substrate.GetAtoms()
                        if (
                            atom.GetIdx()
                            in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[C;X4](O)O")
                            )
                        )
                        & (atom.GetAtomicNum() == 6)
                    ]
                    self.soms = corrected_soms_acetal
                    self.log("Acetal elimination detected. Corrected SoMs.")

                # Correct SoMs for ester hydrolysis
                if (
                    len(
                        set(self.soms).intersection(
                            set(
                                self.substrate.GetSubstructMatch(
                                    MolFromSmarts("[*][C](=O)[O][*]")
                                )
                            )
                        )
                    )
                    > 0
                ):
                    corrected_soms_ester = [
                        atom.GetIdx()
                        for atom in self.substrate.GetAtoms()
                        if (
                            atom.GetIdx()
                            in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[*][C](=O)[O][*]")
                            )
                        )
                        & (
                            atom.GetIdx()
                            in self.substrate.GetSubstructMatch(
                                MolFromSmarts("[C](=O)")
                            )
                        )
                        & (atom.GetAtomicNum() == 6)
                    ]
                    self.soms = corrected_soms_ester
                    self.log("Ester hydrolysis detected. Corrected SoMs.")

                # Correct SoMs for the hydrolysis of sulfonamines
                if len(self.soms) == 1:
                    som = self.soms[0]
                    if (som in self.substrate.GetSubstructMatch(MolFromSmarts("[S](=O)(=O)[N]"))) \
                    or (som in self.substrate.GetSubstructMatch(MolFromSmarts("[N][S](=O)(=O)[N]"))):
                        if (som not in self.substrate.GetSubstructMatch(MolFromSmarts("[N][S](=O)(=O)[O]"))) \
                            and (som not in self.substrate.GetSubstructMatch(MolFromSmarts("[S](=O)(=O)[N][O]"))):
                            self.soms = [atom.GetIdx() for atom in self.substrate.GetAtoms() if atom.GetSymbol() == 'S']
                            self.log("Sulfonamine hydrolysis detected. Corrected SoMs.")

                # Correct SoMs for the hydrolysis of sulfones
                if len(self.soms) == 1:
                    som = self.soms[0]
                    if som in self.substrate.GetSubstructMatch(MolFromSmarts("[*][*][S](=O)(=O)[*]")):
                        self.soms = [atom.GetIdx() for atom in self.substrate.GetAtoms() if atom.GetSymbol() == 'S']
                        self.log("Sulfone hydrolysis detected. Corrected SoMs.")

                # Correct SoMs on piperazine ring opening
                if len(set(self.soms).intersection(set(self.substrate.GetSubstructMatch(MolFromSmarts("N1CCNCC1"))))) != 0:
                    additional_soms = []
                    for som in self.soms:
                        if som in self.substrate.GetSubstructMatch(MolFromSmarts("N1CCNCC1")):
                            neighbors = self.substrate.GetAtomWithIdx(som).GetNeighbors()
                            for neighbor in neighbors:
                                if neighbor.GetSymbol() == "C":
                                    additional_soms.append(neighbor.GetIdx())
                    self.soms.extend(additional_soms)
                    self.log("Piperazine ring opening detected. Corrected SoMs.")

                # Correct SoMs on morpholine ring opening
                if len(set(self.soms).intersection(set(self.substrate.GetSubstructMatch(MolFromSmarts("O1CCNCC1"))))) != 0:
                    additional_soms = []
                    for som in self.soms:
                        if som in self.substrate.GetSubstructMatch(MolFromSmarts("O1CCNCC1")):
                            neighbors = self.substrate.GetAtomWithIdx(som).GetNeighbors()
                            for neighbor in neighbors:
                                if neighbor.GetSymbol() == "C":
                                    additional_soms.append(neighbor.GetIdx())
                    self.soms.extend(additional_soms)
                    self.log("Morpholine ring opening detected. Corrected SoMs.")

                # Correct SoMs for the hydrolysis of 1,3-dioxolane rings
                matches = self.substrate.GetSubstructMatch(MolFromSmarts("O1COcc1"))
                for match in matches:
                    if len(set(self.soms).intersection(set(match))) != 0:
                        corrected_soms = [
                            atom.GetIdx()
                            for atom in self.substrate.GetAtoms()
                            if (
                                atom.GetIdx()
                                in self.substrate.GetSubstructMatch(MolFromSmarts("O1COcc1"))
                            )
                            & (atom.GetAtomicNum() == 6)
                            & (atom.GetIsAromatic() == False)
                        ]
                        self.soms = corrected_soms
                        self.log("1,3-Dioxolane ring opening detected. Corrected SoMs.")

                self.log("Simple elimination successful.")
                return True
            except:
                self.log("Simple elimination matching failed.")
        return False

    def _handle_redox_reaction(self):
        """
        Check for redox reactions.

        Returns:
            bool: True if a redox reaction is found, False otherwise.
        """
        self.log("Checking for MCS matching...")
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], self.params)
        if mcs.numAtoms > 0:
            if mcs.numAtoms == (self.substrate.GetNumHeavyAtoms() - 1):
                for atom in self.substrate.GetAtoms():
                    if atom.GetIdx() not in self.substrate.GetSubstructMatch(
                        mcs.queryMol
                    ):
                        neighbors = atom.GetNeighbors()
                        for neighbor in neighbors:
                            try:
                                if (
                                    neighbor.GetIdx()
                                    in self.metabolite.GetSubstructMatch(mcs.queryMol)
                                ):
                                    self.log("Redox matching successful.")
                                    self.soms.append(neighbor.GetIdx())
                                    if atom.GetAtomicNum() not in [9, 17, 35, 53]:
                                        self.log("Redox reaction of non-halogen bond.")
                                        self.soms.append(atom.GetIdx())
                                    else:
                                        self.log("Redox reaction of halogen bond.")

                                    # Add a correction for C-N bond redox reactions (only the carbon is the som)
                                    covered_atom_types = [
                                        self.substrate.GetAtomWithIdx(
                                            atom_id
                                        ).GetAtomicNum()
                                        for atom_id in self.soms
                                    ]
                                    if (
                                        6 in covered_atom_types
                                        and 7 in covered_atom_types
                                    ):
                                        corrected_soms_cn_redox = [
                                            atom_id
                                            for atom_id in self.soms
                                            if self.substrate.GetAtomWithIdx(
                                                atom_id
                                            ).GetAtomicNum()
                                            == 6
                                        ]
                                        self.soms = corrected_soms_cn_redox
                                        self.log(
                                            "C-N redox reaction detected. Corrected SoMs."
                                        )

                                    return True
                            except KeyError:
                                self.log("Redox matching failed -- KeyError.")
                                return False
            else:
                self.log("Redox matching failed.")
        return False

    def _handle_complex_non_redox_reaction_global_subgraph_isomorphism_matching(self):
        """
        Check for complex non-redox reactions using subgraph isomorphism matching.

        Returns:
            bool: True if a complex non-redox reaction is found, False otherwise.
        """

        GS = self.mol_to_nx(self.substrate)
        GM = self.mol_to_nx(self.metabolite)

        # Check if the substrate is a subgraph of the metabolite or vice versa
        graph_matching = isomorphism.GraphMatcher(
            GS, GM, node_match=isomorphism.categorical_node_match(["atomic_num"], [0])
        )
        if graph_matching.is_isomorphic():
            self.log(
                "Full graph matching found! Metabolite is an isomorphic subgraph of the substrate."
            )
            self.mapping = graph_matching.mapping
            already_matched_metabolite_atom_indices = set(self.mapping.values())
        else:
            graph_matching = isomorphism.GraphMatcher(
                GM,
                GS,
                node_match=isomorphism.categorical_node_match(["atomic_num"], [0]),
            )
            if graph_matching.is_isomorphic():
                self.log(
                    "Full graph matching found! Substrate is an isomorphic subgraph of the metabolite."
                )
                self.mapping = graph_matching.mapping
                already_matched_metabolite_atom_indices = set(self.mapping.keys())
            else:
                self.log(
                    "No full graph matching found. Moving on to partial graph matching..."
                )
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
                except:
                    self.mapping[atom_s.GetIdx()] = -1

        # Compute the SoMs by comparing the degree and the atomic numbers of the atoms in the substrate and the metabolite
        # If 1.) the degree or 2.) the neighbors of the atoms in the substrate and the metabolite are different (in terms of atomic number),
        # or 3.) the number of bonded hydrogens is different, then the atom is a SoM.
        self.soms = [
            atom_id_s
            for atom_id_s, atom_id_m in self.mapping.items()
            if atom_id_m != -1
            and (
                self.substrate.GetAtomWithIdx(atom_id_s).GetDegree()
                != self.metabolite.GetAtomWithIdx(atom_id_m).GetDegree()
                or set(
                    [
                        neighbor.GetAtomicNum()
                        for neighbor in self.substrate.GetAtomWithIdx(
                            atom_id_s
                        ).GetNeighbors()
                    ]
                )
                != set(
                    [
                        neighbor.GetAtomicNum()
                        for neighbor in self.metabolite.GetAtomWithIdx(
                            atom_id_m
                        ).GetNeighbors()
                    ]
                )
                or self.substrate.GetAtomWithIdx(atom_id_s).GetTotalNumHs()
                != self.metabolite.GetAtomWithIdx(atom_id_m).GetTotalNumHs()
            )
        ]

        return True

    def _handle_complex_non_redox_reaction_largest_common_subgraph_matching(self):
        """
        Check for complex non-redox reactions using largest common subgraph matching (maximum common substructure).

        Returns:
            bool: True if a complex non-redox reaction is found, False otherwise.
        """

        self.log("Checking for partial graph matching...")

        params = self.params
        params.BondTyper = rdFMCS.BondCompare.CompareAny
        mcs = rdFMCS.FindMCS([self.substrate, self.metabolite], params)

        if mcs.numAtoms > 0:
            self.log("MCS matching found!")
            highlights_substrate = self.substrate.GetSubstructMatch(mcs.queryMol)
            highlights_metabolite = self.metabolite.GetSubstructMatch(mcs.queryMol)
            self.mapping = {
                i: j for i, j in zip(highlights_substrate, highlights_metabolite)
            }

            # Map the unmapped atoms in the substrate to the remaining atoms in the metabolite
            # based on the Tanimoto similarity of their Morgan fingerprints
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            atom_based_fp_metabolite = [
                mfpgen.GetFingerprint(self.metabolite, fromAtoms=[atom_m.GetIdx()])
                for atom_m in self.metabolite.GetAtoms()
            ]
            already_matched_atom_indices = set(highlights_metabolite)

            for atom_s in self.substrate.GetAtoms():
                if atom_s.GetIdx() not in highlights_substrate:
                    # Calculate the Morgan fingerprint for the atom in the substrate
                    atom_based_fp_substrate = mfpgen.GetFingerprint(
                        self.substrate,
                        fromAtoms=[atom_s.GetIdx()],
                    )
                    # Calculate the Tanimoto similarity between the atom in the substrate and all atoms in the metabolite
                    similarities = [
                        TanimotoSimilarity(
                            atom_based_fp_substrate,
                            atom_based_fp_metabolite[atom_m.GetIdx()],
                        )
                        for atom_m in self.metabolite.GetAtoms()
                    ]

                    # Assign the atom in the metabolite with the highest similarity to the atom in the substrate
                    for _ in range(self.metabolite.GetNumHeavyAtoms()):
                        if (self.mapping.get(atom_s.GetIdx()) is None) and (
                            np.argmax(similarities) not in already_matched_atom_indices
                        ):
                            self.mapping[atom_s.GetIdx()] = int(np.argmax(similarities))
                            already_matched_atom_indices.add(np.argmax(similarities))
                            break
                        else:
                            similarities[np.argmax(similarities)] = 0

                    # If no atom in the metabolite is assigned to the atom in the substrate, assign the first available atom
                    if self.mapping.get(atom_s.GetIdx()) is None:
                        try:
                            first_of_remaining_unmapped_ids = [
                                i
                                for i in range(self.metabolite.GetNumHeavyAtoms())
                                if i not in already_matched_atom_indices
                            ][0]
                            self.mapping[
                                atom_s.GetIdx()
                            ] = first_of_remaining_unmapped_ids
                            already_matched_atom_indices.add(
                                first_of_remaining_unmapped_ids
                            )
                        except:
                            self.mapping[atom_s.GetIdx()] = -1

            # Compute the SoMs by comparing the degree and the atomic numbers of the atoms in the substrate and the metabolite
            # If 1.) the degree or 2.) the neighbors of the atoms in the substrate and the metabolite are different (in terms of atomic number),
            # or 3.) the number of bonded hydrogens is different, then the atom is a SoM.
            self.soms = [
                atom_id_s
                for atom_id_s, atom_id_m in self.mapping.items()
                if atom_id_m != -1
                and (
                    self.substrate.GetAtomWithIdx(atom_id_s).GetDegree()
                    != self.metabolite.GetAtomWithIdx(atom_id_m).GetDegree()
                    or set(
                        [
                            neighbor.GetAtomicNum()
                            for neighbor in self.substrate.GetAtomWithIdx(
                                atom_id_s
                            ).GetNeighbors()
                        ]
                    )
                    != set(
                        [
                            neighbor.GetAtomicNum()
                            for neighbor in self.metabolite.GetAtomWithIdx(
                                atom_id_m
                            ).GetNeighbors()
                        ]
                    )
                    or self.substrate.GetAtomWithIdx(atom_id_s).GetTotalNumHs()
                    != self.metabolite.GetAtomWithIdx(atom_id_m).GetTotalNumHs()
                )
            ]

            # Add an exception for the reduction of nitro groups
            if len(self.soms) == 3:
                if set(
                    [
                        self.substrate.GetAtomWithIdx(som).GetAtomicNum()
                        for som in self.soms
                    ]
                ) == {6, 7, 8}:
                    corrected_nitro_soms = [
                        som
                        for som in self.soms
                        if self.substrate.GetAtomWithIdx(som).GetAtomicNum() == 7
                    ]
                    self.soms = corrected_nitro_soms
                    self.log("Nitro reduction detected. Corrected SoMs.")

            # Add an exception for the reduction of thiourea groups
            smarts_thiourea = "NC(=S)N"
            if self.substrate.HasSubstructMatch(MolFromSmarts(smarts_thiourea)):
                corrected_thiourea_soms = [
                    som
                    for som in self.soms
                    if self.substrate.GetAtomWithIdx(som).GetAtomicNum() == 16
                ]
                self.soms = corrected_thiourea_soms
                self.log("Thiourea reduction detected. Corrected SoMs.")

            # Correct SoMs for the hydrolysis of oxacyclopropane rings
            smarts_oxycyclopropane = "C1OC1"
            if (
                len(
                    set(self.soms).intersection(
                        set(
                            self.substrate.GetSubstructMatch(
                                MolFromSmarts(smarts_oxycyclopropane)
                            )
                        )
                    )
                )
                > 0
            ):
                corrected_soms_oxacyclopropane = [
                    atom.GetIdx()
                    for atom in self.substrate.GetAtoms()
                    if (
                        atom.GetIdx()
                        in self.substrate.GetSubstructMatch(
                            MolFromSmarts(smarts_oxycyclopropane)
                        )
                    )
                    & (atom.GetAtomicNum() == 6)
                ]
                self.soms = corrected_soms_oxacyclopropane
                self.log("Oxacyclopropane hydrolysis detected. Corrected SoMs.")

            return True

        else:
            return False

    def find_soms(self):
        """
        Find the SoMs in the substrate and the metabolite.

        Returns:
            soms (List[int]): List of SoMs.
        """
        self._initialize_atom_notes()
        self.log(
            f"Substrate ID: {self.substrate_id}, metabolite ID: {self.metabolite_id}"
        )

        if self.substrate.GetNumHeavyAtoms() < self.metabolite.GetNumHeavyAtoms():
            self.log(
                "Substrate has less heavy atoms than the metabolite. Checking for simple addition..."
            )
            if self._handle_simple_addition():
                return sorted(self.soms)
            else:
                self.log("No simple addition found.")

        elif self.substrate.GetNumHeavyAtoms() > self.metabolite.GetNumHeavyAtoms():
            self.log(
                "Substrate has more heavy atoms than the metabolite. Checking for simple elimination..."
            )
            if self._handle_simple_elimination():
                return sorted(self.soms)
            else:
                self.log("No simple elimination found.")

        else:
            if self.equal_number_halogens(self.substrate, self.metabolite):
                self.log(
                    "Complex reaction with equal number of heavy atoms and equal number of halogen atoms in substrate and metabolite, i.e. maybe a simple redox reaction."
                )
                if self._handle_redox_reaction():
                    return sorted(self.soms)
                else:
                    self.log("No simple redox reaction found.")

        self.log(
            "Complex non-redox reaction. Checking for global subgraph isomorphism matching..."
        )
        if (
            self._handle_complex_non_redox_reaction_global_subgraph_isomorphism_matching()
        ):
            return sorted(self.soms)

        else:
            self.log(
                "No global subgraph isomorphism matching found. Checking for largest common subgraph matching..."
            )
            if (
                self._handle_complex_non_redox_reaction_largest_common_subgraph_matching()
            ):
                return sorted(self.soms)
            else:
                self.log("No partial graph matching matching found.")

        self.log("No SoMs found.")
        return sorted(self.soms)


def get_soms(substrate, metabolite, substrate_id, metabolite_id, logger_path):
    """
    Find the SoMs in the substrate and the metabolite.

    Args:
        substrate (RDKit Mol): Substrate molecule.
        metabolite (RDKit Mol): Metabolite molecule.
        substrate_id (int): Substrate ID.
        metabolite_id (int): Metabolite ID.
        logger_path (str): Path to the log file.

    Returns:
        soms (List[int]): List of SoMs.
    """
    som_finder = SOMFinder(
        substrate, metabolite, substrate_id, metabolite_id, logger_path
    )
    return som_finder.find_soms()
