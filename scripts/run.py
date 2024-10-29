"""This script predicts Sites of Metabolism (SoMs) for unseen data using pairs of molecular structures (substrate/metabolite) provided in either InChI or SMILES format.

The script performs the following steps:
1. Parses command-line arguments to get input and output paths, input data type, and filter size.
2. Reads the input data from a CSV file.
3. Ensures necessary columns are present in the data.
4. Converts molecular structures from InChI or SMILES to RDKit Mol objects.
5. Curates the data and predicts SoMs for each reaction.
6. Symmetrizes the predicted SoMs.
7. Outputs the annotated data to SDF files.
8. Merges all SoMs from the same substrates and outputs the merged data to a single SDF file.

Command-line arguments:
    -i, --inputPath: str, required
        The path to the input data.
    -o, --outputPath: str, required
        The path for the output data.
    -t, --type: str, required
        The type of input data. Choose between "inchi" and "smiles".
    -f, --filter_size: int, optional, default=40
        The maximum number of heavy atoms tolerated in both substrate and metabolite prior to running redox matching or MCS matching.
        The runtime can get very high for large molecules.

Example usage:
    python run.py -i input.csv -o output/ -t smiles -f 40
"""

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromInchi, MolFromSmiles, MolToInchi, PandasTools
from tqdm import tqdm

from soman.annotator import annotate_soms
from soman.utils import concat_lists, curate_data, log, symmetrize_soms

np.random.seed(seed=42)
tqdm.pandas()


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser("Predicting SoMs for unseen data.")

    parser.add_argument(
        "-i",
        "--inputPath",
        type=str,
        required=True,
        help="The path to the input data.",
    )

    parser.add_argument(
        "-o",
        "--outputPath",
        type=str,
        required=True,
        help="The path for the output data.",
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        help='The type of input data. Choose between "inchi" and "smiles".',
    )

    parser.add_argument(
        "-f",
        "--filter_size",
        type=int,
        required=False,
        default=45,
        help="The maximum number of heavy atoms tolerated in both substrate and metabolite prior to running redox matching or MCS matching.\
              The runtime can get very high for large molecules.",
    )

    parser.add_argument(
        "-e",
        "--ester_hydrolysis",
        required=False,
        help="Per default, SOMAN annotates ester hydrolyses with the same logic as dealkylation reactions.\
              If the -e argument is set, the annotation of ester hydrolysis is consistent with MetaQSAR.",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    logger_path = os.path.join(args.outputPath, "logs.txt")

    if os.path.exists(args.outputPath):
        shutil.rmtree(args.outputPath)
    os.makedirs(args.outputPath)

    data = pd.read_csv(args.inputPath, header="infer")

    if "substrate_id" not in data.columns:
        data["substrate_id"] = "n.a."
    if "substrate_name" not in data.columns:
        data["substrate_name"] = "n.a."
    if "metabolite_id" not in data.columns:
        data["metabolite_id"] = "n.a."
    if "metabolite_name" not in data.columns:
        data["metabolite_name"] = "n.a."

    if args.type == "inchi":
        data["substrate_mol"] = data.iloc[:, 0].map(MolFromInchi)
        data["metabolite_mol"] = data.iloc[:, 1].map(MolFromInchi)
    elif args.type == "smiles":
        data["substrate_mol"] = data.iloc[:, 0].map(MolFromSmiles)
        data["metabolite_mol"] = data.iloc[:, 1].map(MolFromSmiles)
    else:
        raise ValueError("Invalid type argument.")

    data.rename(
        columns={
            data.columns[0]: "substrate_string",
            data.columns[1]: "substrate_string",
        },
        inplace=True,
    )
    log(logger_path, f"Data set contains {len(data)} reactions.")

    data = curate_data(data, logger_path)
    # data = standardize_data(data, logger_path)

    # Predict SoMs and re-annotate topologically symmetric SoMs
    data[["soms", "annotation_rule"]] = data.progress_apply(
        lambda x: annotate_soms(
            (x.substrate_mol, x.substrate_id),
            (x.metabolite_mol, x.metabolite_id),
            logger_path,
            filter_size=args.filter_size,
            ester_hydrolysis=args.ester_hydrolysis,
        ),
        axis=1,
        result_type="expand",
    )
    data["soms"] = data.apply(
        lambda x: symmetrize_soms(x.substrate_mol, x.soms), axis=1
    )

    # Output annotations
    PandasTools.WriteSDF(
        df=data,
        out=os.path.join(args.outputPath, "substrates.sdf"),
        molColName="substrate_mol",
        properties=[
            "substrate_id",
            "substrate_name",
            "metabolite_id",
            "metabolite_name",
            "soms",
            "annotation_rule",
        ],
    )

    PandasTools.WriteSDF(
        df=data,
        out=os.path.join(args.outputPath, "metabolites.sdf"),
        molColName="metabolite_mol",
        properties=[
            "substrate_id",
            "substrate_name",
            "metabolite_id",
            "metabolite_name",
            "soms",
        ],
    )

    #################### Merge all soms from the same substrates and output annotated data ####################
    # One substrate can undergo multiple reactions, leading to multiple metabolites.
    # This step merges all the soms from the same substrate and outputs the data in a single SDF file.

    data["substrate_inchi"] = data.substrate_mol.map(MolToInchi)
    data_grouped = data.groupby("substrate_inchi", as_index=False).agg(
        {"soms": concat_lists, "substrate_id": list}
    )
    data_grouped_first = data.groupby("substrate_inchi", as_index=False).first()[
        ["substrate_inchi", "substrate_name", "substrate_mol"]
    ]

    data_merged = data_grouped.merge(data_grouped_first, how="inner")

    PandasTools.WriteSDF(
        df=data_merged,
        out=os.path.join(args.outputPath, "merged.sdf"),
        molColName="substrate_mol",
        properties=["substrate_id", "substrate_name", "soms"],
    )

    log(
        logger_path,
        f"Average number of soms per compound: {round(np.mean(np.array([len(lst) for lst in data_merged.soms.values])), 2)}",
    )
    log(logger_path, f"Total runtime: {datetime.now() - start}")
