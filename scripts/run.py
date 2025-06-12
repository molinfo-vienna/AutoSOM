"""Predicts Sites of Metabolism (SOMs) for unseen data. Uses pairs of molecular \
structures (substrate/metabolite) provided in SMILES format.

The script performs the following steps:
1. Parses command-line arguments to get input and output paths, input data type, and filter size.
2. Reads the input data from a CSV file.
3. Ensures necessary columns are present in the data.
4. Converts molecular structures from SMILES to RDKit Mol objects.
5. Curates the data and predicts SOMs for each reaction.
6. Symmetrizes the predicted SOMs.
7. Outputs the annotated data to SDF files.
8. Merges all SOMs from the same substrates and outputs the merged data to a single SDF file.

Command-line arguments:
    -i, --inputPath: str, required
        The path to the input data.
    -o, --outputPath: str, required
        The path for the output data.
    -t, --timeout: int, optional, default=20
        The timeout for the SOM annotation in seconds.
    -e, --ester_hydrolysis: bool, optional
        Per default, SOMAN annotates ester hydrolyses with the same logic as dealkylation reactions.
        If the -e argument is set, the annotation of ester hydrolysis is consistent with MetaQSAR.

Example usage:
    python run.py -i input.csv -o output/ -f 30 -e
"""

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools  # type: ignore
from rdkit.Chem import MolFromSmiles, MolToSmiles
from tqdm import tqdm

from autosom.autosom import annotate_soms
from autosom.utils import (
    check_and_collapse_substrate_id,
    concat_lists,
    log,
    symmetrize_soms,
)

np.random.seed(seed=42)
tqdm.pandas()


if __name__ == "__main__":
    start = datetime.now()
    parser = argparse.ArgumentParser("Predicting SOMs for unseen data.")

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
        "--timeout",
        type=int,
        required=False,
        default=20,
        help="The timeout for the SOM annotation in seconds.",
    )

    parser.add_argument(
        "-e",
        "--ester_hydrolysis",
        required=False,
        help="Per default, SOMAN annotates ester hydrolyses with \
            the same logic as dealkylation reactions.\
                If the -e argument is set, the annotation \
                    of ester hydrolysis is consistent with MetaQSAR.",
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    logger_path = os.path.join(args.outputPath, "logs.txt")

    if os.path.exists(args.outputPath):
        shutil.rmtree(args.outputPath)
    os.makedirs(args.outputPath)

    data = pd.read_csv(args.inputPath, header="infer")

    data["substrate_mol"] = data.substrate_smiles.map(MolFromSmiles)
    data["metabolite_mol"] = data.metabolite_smiles.map(MolFromSmiles)

    log(logger_path, f"Data set contains {len(data)} reactions.")

    # Predict SOMs
    params = (logger_path, args.timeout, args.ester_hydrolysis)
    data[["soms", "annotation_rule", "time"]] = data.progress_apply(
        lambda x: annotate_soms(
            params,
            (x.substrate_mol, x.substrate_id),
            (x.metabolite_mol, x.metabolite_id),
        ),
        axis=1,
        result_type="expand",
    )
    # Re-annotate topologically symmetric SoMs
    data["soms"] = data.apply(
        lambda x: symmetrize_soms(x.substrate_mol, x.soms), axis=1
    )

    # Output annotations
    data = data.dropna(
        subset=["substrate_mol", "metabolite_mol"]
    )  # Drop rows with None values (invalid substrate or metabolite Mol objects)
    data["sdf_id"] = data["substrate_id"]
    PandasTools.WriteSDF(
        df=data,
        out=os.path.join(args.outputPath, "substrates.sdf"),
        idName="sdf_id",
        molColName="substrate_mol",
        properties=[column for column in data.columns if "mol" not in column],
    )

    PandasTools.WriteSDF(
        df=data,
        out=os.path.join(args.outputPath, "metabolites.sdf"),
        idName="sdf_id",
        molColName="metabolite_mol",
        properties=[column for column in data.columns if "mol" not in column],
    )

    data.to_csv(
        os.path.join(args.outputPath, "annotated_data.csv"),
        columns=[column for column in data.columns if "mol" not in column],
        header=True,
        index=False,
    )

    # Merge all soms from the same substrates and output annotated data
    # One substrate can undergo multiple reactions, leading to multiple metabolites.
    # This step merges all the soms from the same substrate and outputs the data
    # in a single SDF file.

    data["substrate_canonical_smiles"] = data["substrate_mol"].map(MolToSmiles)
    data["metabolite_canonical_smiles"] = data["metabolite_mol"].map(MolToSmiles)

    data_grouped = data.groupby("substrate_canonical_smiles", as_index=False).agg(
        {"soms": concat_lists, "substrate_id": check_and_collapse_substrate_id}
    )
    # Get only the first entry if multiple entries exist for the same substrate
    data_grouped_first = data.groupby(
        "substrate_canonical_smiles", as_index=False
    ).first()[
        [
            column
            for column in data.columns
            if "metabolite" not in column
            and "annotation_rule" not in column
            and "soms" not in column
        ]
    ]
    data_grouped_first["substrate_id"] = data_grouped_first["substrate_id"].astype(int)

    data_merged = data_grouped.merge(data_grouped_first, how="inner")
    data_merged["sdf_id"] = data_merged["substrate_id"]
    PandasTools.WriteSDF(
        df=data_merged,
        out=os.path.join(args.outputPath, "merged.sdf"),
        idName="sdf_id",
        molColName="substrate_mol",
        properties=[column for column in data_merged.columns if "mol" not in column],
    )

    log(
        logger_path,
        f"Average number of soms per compound: \
        {round(np.mean(np.array([len(lst) for lst in data_merged.soms.values])), 2)}",
    )
    log(logger_path, f"Total runtime: {datetime.now() - start}")
