import argparse
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm

from rdkit.Chem import MolFromInchi, MolFromSmiles, MolToInchi, PandasTools

from soman.soman import get_soms
from soman.utils import (
    concat_lists,
    curate_data,
    filter_data,
    standardize_data,
    symmetrize_soms,
)

np.random.seed(seed=42)
tqdm.pandas()


def run():
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
        data["substrate_mol"] = data.iloc[:, 0].map(lambda x: MolFromInchi(x))
        data["metabolite_mol"] = data.iloc[:, 1].map(lambda x: MolFromInchi(x))
    elif args.type == "smiles":
        data["substrate_mol"] = data.iloc[:, 0].map(lambda x: MolFromSmiles(x))
        data["metabolite_mol"] = data.iloc[:, 1].map(lambda x: MolFromSmiles(x))
    else:
        raise ValueError("Invalid type argument.")

    data.rename(
        columns={
            data.columns[0]: "substrate_string",
            data.columns[1]: "substrate_string",
        },
        inplace=True,
    )
    print(f"Data set contains {len(data)} reactions.")

    data = curate_data(data)
    data = filter_data(data, 30)
    # data = standardize_data(data)

    # Predict SoMs and re-annotate topologically symmetric SoMs
    data["soms"] = data.progress_apply(
        lambda x: get_soms(
            x.substrate_mol,
            x.metabolite_mol,
            x.substrate_id,
            x.metabolite_id,
            logger_path=os.path.join(args.outputPath, "logs.txt"),
        ),
        axis=1,
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

    data["substrate_inchi"] = data.substrate_mol.map(lambda x: MolToInchi(x))
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

    print(
        f"Average number of soms per compound: {round(np.mean(np.array([len(lst) for lst in data_merged.soms.values])), 2)}"
    )


if __name__ == "__main__":
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

    args = parser.parse_args()
    run()
