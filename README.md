**AutoSOM**: A pipeline to automatically annotate the Sites of Metabolism (SOMs) of substrate-metabolite pairs.

### Installation

1. Create a conda environment with the required python version:

```sh
conda create --name autosom-env python=3.11
```

2. Activate the environment:

```sh
conda activate autosom-env
```

3. Install package:

```sh
pip install autosom
```


### Usage

To annotate data, please run:

```sh
python scripts/run.py -i INPUT_PATH -o OUTPUT_PATH -t[OPTIONAL] -e[OPTIONAL]
```

The `INPUT_PATH` is the path to your input data. The file format must be .csv. It should contain a "substrate_smiles" and a "metabolite_smiles" column containing the SMILES string of the substrate and metabolite, respectively, and a "substrate_id" column and "metabolite_id" column containing numerical identifiers of the substrate and metabolite, respectively. Any number and naming of additional column(s) is allowed. The ordering of columns is not important.

The `OUTPUT_PATH` is the path where the output (annotated) data as well as the log file will be written.

The `-t` flag is optional and controls the number of seconds allowed for the annotator to complete. Default is 20 seconds.

The `-e` flag controls is optional and controls the strategy for annotating ester hydrolyses. Per default, AutoSOM annotates ester hydrolyses with the same logic as dealkylation reactions (on the alkyl C-atom). If the -e argument is set, the annotation is on the carbonyl C-atom, which is consistent with the MetaQSAR data set.


### Visualization

You can use the `visualize_results` Jupyter Notebook to visualize your results. For this, you'll first need to install the `ipykernel` and `ipywidgets` packages with pip.
