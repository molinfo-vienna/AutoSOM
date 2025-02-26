**AutoSOM**: A pipeline to automatically annotate the Sites of Metabolism (SoMs) from substrate-metabolite pairs. SoMs, also know as metabolic hotspots, are the atoms where metabolic reactions are initiated.

### Installation

Clone the repository and cd into the repository root:

`git clone https://github.com/molinfo-vienna/AutoSOM.git`

`cd AutoSOM`

Create a conda environment with the required python version:

`conda create --name autosom-env python=3.11`

Activate the environment:

`conda activate autosom-env`

Install autosom package:

`pip install -e .`


### Usage

To annotate data, please run:

`python scripts/run.py -i INPUT_PATH -o OUTPUT_PATH -t TYPE -f FILTER_SIZE -e`

The `INPUT_PATH` is the path to your input data. The file format must be .csv. It should contain a "substrate_smiles" and a "metabolite_smiles" column containing the SMILES string of the substrate and metabolite, respectively, and a "substrate_id" column and "metabolite_id" column containing numerical identifiers of the substrate and metabolite, respectively. Any number and naming of additional column(s) is allowed. The ordering of columns is not important.

The `OUTPUT_PATH` is the path where the output (annotated) data as well as the log file will be written.

The `FILTER_SIZE` indicates the maximum number of heavy atoms tolerated in both substrate and metabolite prior to running some MCS matching operations. The default value is 45. The lower the value, the faster the algorithm runs, but the more reactions are filtered out.

The `-e` flag controls the strategy for annotating ester hydrolyses. Per default, AutoSOM annotates ester hydrolyses with the same logic as dealkylation reactions (on the alkyl C-atom). If the -e argument is set, the annotation is on the carbonyl C-atom, which is consistent with the MetaQSAR data set.


### Sandbox

You can use the `visualize_results` Jupyter Notebook to visualize your results. For this, you'll first need to install the `ipykernel` and `ipywidgets` packages with pip. You can choose this option directly when installing the package by running:

`pip install -e .[vis]`
