**SOMAN**: A pipeline to automatically annotate the Sites of Metabolism (SoMs) from substrate-metabolite pairs. SoMs, also know as metabolic hotspots, are the atoms where metabolic reactions are initiated.

### Installation

Clone the repository and cd into the repository root:

`git clone https://github.com/molinfo-vienna/SOMAN.git`

`cd SOMAN`

Create a conda environment with the required python version:

`conda env create --name soman python=3.11`

Activate the environment:

`conda activate soman`

Install soman package:

`pip install -e .`


### Usage

To annotate data, please run:

`python scripts/run.py -i INPUT_PATH -o OUTPUT_PATH -t TYPE`

The `INPUT_PATH` is the path to your input data. The file format must be .csv. The first and second columns should contain either smiles or inchi of the substrate and metabolite, respectively.

The `OUTPUT_PATH` is the path where the output (annotated) data as well as the log file will be written.

The `TYPE` indicates whether the input data contains SMILES of InChIs. Please choose between `smiles` and `inchi`.


### Sandbox

You can use the `sandbox` Jupyter Notebook to visualize your results. For this, you'll first need to install the `ipykernel` and `ipywidgets` packages with pip.
