from setuptools import setup, find_packages

setup(
    name="soman",
    version="1.0",
    author="Roxane Jacob",
    description="Annotates the Sites of Metabolism (SoMs) of substrate-metabolite pairs.",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "networkx==3.3",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "rdkit==2024.3.1",
        "tqdm==4.66.4",
    ],
)
