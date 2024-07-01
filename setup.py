from setuptools import setup, find_packages

setup(
    name="soman",
    version="1.0",
    author="Roxane Jacob",
    description="Annotates the Sites of Metabolism (SoMs) of substrate-metabolite pairs.",
    python_requires="==3.12.4",
    packages=find_packages(),
    install_requires=[
        "networkx==3.3",
        "rdkit==2024.03.3",
        "tqdm==4.66.4",
    ],
)
