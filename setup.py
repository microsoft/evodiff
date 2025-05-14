import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evodiff-test2",
    version="1.0.1",
    description="Python package for generation of protein sequences and evolutionary alignments via discrete diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/evodiff",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "biopython",
        "blosum",
        "fair-esm",
        "lmdb",
        "matplotlib",
        "mdanalysis",
        "numpy>=1.25,<2.0",
        "pandas",
        "pdb-tools",
        "requests",
        "scikit-learn",
        "seaborn",
        "sequence-models",
        "torch_geometric",
        "torch-scatter",
        "tqdm"
    ],
    python_requires='>=3.9.0',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'': ['config/*']},
)