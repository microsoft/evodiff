import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evodiff",
    version="1.1.0",
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
        "pandas",
        "lmdb",
        "numpy",
        "sequence-models",
        "mlflow",
        "scikit-learn",
        "blosum",
        "seaborn",
        "matplotlib",
        "fair-esm",
        "tqdm",
        "biotite",
        "requests",
        "mdanalysis",
        "pdb-tools"
    ],
    python_requires='>=3.8.5',
    include_package_data=True,
    packages=setuptools.find_packages(),
    package_data={'': ['config/*']},
)
