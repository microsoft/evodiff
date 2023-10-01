# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:23.09-py3

LABEL author="Colby T. Ford <colby@tuple.xyz>"

## Environment Settings
ENV DEBIAN_FRONTEND=noninteractive

## Install Basic Dependencies
RUN apt-get clean && \
    apt-get update && \
    apt-get -y install \
        sudo \
        git \
        curl \
        wget

## Install Conda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#     /bin/bash ~/miniconda.sh -b -p /opt/conda
# ENV PATH=$CONDA_DIR/bin:$PATH

## Pip Install Requirements
RUN python -m pip install \
        pandas \
        numpy \
        lmdb \
        git+https://github.com/microsoft/protein-sequence-models.git \
        mlflow \
        scikit-learn \
        seaborn \
        blosum \
        matplotlib \
        fair-esm \
        biotite \
        torch \
        torchvision \
        torch_geometric \
        torch_scatter \
        tqdm \
        git+https://github.com/HeliXonProtein/OmegaFold.git \
        mdanalysis \
        pdb-tools

## Clone EvoDiff GitHub Repo (to get scripts)
RUN git clone https://github.com/microsoft/evodiff && \
    cd evodiff && \
    git checkout 012b943

## Set Working Directory
WORKDIR /workspace/evodiff

## Pip Install EvoDiff
RUN python -m pip install evodiff

## Get UniRef50 Data
# RUN wget https://evodiff.blob.core.windows.net/evodiff-data/uniref50.tar.gz && \
#     tar -xzf uniref50.tar.gz && \
#     rm uniref50.tar.gz