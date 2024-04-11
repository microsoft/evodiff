FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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
        wget \
        g++

## Pip Install EvoDiff (and other dependencies)
RUN pip install evodiff \
        torch_geometric \
        torch_scatter

## Clone EvoDiff GitHub Repo (to get scripts)
RUN git clone https://github.com/microsoft/evodiff

## Set Working Directory
WORKDIR /workspace/evodiff

## Get UniRef50 Data
RUN cd data && \
    # wget https://evodiff.blob.core.windows.net/evodiff-data/uniref50.tar.gz && \
    wget -O uniref50.tar.gz https://zenodo.org/records/6564798/files/uniref50.tar.gz?download=1 && \
    tar -xzf uniref50.tar.gz && \
    rm uniref50.tar.gz
