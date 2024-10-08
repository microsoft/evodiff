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

## Get EvoDiff release from GitHub 
ENV EVODIFF_VERSION="v.1.0.0"
RUN mkdir evodiff && \
    wget -O evodiff.tar.gz https://codeload.github.com/microsoft/evodiff/tar.gz/refs/tags/${EVODIFF_VERSION} && \
    tar -xzf evodiff.tar.gz && \
    mv evodiff-${EVODIFF_VERSION}/* evodiff && \
    rm evodiff.tar.gz

## Set Working Directory
WORKDIR /workspace/evodiff

## Get UniRef50 Data
RUN cd data && \
    # wget https://evodiff.blob.core.windows.net/evodiff-data/uniref50.tar.gz && \
    wget -O uniref50.tar.gz https://zenodo.org/records/6564798/files/uniref50.tar.gz?download=1 && \
    tar -xzf uniref50.tar.gz && \
    rm uniref50.tar.gz

## Install EvoDiff (+ dependencies)
RUN python setup.py install