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
ENV EVODIFF_VERSION="1.1.0"
# ENV REPO_OWNER="colbyford"
ENV REPO_OWNER="microsoft"
RUN mkdir evodiff && \
    wget -O evodiff.tar.gz https://codeload.github.com/${REPO_OWNER}/evodiff/tar.gz/refs/tags/v${EVODIFF_VERSION} && \
    tar -xzf evodiff.tar.gz && \
    mv evodiff-${EVODIFF_VERSION}/* evodiff && \
    rm evodiff.tar.gz

## Set Working Directory
WORKDIR /workspace/evodiff

## Install EvoDiff (+ dependencies)
RUN pip install -r requirements.txt

## Get UniRef50 Data
RUN cd data && \
    # wget https://evodiff.blob.core.windows.net/evodiff-data/uniref50.tar.gz && \
    wget -O uniref50.tar.gz https://zenodo.org/records/6564798/files/uniref50.tar.gz?download=1 && \
    tar -xzf uniref50.tar.gz && \
    rm uniref50.tar.gz

