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
        wget

## Pip Install EvoDiff
RUN pip install evodiff

## Clone EvoDiff GitHub Repo (to get scripts)
RUN git clone https://github.com/microsoft/evodiff && \
    cd evodiff

## Set Working Directory
WORKDIR /workspace/evodiff

## Get UniRef50 Data
# RUN wget -O data/uniref50.tar.gz https://evodiff.blob.core.windows.net/evodiff-data/uniref50.tar.gz && \
RUN wget -O data/uniref50.tar.gz https://zenodo.org/records/6564798/files/uniref50.tar.gz?download=1 && \
    tar -xzf data/uniref50.tar.gz && \
    rm data/uniref50.tar.gz
