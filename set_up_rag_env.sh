#!/usr/bin/env bash

# Env setup for Ola:
module purge
module load cuda/10.1 NCCL/2.5.6-1-cuda.10.1 anaconda3/5.0.1
conda create -y -n rag-hf python=3.7
conda activate rag-hf
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

# assumes you have already done git clone git@github.com:patrick-s-h-lewis/transformers.git, and you are in project root
git remote add upstream https://github.com/huggingface/transformers.git
git checkout -b rag origin/rag
pip install -e ".[dev]"

# additional RAG requirement is HF NLP - discuss how to handle this later - I install locally for easier debugging etc
git clone git@github.com:huggingface/nlp.git nlp
cd nlp
git checkout -b add-indexed-dataset-2 origin/add-indexed-dataset-2
pip install -e .
cd ..

# need faiss
pip install faiss-gpu
