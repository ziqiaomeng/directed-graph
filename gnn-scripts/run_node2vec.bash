#!/usr/bin/env bash

for dataset in cora citeseer pubmed
do
model=node2vec
python ${model}/src/main.py \
      --input ../dataset/processed/${dataset}/${dataset}_train.txt \
      --output ./output/${model}/${dataset}.embeddings
done