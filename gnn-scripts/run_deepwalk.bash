#!/usr/bin/env bash


#for dataset in cora citeseer pubmed cit-HepTh cit-HepPh slashdot
for dataset in cora citeseer pubmed
do
python deepwalk/deepwalk/__main__.py \
      --format edgelist \
      --input ../dataset/processed/${dataset}/${dataset}_train.txt \
      --output ./output/deepwalk/${dataset}.embeddings
done