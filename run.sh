#!/usr/bin/env bash
for dataset in cora citeseer pubmed
do
python autoencoder.py --dataset=${dataset}
done