#!/usr/bin/env bash
for dataset in cora citeseer pubmed
do
python dataloader.py --dataset ${dataset}
done