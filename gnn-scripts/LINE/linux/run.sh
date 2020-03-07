#!/usr/bin/env bash
#for dataset in cora citeseer pubmed cit-HepTh cit-HepPh slashdot
#for dataset in slashdot
#for dataset in cora citeseer pubmed
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result line.cpp -o line -lgsl -lm -lgslcblas
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result reconstruct.cpp -o reconstruct
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result normalize.cpp -o normalize
g++ -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result concatenate.cpp -o concatenate

for dataset in cora
do
embedding_file=../../../dataset/processed/${dataset}/${dataset}_train.txt
output=../../output/LINE/${dataset}.embeddings
./line -train ${embedding_file} -output ${output}1  -binary 0 -size 64 -order 1 -negative 5 -samples 100 -threads 40
./line -train ${embedding_file} -output ${output}2  -binary 0 -size 64 -order 2 -negative 5 -samples 10 -threads 40
./normalize -input ${output}1 -output ${output}normalized1 -binary 0
./normalize -input ${output}2 -output ${output}normalized2 -binary 0
./concatenate -input1 ${output}normalized1 -input2 ${output}normalized2 -output ${output} -binary 0
done
#./line -train net_youtube_dense.txt -output vec_2nd_wo_norm.txt -binary 1 -size 128 -order 2 -negative 5 -samples 10000 -threads 40
