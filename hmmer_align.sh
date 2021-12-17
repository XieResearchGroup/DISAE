#!/bin/bash

root=Data/HMDB/HMDB_hmm
fastaroot=Data/HMDB/HMDB_clustered_fasta
saveroot=Data/HMDB/HMDB_hmmer_aln

mkdir ${saveroot}

for filename in ${root}/*
do
    basename=${filename##*/}
    clustername=${basename%%.*}
    echo ${clustername}
    hmmalign \
        -o ${saveroot}/${clustername}.sto \
        ${filename} ${fastaroot}/${clustername}.fasta > hmmalign.log 2>&1
done
