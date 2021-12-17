#!/bin/bash

root=Data/HMDB/HMDB_clustered_fasta
saveroot=Data/HMDB/HMDB_clustalo_aln_fasta_format

# rm -r ${saveroot}
mkdir ${saveroot}

for filename in ${root}/*
do
    clustalo -i $filename -t Protein --infmt=fasta \
        -o ${saveroot}/${filename##*/}.sto --outfmt=fa \
        --threads=100 > clustalo.log 2>&1
done
