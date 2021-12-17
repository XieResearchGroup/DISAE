#!/bin/bash

root=Data/HMDB/HMDB_clustalo_aln_fasta_format
saveroot=Data/HMDB/HMDB_hmm

mkdir ${saveroot}

for filename in ${root}/*
do
    base=${filename##*/}
    hmmbuild ${saveroot}/${base%%.*}.hmm ${filename} > hmmbuild.log 2>&1
done
