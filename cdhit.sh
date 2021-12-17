#!/bin/bash


#############################################################
# To get clusters based on sequence similarity with cd-hit  #
# cd-hit package can be installed with conda. However, the  #
# psi-cd-hit.pl will not directly be available after the    #
# installation. Path to the source file need to be added to #
# PATH manually. psi-cd-hid.pl also needs blastp, which can #
# also be installed with conda.                             #
#############################################################

root=Data/HMDB
saveroot=${root}/HMDB_clustered_corpora
n_tread=100
f=protein.fasta

mkdir ${saveroot}/hmdb_cdhit_clusters90
mkdir ${saveroot}/hmdb_cdhit_clusters60
mkdir ${saveroot}/hmdb_cdhit_clusters30
mkdir ${saveroot}/hmdb_cdhit_clusters9060
mkdir ${saveroot}/hmdb_cdhit_clusters906030

cd-hit  -i ${root}/${f} \
	-o ${saveroot}/hmdb_cdhit_clusters90/hmdb_cluster90 -c 0.9 -n 5 \
	-g 1 -G 0 -aS 0.8 \
	-d 100 -p 1 -T ${n_tread} -M 0 >> hmdb_cdhit_cluster.log
cd-hit  -i ${root}/${f} \
	-o ${saveroot}/hmdb_cdhit_clusters60/hmdb_cluster60 -c 0.6 -n 4 \
	-g 1 -G 0 -aS 0.8 \
	-d 100 -p 1 -T ${n_tread} -M 0 >> hmdb_cdhit_cluster.log

psi-cd-hit.pl -i ${root}/${f} -o ${saveroot}/hmdb_cdhit_clusters30/hmdb_cluster30 \
	-c 0.3 -ce 1e-6 -aS 0.8 -G 0 -g 1 -exec local -para 8 -blp 4

clstr_rev.pl \
	${saveroot}/hmdb_cdhit_clusters90/hmdb_cluster90.clstr \
	${saveroot}/hmdb_cdhit_clusters60/hmdb_cluster60.clstr \
	> ${saveroot}/hmdb_cdhit_clusters9060/hmdb_cluster9060.clstr

clstr_rev.pl \
	${saveroot}/hmdb_cdhit_clusters9060/hmdb_cluster9060.clstr \
	${saveroot}/hmdb_cdhit_clusters30/hmdb_cluster30.clstr \
	> ${saveroot}/hmdb_cdhit_clusters906030/hmdb_cluster906030.clstr

