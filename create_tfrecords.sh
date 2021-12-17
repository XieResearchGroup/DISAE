#!/usr/bin/env bash

# where to output tfrecords
OUTPUT_DIR=Data/HMP_metagenome/HMP_corpus/triplets_tfrecords/
# for debug
# OUTPUT_DIR=test/HMP_corpus/triplets_tfrecords/
# where are the corpora
filelist=( Data/HMP_metagenome/HMP_corpus/triplets_wo_IDs/* )
# for debug
# filelist=( test/HMP_corpus/triplets_wo_IDs/* )
num_files=${#filelist[@]} # 102397

# triplet vocabs differ from singlet vocabs
vocabfile=Data/Vocab/vocab_triplets.txt
max_iter=1600
# # how many files to process at a time : (chunk_size * max_iter) > num_files
chunk_size=64

# num_files=5 # for debugging
# max_iter=2 # for debugging
# chunk_size=3 # for debugging

# Albert-specific settings
dupefactor=20 # coverage = maskedprob*dupefactor
maxseqlen=256
maxpredperseq=40
maskedprob=0.15

for i in $( seq 0 $max_iter )
do
	if [ $i -eq $max_iter ]; then
		i1=$(( i*chunk_size))
		i2=$(( num_files - 1 ))
	else
		i1=$(( i*chunk_size ))
		i2=$(( i*chunk_size + chunk_size -1 ))
	fi

	for j in $( seq $i1 $i2)
	do
		infile=${filelist[$j]}
		base=${infile##*/}
        base=${base%%.*}
		if [ $j -eq $i2 ]; then
		    python -m microbiomemeta.data.create_pretraining_data \
			    --do_whole_word_mask \
			    --input_file=$infile \
			    --output_file=${OUTPUT_DIR}${base}.tfrecord \
			    --vocab_file=${vocabfile} \
			    --do_lower_case \
			    --dupe_factor=${dupefactor} \
			    --random_seed=${RANDOM} \
			    --max_seq_length=${maxseqlen} --max_predictions_per_seq=${maxpredperseq} --masked_lm_prob=${maskedprob} &&
			sleep 1s
		else
		    python -m microbiomemeta.data.create_pretraining_data \
			    --do_whole_word_mask \
			    --input_file=$infile \
			    --output_file=${OUTPUT_DIR}${base}.tfrecord \
			    --vocab_file=${vocabfile} \
			    --do_lower_case \
			    --dupe_factor=${dupefactor} \
			    --random_seed=${RANDOM} \
			    --max_seq_length=${maxseqlen} --max_predictions_per_seq=${maxpredperseq} --masked_lm_prob=${maskedprob} &
		fi
	done
done
