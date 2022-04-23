# DISAE
## MSA-Regularized Protein Sequence Transformer toward Predicting Genome-Wide Chemical-Protein Interactions: Application to GPCRome Deorphanization


Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg


This is the repository to replicate experiments for the fine-tuning of classifier with pretrained ALBERT in the paper [DISAE](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c01285).
## ----------- INSTRUCTION -----------
### 1. Install Prerequisites
- python 3.7
- Pytorch
- rdkit
- Transformers ([Huggingface](https://huggingface.co/transformers/). version 2.3.0)

### 2. Clone this repository

### 3. Download Data
All data could be download [here](https://zenodo.org/record/4892316#.YLcxR6hKiUk) and put it under this repository, i.e. in the same directory as the finetuning_train.py.

There will be four subdirectories in the data folder.

![image](https://user-images.githubusercontent.com/33879882/88445795-246b9a00-cdf3-11ea-9757-1afabd87dc39.png)

- activity: gives you the  train/dev/test set split based on protein similarity at threshold of bitscore 0.035
- albertdata: gives you pretrained ALBERT model. The ALBERT is pretraind on distilled triplets of whole Pfam
- Integrated: gives collected chemicals from several database
- protein: gives you mapping from uniprot ID to triplets form

### 4. Generate clusters:
    1. Cluster your protein dataset with `cdhit.sh`. Input is fasta file with all protein sequences in your dataset.
    2. Apply multi-sequence alignment to the clusters with Clustal Omega. (`clustalo.sh`)
    3. Build hmm profiles for the clusters with hmmbuild. (`hmmer_build.sh`)
    4. Redo multi-sequence alignment with the hmm profiles and HMP clusters with HMMER. (`hmmer_align.sh`)
    5. Construct corpus (singlets and triplets, represent sequence and all sequences) with `construct_hmp_singlets_and_triplets.py`. This step could take long if use only one CPU. Multiprocessing can significantly reduce computing time.
    6. Generate TFRecord with the corpus with `create_tfrecords.sh`.

### 5. Run Finetuning
To run ALBERT model (default: ALBERRT frozen transformer):
```
python finetuning_train.py --protein_embedding_type="albert"
```
To try other freezing options, change "frozen_list" to choose modules to be frozen.


To run LSTM model:
```
python finetuning_train.py --protein_embedding_type="lstm"
```


![distilled-and-architecture](https://user-images.githubusercontent.com/33879882/91499613-b9006680-e88f-11ea-8374-fe7ce529666a.png)

