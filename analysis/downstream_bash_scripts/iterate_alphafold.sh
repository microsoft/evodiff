#!/bin/bash

NUM_SEQS=51 # +1 since indexed at 1
run='/home/v-salamdari/Desktop/DMs/amlt-generate/diff-oaardm-msa-random'
seqs='valid_query' # or gen_query

for ((i=1; i<=$NUM_SEQS; i++ )); do
  echo $i
  python3 docker/run_docker.py \
    --fasta_paths=$run/gen-$i/$seqs.fasta \
    --data_dir=/data/ALPHAFOLD_DOWNLOAD_DIR/ \
    --output_dir=$run/gen-$i/ \
    --max_template_date=2023-05-05 \
    --use_precomputed_msas=True

done