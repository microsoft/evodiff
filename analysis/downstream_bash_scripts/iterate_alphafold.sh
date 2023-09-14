#!/bin/bash

# Generate MSA for sequence, and get alphafold prediction

strings=(
        #1qjg
        #1prw
        #1bcf
        #5tpn
        #5wn9
        #5ius
        #1ycr
        #7p19
        #2kl8
        #7mrx
        #5trv
        #6exz
        #6e6r
        #6vw1
        #4jhw
        #4zyp
        #3ixt
        4qvf
        2xa0
        )

DOWNLOAD_DIR=/data/ALPHAFOLD_DOWNLOAD_DIR
output_dir=/home/v-salamdari/Desktop/DMs/cond-gen/scaffolding-pdbs/

for i in "${strings[@]}"; do

fasta_path=/home/v-salamdari/Desktop/DMs/cond-gen/scaffolding-pdbs/${i}.fasta

python3 docker/run_docker.py \
  --fasta_paths=$fasta_path \
  --max_template_date=2023-07-13 \
  --data_dir=$DOWNLOAD_DIR \
  --output_dir=$output_dir
  --models_to_relax=best
  --num_multimer_predictions_per_model=1
done