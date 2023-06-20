home='/home/v-salamdari/Desktop/DMs/blobfuse'
runs=('test-data') #"d3pm/blosum-640M-0" "d3pm/random-640M-0" "d3pm/oaardm-640M" "d3pm/soar-640M" "esm-1b" "sequence/blosum-0-seq" "sequence/oaardm" "d3pm-final/random-0-seq" "pretrain21/cnn-38M") # "d3pm/blosum-640M-0")

NUM_SEQS=249 #num_seqs-1
for run in "${runs[@]}"; do
lengths=(64 128 256 384)
for i in "${lengths[@]}"; do
      if [ ! -d $home/$run/pdb/$i ] ; then
            mkdir -p $home/$run/pdb/$i/
            mkdir -p $home/$run/mpnn/$i/
            echo $i
            omegafold $home/$run/generated_samples_string_$i.fasta $home/$run/pdb/$i/
      fi
      for ((j=0; j<=$NUM_SEQS; j++ )); do
            echo $j
            python ../protein_mpnn_run.py \
                    --pdb_path  $home/$run/pdb/$i/SEQUENCE_$j.pdb \
                    --pdb_path_chains "A" \
                    --out_folder $home/$run/mpnn/$i/sampled_sequences_$j.fasta \
                    --num_seq_per_target 1 \
                    --sampling_temp "0.1" \
                    --seed 37 \
                    --batch_size 1\
                    --save_score 1
      done
done
done
