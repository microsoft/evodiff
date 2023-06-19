#rm -rf oaardm-seq/esmif/
#rm -rf oaardm-seq/pdb/

NUM_SEQS=19 #num_seqs-1
run='/home/v-salamdari/Desktop/DMs/blobfuse/d3pm/random-640M'
lengths=(32 64 128 256) # 384 512) # 1024 2048)
#out='/home/v-salamdari/Desktop/ProteinMPNN/dms_examples/random-01-seq'


for i in "${lengths[@]}"; do
	if [ ! -d $run/pdb/$i ] ; then
		mkdir -p $run/pdb/$i/
		mkdir -p $run/mpnn/$i/ 
		echo $i 
		omegafold $run/generated_samples_string_$i.fasta $run/pdb/$i/
	fi 
	for ((j=0; j<=$NUM_SEQS; j++ )); do 
	#for j in {0..20}; do
		echo $j
		#python sample_sequences.py $run/pdb/$i/SEQUENCE_$j.pdb --chain A --temperature 1 --num-samples 1 --outpath $run/esmif/$i/sampled_sequences_$j.fasta
		#python score_log_likelihoods.py $run/pdb/$i/SEQUENCE_$j.pdb $run/esmif/$i/sampled_sequences_$j.fasta --chain A --outpath $run/esmif/$i/sequence_scores_$j.csv
		python ../protein_mpnn_run.py \
        		--pdb_path  $run/pdb/$i/SEQUENCE_$j.pdb \
        		--pdb_path_chains "A" \
        		--out_folder $run/mpnn/$i/sampled_sequences_$j.fasta \
        		--num_seq_per_target 1 \
        		--sampling_temp "0.1" \
        		--seed 37 \
        		--batch_size 1\
			--save_score 1
	done
done
