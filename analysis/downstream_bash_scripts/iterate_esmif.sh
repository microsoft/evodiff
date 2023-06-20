#rm -rf oaardm-seq/esmif/
#rm -rf oaardm-seq/pdb/

NUM_SEQS=19 #num_seqs-1
run='sequence/oaardm'
lengths=(32 64 128 256) # 384 512) # 1024 2048)

for i in "${lengths[@]}"; do
	if [ ! -d $run/pdb/$i ] ; then
		mkdir -p $run/pdb/$i/
		mkdir -p $run/esmif/$i/ 
		echo $i 
		omegafold $run/generated_samples_string_$i.fasta $run/pdb/$i/
	fi 
	for ((j=0; j<=$NUM_SEQS; j++ )); do 
	#for j in {0..20}; do
		echo $j
		python sample_sequences.py $run/pdb/$i/SEQUENCE_$j.pdb --chain A --temperature 1 --num-samples 1 --outpath $run/esmif/$i/sampled_sequences_$j.fasta
		
		python score_log_likelihoods.py $run/pdb/$i/SEQUENCE_$j.pdb $run/esmif/$i/sampled_sequences_$j.fasta --chain A --outpath $run/esmif/$i/sequence_scores_$j.csv
	done
done
