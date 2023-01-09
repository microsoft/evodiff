import csv
import argparse
import torch
import numpy as np
import pandas as pd
import os
import json
from scipy.stats import spearmanr
from sequence_models.datasets import A2MZeroShotDataset
from sequence_models.constants import MSA_ALPHABET, PROTEIN_ALPHABET, ALL_AAS, PAD
from dms.utils import Tokenizer
from dms.model import ByteNetLMTime
from tqdm import tqdm

## Takes in csv file with generated sequences and performs downstream tasks
def main():
    _ = torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('config_fpath')
    parser.add_argument('out_fpath', type=str, nargs='?', default=os.getenv('PT_OUTPUT_DIR', '/tmp') + '/')
    parser.add_argument('--task', type=str,  default='seq')
    parser.add_argument('--mask', type=str,  default='autoreg')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--final_norm', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    args = parser.parse_args()

    _ = torch.manual_seed(0)
    np.random.seed(0)

    with open(args.config_fpath, 'r') as f:
        config = json.load(f)

    d_embed = config['d_embed']
    d_model = config['d_model']
    n_layers = config['n_layers']
    kernel_size = config['kernel_size']
    r = config['r']
    if 'rank' in config:
        weight_rank = config['rank']
    else:
        weight_rank = None
    if 'slim' in config:
        slim = config['slim']
    else:
        slim = True
    if 'activation' in config:
        activation = config['activation']
    else:
        activation = 'relu'

    # relevant dirs
    home = '/home/v-salamdari/'
    data_top_dir = home + 'Desktop/DMs/data/'
    zero_shot_home = home + 'Desktop/from_data_blobfuse/zero_shot/'
    protein_gym_home = zero_shot_home + 'ProteinGym/'
    checkpoint_dir = home + 'Desktop/from_data_blobfuse/'

    # This data is directly from git clone https://huggingface.co/datasets/ICML2022/ProteinGym
    sub_file_path = protein_gym_home + 'ProteinGym_substitutions/'
    ref_file_path = protein_gym_home + 'ProteinGym_reference_file_substitutions.csv'

    # TODO need to finish MSAs - save for later
    indels_path = protein_gym_home + 'ProteinGym_indels/'
    mut_msa_dir = zero_shot_home + 'MSA_files/'
    dataset = A2MZeroShotDataset(data_dir=mut_msa_dir, selection_type='MaxHamming', n_sequences=64)

    # TODO re-write "LOAD MODEL as a generate function and call here"
    if args.task == 'seq':
        is_sequences=True
    else:
        is_sequences=False
    causal = False
    n_tokens = len(MSA_ALPHABET)
    if args.mask == 'autoreg' or args.mask == 'so':
        tokenizer = Tokenizer()
        diffusion_timesteps = None  # Not input to model
        state_dict = checkpoint_dir + 'oaardm-seq/' + 'checkpoint1786803.tar'
        if args.mask == 'so':
            n_tokens = len(PROTEIN_ALPHABET)
            tokenizer = Tokenizer(protein_alphabet=PROTEIN_ALPHABET, all_aas=ALL_AAS, pad=PAD)
            causal = True
            state_dict = checkpoint_dir + 'soardm-seq/cnn-38M/' + 'checkpoint1014928.tar'
    elif args.mask == 'blosum' or args.mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat", sequences=is_sequences)
        diffusion_timesteps = config['diffusion_timesteps']
    else:
        print("Select 'blosum', 'random', 'autoreg', or 'so' for --mask")
    masking_idx = tokenizer.mask_id

    model = ByteNetLMTime(n_tokens, d_embed, d_model, n_layers, kernel_size, r,
                          causal=causal, padding_idx=masking_idx, rank=weight_rank, dropout=args.dropout,
                          tie_weights=args.tie_weights, final_ln=args.final_norm, slim=slim, activation=activation,
                          timesteps=diffusion_timesteps)  # works w/ time and non-time models (when diffusion_timesteps is None)

    # Load checkpoints TODO rewrite reload as load checkpoint function in generate
    print("Loading checkpoint from...", state_dict)
    sd = torch.load(state_dict, map_location=torch.device('cpu'))
    msd = sd['model_state_dict']
    if args.mask == 'so':
        msd = {k.split('module.')[1]: v for k, v in msd.items()}
    else:
        msd = {k.split('module.')[0]: v for k,v in msd.items()}
    model.load_state_dict(msd)

    #####
    # Start zero shot loop for sequences
    ####
    if args.task == 'seq':
        # get list of data
        ref_df = parse_dataset(ref_file_path)

        correlations = []
        # Iterate over sequences in dataset (treat each assay as unique - even if repeat protein seq entry)
        for i, entry in ref_df.iterrows():
            print("Running", entry['UniProt_ID'])
            if entry['UniProt_ID'] == 'HIS7_YEAST' or entry['UniProt_ID']=='SPG1_STRSG':
                print("skipping", entry['UniProt_ID'])
                pass
            else:
                offset = entry['MSA_start'] # does not change b/w MSAs
                file = entry['DMS_filename']
                current_file_path = sub_file_path + file

                # relevant sequence info
                current_wild_sequence = entry['target_seq']
                seq_len = entry['seq_len']
                print("sequence length", seq_len)

                if args.mask == 'so':
                    # Batch mutations per sequence so faster
                    max_batch_size = 50
                    # extract info from corresponding mut file-> each mutated sequence, and its corresponding DMS score
                    # entry seq has 1 extra row (row 0 of entry sequences is the wild-type seq)
                    entry_scores, entry_sequences = parse_mut_file(current_file_path, offset, current_wild_sequence, tokenizer,
                                                                   mask=args.mask)
                    batches = round(len(entry_sequences)/max_batch_size)
                    tokenized = torch.stack([torch.tensor(tokenizer.tokenizeMSA(s)) for s in entry_sequences])
                    model_log_prob_scores = []
                    # Run batches of each sequence mutant through model
                    for i in tqdm(range(batches)):
                        tokenized_batch = tokenized[max_batch_size*i:max_batch_size+max_batch_size*i]
                        batch_size = len(tokenized_batch)
                        timestep=torch.tensor([0]*batch_size)
                        with torch.no_grad():
                            p = model(tokenized_batch, timestep) # timestep is a placeholder here -> not used in autoreg or so
                            model_pred = torch.nn.functional.log_softmax(p[:, :, :tokenizer.K-6], dim=2)
                            avg_score = model_pred.sum(axis=(1,2))/torch.tensor(seq_len).unsqueeze(0).expand(batch_size) # avg score = sum of scores / sequence length
                            model_log_prob_scores.extend(avg_score.tolist())
                    # Get Fitness = mut log prob - wild-type log prob
                    wild_score = model_log_prob_scores[0] # first seq is wild seq
                    fitness = [lp - wild_score for lp in model_log_prob_scores]
                    corr, _ = spearmanr(entry_scores, fitness[1:])  # ignore first row (wild) for comparison
                    print(corr)
                    correlations.append(corr)

                elif args.mask == 'autoreg':
                    entry_scores, entry_sequences, mask_locs, wild_index = parse_mut_file(current_file_path, offset,
                                                                                          current_wild_sequence,
                                                                                          tokenizer,
                                                                                          mask=args.mask)
                    fitness = []
                    scores = []
                    for i in tqdm(range(len(entry_sequences))):
                        seq = entry_sequences[i]
                        tokenized = torch.tensor(tokenizer.tokenizeMSA(seq)).unsqueeze(0)
                        timestep = torch.tensor([0]) # placeholder
                        with torch.no_grad():
                            p = model(tokenized, timestep) # timestep is a placeholder here -> not used in autoreg or so
                            model_pred = torch.nn.functional.log_softmax(p[:, i, :tokenizer.K - 6], dim=1).squeeze()
                            model_pred = model_pred.tolist()
                            wild_score = model_pred[wild_index[i]]
                            seq_fitness = np.array([lp - wild_score for lp in model_pred])
                            seq_scores = entry_scores[i]
                            select_scores = seq_scores.astype(bool) # we only want scores for entires in database
                            fitness.extend(seq_fitness[select_scores])
                            scores.extend(seq_scores[select_scores])
                    corr, _ = spearmanr(scores, fitness)
                    print(corr)
                    correlations.append(corr)


                # TODO for SO
                # Run each row through model-> get a log likelihood score for each sequence -> compare that to DMs Score
                # TODO For OA
                # Run each "position" through model -> mask each position, then get fitness by doing log_p_wt - log_p_mut
                # Compare the fitness with DMs score
                # This should be your old code..
                # TODO not sure what to do with this in D3PM
                # actually this current method may work for D3PM? with some mods -> dont delete just yet
                # Still need to iterate over each sequence though

            with open(zero_shot_home+'seq-corr-'+str(args.mask)+'.csv', 'w') as f:
                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerow(correlations)

    elif args.task == 'msa':
        # TODO finish msa
        return None


def parse_dataset(ref_file_path):
    # Get all files in ref_file path
    ref_df = pd.read_csv(ref_file_path, header=0)
    #unique_df = ref_df.drop_duplicates(subset=['UniProt_ID'], keep='first')
    #repeat_entries = ref_df[ref_df.duplicated(subset=['UniProt_ID'], keep=False)]['UniProt_ID'].unique()
    return ref_df #repeat_entries, unique_df, ref_df

def parse_mut_file(file_path, offset, wild_sequence, tokenizer, mask='so'):
    """
    Parse csv file in ProteinGym_substitution folder
    file_path: file path correspnding to csv file for seq of interest
    seq_len: seq length
    tokenizer:
    offset: where does seq start according to reference file

    For SO:
    mutated_sequences: mutant sequence corresponding to each mutant row in df
    scores: each score for each associated mutant

    For OAARDM:
    mut_array: [seq_len by K array] with all assay scores corresponding to mut locs, mut residues in df, zero==no value in df
    mutated_sequences: mutant sequence with a mask token at each index location captured in data frame
      * for each row in mut_array that isn't empty == generate a sequence with a mask at that index location
    scores: list of scores for each associated mut position
      * row sliced out of mut_array

    """
    scores = []
    if mask == 'so':
        mutated_sequences = [wild_sequence]  # wild index at 0
        curr_df = pd.read_csv(file_path, header=0)
        for i, row in curr_df.iterrows():
            mutated_seq_row = list(wild_sequence)
            mut_info = row['mutant']
            scores.append(row['DMS_score'])
            if ':' in mut_info:
                # Case where there are more than 1 mutation for associated score
                mut_info = mut_info.split(':')
                for m in mut_info:
                    mut_aa = m[-1]
                    pos = int(m[1:-1]) - int(offset)
                    mutated_seq_row[pos] = mut_aa
            else:
                # Case where there is only 1 mut per score
                mut_aa = mut_info[-1]
                pos = int(mut_info[1:-1]) - int(offset)
                mutated_seq_row[pos] = mut_aa
            mutated_sequences.append(''.join(mutated_seq_row))
        return scores, mutated_sequences

    elif mask == 'autoreg':
        mutated_sequences = []  # wild index at 0
        # Get data from file
        curr_df = pd.read_csv(file_path, header=0)
        all_rows = np.array(curr_df['mutant'])
        all_rows = [mut.split(':') for mut in all_rows]
        all_muts = []
        all_double_muts = [] # TODO deal w. double muts later
        for mut in all_rows:
            if len(mut) >= 2: # TODO deal with double muts later
                temp_mut = []
                for m in mut:
                    temp_mut.append(m[1:-1])
                all_double_muts.append(temp_mut)
            else:
                all_muts.append(mut[0][1:-1])
        all_muts = np.unique(all_muts)

        mask_locs = []
        wild_index = []
        for mut in all_muts:
            # Generate a masked input for each unique mutation
            mutated_seq_row = list(wild_sequence)
            # Case where there is only 1 mut per score
            pos = int(mut) - int(offset)
            mask_locs.append(pos)
            mutated_seq_row[pos] = '#'
            mutated_sequences.append(''.join(mutated_seq_row))

            # Get scores for that mutation
            mut_df = curr_df[curr_df["mutant"].str.contains(mut)]
            mut_array = np.zeros((tokenizer.K-6))  # initiate an array to hold mutant data
            for i, row in mut_df.iterrows():
                if row['DMS_score'] == 0:
                    row['DMS_score'] += 1e-6  # cant be zero or bool array won't catch as entry
                mut_info = row['mutant']
                # Case where there is only 1 mut per score
                wild_aa = tokenizer.a_to_i[mut_info[0]]
                mut_aa = tokenizer.a_to_i[mut_info[-1]]
                mut_array[mut_aa] = row['DMS_score']
            wild_index.append(wild_aa)
            scores.append(mut_array)
        return scores, mutated_sequences, mask_locs, wild_index

def calc_fitness(tokenized, model_pred):
    # Get fitness score using log(Pmut/Pwild) == log lik Pmut - log lik Pwild
    # Inputs here are expected to be log-prob normalized
    wild_pred = torch.zeros((len(tokenized)))
    for pos, og_res in enumerate(tokenized):
        wild_pred[pos] = model_pred[pos, og_res] # prob it predicts the original residue
    wild_pred = wild_pred.unsqueeze(-1).expand(-1, model_pred.shape[1]) #expand to have same dim as model_pred
    fitness = model_pred - wild_pred
    return fitness

if __name__ == '__main__':
    main()