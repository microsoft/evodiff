import csv
import argparse
import torch
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr
from sequence_models.datasets import A2MZeroShotDataset
from sequence_models.constants import MSA_ALPHABET, PROTEIN_ALPHABET, ALL_AAS, PAD
from dms.utils import Tokenizer
from dms.model import ByteNetLMTime

## Takes in csv file with generated sequences and performs downstream tasks
def main():
    _ = torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,  default='seq')
    parser.add_argument('--mask', type=str,  default='autoreg')
    args = parser.parse_args()

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
    if args.mask == 'autoreg' or args.mask == 'so' or args.mask == 'reference':
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
        diffusion_timesteps=500
    masking_idx = tokenizer.mask_id

    model = ByteNetLMTime(n_tokens, 8, 1024, 16, 5, 128,
                          causal=causal, padding_idx=masking_idx, rank=None, dropout=0.0,
                          final_ln=True, slim=True, activation='gelu',
                          timesteps=diffusion_timesteps)

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
        repeat_entries, unique_df, ref_df = parse_dataset(ref_file_path)

        correlations = []
        # Iterate over sequences in dataset
        for i, row in unique_df.iterrows():
            print("Running", row['UniProt_ID'])
            offset = row['MSA_start'] # does not change b/w MSAs
            if row['UniProt_ID'] in repeat_entries:
                print("found multiple files")
                files_df = ref_df[ref_df['UniProt_ID'] == row['UniProt_ID']]  # This will index all the files you need
                files = [row for row in files_df['DMS_filename']]
            else:
                files = [row['DMS_filename']]
            print(files)
            current_files_path = [sub_file_path + f for f in files]

            # relevant sequence info
            current_target_sequence = row['target_seq']
            seq_len = row['seq_len']

            # extract info from corresponding mut file, and bool of residues/mut for which we have data
            dataset_fitness, dataset_locations = parse_mut_file(current_files_path, seq_len, tokenizer, offset)

            # feed sequence through model
            tokenized = torch.tensor(tokenizer.tokenizeMSA(current_target_sequence)) # always use MSA-> related to input format
            timestep=torch.tensor([0]) # TODO not sure what to do with this in D3PM
            with torch.no_grad():
                model_pred = model(tokenized.unsqueeze(0), timestep) # timestep is a placeholder here -> not used in autoreg
                model_pred = torch.nn.functional.log_softmax(model_pred[:, :, :tokenizer.K], dim=2).squeeze()

            # Get Fitness = mut log prob - wild-type log prob
            model_fitness = calc_fitness(tokenized, model_pred)

            # Use bool to select pos/tokens in original dataset
            #print(dataset_fitness[dataset_locations], model_fitness[dataset_locations])
            corr, _ = spearmanr(dataset_fitness[dataset_locations], model_fitness[dataset_locations])
            correlations.append(corr)

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
    unique_df = ref_df.drop_duplicates(subset=['UniProt_ID'], keep='first')
    repeat_entries = ref_df[ref_df.duplicated(subset=['UniProt_ID'], keep=False)]['UniProt_ID'].unique()
    return repeat_entries, unique_df, ref_df

def parse_mut_file(file_path, seq_len, tokenizer, offset):
    """
    Parse csv file in ProteinGym_substitution folder
    file_path: file path correspnding to csv file for seq of interest
    seq_len: seq length
    tokenizer:
    offset: where does seq start according to reference file
    """
    mut_array = np.zeros((seq_len, tokenizer.K)) # initiate an array to hold mutant data
    for file in file_path:
        curr_df = pd.read_csv(file, header=0)
        if curr_df.columns[0] != 'mutant':
            pass # TODO skipping pointer files for now - need to download manually (I think there is only 1 in this dataset)
        else:
            for i, row in curr_df.iterrows():
                if row['DMS_score'] == 0:
                    row['DMS_score'] += 1e-6 # cant be zero or bool array won't catch as entry

                mut_info = row['mutant']
                if ':' in mut_info:
                    # Case where there are more than 1 mutation for associated score
                    mut_info = mut_info.split(':')
                    for m in mut_info:
                        mut_aa = tokenizer.a_to_i[m[-1]]
                        pos = int(m[1:-1]) - int(offset)
                        mut_array[pos, mut_aa] = row['DMS_score']
                else:
                    # Case where there is only 1 mut per score
                    mut_aa = tokenizer.a_to_i[mut_info[-1]]
                    pos = int(mut_info[1:-1]) - int(offset)
                    mut_array[pos, mut_aa] = row['DMS_score']

    mut_bool = mut_array.astype(bool) # only interested in data in dataset
    return mut_array, mut_bool

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