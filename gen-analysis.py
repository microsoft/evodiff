import numpy as np
import os
import pathlib
import argparse
from analysis.plot import aa_reconstruction_parity_plot, msa_substitution_rate, msa_pairwise_interactions
from dms.utils import Tokenizer
import csv
from sequence_models.collaters import MSAAbsorbingCollater
from dms.collaters import D3PMCollaterMSA
from sequence_models.constants import MSA_ALPHABET

home = str(pathlib.Path.home())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_fpath', type=str, nargs='?',
                        default='/Users/nityathakkar/Desktop/research/msr/model_output/')
    parser.add_argument('--mask', type=str, default='autoreg')
    parser.add_argument('--subsampling', type=str, default='MaxHamming')
    parser.add_argument('--start-valid', action='store_true')
    parser.add_argument('--amlt', action='store_true')
    args = parser.parse_args()

    project_dir = home + '/Desktop/DMs/'
    try:
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/'
    except:
        data_top_dir = 'data/'

    if args.mask == 'autoreg':
        tokenizer = Tokenizer()
    elif args.mask == 'blosum' or args.mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")
    else:
        print("mask must be: 'autoreg', 'blosum', or 'random'")
    print(tokenizer.alphabet)
    if args.start_valid:
        start_valid='_valid'
    else:
        start_valid=''

    # Generate a reference file
    #tokenizer = preprocess_train_msa(data_dir=os.path.join(data_top_dir,'openfold/'), arg_mask=args.mask,
    #                                 selection_type=args.subsampling, num_samples=100)
    if args.start_valid and args.amlt:
        train_msas, gen_msas = preprocess_amlt_outputs(args.out_fpath, arg_mask=args.mask, data_top_dir=data_top_dir)
    elif args.start_valid:
        train_msas = preprocess_train_msa_file(args.out_fpath, arg_mask=args.mask, data_top_dir=data_top_dir)
        gen_msas = np.load(args.out_fpath + 'generated_msas.npy')
    else:
        train_msas = np.load(
            project_dir + 'ref/' + args.mask + args.subsampling + start_valid + '_tokenized_openfold_train_msas.npy')
        gen_msas = np.load(args.out_fpath+'generated_msas.npy')

    #train_msas = np.load(project_dir+'ref/'+args.mask+args.subsampling+start_valid+'_tokenized_openfold_train_msas.npy')
    aa_reconstruction_parity_plot(project_dir, args.out_fpath, 'generated_msas.a3m', msa=True, start_valid=args.start_valid)
    msa_substitution_rate(gen_msas, train_msas, tokenizer.all_aas[:-7], args.out_fpath)
    msa_pairwise_interactions(gen_msas, train_msas, tokenizer.all_aas[:-7], args.out_fpath)

def preprocess_amlt_outputs(out_fpath, arg_mask, data_top_dir):
    if arg_mask == 'autoreg':
        tokenizer = Tokenizer()
    elif arg_mask == 'blosum' or arg_mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")

    gen_files = os.listdir(out_fpath)
    valid_msa_arr = np.zeros((len(gen_files), 64, 256)) + tokenizer.pad_id
    gen_msa_arr = np.zeros((len(gen_files), 64, 256)) + tokenizer.pad_id

    msa = 0
    num_seqs = 0
    gen_msa = 0
    gen_num_seqs = 0
    all_valid = out_fpath + 'valid_msas.a3m'
    all_gen = out_fpath + 'generated_msas.a3m'

    for i in range(len(gen_files)):
        valid_file = out_fpath + 'gen-' + str(i + 1) + '/' + 'valid_msas.a3m'
        gen_file = out_fpath + 'gen-' + str(i + 1) + '/' + 'generated_msas.a3m'
        os.system("cat " + valid_file + " >> " + all_valid)
        os.system("cat " + gen_file + " >> " + all_gen)

        with open(valid_file, 'r') as file:
            filecontent = csv.reader(file)
            for row in filecontent:
                if 'SEQUENCE' in row[0]:
                    msa += 1
                    num_seqs = 0
                elif '>' in row[0]:
                    pass
                else:
                    # print(msa-1, num_seqs, len(tokenizer.tokenizeMSA(row[0])))
                    tokenized_seq = tokenizer.tokenizeMSA(row[0])
                    valid_msa_arr[msa - 1, num_seqs, :len(tokenized_seq)] = tokenized_seq
                    num_seqs += 1

        with open(gen_file, 'r') as file:
            filecontent = csv.reader(file)
            for row in filecontent:
                if 'SEQUENCE' in row[0]:
                    gen_msa += 1
                    gen_num_seqs = 0
                elif '>' in row[0]:
                    pass
                else:
                    # print(gen_msa-1, gen_num_seqs, len(tokenizer.tokenizeMSA(row[0])))
                    tokenized_seq = tokenizer.tokenizeMSA(row[0])
                    gen_msa_arr[gen_msa - 1, gen_num_seqs, :len(tokenized_seq)] = tokenized_seq
                    gen_num_seqs += 1
    return valid_msa_arr, gen_msa_arr

def preprocess_train_msa_file(out_fpath, f_name='valid_msas.a3m', arg_mask='autoreg', data_top_dir='data/'):
    if arg_mask == 'autoreg':
        tokenizer = Tokenizer()
    elif arg_mask == 'blosum' or arg_mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")


    total_msas = 0
    with open(out_fpath + f_name, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            if 'SEQUENCE' in row[0]:
                total_msas += 1

    msa = 0
    msa_arr = np.zeros((total_msas, 64, 256)) + tokenizer.pad_id
    # print(msa_arr)

    with open(out_fpath + f_name, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            if 'SEQUENCE' in row[0]:
                msa += 1
                num_seqs = 0
            elif '>' in row[0]:
                pass
            else:
                # print(msa-1, num_seqs, len(tokenizer.tokenizeMSA(row[0])))
                tokenized_seq = tokenizer.tokenizeMSA(row[0])
                msa_arr[msa - 1, num_seqs, :len(tokenized_seq)] = tokenized_seq
                num_seqs += 1
    return msa_arr

if __name__ == '__main__':
    main()