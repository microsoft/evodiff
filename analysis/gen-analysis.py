import numpy as np
import os
import pathlib
import argparse
from evodiff.plot import aa_reconstruction_parity_plot, msa_substitution_rate, msa_pairwise_interactions, plot_percent_similarity
from evodiff.utils import Tokenizer
import csv
from sequence_models.collaters import MSAAbsorbingCollater
from evodiff.collaters import D3PMCollaterMSA
from sequence_models.constants import MSA_ALPHABET

home = str(pathlib.Path.home())

# Used to evaluate/condense all MSA runs that were generated on amulet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_fpath', type=str, nargs='?',
                        default='/Users/nityathakkar/Desktop/research/msr/model_output/')
    parser.add_argument('--mask', type=str, default='autoreg')
    parser.add_argument('--subsampling', type=str, default='MaxHamming')
    parser.add_argument('--start-valid', action='store_true')
    parser.add_argument('--amlt', action='store_true')
    parser.add_argument('--idr', action='store_true')
    parser.add_argument('--start-query', action='store_true') # if starting from query -> gen msa
    parser.add_argument('--start-msa', action='store_true') # if starting from msa, gen-> query
    args = parser.parse_args()

    project_dir = home + '/Desktop/DMs/'
    try:
        data_top_dir = os.getenv('AMLT_DATA_DIR') + '/'
    except:
        data_top_dir = '../data/'

    if args.mask == 'autoreg':
        tokenizer = Tokenizer()
    elif args.mask == 'blosum' or args.mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")
    else:
        print("mask must be: 'autoreg', 'blosum', or 'random'")
    print(tokenizer.alphabet)

    print("START VALID", args.start_valid)
    if args.amlt:
        if args.start_valid:
            train_msas, gen_msas = preprocess_amlt_outputs(args.out_fpath, arg_mask=args.mask, data_top_dir=data_top_dir,
                                                       idr=args.idr, start_valid=args.start_valid)
        else:
            tokenizer = sample_train_msa(os.path.join(data_top_dir,'openfold/'), arg_mask=args.mask,
                                         selection_type=args.subsampling, num_samples=50, out_path=project_dir + 'ref/')
            gen_msas = preprocess_amlt_outputs(args.out_fpath, arg_mask=args.mask, data_top_dir=data_top_dir,
                                               idr=args.idr, start_valid=args.start_valid)
            train_msas = np.load(project_dir + 'ref/' + args.mask + args.subsampling + '_tokenized_openfold_train_msas.npy')
    else:
        train_msas = np.load(
            project_dir + 'ref/' + args.mask + args.subsampling + '_tokenized_openfold_train_msas.npy')
        gen_msas = np.load(args.out_fpath+'generated_msas.npy')
    if args.start_msa:
        gen_file = 'gen_msas_onlyquery.txt'
    elif args.start_query:
        gen_file = 'gen_msas_onlymsa.txt'
    else:
        gen_file = 'generated_msas.a3m'
    aa_reconstruction_parity_plot(project_dir, args.out_fpath, gen_file, msa=True, start_valid=args.start_valid,
                                  start_msa=args.start_msa, start_query=args.start_query)
    msa_substitution_rate(gen_msas, train_msas, tokenizer.all_aas[:-7], args.out_fpath)
    msa_pairwise_interactions(gen_msas, train_msas, tokenizer.all_aas[:-7], args.out_fpath)


def preprocess_amlt_outputs(out_fpath, arg_mask, data_top_dir, idr=False, start_valid=False):
    if arg_mask == 'autoreg':
        tokenizer = Tokenizer()
    elif arg_mask == 'blosum' or arg_mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")
    if idr:
        tag='idr-'
    else:
        tag='gen-'

    gen_files = [filename for filename in os.listdir(out_fpath) if filename.startswith("gen")]
    print(gen_files)
    if start_valid:
        valid_msa_arr = np.zeros((len(gen_files), 64, 512)) + tokenizer.pad_id
    gen_msa_arr = np.zeros((len(gen_files), 64, 512)) + tokenizer.pad_id

    msa = 0
    num_seqs = 0
    gen_msa = 0
    gen_num_seqs = 0
    if start_valid:
        all_valid = out_fpath + 'valid_msas.a3m'
    all_gen = out_fpath + 'generated_msas.a3m'

    all_query_valid = out_fpath + 'valid_msas_onlyquery'
    all_msa_valid = out_fpath + 'valid_msas_onlymsa'
    all_query_gen = out_fpath + 'gen_msas_onlyquery'
    all_msa_gen = out_fpath + 'gen_msas_onlymsa'

    all_query_valid_write_str = ""
    all_query_gen_write_str = ""

    for i in range(len(gen_files)):
        if start_valid:
            valid_file = out_fpath + tag + str(i + 1) + '/' + 'valid_msas.a3m'
            os.system("cat " + valid_file + " >> " + all_valid)
        gen_file = out_fpath + tag + str(i + 1) + '/' + 'generated_msas.a3m'
        os.system("cat " + gen_file + " >> " + all_gen)
        if start_valid:
            with open(valid_file, 'r') as file:
                filecontent = csv.reader(file)
                for row in filecontent:
                    if 'SEQUENCE' in row[0] or 'seq 0' in row[0]:
                        msa += 1
                        num_seqs = 0
                        query_seq = True
                        all_query_valid_write_str+='>SEQUENCE'+str(i)+'\n'
                    elif '>' in row[0]:
                        query_seq = False
                    else:
                        if query_seq:
                            os.system("echo " + row[0] + " >> " + all_query_valid + '.txt')
                            all_query_valid_write_str += row[0]+'\n'
                        else:
                            os.system("echo " + row[0] + " >> " + all_msa_valid + '.txt')
                        tokenized_seq = tokenizer.tokenizeMSA(row[0])
                        valid_msa_arr[msa - 1, num_seqs, :len(tokenized_seq)] = tokenized_seq
                        num_seqs += 1
        with open(gen_file, 'r') as file:
            filecontent = csv.reader(file)
            write_sequence = '' # POTS models are written with seqs on multiple lines of out file, make sure this works on your msa files
            for row in filecontent:
                if 'SEQUENCE_' in row[0] or 'seq 0' in row[0] or 'MSA_0' in row[0]:
                    gen_msa += 1
                    gen_num_seqs = 0
                    query_seq = True
                    all_query_gen_write_str += '>SEQUENCE' + str(i) + '\n'
                elif '>' in row[0] or 'tr' in row[0]:
                    tokenized_seq = tokenizer.tokenizeMSA(write_sequence)
                    gen_msa_arr[gen_msa - 1, gen_num_seqs, :len(tokenized_seq)] = tokenized_seq
                    gen_num_seqs += 1
                    if query_seq:
                        os.system("echo " + write_sequence + " >> " + all_query_gen + '.txt')
                        all_query_gen_write_str += write_sequence + '\n'
                        query_seq = False
                    else:
                        os.system("echo " + write_sequence + " >> " + all_msa_gen + '.txt')
                    write_sequence=''
                else:
                    write_sequence += row[0]
            os.system("echo " + write_sequence + " >> " + all_msa_gen + '.txt')
    with open(all_query_valid +'.a3m', 'a') as f:
        f.write(all_query_valid_write_str)
        f.close()
    with open(all_query_gen +'.a3m', 'a') as f:
        f.write(all_query_gen_write_str)
        f.close()
    if start_valid:
        return valid_msa_arr, gen_msa_arr
    else:
        return gen_msa_arr

def preprocess_train_msa_file(out_fpath, f_name='valid_msas.a3m', arg_mask='autoreg', data_top_dir='data/'):
    if arg_mask == 'autoreg':
        tokenizer = Tokenizer()
    elif arg_mask == 'blosum' or arg_mask == 'random':
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")

    total_msas = 0
    print(out_fpath+f_name)
    with open(out_fpath + f_name, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            print(row)
            if 'SEQUENCE' in row[0]:
                total_msas += 1

    msa = 0
    msa_arr = np.zeros((total_msas, 64, 512)) + tokenizer.pad_id

    with open(out_fpath + f_name, 'r') as file:
        print(f_name)
        filecontent = csv.reader(file)
        for row in filecontent:
            if 'SEQUENCE' in row[0]:
                print(msa)
                msa += 1
                num_seqs = 0
            elif '>' in row[0]:
                pass
            else:
                tokenized_seq = tokenizer.tokenizeMSA(row[0])
                msa_arr[msa - 1, num_seqs, :len(tokenized_seq)] = tokenized_seq
                num_seqs += 1
    return msa_arr


def sample_train_msa(train_msa_path, arg_mask='blosum', num_samples=2, selection_type='MaxHamming',
                         out_path='../DMs/ref/'):
    from sequence_models.datasets import A3MMSADataset
    from torch.utils.data import DataLoader
    train_msas = []

    data_top_dir = '../data/'
    if arg_mask == 'autoreg':
        tokenizer = Tokenizer()
        collater = MSAAbsorbingCollater(alphabet=MSA_ALPHABET)
    elif arg_mask == 'blosum' or arg_mask == 'random':
        diffusion_timesteps = 500
        tokenizer = Tokenizer(path_to_blosum=data_top_dir + "blosum62-special-MSA.mat")
        if arg_mask == 'random':
            Q_prod, Q_t = tokenizer.q_random_schedule(timesteps=diffusion_timesteps)
        if arg_mask == 'blosum':
            Q_prod, Q_t = tokenizer.q_blosum_schedule(timesteps=diffusion_timesteps)
        collater = D3PMCollaterMSA(tokenizer=tokenizer, num_timesteps=diffusion_timesteps, Q=Q_t, Q_bar=Q_prod)

    dataset = A3MMSADataset(data_dir=train_msa_path, selection_type=selection_type, n_sequences=64, max_seq_len=256)
    loader = DataLoader(dataset, batch_size=num_samples, collate_fn=collater, num_workers=8)

    count = 0
    for batch in loader:
        if arg_mask == 'blosum' or arg_mask == 'random':
            src, src_one_hot, timestep, tgt, tgt_one_hot, Q, Q_prod, q = batch
        else:
            src, tgt, mask = batch
        if count < 1:
            train_msas.append(tgt)
            count += 1
        else:
            break
    train_msas = np.concatenate(train_msas, axis=0)
    # return train_msas # array of shape #msas, 64, 256
    np.save(out_path + arg_mask + selection_type + '_tokenized_openfold_train_msas', train_msas)

    return tokenizer

if __name__ == '__main__':
    main()