import numpy as np
import os
import pathlib
import argparse
home = str(pathlib.Path.home())
from plot import aa_reconstruction_parity_plot, msa_substitution_rate, msa_pairwise_interactions
from dms.utils import Tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_fpath', type=str, nargs='?',
                        default='/Users/nityathakkar/Desktop/research/msr/model_output/')
    parser.add_argument('--mask', type=str, default='autoreg')
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

    # Downstream tasks
    gen_msas = np.load(args.out_fpath+'generated_msas.npy')
    train_msas = np.load(project_dir+'ref/'+args.mask+'_tokenized_openfold_train_msas.npy')
    aa_reconstruction_parity_plot(project_dir, args.out_fpath, 'generated_msas.a3m', msa=True)
    msa_substitution_rate(gen_msas, train_msas, tokenizer.all_aas[:-7], args.out_fpath)
    msa_pairwise_interactions(gen_msas, train_msas, tokenizer.all_aas[:-7], args.out_fpath)


if __name__ == '__main__':
    main()