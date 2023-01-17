import os
from os import path
from tqdm import tqdm
from scipy.spatial.distance import hamming, cdist

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from dms.utils import Tokenizer
from sequence_models.utils import parse_fasta
from sequence_models.constants import PROTEIN_ALPHABET, trR_ALPHABET, PAD, GAP


class TRRMSADataset(Dataset):
    """Build dataset for trRosetta data: MSA Absorbing Diffusion model"""

    def __init__(self, selection_type, n_sequences, max_seq_len, data_dir=None):
        """
        Args:
            selection_type: str,
                MSA selection strategy of random or MaxHamming
            n_sequences: int,
                number of sequences to subsample down to
            max_seq_len: int,
                maximum MSA sequence length
            data_dir: str,
                if you have a specified npz directory
        """

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        # MSAs should be in the order of npz_dir
        all_files = os.listdir(self.data_dir)
        if 'trrosetta_lengths.npz' in all_files:
            all_files.remove('trrosetta_lengths.npz')
        all_files = sorted(all_files)
        self.filenames = all_files  # IDs of samples to include

        # Number of sequences to subsample down to
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type

        alphabet = trR_ALPHABET + PAD
        self.tokenizer = Tokenizer(alphabet)
        self.alpha = np.array(list(alphabet))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):  # TODO: add error checking?
        filename = self.filenames[idx]
        data = np.load(self.data_dir + filename)
        # Grab sequence info
        msa = data['msa']

        msa_seq_len = len(msa[0])
        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        sliced_msa = msa[:, slice_start: slice_start + self.max_seq_len]
        anchor_seq = sliced_msa[0]  # This is the query sequence in MSA

        sliced_msa = [list(seq) for seq in sliced_msa if (list(set(seq)) != [self.tokenizer.alphabet.index(GAP)])]
        sliced_msa = np.asarray(sliced_msa)
        msa_num_seqs = len(sliced_msa)

        # If fewer sequences in MSA than self.n_sequences, create sequences padded with PAD token based on 'random' or
        # 'MaxHamming' selection strategy
        if msa_num_seqs < self.n_sequences:
            output = np.full(shape=(self.n_sequences, seq_len), fill_value=self.tokenizer.pad_id)
            output[:msa_num_seqs] = sliced_msa
        elif msa_num_seqs > self.n_sequences:
            if self.selection_type == 'random':
                random_idx = np.random.choice(msa_num_seqs - 1, size=self.n_sequences - 1, replace=False) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate((anchor_seq, sliced_msa[random_idx]), axis=0)
            elif self.selection_type == 'non-random':
                output = sliced_msa[:64]
            elif self.selection_type == "MaxHamming":
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]
                random_ind = np.random.choice(msa_ind)
                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((self.n_sequences - 2, m))

                for i in range(self.n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0) # (1,num_choices)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
        else:
            output = sliced_msa
        output = [''.join(seq) for seq in self.alpha[output]]
        print("shape of msa", len(output), len(output[0]))
        #print(output) # check that there are no all-msa rows
        #import pdb; pdb.set_trace()
        return output

def get_msa_depth(data_dir, all_files, save_file, dataset='openfold'): # TODO combine this function w/ nitya old functions (find nitya old functions)
    msa_depth = []
    for filename in tqdm(all_files):
        path = data_dir + filename
        if dataset == 'openfold':
            path+='/a3m/bfd_uniclust_hits.a3m'
        #data = np.load(path, allow_pickle=True)
        if os.path.exists(path):
            parsed_msa = parse_fasta(path)
        else:
            parsed_msa = parse_fasta(data_dir + filename + '/a3m/uniclust30.a3m') # TODO why are there two different filenames?
        msa_depth.append(len(parsed_msa))
        #if len(parsed_msa) < 64:
        #    print(len(parsed_msa))
    np.savez_compressed(data_dir+save_file, np.asarray(msa_depth))


class A3MMSADataset(Dataset):
    """Build dataset for A3M data: MSA Absorbing Diffusion model"""

    def __init__(self, selection_type, n_sequences, max_seq_len, data_dir=None, min_depth=None):
        """
        Args:
            selection_type: str,
                MSA selection strategy of random or MaxHamming
            n_sequences: int,
                number of sequences to subsample down to
            max_seq_len: int,
                maximum MSA sequence length
            data_dir: str,
                if you have a specified data directory
        """

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        all_files = os.listdir(self.data_dir) # TODO clean this up, why is rosetta_lengths in this dir? have code create these files in this class
        if 'openfold_lengths.npz' in all_files:
            all_files.remove('openfold_lengths.npz')
        if 'trrosetta_test_lengths.npz' in all_files:
            all_files.remove('trrosetta_test_lengths.npz')
        if 'openfold_depths.npz' in all_files:
            all_files.remove('openfold_depths.npz')
        elif 'openfold_depths.npz' not in all_files:
            get_msa_depth(data_dir, sorted(all_files), 'openfold_depths.npz')
        all_files = sorted(all_files)

        #print(all_files)

        ## Constructor; loop through once, find length>min, reindex file/length/depth
        if min_depth is not None: # reindex, filtering out MSAs < min_depth
            _depths = np.load(data_dir+'openfold_depths.npz')['arr_0']
            depths = pd.DataFrame(_depths, columns=['depth'])
            depths = depths[depths['depth'] >= 64]
            keep_idx = depths.index

            _lengths = np.load(data_dir+'openfold_lengths.npz')['ells']
            lengths = np.array(_lengths)[keep_idx]
            filtered_files = np.array(all_files)[keep_idx]

        self.filenames = filtered_files  # IDs of samples to include
        self.lengths = lengths
        ## Add attribute here: Get new lengths for batching self.length, pass dataset.lengths to batch sampler (instead of loading twice)
        ## TODO: self.lengths= # where is this being used?

        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type
        alphabet=PROTEIN_ALPHABET
        self.tokenizer = Tokenizer(alphabet)
        #self.tokenizer = tokenizer
        self.alpha = np.array(list(alphabet))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):  # TODO: add error checking?
        filename = self.filenames[idx]
        if path.exists(self.data_dir + filename + '/a3m/uniclust30.a3m'):
            parsed_msa = parse_fasta(self.data_dir + filename + '/a3m/uniclust30.a3m')
        elif path.exists(self.data_dir + filename + '/a3m/bfd_uniclust_hits.a3m'):
            parsed_msa = parse_fasta(self.data_dir + filename + '/a3m/bfd_uniclust_hits.a3m')
        else: # TODO what is this line for?
            parsed_msa = parse_fasta(self.data_dir + filename)
            # print(filename)
            # raise ValueError("file does not exist")
        #print("parsed", parsed_msa[0])
        aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
        aligned_msa = [''.join(seq) for seq in aligned_msa]
        #print("aligned", aligned_msa[0])

        # with open('/home/t-nthakkar/msa_' + str(idx) + '.txt', 'a') as f:
        #     for seq in aligned_msa:
        #         f.write(seq)
        #         f.write('\n')

        tokenized_msa = [self.tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
        tokenized_msa = np.array([l.tolist() for l in tokenized_msa])

        msa_seq_len = len(tokenized_msa[0])

        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        sliced_msa = tokenized_msa[:, slice_start: slice_start + self.max_seq_len]
        anchor_seq = sliced_msa[0]  # This is the query sequence in MSA

        # gap_str = '-' * msa_seq_len
        # parsed_msa = [seq.upper() for seq in parsed_msa if seq != gap_str]

        sliced_msa = [seq for seq in sliced_msa if (list(set(seq)) != [self.tokenizer.alphabet.index('-')])]
        msa_num_seqs = len(sliced_msa)
        #print("msa num seqs", msa_num_seqs)
        # If fewer sequences in MSA than self.n_sequences, create sequences padded with PAD token based on 'random' or
        # 'MaxHamming' selection strategy
        if msa_num_seqs < self.n_sequences: # TODO this should not be called anymore
            print("msa_num_seqs < self.n_sequences should not be called")
            print(msa_num_seqs)
            print("seq len", seq_len)
            output = np.full(shape=(self.n_sequences, seq_len), fill_value=self.tokenizer.pad_id)
            output[:msa_num_seqs] = sliced_msa
        elif msa_num_seqs > self.n_sequences:
            if self.selection_type == 'random':
                random_idx = np.random.choice(msa_num_seqs - 1, size=self.n_sequences - 1, replace=False) + 1
                anchor_seq = np.expand_dims(anchor_seq, axis=0)
                output = np.concatenate((anchor_seq, np.array(sliced_msa)[random_idx.astype(int)]), axis=0)
            elif self.selection_type == "MaxHamming":
                output = [list(anchor_seq)]
                msa_subset = sliced_msa[1:]
                msa_ind = np.arange(msa_num_seqs)[1:]
                random_ind = np.random.choice(msa_ind)
                random_seq = sliced_msa[random_ind]
                output.append(list(random_seq))
                random_seq = np.expand_dims(random_seq, axis=0)
                msa_subset = np.delete(msa_subset, (random_ind - 1), axis=0)
                m = len(msa_ind) - 1
                distance_matrix = np.ones((self.n_sequences - 2, m))

                for i in range(self.n_sequences - 2):
                    curr_dist = cdist(random_seq, msa_subset, metric='hamming')
                    curr_dist = np.expand_dims(np.array(curr_dist), axis=0)  # shape is now (1,msa_num_seqs)
                    distance_matrix[i] = curr_dist
                    col_min = np.min(distance_matrix, axis=0)  # (1,num_choices)
                    max_ind = np.argmax(col_min)
                    random_ind = max_ind
                    random_seq = msa_subset[random_ind]
                    output.append(list(random_seq))
                    random_seq = np.expand_dims(random_seq, axis=0)
                    msa_subset = np.delete(msa_subset, random_ind, axis=0)
                    distance_matrix = np.delete(distance_matrix, random_ind, axis=1)
        else:
            output = sliced_msa

        output = [''.join(seq) for seq in self.alpha[output]]
        # print("parsed", parsed_msa[1],
        #       "\naligned", aligned_msa[1],
        #       "\noutput", output[1]) # check that there are no all-msa rows
        # print(len(parsed_msa), len(aligned_msa), len(output))
        return output
