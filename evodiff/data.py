import os
from tqdm import tqdm
from scipy.spatial.distance import hamming, cdist

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from evodiff.utils import Tokenizer
from sequence_models.utils import parse_fasta
from sequence_models.constants import PROTEIN_ALPHABET, trR_ALPHABET, PAD, GAP
from collections import Counter

def read_openfold_files(data_dir, filename):
    """
    Helper function to read the openfold files

    inputs:
        data_dir : path to directory with data
        filename: MSA name

    outputs:
        path: path to .a3m file
    """
    if os.path.exists(data_dir + filename + '/a3m/uniclust30.a3m'):
        path = data_dir + filename + '/a3m/uniclust30.a3m'
    elif os.path.exists(data_dir + filename + '/a3m/bfd_uniclust_hits.a3m'):
        path = data_dir + filename + '/a3m/bfd_uniclust_hits.a3m'
    else:
        raise Exception("Missing filepaths")
    return path

def read_idr_files(data_dir, filename):
    """
    Helper function to read the idr files

    inputs:
        data_dir : path to directory with data
        filename: IDR name

    outputs:
        path: path to IDR file
    """
    if os.path.exists(data_dir + filename):
        path = data_dir + filename
    else:
        raise Exception("Missing filepaths")
    return path

def get_msa_depth_lengths(data_dir, all_files, save_depth_file, save_length_file, idr=False):
    """
    Function to compute openfold and IDR dataset depths

    inputs:
        data_dir : path to directory with data
        all_files: all filenames
        save_depth_file: file to save depth values in
        save_length_file: file to save length values in
    """
    msa_depth = []
    msa_lengths = []
    for filename in tqdm(all_files):
        if idr:
            path = read_idr_files(data_dir, filename)
        else:
            path = read_openfold_files(data_dir, filename)
        parsed_msa = parse_fasta(path)
        msa_depth.append(len(parsed_msa))
        msa_lengths.append(len(parsed_msa[0]))  # all seq in MSA are same length
    np.savez_compressed(data_dir+save_depth_file, np.asarray(msa_depth))
    np.savez_compressed(data_dir + save_length_file, np.asarray(msa_lengths))

def get_idr_query_index(data_dir, all_files, save_file):
    """
    Function to get IDR query index

    inputs:
        data_dir : path to directory with data
        all_files: all filenames
        save_file: file to save query indexes in
    """
    query_idxs = []
    for filename in tqdm(all_files):
        msa_data, msa_names = parse_fasta(data_dir + filename, return_names=True)
        query_idx = [i for i, name in enumerate(msa_names) if name == filename.split('_')[0]][0]  # get query index
        query_idxs.append(query_idx)
    np.savez_compressed(data_dir + save_file, np.asarray(query_idxs))

def get_sliced_gap_depth_openfold(data_dir, all_files, save_file, max_seq_len=512):
    """
    Function to compute make sure every MSA has 64 sequences

    inputs:
        data_dir : path to directory with data
        all_files: all filenames
        save_file: file to save data to
    """
    sliced_depth = []
    for filename in tqdm(all_files):
        path=read_openfold_files(data_dir, filename)
        parsed_msa = parse_fasta(path)
        sliced_msa_depth = [seq for seq in parsed_msa if (Counter(seq)[GAP]) <= max_seq_len] # Only append seqs with gaps<512
        sliced_depth.append(len(sliced_msa_depth))

    np.savez_compressed(data_dir + save_file, np.asarray(sliced_depth))


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

    def __getitem__(self, idx):
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
        print(output) # check that there are no all-msa rows
        #import pdb; pdb.set_trace()
        return output


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
        alphabet = PROTEIN_ALPHABET
        self.tokenizer = Tokenizer(alphabet)
        self.alpha = np.array(list(alphabet))
        self.gap_idx = self.tokenizer.alphabet.index(GAP)

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        [print("Excluding", x) for x in os.listdir(self.data_dir) if x.endswith('.npz')]
        all_files = [x for x in os.listdir(self.data_dir) if not x.endswith('.npz')]
        all_files = sorted(all_files)
        print("unfiltered length", len(all_files))

        ## Filter based on depth (keep > 64 seqs/MSA)
        if not os.path.exists(data_dir + 'openfold_lengths.npz'):
            raise Exception("Missing openfold_lengths.npz in openfold/")
        if not os.path.exists(data_dir + 'openfold_depths.npz'):
            #get_msa_depth_openfold(data_dir, sorted(all_files), 'openfold_depths.npz')
            raise Exception("Missing openfold_depths.npz in openfold/")
        if min_depth is not None: # reindex, filtering out MSAs < min_depth
            _depths = np.load(data_dir+'openfold_depths.npz')['arr_0']
            depths = pd.DataFrame(_depths, columns=['depth'])
            depths = depths[depths['depth'] >= min_depth]
            keep_idx = depths.index

            _lengths = np.load(data_dir+'openfold_lengths.npz')['ells']
            lengths = np.array(_lengths)[keep_idx]
            all_files = np.array(all_files)[keep_idx]
            print("filter MSA depth > 64", len(all_files))

        # Re-filter based on high gap-contining rows
        if not os.path.exists(data_dir + 'openfold_gap_depths.npz'):
            #get_sliced_gap_depth_openfold(data_dir, all_files, 'openfold_gap_depths.npz', max_seq_len=max_seq_len)
            raise Exception("Missing openfold_gap_depths.npz in openfold/")
        _gap_depths = np.load(data_dir + 'openfold_gap_depths.npz')['arr_0']
        gap_depths = pd.DataFrame(_gap_depths, columns=['gapdepth'])
        gap_depths = gap_depths[gap_depths['gapdepth'] >= min_depth]
        filter_gaps_idx = gap_depths.index
        lengths = np.array(lengths)[filter_gaps_idx]
        all_files = np.array(all_files)[filter_gaps_idx]
        print("filter rows with GAPs > 512", len(all_files))

        self.filenames = all_files  # IDs of samples to include
        self.lengths = lengths # pass to batch sampler
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = read_openfold_files(self.data_dir, filename)
        parsed_msa = parse_fasta(path)

        aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
        aligned_msa = [''.join(seq) for seq in aligned_msa]

        tokenized_msa = [self.tokenizer.tokenizeMSA(seq) for seq in aligned_msa]
        tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
        msa_seq_len = len(tokenized_msa[0])

        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        # Slice to 512
        sliced_msa_seq = tokenized_msa[:, slice_start: slice_start + self.max_seq_len]
        anchor_seq = sliced_msa_seq[0]  # This is the query sequence in MSA

        # slice out all-gap rows
        sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [self.gap_idx])]
        msa_num_seqs = len(sliced_msa)

        if msa_num_seqs < self.n_sequences:
            print("before for len", len(sliced_msa_seq))
            print("msa_num_seqs < self.n_sequences should not be called")
            print("tokenized msa shape", tokenized_msa.shape)
            print("tokenized msa depth", len(tokenized_msa))
            print("sliced msa depth", msa_num_seqs)
            print("used to set slice")
            print("msa_seq_len", msa_seq_len)
            print("self max seq len", self.max_seq_len)
            print(slice_start)
            import pdb; pdb.set_trace()
            output = np.full(shape=(self.n_sequences, seq_len), fill_value=self.tokenizer.pad_id)
            output[:msa_num_seqs] = sliced_msa
            raise Exception("msa num_seqs < self.n_sequences, indicates dataset not filtered properly")
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
        return output


class IDRDataset(Dataset):
    """Build dataset for IDRs"""

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
        alphabet = PROTEIN_ALPHABET
        self.tokenizer = Tokenizer(alphabet)
        self.alpha = np.array(list(alphabet))
        self.gap_idx = self.tokenizer.alphabet.index(GAP)

        # Get npz_data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise FileNotFoundError(data_dir)

        [print("Excluding", x) for x in os.listdir(self.data_dir) if x.endswith('.npz')]
        all_files = [x for x in os.listdir(self.data_dir) if not x.endswith('.npz')]
        all_files = sorted(all_files)
        print("unfiltered length", len(all_files))

        ## Filter based on depth (keep > 64 seqs/MSA)
        if not os.path.exists(data_dir + 'idr_lengths.npz'):
            raise Exception("Missing idr_lengths.npz in human_idr_alignments/human_protein_alignments/")
        if not os.path.exists(data_dir + 'idr_depths.npz'):
            #get_msa_depth_openfold(data_dir, sorted(all_files), 'openfold_depths.npz')
            raise Exception("Missing idr_depths.npz in human_idr_alignments/human_protein_alignments/")
        _depths = np.load(data_dir + 'idr_depths.npz')['arr_0']
        depths = pd.DataFrame(_depths, columns=['depth'])

        if min_depth is not None: # reindex, filtering out MSAs < min_depth
            raise Exception("MIN DEPTH CONSTRAINT NOT CURRENTLY WORKING ON IDRS")
        #    depths = depths[depths['depth'] >= min_depth]
        #keep_idx = depths.index

        _lengths = np.load(data_dir + 'idr_lengths.npz')['arr_0']
        lengths = pd.DataFrame(_lengths, columns=['length'])
        if max_seq_len is not None:
            lengths = lengths[lengths['length'] <= max_seq_len]
        keep_idx = lengths.index

        lengths = np.array(_lengths)[keep_idx]
        all_files = np.array(all_files)[keep_idx]
        print("filter MSA length >", max_seq_len, len(all_files))

        _query_idxs = np.load(data_dir+'idr_query_idxs.npz')['arr_0']
        query_idxs = np.array(_query_idxs)[keep_idx]

        self.filenames = all_files  # IDs of samples to include
        self.lengths = lengths # pass to batch sampler
        self.n_sequences = n_sequences
        self.max_seq_len = max_seq_len
        self.selection_type = selection_type
        self.query_idxs = query_idxs


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        print(filename)
        path = read_idr_files(self.data_dir, filename)
        parsed_msa = parse_fasta(path)
        aligned_msa = [[char for char in seq if (char.isupper() or char == '-') and not char == '.'] for seq in parsed_msa]
        aligned_msa = [''.join(seq) for seq in aligned_msa]

        tokenized_msa = [self.tokenizer.tokenizeMSA(seq) for seq in aligned_msa]

        tokenized_msa = np.array([l.tolist() for l in tokenized_msa])
        msa_seq_len = len(tokenized_msa[0])
        print("msa_seq_len", msa_seq_len, "max seq len", self.max_seq_len)

        if msa_seq_len > self.max_seq_len:
            slice_start = np.random.choice(msa_seq_len - self.max_seq_len + 1)
            seq_len = self.max_seq_len
        else:
            slice_start = 0
            seq_len = msa_seq_len

        # Slice to 512
        sliced_msa_seq = tokenized_msa[:, slice_start: slice_start + self.max_seq_len]
        #print(sliced_msa_seq.shape)
        query_idx = self.query_idxs[idx]
        anchor_seq = tokenized_msa[query_idx]  # This is the query sequence
        print("anchor seq", len(anchor_seq))
        # Remove query from MSA?
        #del tokenized_msa[query_idx]

        # slice out all-gap rows
        sliced_msa = [seq for seq in sliced_msa_seq if (list(set(seq)) != [self.gap_idx])]
        msa_num_seqs = len(sliced_msa)

        # if msa_num_seqs < self.n_sequences:
        #     raise Exception("msa num_seqs < self.n_sequences, indicates dataset not filtered properly")
        if msa_num_seqs > self.n_sequences:
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
        return output
