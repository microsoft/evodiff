import torch
import numpy as np
from sequence_models.constants import MASK, MSA_PAD, MSA_ALPHABET, MSA_AAS, GAP, START, STOP
from evodiff.constants import BLOSUM_ALPHABET
from sklearn.preprocessing import normalize
import itertools
from collections import Counter, OrderedDict
import csv
import pandas as pd

def loadMatrix(path):
    """
    Taken from https://pypi.org/project/blosum/
    Edited slightly from original implementation

    Reads a Blosum matrix from file. Changed slightly to read in larger blosum matrix
    File in a format like:
        https://www.ncbi.nlm.nih.gov/IEB/ToolBox/C_DOC/lxr/source/data/BLOSUM62
    Input:
        path: str, path to a file.
    Returns:
        blosumDict: Dictionary, The blosum dict
    """

    with open(path, "r") as f:
        content = f.readlines()

    blosumDict = {}

    header = True
    for line in content:
        line = line.strip()

        # Skip comments starting with #
        if line.startswith(";"):
            continue

        linelist = line.split()

        # Extract labels only once
        if header:
            labelslist = linelist
            header = False

            # Check if all AA are covered
            #if not len(labelslist) == 25:
            #    print("Blosum matrix may not cover all amino-acids")
            continue

        if not len(linelist) == len(labelslist) + 1:
            print(len(linelist), len(labelslist))
            # Check if line has as may entries as labels
            raise EOFError("Blosum file is missing values.")

        # Add Line/Label combination to dict
        for index, lab in enumerate(labelslist, start=1):
            blosumDict[f"{linelist[0]}{lab}"] = float(linelist[index])

    # Check quadratic
    if not len(blosumDict) == len(labelslist) ** 2:
        print(len(blosumDict), len(labelslist))
        raise EOFError("Blosum file is not quadratic.", len(blosumDict), len(labelslist)**2)
    return blosumDict


def cumprod_matrix(a):
    """
    Takes a list of transition matrices and ouputs a list of the cumulative products (Q_bar) at each timestep
    """
    a_bar = [a[0]]  # initialize w/ first item in list
    start = a[0]
    for i in range(len(a) - 1):
        a_prod_temp = torch.mm(start, a[i + 1])
        start = a_prod_temp
        a_bar.append(a_prod_temp)  # update start
    return a_bar

def softmax(x):
    """
    Compute softmax over x
    """
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def double_stochastic(q):
    q_norm = normalize(q, axis=1, norm='l1')
    while not np.isclose(np.min(np.sum(q_norm, axis=0)), 1): # only checking that one value converges to 1 (prob best to do all 4 min/max)
        q_norm = normalize(q_norm, axis=0, norm='l1')
        q_norm = normalize(q_norm, axis=1, norm='l1')
    return q_norm

def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999, max=8):
    """
    Variance schedule for adding noise
    Start/End will control the magnitude of sigmoidal and cosine schedules.
    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == 'sohl-dickstein':
        betas = torch.linspace(0,num_timesteps-1, num_timesteps)
        betas = 1/(num_timesteps - betas + 1)
    elif schedule == "cosine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.cos(betas) * (end - start) + start
    elif schedule == "exp":
        betas = torch.linspace(0, max, num_timesteps)
        betas = torch.exp(betas) * (end - start) + start
    else:
        print("Must select a valid schedule; ['linear', 'sohl-dickstein', 'cosine', 'exp']")
    return betas

def read_fasta(fasta_path, seq_file, info_file, index_file):
    """
    Read fasta and extract sequences, write out a corresponding index file w/ headers
    Only needs to be done 1x to clean data
    """
    with open(fasta_path) as f_in, open(seq_file, 'w') as f_out, open(info_file, 'w') as info_out, open(index_file, 'w') as i_out:
        current_seq = ''  # sequence string
        index = 0
        for line in f_in:
            if line[0] == '>':
                # print(line)
                i_out.write(str(index)+"\n")
                info_out.write(line)  # line containing seq info
                index+=1
                current_seq += "\n"
                # print(len(current_seq))
                f_out.write(current_seq)
                current_seq = ''  # new line for new seq
            else:
                current_seq += line[:-1]

def parse_fasta(seq_file, idx):
    """
    Reads seq_file from processing steps, and will extract sequence at a given index
    """
    sequence = ''
    with open(seq_file) as f_in:
        for l, line in enumerate(f_in):
            if l == idx:
                sequence += (line[:-1])
                break
    return sequence


class Tokenizer(object):
    """Convert between strings and index"""
    def __init__(self, protein_alphabet=MSA_ALPHABET, pad=MSA_PAD, mask=MASK, all_aas=MSA_AAS, gap=GAP, start=START,
                 stop=STOP, path_to_blosum=None, sequences=False):
        self.alphabet = list("".join(protein_alphabet))
        self.all_aas = list("".join(all_aas))
        self.pad = pad
        self.mask = mask
        self.gap = gap
        self.start = start
        self.stop = stop
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = np.array(self.alphabet)
        if path_to_blosum is not None:
            self.matrix = loadMatrix(path_to_blosum)
            self.matrix_dict = dict(self.matrix)
        self.sequences = sequences # only needed for D3PM MSA vs Seq
        self.K = len(self.all_aas)
        if self.sequences: # This only matters for D3PM models
            self.K = len(self.all_aas[:-1]) # slice out GAPS for sequences
        #print("K is :", self.K)

    @property
    def pad_id(self):
         return self.tokenize(self.pad)[0]

    @property
    def mask_id(self):
        return self.tokenize(self.mask)[0]

    @property
    def gap_id(self):
        return self.tokenize(self.gap)[0]

    @property
    def start_id(self):
        return self.tokenize(self.start)[0]

    @property
    def stop_id(self):
        return self.tokenize(self.stop)[0]

    def q_blosum(self):
        q = np.array([i for i in self.matrix_dict.values()])
        q = q.reshape((len(self.all_aas),len(self.all_aas)))
        q = softmax(q)
        q = double_stochastic(q)
        q = torch.tensor(q)
        # REORDER BLOSUM MATRIX BASED ON MSA_ALPHABET (self.alphabet, self.a_to_i)
        new_q = q.clone()
        i2_to_a = np.array(list(BLOSUM_ALPHABET))
        for i, row in enumerate(new_q):
            for j, value in enumerate(row):
                ind1, ind2 = [i, j]
                key = i2_to_a[ind1], i2_to_a[ind2]
                new1, new2 = [self.a_to_i[k] for k in key]
                new_q[new1, new2] = q[ind1, ind2]
        #IF TRAINING SEQUENCES - DROP GAP
        if self.sequences:
            new_q = new_q[:-1, :-1]
        return new_q

    def q_blosum_schedule(self, timesteps=500, schedule='exp', max=6):
        """
        betas = 'exp' use exp scheme for beta schedule
        """
        print(schedule)
        q = self.q_blosum()
        betas = _beta_schedule(timesteps, schedule=schedule, max=max)
        betas = betas / betas.max() + 1/timesteps
        Q_t = [] # scheduled matrix
        for i in range(timesteps):
            q_non_diag = torch.ones((self.K,self.K)) * q * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(self.K)) * norm_constant
            R = q_diag + q_non_diag
            Q_t.append(R)
        Q_prod = cumprod_matrix(Q_t)
        Q_prod = torch.stack(Q_prod) # cumprod of matrices
        Q_t = torch.stack(Q_t) # scheduled matrix
        return Q_prod, Q_t

    def q_random_schedule(self, timesteps=500, schedule='sohl-dickstein'):
        print(schedule)
        betas = _beta_schedule(timesteps, schedule=schedule)
        Q_t = []  # scheduled matrix
        for i in range(len(betas)):
            q_non_diag = torch.ones((self.K,self.K)) / self.K * betas[i]
            norm_constant = (1 - (q_non_diag).sum(axis=0))
            q_diag = torch.tensor(np.identity(self.K)) * norm_constant
            R = q_diag + q_non_diag
            Q_t.append(R)
        Q_prod = cumprod_matrix(Q_t)
        Q_prod = torch.stack(Q_prod)  # cumprod of matrices
        Q_t = torch.stack(Q_t)  # scheduled matrix
        return Q_prod, Q_t

    def tokenize(self, seq):
        return np.array([self.a_to_i[a] for a in seq[0]]) # for nested lists

    def tokenizeMSA(self, seq):
        return np.array([self.a_to_i[a] for a in seq]) # not nested

    def untokenize(self, x):
        if torch.is_tensor(x):
            return "".join([self.i_to_a[int(t.item())] for t in x])
        else:
            return "".join([self.i_to_a[t] for t in x])

    def one_hot(self, tokenized):
        "one hot encode according to indexing"
        #print(tokenized, self.K)
        x_onehot = torch.nn.functional.one_hot(tokenized, num_classes=self.K)
        return x_onehot.to(torch.double)

    def undo_one_hot(self, x_onehot):
        "one hot -> seq"
        tokenized = [np.where(r==1)[0] for r in x_onehot]
        return tokenized

def parse_txt(fasta_file):
    "Read output of PGP seqs from text file"
    train_seqs = []
    with open(fasta_file, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            if len(row) >= 1:
                if row[0][0] != '>':
                    train_seqs.append(str(row[0]))
    return train_seqs

def removekey(d, list_of_keys):
    r = d.copy()
    for key in list_of_keys:
        del r[key]
    return r

def csv_to_dict(generate_file):
    seqs = ''
    with open(generate_file, 'r') as file:
        filecontent = csv.reader(file)
        for row in filecontent:
            if len(row) >= 1:
                if row[0][0] != '>':
                    seqs += str(row[0])
    aminos_gen = Counter(
        {'A': 0, 'M': 0, 'R': 0, 'T': 0, 'D': 0, 'Y': 0, 'P': 0, 'F': 0, 'L': 0, 'E': 0, 'W': 0, 'I': 0, 'N': 0, 'S': 0, \
         'K': 0, 'Q': 0, 'H': 0, 'V': 0, 'G': 0, 'C': 0, 'X': 0, 'B': 0, 'Z': 0, 'U': 0, 'O': 0, 'J': 0, '-': 0})
    aminos_gen.update(seqs)

    order_of_keys = ['A','M','R','T','D','Y','P','F','L','E','W','I','N','S',
                     'K','Q','H','V','G','C','X','B','Z','J','O','U','-']
    list_of_tuples = [(key, aminos_gen[key]) for key in order_of_keys]
    aminos_gen_ordered = OrderedDict(list_of_tuples)
    return aminos_gen_ordered

def normalize_list(list):
    norm = sum(list)
    new_list = [item / norm for item in list]
    return new_list

def get_matrix(all_pairs, all_aa_pairs, alphabet):
    count_map = {}
    for i in all_pairs:
        count_map[i] = count_map.get(i, 0) + (1 / 63)
    for aa_pair in all_aa_pairs:
        if aa_pair not in count_map.keys():
            pass
            count_map[aa_pair] = 0
    _dict = {k: count_map[k] for k in sorted(count_map.keys())}
    _matrix = list(_dict.values())
    _matrix = np.asarray(_matrix).reshape(len(alphabet), len(alphabet))
    return _matrix


def get_pairs(array, alphabet):
    all_pairs = []
    all_q_val = []
    for b in np.arange(array.shape[0]):
        curr_msa = array[b]
        for col in np.arange(curr_msa.shape[1]):
            q_val = curr_msa[0, col]
            if q_val < len(alphabet):
                q_val = curr_msa[0, col]
                all_q_val.append(q_val)
                col_vals = list(curr_msa[1:, col])
                col_vals = filter(lambda val: val < len(alphabet), col_vals)
                curr_pairs = [(q_val, v) for v in col_vals]
                all_pairs.append(curr_pairs)
    all_pairs = list(itertools.chain(*all_pairs))
    return all_pairs


def normalize_matrix(data, alphabet):
    alpha_labels = list(alphabet)
    table = pd.DataFrame(data, index=alpha_labels, columns=alpha_labels)
    table = table / table.sum(axis=0)  # normalize
    table.fillna(0, inplace=True)

    table_vals = table.values
    table_diag_vals = np.diag(table)
    return table, table_vals, table_diag_vals

def extract_seq_a3m(generate_file):
    "Get sequences from A3M file"
    list_of_seqs = []
    with open(generate_file, 'r') as file:
            filecontent = csv.reader(file)
            for row in filecontent:
                if len(row) >= 1:
                    if row[0][0] != '>':
                        list_of_seqs.append(str(row[0]))
    return list_of_seqs[1:]

def get_pairwise(msa, alphabet):
    all_pairs = []
    queries = msa[:, 0, :]
    for row in queries:
        row = row.astype(int)
        curr_query = list(row[row < len(alphabet)])
        curr_query = [alphabet[c] for c in curr_query if c < len(alphabet)]
        curr_pairs = itertools.permutations(curr_query, 2)
        all_pairs.append(list(curr_pairs))
    all_pairs = list(itertools.chain(*all_pairs))
    return all_pairs

def download_model(model_name):
    #url = f"https://.. {model_name} .. " # TODO add links when uploaded to Zenodo
    #state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
    state_dict = "zenodo/checkpoints/"+model_name+".tar"
    return state_dict

def download_generated_sequences(model_name):
    sequence_list = "curl -O"
    return sequence_list

def get_valid_msas(data_top_dir, data_dir='openfold/', selection_type='MaxHamming', n_sequences=64, max_seq_len=512,
                   out_path='../DMs/ref/'):
    from evodiff.data import A3MMSADataset
    import os
    from torch.utils.data import Subset
    from sequence_models.collaters import MSAAbsorbingCollater
    from evodiff.collaters import D3PMCollaterMSA
    from torch.utils.data import DataLoader
    import tqdm as tqdm

    valid_msas = []
    query_msas = []
    seq_lens = []

    _ = torch.manual_seed(1) # same seeds as training
    np.random.seed(1)

    dataset = A3MMSADataset(selection_type, n_sequences, max_seq_len, data_dir=os.path.join(data_top_dir,data_dir), min_depth=64)

    train_size = len(dataset)
    random_ind = np.random.choice(train_size, size=(train_size - 10000), replace=False)
    val_ind = np.delete(np.arange(train_size), random_ind)
    ds_valid = Subset(dataset, val_ind)

    return ds_valid
