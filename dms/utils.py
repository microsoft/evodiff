#import blosum as bl
from dms.data import loadMatrix
import torch
import numpy as np
from sequence_models.constants import SPECIALS, MASK
from dms.constants import ALL_AAS, PAD, BLOSUM62_AAS
from sklearn.preprocessing import normalize

data_dir = '/home/v-salamdari/Desktop/DMs/data/' # TODO fix this

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def double_stochastic(q):
    q_norm = normalize(q, axis=1, norm='l1')
    while not np.isclose(np.min(np.sum(q_norm, axis=0)), 1): # only checking that one value converges to 1 (prob best to do all 4 min/max)
        q_norm = normalize(q_norm, axis=0, norm='l1')
        q_norm = normalize(q_norm, axis=1, norm='l1')
    return q_norm

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
    def __init__(self, all_aas=ALL_AAS, specials=SPECIALS, pad=PAD, mask=MASK,
                 path_to_blosum=data_dir+"blosum62.mat"):
        self.matrix = loadMatrix(path_to_blosum)
        self.matrix_dict = dict(self.matrix)
        self.all_aas = list(all_aas)
        self.alphabet = list("".join(all_aas+pad+specials))
        self.pad = pad
        self.mask = mask
        self.vocab = sorted(set("".join(all_aas)))
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = np.array(self.alphabet)

    @property
    def pad_id(self):
         return self.tokenize(self.pad)[0]

    @property
    def mask_id(self):
        return self.tokenize(self.mask)[0]

    @property
    def q_blosum(self):
        q = np.array([i for i in self.matrix_dict.values()])
        q = q.reshape((len(self.all_aas), len(self.all_aas)))
        q = softmax(q)
        q = double_stochastic(q)
        return q

    @property
    def q_random(self):
        q = np.eye(len(self.all_aas)) + 1 / 10  # arbitrary, set diag to zero assign other transitions some prob
        q = double_stochastic(q)  # normalize so rows += 1
        return q

    def q_blosum_alpha_t(self, alpha_t=0.03):
        q = self.q_blosum
        q_diag = np.identity(len(self.all_aas)) * q
        q_non_diag = (1 - np.identity(len(self.all_aas))) * q
        q_alpha_t = double_stochastic((q_diag + np.dot(q_non_diag, np.array(alpha_t))))
        return q_alpha_t

    def tokenize(self, seq):
        return np.array([self.a_to_i[a] for a in seq[0]]) # seq is a tuple with empty second dim

    def untokenize(self, x):
        if torch.is_tensor(x):
            return "".join([self.i_to_a[int(t.item())] for t in x])
        else:
            return "".join([self.i_to_a[t] for t in x])

    def one_hot(self, seq):
        "one hot encode according to indexing"
        x_onehot = np.zeros((len(seq), len(self.all_aas)))
        for i, a in enumerate(seq):
            one_index = self.a_to_i[a]
            x_onehot[i][one_index] = 1
        return x_onehot
