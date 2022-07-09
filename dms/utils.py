import blosum as bl
import numpy as np
from sequence_models.constants import ALL_AAS, SPECIALS, MASK
from dms.constants import PAD, BLOSUM62_ALPHABET, ROUND

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def norm_q(q):
    "Normalize transition matrix, ensures that rows sum to 1"
    q_norm = np.zeros(q.shape)
    for i in range(q.shape[0]):
        _norm = q[i]/q[i].sum()
        q_norm[i] = _norm.round(ROUND)
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

def tokenize_blosum(seq):
    return tuple(Tokenizer().a_to_i[a] for a in seq) # use for blosum

class Tokenizer(object):
    """Convert between strings and index"""
    def __init__(self, all_aas=ALL_AAS, specials=SPECIALS, pad=PAD, mask=MASK):
        self.alphabet = sorted(set("".join(pad+all_aas+specials)))
        self.pad = pad
        self.mask = mask
        self.vocab = sorted(set("".join(all_aas)))
        self.a_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_a = np.array(self.alphabet)

    @property
    def pad_id(self):
        return self.alphabet.index(self.pad)

    @property
    def mask_id(self):
        return self.alphabet.index(self.mask)

    def tokenize(self, seq):
        return np.array([self.a_to_i[a] for a in seq[0]]) # seq is a tuple with empty second dim

    def untokenize(self, x):
        if x.type() == 'torch.FloatTensor':
            return "".join([self.i_to_a[int(t.item())] for t in x])
        else:
            return "".join([self.i_to_a[t] for t in x])

class Blosum62(object):
    """
    Tokenizer for Blosum62 - Order of BLOSUM matrices controls one hot indexing
    diff that AA alphabet -- but probably can combine these two at some point. No need for
    2 indexing schemes
    """
    def __init__(self, tokenizer=Tokenizer(), alphabet=BLOSUM62_ALPHABET, path_to_blosum="data/blosum62.mat", num_aas=23):
        self.tokenizer = tokenizer
        self.alphabet=BLOSUM62_ALPHABET
        self.matrix = bl.BLOSUM(path_to_blosum)
        self.matrix_dict = dict(self.matrix)
        self.b_to_i = {u: i for i, u in enumerate(self.alphabet)}
        self.i_to_b = np.array([a for a in self.alphabet])
        self.num_aas = num_aas

    @property
    def q_blosum(self):
        q = np.array([i for i in self.matrix_dict.values()])
        q = q.reshape((self.num_aas, self.num_aas))
        q = softmax(q)
        q = norm_q(q)
        return q

    @property
    def q_random(self):
        q = np.eye(23) + 1 / 10 # arbitrary, set diagnoal to zero assign other transitions some prob
        q = norm_q(q) # normalize so rows += 1
        return q

    def blosum_dict(self):
        blosum_dict = dict(self.matrix)
        keys = [key for key in blosum_dict.keys()]
        keys_tokenized = [tokenize_blosum(key) for key in keys]
        d = dict(zip(keys_tokenized, blosum_dict.values()))
        return d

    def one_hot(self, seq):
        x_onehot = np.zeros((len(seq), self.num_aas))
        for i, a in enumerate(seq):
            one_index = self.b_to_i[a]
            x_onehot[i][one_index] = 1
        return x_onehot