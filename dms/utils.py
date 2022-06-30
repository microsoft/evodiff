import blosum as bl
import numpy as np
from sequence_models.constants import ALL_AAS, SPECIALS, MASK
from dms.constants import PAD

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
    "Generate a dictionary of tuples"
    def __init__(self, tokenizer=Tokenizer(), matrix=bl.BLOSUM(62)):
        self.tokenizer = tokenizer
        self.matrix = matrix

    @property
    def matrix_dict(self):
        return dict(self.matrix)

    def blosum_dict(self):
        blosum_dict = dict(self.matrix)
        keys = [key for key in blosum_dict.keys()]
        keys_tokenized = [tokenize_blosum(key) for key in keys]
        d = dict(zip(keys_tokenized, blosum_dict.values()))
        return d